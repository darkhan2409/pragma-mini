from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .buckets import BucketSpec, edges_artifact, fit_edges
from .config import (
    EVENT_TYPE_PROFILE,
    EVENT_TYPES,
    FIT_DATASET,
    KIND_METADATA,
    SCHEMA_VERSION,
    FieldSpec,
    Settings,
    feature_specs,
    numeric_specs,
    predictable_specs,
    specs_for,
)
from .raw import RawDataset, parse_payloads
from .stats import FieldAccumulator, field_stats_entry, metadata_entry, unigram_entry, value_counts_table


# ============================================================
# ИДЕЯ
# ============================================================
#
# Первый проход: считаем всё обучаемое ТОЛЬКО на train.
#
# Fit-набор это объединение историй валидных train-примеров.
# Истории это префиксы ленты, а причины пропуска монотонны по
# cutoff, поэтому объединение историй клиента равно одному
# префиксу до его последнего валидного train-cutoff. Значит
# каждое событие учитывается ровно один раз, сколько бы
# примеров его ни включало.
#
# Namespace profile это отдельный набор: только те as-of
# снимки, которые реально выбраны валидными train-примерами.
# Снимки первых месяцев не попадают сюда никогда (правило 90
# дней), но как события profile_snapshot в ленте они есть.
# ============================================================


@dataclass(frozen=True)
class FitScope:
    """
    Кто и до какого момента входит в обучаемые статистики.
    """

    clients: np.ndarray            # отсортированные client_id
    cutoff: np.ndarray             # datetime64[us], параллельно clients
    snapshots: set[tuple[int, datetime]]

    @property
    def n_clients(self) -> int:
        return int(self.clients.size)

    @property
    def max_cutoff(self) -> datetime | None:
        if self.cutoff.size == 0:
            return None
        return self.cutoff.max().astype("datetime64[us]").astype(datetime)

    def mask_for(self, client_id: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """
        Какие строки ленты входят в fit-набор.
        """

        if self.clients.size == 0 or client_id.size == 0:
            return np.zeros(client_id.size, dtype=bool)

        position = np.searchsorted(self.clients, client_id)

        position = np.clip(position, 0, self.clients.size - 1)

        known = self.clients[position] == client_id

        limit = self.cutoff[position]

        return known & (ts < limit)


def fit_scope(cutoff_index: pa.Table) -> FitScope:

    rows = cutoff_index.filter(pc.equal(cutoff_index.column("dataset"), FIT_DATASET))

    if rows.num_rows == 0:
        return FitScope(np.array([], np.int64), np.array([], "datetime64[us]"), set())

    client_id = rows.column("client_id").to_numpy()
    cutoff = rows.column("cutoff").to_numpy()
    snapshot_ts = rows.column("snapshot_ts").to_numpy(zero_copy_only=False)

    order = np.lexsort((cutoff, client_id))

    client_id = client_id[order]
    cutoff = cutoff[order]

    # Последний валидный train-cutoff клиента.
    last = np.ones(client_id.size, dtype=bool)
    last[:-1] = client_id[:-1] != client_id[1:]

    snapshots = {
        (int(cid), ts.astype("datetime64[us]").astype(datetime))
        for cid, ts in zip(rows.column("client_id").to_numpy(), snapshot_ts)
        if not np.isnat(ts)
    }

    return FitScope(clients=client_id[last], cutoff=cutoff[last], snapshots=snapshots)


# ============================================================
# АККУМУЛЯТОРЫ
# ============================================================


def new_accumulators() -> dict[tuple[str, str], FieldAccumulator]:
    return {spec.key: FieldAccumulator(spec) for spec in feature_specs()}


def accumulate_events(raw: RawDataset, scope: FitScope, accumulators: dict[tuple[str, str], FieldAccumulator]) -> None:
    """
    Проход по ленте: event_type и поля payload каждого типа.
    """

    for batch in raw.iter_row_groups("timeline"):

        mask = scope.mask_for(batch.column("client_id").to_numpy(), batch.column("ts").to_numpy())

        if not mask.any():
            continue

        rows = batch.filter(pa.array(mask))

        accumulators[("timeline", "event_type")].update(rows.column("event_type"))

        event_type_column = rows.column("event_type")

        for event_type in EVENT_TYPES:

            typed = rows.filter(pc.equal(event_type_column, event_type))

            if typed.num_rows == 0:
                continue

            parsed = parse_payloads(event_type, typed.column("payload"))

            for name in parsed.schema.names:
                key = (event_type, name)
                if key in accumulators:
                    accumulators[key].update(parsed.column(name))


def select_snapshots(raw: RawDataset, scope: FitScope) -> pa.Table:
    """
    Снимки профиля, выбранные как as-of валидными train-примерами.
    """

    profile = raw.read("profile")

    if profile.num_rows == 0 or not scope.snapshots:
        return profile.slice(0, 0)

    client_id = profile.column("client_id").to_numpy()
    ts = profile.column("ts").to_numpy()

    keys = np.array(
        sorted((cid, np.datetime64(stamp, "us").astype(np.int64)) for cid, stamp in scope.snapshots),
        dtype=np.int64,
    )

    wanted = set(map(tuple, keys))

    mask = np.array(
        [(int(cid), int(stamp.astype("datetime64[us]").astype(np.int64))) in wanted for cid, stamp in zip(client_id, ts)],
        dtype=bool,
    )

    return profile.filter(pa.array(mask))


def accumulate_profile(raw: RawDataset, scope: FitScope, accumulators: dict[tuple[str, str], FieldAccumulator]) -> pa.Table:

    snapshots = select_snapshots(raw, scope)

    for spec in specs_for("profile"):

        if spec.kind == KIND_METADATA:
            continue

        accumulators[spec.key].update(snapshots.column(spec.field))

    return snapshots


# ============================================================
# ARTIFACTS
# ============================================================


def fit_buckets(accumulators: dict[tuple[str, str], FieldAccumulator], settings: Settings) -> dict[tuple[str, str], BucketSpec]:

    specs: dict[tuple[str, str], BucketSpec] = {}

    for spec in numeric_specs():

        acc = accumulators[spec.key]

        specs[spec.key] = fit_edges(
            spec.namespace,
            spec.field,
            acc.numeric_values(),
            settings.buckets_for(spec),
        )

    return specs


def value_counts_path(spec: FieldSpec) -> str:
    return f"value_counts/{spec.namespace}__{spec.field}.parquet"


def build_artifacts(
    accumulators: dict[tuple[str, str], FieldAccumulator],
    buckets: dict[tuple[str, str], BucketSpec],
    scope: FitScope,
    settings: Settings,
) -> tuple[dict, dict, dict, dict[str, pa.Table]]:
    """
    field_stats, unigram_baselines, bucket_edges и таблицы
    полных распределений numeric-полей.
    """

    from .config import REGISTRY

    records: dict[str, int] = {}

    for namespace in sorted({spec.namespace for spec in feature_specs()}):
        namespace_specs = [spec for spec in feature_specs() if spec.namespace == namespace]
        records[namespace] = accumulators[namespace_specs[0].key].n_total if namespace_specs else 0

    fit_echo = {
        "dataset": FIT_DATASET,
        "n_fit_clients": scope.n_clients,
        "fit_cutoff_max": scope.max_cutoff.isoformat() if scope.max_cutoff else None,
        "n_fit_snapshots": len(scope.snapshots),
        "records": records,
        "rule": {
            "events": "события ленты train-клиентов с ts < последний валидный train-cutoff клиента, каждое один раз",
            "profile": "as-of снимки, выбранные валидными train-примерами, каждый один раз",
        },
    }

    stats_fields: dict[str, dict[str, dict]] = {}
    unigram_fields: dict[str, dict[str, dict]] = {}
    distributions: dict[str, pa.Table] = {}

    for spec in REGISTRY.values():

        section = stats_fields.setdefault(spec.namespace, {})

        if not spec.is_feature:
            section[spec.field] = metadata_entry(spec)
            continue

        acc = accumulators[spec.key]
        bucket = buckets.get(spec.key)

        path = value_counts_path(spec) if spec.is_numeric else None

        section[spec.field] = field_stats_entry(spec, acc, bucket, settings, path)

        if spec.is_numeric:
            distributions[path] = value_counts_table(acc)

    for spec in predictable_specs():
        unigram_fields.setdefault(spec.namespace, {})[spec.field] = unigram_entry(
            spec, accumulators[spec.key], buckets.get(spec.key)
        )

    field_stats = {
        "schema_version": SCHEMA_VERSION,
        "fit": fit_echo,
        "thresholds": settings.as_dict()["flag_thresholds"],
        "rare_count_threshold": settings.rare_count_threshold,
        "fields": stats_fields,
    }

    unigram_baselines = {
        "schema_version": SCHEMA_VERSION,
        "fit": fit_echo,
        "note": "baseline «всегда самое частое»: мода и распределение по непустым значениям; numeric кодируется bucket'ами",
        "fields": unigram_fields,
    }

    bucket_edges = edges_artifact(buckets, settings.default_buckets, SCHEMA_VERSION)

    return bucket_edges, field_stats, unigram_baselines, distributions


def run_fit(raw: RawDataset, cutoff_index: pa.Table, settings: Settings):
    """
    Полный первый проход.
    """

    scope = fit_scope(cutoff_index)

    accumulators = new_accumulators()

    accumulate_events(raw, scope, accumulators)
    accumulate_profile(raw, scope, accumulators)

    buckets = fit_buckets(accumulators, settings)

    bucket_edges, field_stats, unigram_baselines, distributions = build_artifacts(
        accumulators, buckets, scope, settings
    )

    return scope, buckets, bucket_edges, field_stats, unigram_baselines, distributions
