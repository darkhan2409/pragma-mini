from __future__ import annotations

from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.generator.profile import month_start, next_month

from .config import OBSERVATION_SOURCE, Settings
from .raw import RawDataset
from .splits import dataset_for


# ============================================================
# ИДЕЯ
# ============================================================
#
# Пример это пара (client_id, cutoff), cutoff это начало
# следующего месяца. Кандидаты строятся для каждого клиента и
# каждого полного месяца RAW; каждый получает статус: валиден
# и попал в датасет, валиден, но сочетание не используется,
# или пропущен с причиной.
#
# История примера это префикс ленты: все события с ts < cutoff.
# Так как лента отсортирована по (ts, seq) и seq плотный,
# префикс задаётся одним числом seq_end.
#
# Причины пропуска монотонны по cutoff: если клиент валиден на
# каком-то cutoff, он валиден и на всех более поздних.
# ============================================================


TS = pa.timestamp("us")

CUTOFF_INDEX_SCHEMA = pa.schema(
    [
        ("client_id", pa.int64()),
        ("cutoff", TS),
        ("observation_month", TS),
        ("client_group", pa.string()),
        ("month_role", pa.string()),
        ("dataset", pa.string()),
        ("valid", pa.bool_()),
        ("skip_reason", pa.string()),
        ("observation_start", TS),
        ("observation_days", pa.int64()),
        ("n_events", pa.int64()),
        ("seq_end", pa.int64()),
        ("snapshot_ts", TS),
    ]
)

EXAMPLES_SCHEMA = pa.schema(
    [
        ("client_id", pa.int64()),
        ("cutoff", TS),
        ("observation_month", TS),
        ("client_group", pa.string()),
        ("month_role", pa.string()),
        ("dataset", pa.string()),
        ("observation_start", TS),
        ("observation_days", pa.int64()),
        ("n_events", pa.int64()),
        ("seq_end", pa.int64()),
        ("snapshot_ts", TS),
    ]
)


# ============================================================
# СЕТКА МЕСЯЦЕВ
# ============================================================


def month_grid(history_start: datetime, feature_end: datetime) -> list[datetime]:
    """
    Начала полных месяцев наблюдения внутри [history_start, feature_end).
    Неполный месяц на любом краю отбрасывается.
    """

    first = month_start(history_start)

    if first < history_start:
        first = next_month(first)

    months: list[datetime] = []

    month = first

    while next_month(month) <= feature_end:
        months.append(month)
        month = next_month(month)

    return months


def cutoff_of(month: datetime) -> datetime:
    return next_month(month)


# ============================================================
# КАНДИДАТЫ
# ============================================================


def _client_runs(client_ids: np.ndarray) -> dict[int, tuple[int, int]]:
    """
    Границы [lo, hi) непрерывного блока каждого клиента в массиве,
    отсортированном по client_id.
    """

    if client_ids.size == 0:
        return {}

    change = np.flatnonzero(np.diff(client_ids)) + 1

    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [client_ids.size]])

    return {int(client_ids[lo]): (int(lo), int(hi)) for lo, hi in zip(starts, ends)}


def _sorted_by_client_ts(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:

    client_ids = table.column("client_id").to_numpy()
    ts = table.column("ts").to_numpy()

    order = np.lexsort((ts, client_ids))

    return client_ids[order], ts[order]


def first_seen_by_client(raw: RawDataset, source: str) -> dict[int, datetime | None]:

    coverage = raw.read("source_coverage")

    rows = coverage.filter(pc.equal(coverage.column("source"), source))

    return {
        int(client_id): (None if seen is None else seen)
        for client_id, seen in zip(rows.column("client_id").to_pylist(), rows.column("first_seen").to_pylist())
    }


def build_cutoff_index(
    raw: RawDataset,
    groups: dict[int, str],
    roles: dict[datetime, str],
    settings: Settings,
) -> pa.Table:

    months = sorted(roles)

    cutoffs = [cutoff_of(month) for month in months]

    cutoffs_np = np.array([np.datetime64(cutoff, "us") for cutoff in cutoffs])

    history_start = raw.manifest.history_start

    # --------------------------------------------------------
    # ИСТОЧНИКИ
    # --------------------------------------------------------

    event_clients, event_ts = _sorted_by_client_ts(raw.read("timeline", ["client_id", "ts"]))
    event_runs = _client_runs(event_clients)

    profile_clients, profile_ts = _sorted_by_client_ts(raw.read("profile", ["client_id", "ts"]))
    profile_runs = _client_runs(profile_clients)

    first_seen = first_seen_by_client(raw, OBSERVATION_SOURCE)

    # --------------------------------------------------------
    # КАНДИДАТЫ
    # --------------------------------------------------------

    columns: dict[str, list] = {name: [] for name in CUTOFF_INDEX_SCHEMA.names}

    for client_id in sorted(groups):

        group = groups[client_id]

        lo, hi = event_runs.get(client_id, (0, 0))
        ts_client = event_ts[lo:hi]

        plo, phi = profile_runs.get(client_id, (0, 0))
        profile_client = profile_ts[plo:phi]

        seq_ends = np.searchsorted(ts_client, cutoffs_np, side="left")
        snapshot_positions = np.searchsorted(profile_client, cutoffs_np, side="left") - 1

        seen = first_seen.get(client_id)

        observation_start = None if seen is None else max(history_start, seen)

        for month, cutoff, seq_end, snapshot_position in zip(months, cutoffs, seq_ends, snapshot_positions):

            role = roles[month]

            n_events = int(seq_end)

            snapshot_ts = (
                None
                if snapshot_position < 0
                else profile_client[snapshot_position].astype("datetime64[us]").astype(datetime)
            )

            observation_days = None if observation_start is None else (cutoff - observation_start).days

            # Приоритет причин фиксирован и монотонен по cutoff.
            if observation_start is None:
                reason = "no_transactions_coverage"
            elif observation_days < settings.min_observation_days:
                reason = "insufficient_observation"
            elif n_events == 0:
                reason = "no_events"
            elif snapshot_ts is None:
                reason = "no_profile_snapshot"
            else:
                reason = None

            valid = reason is None

            dataset = dataset_for(group, role) if valid else None

            if valid and dataset is None:
                reason = "unused_combination"

            columns["client_id"].append(client_id)
            columns["cutoff"].append(cutoff)
            columns["observation_month"].append(month)
            columns["client_group"].append(group)
            columns["month_role"].append(role)
            columns["dataset"].append(dataset)
            columns["valid"].append(valid)
            columns["skip_reason"].append(reason)
            columns["observation_start"].append(observation_start)
            columns["observation_days"].append(observation_days)
            columns["n_events"].append(n_events)
            columns["seq_end"].append(n_events)
            columns["snapshot_ts"].append(snapshot_ts)

    return pa.table(
        {name: pa.array(values, type=CUTOFF_INDEX_SCHEMA.field(name).type) for name, values in columns.items()},
        schema=CUTOFF_INDEX_SCHEMA,
    )


def examples_of(cutoff_index: pa.Table, dataset: str) -> pa.Table:

    rows = cutoff_index.filter(pc.equal(cutoff_index.column("dataset"), dataset))

    return rows.select(EXAMPLES_SCHEMA.names)


def skip_summary(cutoff_index: pa.Table) -> dict[str, int]:

    reasons = cutoff_index.column("skip_reason")

    counted = pc.value_counts(pc.drop_null(reasons))

    return {item["values"]: int(item["counts"]) for item in sorted(counted.to_pylist(), key=lambda item: item["values"])}


def cutoff_summary(cutoff_index: pa.Table) -> dict[str, dict[str, int]]:
    """
    По каждому cutoff: сколько валидных и сколько по каждой причине.
    """

    summary: dict[str, dict[str, int]] = {}

    cutoffs = cutoff_index.column("cutoff").to_pylist()
    reasons = cutoff_index.column("skip_reason").to_pylist()
    valids = cutoff_index.column("valid").to_pylist()

    for cutoff, reason, valid in zip(cutoffs, reasons, valids):
        bucket = summary.setdefault(cutoff.isoformat(), {})
        key = "valid" if valid else reason
        bucket[key] = bucket.get(key, 0) + 1

    return summary
