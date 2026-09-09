from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.generator.config import PROFILE_FIELDS
from src.generator.version import RAW_SCHEMA_REVISION

from .artifacts import TableWriter, write_table
from .buckets import BUCKET_DTYPE, BucketSpec, apply_buckets
from .config import (
    CLIENT_GROUPS,
    DATASET_NAMES,
    EVENT_TYPES,
    KIND_METADATA,
    FieldSpec,
    payload_fields,
    specs_for,
)
from .cutoffs import EXAMPLES_SCHEMA, examples_of
from .raw import RawDataset, parse_payloads


# ============================================================
# ИДЕЯ
# ============================================================
#
# Второй проход: широкая типизированная таблица событий.
#
# Одна строка это одно событие ленты. Поля payload разложены
# по колонкам <event_type>__<field> со своим типом; строка
# заполняет только колонки своего типа, остальные null.
# Рядом с numeric-полем лежит <...>__bucket, посчитанный по
# границам, обученным на train.
#
# События хранятся по группам клиентов один раз, а примеры это
# ссылки (client_id, cutoff, seq_end, snapshot_ts). Потребитель
# ОБЯЗАН брать префикс seq < seq_end: события train-клиентов
# намеренно содержат val- и test-месяцы, они нужны примерам
# val_time и test_time.
# ============================================================


EVENTS_KEYS = ("client_id", "seq", "ts", "event_type")

PROFILE_KEYS = ("client_id", "ts", "snapshot_month")


def event_field_specs(event_type: str, revision: int = RAW_SCHEMA_REVISION) -> list[FieldSpec]:
    """
    Поля payload типа события в порядке контракта.
    """

    from .config import REGISTRY

    return [
        REGISTRY[(event_type, name)] for name in payload_fields(event_type, revision)
    ]


def events_schema(revision: int = RAW_SCHEMA_REVISION) -> pa.Schema:
    """
    Широкая схема событий processed.

    Зависит от ревизии схемы RAW: набор ревизии 1 не знает
    session_id у операций и баннеров, и добавлять ему пустые
    колонки нельзя. Иначе уже собранные наборы перестали бы
    открываться из-за поля, которого у них и не было.
    """

    fields = [
        ("client_id", pa.int64()),
        ("seq", pa.int64()),
        ("ts", pa.timestamp("us")),
        ("event_type", pa.string()),
    ]

    for event_type in EVENT_TYPES:
        for spec in event_field_specs(event_type, revision):
            fields.append((spec.column, spec.arrow_type))
            if spec.is_numeric:
                fields.append((spec.bucket_column, BUCKET_DTYPE))

    return pa.schema(fields)


def profile_schema() -> pa.Schema:

    from .config import REGISTRY

    fields = [
        ("client_id", pa.int64()),
        ("ts", pa.timestamp("us")),
        ("snapshot_month", pa.timestamp("us")),
    ]

    for name in PROFILE_FIELDS:
        spec = REGISTRY[("profile", name)]
        fields.append((spec.field, spec.arrow_type))
        if spec.is_numeric:
            fields.append((f"{spec.field}__bucket", BUCKET_DTYPE))

    return pa.schema(fields)


# ============================================================
# РАЗБРОС В ШИРОКУЮ ТАБЛИЦУ
# ============================================================


def scatter(column: pa.Array, mask: np.ndarray, length: int, arrow_type: pa.DataType) -> pa.Array:
    """
    Значения плотной колонки раскладываются по позициям mask,
    остальные позиции становятся null.
    """

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    # Позиция строки внутри плотной колонки; вне маски берётся
    # добавленный в хвост null.
    positions = np.full(length, len(column), dtype=np.int64)
    positions[mask] = np.arange(len(column), dtype=np.int64)

    padded = pa.concat_arrays([column.cast(arrow_type), pa.nulls(1, arrow_type)])

    return pc.take(padded, pa.array(positions))


def widen_batch(batch: pa.Table, buckets: dict[tuple[str, str], BucketSpec], schema: pa.Schema) -> pa.Table:
    """
    Батч ленты в широкую типизированную форму.
    """

    length = batch.num_rows

    columns: dict[str, pa.Array] = {
        "client_id": batch.column("client_id").combine_chunks(),
        "seq": batch.column("seq").combine_chunks(),
        "ts": batch.column("ts").combine_chunks(),
        "event_type": batch.column("event_type").combine_chunks(),
    }

    event_type_column = batch.column("event_type")

    for event_type in EVENT_TYPES:

        mask = pc.equal(event_type_column, event_type).to_numpy(zero_copy_only=False)
        mask = np.asarray(mask, dtype=bool)

        specs = event_field_specs(event_type)

        if not mask.any():
            for spec in specs:
                columns[spec.column] = pa.nulls(length, spec.arrow_type)
                if spec.is_numeric:
                    columns[spec.bucket_column] = pa.nulls(length, BUCKET_DTYPE)
            continue

        typed = batch.filter(pa.array(mask))

        parsed = parse_payloads(event_type, typed.column("payload"))

        for spec in specs:

            dense = parsed.column(spec.field).combine_chunks()

            columns[spec.column] = scatter(dense, mask, length, spec.arrow_type)

            if spec.is_numeric:
                dense_buckets = apply_buckets(dense, buckets[spec.key])
                columns[spec.bucket_column] = scatter(dense_buckets, mask, length, BUCKET_DTYPE)

    return pa.table({name: columns[name] for name in schema.names}, schema=schema)


def widen_profile(profile: pa.Table, buckets: dict[tuple[str, str], BucketSpec], schema: pa.Schema) -> pa.Table:

    columns: dict[str, pa.Array] = {name: profile.column(name).combine_chunks() for name in PROFILE_KEYS}

    for spec in specs_for("profile"):

        if spec.kind == KIND_METADATA:
            continue

        column = profile.column(spec.field).combine_chunks()

        columns[spec.field] = column

        if spec.is_numeric:
            columns[f"{spec.field}__bucket"] = apply_buckets(column, buckets[spec.key])

    return pa.table({name: columns[name] for name in schema.names}, schema=schema)


# ============================================================
# ЗАПИСЬ
# ============================================================


def group_masks(client_ids: np.ndarray, groups: dict[int, str]) -> dict[str, np.ndarray]:

    assigned = np.array([groups.get(int(client_id), "") for client_id in client_ids], dtype=object)

    return {group: (assigned == group) for group in CLIENT_GROUPS}


def write_processed(
    raw: RawDataset,
    cutoff_index: pa.Table,
    groups: dict[int, str],
    buckets: dict[tuple[str, str], BucketSpec],
    out_dir: Path,
) -> dict[str, int]:
    """
    Пишет события и профиль по группам клиентов, примеры по
    датасетам и общий индекс кандидатов.
    """

    out_dir = Path(out_dir)

    counts: dict[str, int] = {}

    schema = events_schema(raw.manifest.revision)

    writers = {
        group: TableWriter(out_dir / "clients" / f"{group}_clients" / "events.parquet", schema)
        for group in CLIENT_GROUPS
    }

    for batch in raw.iter_row_groups("timeline"):

        wide = widen_batch(batch, buckets, schema)

        masks = group_masks(batch.column("client_id").to_numpy(), groups)

        for group, mask in masks.items():
            if mask.any():
                writers[group].write(wide.filter(pa.array(mask)))

    for group, writer in writers.items():
        counts[f"clients/{group}_clients/events"] = writer.close()

    # --------------------------------------------------------
    # ПРОФИЛЬ
    # --------------------------------------------------------

    profile_out = profile_schema()

    profile = raw.read("profile")

    wide_profile = widen_profile(profile, buckets, profile_out)

    masks = group_masks(profile.column("client_id").to_numpy(), groups)

    for group, mask in masks.items():

        rows = wide_profile.filter(pa.array(mask)) if mask.any() else profile_out.empty_table()

        path = out_dir / "clients" / f"{group}_clients" / "profile.parquet"

        write_table(path, rows, profile_out)

        counts[f"clients/{group}_clients/profile"] = rows.num_rows

    # --------------------------------------------------------
    # ПРИМЕРЫ И ИНДЕКС
    # --------------------------------------------------------

    for dataset in DATASET_NAMES:

        rows = examples_of(cutoff_index, dataset)

        write_table(out_dir / dataset / "examples.parquet", rows, EXAMPLES_SCHEMA)

        counts[f"{dataset}/examples"] = rows.num_rows

    write_table(out_dir / "cutoff_index.parquet", cutoff_index, cutoff_index.schema)

    counts["cutoff_index"] = cutoff_index.num_rows

    return counts
