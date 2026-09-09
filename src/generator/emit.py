from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    EVENT_TYPE_PRIORITY,
    FEATURE_END,
    HISTORY_START,
    LABEL_END,
    MAX_EVENTS_PER_HISTORY,
    MAX_TOKENS_PER_EVENT,
    PRESETS,
    PROFILE_FIELDS,
    RAW_DIR,
    SEED,
    SOURCE_AVAILABILITY,
)
from .coverage import coverage_rows
from .derive import derive_labels
from .history import EVENT_TABLES, generate_client_history, observed
from .profile import snapshot_row
from .timeline import timeline_rows
from .version import (
    CONFIG_KEY,
    DEFAULT_VERSION,
    MANIFEST_KEY,
    RAW_SCHEMA_REVISION,
    REVISION_KEY,
    V1,
    VERSIONS,
    check_version,
    generation_config,
    revision_for,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Клиенты независимы и детерминированы по ключу, поэтому
# датасет режется на чанки и генерируется параллельно.
#
# Каждый чанк пишет свои part-файлы, затем главный процесс
# склеивает их в порядке чанков: результат не зависит
# ни от числа воркеров, ни от порядка завершения.
#
# В RAW попадает только окно признаков: history.before(FEATURE_END).
# Будущее нужно исключительно метке.
#
# Вывод: data/raw/<preset>/<table>.parquet + manifest.json
# ============================================================


# ============================================================
# СХЕМЫ
# ============================================================

PROFILE_FIELD_TYPES: dict[str, pa.DataType] = {
    "age": pa.int64(),
    "gender": pa.string(),
    "family_status": pa.string(),
    "children": pa.int64(),
    "education": pa.string(),
    "region": pa.string(),
    "housing_type": pa.string(),
    "pensioner": pa.bool_(),
    "income_type": pa.string(),
    "declared_income": pa.int64(),
    "industry": pa.string(),
    "salary_day": pa.int64(),
    "relationship_months": pa.int64(),
    "contracts_count": pa.int64(),
    "active_contracts": pa.int64(),
    "holds_credit_card": pa.bool_(),
    "holds_debit_card": pa.bool_(),
    "holds_deposit": pa.bool_(),
    "credit_limit": pa.float64(),
    "credit_utilization": pa.float64(),
}


SCHEMAS: dict[str, pa.Schema] = {
    "profile": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("snapshot_month", pa.timestamp("us")),
        ]
        + [(field, PROFILE_FIELD_TYPES[field]) for field in PROFILE_FIELDS]
    ),

    "transactions": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("amount", pa.int64()),
            ("direction", pa.string()),
            ("mcc", pa.string()),
            ("merchant_city", pa.string()),
            ("merchant_country", pa.string()),
            ("is_online", pa.bool_()),
            ("is_subscription", pa.bool_()),
        ]
    ),

    "product_events": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("product_type", pa.string()),
            ("amount_or_limit", pa.float64()),
            ("term", pa.int64()),
            ("product_subtype", pa.string()),
            ("timestamp_quality", pa.string()),
        ]
    ),

    "communications": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("channel", pa.string()),
            ("template", pa.string()),
            ("day_of_week", pa.int64()),
            ("hour", pa.int64()),
            ("delivered", pa.bool_()),
        ]
    ),

    "app_screens": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("session_id", pa.string()),
            ("firebase_screen", pa.string()),
            ("product", pa.string()),
            ("funnel_stage", pa.string()),
            ("reject_reason", pa.string()),
        ]
    ),

    "app_operations": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("domain", pa.string()),
            ("operation", pa.string()),
            ("status", pa.string()),
        ]
    ),

    "banners": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("slot", pa.string()),
            ("offer", pa.string()),
            ("action", pa.string()),
        ]
    ),

    "source_coverage": pa.schema(
        [
            ("client_id", pa.int64()),
            ("source", pa.string()),
            ("availability_start", pa.timestamp("us")),
            ("first_seen", pa.timestamp("us")),
        ]
    ),

    "timeline": pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", pa.timestamp("us")),
            ("seq", pa.int64()),
            ("event_type", pa.string()),
            ("payload", pa.string()),
        ]
    ),

    "labels": pa.schema(
        [
            ("client_id", pa.int64()),
            ("label_start", pa.timestamp("us")),
            ("label_end", pa.timestamp("us")),
            ("product_open_90d", pa.bool_()),
        ]
    ),
}

# ============================================================
# РЕВИЗИИ СХЕМЫ
# ============================================================
#
# SCHEMAS выше это ревизия 1, и она заморожена: её байты
# лежат на диске и проверяются золотыми хэшами.
#
# В ревизии 2 app_operations и banners получают session_id
# ровно там же, где он стоит у app_screens, сразу после ts.
# Место важно: payload ленты это names[2:], то есть порядок
# колонок и порядок ключей payload это одно и то же.
# ============================================================


def _with_session_id(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [schema.field(0), schema.field(1), pa.field("session_id", pa.string())]
        + [schema.field(index) for index in range(2, len(schema.names))]
    )


SCHEMAS_R2: dict[str, pa.Schema] = {
    **SCHEMAS,
    "app_operations": _with_session_id(SCHEMAS["app_operations"]),
    "banners": _with_session_id(SCHEMAS["banners"]),
}

SCHEMAS_BY_REVISION: dict[int, dict[str, pa.Schema]] = {1: SCHEMAS, 2: SCHEMAS_R2}


def schemas_for(revision: int) -> dict[str, pa.Schema]:

    if revision not in SCHEMAS_BY_REVISION:
        raise ValueError(f"неизвестная ревизия схемы RAW: {revision!r}")

    return SCHEMAS_BY_REVISION[revision]


PARTS_DIR = "parts"


def table_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.parquet"


def part_path(out_dir: Path, name: str, chunk_index: int) -> Path:
    return out_dir / PARTS_DIR / f"{name}-{chunk_index:05d}.parquet"


# ============================================================
# PARQUET SINK
# ============================================================


class ParquetSink:

    def __init__(self, paths: dict[str, Path], schemas: dict[str, pa.Schema] = SCHEMAS) -> None:

        self.paths = paths
        self.schemas = schemas
        self.writers: dict[str, pq.ParquetWriter | None] = {name: None for name in schemas}
        self.counts: dict[str, int] = {name: 0 for name in schemas}

    def write(self, name: str, records: list[dict]) -> None:

        if not records:
            return

        # Лишние ключи словаря схема отбрасывает молча: событие
        # ревизии 2 в ревизии 1 теряет session_id и совпадает
        # с прежними байтами.
        table = pa.Table.from_pylist(records, schema=self.schemas[name])

        writer = self.writers[name]

        if writer is None:
            self.paths[name].parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(self.paths[name], self.schemas[name], compression="zstd")
            self.writers[name] = writer

        writer.write_table(table)

        self.counts[name] += len(records)

    def close(self) -> None:

        for name, writer in self.writers.items():

            if writer is not None:
                writer.close()
                continue

            self.paths[name].parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist([], schema=self.schemas[name]),
                self.paths[name],
                compression="zstd",
            )


# ============================================================
# ОДИН ЧАНК
# ============================================================

DEFAULT_CHUNK_CLIENTS = 25


def raw_root(version: str) -> Path:
    """
    Корень RAW для версии: v1 остаётся на прежнем месте,
    v2 уходит в отдельный подкаталог и ничего не перезаписывает.
    """

    return RAW_DIR if version == V1 else RAW_DIR / version


def generate_chunk(
    first_client: int,
    last_client: int,
    paths: dict[str, Path],
    version: str = DEFAULT_VERSION,
) -> dict[str, int]:
    """
    Клиенты [first_client, last_client) в файлы paths.
    """

    check_version(version)

    schemas = schemas_for(revision_for(version))

    sink = ParquetSink(paths, schemas)

    buffers: dict[str, list[dict]] = {name: [] for name in schemas}

    for client_id in range(first_client, last_client):

        history = generate_client_history(
            client_id=client_id,
            start=HISTORY_START,
            end=LABEL_END,
            version=version,
        )

        # Окно признаков в наблюдаемом виде: шум применён один раз,
        # дальше и таблицы, и лента строятся из этого же среза.
        feature = observed(history.before(FEATURE_END))

        # ----------------------------------------------------
        # ПОТОКИ СОБЫТИЙ
        # ----------------------------------------------------

        for table_name in EVENT_TABLES:
            for event in feature.events(table_name):
                buffers[table_name].append(asdict(event))

        # ----------------------------------------------------
        # ПРОФИЛЬ
        # ----------------------------------------------------

        for snapshot in feature.profile:
            buffers["profile"].append(snapshot_row(snapshot))

        # ----------------------------------------------------
        # ЛЕНТА
        # ----------------------------------------------------

        buffers["timeline"].extend(timeline_rows(feature))

        # ----------------------------------------------------
        # ПОКРЫТИЕ И МЕТКА
        # ----------------------------------------------------

        buffers["source_coverage"].extend(coverage_rows(client_id))

        buffers["labels"].append(asdict(derive_labels(history)))

    for table_name, records in buffers.items():
        sink.write(table_name, records)

    sink.close()

    return sink.counts


def _chunk_job(args: tuple) -> tuple[int, dict[str, int]]:

    chunk_index, first, last, out_dir, version = args

    paths = {name: part_path(out_dir, name, chunk_index) for name in SCHEMAS}

    return chunk_index, generate_chunk(first, last, paths, version)


# ============================================================
# СКЛЕЙКА
# ============================================================


def merge_parts(out_dir: Path, chunk_count: int, revision: int = 1) -> None:

    for name, schema in schemas_for(revision).items():

        writer = pq.ParquetWriter(table_path(out_dir, name), schema, compression="zstd")

        for chunk_index in range(chunk_count):

            part = part_path(out_dir, name, chunk_index)

            table = pq.read_table(part)

            if table.num_rows:
                writer.write_table(table)

            part.unlink()

        writer.close()

    parts_dir = out_dir / PARTS_DIR

    if parts_dir.exists() and not any(parts_dir.iterdir()):
        parts_dir.rmdir()


# ============================================================
# ДАТАСЕТ
# ============================================================


def clean_output(out_dir: Path) -> None:

    for name in SCHEMAS:
        path = table_path(out_dir, name)
        if path.exists():
            path.unlink()

    parts_dir = out_dir / PARTS_DIR

    if parts_dir.exists():
        for part in parts_dir.iterdir():
            part.unlink()
        parts_dir.rmdir()

    manifest = out_dir / "manifest.json"

    if manifest.exists():
        manifest.unlink()


def generate_dataset(
    total_clients: int,
    chunk_clients: int = DEFAULT_CHUNK_CLIENTS,
    out_dir: Path | None = None,
    workers: int = 1,
    version: str = DEFAULT_VERSION,
) -> dict[str, int]:

    check_version(version)

    out_dir = Path(out_dir) if out_dir is not None else raw_root(version) / f"clients_{total_clients}"

    out_dir.mkdir(parents=True, exist_ok=True)

    clean_output(out_dir)

    starts = list(range(0, total_clients, chunk_clients))

    jobs = [
        (index, first, min(first + chunk_clients, total_clients), out_dir, version)
        for index, first in enumerate(starts)
    ]

    counts: dict[str, int] = {name: 0 for name in SCHEMAS}

    done = 0

    def report(chunk_counts: dict[str, int]) -> None:
        nonlocal done
        done += 1
        for name, value in chunk_counts.items():
            counts[name] += value
        print(f"chunks: {done}/{len(jobs)}  clients: {counts['labels']:,}/{total_clients:,}")

    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            _, chunk_counts = _chunk_job(job)
            report(chunk_counts)
    else:
        with Pool(processes=min(workers, len(jobs))) as pool:
            for _, chunk_counts in pool.imap_unordered(_chunk_job, jobs):
                report(chunk_counts)

    merge_parts(out_dir, len(jobs), revision_for(version))

    manifest = {
        "seed": SEED,
        "total_clients": total_clients,
        "chunk_clients": chunk_clients,
        "history_start": HISTORY_START.isoformat(),
        "feature_end": FEATURE_END.isoformat(),
        "label_end": LABEL_END.isoformat(),
        "source_availability": {
            source: ts.isoformat() for source, ts in SOURCE_AVAILABILITY.items()
        },
        "event_type_priority": EVENT_TYPE_PRIORITY,
        "max_tokens_per_event": MAX_TOKENS_PER_EVENT,
        "max_events_per_history": MAX_EVENTS_PER_HISTORY,
        "rows": counts,
    }

    # Манифесты v1 уже лежат на диске: добавление ключей
    # сделало бы их невоспроизводимыми побайтово.
    if version != V1:
        manifest[MANIFEST_KEY] = version
        manifest[REVISION_KEY] = revision_for(version)
        manifest[CONFIG_KEY] = generation_config(version)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return counts


# ============================================================
# CLI
# ============================================================


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def main() -> None:

    parser = argparse.ArgumentParser(description="Генерация RAW-датасета")

    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--chunk-clients", type=int, default=DEFAULT_CHUNK_CLIENTS)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--version", choices=VERSIONS, default=DEFAULT_VERSION)

    args = parser.parse_args()

    root = raw_root(args.version)

    if args.clients is not None:
        total_clients = args.clients
        out_dir = args.out or root / f"clients_{total_clients}"
    else:
        total_clients = PRESETS[args.preset]
        out_dir = args.out or root / args.preset

    counts = generate_dataset(
        total_clients=total_clients,
        chunk_clients=args.chunk_clients,
        out_dir=out_dir,
        workers=args.workers,
        version=args.version,
    )

    print()
    print("=" * 60)
    print(f"RAW DATASET GENERATED  ({args.version})")
    print("=" * 60)

    for table_name, count in counts.items():
        print(f"{table_name:20s}{count:,}")

    print()
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
