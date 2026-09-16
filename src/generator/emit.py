from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import params as params_module
from . import rng as rng_module
from .config import (
    EVENT_TYPE_PRIORITY,
    GENERATOR_VERSION,
    HISTORY_END,
    HISTORY_START,
    PRESETS,
    RAW_DIR,
    REGISTRY_START,
    SCHEMA_VERSION,
    SEED,
    SOURCES,
    SOURCE_AVAILABILITY,
    SOURCE_PRECISION,
    TIME_PRECISIONS,
    key_catalogue,
)
from .profile import PROFILE_SCHEMA
from .world import communities, geography, merchants, products as product_catalog


# ============================================================
# ВЫГРУЗКА RAW
# ============================================================
#
# Единица симуляции это СООБЩЕСТВО, а runtime-чанк только
# распределяет сообщества между воркерами. Поэтому при одном
# seed и одних параметрах содержимое датасета не зависит ни от
# числа воркеров, ни от размера чанка, ни от порядка завершения.
#
# Контрольная сумма считается по строкам без учёта порядка:
# сумма их отпечатков по модулю 2**128 плюс число строк.
# ============================================================


EVENTS_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("client_id", pa.string()),
        ("event_type", pa.string()),
        ("source", pa.string()),
        ("event_time", pa.timestamp("us")),
        ("record_time", pa.timestamp("us")),
        ("effective_at", pa.timestamp("us")),
        ("time_precision", pa.string()),
        ("sequence_number", pa.int64()),
        ("event_version", pa.int32()),
        ("change_initiator", pa.string()),
        ("correlation_id", pa.string()),
        ("link_type", pa.string()),
        ("is_test_account", pa.bool_()),
        ("payload", pa.string()),
    ]
)

COVERAGE_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("source", pa.string()),
        ("first_available_at", pa.timestamp("us")),
        ("last_available_at", pa.timestamp("us")),
        ("first_seen", pa.timestamp("us")),
        ("coverage_status", pa.string()),
        ("coverage_reason", pa.string()),
        ("opening_state", pa.string()),
    ]
)

TRUTH_EVENTS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("ts", pa.timestamp("us")),
        ("kind", pa.string()),
        ("key", pa.string()),
        ("value", pa.string()),
    ]
)

TRUTH_RELATIONSHIPS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("counterpart_id", pa.string()),
        ("counterpart_kind", pa.string()),
        ("counterpart_client_id", pa.string()),
        ("relation_type", pa.string()),
        ("strength", pa.float64()),
        ("typical_frequency", pa.float64()),
        ("typical_amount_low", pa.int64()),
        ("typical_amount_high", pa.int64()),
        ("valid_from", pa.timestamp("us")),
        ("valid_to", pa.timestamp("us")),
        ("household_id", pa.string()),
    ]
)

GEOGRAPHY_SCHEMA = pa.schema(
    [
        ("settlement_id", pa.string()),
        ("name", pa.string()),
        ("region", pa.string()),
        ("settlement_type", pa.string()),
        ("population_weight", pa.float64()),
        ("districts", pa.string()),
        ("regional_capital", pa.string()),
    ]
)

MERCHANTS_SCHEMA = pa.schema(
    [
        ("outlet_id", pa.string()),
        ("merchant_id", pa.string()),
        ("brand", pa.string()),
        ("merchant_name", pa.string()),
        ("sector", pa.string()),
        ("category", pa.string()),
        ("subcategory", pa.string()),
        ("mcc", pa.string()),
        ("settlement", pa.string()),
        ("region", pa.string()),
        ("settlement_type", pa.string()),
        ("district", pa.string()),
        ("channel", pa.string()),
        ("price_segment", pa.string()),
        ("opening_hour", pa.int32()),
        ("closing_hour", pa.int32()),
        ("popularity", pa.float64()),
        ("country", pa.string()),
        ("is_online", pa.bool_()),
    ]
)

PRODUCTS_SCHEMA = pa.schema(
    [
        ("product_id", pa.string()),
        ("product_code", pa.string()),
        ("product_family", pa.string()),
        ("product_name", pa.string()),
        ("group", pa.string()),
        ("product_version", pa.int32()),
        ("tariff_version", pa.int32()),
        ("status", pa.string()),
        ("valid_from", pa.timestamp("us")),
        ("valid_to", pa.timestamp("us")),
        ("announced_known_period_start", pa.string()),
        ("announced_known_period_end", pa.string()),
        ("announced_date_precision", pa.string()),
        ("announced_simulation_effective_at", pa.timestamp("us")),
        ("sales_start_known_period_start", pa.string()),
        ("sales_start_known_period_end", pa.string()),
        ("sales_start_date_precision", pa.string()),
        ("sales_start_simulation_effective_at", pa.timestamp("us")),
        ("sales_end_known_period_start", pa.string()),
        ("sales_end_known_period_end", pa.string()),
        ("sales_end_date_precision", pa.string()),
        ("sales_end_simulation_effective_at", pa.timestamp("us")),
        ("service_end_known_period_start", pa.string()),
        ("service_end_known_period_end", pa.string()),
        ("service_end_date_precision", pa.string()),
        ("service_end_simulation_effective_at", pa.timestamp("us")),
        ("eligibility", pa.string()),
        ("channels", pa.string()),
        ("terms", pa.string()),
        ("applies_to", pa.string()),
        ("notice_days", pa.int32()),
        ("allow_multiple", pa.bool_()),
        ("max_active_holdings", pa.int32()),
        ("compatibility_rules", pa.string()),
        ("replacement_rules", pa.string()),
        ("predecessor", pa.string()),
        ("successor", pa.string()),
        ("migration_policy", pa.string()),
        ("source_url", pa.string()),
        ("evidence_at", pa.string()),
        ("confidence", pa.string()),
        ("unresolved_source", pa.bool_()),
        ("is_synthetic", pa.bool_()),
        ("note", pa.string()),
    ]
)


TABLES = {
    "events": ("events.parquet", EVENTS_SCHEMA),
    "profile": ("profile.parquet", PROFILE_SCHEMA),
    "source_coverage": ("source_coverage.parquet", COVERAGE_SCHEMA),
    "truth_events": ("truth/events.parquet", TRUTH_EVENTS_SCHEMA),
    "truth_relationships": ("truth/relationships.parquet", TRUTH_RELATIONSHIPS_SCHEMA),
}

PARTS_DIR = "parts"


# ============================================================
# КОНТРОЛЬНАЯ СУММА СОДЕРЖИМОГО
# ============================================================


class ContentDigest:
    """
    Отпечаток набора строк, не зависящий от их порядка.
    """

    MODULUS = 2 ** 128

    def __init__(self) -> None:
        self.total = 0
        self.rows = 0

    def add(self, row: dict) -> None:
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
        self.total = (self.total + int.from_bytes(digest, "little")) % self.MODULUS
        self.rows += 1

    def extend(self, rows: list) -> None:
        for row in rows:
            self.add(row)

    def merge(self, other: tuple) -> None:
        total, rows = other
        self.total = (self.total + total) % self.MODULUS
        self.rows += rows

    def as_tuple(self) -> tuple:
        return (self.total, self.rows)

    def value(self) -> str:
        return hashlib.sha256(
            f"{self.total}:{self.rows}".encode("utf-8")
        ).hexdigest()


# ============================================================
# ЗАПИСЬ
# ============================================================


def _write(path: Path, rows: list, schema: pa.Schema) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows, schema=schema)

    pq.write_table(table, path, compression="zstd")


def _truth_clients_schema(rows: list) -> pa.Schema:

    base = [
        ("client_id", pa.string()),
        ("client_ordinal", pa.int64()),
        ("community_id", pa.int64()),
        ("archetype", pa.string()),
        ("life_stage", pa.string()),
        ("hcb_role", pa.string()),
        ("activity_mode", pa.string()),
        ("settlement", pa.string()),
        ("settlement_type", pa.string()),
        ("true_income", pa.int64()),
        ("visible_share", pa.float64()),
        ("is_test_account", pa.bool_()),
        ("registered_in_window", pa.bool_()),
        ("vanished_after_registration", pa.bool_()),
        ("night_segment", pa.bool_()),
        ("household_id", pa.string()),
        ("final_state", pa.string()),
        ("hidden_cash", pa.int64()),
        ("hidden_other_bank", pa.int64()),
    ]

    known = {name for name, _ in base}

    traits = sorted({name for row in rows for name in row if name not in known})

    return pa.schema(base + [(name, pa.float64()) for name in traits])


# ============================================================
# ЗАДАЧА ВОРКЕРА
# ============================================================


_WORKER: dict = {}


def _worker_init(seed: int, params_path: str | None, catalog_scale: float | None,
                 community_size: int | None) -> None:

    settings = _build_params(params_path, catalog_scale, community_size)

    params_module.activate(settings)
    rng_module.configure(seed, settings.fingerprint())

    _WORKER["ready"] = True


def _build_params(params_path: str | None, catalog_scale: float | None,
                  community_size: int | None):

    settings = params_module.load(params_path)

    overrides: dict = {}

    if catalog_scale is not None:
        overrides["merchants"] = {"catalog_scale": float(catalog_scale)}

    if community_size is not None:
        overrides["relationships"] = {"community_size": int(community_size)}

    if overrides:
        settings = settings.with_overrides(overrides)

    return settings


def _run_batch(job: tuple) -> tuple:
    """
    Пакет сообществ: симуляция и запись своих part-файлов.
    """

    batch_index, community_ids, total_clients, out_dir = job

    from .engine import run_community

    rows: dict[str, list] = {name: [] for name in TABLES}
    rows["truth_clients"] = []

    for community_id in community_ids:

        members = communities.members(community_id, total_clients)

        if not members:
            continue

        result = run_community(community_id, members)

        rows["events"].extend(result.events)
        rows["profile"].extend(result.profile_versions)
        rows["source_coverage"].extend(result.coverage)
        rows["truth_clients"].extend(result.truth_clients)
        rows["truth_events"].extend(result.truth_events)
        rows["truth_relationships"].extend(result.truth_relationships)

    out = Path(out_dir)

    digests: dict[str, tuple] = {}
    counts: dict[str, int] = {}

    for name, (relative, schema) in TABLES.items():
        digest = ContentDigest()
        digest.extend(rows[name])
        digests[name] = digest.as_tuple()
        counts[name] = digest.rows
        _write(out / PARTS_DIR / f"{name}-{batch_index:05d}.parquet", rows[name], schema)

    digest = ContentDigest()
    digest.extend(rows["truth_clients"])
    digests["truth_clients"] = digest.as_tuple()
    counts["truth_clients"] = digest.rows

    _write(
        out / PARTS_DIR / f"truth_clients-{batch_index:05d}.parquet",
        rows["truth_clients"],
        _truth_clients_schema(rows["truth_clients"]),
    )

    return batch_index, digests, counts


# ============================================================
# КАТАЛОГИ
# ============================================================


def _period(period, moment) -> dict:

    if period is None:
        return {"start": None, "end": None, "precision": "unknown", "moment": moment}

    return {
        "start": period.start.isoformat() if period.start else None,
        "end": period.end.isoformat() if period.end else None,
        "precision": period.precision,
        "moment": moment,
    }


def _write_catalogs(out: Path) -> dict:

    catalog = product_catalog.catalog()

    product_rows = []

    for row in catalog.rows:

        announced = _period(row.announced, row.announced_at)
        sales_start = _period(row.sales_start, row.sales_start_at)
        sales_end = _period(row.sales_end, row.sales_end_at)
        service_end = _period(row.service_end, row.service_end_at)

        product_rows.append(
            {
                "product_id": row.product_id,
                "product_code": row.product_code,
                "product_family": row.product_family,
                "product_name": row.product_name,
                "group": row.group,
                "product_version": row.product_version,
                "tariff_version": row.tariff_version,
                "status": row.status,
                "valid_from": row.valid_from,
                "valid_to": None if row.valid_to.year >= 9999 else row.valid_to,
                "announced_known_period_start": announced["start"],
                "announced_known_period_end": announced["end"],
                "announced_date_precision": announced["precision"],
                "announced_simulation_effective_at": announced["moment"],
                "sales_start_known_period_start": sales_start["start"],
                "sales_start_known_period_end": sales_start["end"],
                "sales_start_date_precision": sales_start["precision"],
                "sales_start_simulation_effective_at": sales_start["moment"],
                "sales_end_known_period_start": sales_end["start"],
                "sales_end_known_period_end": sales_end["end"],
                "sales_end_date_precision": sales_end["precision"],
                "sales_end_simulation_effective_at": sales_end["moment"],
                "service_end_known_period_start": service_end["start"],
                "service_end_known_period_end": service_end["end"],
                "service_end_date_precision": service_end["precision"],
                "service_end_simulation_effective_at": service_end["moment"],
                "eligibility": json.dumps(row.eligibility, ensure_ascii=False, default=str),
                "channels": json.dumps(list(row.channels), ensure_ascii=False),
                "terms": json.dumps(row.terms, ensure_ascii=False, default=str),
                "applies_to": row.applies_to,
                "notice_days": row.notice_days,
                "allow_multiple": row.allow_multiple,
                "max_active_holdings": row.max_active_holdings,
                "compatibility_rules": json.dumps(row.compatibility_rules, ensure_ascii=False),
                "replacement_rules": json.dumps(row.replacement_rules, ensure_ascii=False),
                "predecessor": row.predecessor,
                "successor": row.successor,
                "migration_policy": row.migration_policy,
                "source_url": row.source_url,
                "evidence_at": row.evidence_at.isoformat() if row.evidence_at else None,
                "confidence": row.confidence,
                "unresolved_source": row.unresolved_source,
                "is_synthetic": row.is_synthetic,
                "note": row.note,
            }
        )

    _write(out / "catalog" / "products.parquet", product_rows, PRODUCTS_SCHEMA)

    geography_rows = [
        {
            "settlement_id": item.settlement_id,
            "name": item.name,
            "region": item.region,
            "settlement_type": item.settlement_type,
            "population_weight": item.population_weight,
            "districts": json.dumps(list(item.districts), ensure_ascii=False),
            "regional_capital": item.regional_capital,
        }
        for item in geography.settlements()
    ]

    _write(out / "catalog" / "geography.parquet", geography_rows, GEOGRAPHY_SCHEMA)

    merchant_rows = []

    for item in merchants.iter_catalog():
        merchant_rows.append(
            {
                "outlet_id": item.outlet_id,
                "merchant_id": item.merchant_id,
                "brand": item.brand,
                "merchant_name": item.merchant_name,
                "sector": item.sector,
                "category": item.category,
                "subcategory": item.subcategory,
                "mcc": item.mcc,
                "settlement": item.settlement,
                "region": item.region,
                "settlement_type": item.settlement_type,
                "district": item.district,
                "channel": item.channel,
                "price_segment": item.price_segment,
                "opening_hour": item.opening_hour,
                "closing_hour": item.closing_hour,
                "popularity": item.popularity,
                "country": item.country,
                "is_online": item.is_online,
            }
        )

    _write(out / "catalog" / "merchants.parquet", merchant_rows, MERCHANTS_SCHEMA)

    return {
        "products": len(product_rows),
        "geography": len(geography_rows),
        "merchants": len(merchant_rows),
        "timeline_sha256": catalog.timeline_sha256,
        "unresolved_sources": len(catalog.unresolved_sources),
    }


# ============================================================
# СКЛЕЙКА
# ============================================================


def _merge_parts(out: Path, name: str, batches: int, schema: pa.Schema | None = None) -> int:

    relative = TABLES[name][0] if name in TABLES else f"truth/clients.parquet"

    target = out / relative

    target.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = 0

    for index in range(batches):

        part = out / PARTS_DIR / f"{name}-{index:05d}.parquet"

        if not part.exists():
            continue

        table = pq.read_table(part)

        if writer is None:
            writer = pq.ParquetWriter(target, table.schema, compression="zstd")

        if table.num_rows:
            writer.write_table(table)
            rows += table.num_rows

        part.unlink()

    if writer is None:
        empty = schema or (TABLES[name][1] if name in TABLES else None)
        if empty is not None:
            _write(target, [], empty)
    else:
        writer.close()

    return rows


def _file_hashes(out: Path) -> dict:

    hashes = {}

    for path in sorted(out.rglob("*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(out)).replace("\\", "/")] = digest

    return hashes


# ============================================================
# ГЕНЕРАЦИЯ
# ============================================================


def generate_dataset(
    total_clients: int,
    out_dir: Path,
    seed: int = SEED,
    workers: int = 1,
    chunk_clients: int = 256,
    params_path: str | None = None,
    catalog_scale: float | None = None,
    community_size: int | None = None,
    resume: bool = False,
    quiet: bool = False,
) -> dict:

    out = Path(out_dir)

    settings = _build_params(params_path, catalog_scale, community_size)

    params_module.activate(settings)
    rng_module.configure(seed, settings.fingerprint())

    if out.exists() and not resume:
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    size = settings.relationships.community_size

    community_count = communities.community_count(total_clients)

    per_batch = max(1, chunk_clients // size)

    batches = [
        tuple(range(start, min(start + per_batch, community_count)))
        for start in range(0, community_count, per_batch)
    ]

    jobs = [
        (index, community_ids, total_clients, str(out))
        for index, community_ids in enumerate(batches)
    ]

    if resume:
        jobs = [
            job
            for job in jobs
            if not (out / PARTS_DIR / f"events-{job[0]:05d}.parquet").exists()
        ]

    digests = {name: ContentDigest() for name in list(TABLES) + ["truth_clients"]}
    counts = {name: 0 for name in list(TABLES) + ["truth_clients"]}

    done = 0

    def report(result) -> None:
        nonlocal done
        _, batch_digests, batch_counts = result
        done += 1
        for name, value in batch_digests.items():
            digests[name].merge(value)
            counts[name] += batch_counts[name]
        if not quiet:
            print(f"batches: {done}/{len(jobs)}  events: {counts['events']:,}")

    if workers <= 1 or len(jobs) <= 1:
        _worker_init(seed, params_path, catalog_scale, community_size)
        for job in jobs:
            report(_run_batch(job))
    else:
        with Pool(
            processes=min(workers, len(jobs)),
            initializer=_worker_init,
            initargs=(seed, params_path, catalog_scale, community_size),
        ) as pool:
            for result in pool.imap_unordered(_run_batch, jobs):
                report(result)

    for name in TABLES:
        _merge_parts(out, name, len(batches))

    _merge_parts(out, "truth_clients", len(batches))

    parts_dir = out / PARTS_DIR

    if parts_dir.exists():
        for leftover in parts_dir.iterdir():
            leftover.unlink()
        parts_dir.rmdir()

    catalog_info = _write_catalogs(out)

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "total_clients": total_clients,
        "community_size": size,
        "communities": community_count,
        "chunk_clients": chunk_clients,
        "history_start": HISTORY_START.isoformat(),
        "history_end": HISTORY_END.isoformat(),
        "registry_start": REGISTRY_START.isoformat(),
        "extract_time": HISTORY_END.isoformat(),
        "sources": {
            source: {
                "available_from": SOURCE_AVAILABILITY[source].isoformat(),
                "time_precision": SOURCE_PRECISION[source],
                "defect_profile": {
                    "record_delay_minutes": list(
                        settings.defects.record_delay_minutes.get(source, (0, 60))
                    ),
                    "duplicate_share": settings.defects.duplicate_share.get(source, 0.0),
                    "correction_share": settings.defects.correction_share.get(source, 0.0),
                    "late_arrival_share": settings.defects.late_arrival_share.get(source, 0.0),
                    "outage_days_per_year": settings.defects.outage_days_per_year.get(source, 0.0),
                },
            }
            for source in SOURCES
        },
        "time_precisions": list(TIME_PRECISIONS),
        "event_type_priority": EVENT_TYPE_PRIORITY,
        "key_catalogue": key_catalogue(),
        "schema_changes": [dict(item) for item in settings.defects.schema_changes],
        "conflict_rules": [
            "подтверждённое profile_change важнее анкеты заявки",
            "анкета заявки важнее системного пересчёта профиля",
            "системный пересчёт важнее косвенных признаков транзакций",
            "при равном record_time выигрывает большая event_version",
        ],
        "bank_timeline": [
            {key: (value.isoformat() if isinstance(value, (date, datetime)) else value)
             for key, value in item.items()}
            for item in product_catalog.catalog().bank_timeline
        ],
        "product_timeline_sha256": catalog_info["timeline_sha256"],
        "unresolved_sources": catalog_info["unresolved_sources"],
        "catalog_rows": {
            "products": catalog_info["products"],
            "geography": catalog_info["geography"],
            "merchants": catalog_info["merchants"],
        },
        "rows": counts,
        "content_sha256": {name: digests[name].value() for name in digests},
        "generation_config": settings.as_dict(),
        "generation_config_sha256": settings.fingerprint(),
        "calibration_targets": settings.calibration.as_list(),
    }

    manifest["file_sha256"] = _file_hashes(out)

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
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
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--chunk-clients", type=int, default=256)
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--catalog-scale", type=float, default=None)
    parser.add_argument("--community-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.clients is not None:
        total = args.clients
        out = args.out or RAW_DIR / f"clients_{total}"
    else:
        total = PRESETS[args.preset]
        out = args.out or RAW_DIR / args.preset

    counts = generate_dataset(
        total_clients=total,
        out_dir=out,
        seed=args.seed,
        workers=args.workers,
        chunk_clients=args.chunk_clients,
        params_path=args.params,
        catalog_scale=args.catalog_scale,
        community_size=args.community_size,
        resume=args.resume,
    )

    print()
    print("=" * 60)
    print("RAW DATASET GENERATED")
    print("=" * 60)

    for name, value in counts.items():
        print(f"{name:24s}{value:,}")

    print()
    print(f"output: {out}")


if __name__ == "__main__":
    main()
