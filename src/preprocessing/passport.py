from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .artifacts import _md_table
from .rawdata import (
    CONTENT_TABLES,
    ParsedPayload,
    RawContractError,
    RawDataset,
    iter_event_types,
    parse_payloads,
)
from .settings import GROUPS, PreprocessingConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 1: понять, что пришло, за какой период и можно ли этим
# данным доверять. Ничего не преобразуется и не обучается.
#
# Два разных вердикта, которые нельзя смешивать:
#
#   errors        структурные поломки: нет файла, не сошлись
#                 контрольные суммы или схема. Группа блокируется.
#   limitations   честные ограничения данных: горизонт короче
#                 согласованного, источник подключён поздно,
#                 нарушения контракта payload в отдельных строках.
#                 Обработка возможна, готовность не объявляется.
#
# Пустой месяц у источника — не ошибка: период генерации,
# доступность источника и реальная активность различаются и
# печатаются отдельно.
# ============================================================


STAGE = "passport"
STAGE_VERSION = "5.0.0"

# Статусы по возрастанию тяжести. Ready означает «данные пригодны
# целиком»; всё остальное запрещает объявлять набор готовым.
STATUS_READY = "ready"
STATUS_HORIZON_SHORT = "horizon_short"
STATUS_CONTRACT_MISMATCH = "contract_mismatch"
STATUS_BLOCKED = "blocked"

STATUS_ORDER: tuple[str, ...] = (
    STATUS_READY,
    STATUS_HORIZON_SHORT,
    STATUS_CONTRACT_MISMATCH,
    STATUS_BLOCKED,
)

# Статусы, при которых препроцессинг не должен идти дальше без
# явного разрешения на диагностику.
NOT_USABLE: frozenset[str] = frozenset({STATUS_CONTRACT_MISMATCH, STATUS_BLOCKED})

# Виды нарушений контракта payload: каждое означает, что данные
# расходятся с собственным каталогом ключей манифеста.
CONTRACT_VIOLATION_KINDS: tuple[str, ...] = (
    "missing_required",
    "type_mismatch",
    "unexpected_key",
    "unparseable",
    "null_payload",
)

# Структурно обязательные поля конверта: без любого из них
# строка необрабатываема. Пустое значение здесь не «расхождение
# с каталогом ключей», а поломка выгрузки: порядок и связи по
# такой строке построить нельзя, а молча выбросить её нельзя
# тем более. Поэтому null в них блокирует группу, и
# диагностический флаг этого не снимает.
#
# payload в список не входит: пустой payload canonical уже
# обрабатывает как null_payload и строку сохраняет с причиной.
REQUIRED_ENVELOPE_FIELDS: tuple[str, ...] = (
    "event_id",
    "client_id",
    "event_type",
    "source",
    "event_time",
)

MONTH_NOT_GENERATED = "—"
MONTH_SOURCE_UNAVAILABLE = "n/a"


# ============================================================
# АККУМУЛЯТОРЫ
# ============================================================


@dataclass
class PayloadStats:
    rows: int = 0
    nulls: Counter = field(default_factory=Counter)
    violations: Counter = field(default_factory=Counter)
    by_field: Counter = field(default_factory=Counter)
    samples: list[dict] = field(default_factory=list)

    def merge(self, parsed: ParsedPayload) -> None:
        self.rows += parsed.table.num_rows
        for name in parsed.table.column_names:
            self.nulls[name] += parsed.table.column(name).null_count
        self.violations.update(parsed.counts)
        self.by_field.update(parsed.by_field)
        room = ParsedPayload.SAMPLE_LIMIT - len(self.samples)
        if room > 0:
            self.samples.extend(parsed.samples[:room])


@dataclass
class EventsScan:
    rows: int = 0
    row_groups: int = 0
    by_source: Counter = field(default_factory=Counter)
    by_type: Counter = field(default_factory=Counter)
    event_time_min: datetime | None = None
    event_time_max: datetime | None = None
    pair_hashes: list[np.ndarray] = field(default_factory=list)
    clients: set = field(default_factory=set)
    month_counts: Counter = field(default_factory=Counter)
    before_start_by_source: Counter = field(default_factory=Counter)
    before_start_by_type: Counter = field(default_factory=Counter)
    after_extract_by_source: Counter = field(default_factory=Counter)
    payload: dict[str, PayloadStats] = field(default_factory=dict)
    unknown_types: Counter = field(default_factory=Counter)
    required_nulls: Counter = field(default_factory=Counter)

    # Клиенты, чьи строки лежат в файле не подряд. Весь
    # препроцессинг читает ленту блоками целых клиентов, и
    # разорванный блок молча теряет часть истории.
    split_clients: set = field(default_factory=set)
    finished_clients: set = field(default_factory=set)
    current_client: str | None = None


def _run_starts(values: np.ndarray) -> list:
    """
    Первое значение каждого непрерывного блока.
    """

    if values.size == 0:
        return []

    change = np.empty(values.size, dtype=bool)
    change[0] = True
    change[1:] = values[1:] != values[:-1]

    return values[change].tolist()


def _counter(column: pa.Array | pa.ChunkedArray) -> Counter:
    counted = pc.value_counts(column)
    return Counter(
        {("null" if item["values"] is None else str(item["values"])): int(item["counts"]) for item in counted.to_pylist()}
    )


def _min_max(column: pa.Array | pa.ChunkedArray) -> tuple[datetime | None, datetime | None]:
    if column.null_count == len(column):
        return None, None
    stats = pc.min_max(column).as_py()
    return stats["min"], stats["max"]


def _merge_min(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    return candidate if current is None or candidate < current else current


def _merge_max(current: datetime | None, candidate: datetime | None) -> datetime | None:
    if candidate is None:
        return current
    return candidate if current is None or candidate > current else current


def _hash_ids(event_id: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """
    uint64-отпечаток event_id без хранения самих строк: повторы
    находятся через np.unique в конце.

    Считается blake2b: pandas ради одной хеш-функции держать
    незачем, а его реализация между версиями не обещана
    стабильной.
    """

    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")
            for value in event_id.to_pylist()
        ),
        dtype=np.uint64,
        count=len(event_id),
    )


def scan_events(raw: RawDataset, config: PreprocessingConfig) -> EventsScan:

    manifest = raw.manifest
    scan = EventsScan()

    history_start = np.datetime64(manifest.history_start, "us")
    extract_time = np.datetime64(manifest.extract_time, "us")

    for _, chunk in raw.iter_row_groups("events"):

        scan.row_groups += 1
        scan.rows += chunk.num_rows

        if chunk.num_rows == 0:
            continue

        # Пустое обязательное поле конверта: считается по всем
        # строкам, до любого разбора payload.
        for name in REQUIRED_ENVELOPE_FIELDS:
            nulls = chunk.column(name).null_count
            if nulls:
                scan.required_nulls[name] += nulls

        scan.by_source.update(_counter(chunk.column("source")))
        scan.by_type.update(_counter(chunk.column("event_type")))

        event_time = chunk.column("event_time")

        low, high = _min_max(event_time)
        scan.event_time_min = _merge_min(scan.event_time_min, low)
        scan.event_time_max = _merge_max(scan.event_time_max, high)

        event_np = event_time.to_numpy(zero_copy_only=False).astype("datetime64[us]")

        sources = np.asarray(chunk.column("source").to_pylist(), dtype=object)

        # --- повторы идентификаторов ---
        #
        # Запись приходит в выгрузку ровно один раз. Повтор
        # event_id — поломка, а не дефект доставки, и считается
        # по идентификатору, а не по паре с версией.

        scan.pair_hashes.append(_hash_ids(chunk.column("event_id")))

        # --- клиенты ---

        scan.clients.update(pc.unique(chunk.column("client_id")).to_pylist())

        # Строки клиента обязаны лежать подряд: и паспорт, и
        # canonical, и история читают ленту блоками целых
        # клиентов. Клиент, появившийся второй раз после другого,
        # потерял бы часть истории молча.
        for client in _run_starts(np.asarray(chunk.column("client_id").to_pylist(), dtype=object)):

            if client == scan.current_client:
                continue

            if scan.current_client is not None:
                scan.finished_clients.add(scan.current_client)

            if client in scan.finished_clients:
                scan.split_clients.add(client)

            scan.current_client = client

        # --- месяцы ---

        months = pc.strftime(event_time, "%Y-%m")
        keyed = pc.binary_join_element_wise(chunk.column("source"), months, "|")
        scan.month_counts.update(_counter(keyed))

        before = event_np < history_start
        if before.any():
            scan.before_start_by_source.update(Counter(sources[before].tolist()))
            types = np.asarray(chunk.column("event_type").to_pylist(), dtype=object)
            scan.before_start_by_type.update(Counter(types[before].tolist()))

        # Событие на границе выгрузки или позже: в честной выгрузке
        # таких строк быть не должно, и их число печатается.
        after = event_np >= extract_time
        if after.any():
            scan.after_extract_by_source.update(Counter(sources[after].tolist()))

        # --- payload по каталогу ---

        for event_type, rows in iter_event_types(chunk):

            info = manifest.catalogue.get(event_type)

            if info is None:
                scan.unknown_types[event_type] += rows.num_rows
                continue

            parsed = parse_payloads(info, rows.column("payload"))

            scan.payload.setdefault(event_type, PayloadStats()).merge(parsed)

    return scan


# ============================================================
# ПРОФИЛЬ, ПОКРЫТИЕ, КАТАЛОГИ
# ============================================================


def _quantiles(values: np.ndarray, points=(0.5, 0.9, 0.99, 1.0)) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {f"p{int(point * 100)}": float(np.quantile(values, point)) for point in points}


def scan_profile(raw: RawDataset) -> dict[str, Any]:

    rows = 0
    per_client: Counter = Counter()
    null_counts: Counter = Counter()
    columns: list[str] = []

    for _, chunk in raw.iter_row_groups("profile"):

        rows += chunk.num_rows
        columns = chunk.column_names

        if chunk.num_rows == 0:
            continue

        per_client.update(_counter(chunk.column("client_id")))

        for name in columns:
            null_counts[name] += chunk.column(name).null_count

    repeated = sorted(name for name, count in per_client.items() if count > 1)

    return {
        "rows": rows,
        "clients": len(per_client),
        "clients_with_several_rows": repeated[:10],
        "rule": "профиль это одна итоговая строка на клиента; вторая строка это поломка контракта",
        "null_share": {name: round(null_counts[name] / rows, 6) for name in columns} if rows else {},
    }


def scan_coverage(raw: RawDataset) -> dict[str, Any]:

    table = raw.read("source_coverage")

    by_source: dict[str, dict] = {}

    for source in sorted(pc.unique(table.column("source")).to_pylist()):

        rows = table.filter(pc.equal(table.column("source"), source))

        by_source[source] = {
            "rows": rows.num_rows,
            "first_available_at": sorted(
                {value.isoformat() for value in rows.column("first_available_at").to_pylist() if value is not None}
            ),
            "first_seen_null": rows.column("first_seen").null_count,
            "last_available_at_set": rows.num_rows - rows.column("last_available_at").null_count,
            "opening_state_set": rows.num_rows - rows.column("opening_state").null_count,
            "status": dict(sorted(_counter(rows.column("coverage_status")).items())),
            "reason": dict(sorted(_counter(rows.column("coverage_reason")).items())),
        }

    with_opening = table.filter(pc.is_valid(table.column("opening_state")))

    return {
        "rows": table.num_rows,
        "clients": len(pc.unique(table.column("client_id"))),
        "status": dict(sorted(_counter(table.column("coverage_status")).items())),
        "reason": dict(sorted(_counter(table.column("coverage_reason")).items())),
        "clients_with_opening_state": len(pc.unique(with_opening.column("client_id"))) if with_opening.num_rows else 0,
        "by_source": by_source,
    }


def scan_catalogs(raw: RawDataset) -> dict[str, Any]:

    out: dict[str, Any] = {
        "note": "справочники расшифровки: наличие продукта в каталоге не делает его известным клиенту в прошлом",
    }

    if raw.exists("products"):
        products = raw.read("products")
        out["products"] = {
            "rows": products.num_rows,
            "product_ids": len(pc.unique(products.column("product_id"))),
            "product_codes": len(pc.unique(products.column("product_code"))),
            "max_product_version": pc.max(products.column("product_version")).as_py(),
            "valid_from": dict(zip(("min", "max"), _min_max(products.column("valid_from")))),
            "status": dict(sorted(_counter(products.column("status")).items())),
            "synthetic_rows": int(pc.sum(products.column("is_synthetic")).as_py() or 0),
        }

    if raw.exists("merchants"):
        merchants = raw.read("merchants")
        out["merchants"] = {
            "rows": merchants.num_rows,
            "merchant_ids": len(pc.unique(merchants.column("merchant_id"))),
            "outlet_ids": len(pc.unique(merchants.column("outlet_id"))),
            "brands": len(pc.unique(merchants.column("brand"))),
            "mcc": len(pc.unique(merchants.column("mcc"))),
            "settlements": len(pc.unique(merchants.column("settlement"))),
            "online_share": round(
                int(pc.sum(merchants.column("is_online")).as_py() or 0) / max(1, merchants.num_rows), 6
            ),
        }

    if raw.exists("geography"):
        geography = raw.read("geography")
        out["geography"] = {
            "rows": geography.num_rows,
            "regions": len(pc.unique(geography.column("region"))),
        }

    return out


# ============================================================
# МЕСЯЦЫ, ГОРИЗОНТ, ИСТОЧНИКИ
# ============================================================


def month_range(start: datetime, end: datetime) -> list[str]:
    """
    Месяцы YYYY-MM от месяца start до месяца, содержащего end − 1 мкс
    (end исключительно).
    """

    last = end - timedelta(microseconds=1)

    months: list[str] = []

    year, month = start.year, start.month

    while (year, month) <= (last.year, last.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1

    return months


def month_table(scan: EventsScan, raw: RawDataset, config: PreprocessingConfig) -> dict[str, Any]:
    """
    Источник × месяц. Три разных клетки: месяц вне периода
    генерации, источник ещё не доступен, число событий (в том
    числе ноль).
    """

    manifest = raw.manifest

    start = min(config.required_history_start, manifest.history_start)
    months = month_range(start, manifest.history_end)

    generated_from = manifest.history_start.strftime("%Y-%m")

    table: dict[str, dict[str, Any]] = {}
    empty_generated: dict[str, list[str]] = {}

    for source in sorted(manifest.sources):

        available_from = manifest.sources[source].available_from.strftime("%Y-%m")

        row: dict[str, Any] = {}
        empties: list[str] = []

        for month in months:
            if month < generated_from:
                row[month] = MONTH_NOT_GENERATED
            elif month < available_from:
                row[month] = MONTH_SOURCE_UNAVAILABLE
            else:
                count = scan.month_counts.get(f"{source}|{month}", 0)
                row[month] = count
                if count == 0:
                    empties.append(month)

        table[source] = row
        if empties:
            empty_generated[source] = empties

    return {
        "months": months,
        "legend": {
            MONTH_NOT_GENERATED: "месяц вне периода генерации RAW",
            MONTH_SOURCE_UNAVAILABLE: "источник ещё не подключён в банке",
            "0": "источник доступен, событий нет: пустой месяц сам по себе не ошибка",
        },
        "table": table,
        "empty_months_of_available_sources": empty_generated,
        "before_history_start": {
            "rows": sum(scan.before_start_by_source.values()),
            "by_source": dict(sorted(scan.before_start_by_source.items())),
            "by_type": dict(sorted(scan.before_start_by_type.items())),
        },
        "event_time_at_or_after_extract": {
            "rows": sum(scan.after_extract_by_source.values()),
            "by_source": dict(sorted(scan.after_extract_by_source.items())),
        },
    }


def horizon_check(raw: RawDataset, config: PreprocessingConfig, groups: tuple[str, ...]) -> dict[str, Any]:

    manifest = raw.manifest

    reasons: list[str] = []

    start_ok = manifest.history_start <= config.required_history_start

    if not start_ok:
        reasons.append(
            f"согласованный горизонт с {config.required_history_start.date()} не обеспечен: "
            f"history_start = {manifest.history_start.date()}"
        )

    per_group: dict[str, dict] = {}

    for group in groups:

        window = config.windows[group]

        end_ok = manifest.history_end >= window.final_cutoff
        extract_ok = manifest.extract_time >= window.final_cutoff

        per_group[group] = {
            "final_cutoff": window.final_cutoff.isoformat(),
            "history_end_ok": end_ok,
            "extract_time_ok": extract_ok,
        }

        if not end_ok:
            reasons.append(
                f"{group}: history_end {manifest.history_end.date()} раньше конечного cutoff {window.final_cutoff.date()}"
            )
        if not extract_ok:
            reasons.append(
                f"{group}: extract_time {manifest.extract_time.date()} раньше конечного cutoff {window.final_cutoff.date()}"
            )

    return {
        "required_history_start": config.required_history_start.isoformat(),
        "history_start": manifest.history_start.isoformat(),
        "history_start_ok": start_ok,
        "history_end": manifest.history_end.isoformat(),
        "extract_time": manifest.extract_time.isoformat(),
        "groups": per_group,
        "ok": not reasons,
        "reasons": reasons,
    }


def sources_check(raw: RawDataset, config: PreprocessingConfig) -> dict[str, Any]:

    manifest = raw.manifest

    base: dict[str, dict] = {}
    late_base: list[str] = []
    missing_base: list[str] = []

    for source in config.base_sources:

        info = manifest.sources.get(source)

        if info is None:
            missing_base.append(source)
            continue

        ok = info.available_from <= config.required_history_start

        base[source] = {"available_from": info.available_from.isoformat(), "covers_required_start": ok}

        if not ok:
            late_base.append(source)

    others = {
        name: {"available_from": info.available_from.isoformat()}
        for name, info in sorted(manifest.sources.items())
        if name not in config.base_sources
    }

    return {
        "base": base,
        "base_missing_in_manifest": missing_base,
        "base_available_later_than_required": late_base,
        "late_connected_sources": others,
        "schema_changes": list(manifest.schema_changes),
    }


# ============================================================
# ПАСПОРТ
# ============================================================


def _events_summary(scan: EventsScan, config: PreprocessingConfig) -> dict[str, Any]:

    ids = np.concatenate(scan.pair_hashes) if scan.pair_hashes else np.zeros(0, dtype=np.uint64)
    _, counts = np.unique(ids, return_counts=True)
    repeated_ids = int((counts > 1).sum())
    repeated_rows = int((counts[counts > 1] - 1).sum())

    return {
        "rows": scan.rows,
        "row_groups": scan.row_groups,
        "clients": len(scan.clients),
        "event_time": {"min": scan.event_time_min, "max": scan.event_time_max},
        "by_source": dict(sorted(scan.by_source.items())),
        "by_event_type": dict(sorted(scan.by_type.items())),
        # Запись приходит в выгрузку ровно один раз. Повтор
        # event_id — поломка контракта, а не дефект доставки.
        "repeated_event_ids": {"ids": repeated_ids, "extra_rows": repeated_rows},
        "reversal_like": {
            name: scan.by_type.get(name, 0) for name in ("refund", "reversal", "chargeback")
        },
        "required_nulls": dict(sorted(scan.required_nulls.items())),
    }


def _payload_summary(scan: EventsScan, raw: RawDataset) -> dict[str, Any]:

    per_type: dict[str, dict] = {}
    violations_total: Counter = Counter()

    for event_type, stats in sorted(scan.payload.items()):

        info = raw.manifest.catalogue[event_type]

        per_type[event_type] = {
            "rows_checked": stats.rows,
            "null_share": {
                item.name: round(stats.nulls.get(item.name, 0) / stats.rows, 6) if stats.rows else 0.0
                for item in info.fields
            },
            "violations": dict(sorted(stats.violations.items())),
            "violations_by_field": dict(sorted(stats.by_field.items())),
            "samples": stats.samples,
        }

        violations_total.update(stats.violations)

    return {
        "rule": (
            "ключи проверены потоково по всем строкам; отсутствующий необязательный ключ — пропуск, "
            "отсутствующий обязательный, лишний ключ и значение не того типа — нарушения"
        ),
        "rows_checked": sum(stats.rows for stats in scan.payload.values()),
        "violations_total": dict(sorted(violations_total.items())),
        "unknown_event_types": dict(sorted(scan.unknown_types.items())),
        "by_event_type": per_type,
    }


def _file_inventory(raw: RawDataset) -> list[dict]:

    inventory: list[dict] = []

    for name in raw.listed_files():

        if name.startswith("truth/"):
            inventory.append({"file": name, "role": "truth: присутствует, не читается"})
            continue

        path = raw.raw_dir / name

        try:
            metadata = pq.read_metadata(path)
            schema = pq.read_schema(path).remove_metadata()
        except Exception as error:  # noqa: BLE001 — повреждённый файл это результат, а не сбой
            inventory.append({"file": name, "role": f"не читается: {type(error).__name__}"})
            continue

        inventory.append(
            {
                "file": name,
                "rows": int(metadata.num_rows),
                "row_groups": int(metadata.num_row_groups),
                "columns": [f"{item.name}: {item.type}" for item in schema],
            }
        )

    return inventory


def build_passport(
    raw_dir,
    config: PreprocessingConfig,
    group: str | None = None,
) -> dict[str, Any]:
    """
    Паспорт одной RAW-группы. Никогда не бросает исключение на
    плохих данных: поломка становится записью в errors, а статус —
    blocked.
    """

    raw_dir = Path(raw_dir)

    errors: list[str] = []
    contract_violations: list[str] = []
    limitations: list[str] = []

    groups: tuple[str, ...] = (group,) if group is not None else GROUPS

    try:
        raw = RawDataset(raw_dir)
    except RawContractError as error:
        return {
            "stage": STAGE,
            "stage_version": STAGE_VERSION,
            "group": group,
            "status": STATUS_BLOCKED,
            "usable": False,
            "errors": [str(error)],
            "contract_violations": [],
            "limitations": [],
            "config": config.section(STAGE),
        }

    report: dict[str, Any] = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "group": group,
        "raw": raw.manifest.echo(),
        "config": config.section(STAGE),
    }

    # --- структура ---

    missing = raw.missing_required_files()
    for name in missing:
        errors.append(f"нет обязательного файла {name}")

    checks: dict[str, Any] = {
        "missing_required_files": missing,
        "files": raw.verify_files(),
        "rows": raw.verify_rows() if not missing else {},
        "schemas": raw.verify_schemas() if not missing else {},
    }

    for name, item in checks["files"].items():
        if item["status"] == "mismatch":
            errors.append(f"sha256 файла {name} не совпадает с манифестом")
        elif item["status"] == "missing" and not name.startswith("truth/"):
            errors.append(f"файл {name} назван манифестом, но отсутствует")
        elif item["status"] == "unlisted":
            limitations.append(f"файл {name} не назван в манифесте")

    for table, item in checks["rows"].items():
        if item["status"] == "mismatch":
            errors.append(f"число строк {table}: манифест {item['expected']}, файл {item['actual']}")
        elif item["status"] == "unreadable":
            errors.append(f"файл таблицы {table} не читается как parquet")

    for table, item in checks["schemas"].items():
        if item["status"] == "mismatch":
            errors.append(f"схема {table}: " + "; ".join(item["differences"]))

    if not errors:
        checks["content"] = raw.verify_content()
        for table, item in checks["content"].items():
            if item["status"] == "mismatch":
                errors.append(f"content_sha256 таблицы {table} не совпадает с манифестом")
    else:
        checks["content"] = {}

    if raw.manifest.world_seed is None:
        limitations.append("в манифесте нет world_seed: общий мир групп манифестом не подтверждается")

    report["checks"] = checks
    report["files"] = _file_inventory(raw)

    if errors:
        report.update(
            {
                "status": STATUS_BLOCKED,
                "usable": False,
                "errors": errors,
                "contract_violations": contract_violations,
                "limitations": limitations,
            }
        )
        return report

    # --- содержимое ---

    scan = scan_events(raw, config)

    report["events"] = _events_summary(scan, config)
    report["payload"] = _payload_summary(scan, raw)
    report["profile"] = scan_profile(raw)
    report["coverage"] = scan_coverage(raw)
    report["catalogs"] = scan_catalogs(raw)
    report["months"] = month_table(scan, raw, config)
    report["horizon"] = horizon_check(raw, config, groups)
    report["sources"] = sources_check(raw, config)

    # --- пустые обязательные поля конверта ---
    #
    # Это структурная поломка, а не расхождение с каталогом
    # ключей: по такой строке нельзя ни установить порядок, ни
    # выбрать версию, ни построить связь. Группа блокируется, и
    # режим диагностики этого не снимает.

    for name, count in sorted(scan.required_nulls.items()):
        errors.append(f"поле {name}: null в {count} строках (обязательное поле конверта)")

    # --- разорванный блок клиента ---

    if scan.split_clients:
        listed = ", ".join(sorted(scan.split_clients)[:5])
        errors.append(
            f"строки клиента лежат не подряд ({len(scan.split_clients)} клиентов: {listed}): "
            "пачки целых клиентов собрать нельзя"
        )

    # --- нарушения входного контракта ---
    #
    # Данные расходятся с каталогом ключей, который объявляет сам
    # манифест. Это дефект выгрузки, а не ограничение наблюдения:
    # набор с такими строками не станет готовым и после того, как
    # горизонт станет полным. Чинится в генераторе.

    for event_type, rows in sorted(scan.unknown_types.items()):
        contract_violations.append(
            f"тип события {event_type} ({rows} строк) отсутствует в каталоге ключей манифеста"
        )

    for event_type, item in sorted(report["payload"]["by_event_type"].items()):
        for key, count in sorted(item["violations_by_field"].items()):
            kind, _, field_name = key.partition(":")
            if kind not in CONTRACT_VIOLATION_KINDS:
                continue
            place = f"{event_type}.{field_name}" if field_name not in ("", "None") else event_type
            contract_violations.append(f"{place}: {kind} в {count} строках")

    repeated = report["events"]["repeated_event_ids"]
    if repeated["ids"]:
        errors.append(
            f"повторяющийся event_id: {repeated['ids']} идентификаторов, "
            f"{repeated['extra_rows']} лишних строк; запись обязана приходить один раз"
        )

    before = report["months"]["before_history_start"]["rows"]
    if before:
        limitations.append(f"записей до history_start: {before} (реестр договоров, контекст, не история)")

    after = report["months"]["event_time_at_or_after_extract"]["rows"]
    if after:
        limitations.append(f"событий на границе extract_time или позже: {after}")

    for source in report["sources"]["base_available_later_than_required"]:
        limitations.append(
            f"базовый источник {source} доступен с {raw.manifest.sources[source].available_from.date()}, "
            f"позже согласованного начала"
        )

    for source in report["sources"]["base_missing_in_manifest"]:
        errors.append(f"базовый источник {source} отсутствует в manifest.sources")

    limitations.extend(report["horizon"]["reasons"])

    # Самый тяжёлый из сработавших вердиктов. Причины каждого
    # лежат в своём списке и не смешиваются между собой.

    verdicts = [STATUS_READY]

    if not report["horizon"]["ok"]:
        verdicts.append(STATUS_HORIZON_SHORT)

    if contract_violations:
        verdicts.append(STATUS_CONTRACT_MISMATCH)

    if errors:
        verdicts.append(STATUS_BLOCKED)

    status = max(verdicts, key=STATUS_ORDER.index)

    report.update(
        {
            "status": status,
            "usable": status not in NOT_USABLE,
            "errors": errors,
            "contract_violations": contract_violations,
            "limitations": limitations,
        }
    )

    return report


# ============================================================
# MARKDOWN
# ============================================================


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_passport_md(report: dict) -> str:

    out: list[str] = []

    group = report.get("group") or "—"
    out.append(f"# Паспорт RAW: группа {group}\n")
    out.append(f"Статус: **{report['status']}**.\n")

    if report.get("errors"):
        out.append("## Ошибки структуры (блокируют)\n")
        out.extend(f"- {item}" for item in report["errors"])
        out.append("")

    if report.get("contract_violations"):
        out.append("## Нарушения входного контракта (блокируют)\n")
        out.append(
            "Данные расходятся с каталогом ключей собственного манифеста. "
            "Набор с такими строками не может быть объявлен готовым; "
            "исправление принадлежит генератору.\n"
        )
        out.extend(f"- {item}" for item in report["contract_violations"])
        out.append("")

    if report.get("limitations"):
        out.append("## Известные ограничения\n")
        out.extend(f"- {item}" for item in report["limitations"])
        out.append("")

    raw = report.get("raw")
    if raw:
        out.append("## Манифест\n")
        out.append(
            _md_table(
                [
                    ["schema_version", raw["schema_version"]],
                    ["generator_version", raw["generator_version"]],
                    ["seed / world_seed", f"{raw['seed']} / {raw['world_seed']}"],
                    ["total_clients", raw["total_clients"]],
                    ["history_start", raw["history_start"]],
                    ["history_end", raw["history_end"]],
                    ["extract_time", raw["extract_time"]],
                    ["registry_start", raw["registry_start"]],
                ],
                ["ключ", "значение"],
            )
        )

    checks = report.get("checks")
    if checks:
        out.append("\n## Сверка с манифестом\n")
        rows = []
        for name, item in sorted(checks.get("files", {}).items()):
            rows.append(["файл", name, item["status"]])
        for name, item in sorted(checks.get("rows", {}).items()):
            rows.append(["строки", name, f"{item['status']} ({item['expected']} / {item['actual']})"])
        for name, item in sorted(checks.get("schemas", {}).items()):
            rows.append(["схема", name, item["status"]])
        for name, item in sorted(checks.get("content", {}).items()):
            rows.append(["содержимое", name, item["status"]])
        out.append(_md_table(rows, ["проверка", "объект", "результат"]))

    if report["status"] == STATUS_BLOCKED:
        return "\n".join(out) + "\n"

    events = report["events"]
    out.append("\n## Лента событий\n")
    out.append(
        _md_table(
            [
                ["строк", events["rows"]],
                ["клиентов", events["clients"]],
                ["event_time", f"{_fmt(events['event_time']['min'])} … {_fmt(events['event_time']['max'])}"],
                [
                    "повторов event_id",
                    f"{events['repeated_event_ids']['ids']} идентификаторов / "
                    f"{events['repeated_event_ids']['extra_rows']} лишних строк",
                ],
                ["refund / reversal / chargeback", ", ".join(f"{k}={v}" for k, v in events["reversal_like"].items())],
            ],
            ["показатель", "значение"],
        )
    )

    out.append("\n### По источникам\n")
    out.append(
        _md_table(
            [
                [source, count]
                for source, count in sorted(events["by_source"].items())
            ],
            ["источник", "строк"],
        )
    )

    out.append("\n### По типам событий\n")
    out.append(
        _md_table(
            [[name, count] for name, count in sorted(events["by_event_type"].items(), key=lambda item: -item[1])],
            ["тип", "строк"],
        )
    )


    payload = report["payload"]
    out.append("\n## Payload по каталогу ключей\n")
    out.append(f"{payload['rule']}. Проверено строк: {payload['rows_checked']}.\n")
    if payload["violations_total"]:
        out.append(
            "Нарушения контракта: "
            + ", ".join(f"{k}={v}" for k, v in sorted(payload["violations_total"].items()))
            + "\n"
        )
    else:
        out.append("Нарушений контракта нет.\n")
    if payload["unknown_event_types"]:
        out.append("Типы вне каталога: " + ", ".join(f"{k}={v}" for k, v in payload["unknown_event_types"].items()) + "\n")

    rows = []
    for event_type, item in sorted(payload["by_event_type"].items()):
        high = sorted(item["null_share"].items(), key=lambda pair: -pair[1])[:4]
        rows.append(
            [
                event_type,
                item["rows_checked"],
                ", ".join(f"{k}={v}" for k, v in sorted(item["violations"].items())) or "—",
                ", ".join(f"{name}={share:.2f}" for name, share in high if share > 0) or "—",
            ]
        )
    out.append(_md_table(rows, ["тип", "строк", "нарушения", "самые частые пропуски"]))

    profile = report["profile"]
    out.append("\n## Профиль\n")
    out.append(
        _md_table(
            [
                ["строк", profile["rows"]],
                ["клиентов", profile["clients"]],
                ["клиентов с лишними строками", len(profile["clients_with_several_rows"])],
            ],
            ["показатель", "значение"],
        )
    )

    coverage = report["coverage"]
    out.append("\n## Покрытие источников\n")
    out.append(
        _md_table(
            [
                [
                    source,
                    item["rows"],
                    ", ".join(item["first_available_at"]),
                    item["first_seen_null"],
                    item["last_available_at_set"],
                    item["opening_state_set"],
                    ", ".join(f"{k}={v}" for k, v in item["status"].items()),
                    ", ".join(f"{k}={v}" for k, v in item["reason"].items()),
                ]
                for source, item in sorted(coverage["by_source"].items())
            ],
            ["источник", "строк", "first_available_at", "first_seen null", "last_available_at задан", "opening_state", "статусы", "причины"],
        )
    )
    out.append(f"\nКлиентов с opening_state: {coverage['clients_with_opening_state']}.\n")

    months = report["months"]
    out.append("\n## Источник × месяц\n")
    out.append(
        "Обозначения: "
        + "; ".join(f"`{key}` — {value}" for key, value in months["legend"].items())
        + ".\n"
    )
    header = ["источник"] + months["months"]
    rows = [[source] + [months["table"][source][month] for month in months["months"]] for source in sorted(months["table"])]
    out.append(_md_table(rows, header))
    out.append(
        f"\nЗаписей до history_start: {months['before_history_start']['rows']} "
        f"({_fmt(months['before_history_start']['by_type']) if months['before_history_start']['rows'] else '—'}). "
        f"Событий на границе extract_time или позже: {months['event_time_at_or_after_extract']['rows']}.\n"
    )

    horizon = report["horizon"]
    out.append("\n## Горизонт\n")
    out.append(
        _md_table(
            [
                ["согласованное начало", horizon["required_history_start"]],
                ["history_start", f"{horizon['history_start']} ({'ok' if horizon['history_start_ok'] else 'короче'})"],
                ["history_end", horizon["history_end"]],
                ["extract_time", horizon["extract_time"]],
            ]
            + [
                [
                    f"группа {name}",
                    f"конечный cutoff {item['final_cutoff']}: history_end {'ok' if item['history_end_ok'] else 'мало'}, "
                    f"extract_time {'ok' if item['extract_time_ok'] else 'мало'}",
                ]
                for name, item in sorted(horizon["groups"].items())
            ],
            ["показатель", "значение"],
        )
    )

    sources = report["sources"]
    out.append("\n## Источники\n")
    out.append(
        _md_table(
            [
                [source, item["available_from"], "да" if item["covers_required_start"] else "нет"]
                for source, item in sorted(sources["base"].items())
            ],
            ["базовый источник", "available_from", "покрывает согласованное начало"],
        )
    )
    out.append("\nПодключены позже:\n")
    out.append(
        _md_table(
            [[source, item["available_from"]] for source, item in sources["late_connected_sources"].items()],
            ["источник", "available_from"],
        )
    )
    if sources["schema_changes"]:
        out.append("\nДатированные изменения схемы:\n")
        out.append(
            _md_table(
                [[item.get("source"), item.get("field"), item.get("from"), item.get("reason")] for item in sources["schema_changes"]],
                ["источник", "поле", "с даты", "причина"],
            )
        )

    catalogs = report["catalogs"]
    out.append("\n## Справочники\n")
    out.append(f"{catalogs['note']}.\n")
    rows = []
    for name in ("products", "merchants", "geography"):
        if name in catalogs:
            rows.append([name, ", ".join(f"{k}={_fmt(v)}" for k, v in catalogs[name].items())])
    out.append(_md_table(rows, ["таблица", "сводка"]))

    out.append("\n## Файлы\n")
    out.append(
        _md_table(
            [
                [item["file"], item.get("rows", "—"), item.get("row_groups", "—"), item.get("role", f"{len(item.get('columns', []))} колонок")]
                for item in report["files"]
            ],
            ["файл", "строк", "row groups", "заметка"],
        )
    )

    return "\n".join(out) + "\n"


__all__ = [
    "CONTRACT_VIOLATION_KINDS",
    "NOT_USABLE",
    "STAGE",
    "STAGE_VERSION",
    "STATUS_BLOCKED",
    "STATUS_CONTRACT_MISMATCH",
    "STATUS_HORIZON_SHORT",
    "STATUS_ORDER",
    "STATUS_READY",
    "build_passport",
    "month_range",
    "render_passport_md",
    "scan_events",
]
