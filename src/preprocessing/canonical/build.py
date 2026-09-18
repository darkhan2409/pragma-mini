from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..artifacts import TableWriter, _md_table, write_json, write_table, write_text
from ..rawdata import RawDataset
from ..settings import PreprocessingConfig
from .entities import extract_mentions, extract_transfers, order_transfers
from .events import (
    TEXT_NORMALIZATION,
    build_batch,
    canonical_schema,
    iter_client_batches,
)
from .links import build_link_report, scan_entities
from .registry import build_registry, registry_as_dict, registry_digest
from .schema import (
    CLIENT_INDEX_SCHEMA,
    DEDUPE_LOG_SCHEMA,
    DERIVED_COLUMNS,
    MENTIONS_SCHEMA,
    REJECTS_SCHEMA,
    SCHEMA_VERSION,
    TRANSFERS_SCHEMA,
    VERSION_ROLE_CONFLICT,
    VERSION_ROLE_CORRECTION,
    VERSION_ROLE_REDELIVERY,
    coverage_schema,
    payload_columns,
)
from .sidecars import build_coverage, build_profile


# ============================================================
# ИДЕЯ
# ============================================================
#
# Оркестрация этапа 2: RAW одной группы -> canonical.
#
# Лента читается пачками целых клиентов и пишется по одному row
# group на пачку, поэтому память ограничена пачкой, а не файлом.
# Профиль, покрытие, сущности, переводы, журнал дублей и отчёт о
# связях собираются по дороге.
#
# Число строк обязано сойтись: в canonical ровно столько строк,
# сколько их в RAW. Всё, что нельзя разобрать, остаётся строкой с
# причиной, а не исчезает.
# ============================================================


STAGE = "canonical"
STAGE_VERSION = "4.4.0"

STATUS_OK = "ok"
STATUS_ROW_COUNT_MISMATCH = "row_count_mismatch"

EVENTS_FILE = "events.parquet"
PROFILE_FILE = "profile.parquet"
COVERAGE_FILE = "coverage.parquet"
CLIENT_INDEX_FILE = "client_index.parquet"
MENTIONS_FILE = "entities/mentions.parquet"
TRANSFERS_FILE = "entities/transfers.parquet"
DEDUPE_FILE = "dedupe_log.parquet"
REJECTS_FILE = "rejects.parquet"
REGISTRY_FILE = "field_registry.json"
REPORT_JSON_FILE = "canonical_report.json"
REPORT_MD_FILE = "canonical_report.md"


@dataclass
class CanonicalResult:
    report: dict
    outputs: list[Path]


def build_client_index(raw: RawDataset) -> dict[str, int]:
    """
    Плотный внутренний индекс клиента: устойчивый номер в
    лексикографическом порядке client_id.

    Список собирается из покрытия, профиля и ленты: клиент без
    событий не исчезает, а клиент, которого нет в покрытии, не
    теряется.
    """

    ids: set[str] = set()

    for table in ("source_coverage", "profile"):
        if raw.exists(table):
            ids.update(raw.read(table, ["client_id"]).column("client_id").to_pylist())

    for _, chunk in raw.iter_row_groups("events", ["client_id"]):
        ids.update(pc.unique(chunk.column("client_id")).to_pylist())

    return {value: index for index, value in enumerate(sorted(ids))}


def locate_clients(events_path: Path, clients: list[dict]) -> None:
    """
    Проставляет адрес клиента по ФАКТИЧЕСКОЙ разбивке файла.

    Пачка ложится в один row group, но это свойство писателя, а не
    контракт. Границы читаются из метаданных, поэтому адрес верен
    и если писатель однажды порежет файл иначе.
    """

    metadata = pq.read_metadata(events_path)

    boundaries = [0]
    for index in range(metadata.num_row_groups):
        boundaries.append(boundaries[-1] + metadata.row_group(index).num_rows)

    edges = np.asarray(boundaries, dtype=np.int64)

    for item in clients:

        start = item["global_row_start"]

        if start is None or not item["row_count"]:
            continue

        group = int(np.searchsorted(edges, start, side="right") - 1)

        item["row_group"] = group
        item["row_offset"] = int(start - edges[group])
        item["spans_row_groups"] = bool(start + item["row_count"] > edges[group + 1])


def build_group(
    raw_dir: Path,
    out_dir: Path,
    config: PreprocessingConfig,
    group: str | None,
) -> CanonicalResult:

    raw = RawDataset(raw_dir)
    manifest = raw.manifest

    out_dir = Path(out_dir)

    schema = canonical_schema(manifest)
    payload_names = [name for name, _ in payload_columns(manifest)]

    client_index = build_client_index(raw)

    # --- лента ---

    events_writer = TableWriter(out_dir / EVENTS_FILE, schema)
    mentions_writer = TableWriter(out_dir / MENTIONS_FILE, MENTIONS_SCHEMA)

    # Отчёт о сущностях копится по пачкам: читать всю таблицу
    # упоминаний обратно значит держать в памяти целую группу.
    entity_scan: dict[tuple[str, str], dict] = {}

    dedupe: list[dict] = []
    rejects: list[dict] = []
    clients: list[dict] = []
    transfers: list[dict] = []

    counts: Counter = Counter()
    roles: Counter = Counter()
    flags: Counter = Counter()

    row_start = 0
    row_group = 0

    for batch in iter_client_batches(raw, config.batch_clients):

        result = build_batch(
            raw,
            config,
            batch,
            payload_names,
            schema,
            client_index,
            row_group,
            row_start,
        )

        events_writer.write(result.table)

        mentions = extract_mentions(result.table)
        if mentions.num_rows:
            mentions_writer.write(mentions)
            scan_entities(mentions, entity_scan)

        transfers.extend(extract_transfers(result.table))

        dedupe.extend(result.dedupe_log)
        rejects.extend(result.rejects)
        clients.extend(result.clients)

        for key, value in result.counts.items():
            counts[key] += value

        roles.update(result.table.column("version_role").to_pylist())

        for name in (
            "is_exact_duplicate",
            "before_window",
            "at_or_after_extract",
            "time_finer_than_precision",
            "ambiguous_local_time",
            "balance_chain_gap",
        ):
            flags[name] += int(pc.sum(result.table.column(name)).as_py() or 0)

        flags["payload_violations"] += int(
            pc.sum(pc.is_valid(result.table.column("payload_violations"))).as_py() or 0
        )
        flags["known_missing"] += int(
            pc.sum(pc.is_valid(result.table.column("known_missing"))).as_py() or 0
        )

        row_start += result.table.num_rows
        row_group += 1

    events_rows = events_writer.close()
    mentions_rows = mentions_writer.close()

    # --- спутники ---

    profile_table, profile_report = build_profile(raw, client_index)
    write_table(out_dir / PROFILE_FILE, profile_table)

    coverage_table, coverage_report = build_coverage(raw, client_index)
    write_table(out_dir / COVERAGE_FILE, coverage_table)

    transfers_table = order_transfers(transfers)
    write_table(out_dir / TRANSFERS_FILE, transfers_table, TRANSFERS_SCHEMA)

    write_table(
        out_dir / DEDUPE_FILE,
        pa.Table.from_pylist(dedupe, schema=DEDUPE_LOG_SCHEMA) if dedupe else DEDUPE_LOG_SCHEMA.empty_table(),
        DEDUPE_LOG_SCHEMA,
    )

    write_table(
        out_dir / REJECTS_FILE,
        pa.Table.from_pylist(rejects, schema=REJECTS_SCHEMA) if rejects else REJECTS_SCHEMA.empty_table(),
        REJECTS_SCHEMA,
    )

    # Клиенты без событий тоже в индексе: отсутствие событий это
    # наблюдение, а не повод исчезнуть.
    with_events = {item["client_id"] for item in clients}

    # Признак тестового счёта у такого клиента брать неоткуда,
    # кроме покрытия: в ленте у него строк нет. Проставленный
    # False означал бы «это настоящий клиент», и тестовые счета
    # без событий попадали бы в обучение.
    test_accounts = {
        row["client_id"]
        for row in coverage_table.select(["client_id", "coverage_reason"]).to_pylist()
        if row["coverage_reason"] == "test_account"
    }

    for client_id, index in sorted(client_index.items(), key=lambda item: item[1]):
        if client_id not in with_events:
            clients.append(
                {
                    "client_idx": index,
                    "client_id": client_id,
                    "row_group": None,
                    "row_offset": None,
                    "global_row_start": None,
                    "row_count": 0,
                    "spans_row_groups": False,
                    "event_time_min": None,
                    "event_time_max": None,
                    "is_test_account": client_id in test_accounts,
                }
            )

    clients.sort(key=lambda item: item["client_idx"])

    locate_clients(out_dir / EVENTS_FILE, clients)

    write_table(
        out_dir / CLIENT_INDEX_FILE,
        pa.Table.from_pylist(clients, schema=CLIENT_INDEX_SCHEMA),
        CLIENT_INDEX_SCHEMA,
    )

    # --- реестр полей ---

    extra = [("derived", name, dtype, description) for name, _, dtype, description in DERIVED_COLUMNS]
    entries = build_registry(manifest, extra)

    registry = registry_as_dict(entries, timezone=config.timezone)

    write_json(out_dir / REGISTRY_FILE, registry)

    # --- связи ---

    link_report = build_link_report(
        out_dir / EVENTS_FILE,
        entity_scan,
        transfers_table,
        coverage_table,
        manifest.history_start,
    )

    # --- отчёт ---

    raw_rows = manifest.rows.get("events", 0)

    status = STATUS_OK if events_rows == raw_rows else STATUS_ROW_COUNT_MISMATCH

    report: dict[str, Any] = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "group": group,
        "status": status,
        "raw": manifest.echo(),
        "config": config.section(STAGE),
        "rows": {
            "raw_events": raw_rows,
            "canonical_events": events_rows,
            "difference": events_rows - raw_rows,
            "rule": "одна строка RAW это одна строка canonical: дубли и конфликты остаются с пометкой",
            "profile": profile_table.num_rows,
            "coverage": coverage_table.num_rows,
            "mentions": mentions_rows,
            "transfer_sides": transfers_table.num_rows,
            "dedupe_log": len(dedupe),
            "rejects": len(rejects),
            "clients": len(clients),
            "clients_without_events": sum(1 for item in clients if not item["row_count"]),
        },
        "versions": {
            "roles": dict(sorted(roles.items())),
            "corrections": roles.get(VERSION_ROLE_CORRECTION, 0),
            "redeliveries": roles.get(VERSION_ROLE_REDELIVERY, 0),
            "conflicts": roles.get(VERSION_ROLE_CONFLICT, 0),
            "rule": (
                "сравнение идёт только внутри одного event_id; повторная доставка это та же версия с тем же "
                "содержимым, конфликт это та же версия с другим содержимым, разные события с похожими полями "
                "дублями не считаются"
            ),
        },
        "flags": dict(sorted(flags.items())),
        "payload": {
            "columns": len(payload_names),
            "physical_fields": sum(1 for item in entries if item.role == "payload"),
            "violations": {key: value for key, value in sorted(counts.items()) if key != "rows"},
            "rule": "нарушение контракта обнуляет ячейку и остаётся в payload_violations строки",
        },
        "text_normalization": TEXT_NORMALIZATION,
        "timezone": {
            "contract": config.timezone,
            "rule": "значения остаются наивными, как в выгрузке; пояс объявлен, а не применён",
            "ambiguous_intervals": [item.as_dict() for item in config.ambiguous_local_intervals],
            "ambiguous_rows": flags.get("ambiguous_local_time", 0),
        },
        "profile": profile_report,
        "coverage": coverage_report,
        "links": link_report,
        "registry": {
            "file": REGISTRY_FILE,
            "digest": registry_digest(entries),
            "counts": registry["counts"],
            "rules": registry["rules"],
        },
    }

    outputs = [
        out_dir / EVENTS_FILE,
        out_dir / PROFILE_FILE,
        out_dir / COVERAGE_FILE,
        out_dir / CLIENT_INDEX_FILE,
        out_dir / MENTIONS_FILE,
        out_dir / TRANSFERS_FILE,
        out_dir / DEDUPE_FILE,
        out_dir / REJECTS_FILE,
        out_dir / REGISTRY_FILE,
    ]

    # Отчёт пишет сам этап: каталог canonical должен отвечать на
    # вопрос о себе без помощи оркестратора, иначе читатель слоя
    # не найдёт ни границ выгрузки, ни правил сборки.
    write_json(out_dir / REPORT_JSON_FILE, report)
    write_text(out_dir / REPORT_MD_FILE, render_canonical_md(report))

    outputs += [out_dir / REPORT_JSON_FILE, out_dir / REPORT_MD_FILE]

    return CanonicalResult(report, outputs)


# ============================================================
# MARKDOWN
# ============================================================


def render_canonical_md(report: dict) -> str:

    out: list[str] = []

    group = report.get("group") or "—"

    out.append(f"# Canonical: группа {group}\n")
    out.append(f"Статус: **{report['status']}**.\n")

    rows = report["rows"]

    out.append("## Строки\n")
    out.append(
        _md_table(
            [
                ["строк RAW", rows["raw_events"]],
                ["строк canonical", rows["canonical_events"]],
                ["разница", rows["difference"]],
                ["версии профиля", rows["profile"]],
                ["строки покрытия", rows["coverage"]],
                ["упоминания сущностей", rows["mentions"]],
                ["стороны переводов", rows["transfer_sides"]],
                ["журнал дублей и конфликтов", rows["dedupe_log"]],
                ["неразобранные строки", rows["rejects"]],
                ["клиентов", rows["clients"]],
                ["из них без событий", rows["clients_without_events"]],
            ],
            ["показатель", "значение"],
        )
    )
    out.append(f"\n{rows['rule']}.\n")

    versions = report["versions"]

    out.append("\n## Версии, дубли и конфликты\n")
    out.append(
        _md_table(
            [[name, count] for name, count in versions["roles"].items()],
            ["роль строки", "строк"],
        )
    )
    out.append(f"\n{versions['rule']}.\n")

    out.append("\n## Наблюдаемость\n")
    out.append(
        _md_table(
            [[name, count] for name, count in report["flags"].items()],
            ["признак", "строк"],
        )
    )

    payload = report["payload"]

    out.append("\n## Payload\n")
    out.append(
        _md_table(
            [
                ["физических полей", payload["physical_fields"]],
                ["колонок в таблице", payload["columns"]],
                ["нарушения контракта", payload["violations"] or "—"],
            ],
            ["показатель", "значение"],
        )
    )
    out.append(f"\n{payload['rule']}.\n")

    timezone = report["timezone"]

    out.append("\n## Время\n")
    out.append(
        _md_table(
            [
                ["часовой пояс контракта", timezone["contract"]],
                ["правило", timezone["rule"]],
                [
                    "неоднозначные интервалы",
                    ", ".join(f"{item['start']}…{item['end']} ({item['reason']})" for item in timezone["ambiguous_intervals"]) or "—",
                ],
                ["строк в них", timezone["ambiguous_rows"]],
            ],
            ["показатель", "значение"],
        )
    )

    links = report["links"]

    out.append("\n## Связи\n")

    causes = links["causes"]
    if causes.get("references"):
        out.append(
            _md_table(
                [
                    ["ссылок на причину", causes["references"]],
                    ["разрешено", causes["resolved"]],
                    ["не разрешено", causes["unresolved"] or "—"],
                    ["у другого клиента", causes["cross_client"]],
                    ["причина позже следствия", causes["cause_after_effect"]],
                    [
                        "из них у причины потеряно время",
                        causes["cause_after_effect_with_coarse_cause"],
                    ],
                ],
                ["показатель", "значение"],
            )
        )
        out.append(f"\n{causes['rule']}.\n")

    transfers = links["transfers"]
    if transfers.get("sides"):
        out.append("\n### Переводы\n")
        out.append(
            _md_table(
                [
                    ["сторон", transfers["sides"]],
                    ["переводов", transfers["transfers"]],
                    ["обе стороны есть в выгрузке", transfers["both_sides_in_full_extract"]],
                    ["одна сторона в выгрузке", transfers["one_side_in_full_extract"]],
                ],
                ["показатель", "значение"],
            )
        )
        out.append(f"\n{transfers['rule']}.\n")

    entities = links["entities"]
    if entities.get("entities"):
        out.append("\n### Сущности\n")
        out.append(
            _md_table(
                [
                    [
                        kind,
                        item["entities"],
                        "да" if item["opening_applicable"] else "нет",
                        item["with_opening"],
                        item["without_opening"],
                        ", ".join(f"{k}={v}" for k, v in item["reasons"].items()) or "—",
                    ]
                    for kind, item in entities["by_kind"].items()
                ],
                ["вид", "всего", "открытие определено", "с открытием", "без открытия", "причины"],
            )
        )
        out.append(f"\n{entities['rule']}.\n")

    chains = links["chains"]
    if chains.get("chains"):
        out.append("\n### Цепочки\n")
        out.append(
            _md_table(
                [
                    [
                        link_type,
                        item["chains"],
                        ", ".join(f"{size}:{count}" for size, count in item["sizes"].items()),
                        ", ".join(f"{k}={v}" for k, v in item["event_types"].items()),
                    ]
                    for link_type, item in chains["by_link_type"].items()
                ],
                ["связь", "цепочек", "размеры", "типы событий"],
            )
        )

    profile = report["profile"]
    coverage = report["coverage"]

    out.append("\n## Профиль и покрытие\n")
    out.append(
        _md_table(
            [
                ["версий профиля", profile["rows"]],
                ["клиентов в профиле", profile["clients"]],
                ["правило профиля", profile["rule"]],
                ["строк покрытия", coverage["rows"]],
                ["opening_state", ", ".join(f"{k}={v}" for k, v in coverage["opening_state"].items())],
                [
                    "ключи opening_state",
                    ", ".join(f"{k}={v}" for k, v in coverage["opening_state_keys"].items()) or "—",
                ],
                ["правило покрытия", coverage["rule"]],
            ],
            ["показатель", "значение"],
        )
    )

    registry = report["registry"]

    out.append("\n## Реестр физических полей\n")
    out.append(
        _md_table(
            [
                ["всего записей", registry["counts"]["total"]],
                ["полей payload", registry["counts"]["by_owner_kind"]["payload"]],
                ["из них разных имён", registry["counts"]["distinct_payload_names"]],
                ["конверт", registry["counts"]["by_owner_kind"]["envelope"]],
                ["профиль", registry["counts"]["by_owner_kind"]["profile"]],
                ["покрытие", registry["counts"]["by_owner_kind"]["coverage"]],
                ["производные", registry["counts"]["by_owner_kind"]["derived"]],
            ],
            ["показатель", "значение"],
        )
    )
    out.append(f"\n{registry['rules']['identity']}.\n")

    return "\n".join(out) + "\n"


__all__ = [
    "CLIENT_INDEX_FILE",
    "REGISTRY_FILE",
    "REPORT_JSON_FILE",
    "REPORT_MD_FILE",
    "locate_clients",
    "COVERAGE_FILE",
    "DEDUPE_FILE",
    "EVENTS_FILE",
    "MENTIONS_FILE",
    "PROFILE_FILE",
    "REJECTS_FILE",
    "STAGE",
    "STAGE_VERSION",
    "STATUS_OK",
    "STATUS_ROW_COUNT_MISMATCH",
    "TRANSFERS_FILE",
    "CanonicalResult",
    "build_client_index",
    "build_group",
    "render_canonical_md",
]
