from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq



# ============================================================
# ИДЕЯ
# ============================================================
#
# Отчёт о связях: что сошлось, что не сошлось и по какой
# причине. Ничего не достраивается по догадке.
#
# Отсутствующая ссылка это не всегда ошибка. Объект мог быть
# открыт до начала наблюдения, вторая сторона перевода может
# жить вне банка или вне этой выборки клиентов, а цель ссылки
# могла ещё не поступить. Отчёт называет доступные признаки и
# честно говорит, когда различить причины по данным нельзя.
# ============================================================


CAUSE_FIELD = "cause_event_id"

REASON_NOT_IN_DATASET = "target_not_in_dataset"
REASON_BEFORE_WINDOW = "first_mention_before_history_start"
REASON_PRE_WINDOW_CONTRACTS = "client_has_contracts_before_window"
REASON_UNKNOWN = "not_observed_in_dataset"

OPENING_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "account": ("opened",),
    "card": ("activated", "reissued"),
    "contract": ("opened",),
    "application": ("submitted",),
    "case": ("opened",),
}


def _quantiles(counter: Counter) -> dict:
    if not counter:
        return {}
    return {str(size): count for size, count in sorted(counter.items())}


def check_causes(events_path: Path) -> dict:
    """
    cause_event_id: существует ли цель, у того ли клиента и не
    позже ли она следствия.
    """

    parquet = pq.ParquetFile(events_path)

    referenced: set[str] = set()

    for index in range(parquet.num_row_groups):
        column = parquet.read_row_group(index, columns=[CAUSE_FIELD]).column(CAUSE_FIELD)
        referenced.update(value for value in column.to_pylist() if value)

    if not referenced:
        return {"references": 0, "rule": "ссылок на событие-причину в выгрузке нет"}

    known: dict[str, dict] = {}

    columns = ["event_id", "client_id", "event_time", "event_version", "time_precision"]

    if "timestamp_quality" in parquet.schema_arrow.names:
        columns.append("timestamp_quality")

    for index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(index, columns=columns)
        for row in chunk.to_pylist():
            if row["event_id"] in referenced:
                current = known.get(row["event_id"])
                if current is None or row["event_version"] < current["event_version"]:
                    known[row["event_id"]] = row

    total = 0
    resolved = 0
    cross_client = 0
    cause_after_effect = 0
    cause_after_effect_coarse = 0
    missing: Counter = Counter()

    for index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(
            index, columns=["client_id", "event_time", CAUSE_FIELD, "event_type"]
        )
        for row in chunk.to_pylist():
            cause = row[CAUSE_FIELD]
            if not cause:
                continue
            total += 1
            target = known.get(cause)
            if target is None:
                missing[REASON_NOT_IN_DATASET] += 1
                continue
            resolved += 1
            if target["client_id"] != row["client_id"]:
                cross_client += 1
            if target["event_time"] > row["event_time"]:
                cause_after_effect += 1
                # Витрина теряет время у части записей: у такой
                # причины день известен, а час нет, и порядок
                # внутри дня восстановить нельзя.
                if target.get("time_precision") != "second" or target.get("timestamp_quality") == "date_only":
                    cause_after_effect_coarse += 1

    return {
        "references": total,
        "resolved": resolved,
        "unresolved": dict(missing),
        "cross_client": cross_client,
        "cause_after_effect": cause_after_effect,
        "cause_after_effect_with_coarse_cause": cause_after_effect_coarse,
        "rule": (
            "цель ищется среди событий той же группы; причина, оказавшаяся позже следствия по времени "
            "бизнеса, почти всегда это запись с потерянным временем, и такие случаи считаются отдельной строкой"
        ),
    }


def check_transfers(transfers: pa.Table) -> dict:

    if transfers.num_rows == 0:
        return {"sides": 0, "rule": "переводов в выгрузке нет"}

    sides = Counter(transfers.column("side").to_pylist())

    by_id: dict[str, set[str]] = defaultdict(set)
    for transfer_id, client_id in zip(
        transfers.column("transfer_id").to_pylist(), transfers.column("client_id").to_pylist()
    ):
        by_id[transfer_id].add(client_id)

    return {
        "sides": transfers.num_rows,
        "transfers": len(by_id),
        "by_side": dict(sorted(sides.items())),
        "both_sides_in_full_extract": sum(1 for clients in by_id.values() if len(clients) > 1),
        "one_side_in_full_extract": sum(1 for clients in by_id.values() if len(clients) == 1),
        "rule": (
            "это статистика по ВСЕЙ выгрузке для отчёта качества, а не признак строки. "
            "Обе стороны, найденные в выгрузке, не значат, что на конкретном cutoff встречная "
            "сторона уже известна: парность на дату считает этап истории. Одна сторона не значит "
            "потерю: вторая либо вне банка, либо у клиента за пределами этой выборки"
        ),
    }


def scan_entities(mentions: pa.Table, seen: dict[tuple[str, str], dict]) -> None:
    """
    Складывает пачку упоминаний в накопитель по сущностям.

    Считается по пачкам, а не по всей таблице упоминаний сразу:
    отчёт о сущностях не должен требовать памяти в размер всей
    группы, когда сам слой собирается пачками клиентов.
    """

    if mentions.num_rows == 0:
        return

    for row in mentions.select(
        ["entity_kind", "entity_id", "client_id", "event_time", "is_transition", "transition"]
    ).to_pylist():

        key = (row["entity_kind"], row["entity_id"])

        item = seen.get(key)

        if item is None:
            item = {
                "client_id": row["client_id"],
                "first_event_time": row["event_time"],
                "opened": False,
                "mentions": 0,
            }
            seen[key] = item

        item["mentions"] += 1

        # Страховка: в норме сюда не доходит строка без времени —
        # паспорт блокирует такую выгрузку, а canonical падает
        # раньше с названием поля. Но сравнивать неизвестное
        # время нельзя ни при каких обстоятельствах.
        known = [value for value in (item["first_event_time"], row["event_time"]) if value is not None]
        item["first_event_time"] = min(known) if known else None

        if row["is_transition"] and row["transition"] in OPENING_TRANSITIONS.get(row["entity_kind"], ()):
            item["opened"] = True


def check_entities(seen: dict[tuple[str, str], dict], coverage: pa.Table, history_start: datetime) -> dict:
    """
    У каждой ли наблюдаемой сущности есть наблюдаемое открытие.

    На вход идёт накопитель, собранный по пачкам.
    """

    if not seen:
        return {"entities": 0}

    pre_window: dict[str, int] = {}

    if coverage.num_rows:
        for row in coverage.select(["client_id", "opening_state_values"]).to_pylist():
            for key, value in row["opening_state_values"] or []:
                if key == "contracts_before_window" and value:
                    pre_window[row["client_id"]] = max(pre_window.get(row["client_id"], 0), int(value))

    per_kind: dict[str, dict] = {}

    for (kind, _), item in seen.items():

        applicable = kind in OPENING_TRANSITIONS

        slot = per_kind.setdefault(
            kind,
            {
                "entities": 0,
                "opening_applicable": applicable,
                "with_opening": 0,
                "without_opening": 0,
                "reasons": Counter(),
            },
        )

        slot["entities"] += 1

        # У предложения нет собственного открытия: это ссылка на
        # кампанию, а не продукт клиента.
        if not applicable:
            continue

        if item["opened"]:
            slot["with_opening"] += 1
            continue

        slot["without_opening"] += 1

        if item["first_event_time"] is not None and item["first_event_time"] < history_start:
            slot["reasons"][REASON_BEFORE_WINDOW] += 1
        elif item["client_id"] in pre_window:
            slot["reasons"][REASON_PRE_WINDOW_CONTRACTS] += 1
        else:
            slot["reasons"][REASON_UNKNOWN] += 1

    for slot in per_kind.values():
        slot["reasons"] = dict(sorted(slot["reasons"].items()))

    return {
        "entities": len(seen),
        "by_kind": dict(sorted(per_kind.items())),
        "rule": (
            "открытие ищется среди наблюдаемых переходов и только там, где оно определено; "
            "отсутствие открытия объясняется первым упоминанием до начала окна или заявленным "
            "opening_state клиента, иначе остаётся ненаблюдаемым и не достраивается"
        ),
    }


def check_chains(events_path: Path) -> dict:
    """
    Цепочки по correlation_id: сколько, какого размера и из чего.
    """

    parquet = pq.ParquetFile(events_path)

    chains: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(index, columns=["correlation_id", "link_type", "event_type"])
        for correlation, link_type, event_type in zip(
            chunk.column("correlation_id").to_pylist(),
            chunk.column("link_type").to_pylist(),
            chunk.column("event_type").to_pylist(),
        ):
            if not correlation:
                continue
            chains[(link_type or "unset", correlation)][event_type] += 1

    by_link: dict[str, dict] = {}

    for (link_type, _), composition in chains.items():
        slot = by_link.setdefault(link_type, {"chains": 0, "sizes": Counter(), "event_types": Counter()})
        slot["chains"] += 1
        slot["sizes"][sum(composition.values())] += 1
        slot["event_types"].update(composition)

    for slot in by_link.values():
        slot["sizes"] = _quantiles(slot["sizes"])
        slot["event_types"] = dict(sorted(slot["event_types"].items(), key=lambda item: -item[1])[:8])

    return {
        "chains": len(chains),
        "by_link_type": dict(sorted(by_link.items())),
        "rule": "цепочка это correlation_id; незавершённая цепочка остаётся незавершённой",
    }


def build_link_report(
    events_path: Path,
    entity_scan: dict[tuple[str, str], dict],
    transfers: pa.Table,
    coverage: pa.Table,
    history_start: datetime,
) -> dict:

    return {
        "causes": check_causes(events_path),
        "transfers": check_transfers(transfers),
        "entities": check_entities(entity_scan, coverage, history_start),
        "chains": check_chains(events_path),
    }


__all__ = [
    "CAUSE_FIELD",
    "OPENING_TRANSITIONS",
    "REASON_NOT_IN_DATASET",
    "REASON_PRE_WINDOW_CONTRACTS",
    "REASON_UNKNOWN",
    "build_link_report",
    "check_causes",
    "check_chains",
    "check_entities",
    "scan_entities",
    "check_transfers",
]
