from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
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


def check_transfers(transfers: pa.Table) -> dict:

    if transfers.num_rows == 0:
        return {"sides": 0, "rule": "переводов в выгрузке нет"}

    sides = Counter(transfers.column("side").to_pylist())

    # Парность считается по СТОРОНАМ, а не по числу клиентов:
    # перевод между своими счетами наблюдается целиком у одного
    # человека, и раньше он попадал в «видна одна сторона».
    by_id: dict[str, set[str]] = defaultdict(set)
    for transfer_id, side in zip(
        transfers.column("transfer_id").to_pylist(), transfers.column("side").to_pylist()
    ):
        by_id[transfer_id].add(side)

    return {
        "sides": transfers.num_rows,
        "transfers": len(by_id),
        "by_side": dict(sorted(sides.items())),
        "both_sides_in_full_extract": sum(1 for kinds in by_id.values() if len(kinds) > 1),
        "one_side_in_full_extract": sum(1 for kinds in by_id.values() if len(kinds) == 1),
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


def check_entities(seen: dict[tuple[str, str], dict], history_start: datetime) -> dict:
    """
    У каждой ли наблюдаемой сущности есть наблюдаемое открытие.

    На вход идёт накопитель, собранный по пачкам.
    """

    if not seen:
        return {"entities": 0}

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
        else:
            slot["reasons"][REASON_UNKNOWN] += 1

    for slot in per_kind.values():
        slot["reasons"] = dict(sorted(slot["reasons"].items()))

    return {
        "entities": len(seen),
        "by_kind": dict(sorted(per_kind.items())),
        "rule": (
            "открытие ищется среди наблюдаемых переходов и только там, где оно определено; "
            "отсутствие открытия объясняется первым упоминанием до начала окна, иначе "
            "остаётся ненаблюдаемым и не достраивается"
        ),
    }


# Деловые ключи payload, каждый из которых собирает свою
# цепочку. Метки связи в конверте больше нет: вид цепочки задаёт
# имя ключа, и переименовать его молча нельзя.
CHAIN_FIELDS: tuple[str, ...] = (
    "application_id",
    "contract_id",
    "case_id",
    "offer_id",
    "session_id",
    "transfer_id",
)


def check_chains(events_path: Path) -> dict:
    """
    Цепочки по деловым ключам payload: сколько, какого размера и
    из чего.

    Одна запись может входить в несколько цепочек сразу: платёж
    из приложения принадлежит и договору, и сессии. Это не
    двойной счёт, а два разных разреза одной ленты.
    """

    parquet = pq.ParquetFile(events_path)

    present = [name for name in CHAIN_FIELDS if name in parquet.schema_arrow.names]

    if not present:
        return {"chains": 0, "by_field": {}, "rule": "деловых ключей связи в выгрузке нет"}

    chains: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for index in range(parquet.num_row_groups):

        chunk = parquet.read_row_group(index, columns=[*present, "event_type"])

        types = chunk.column("event_type").to_pylist()

        for field_name in present:
            for value, event_type in zip(chunk.column(field_name).to_pylist(), types):
                if not value:
                    continue
                chains[(field_name, value)][event_type] += 1

    by_field: dict[str, dict] = {}

    for (field_name, _), composition in chains.items():
        slot = by_field.setdefault(field_name, {"chains": 0, "sizes": Counter(), "event_types": Counter()})
        slot["chains"] += 1
        slot["sizes"][sum(composition.values())] += 1
        slot["event_types"].update(composition)

    for slot in by_field.values():
        slot["sizes"] = _quantiles(slot["sizes"])
        slot["event_types"] = dict(sorted(slot["event_types"].items(), key=lambda item: -item[1])[:8])

    return {
        "chains": len(chains),
        "by_field": dict(sorted(by_field.items())),
        "rule": (
            "цепочка это деловой ключ payload; одна запись может входить в несколько цепочек, "
            "незавершённая цепочка остаётся незавершённой"
        ),
    }


def build_link_report(
    events_path: Path,
    entity_scan: dict[tuple[str, str], dict],
    transfers: pa.Table,
    history_start: datetime,
) -> dict:

    return {
        "transfers": check_transfers(transfers),
        "entities": check_entities(entity_scan, history_start),
        "chains": check_chains(events_path),
    }


__all__ = [
    "OPENING_TRANSITIONS",
    "REASON_NOT_IN_DATASET",
    "REASON_PRE_WINDOW_CONTRACTS",
    "REASON_UNKNOWN",
    "build_link_report",
    "check_chains",
    "check_entities",
    "scan_entities",
    "check_transfers",
]
