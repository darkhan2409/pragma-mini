from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime



# ============================================================
# ИДЕЯ
# ============================================================
#
# Активность по месяцам различает четыре случая, и разница между
# ними принципиальна:
#
#   has_client_action        клиент что-то сделал сам;
#   records_without_action   записи есть, но все от банка,
#                            системы или внешнего источника;
#   no_records               записей нет вовсе.
#
# Пустой месяц это доказанное молчание: если события нет,
# значит клиент его не совершал. Отдельной витрины покрытия,
# которая объявляла бы месяц ненаблюдаемым, у выгрузки нет.
#
# Считается только до cutoff. Будущая дата возвращения и итоговая
# длительность паузы не рассчитываются: их знание лежит за
# границей среза.
# ============================================================


HAS_CLIENT_ACTION = "has_client_action"
RECORDS_WITHOUT_ACTION = "records_without_client_action"
NO_RECORDS = "no_records"

MONTH_STATES: tuple[str, ...] = (
    HAS_CLIENT_ACTION,
    RECORDS_WITHOUT_ACTION,
    NO_RECORDS,
)

from ..projection import CLIENT_ACTION_EVENT_TYPES


@dataclass(frozen=True)
class ActivityMonth:
    month: str
    state: str
    events: int
    client_actions: int
    partial: bool

    def as_dict(self) -> dict:
        return {
            "month": self.month,
            "state": self.state,
            "events": self.events,
            "client_actions": self.client_actions,
            "partial": self.partial,
        }


def _month_key(moment: datetime) -> str:
    return f"{moment.year:04d}-{moment.month:02d}"


def _months_before(start: datetime, cutoff: datetime) -> list[tuple[int, int]]:
    """
    Месяцы, у которых есть хоть один прожитый до cutoff день.

    Граница исключительная и здесь: месяц, начинающийся ровно в
    cutoff, не прожит ни на день и месяцем наблюдения не
    считается.
    """

    out: list[tuple[int, int]] = []

    year, month = start.year, start.month

    while datetime(year, month, 1) < cutoff:
        out.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1

    return out


def _next_month(moment: datetime) -> datetime:
    return (
        datetime(moment.year + 1, 1, 1)
        if moment.month == 12
        else datetime(moment.year, moment.month + 1, 1)
    )


def activity_months(
    rows: list[dict],
    observed_start: datetime | None,
    cutoff: datetime,
) -> list[ActivityMonth]:
    """
    Месяцы от наблюдаемого начала до cutoff.

    Неполным месяц называется только тогда, когда граница
    наблюдения действительно прошла внутри него: первый — если
    наблюдение началось не первого числа, последний — если cutoff
    не пришёлся ровно на 00:00 первого числа. Месяц, совпавший с
    границей ровно, полон, и объявлять его неполным значит терять
    настоящий месяц наблюдения.
    """

    if observed_start is None:
        return []

    events: dict[str, int] = {}
    actions: dict[str, int] = {}

    for row in rows:
        key = _month_key(row["event_time"])
        events[key] = events.get(key, 0) + 1
        if row["type"] in CLIENT_ACTION_EVENT_TYPES:
            actions[key] = actions.get(key, 0) + 1

    months = _months_before(observed_start, cutoff)

    out: list[ActivityMonth] = []

    for index, (year, month) in enumerate(months):

        key = f"{year:04d}-{month:02d}"

        count = events.get(key, 0)
        client = actions.get(key, 0)


        if client:
            state = HAS_CLIENT_ACTION
        elif count:
            state = RECORDS_WITHOUT_ACTION
        else:
            state = NO_RECORDS

        first_day = datetime(year, month, 1)

        # Неполон месяц только тогда, когда граница наблюдения
        # действительно прошла внутри него.
        partial = (index == 0 and observed_start > first_day) or (
            index == len(months) - 1 and cutoff < _next_month(first_day)
        )

        out.append(ActivityMonth(key, state, count, client, partial))

    return out


def activity_summary(months: list[ActivityMonth]) -> dict:
    """
    Доли считаются по всем месяцам наблюдения: пустой месяц это
    настоящий ноль, а не пробел.
    """

    counts = {state: sum(1 for item in months if item.state == state) for state in MONTH_STATES}

    silent = sum(1 for item in months if item.state != HAS_CLIENT_ACTION)

    return {
        "months": len(months),
        "by_state": counts,
        "share_without_client_action": (silent / len(months)) if months else None,
        "denominator": "все месяцы от первой записи клиента до cutoff",
        "current_pause_months": _current_pause(months),
    }



def _current_pause(months: list[ActivityMonth]) -> int | None:
    """
    Доказанная пауза к cutoff: сколько месяцев подряд не было
    действий клиента.
    """

    pause = 0

    for item in reversed(months):

        if item.state == HAS_CLIENT_ACTION:
            return pause

        pause += 1

    return pause



__all__ = [
    "HAS_CLIENT_ACTION",
    "MONTH_STATES",
    "NO_RECORDS",
    "RECORDS_WITHOUT_ACTION",
    "ActivityMonth",
    "activity_months",
    "activity_summary",
]
