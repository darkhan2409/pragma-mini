from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .keys import TIMING_KEYS


# ============================================================
# ИДЕЯ
# ============================================================
#
# Временные признаки, которых нет ни в календаре, ни в самом
# событии: сколько прошло с прошлого события, сколько живёт
# продукт, сколько осталось до известного планового платежа.
#
# Календарь сюда НЕ входит. Час суток, день недели и день месяца
# это отдельный временной канал модели, он считается из
# event_time в preprocessing/calendar.py и здесь не дублируется.
#
# Точность не придумывается: интервал считается в объявленной
# времени события. Время точное, поэтому интервал это прямая
# разность двух моментов: усекать и согласовывать точности
# больше нечего.
#
# Всё считается только по видимым на cutoff событиям. Будущая
# дата возврата, итоговая длительность паузы и прочее знание из
# будущего не рассчитываются.
# ============================================================


HOUR = 3600.0
DAY = 86400.0


def interval_hours(later: datetime, earlier: datetime) -> float:
    """
    Часы между двумя событиями.

    Время события точное, поэтому усекать и согласовывать
    точности больше нечего: разность берётся как есть.
    """

    return _hours(later, earlier)

# Событие «доход»: с него считается время до следующей траты.
INCOME_TYPES: frozenset[str] = frozenset(
    {"salary_credit", "pension_credit", "other_income_credit", "transfer_in", "p2p_in", "cash_deposit"}
)

# Событие, открывающее продукт: от него считается возраст.
OPENING_TYPES: frozenset[str] = frozenset(
    {"product_opened", "account_opened", "card_activated"}
)


@dataclass(frozen=True)
class EventTiming:
    """
    Временные признаки одного видимого события.
    """

    since_previous_hours: float | None
    since_same_type_hours: float | None
    since_last_income_hours: float | None
    age_of_history_days: float | None
    days_to_due: float | None = None

    def as_dict(self) -> dict:
        return {
            "since_previous_hours": self.since_previous_hours,
            "since_same_type_hours": self.since_same_type_hours,
            "since_last_income_hours": self.since_last_income_hours,
            "age_of_history_days": self.age_of_history_days,
            "days_to_due": self.days_to_due,
        }

    def model_values(self) -> dict[str, float]:
        """
        Интервалы, которые получает модель, под объявленными
        ключами. Пустой интервал признаком не становится: «не с
        чем сравнивать» это не ноль.
        """

        return {
            name: value
            for name, value in self.as_dict().items()
            if name in TIMING_KEYS and value is not None
        }


def _hours(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / HOUR


def _days(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / DAY


def timings(rows: list[dict], observed_start: datetime | None) -> list[EventTiming]:
    """
    Интервалы по видимой истории клиента в бизнес-порядке.

    Первое событие интервала не имеет: None здесь означает «не с
    чем сравнивать», а не ноль.
    """

    previous: datetime | None = None
    per_type: dict[str, datetime] = {}
    last_income: datetime | None = None

    out: list[EventTiming] = []

    for row in rows:

        moment = row["event_time"]
        event_type = row["event_type"]

        due = row.get("due_date")

        out.append(
            EventTiming(
                since_previous_hours=(
                    None if previous is None else interval_hours(moment, previous)
                ),
                since_same_type_hours=(
                    None
                    if event_type not in per_type
                    else interval_hours(moment, per_type[event_type])
                ),
                since_last_income_hours=(
                    None if last_income is None else interval_hours(moment, last_income)
                ),
                age_of_history_days=(
                    None if observed_start is None else _days(moment, observed_start)
                ),
                days_to_due=_days_to_due(moment, due),
            )
        )

        previous = moment
        per_type[event_type] = moment

        if event_type in INCOME_TYPES:
            last_income = moment

    return out


def _days_to_due(moment: datetime, due) -> float | None:
    """
    Дней до планового платежа, известного из графика.

    График это законное знание банка на момент договора, поэтому
    будущая дата здесь допустима. Нечитаемая дата признаком не
    становится.
    """

    if not due:
        return None

    try:
        planned = datetime.fromisoformat(str(due))
    except ValueError:
        return None

    return _days(planned, moment)


def product_ages(rows: list[dict], cutoff: datetime) -> dict[str, float]:
    """
    Возраст сущности в днях на cutoff, считая от ВИДИМОГО
    открытия. Сущность без наблюдаемого открытия возраста не
    получает: выдумывать его нельзя.
    """

    opened: dict[str, datetime] = {}

    for row in rows:

        if row["event_type"] not in OPENING_TYPES:
            continue

        known = row["event_time"]

        for column in ("contract_ref", "account_ref", "card_ref"):
            ref = row.get(column)
            if ref is not None and ref not in opened:
                opened[ref] = known

    return {ref: _days(cutoff, moment) for ref, moment in sorted(opened.items())}


def relationship_days(observed_start: datetime | None, cutoff: datetime) -> float | None:
    return None if observed_start is None else _days(cutoff, observed_start)


__all__ = [
    "INCOME_TYPES",
    "OPENING_TYPES",
    "EventTiming",
    "interval_hours",
    "product_ages",
    "relationship_days",
    "timings",
]
