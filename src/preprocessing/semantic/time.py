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
# точности события. У записи дневной точности час и минута не
# наблюдались, что бы ни стояло в поле времени, поэтому её
# интервалы кратны суткам: оба момента усекаются до более грубой
# точности пары. Признак «время суток» из такого события тоже не
# выводится — это работа календаря и его флага hour_known.
#
# Всё считается только по видимым на cutoff событиям. Будущая
# дата возврата, итоговая длительность паузы и прочее знание из
# будущего не рассчитываются.
# ============================================================


HOUR = 3600.0
DAY = 86400.0

DATE_ONLY = "date_only"

# Объявленные точности от точной к грубой. Точность пары событий
# это более грубая из двух.
PRECISION_ORDER: tuple[str, ...] = ("second", "minute", "day")


class PrecisionError(ValueError):
    """
    Источник объявил точность, которой слой не знает: судить о
    времени по ней нельзя.
    """


def effective_precision(row: dict) -> str:
    """
    Насколько точно источник знает время этой записи.

    Дневное качество отметки в payload значит то же, что дневная
    точность источника: час и минута не наблюдались. Правило одно
    на весь слой: интервалы, порядок в цепочках и hour_known
    смотрят сюда.
    """

    if row.get("timestamp_quality") == DATE_ONLY:
        return "day"

    declared = row.get("time_precision") or "second"

    if declared not in PRECISION_ORDER:
        raise PrecisionError(
            f"неизвестная объявленная точность времени {declared!r}: "
            "судить о времени события по ней нельзя"
        )

    return declared


def coarser(first: str, second: str) -> str:
    return max(first, second, key=PRECISION_ORDER.index)


def floor_to_precision(moment: datetime, precision: str) -> datetime:
    """
    Момент, усечённый до объявленной точности: то, что источник
    действительно знает о времени.
    """

    if precision == "day":
        return moment.replace(hour=0, minute=0, second=0, microsecond=0)

    if precision == "minute":
        return moment.replace(second=0, microsecond=0)

    return moment


def interval_hours(
    later: datetime, later_precision: str, earlier: datetime, earlier_precision: str
) -> float:
    """
    Часы между двумя событиями в точности пары: оба момента
    усечены до более грубой из двух точностей. У пары с дневной
    записью результат кратен 24.
    """

    precision = coarser(later_precision, earlier_precision)

    return _hours(floor_to_precision(later, precision), floor_to_precision(earlier, precision))

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
    time_precision: str
    days_to_due: float | None = None

    def as_dict(self) -> dict:
        return {
            "since_previous_hours": self.since_previous_hours,
            "since_same_type_hours": self.since_same_type_hours,
            "since_last_income_hours": self.since_last_income_hours,
            "age_of_history_days": self.age_of_history_days,
            "time_precision": self.time_precision,
            "days_to_due": self.days_to_due,
        }

    def model_values(self) -> dict[str, float]:
        """
        Интервалы, которые получает модель, под объявленными
        ключами. Пустой интервал признаком не становится: «не с
        чем сравнивать» это не ноль.

        time_precision сюда не входит: объявленная точность это
        признак качества, он остаётся внутри слоя.
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
    чем сравнивать», а не ноль. Каждый интервал считается в
    точности своей пары событий.
    """

    previous: tuple[datetime, str] | None = None
    per_type: dict[str, tuple[datetime, str]] = {}
    last_income: tuple[datetime, str] | None = None

    out: list[EventTiming] = []

    for row in rows:

        moment = row["event_time"]
        event_type = row["event_type"]
        precision = effective_precision(row)

        # Возраст истории и дни до платежа считаются от момента,
        # каким его знает источник: у дневной записи это её дата.
        known = floor_to_precision(moment, precision)

        due = row.get("due_date")

        out.append(
            EventTiming(
                since_previous_hours=(
                    None if previous is None else interval_hours(moment, precision, *previous)
                ),
                since_same_type_hours=(
                    None
                    if event_type not in per_type
                    else interval_hours(moment, precision, *per_type[event_type])
                ),
                since_last_income_hours=(
                    None if last_income is None else interval_hours(moment, precision, *last_income)
                ),
                age_of_history_days=(
                    None
                    if observed_start is None
                    else _days(known, floor_to_precision(observed_start, precision))
                ),
                time_precision=precision,
                days_to_due=_days_to_due(known, due),
            )
        )

        previous = (moment, precision)
        per_type[event_type] = (moment, precision)

        if event_type in INCOME_TYPES:
            last_income = (moment, precision)

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
    получает: выдумывать его нельзя. Момент открытия берётся в
    объявленной точности: у дневной записи это дата.
    """

    opened: dict[str, datetime] = {}

    for row in rows:

        if row["event_type"] not in OPENING_TYPES:
            continue

        known = floor_to_precision(row["event_time"], effective_precision(row))

        for column in ("contract_ref", "account_ref", "card_ref"):
            ref = row.get(column)
            if ref is not None and ref not in opened:
                opened[ref] = known

    return {ref: _days(cutoff, moment) for ref, moment in sorted(opened.items())}


def relationship_days(observed_start: datetime | None, cutoff: datetime) -> float | None:
    return None if observed_start is None else _days(cutoff, observed_start)


__all__ = [
    "DATE_ONLY",
    "INCOME_TYPES",
    "OPENING_TYPES",
    "PRECISION_ORDER",
    "EventTiming",
    "PrecisionError",
    "coarser",
    "effective_precision",
    "floor_to_precision",
    "interval_hours",
    "product_ages",
    "relationship_days",
    "timings",
]
