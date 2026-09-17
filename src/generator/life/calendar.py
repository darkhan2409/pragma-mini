from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .. import params as params_module
from ..rng import state_cache


# ============================================================
# КАЛЕНДАРЬ
# ============================================================
#
# Поведение зависит от часа и части суток, дня недели, рабочего
# или выходного дня, зарплатных окон, начала и конца месяца,
# сезона, праздников Казахстана, учебного года и отпусков.
#
# Праздники заданы таблицей: дат, которых в таблице нет,
# генератор не выдумывает.
# ============================================================


@dataclass(frozen=True)
class DayContext:
    ts: datetime
    weekday: int
    is_weekend: bool
    is_holiday: bool
    holiday: str | None
    pre_holiday: str | None
    post_holiday: bool
    month: int
    is_month_start: bool
    is_winter: bool
    is_school_year: bool
    is_vacation_season: bool


@state_cache
def _holiday_map() -> dict:
    """
    Дата -> имя праздника для всего окна наблюдения.
    """

    settings = params_module.active().seasonality

    table: dict[date, str] = {}

    for year in range(2017, 2031):

        for month, day, name in settings.fixed_holidays:
            try:
                table[date(year, month, day)] = name
            except ValueError:
                continue

        kurban = settings.kurban_ait.get(year)

        if kurban:
            table[date(year, kurban[0], kurban[1])] = "kurban_ait"

    return table


@state_cache
def _pre_holiday_map() -> dict:
    """
    Дата -> праздник, к которому готовятся.
    """

    settings = params_module.active().seasonality

    table: dict[date, str] = {}

    for moment, name in _holiday_map().items():

        window = settings.pre_holiday_days.get(name)

        if not window:
            continue

        for offset in range(1, window + 1):
            day = moment - timedelta(days=offset)
            table.setdefault(day, name)

    return table


@state_cache
def _post_holiday_days() -> frozenset:

    settings = params_module.active().seasonality

    days: set[date] = set()

    for moment, name in _holiday_map().items():
        if name != "new_year":
            continue
        for offset in range(1, settings.post_holiday_dip_days + 1):
            days.add(moment + timedelta(days=offset))

    return frozenset(days)


def day_context(ts: datetime) -> DayContext:

    settings = params_module.active().seasonality

    day = ts.date()

    holiday = _holiday_map().get(day)

    return DayContext(
        ts=ts,
        weekday=ts.weekday(),
        is_weekend=ts.weekday() >= 5,
        is_holiday=holiday is not None,
        holiday=holiday,
        pre_holiday=_pre_holiday_map().get(day),
        post_holiday=day in _post_holiday_days(),
        month=ts.month,
        is_month_start=ts.day <= settings.month_start_days,
        is_winter=ts.month in settings.winter_months,
        is_school_year=ts.month in settings.school_year_months,
        is_vacation_season=ts.month in settings.vacation_months,
    )


def is_business_day(ts: datetime) -> bool:
    return ts.weekday() < 5 and _holiday_map().get(ts.date()) is None


def previous_business_day(ts: datetime) -> datetime:
    moment = ts
    for _ in range(10):
        if is_business_day(moment):
            return moment
        moment -= timedelta(days=1)
    return moment


def next_business_day(ts: datetime) -> datetime:
    moment = ts
    for _ in range(10):
        if is_business_day(moment):
            return moment
        moment += timedelta(days=1)
    return moment


# ============================================================
# СЕЗОННОСТЬ КАТЕГОРИЙ
# ============================================================


def category_factor(category: str, ts: datetime, region: str | None = None) -> float:
    """
    Множитель интенсивности категории: месяц, выходные,
    праздничное окно, зимний регион.
    """

    settings = params_module.active().seasonality

    context = day_context(ts)

    factor = 1.0

    month_table = settings.month_factors.get(category)

    if month_table:
        factor *= month_table.get(context.month, 1.0)

    if context.is_weekend:
        factor *= settings.weekend_factors.get(category, 1.0)

    for name in (context.holiday, context.pre_holiday):
        if name:
            factor *= settings.holiday_factors.get(name, {}).get(category, 1.0)

    if context.post_holiday:
        factor *= settings.post_holiday_dip

    if context.is_month_start:
        factor *= settings.month_start_boost.get(category, 1.0)

    if category == "utilities" and context.is_winter:
        factor *= settings.winter_region_factor.get(region or "", 1.0)

    return float(factor)


def month_factor(ts: datetime) -> float:
    """
    Общий уровень трат месяца: декабрь дороже февраля.
    """

    return float(params_module.active().seasonality.month_factor.get(ts.month, 1.0))


def payday_factor(ts: datetime, last_payday: datetime | None, next_payday: datetime | None) -> float:
    """
    Первые дни после зарплаты тратят больше, перед ней меньше.
    """

    settings = params_module.active().seasonality

    factor = 1.0

    if last_payday is not None:
        elapsed = (ts - last_payday).days
        if 0 <= elapsed <= settings.payday_window_days:
            factor *= settings.payday_boost

    if next_payday is not None:
        remaining = (next_payday - ts).days
        if 0 <= remaining <= settings.pre_payday_days:
            factor *= settings.pre_payday_factor

    return float(factor)


# ============================================================
# ЧАСЫ
# ============================================================


def hour_weights(ts: datetime, night_segment: bool = False) -> tuple:
    """
    Часовой профиль дня. Будни и выходные различаются формой,
    а не только уровнем.
    """

    settings = params_module.active().activity

    base = (
        settings.hour_profile_weekend
        if ts.weekday() >= 5
        else settings.hour_profile_weekday
    )

    if not night_segment:
        return base

    boosted = list(base)

    for hour in settings.night_hours:
        boosted[hour] *= settings.night_boost

    total = sum(boosted)

    return tuple(value / total for value in boosted)


def month_start(ts: datetime) -> datetime:
    return datetime(ts.year, ts.month, 1)


def next_month(ts: datetime) -> datetime:
    return datetime(ts.year + 1, 1, 1) if ts.month == 12 else datetime(ts.year, ts.month + 1, 1)


def month_end(ts: datetime) -> datetime:
    return next_month(ts) - timedelta(seconds=1)


def add_months(anchor: datetime, months: int) -> datetime:
    index = anchor.month - 1 + months
    year = anchor.year + index // 12
    month = index % 12 + 1
    return anchor.replace(year=year, month=month, day=min(anchor.day, 28))


def month_index(ts: datetime) -> int:
    return ts.year * 12 + ts.month


def day_in_month(ts: datetime, day: int) -> datetime:
    try:
        return ts.replace(day=day)
    except ValueError:
        return month_end(ts).replace(hour=0, minute=0, second=0, microsecond=0)


__all__ = [
    "month_factor",
    "DayContext",
    "add_months",
    "category_factor",
    "day_context",
    "day_in_month",
    "hour_weights",
    "is_business_day",
    "month_end",
    "month_index",
    "month_start",
    "next_business_day",
    "next_month",
    "payday_factor",
    "previous_business_day",
]
