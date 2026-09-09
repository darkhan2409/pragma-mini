from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from .categories import (
    CATEGORIES,
    DIRECTION_CREDIT,
    DIRECTION_DEBIT,
    GROUP_BASE_WEIGHT,
    GROUPS,
    MCC_CASH_IN,
    MCC_SALARY,
    MCC_TRANSFER_IN,
)
from .persona import Persona, draw_persona
from .rng import (
    COMPONENT_CONTENT,
    COMPONENT_COUNT,
    COMPONENT_TIME,
    NS_TX,
    KeyedRandom,
    day_rng,
    event_rng,
)
from .season import season_factor
from .trajectory import BehaviorState, behavior_state
from .world import (
    CITY_TAIL,
    FOREIGN_COUNTRIES,
    FOREIGN_COUNTRY_WEIGHTS,
    HOME_COUNTRY,
    MAIN_CITY_SHARE,
)


# ============================================================
# КОНТРАКТ
# ============================================================
#
#     ts, amount, direction, mcc, merchant_city,
#     merchant_country, is_online, is_subscription
#
# Валюты в V1 сознательно нет: все суммы в тенге.
#
# Поток состоит из трёх частей:
#   покупки       пуассоновский поток по категориям
#   зарплата      ежемесячное зачисление в день зарплаты
#   подписки      ежемесячное списание фиксированной суммы
# ============================================================


@dataclass(frozen=True)
class TransactionEvent:
    client_id: int
    ts: datetime

    amount: int
    direction: str
    mcc: str
    merchant_city: str | None
    merchant_country: str
    is_online: bool
    is_subscription: bool


# ============================================================
# ИНТЕНСИВНОСТЬ ПОКУПОК
# ============================================================

WINTER_MONTHS = frozenset({12, 1, 2})

WINTER_REDUCTION = {
    "Astana": 0.94, "Petropavl": 0.94, "Kostanay": 0.95, "Pavlodar": 0.95,
    "Karaganda": 0.95, "Oskemen": 0.95, "Semey": 0.96, "Aktobe": 0.97,
    "Oral": 0.97, "Atyrau": 0.98, "Almaty": 0.98, "Taldykorgan": 0.98,
    "Taraz": 0.99, "Kyzylorda": 0.99, "Shymkent": 1.00,
}


def daily_purchase_rate(persona: Persona, ts: datetime) -> float:
    """
    Ожидаемое число покупок за день.
    """

    rate = 0.25 + 1.20 * persona.activity

    if ts.weekday() >= 5:
        rate *= 1.08

    if ts.month in WINTER_MONTHS:
        rate *= WINTER_REDUCTION.get(persona.region, 0.98)

    if ts.month == 12 and ts.day >= 15:
        rate *= 1.15

    if ts.month == 1 and 2 <= ts.day <= 10:
        rate *= 0.90

    rate *= behavior_state(persona.client_id, ts).activity_multiplier

    return float(rate)


# ============================================================
# ВЫБОР КАТЕГОРИИ
# ============================================================

BASE_WEIGHTS = np.array([GROUP_BASE_WEIGHT[g] for g in GROUPS], dtype=float)


def _mask(groups: set[str]) -> np.ndarray:
    return np.array([g in groups for g in GROUPS], dtype=float)


DIGITAL_GROUPS = {"marketplace", "delivery", "telecom"}
MOBILITY_GROUPS = {"taxi", "fuel", "travel", "hotel", "airline"}
INCOME_GROUPS = {
    "restaurant", "electronics", "clothing", "shoes",
    "furniture", "beauty", "travel", "hotel", "airline",
}
HEALTH_GROUPS = {"pharmacy", "medical"}

DISCRETIONARY_GROUPS = {
    "restaurant", "coffee", "clothing", "shoes", "electronics", "furniture",
    "home_goods", "cosmetics", "beauty", "sports", "entertainment", "cinema",
    "travel", "hotel", "airline", "gambling",
}
ESSENTIAL_GROUPS = {"grocery", "pharmacy", "utilities", "transit", "telecom"}

ESSENTIAL_STRESS_SLOPE = 0.25

_DIGITAL = _mask(DIGITAL_GROUPS)
_MOBILITY = _mask(MOBILITY_GROUPS)
_INCOME = _mask(INCOME_GROUPS)
_HEALTH = _mask(HEALTH_GROUPS)
_DISCRETIONARY = _mask(DISCRETIONARY_GROUPS)
_ESSENTIAL = _mask(ESSENTIAL_GROUPS)


@lru_cache(maxsize=131_072)
def persona_group_vector(client_id: int) -> np.ndarray:
    """
    Постоянные предпочтения категорий у клиента.
    """

    persona = draw_persona(client_id)

    digital = 0.70 + 0.80 * persona.digital_affinity
    mobility = 0.65 + 1.10 * persona.mobility

    income_ratio = persona.declared_income / 350_000
    income = min(1.60, max(0.70, income_ratio ** 0.20))

    health = min(1.45, max(0.70, 0.70 + (persona.age - 18) / 70))

    factor = np.ones(len(GROUPS), dtype=float)
    factor *= np.where(_DIGITAL > 0, digital, 1.0)
    factor *= np.where(_MOBILITY > 0, mobility, 1.0)
    factor *= np.where(_INCOME > 0, income, 1.0)
    factor *= np.where(_HEALTH > 0, health, 1.0)

    return factor


@lru_cache(maxsize=65_536)
def season_group_vector(day: int, region: str) -> np.ndarray:

    ts = datetime.fromordinal(day)

    return np.array(
        [season_factor(group=g, ts=ts, region=region) for g in GROUPS],
        dtype=float,
    )


def stress_group_vector(state: BehaviorState) -> np.ndarray:
    """
    В кредитном стрессе состав покупок смещается:
    дискреционные категории сжимаются, базовые растут.
    """

    factor = np.ones(len(GROUPS), dtype=float)
    factor *= np.where(_DISCRETIONARY > 0, state.discretionary_multiplier, 1.0)
    factor *= np.where(
        _ESSENTIAL > 0,
        1.0 + ESSENTIAL_STRESS_SLOPE * state.credit_stress,
        1.0,
    )

    return factor


def choose_group(persona: Persona, ts: datetime, rng: KeyedRandom) -> str:

    state = behavior_state(persona.client_id, ts)

    weights = (
        BASE_WEIGHTS
        * persona_group_vector(persona.client_id)
        * season_group_vector(ts.toordinal(), persona.region)
        * stress_group_vector(state)
    )

    cumulative = np.cumsum(weights)

    index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))

    return GROUPS[min(index, len(GROUPS) - 1)]


# ============================================================
# ВРЕМЯ
# ============================================================

HOUR_PROFILE = np.array(
    [
        0.010, 0.005, 0.003, 0.003, 0.003, 0.008,
        0.025, 0.055, 0.075, 0.080, 0.075, 0.070,
        0.080, 0.080, 0.070, 0.065, 0.070, 0.080,
        0.095, 0.100, 0.085, 0.065, 0.040, 0.020,
    ],
    dtype=float,
)


@lru_cache(maxsize=256)
def hour_cumulative(hours: tuple[int, ...]) -> np.ndarray:

    weights = HOUR_PROFILE.copy()

    allowed = set(hours)
    weights *= np.array([1.0 if h in allowed else 0.0 for h in range(24)])

    if weights.sum() <= 0:
        weights = HOUR_PROFILE.copy()

    return np.cumsum(weights)


def draw_time(day: datetime, hours: tuple[int, ...], rng: KeyedRandom) -> datetime:

    cumulative = hour_cumulative(hours)

    hour = min(
        23,
        int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right")),
    )

    return day.replace(
        hour=hour,
        minute=rng.integers(0, 60),
        second=rng.integers(0, 60),
        microsecond=0,
    )


# ============================================================
# ГЕОГРАФИЯ ТОЧКИ
# ============================================================


def draw_location(
    persona: Persona,
    group: str,
    is_online: bool,
    rng: KeyedRandom,
) -> tuple[str | None, str]:
    """
    Город и страна торговой точки.

    Зарубежная точка: страна из хвоста, город не разбирается.
    """

    category = CATEGORIES[group]

    foreign_share = category.foreign_share * (0.5 + 1.5 * persona.mobility)

    if rng.random() < foreign_share:
        country = str(rng.choice(FOREIGN_COUNTRIES, p=FOREIGN_COUNTRY_WEIGHTS))
        return None, country

    if is_online:
        # Онлайн-точка зарегистрирована в крупном городе.
        city = "Almaty" if rng.random() < 0.65 else "Astana"
        return city, HOME_COUNTRY

    if rng.random() < MAIN_CITY_SHARE:
        return persona.region, HOME_COUNTRY

    tail = CITY_TAIL[persona.region]

    return str(rng.choice(tail)), HOME_COUNTRY


# ============================================================
# ПОКУПКА
# ============================================================


def draw_amount(group: str, persona: Persona, state: BehaviorState, rng: KeyedRandom) -> int:

    category = CATEGORIES[group]

    raw = rng.lognormal(math.log(category.typical_amount), category.log_sigma)

    income_factor = min(1.40, max(0.80, (persona.declared_income / 350_000) ** 0.12))

    amount = raw * income_factor * state.spending_multiplier

    return int(round(min(5_000_000.0, max(100.0, amount)) / 10) * 10)


def generate_purchase(client_id: int, day: datetime, index: int) -> TransactionEvent:

    persona = draw_persona(client_id)
    state = behavior_state(client_id, day)

    rng = event_rng(NS_TX, client_id, day.toordinal(), index, COMPONENT_CONTENT)

    group = choose_group(persona, day, rng)
    category = CATEGORIES[group]

    mcc = str(rng.choice(category.mccs, p=category.mcc_weights))

    online_share = min(
        1.0,
        category.online_share * (0.55 + 0.90 * persona.digital_affinity),
    )
    is_online = rng.random() < online_share

    city, country = draw_location(persona, group, is_online, rng)

    amount = draw_amount(group, persona, state, rng)

    time_rng = event_rng(NS_TX, client_id, day.toordinal(), index, COMPONENT_TIME)

    hours = tuple(range(24)) if is_online else category.hours

    ts = draw_time(day, hours, time_rng)

    return TransactionEvent(
        client_id=client_id,
        ts=ts,
        amount=amount,
        direction=DIRECTION_DEBIT,
        mcc=mcc,
        merchant_city=city,
        merchant_country=country,
        is_online=is_online,
        is_subscription=False,
    )


# ============================================================
# ПОДПИСКИ
# ============================================================
#
# Регулярные списания одной суммы в один и тот же день месяца.
# Для sequence-модели это самый чистый повторяющийся паттерн.
# ============================================================

SUBSCRIPTION_GROUPS = tuple(
    group for group, category in CATEGORIES.items() if category.subscription_share > 0
)

SUBSCRIPTION_WEIGHTS = tuple(
    CATEGORIES[group].subscription_share for group in SUBSCRIPTION_GROUPS
)


@dataclass(frozen=True)
class Subscription:
    group: str
    mcc: str
    amount: int
    day_of_month: int


@lru_cache(maxsize=131_072)
def client_subscriptions(client_id: int) -> tuple[Subscription, ...]:

    persona = draw_persona(client_id)

    rng = event_rng(NS_TX, client_id, 0, 0, COMPONENT_CONTENT)

    count = rng.poisson(0.6 + 2.2 * persona.digital_affinity)
    count = min(count, 5)

    subscriptions: list[Subscription] = []
    used: set[str] = set()

    for _ in range(count):

        group = str(rng.choice(SUBSCRIPTION_GROUPS, p=SUBSCRIPTION_WEIGHTS))

        if group in used:
            continue

        used.add(group)

        category = CATEGORIES[group]

        amount = int(
            round(
                rng.lognormal(math.log(category.typical_amount), 0.35)
                * min(1.4, max(0.8, (persona.declared_income / 350_000) ** 0.15))
                / 10
            )
            * 10
        )

        subscriptions.append(
            Subscription(
                group=group,
                mcc=str(rng.choice(category.mccs, p=category.mcc_weights)),
                amount=max(500, amount),
                day_of_month=rng.integers(1, 29),
            )
        )

    return tuple(subscriptions)


def subscription_events(
    client_id: int,
    start: datetime,
    end: datetime,
) -> list[TransactionEvent]:

    persona = draw_persona(client_id)

    events: list[TransactionEvent] = []

    for index, subscription in enumerate(client_subscriptions(client_id)):

        month = datetime(start.year, start.month, 1)

        while month < end:

            try:
                due = month.replace(day=subscription.day_of_month)
            except ValueError:
                due = month

            if start <= due < end:

                rng = event_rng(
                    NS_TX,
                    client_id,
                    due.toordinal(),
                    900 + index,
                    COMPONENT_CONTENT,
                )

                # Подписка списывается в фиксированный час,
                # но не строго в одну и ту же секунду.
                ts = due.replace(
                    hour=rng.integers(2, 7),
                    minute=rng.integers(0, 60),
                    second=rng.integers(0, 60),
                    microsecond=0,
                )

                city, country = draw_location(persona, subscription.group, True, rng)

                events.append(
                    TransactionEvent(
                        client_id=client_id,
                        ts=ts,
                        amount=subscription.amount,
                        direction=DIRECTION_DEBIT,
                        mcc=subscription.mcc,
                        merchant_city=city,
                        merchant_country=country,
                        is_online=True,
                        is_subscription=True,
                    )
                )

            month = (
                month.replace(year=month.year + 1, month=1)
                if month.month == 12
                else month.replace(month=month.month + 1)
            )

    return events


# ============================================================
# ЗАЧИСЛЕНИЯ
# ============================================================
#
# Зарплата приходит в свой день месяца. Под кредитным стрессом
# она задерживается и иногда не приходит вовсе: это самый
# ранний наблюдаемый признак проблем.
# ============================================================

SALARY_INCOME_TYPES = frozenset(
    {"employed", "state_employee", "self_employed", "business_owner", "pensioner"}
)


def salary_events(
    client_id: int,
    start: datetime,
    end: datetime,
) -> list[TransactionEvent]:

    persona = draw_persona(client_id)

    if persona.income_type not in SALARY_INCOME_TYPES:
        return []

    events: list[TransactionEvent] = []

    month = datetime(start.year, start.month, 1)

    while month < end:

        try:
            due = month.replace(day=persona.salary_day)
        except ValueError:
            due = month

        rng = event_rng(NS_TX, client_id, due.toordinal(), 800, COMPONENT_CONTENT)

        state = behavior_state(client_id, due)
        stress = state.credit_stress

        # Под стрессом зарплата задерживается и иногда пропадает.
        if rng.random() < 0.35 * stress ** 2:
            month = _next_month(month)
            continue

        delay = 0

        if rng.random() < 0.15 + 0.55 * stress:
            delay = rng.integers(1, 2 + int(8 * stress))

        ts = due + timedelta(days=delay)

        ts = ts.replace(
            hour=rng.integers(6, 12),
            minute=rng.integers(0, 60),
            second=rng.integers(0, 60),
            microsecond=0,
        )

        if start <= ts < end:

            share = 0.80 + 0.35 * rng.random()

            amount = int(round(persona.declared_income * share / 100) * 100)

            events.append(
                TransactionEvent(
                    client_id=client_id,
                    ts=ts,
                    amount=amount,
                    direction=DIRECTION_CREDIT,
                    mcc=MCC_SALARY,
                    merchant_city=None,
                    merchant_country=HOME_COUNTRY,
                    is_online=True,
                    is_subscription=False,
                )
            )

        month = _next_month(month)

    return events


def _next_month(month: datetime) -> datetime:
    return (
        month.replace(year=month.year + 1, month=1)
        if month.month == 12
        else month.replace(month=month.month + 1)
    )


def other_credit_events(
    client_id: int,
    start: datetime,
    end: datetime,
) -> list[TransactionEvent]:
    """
    Прочие зачисления: переводы от людей и внесение наличных.
    Под стрессом клиент чаще получает переводы.
    """

    persona = draw_persona(client_id)

    events: list[TransactionEvent] = []

    day = start.replace(hour=0, minute=0, second=0, microsecond=0)

    while day < end:

        state = behavior_state(client_id, day)

        rate = (
            (0.02 + 0.05 * persona.digital_affinity)
            * state.activity_multiplier
            * (1.0 + 1.6 * state.credit_stress)
        )

        count = day_rng(NS_TX, client_id, day.toordinal(), 700).poisson(rate)

        for index in range(count):

            rng = event_rng(NS_TX, client_id, day.toordinal(), 700 + index, COMPONENT_CONTENT)

            is_transfer = rng.random() < 0.75

            amount = int(
                round(
                    rng.lognormal(math.log(25_000 if is_transfer else 60_000), 0.85)
                    / 100
                )
                * 100
            )

            ts = draw_time(day, tuple(range(8, 23)), rng)

            if not (start <= ts < end):
                continue

            events.append(
                TransactionEvent(
                    client_id=client_id,
                    ts=ts,
                    amount=max(500, min(3_000_000, amount)),
                    direction=DIRECTION_CREDIT,
                    mcc=MCC_TRANSFER_IN if is_transfer else MCC_CASH_IN,
                    merchant_city=None if is_transfer else persona.region,
                    merchant_country=HOME_COUNTRY,
                    is_online=is_transfer,
                    is_subscription=False,
                )
            )

        day += timedelta(days=1)

    return events


# ============================================================
# ПОЛНАЯ ИСТОРИЯ
# ============================================================


def generate_transaction_history(
    client_id: int,
    start: datetime,
    end: datetime,
) -> list[TransactionEvent]:
    """
    Все транзакции клиента на [start, end).
    """

    if end <= start:
        raise ValueError("end must be after start")

    persona = draw_persona(client_id)

    events: list[TransactionEvent] = []

    day = start.replace(hour=0, minute=0, second=0, microsecond=0)

    while day < end:

        count = day_rng(NS_TX, client_id, day.toordinal()).poisson(
            daily_purchase_rate(persona, day)
        )

        for index in range(count):

            event = generate_purchase(client_id, day, index)

            if start <= event.ts < end:
                events.append(event)

        day += timedelta(days=1)

    events.extend(subscription_events(client_id, start, end))
    events.extend(salary_events(client_id, start, end))
    events.extend(other_credit_events(client_id, start, end))

    events.sort(key=lambda event: event.ts)

    return events
