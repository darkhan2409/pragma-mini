from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .. import params as params_module
from ..life import calendar as cal
from ..life.persona import Persona
from ..rng import (
    NS_NEEDS,
    COMPONENT_CHANNEL,
    COMPONENT_CONTENT,
    COMPONENT_COUNT,
    COMPONENT_TIME,
    day_rng,
    event_rng,
)
from ..world.dictionaries import CATEGORY_BY_NAME, CATEGORY_NAMES
from .habits import Habits
from .routines import routine_for


# ============================================================
# ПОТРЕБНОСТИ ДНЯ
# ============================================================
#
# Потребность рождается из бюджета, привычек, календаря и
# жизненного состояния, а не из независимого розыгрыша строки.
#
# Часть дня это устойчивая бытовая последовательность, часть
# разовые покупки, часть счета по расписанию.
# ============================================================


@dataclass(frozen=True)
class Intent:
    ts: datetime
    category: str
    zone: str
    online_hint: bool | None = None


def daily_purchase_rate(
    persona: Persona,
    ts: datetime,
    state: str,
    spending_factor: float,
    silenced: frozenset,
) -> float:
    """
    Сколько покупок клиент делает в этот день.
    """

    settings = params_module.active().activity

    if "purchases" in silenced:
        if "other_bank" in silenced:
            return settings.purchases_per_day[persona.activity_mode] * settings.other_bank_residual
        return 0.0

    rate = settings.purchases_per_day[persona.activity_mode]

    rate *= settings.state_factor.get(state, 1.0)
    rate *= settings.role_factor.get(persona.hcb_role, 1.0)
    rate *= persona.visible_share * 1.3

    # Импульсивный клиент ходит по магазинам чаще, а не только
    # тратит больше за раз.
    impulsivity = persona.trait("spending_impulsivity", ts)

    rate *= 0.75 + params_module.active().traits.impulsivity_rate_factor * impulsivity

    if ts.weekday() >= 5:
        rate *= settings.weekend_factor_purchases

    rate *= spending_factor

    return float(max(0.0, rate))


def _category_weights(persona: Persona, habits: Habits, ts: datetime) -> dict:

    weights: dict[str, float] = {}

    era = habits.era_at(ts)

    for name in CATEGORY_NAMES:

        category = CATEGORY_BY_NAME[name]

        if name in ("utilities", "telecom", "internet", "subscription"):
            # Регулярные платежи идут по расписанию счетов,
            # а не случайной покупкой.
            continue

        weight = 1.0 if category.essential else 0.45

        # Частота обратна размеру корзины: мелкие покупки
        # определяют счёт событий, крупные — сумму.
        amounts = params_module.active().amounts

        basket = amounts.basket_median.get(name, amounts.reference_basket)

        weight *= (amounts.reference_basket / max(1, basket)) ** amounts.frequency_from_basket

        if name in habits.favourite_categories:
            weight *= 2.4

        if name in era.favourites:
            weight *= 1.3

        weight *= cal.category_factor(name, ts, persona.region)

        if category.need == "kids" and persona.children == 0:
            weight *= 0.05

        if name == "fuel" and persona.transport != "car":
            weight *= 0.10

        if name in ("transit", "taxi") and persona.transport == "car":
            weight *= 0.35

        if name == "gambling":
            weight *= 0.2 + 1.8 * persona.trait("risk_tolerance", ts)

        if name in ("restaurant", "entertainment", "cinema", "beauty", "cosmetics"):
            weight *= 0.4 + 1.4 * persona.trait("spending_impulsivity", ts)

        weights[name] = weight

    return weights


def daily_intents(
    persona: Persona,
    habits: Habits,
    ts: datetime,
    state: str,
    spending_factor: float,
    silenced: frozenset,
) -> tuple:
    """
    Намерения покупок на день.
    """

    settings = params_module.active()

    rate = daily_purchase_rate(persona, ts, state, spending_factor, silenced)

    if rate <= 0.0:
        return ()

    day = ts.toordinal()

    count_rng = day_rng(NS_NEEDS, persona.client_ordinal, day, COMPONENT_COUNT)

    count = min(settings.activity.max_purchases_per_day, count_rng.poisson(rate))

    if count <= 0:
        return ()

    has_job = persona.income_type in ("employed", "state_employee", "self_employed", "business_owner")

    routine = routine_for(
        weekday=ts.weekday(),
        transport=persona.transport,
        has_job=has_job,
        household_size=persona.household_size,
        has_children=persona.children > 0,
    )

    routine_share = (
        settings.merchants.routine_share_weekend
        if ts.weekday() >= 5
        else settings.merchants.routine_share_weekday
    )

    weights = _category_weights(persona, habits, ts)

    hours = cal.hour_weights(ts, persona.night_segment)

    intents: list[Intent] = []

    for index in range(count):

        item_rng = event_rng(NS_NEEDS, persona.client_ordinal, day, index + 1, COMPONENT_CONTENT)
        time_rng = event_rng(NS_NEEDS, persona.client_ordinal, day, index + 1, COMPONENT_TIME)

        if item_rng.random() < routine_share and routine:

            step = routine[item_rng.integers(0, len(routine))]

            if item_rng.random() <= step.probability:
                hour = item_rng.integers(step.hour_low, step.hour_high)
                moment = ts.replace(
                    hour=int(hour),
                    minute=int(time_rng.integers(0, 60)),
                    second=int(time_rng.integers(0, 60)),
                    microsecond=0,
                )
                intents.append(
                    Intent(ts=moment, category=step.category, zone=step.zone)
                )
                continue

        names = list(weights)
        category = str(item_rng.choice(names, p=[weights[name] for name in names]))

        hour = int(time_rng.choice(24, p=hours))

        moment = ts.replace(
            hour=hour,
            minute=int(time_rng.integers(0, 60)),
            second=int(time_rng.integers(0, 60)),
            microsecond=0,
        )

        intents.append(Intent(ts=moment, category=category, zone="other"))

    intents.sort(key=lambda item: item.ts)

    return tuple(intents)


def cash_need(persona: Persona, ts: datetime, silenced: frozenset) -> bool:
    """
    Нужны ли сегодня наличные.
    """

    settings = params_module.active()

    if "cash" in silenced:
        return False

    per_month = settings.activity.cash_withdrawals_per_month[persona.activity_mode]

    cash_level = settings.geography.cash_share[persona.settlement_type]

    rate = per_month * (0.4 + 1.6 * cash_level) / 30.0

    rng = day_rng(NS_NEEDS, persona.client_ordinal, ts.toordinal(), COMPONENT_CHANNEL)

    return rng.random() < rate


__all__ = ["Intent", "cash_need", "daily_intents", "daily_purchase_rate"]
