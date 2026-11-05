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
    share_factor: float = 1.0,
) -> float:
    """
    Сколько покупок клиент делает в этот день.

    share_factor — какая доля его покупок сейчас идёт через этот
    банк относительно обычной (behaviour/engagement).
    """

    settings = params_module.active().activity

    if "purchases" in silenced:
        return 0.0

    rate = settings.purchases_per_day[persona.activity_mode]

    rate *= settings.state_factor.get(state, 1.0)
    rate *= settings.role_factor.get(persona.hcb_role, 1.0)
    rate *= persona.visible_share * 1.3

    # Импульсивный клиент ходит по магазинам чаще, а не только
    # тратит больше за раз.
    impulsivity = persona.trait("spending_impulsivity", ts)

    rate *= 0.75 + params_module.active().traits.impulsivity_rate_factor * impulsivity

    if cal.is_day_off(ts):
        rate *= settings.weekend_factor_purchases

    rate *= spending_factor
    rate *= share_factor

    return float(max(0.0, rate))


def _category_weights(persona: Persona, habits: Habits, ts: datetime, trip: bool = False) -> dict:

    weights: dict[str, float] = {}

    era = habits.era_at(ts)

    home_only = params_module.active().geography.home_only_categories

    for name in CATEGORY_NAMES:

        category = CATEGORY_BY_NAME[name]

        if name in ("utilities", "telecom", "internet", "subscription", "taxes"):
            # Регулярные платежи идут по расписанию счетов, налог — раз в
            # год к сроку (engine._tax_day), а не случайной покупкой.
            continue

        if trip and name in home_only:
            # В поездке нет домашних дел: налогов, дороги на работу,
            # детсада, ремонта (аудит 2026-11-05, context-03).
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

        population = params_module.active().population

        if persona.transport != "car" and name in population.car_categories:
            weight *= population.car_categories[name]

        if name in ("transit", "taxi") and persona.transport == "car":
            weight *= 0.35

        # Структурные нули: без признака участия категории нет вовсе.
        if (name == "tobacco" and not persona.smokes) or (name == "gambling" and not persona.gambles) \
                or (name == "micromobility" and not persona.scooter):
            continue

        if name == "cinema" and not persona.cinema:
            weight *= 0.03

        allowed = population.category_settlement_types.get(name)
        if allowed and persona.settlement_type not in allowed:
            continue

        weight *= population.category_gender_factor.get(persona.gender, {}).get(name, 1.0)

        age = persona.age_at(ts)
        factor_by_age: dict = {}
        for since, table in population.category_age_factor:
            if age >= since:
                factor_by_age = table
        weight *= factor_by_age.get(name, 1.0)

        if name == "gambling":
            weight *= 0.2 + 1.8 * persona.trait("risk_tolerance", ts)

        if name in ("restaurant", "entertainment", "cinema", "beauty", "cosmetics", "bars"):
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
    share_factor: float = 1.0,
    trip: bool = False,
) -> tuple:
    """
    Намерения покупок на день. trip — день поездки: распорядок
    выходного дня и без домашних категорий.
    """

    settings = params_module.active()

    rate = daily_purchase_rate(persona, ts, state, spending_factor, silenced, share_factor)

    if rate <= 0.0:
        return ()

    day = ts.toordinal()

    count_rng = day_rng(NS_NEEDS, persona.client_ordinal, day, COMPONENT_COUNT)

    count = min(settings.activity.max_purchases_per_day, count_rng.poisson(rate))

    if count <= 0:
        return ()

    has_job = persona.income_type in ("employed", "state_employee", "self_employed", "business_owner")

    # Нерабочий будний день (праздник, перенос) и поездка живут как
    # выходной.
    day_off = trip or cal.is_day_off(ts)

    routine = routine_for(
        weekday=5 if day_off else min(4, ts.weekday()),
        transport=persona.transport,
        has_job=has_job,
        household_size=persona.household_size,
        has_children=persona.children > 0,
    )

    routine_share = (
        settings.merchants.routine_share_weekend
        if day_off
        else settings.merchants.routine_share_weekday
    )

    weights = _category_weights(persona, habits, ts, trip)

    intents: list[Intent] = []

    # Шаг распорядка бывает раз в день, категория — не чаще своего
    # предела (activity.daily_category_cap).
    caps = settings.activity.daily_category_cap
    used: set[int] = set()
    counts: dict[str, int] = {}

    def room(category: str) -> bool:
        return counts.get(category, 0) < caps.get(category, settings.activity.max_purchases_per_day)

    for index in range(count):

        item_rng = event_rng(NS_NEEDS, persona.client_ordinal, day, index + 1, COMPONENT_CONTENT)
        time_rng = event_rng(NS_NEEDS, persona.client_ordinal, day, index + 1, COMPONENT_TIME)

        if item_rng.random() < routine_share and routine:

            position = int(item_rng.integers(0, len(routine)))

            step = routine[position]

            # Шаг распорядка — тоже покупка категории: без признака
            # участия (кино, самокат) или там, где категории нет, его
            # не бывает.
            allowed = (
                step.category in weights
                and (step.category != "cinema" or persona.cinema)
                and position not in used
                and room(step.category)
            )

            if allowed and item_rng.random() <= step.probability:
                used.add(position)
                counts[step.category] = counts.get(step.category, 0) + 1
                # Хронотип сдвигает и распорядок: сова едет на работу и
                # обедает позже.
                hour = min(23, max(0, int(item_rng.integers(step.hour_low, step.hour_high)) + persona.chronotype))
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

        names = [name for name in weights if room(name)]

        if not names:
            continue

        category = str(item_rng.choice(names, p=[weights[name] for name in names]))

        counts[category] = counts.get(category, 0) + 1

        # Час — по профилю клиента и категории: кофе утром, бар вечером.
        hour = int(time_rng.choice(24, p=cal.hour_weights(ts, persona.night_segment, persona.chronotype, category)))

        moment = ts.replace(
            hour=hour,
            minute=int(time_rng.integers(0, 60)),
            second=int(time_rng.integers(0, 60)),
            microsecond=0,
        )

        intents.append(Intent(ts=moment, category=category, zone="other"))

    intents.sort(key=lambda item: item.ts)

    return tuple(intents)


def cash_need(persona: Persona, ts: datetime, silenced: frozenset, share_factor: float = 1.0) -> bool:
    """
    Нужны ли сегодня наличные.
    """

    settings = params_module.active()

    if "cash" in silenced:
        return False

    per_month = settings.activity.cash_withdrawals_per_month[persona.activity_mode]

    cash_level = settings.geography.cash_share[persona.settlement_type]

    rate = per_month * (0.4 + 1.6 * cash_level) / 30.0 * share_factor

    rng = day_rng(NS_NEEDS, persona.client_ordinal, ts.toordinal(), COMPONENT_CHANNEL)

    return rng.random() < rate


__all__ = ["Intent", "cash_need", "daily_intents", "daily_purchase_rate"]
