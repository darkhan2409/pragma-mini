from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from .. import params as params_module
from ..life import calendar as cal
from ..life.persona import Persona
from ..rng import KeyedRandom
from ..world import geography, merchants as catalog
from ..world.dictionaries import CATEGORY_BY_NAME
from ..world.merchants import Outlet
from .habits import Era, Habits


# ============================================================
# ВЫБОР ТОРГОВОЙ ТОЧКИ
# ============================================================
#
# Порядок обязателен:
#
#   потребность -> категория -> география -> канал -> цена
#   -> любимое место, периодическое или новое -> outlet
#
# Расстояние, время суток, день недели, график работы, доход,
# ценовая чувствительность, транспорт и поездка влияют на выбор.
# Сумма НЕ определяется одним MCC.
# ============================================================


ZONE_HOME = "home"
ZONE_WORK = "work"
ZONE_OTHER = "other"


@dataclass(frozen=True)
class Choice:
    outlet: Outlet
    zone: str
    channel: str
    is_favourite: bool


def _zone(persona: Persona, era: Era, ts: datetime, rng: KeyedRandom) -> str:

    settings = params_module.active().geography

    weights = dict(settings.zone_share)

    context = cal.day_context(ts)

    if context.is_weekend or context.is_holiday:
        weights[ZONE_WORK] *= 0.25
        weights[ZONE_HOME] *= 1.35

    if era.work_district == era.home_district:
        weights[ZONE_WORK] *= 0.4
        weights[ZONE_HOME] *= 1.2

    if persona.income_type in ("pensioner", "unemployed", "student"):
        weights[ZONE_WORK] *= 0.3

    return rng.weighted(weights)


def _district_for_zone(era: Era, zone: str, settlement: geography.Settlement, rng: KeyedRandom) -> str:

    if zone == ZONE_HOME:
        return era.home_district

    if zone == ZONE_WORK:
        return era.work_district

    return settlement.districts[rng.integers(0, len(settlement.districts))]


def _open_now(outlet: Outlet, ts: datetime) -> bool:

    if outlet.is_online:
        return True

    if outlet.opening_hour == 0 and outlet.closing_hour == 24:
        return True

    return outlet.opening_hour <= ts.hour < outlet.closing_hour


def _price_fit(outlet: Outlet, persona: Persona, ts: datetime) -> float:
    """
    Насколько ценовой сегмент точки подходит клиенту.
    """

    settings = params_module.active().amounts

    sensitivity = persona.trait("price_sensitivity", ts)

    level = settings.price_segment_factor.get(outlet.price_segment, 1.0)

    income_level = min(2.5, max(0.4, persona.true_income / 320_000))

    distance = abs(math.log(level) - math.log(income_level))

    return math.exp(-distance * (0.6 + 1.6 * sensitivity))


def choose_outlet(
    persona: Persona,
    habits: Habits,
    category: str,
    ts: datetime,
    rng: KeyedRandom,
    online_hint: bool | None = None,
    travel_settlement: str | None = None,
    foreign_country: str | None = None,
) -> Choice | None:
    """
    Точка для покупки категории.
    """

    settings = params_module.active()

    if foreign_country is not None:
        outlet = catalog.foreign_outlet(foreign_country, category, rng.integers(0, 40))
        return Choice(outlet=outlet, zone=ZONE_OTHER, channel="pos", is_favourite=False)

    era = habits.era_at(ts)

    settlement_name = travel_settlement or era.settlement

    settlement = geography.by_name(settlement_name)

    pool = catalog.outlets_of(settlement_name, category)

    if not pool:
        return None

    online_share = CATEGORY_BY_NAME[category].online_share

    digital = persona.trait("digital_affinity", ts)

    ecom_level = settings.geography.ecom_share[settlement.settlement_type]

    wants_online = (
        online_hint
        if online_hint is not None
        else rng.random() < min(0.98, online_share * (0.45 + 1.5 * digital) * (0.7 + 1.5 * ecom_level))
    )

    candidates = [item for item in pool if item.is_online == wants_online]

    if not candidates:
        candidates = list(pool)
        wants_online = candidates[0].is_online

    zone = ZONE_OTHER if wants_online else _zone(persona, era, ts, rng)

    target_district = (
        "online"
        if wants_online
        else _district_for_zone(era, zone, settlement, rng)
    )

    # --- любимое место ---

    favourites = era.favourites.get(category, ())

    if favourites and not wants_online and rng.random() < habits.loyalty:

        # Только те точки, что прошли фильтр онлайна: иначе
        # интернет-магазин возвращался бы с каналом pos.
        by_id = {item.outlet_id: item for item in candidates}

        weights = [item.weight for item in favourites]

        pick = favourites[int(rng.choice(len(favourites), p=weights))]

        outlet = by_id.get(pick.outlet_id)

        if outlet is not None and _open_now(outlet, ts):
            return Choice(outlet=outlet, zone=zone, channel="pos", is_favourite=True)

    # --- обычный выбор ---

    decay = settings.geography.distance_decay

    weights = []

    favourite_ids = {item.outlet_id for item in favourites}

    bonus = settings.traits.favourite_outlet_bonus

    closed_weight = settings.merchants.closed_outlet_weight

    for item in candidates:

        weight = max(1e-6, item.popularity)

        if not wants_online:

            if item.district != target_district:
                weight *= math.exp(-decay)

            # Закрытая точка не обслуживает. Раньше она лишь
            # теряла вес и всё равно иногда выигрывала: каждая
            # восьмая покупка приходилась на нерабочий час.
            if not _open_now(item, ts):
                weight *= closed_weight

        # Привычка тянет к любимой точке и там, где выбор идёт
        # по общему правилу.
        if item.outlet_id in favourite_ids:
            weight *= 1.0 + bonus * habits.loyalty

        weight *= _price_fit(item, persona, ts)

        weights.append(weight)

    if sum(weights) <= 0.0:
        return None

    outlet = candidates[int(rng.choice(len(candidates), p=weights))]

    channel = "ecom" if outlet.is_online else "pos"

    return Choice(outlet=outlet, zone=zone, channel=channel, is_favourite=False)


def purchase_amount(
    persona: Persona,
    category: str,
    outlet: Outlet,
    ts: datetime,
    spending_factor: float,
    rng: KeyedRandom,
) -> int:
    """
    Сумма покупки: корзина категории, ценовой уровень точки,
    доход домохозяйства, чувствительность к цене, сезон и
    размер семьи. MCC на сумму напрямую не влияет.
    """

    settings = params_module.active().amounts

    median = settings.basket_median.get(category, 5_000)
    sigma = settings.basket_sigma.get(category, 0.6)

    amount = median * rng.lognormal(0.0, sigma)

    amount *= settings.price_segment_factor.get(outlet.price_segment, 1.0)

    amount *= params_module.active().geography.price_level[
        outlet.settlement_type if outlet.settlement_type != "foreign" else "metropolis"
    ]

    income_ratio = max(0.25, persona.true_income / 320_000)
    amount *= income_ratio ** settings.income_elasticity

    sensitivity = persona.trait("price_sensitivity", ts)
    amount *= 1.0 - settings.price_sensitivity_factor * (sensitivity - 0.5)

    if category in settings.household_size_categories:
        amount *= (1.0 + 0.2 * max(0, persona.household_size - 1)) ** settings.household_size_elasticity

    amount *= cal.category_factor(category, ts, persona.region)

    amount *= spending_factor

    low, high = settings.amount_bounds

    step = settings.round_to

    return int(min(high, max(low, round(amount / step) * step)))


__all__ = [
    "Choice",
    "ZONE_HOME",
    "ZONE_OTHER",
    "ZONE_WORK",
    "choose_outlet",
    "purchase_amount",
]
