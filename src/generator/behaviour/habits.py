from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..life import calendar as cal
from ..life.persona import Persona
from ..rng import NS_HABITS, keyed_rng, stable_hash
from ..world import merchants
from ..world.dictionaries import CATEGORY_BY_NAME, CATEGORY_NAMES


# ============================================================
# ПРИВЫЧКИ
# ============================================================
#
# У клиента есть любимые торговые точки в частых категориях,
# набор периодически используемых мест и длинный хвост редких
# покупок.
#
# Привычки меняются после переезда, смены работы и изменения
# состава семьи: старые точки перестают быть удобными.
#
# Счета и подписки это не случайные покупки: у них свой день
# и повторяющаяся сумма.
# ============================================================


@dataclass(frozen=True)
class Favourite:
    category: str
    outlet_id: str
    weight: float


@dataclass(frozen=True)
class Era:
    """
    Отрезок жизни с одним набором привычек.
    """

    valid_from: datetime
    settlement: str
    home_district: str
    work_district: str
    favourites: dict
    cause: str


@dataclass(frozen=True)
class Bill:
    kind: str
    category: str
    day_of_month: int
    base_amount: int
    sigma: float
    valid_from: datetime
    valid_to: datetime | None
    autopay: bool


@dataclass(frozen=True)
class Subscription:
    category: str
    outlet_id: str
    day_of_month: int
    amount: int
    amount_after: int
    start_month: int
    end_month: int | None
    change_month: int | None


@dataclass(frozen=True)
class Habits:
    eras: tuple
    bills: tuple
    subscriptions: tuple
    favourite_categories: tuple
    loyalty: float
    new_place_chance: float

    def era_at(self, ts: datetime) -> Era:
        chosen = self.eras[0]
        for era in self.eras:
            if era.valid_from <= ts:
                chosen = era
            else:
                break
        return chosen


def _favourites_for(
    persona: Persona,
    settlement: str,
    categories: tuple,
    salt: int,
) -> dict:

    settings = params_module.active().merchants

    favourites: dict[str, tuple] = {}

    for category in categories:

        pool = merchants.outlets_of(settlement, category)

        if not pool:
            continue

        rng = keyed_rng(NS_HABITS, persona.client_ordinal, salt, stable_hash(category) % (2 ** 20))

        low, high = settings.favourite_outlets_per_category

        # Лояльный клиент держится одной точки, нелояльный
        # разбрасывается по нескольким. Без этого лояльность
        # никак не проявлялась в доле повторных покупок.
        loyalty = persona.trait("merchant_loyalty")

        span = max(0, high - low)

        count = high - int(round(span * loyalty))

        count = min(len(pool), max(low, count))

        weights = [item.popularity for item in pool]

        chosen: list[Favourite] = []
        seen: set[str] = set()

        for index in range(count):
            pick = pool[int(rng.choice(len(pool), p=weights))]
            if pick.outlet_id in seen:
                continue
            seen.add(pick.outlet_id)
            chosen.append(
                Favourite(category=category, outlet_id=pick.outlet_id, weight=1.0 / (index + 1))
            )

        if chosen:
            favourites[category] = tuple(chosen)

    return favourites


def _habit_categories(persona: Persona) -> tuple:

    settings = params_module.active().merchants

    rng = keyed_rng(NS_HABITS, persona.client_ordinal, 1)

    low, high = settings.favourite_categories
    count = rng.integers(low, high + 1)

    essential = [name for name in CATEGORY_NAMES if CATEGORY_BY_NAME[name].essential]

    optional = [name for name in CATEGORY_NAMES if not CATEGORY_BY_NAME[name].essential]

    chosen = list(essential[: min(len(essential), 4)])

    for _ in range(max(0, count - len(chosen))):
        pick = optional[rng.integers(0, len(optional))]
        if pick not in chosen:
            chosen.append(pick)

    return tuple(chosen)


def _bills(persona: Persona, events: tuple) -> tuple:

    settings = params_module.active().amounts

    rng = keyed_rng(NS_HABITS, persona.client_ordinal, 2)

    start = max(config.HISTORY_START, persona.relationship_start)

    autopay_chance = 0.25 + 0.5 * persona.trait("financial_discipline")

    bills: list[Bill] = []

    def add(kind: str, category: str, factor: float = 1.0, valid_from=None) -> None:
        base = int(settings.bill_medians[kind] * factor * rng.uniform(0.75, 1.35))
        bills.append(
            Bill(
                kind=kind,
                category=category,
                day_of_month=int(rng.integers(3, 26)),
                base_amount=base,
                sigma=float(settings.bill_sigma[kind]),
                valid_from=valid_from or start,
                valid_to=None,
                autopay=bool(rng.random() < autopay_chance),
            )
        )

    household_factor = 1.0 + 0.16 * max(0, persona.household_size - 1)

    if persona.housing_type != "with_parents":
        add("utilities", "utilities", household_factor)

    add("telecom", "telecom")

    if persona.settlement_type in ("metropolis", "major_city", "regional_centre", "industrial_town"):
        if rng.random() < 0.72:
            add("internet", "internet")

    if persona.children > 0 and rng.random() < 0.45:
        add("kindergarten", "kids")

    # Ребёнок рождается внутри окна: счёт появляется позже.
    for event in events:
        if event.kind == "child_birth" and rng.random() < 0.35:
            add("kindergarten", "kids", 1.0, event.ts + timedelta(days=int(rng.integers(120, 540))))

    return tuple(bills)


def _subscriptions(persona: Persona, settlement: str) -> tuple:

    settings = params_module.active().amounts

    rng = keyed_rng(NS_HABITS, persona.client_ordinal, 3)

    digital = persona.trait("digital_affinity")

    low, high = settings.subscription_count

    count = min(high, rng.poisson(low + (high - low) * digital * 0.6))

    pool = merchants.outlets_of(settlement, "subscription")

    if not pool:
        return ()

    result: list[Subscription] = []

    first_month = cal.month_index(config.HISTORY_START)
    last_month = cal.month_index(config.HISTORY_END)

    for index in range(count):

        item_rng = keyed_rng(NS_HABITS, persona.client_ordinal, 4, index)

        outlet = pool[int(item_rng.choice(len(pool), p=[item.popularity for item in pool]))]

        amount = int(
            settings.subscription_median * item_rng.lognormal(0.0, settings.subscription_sigma)
        )
        amount = max(500, int(round(amount / 50) * 50))

        start_month = first_month

        if item_rng.random() < 0.35:
            start_month = first_month + int(item_rng.integers(1, max(2, last_month - first_month - 2)))

        end_month = None

        if item_rng.random() < settings.subscription_stop_share:
            end_month = start_month + int(item_rng.integers(2, max(3, last_month - start_month)))

        change_month = None
        amount_after = amount

        if item_rng.random() < settings.subscription_price_change_share:
            change_month = start_month + int(item_rng.integers(3, max(4, last_month - start_month)))
            direction = 1.0 if item_rng.random() < 0.75 else -1.0
            amount_after = int(
                amount * (1.0 + direction * item_rng.uniform(*settings.subscription_price_change))
            )
            amount_after = max(500, int(round(amount_after / 50) * 50))

        result.append(
            Subscription(
                category="subscription",
                outlet_id=outlet.outlet_id,
                day_of_month=int(item_rng.integers(1, 28)),
                amount=amount,
                amount_after=amount_after,
                start_month=start_month,
                end_month=end_month,
                change_month=change_month,
            )
        )

    return tuple(result)


def build_habits(persona: Persona, events: tuple) -> Habits:
    """
    Привычки клиента с учётом переездов и жизненных событий.
    """

    settings = params_module.active().merchants

    categories = _habit_categories(persona)

    eras: list[Era] = [
        Era(
            valid_from=min(config.HISTORY_START, persona.relationship_start),
            settlement=persona.settlement,
            home_district=persona.home_district,
            work_district=persona.work_district,
            favourites=_favourites_for(persona, persona.settlement, categories, 10),
            cause="initial",
        )
    ]

    salt = 11

    for event in events:

        rng = keyed_rng(NS_HABITS, persona.client_ordinal, 5, int(event.ts.toordinal()))

        if event.kind == "move":
            chance = settings.habit_reset_on_move
        elif event.kind == "job_change":
            chance = settings.habit_reset_on_job_change
        elif event.kind == "child_birth":
            chance = settings.habit_reset_on_child_birth
        else:
            continue

        if rng.random() >= chance:
            continue

        last = eras[-1]

        settlement = str(event.payload.get("settlement", last.settlement))
        home_district = str(event.payload.get("district", last.home_district))

        work_district = home_district if event.kind == "move" else last.work_district

        if event.kind == "job_change":
            place = settlement
            districts = merchants.geography.by_name(place).districts
            work_district = districts[rng.integers(0, len(districts))]

        eras.append(
            Era(
                valid_from=event.ts,
                settlement=settlement,
                home_district=home_district,
                work_district=work_district,
                favourites=_favourites_for(persona, settlement, categories, salt),
                cause=event.kind,
            )
        )

        salt += 1

    loyalty_low, loyalty_high = settings.loyalty_to_favourite

    trait = persona.trait("merchant_loyalty")

    loyalty = loyalty_low + (loyalty_high - loyalty_low) * trait

    new_place = params_module.active().geography.new_place_base * (1.4 - 0.8 * trait)

    return Habits(
        eras=tuple(eras),
        bills=_bills(persona, events),
        subscriptions=_subscriptions(persona, persona.settlement),
        favourite_categories=categories,
        loyalty=float(loyalty),
        new_place_chance=float(min(0.85, max(0.05, new_place))),
    )


def bill_amount(bill: Bill, ts: datetime, region: str, rng) -> int:

    amount = bill.base_amount * rng.lognormal(0.0, bill.sigma)

    if bill.kind == "utilities":
        amount *= cal.category_factor("utilities", ts, region)

    return int(max(300, round(amount / 10) * 10))


def subscription_amount(subscription: Subscription, ts: datetime) -> int:

    index = cal.month_index(ts)

    if subscription.change_month is not None and index >= subscription.change_month:
        return subscription.amount_after

    return subscription.amount


def subscription_active(subscription: Subscription, ts: datetime) -> bool:

    index = cal.month_index(ts)

    if index < subscription.start_month:
        return False

    return subscription.end_month is None or index < subscription.end_month


__all__ = [
    "Bill",
    "Era",
    "Favourite",
    "Habits",
    "Subscription",
    "bill_amount",
    "build_habits",
    "subscription_active",
    "subscription_amount",
]
