from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from ..categories import CATEGORIES, GROUP_BASE_WEIGHT, GROUPS
from ..config import HISTORY_START, LABEL_END, SEED
from ..persona import Persona, draw_persona
from ..rng import NS_V2_HABITS
from ..transactions import (
    SUBSCRIPTION_GROUPS,
    SUBSCRIPTION_WEIGHTS,
    persona_group_vector,
)
from ..world import CITY_TAIL, FOREIGN_COUNTRIES, FOREIGN_COUNTRY_WEIGHTS, HOME_COUNTRY
from .config import (
    BILL_BANK_SHARE,
    BILL_KINDS,
    BURST_GROUP_BOOST,
    DRIFT_BLEND_DAYS,
    DRIFT_POINTS,
    DRIFT_REDRAW_HABITS,
    EPISODE_DAYS,
    EPISODE_KINDS,
    EPISODE_WEIGHTS,
    EPISODES,
    FINE_RATE_PER_MONTH,
    HABIT_GROUPS_COUNT,
    HABIT_LOG_SIGMA_FACTOR,
    HABIT_STICKINESS,
    INTERNET_SHARE,
    PLAN_CHANGE_SHARE,
    PRICE_POINT_COUNT,
    PRICE_POINT_GROUPS,
    SALARY_FACTOR,
    SALARY_RAISE_FACTOR,
    SALARY_RAISE_SHARE,
    SECOND_UTILITY_SHARE,
    SERVICE_RATE_PER_MONTH,
    SUBSCRIPTION_LATE_START_SHARE,
    SUBSCRIPTION_PRICE_CHANGE,
    SUBSCRIPTION_PRICE_CHANGE_SHARE,
    SUBSCRIPTION_STOP_SHARE,
    TASTE_SIGMA,
    TAX_PER_YEAR,
    TRIP_GROUP_BOOST,
    BILL_GROUPS,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Привычка это то, чего в v1 не было совсем: у клиента нет
# «своих» магазинов, сумм и дней оплаты, всё рисуется заново
# на каждое событие.
#
# Здесь описан скрытый портрет клиента: вкус по категориям,
# привычная точка и сумма внутри категории, расписание счетов,
# жизненный цикл подписок, зарплата, точки перелома привычек
# и временные эпизоды.
#
# Все даты берутся от КОНСТАНТ горизонта, а не от аргументов
# вызова: иначе продление горизонта сдвинуло бы прошлое.
#
# Ничего из этого модуля в RAW не попадает.
# ============================================================


SLOT_APP = 1
SLOT_TASTE = 2
SLOT_DRIFT = 3
SLOT_EPISODE = 4
SLOT_SUBSCRIPTION = 5
SLOT_SALARY = 6
SLOT_BILL = 7


HORIZON_START = HISTORY_START.toordinal()
HORIZON_END = LABEL_END.toordinal()

TOTAL_MONTHS = (
    (LABEL_END.year - HISTORY_START.year) * 12 + LABEL_END.month - HISTORY_START.month
)


def month_index(ts: datetime) -> int:
    return (ts.year - HISTORY_START.year) * 12 + ts.month - HISTORY_START.month


def month_start(index: int) -> datetime:

    total = HISTORY_START.month - 1 + index

    return datetime(HISTORY_START.year + total // 12, total % 12 + 1, 1)


def slot_rng(client_id: int, slot: int) -> np.random.Generator:
    """
    Отдельный поток на каждый блок скрытых признаков: правка
    одного блока не пересеивает остальные.
    """

    return np.random.default_rng([SEED, NS_V2_HABITS, client_id, slot])


# ============================================================
# СТРУКТУРЫ
# ============================================================


@dataclass(frozen=True)
class Habit:
    """
    Привычная точка внутри категории: тот же MCC, тот же город,
    тот же канал и сумма из узкого диапазона.
    """

    group: str
    mcc: str
    city: str | None
    is_online: bool
    amount_shift: float
    log_sigma: float
    price_points: tuple[int, ...]


@dataclass(frozen=True)
class TasteState:
    taste: tuple[float, ...]
    habits: tuple[Habit, ...]

    def habit(self, group: str) -> Habit | None:
        for item in self.habits:
            if item.group == group:
                return item
        return None


@dataclass(frozen=True)
class DriftPoint:
    day: int
    blend_days: int


@dataclass(frozen=True)
class Episode:
    kind: str
    start_day: int
    end_day: int
    country: str | None


@dataclass(frozen=True)
class SubscriptionV2:
    group: str
    mcc: str
    city: str | None
    day_of_month: int
    start_month: int
    end_month: int | None
    change_month: int | None
    amount: int
    amount_after: int


@dataclass(frozen=True)
class Bill:
    kind: str
    mcc: str
    group: str
    day_of_month: int
    amount: int
    log_sigma: float
    months: tuple[int, ...]


@dataclass(frozen=True)
class AppHabits:
    scenario_bias: tuple[float, ...]
    abandon: float
    depth_factor: float
    retry: float
    biometry_share: float
    favourite_transfer: str
    favourite_payment: str
    support_bias: float
    in_app_bills_share: float
    explore_bias: float


@dataclass(frozen=True)
class ClientHabits:
    client_id: int
    app: AppHabits
    tastes: tuple[TasteState, ...]
    drift_points: tuple[DriftPoint, ...]
    episodes: tuple[Episode, ...]
    subscriptions: tuple[SubscriptionV2, ...]
    bills: tuple[Bill, ...]
    home_city: str
    tail_city: str
    online_city: str
    salary_factor: float
    salary_raise_day: int | None
    salary_raise_factor: float
    foreign_country: str


@dataclass(frozen=True)
class TasteView:
    """
    Вкус на конкретный день: между точками перелома старый
    и новый портреты смешиваются постепенно.
    """

    taste: tuple[float, ...]
    older: TasteState
    newer: TasteState
    mix: float

    def habit(self, group: str, u: float) -> Habit | None:
        state = self.newer if u < self.mix else self.older
        return state.habit(group)


# ============================================================
# СЦЕНАРНЫЕ СКЛОННОСТИ ПРИЛОЖЕНИЯ
# ============================================================

from .scenarios import SCENARIOS  # noqa: E402  (циклов нет: scenarios не знает habits)


TRANSFER_OPERATIONS = ("transfer_phone", "transfer_card", "transfer_own")
PAYMENT_OPERATIONS = ("pay_utility", "pay_mobile", "pay_qr")


def draw_app_habits(persona: Persona) -> AppHabits:

    rng = slot_rng(persona.client_id, SLOT_APP)

    bias = tuple(float(value) for value in rng.lognormal(0.0, 0.45, size=len(SCENARIOS)))

    abandon = float(rng.uniform(0.10, 0.35) * (1.4 - 0.6 * persona.digital_affinity))
    depth_factor = float(rng.uniform(0.7, 1.5))
    retry = float(rng.uniform(0.35, 0.85))
    biometry = float(rng.uniform(0.15, 0.95) * (0.5 + 0.7 * persona.digital_affinity))

    favourite_transfer = TRANSFER_OPERATIONS[int(rng.integers(0, len(TRANSFER_OPERATIONS)))]
    favourite_payment = PAYMENT_OPERATIONS[int(rng.integers(0, len(PAYMENT_OPERATIONS)))]

    support_bias = float(rng.uniform(0.5, 1.8))

    in_app_bills = float(
        min(0.95, max(0.0, rng.uniform(0.15, 0.95) * (0.4 + 0.9 * persona.digital_affinity)))
    )

    explore_bias = float(rng.uniform(0.5, 1.7))

    return AppHabits(
        scenario_bias=bias,
        abandon=min(0.6, abandon),
        depth_factor=depth_factor,
        retry=retry,
        biometry_share=min(0.95, biometry),
        favourite_transfer=favourite_transfer,
        favourite_payment=favourite_payment,
        support_bias=support_bias,
        in_app_bills_share=in_app_bills,
        explore_bias=explore_bias,
    )


# ============================================================
# ВКУС И ПРИВЫЧНЫЕ ТОЧКИ
# ============================================================


def draw_taste(rng: np.random.Generator) -> tuple[float, ...]:
    """
    Индивидуальный вкус по категориям.

    Группы-счета исключены: их доля должна остаться такой же,
    как в v1, потому что именно она гасит слоты покупок.
    """

    values = rng.lognormal(0.0, TASTE_SIGMA, size=len(GROUPS))

    return tuple(
        1.0 if group in BILL_GROUPS else float(value)
        for group, value in zip(GROUPS, values)
    )


def habitual_groups(
    persona: Persona,
    taste: tuple[float, ...],
    count: int,
) -> tuple[str, ...]:
    """
    Категории, в которых у клиента есть «своё место».
    """

    persona_vector = persona_group_vector(persona.client_id)

    weights = [
        0.0
        if group in BILL_GROUPS
        else GROUP_BASE_WEIGHT[group] * persona_vector[index] * taste[index]
        for index, group in enumerate(GROUPS)
    ]

    order = sorted(range(len(GROUPS)), key=lambda index: -weights[index])

    return tuple(GROUPS[index] for index in order[:count])


def draw_habit(
    group: str,
    persona: Persona,
    home_city: str,
    tail_city: str,
    online_city: str,
    rng: np.random.Generator,
) -> Habit:

    category = CATEGORIES[group]

    mcc = str(rng.choice(category.mccs, p=np.array(category.mcc_weights) / sum(category.mcc_weights)))

    is_online = bool(rng.random() < category.online_share)

    if is_online:
        city = online_city
    else:
        city = home_city if rng.random() < 0.80 else tail_city

    amount_shift = float(rng.lognormal(0.0, 0.35))

    price_points: tuple[int, ...] = ()

    if group in PRICE_POINT_GROUPS:

        count = int(rng.integers(PRICE_POINT_COUNT[0], PRICE_POINT_COUNT[1] + 1))

        base = category.typical_amount * amount_shift

        price_points = tuple(
            sorted(
                int(round(base * float(rng.uniform(0.7, 1.4)) / 50) * 50)
                for _ in range(count)
            )
        )

    return Habit(
        group=group,
        mcc=mcc,
        city=city,
        is_online=is_online,
        amount_shift=amount_shift,
        log_sigma=category.log_sigma * HABIT_LOG_SIGMA_FACTOR,
        price_points=price_points,
    )


def draw_taste_states(
    persona: Persona,
    home_city: str,
    tail_city: str,
    online_city: str,
    phases: int,
) -> tuple[TasteState, ...]:

    rng = slot_rng(persona.client_id, SLOT_TASTE)

    taste = draw_taste(rng)

    count = int(rng.integers(HABIT_GROUPS_COUNT[0], HABIT_GROUPS_COUNT[1] + 1))

    groups = habitual_groups(persona, taste, count)

    habits = tuple(
        draw_habit(group, persona, home_city, tail_city, online_city, rng)
        for group in groups
    )

    states = [TasteState(taste=taste, habits=habits)]

    for _ in range(phases - 1):

        # Вкус смещается, а не перерисовывается заново.
        shifted = tuple(
            1.0 if group in BILL_GROUPS else value * float(rng.lognormal(0.0, 0.35))
            for group, value in zip(GROUPS, states[-1].taste)
        )

        redraw = int(rng.integers(DRIFT_REDRAW_HABITS[0], DRIFT_REDRAW_HABITS[1] + 1))

        updated = list(states[-1].habits)

        for _ in range(min(redraw, len(updated))):
            index = int(rng.integers(0, len(updated)))
            updated[index] = draw_habit(
                updated[index].group, persona, home_city, tail_city, online_city, rng
            )

        states.append(TasteState(taste=shifted, habits=tuple(updated)))

    return tuple(states)


def draw_drift_points(client_id: int) -> tuple[DriftPoint, ...]:

    rng = slot_rng(client_id, SLOT_DRIFT)

    count = int(rng.integers(DRIFT_POINTS[0], DRIFT_POINTS[1] + 1))

    days = sorted(
        int(rng.integers(HORIZON_START + 60, HORIZON_END - 30)) for _ in range(count)
    )

    return tuple(
        DriftPoint(
            day=day,
            blend_days=int(rng.integers(DRIFT_BLEND_DAYS[0], DRIFT_BLEND_DAYS[1] + 1)),
        )
        for day in days
    )


def taste_at(habits: ClientHabits, day: int) -> TasteView:
    """
    Портрет клиента на день с учётом постепенного перелома.
    """

    phase = 0
    mix = 0.0

    for index, point in enumerate(habits.drift_points):

        if day >= point.day + point.blend_days:
            phase = index + 1
            mix = 0.0
        elif day >= point.day:
            phase = index
            mix = (day - point.day) / point.blend_days
            break
        else:
            break

    older = habits.tastes[phase]
    newer = habits.tastes[min(phase + 1, len(habits.tastes) - 1)]

    blended = tuple(
        old * (1.0 - mix) + new * mix
        for old, new in zip(older.taste, newer.taste)
    )

    return TasteView(taste=blended, older=older, newer=newer, mix=mix)


# ============================================================
# ЭПИЗОДЫ
# ============================================================


def draw_episodes(client_id: int) -> tuple[Episode, ...]:

    rng = slot_rng(client_id, SLOT_EPISODE)

    count = int(rng.integers(EPISODES[0], EPISODES[1] + 1))

    episodes: list[Episode] = []

    for _ in range(count):

        start = int(rng.integers(HORIZON_START + 30, HORIZON_END - 30))
        length = int(rng.integers(EPISODE_DAYS[0], EPISODE_DAYS[1] + 1))

        kind = str(rng.choice(EPISODE_KINDS, p=EPISODE_WEIGHTS))

        country = (
            str(rng.choice(FOREIGN_COUNTRIES, p=FOREIGN_COUNTRY_WEIGHTS))
            if kind == "trip"
            else None
        )

        episodes.append(
            Episode(kind=kind, start_day=start, end_day=start + length, country=country)
        )

    return tuple(sorted(episodes, key=lambda item: item.start_day))


def episode_at(habits: ClientHabits, day: int) -> Episode | None:

    for episode in habits.episodes:
        if episode.start_day <= day < episode.end_day:
            return episode

    return None


def episode_group_boost(episode: Episode | None) -> dict[str, float]:

    if episode is None:
        return {}

    if episode.kind == "trip":
        return TRIP_GROUP_BOOST

    if episode.kind == "burst":
        return BURST_GROUP_BOOST

    return {}


# ============================================================
# ПОДПИСКИ
# ============================================================


def draw_subscriptions(persona: Persona) -> tuple[SubscriptionV2, ...]:

    rng = slot_rng(persona.client_id, SLOT_SUBSCRIPTION)

    count = min(5, int(rng.poisson(0.6 + 2.2 * persona.digital_affinity)))

    weights = np.array(SUBSCRIPTION_WEIGHTS, dtype=float)
    weights = weights / weights.sum()

    used: list[str] = []
    result: list[SubscriptionV2] = []

    for _ in range(count):

        group = str(rng.choice(SUBSCRIPTION_GROUPS, p=weights))

        if group in used:
            continue

        used.append(group)

        category = CATEGORIES[group]

        amount = int(
            round(
                float(rng.lognormal(math.log(category.typical_amount), 0.35))
                * min(1.4, max(0.8, (persona.declared_income / 350_000) ** 0.15))
                / 10
            )
            * 10
        )
        amount = max(500, amount)

        start_month = 0

        if rng.random() < SUBSCRIPTION_LATE_START_SHARE:
            start_month = int(rng.integers(1, max(2, TOTAL_MONTHS - 3)))

        end_month = None

        if rng.random() < SUBSCRIPTION_STOP_SHARE:
            end_month = int(rng.integers(start_month + 2, TOTAL_MONTHS + 1))

        change_month = None
        amount_after = amount

        if rng.random() < SUBSCRIPTION_PRICE_CHANGE_SHARE:

            last = end_month if end_month is not None else TOTAL_MONTHS

            if last - start_month >= 4:
                change_month = int(rng.integers(start_month + 2, last))
                step = float(
                    rng.uniform(SUBSCRIPTION_PRICE_CHANGE[0], SUBSCRIPTION_PRICE_CHANGE[1])
                )
                direction = 1.0 if rng.random() < 0.75 else -1.0
                amount_after = max(
                    500, int(round(amount * (1.0 + direction * step) / 10) * 10)
                )

        mcc = str(
            rng.choice(
                category.mccs, p=np.array(category.mcc_weights) / sum(category.mcc_weights)
            )
        )

        result.append(
            SubscriptionV2(
                group=group,
                mcc=mcc,
                city="Almaty" if rng.random() < 0.65 else "Astana",
                day_of_month=int(rng.integers(1, 29)),
                start_month=start_month,
                end_month=end_month,
                change_month=change_month,
                amount=amount,
                amount_after=amount_after,
            )
        )

    return tuple(result)


def subscription_amount(subscription: SubscriptionV2, index: int) -> int:

    if subscription.change_month is not None and index >= subscription.change_month:
        return subscription.amount_after

    return subscription.amount


# ============================================================
# СЧЕТА
# ============================================================


def draw_bills(persona: Persona) -> tuple[Bill, ...]:
    """
    Расписание счетов клиента: у каждого свой день, своя сумма
    и свой срок жизни. Это замена случайным платежам v1.
    """

    rng = slot_rng(persona.client_id, SLOT_BILL)

    bills: list[Bill] = []

    every_month = tuple(range(TOTAL_MONTHS + 1))

    def regular(kind: str, day: int, scale: float, months: tuple[int, ...]) -> Bill:
        mcc, base, sigma = BILL_KINDS[kind]
        return Bill(
            kind=kind,
            mcc=mcc,
            group="utilities" if mcc == "4900" else ("telecom" if mcc in ("4814", "4899") else "government"),
            day_of_month=day,
            amount=max(300, int(round(base * scale / 10) * 10)),
            log_sigma=sigma,
            months=months,
        )

    income_scale = min(1.6, max(0.7, (persona.declared_income / 350_000) ** 0.25))

    # Через этот банк проходит не всё: у малоактивного клиента
    # часть счетов оплачивается мимо и в данных не видна.
    bank = min(1.0, BILL_BANK_SHARE[0] + BILL_BANK_SHARE[1] * persona.activity)

    # Квартплата.
    if rng.random() < bank:
        bills.append(
            regular("utility", int(rng.integers(5, 26)), income_scale * float(rng.uniform(0.7, 1.4)), every_month)
        )

    if rng.random() < SECOND_UTILITY_SHARE * bank:
        bills.append(
            regular("utility", int(rng.integers(5, 26)), income_scale * float(rng.uniform(0.3, 0.8)), every_month)
        )

    # Связь: у части клиентов тариф меняется на середине горизонта.
    mobile_day = int(rng.integers(1, 29))
    mobile_scale = float(rng.uniform(0.6, 1.6))

    if rng.random() >= bank:
        pass
    elif rng.random() < PLAN_CHANGE_SHARE:
        switch = int(rng.integers(4, TOTAL_MONTHS - 2))
        bills.append(regular("mobile", mobile_day, mobile_scale, tuple(range(switch))))
        bills.append(
            regular(
                "mobile",
                mobile_day,
                mobile_scale * float(rng.uniform(1.05, 1.6)),
                tuple(range(switch, TOTAL_MONTHS + 1)),
            )
        )
    else:
        bills.append(regular("mobile", mobile_day, mobile_scale, every_month))

    if rng.random() < INTERNET_SHARE * bank:
        bills.append(
            regular("internet", int(rng.integers(1, 29)), float(rng.uniform(0.7, 1.4)), every_month)
        )

    # Штрафы и разовые услуги.
    for kind, rate in (("fine", FINE_RATE_PER_MONTH), ("service", SERVICE_RATE_PER_MONTH)):

        occurrences = int(rng.poisson(rate * bank * (TOTAL_MONTHS + 1)))

        for _ in range(occurrences):
            bills.append(
                regular(
                    kind,
                    int(rng.integers(1, 29)),
                    float(rng.uniform(0.4, 2.2)),
                    (int(rng.integers(0, TOTAL_MONTHS + 1)),),
                )
            )

    # Налог: в свои месяцы года.
    per_year = int(rng.integers(TAX_PER_YEAR[0], TAX_PER_YEAR[1] + 1)) if rng.random() < bank else 0

    tax_months = sorted({int(rng.integers(0, 12)) for _ in range(per_year)})

    tax_day = int(rng.integers(1, 29))

    occurrences = tuple(
        index
        for index in range(TOTAL_MONTHS + 1)
        if month_start(index).month - 1 in tax_months
    )

    if occurrences:
        bills.append(regular("tax", tax_day, float(rng.uniform(0.5, 1.8)), occurrences))

    return tuple(bills)


@dataclass(frozen=True)
class BillDue:
    bill: Bill
    month: int
    due: datetime


def bills_due(habits: ClientHabits) -> tuple[BillDue, ...]:
    """
    Все сроки оплаты на горизонте, по возрастанию даты.
    """

    items: list[BillDue] = []

    for bill in habits.bills:

        for index in bill.months:

            start = month_start(index)

            try:
                due = start.replace(day=bill.day_of_month)
            except ValueError:
                due = start

            items.append(BillDue(bill=bill, month=index, due=due))

    return tuple(sorted(items, key=lambda item: (item.due, item.bill.mcc, item.bill.amount)))


def bill_amount(due: BillDue, rng) -> int:
    """
    Сумма счёта: привычная база с небольшим разбросом.
    """

    value = rng.lognormal(math.log(due.bill.amount), due.bill.log_sigma)

    return int(round(min(3_000_000.0, max(300.0, value)) / 10) * 10)


# ============================================================
# ЗАРПЛАТА
# ============================================================


def draw_salary(client_id: int) -> tuple[float, int | None, float]:

    rng = slot_rng(client_id, SLOT_SALARY)

    factor = float(rng.uniform(SALARY_FACTOR[0], SALARY_FACTOR[1]))

    raise_day: int | None = None
    raise_factor = 1.0

    if rng.random() < SALARY_RAISE_SHARE:
        raise_day = int(rng.integers(HORIZON_START + 120, HORIZON_END - 60))
        raise_factor = float(rng.uniform(SALARY_RAISE_FACTOR[0], SALARY_RAISE_FACTOR[1]))

    return factor, raise_day, raise_factor


# ============================================================
# ПОРТРЕТ ЦЕЛИКОМ
# ============================================================


@lru_cache(maxsize=131_072)
def client_habits(client_id: int) -> ClientHabits:

    persona = draw_persona(client_id)

    drift_points = draw_drift_points(client_id)

    geo = slot_rng(client_id, SLOT_TASTE + 100)

    home_city = persona.region
    tail = CITY_TAIL[persona.region]
    tail_city = str(tail[int(geo.integers(0, len(tail)))])
    online_city = "Almaty" if geo.random() < 0.65 else "Astana"
    foreign_country = str(geo.choice(FOREIGN_COUNTRIES, p=FOREIGN_COUNTRY_WEIGHTS))

    tastes = draw_taste_states(
        persona, home_city, tail_city, online_city, len(drift_points) + 1
    )

    salary_factor, raise_day, raise_factor = draw_salary(client_id)

    return ClientHabits(
        client_id=client_id,
        app=draw_app_habits(persona),
        tastes=tastes,
        drift_points=drift_points,
        episodes=draw_episodes(client_id),
        subscriptions=draw_subscriptions(persona),
        bills=draw_bills(persona),
        home_city=home_city,
        tail_city=tail_city,
        online_city=online_city,
        salary_factor=salary_factor,
        salary_raise_day=raise_day,
        salary_raise_factor=raise_factor,
        foreign_country=foreign_country,
    )


def salary_multiplier(habits: ClientHabits, ts: datetime) -> float:

    if habits.salary_raise_day is not None and ts.toordinal() >= habits.salary_raise_day:
        return habits.salary_factor * habits.salary_raise_factor

    return habits.salary_factor


__all__ = [
    "AppHabits",
    "Bill",
    "BillDue",
    "ClientHabits",
    "DriftPoint",
    "Episode",
    "Habit",
    "SubscriptionV2",
    "TasteState",
    "TasteView",
    "bill_amount",
    "bills_due",
    "client_habits",
    "episode_at",
    "episode_group_boost",
    "month_index",
    "month_start",
    "salary_multiplier",
    "subscription_amount",
    "taste_at",
]
