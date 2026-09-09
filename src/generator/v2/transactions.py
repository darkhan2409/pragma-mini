from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from ..categories import (
    CATEGORIES,
    DIRECTION_CREDIT,
    DIRECTION_DEBIT,
    GROUPS,
    MCC_SALARY,
)
from ..persona import Persona, draw_persona
from ..rng import (
    COMPONENT_CONTENT,
    COMPONENT_LINKED,
    COMPONENT_TIME,
    NS_TX,
    NS_V2_TX,
    KeyedRandom,
    day_rng,
    event_rng,
)
from ..transactions import (
    BASE_WEIGHTS,
    SALARY_INCOME_TYPES,
    TransactionEvent,
    daily_purchase_rate,
    draw_time,
    other_credit_events,
    persona_group_vector,
    season_group_vector,
    stress_group_vector,
)
from ..trajectory import behavior_state
from ..world import (
    FOREIGN_COUNTRIES,
    FOREIGN_COUNTRY_WEIGHTS,
    HOME_COUNTRY,
)
from .config import (
    BILL_GROUPS,
    LINKED_DELAY_SECONDS,
    MARKET_GROUP,
    QUIET_RATE_FACTOR,
    TRIP_FOREIGN_SHARE,
)
from .habits import (
    ClientHabits,
    client_habits,
    episode_at,
    episode_group_boost,
    month_index,
    salary_multiplier,
    subscription_amount,
    taste_at,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# В v1 у клиента не было ни своих магазинов, ни привычных сумм:
# MCC, город и сумма рисовались заново на каждое событие.
#
# В v2 у клиента есть привычные точки внутри своих категорий,
# и часть покупок повторяет привычку, а часть остаётся новой.
# Привычки медленно смещаются в точках перелома и временно
# отклоняются в эпизодах (поездка, всплеск, затишье).
#
# Квартплата, связь и госплатежи ушли из случайной смеси в
# расписание счетов. Слот покупки, выпавший на такую группу,
# ГАСИТСЯ: связанные платежи заменяют независимые, а не
# добавляются к ним.
# ============================================================


BILL_GROUP_SET = frozenset(BILL_GROUPS)


def choose_group_v2(
    persona: Persona,
    ts: datetime,
    taste: tuple[float, ...],
    episode_boost: dict[str, float],
    rng: KeyedRandom,
) -> str:
    """
    Категория покупки: те же множители, что в v1, плюс вкус
    клиента и временное отклонение эпизода.
    """

    state = behavior_state(persona.client_id, ts)

    weights = (
        BASE_WEIGHTS
        * persona_group_vector(persona.client_id)
        * season_group_vector(ts.toordinal(), persona.region)
        * stress_group_vector(state)
        * np.asarray(taste, dtype=float)
    )

    if episode_boost:
        weights = weights * np.array(
            [episode_boost.get(group, 1.0) for group in GROUPS], dtype=float
        )

    cumulative = np.cumsum(weights)

    index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))

    return GROUPS[min(index, len(GROUPS) - 1)]


def draw_habitual_amount(habit, rng: KeyedRandom) -> int:

    if habit.price_points and rng.random() < 0.6:
        return int(habit.price_points[int(rng.integers(0, len(habit.price_points)))])

    category = CATEGORIES[habit.group]

    value = rng.lognormal(
        math.log(category.typical_amount * habit.amount_shift), habit.log_sigma
    )

    return int(round(min(5_000_000.0, max(100.0, value)) / 10) * 10)


def draw_novel_amount(group: str, persona: Persona, ts: datetime, rng: KeyedRandom) -> int:

    category = CATEGORIES[group]
    state = behavior_state(persona.client_id, ts)

    raw = rng.lognormal(math.log(category.typical_amount), category.log_sigma)

    income_factor = min(1.40, max(0.80, (persona.declared_income / 350_000) ** 0.12))

    amount = raw * income_factor * state.spending_multiplier

    return int(round(min(5_000_000.0, max(100.0, amount)) / 10) * 10)


def generate_purchase_v2(
    client_id: int,
    day: datetime,
    index: int,
    habits: ClientHabits,
    persona: Persona,
) -> TransactionEvent | None:
    """
    Одна покупка. None значит, что слот погашен счётом:
    эта покупка в v1 была бы квартплатой или связью.
    """

    rng = event_rng(NS_V2_TX, client_id, day.toordinal(), index, COMPONENT_CONTENT)

    view = taste_at(habits, day.toordinal())
    episode = episode_at(habits, day.toordinal())

    group = choose_group_v2(
        persona, day, view.taste, episode_group_boost(episode), rng
    )

    if group in BILL_GROUP_SET:
        return None

    habit = view.habit(group, rng.random())

    use_habit = habit is not None and rng.random() < _stickiness(habits, group)

    if use_habit:
        mcc = habit.mcc
        is_online = habit.is_online
        city = habit.city
        country = HOME_COUNTRY
        amount = draw_habitual_amount(habit, rng)
    else:
        category = CATEGORIES[group]
        mcc = str(rng.choice(category.mccs, p=category.mcc_weights))
        online_share = min(
            1.0, category.online_share * (0.55 + 0.90 * persona.digital_affinity)
        )
        is_online = rng.random() < online_share
        city = habits.online_city if is_online else _offline_city(habits, rng)
        country = HOME_COUNTRY
        amount = draw_novel_amount(group, persona, day, rng)

    # Поездка: часть покупок уезжает за границу вместе с клиентом.
    foreign_share = CATEGORIES[group].foreign_share * (0.5 + 1.5 * persona.mobility)

    if episode is not None and episode.kind == "trip":
        foreign_share = TRIP_FOREIGN_SHARE

    if rng.random() < foreign_share:
        country = (
            episode.country
            if episode is not None and episode.country
            else str(rng.choice(FOREIGN_COUNTRIES, p=FOREIGN_COUNTRY_WEIGHTS))
        )
        city = None

    time_rng = event_rng(NS_TX, client_id, day.toordinal(), index, COMPONENT_TIME)

    hours = tuple(range(24)) if is_online else CATEGORIES[group].hours

    return TransactionEvent(
        client_id=client_id,
        ts=draw_time(day, hours, time_rng),
        amount=amount,
        direction=DIRECTION_DEBIT,
        mcc=mcc,
        merchant_city=city,
        merchant_country=country,
        is_online=bool(is_online),
        is_subscription=False,
    )


def _stickiness(habits: ClientHabits, group: str) -> float:

    # Липкость привязана к клиенту, а не к событию.
    base = 0.55 + 0.35 * ((habits.client_id * 2654435761) % 1000) / 1000.0

    return base


def _offline_city(habits: ClientHabits, rng: KeyedRandom) -> str:

    return habits.home_city if rng.random() < 0.85 else habits.tail_city


# ============================================================
# СЧЕТА
# ============================================================


def payment_events(
    client_id: int,
    payments,
    start: datetime,
    end: datetime,
    habits: ClientHabits,
) -> list[TransactionEvent]:
    """
    Ровно одна транзакция на каждую запись об оплате.

    Оплата в приложении привязана к своей операции по устойчивому
    ключу (день, сессия, шаг, попытка), а не по совпадению секунды:
    две операции могут прийтись на одну и ту же секунду.
    """

    events: list[TransactionEvent] = []

    for payment in payments:

        if payment.paid_via == "app":

            day_ordinal, index, step, _ = payment.op_key

            rng = event_rng(
                NS_V2_TX, client_id, day_ordinal, index * 100 + step, COMPONENT_LINKED
            )

            ts = payment.ts + timedelta(seconds=int(rng.integers(*LINKED_DELAY_SECONDS)))

        else:
            ts = payment.ts

        if not (start <= ts < end):
            continue

        events.append(
            TransactionEvent(
                client_id=client_id,
                ts=ts,
                amount=payment.amount,
                direction=DIRECTION_DEBIT,
                mcc=payment.mcc,
                merchant_city=habits.online_city,
                merchant_country=HOME_COUNTRY,
                is_online=True,
                is_subscription=False,
            )
        )

    return events


def linked_purchase_events(
    client_id: int,
    purchases,
    start: datetime,
    end: datetime,
    habits: ClientHabits,
    persona: Persona,
) -> list[TransactionEvent]:
    """
    Покупка по QR и заказ на маркете: операция и списание это
    одно и то же действие, разнесённое на десятки секунд.
    """

    events: list[TransactionEvent] = []

    for purchase in purchases:

        day_ordinal, index, step, attempt = purchase.op_key

        rng = event_rng(
            NS_V2_TX, client_id, day_ordinal, index * 100 + step, COMPONENT_LINKED
        )

        ts = purchase.ts + timedelta(seconds=int(rng.integers(*LINKED_DELAY_SECONDS)))

        if not (start <= ts < end):
            continue

        view = taste_at(habits, ts.toordinal())

        if purchase.operation == "market_order":
            group = MARKET_GROUP
            is_online = True
            city = habits.online_city
        else:
            group = _qr_group(habits, view, rng)
            is_online = False
            city = habits.home_city

        habit = view.habit(group, rng.random())

        if habit is not None:
            mcc = habit.mcc
            amount = draw_habitual_amount(habit, rng)
        else:
            category = CATEGORIES[group]
            mcc = str(rng.choice(category.mccs, p=category.mcc_weights))
            amount = draw_novel_amount(group, persona, ts, rng)

        events.append(
            TransactionEvent(
                client_id=client_id,
                ts=ts,
                amount=amount,
                direction=DIRECTION_DEBIT,
                mcc=mcc,
                merchant_city=city,
                merchant_country=HOME_COUNTRY,
                is_online=is_online,
                is_subscription=False,
            )
        )

    return events


def _qr_group(habits: ClientHabits, view, rng: KeyedRandom) -> str:
    """
    По QR платят там, где клиент бывает: в своей офлайн-точке.
    """

    offline = [
        habit.group
        for habit in view.older.habits
        if not habit.is_online and habit.group not in BILL_GROUP_SET
    ]

    if not offline:
        return "grocery"

    return offline[int(rng.integers(0, len(offline)))]


# ============================================================
# ПОДПИСКИ
# ============================================================


def subscription_events_v2(
    client_id: int,
    start: datetime,
    end: datetime,
    habits: ClientHabits,
) -> list[TransactionEvent]:
    """
    Подписка начинается, иногда заканчивается и изредка меняет
    сумму. Город у неё один: это одна и та же площадка.
    """

    events: list[TransactionEvent] = []

    for order, subscription in enumerate(habits.subscriptions):

        day = start.replace(hour=0, minute=0, second=0, microsecond=0)

        month = datetime(start.year, start.month, 1)

        while month < end:

            index = month_index(month)

            active = index >= subscription.start_month and (
                subscription.end_month is None or index < subscription.end_month
            )

            if active:

                try:
                    due = month.replace(day=subscription.day_of_month)
                except ValueError:
                    due = month

                if start <= due < end:

                    rng = event_rng(
                        NS_V2_TX,
                        client_id,
                        due.toordinal(),
                        900 + order,
                        COMPONENT_CONTENT,
                    )

                    ts = due.replace(
                        hour=rng.integers(2, 7),
                        minute=rng.integers(0, 60),
                        second=rng.integers(0, 60),
                        microsecond=0,
                    )

                    events.append(
                        TransactionEvent(
                            client_id=client_id,
                            ts=ts,
                            amount=subscription_amount(subscription, index),
                            direction=DIRECTION_DEBIT,
                            mcc=subscription.mcc,
                            merchant_city=subscription.city,
                            merchant_country=HOME_COUNTRY,
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
# ЗАРПЛАТА
# ============================================================


def salary_events_v2(
    client_id: int,
    start: datetime,
    end: datetime,
    habits: ClientHabits,
) -> list[TransactionEvent]:
    """
    Даты и порядок розыгрыша те же, что в v1: пропуск, задержка,
    время. Меняется только сумма: у клиента стабильный доход
    с редким повышением, а не новый розыгрыш каждый месяц.
    """

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

        stress = behavior_state(client_id, due).credit_stress

        if rng.random() < 0.35 * stress ** 2:
            month = _next_month(month)
            continue

        delay = 0

        if rng.random() < 0.15 + 0.55 * stress:
            delay = rng.integers(1, 2 + int(8 * stress))

        ts = (due + timedelta(days=delay)).replace(
            hour=rng.integers(6, 12),
            minute=rng.integers(0, 60),
            second=rng.integers(0, 60),
            microsecond=0,
        )

        if start <= ts < end:

            noise = 1.0 + 0.02 * (rng.random() * 2.0 - 1.0)

            amount = int(
                round(
                    persona.declared_income * salary_multiplier(habits, ts) * noise / 100
                )
                * 100
            )

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


# ============================================================
# ПОЛНАЯ ИСТОРИЯ
# ============================================================


def generate_transaction_history_v2(
    client_id: int,
    start: datetime,
    end: datetime,
    lifecycle=None,
) -> list[TransactionEvent]:

    if end <= start:
        raise ValueError("end must be after start")

    persona = draw_persona(client_id)
    habits = client_habits(client_id)

    blocks = tuple(lifecycle.card_blocks) if lifecycle is not None else ()

    events: list[TransactionEvent] = []

    day = start.replace(hour=0, minute=0, second=0, microsecond=0)

    while day < end:

        rate = daily_purchase_rate(persona, day)

        episode = episode_at(habits, day.toordinal())

        if episode is not None and episode.kind == "quiet":
            rate *= QUIET_RATE_FACTOR

        count = day_rng(NS_TX, client_id, day.toordinal()).poisson(rate)

        for index in range(count):

            event = generate_purchase_v2(client_id, day, index, habits, persona)

            if event is None:
                continue

            if not (start <= event.ts < end):
                continue

            # Заблокированной картой в магазине не расплатишься.
            if not event.is_online and _blocked(blocks, event.ts):
                continue

            events.append(event)

        day += timedelta(days=1)

    events.extend(subscription_events_v2(client_id, start, end, habits))
    events.extend(salary_events_v2(client_id, start, end, habits))
    events.extend(other_credit_events(client_id, start, end))

    if lifecycle is not None:
        events.extend(
            payment_events(client_id, lifecycle.payments, start, end, habits)
        )
        events.extend(
            linked_purchase_events(
                client_id, lifecycle.linked_purchases, start, end, habits, persona
            )
        )

    events.sort(key=lambda event: (event.ts, event.mcc, event.amount))

    return events


def _blocked(blocks, ts: datetime) -> bool:

    for started, ended in blocks:
        if started <= ts and (ended is None or ts < ended):
            return True

    return False


__all__ = [
    "payment_events",
    "generate_purchase_v2",
    "generate_transaction_history_v2",
    "linked_purchase_events",
    "salary_events_v2",
    "subscription_events_v2",
]
