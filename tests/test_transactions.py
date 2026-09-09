"""
Поток транзакций: контракт, направления, зарплата, подписки.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import pytest

from src.generator.categories import (
    ALL_MCCS,
    CREDIT_MCCS,
    DIRECTION_CREDIT,
    DIRECTION_DEBIT,
    MCC_SALARY,
)
from src.generator.config import FEATURE_END, HISTORY_START
from src.generator.persona import draw_persona
from src.generator.transactions import (
    client_subscriptions,
    generate_transaction_history,
    salary_events,
)
from src.generator.world import FOREIGN_COUNTRIES, HOME_COUNTRY


CLIENTS = [0, 3, 7, 11]


@pytest.fixture(scope="module")
def histories():
    return {
        client_id: generate_transaction_history(client_id, HISTORY_START, FEATURE_END)
        for client_id in CLIENTS
    }


# ============================================================
# КОНТРАКТ
# ============================================================


def test_contract_fields(histories):
    for client_id, events in histories.items():

        assert events

        for event in events:
            assert event.client_id == client_id
            assert event.direction in (DIRECTION_DEBIT, DIRECTION_CREDIT)
            assert event.mcc in ALL_MCCS
            assert event.merchant_country == HOME_COUNTRY or event.merchant_country in FOREIGN_COUNTRIES
            assert isinstance(event.is_online, bool)
            assert isinstance(event.is_subscription, bool)
            assert event.amount > 0


def test_no_currency_field(histories):
    """
    Валюты в V1 нет сознательно.
    """

    event = histories[0][0]

    assert not hasattr(event, "currency")


def test_sorted_and_inside_window(histories):
    for events in histories.values():

        timestamps = [event.ts for event in events]

        assert timestamps == sorted(timestamps)
        assert all(HISTORY_START <= ts < FEATURE_END for ts in timestamps)


def test_reversed_interval_raises():
    with pytest.raises(ValueError):
        generate_transaction_history(0, FEATURE_END, HISTORY_START)


# ============================================================
# НАПРАВЛЕНИЯ
# ============================================================


def test_both_directions_present(histories):
    directions = Counter(
        event.direction for events in histories.values() for event in events
    )

    assert directions[DIRECTION_DEBIT] > directions[DIRECTION_CREDIT] > 0


def test_credit_transactions_use_service_mcc(histories):
    for events in histories.values():
        for event in events:
            if event.direction == DIRECTION_CREDIT:
                assert event.mcc in CREDIT_MCCS


def test_debit_transactions_never_use_service_mcc(histories):
    for events in histories.values():
        for event in events:
            if event.direction == DIRECTION_DEBIT:
                assert event.mcc not in CREDIT_MCCS


# ============================================================
# ЗАРПЛАТА
# ============================================================


def test_salary_lands_near_salary_day():
    client_id = next(
        c for c in range(200) if draw_persona(c).income_type == "employed"
    )

    persona = draw_persona(client_id)

    events = salary_events(client_id, HISTORY_START, FEATURE_END)

    assert len(events) >= 18, "зарплата должна приходить почти каждый месяц"

    for event in events:
        assert event.direction == DIRECTION_CREDIT
        assert event.mcc == MCC_SALARY
        assert event.merchant_city is None

        # День зарплаты или небольшая задержка.
        assert 0 <= (event.ts.day - persona.salary_day) % 31 <= 12


def test_no_salary_without_income():
    idle = [
        c
        for c in range(400)
        if draw_persona(c).income_type in ("student", "unemployed")
    ]

    assert idle

    for client_id in idle[:10]:
        assert salary_events(client_id, HISTORY_START, FEATURE_END) == []


# ============================================================
# ПОДПИСКИ
# ============================================================


def test_subscriptions_repeat_monthly():
    client_id = next(c for c in range(200) if client_subscriptions(c))

    subscription = client_subscriptions(client_id)[0]

    events = [
        event
        for event in generate_transaction_history(client_id, HISTORY_START, FEATURE_END)
        if event.is_subscription and event.mcc == subscription.mcc
    ]

    assert len(events) >= 12

    # Один и тот же день месяца и одна и та же сумма.
    assert {event.ts.day for event in events} == {subscription.day_of_month}
    assert {event.amount for event in events} == {subscription.amount}
    assert all(event.direction == DIRECTION_DEBIT for event in events)


def test_subscriptions_are_deterministic():
    assert client_subscriptions(5) == client_subscriptions(5)


# ============================================================
# ГЕОГРАФИЯ
# ============================================================


def test_foreign_transactions_have_no_city(histories):
    foreign = [
        event
        for events in histories.values()
        for event in events
        if event.merchant_country != HOME_COUNTRY
    ]

    assert foreign, "должны быть зарубежные операции"

    for event in foreign:
        assert event.merchant_city is None


def test_domestic_pos_mostly_in_home_region(histories):
    for client_id, events in histories.items():

        region = draw_persona(client_id).region

        pos = [
            event
            for event in events
            if not event.is_online
            and event.merchant_country == HOME_COUNTRY
            and event.merchant_city is not None
        ]

        assert pos

        home = sum(1 for event in pos if event.merchant_city == region)

        assert home / len(pos) > 0.60
