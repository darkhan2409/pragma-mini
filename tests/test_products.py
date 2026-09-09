"""
Реестр договоров: контракт, сроки, качество времени, владение.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pytest

from src.generator.config import FEATURE_END, HISTORY_START, SOURCE_AVAILABILITY
from src.generator.products import (
    PRODUCT_SUBTYPES,
    PRODUCT_TYPES,
    QUALITY_DATE_ONLY,
    QUALITY_EXACT,
    REGISTRY_START,
    TERM_CHOICES,
    TIMESTAMP_QUALITIES,
    ProductState,
    add_months,
    open_contract,
    prehistory_contracts,
)


POPULATION = 400


@pytest.fixture(scope="module")
def states():
    return {client_id: ProductState(client_id) for client_id in range(POPULATION)}


# ============================================================
# КОНТРАКТ
# ============================================================


def test_event_contract(states):
    for state in states.values():
        for event in state.visible_events():

            assert event.product_type in PRODUCT_TYPES
            assert event.product_subtype in PRODUCT_SUBTYPES[event.product_type]
            assert event.timestamp_quality in TIMESTAMP_QUALITIES

            if event.product_type == "debit_card":
                assert event.amount_or_limit is None
            else:
                assert event.amount_or_limit is not None
                assert event.amount_or_limit > 0

            if event.product_type in TERM_CHOICES:
                assert event.term in TERM_CHOICES[event.product_type]
            else:
                assert event.term is None


def test_no_action_field(states):
    event = states[0].visible_events()[0]

    assert not hasattr(event, "action")


def test_every_client_starts_with_a_debit_card(states):
    for client_id, state in states.items():
        assert "debit_card" in {h.product_type for h in state.holdings}


# ============================================================
# КАЧЕСТВО TIMESTAMP
# ============================================================


def test_date_only_events_are_at_midnight(states):
    seen = Counter()

    for state in states.values():
        for event in state.visible_events():

            seen[event.timestamp_quality] += 1

            if event.timestamp_quality == QUALITY_DATE_ONLY:
                assert (event.ts.hour, event.ts.minute, event.ts.second) == (0, 0, 0)

    assert seen[QUALITY_DATE_ONLY] > 0
    assert seen[QUALITY_EXACT] > 0


def test_cards_have_exact_time_loans_mostly_not(states):
    by_type: dict[str, Counter] = {}

    for state in states.values():
        for event in state.visible_events():
            by_type.setdefault(event.product_type, Counter())[event.timestamp_quality] += 1

    for card in ("debit_card", "credit_card"):
        assert by_type[card][QUALITY_DATE_ONLY] == 0

    # Страховка приходит только датой.
    assert by_type["insurance"][QUALITY_EXACT] == 0

    # У депозитов и кредитов время появляется редко: часть
    # договоров подтягивается из второй системы.
    for slow in ("deposit", "cash_loan"):
        exact = by_type[slow][QUALITY_EXACT]
        total = sum(by_type[slow].values())
        assert exact / total < 0.20, (slow, exact, total)


# ============================================================
# РЕЕСТР СТАРШЕ ОКНА
# ============================================================


def test_registry_reaches_before_history(states):
    early = sum(
        1
        for state in states.values()
        for event in state.visible_events()
        if event.ts < HISTORY_START
    )

    assert early > 0


def test_registry_truncated_at_migration(states):
    for state in states.values():
        for event in state.visible_events():
            assert event.ts >= REGISTRY_START

    assert REGISTRY_START == SOURCE_AVAILABILITY["product_events"]


def test_hidden_contracts_still_count_in_ownership():
    """
    Договор старше миграции реестра не виден как событие,
    но владение по нему учитывается.
    """

    hidden = [
        client_id
        for client_id in range(POPULATION)
        if len(ProductState(client_id).holdings)
        > len(ProductState(client_id).visible_events())
    ]

    assert hidden


# ============================================================
# ВЛАДЕНИЕ
# ============================================================


def test_ownership_follows_term(states):
    for state in states.values():
        for holding in state.holdings:

            if holding.term is None:
                assert holding.closed_at is None
                assert holding.is_open_at(FEATURE_END)
                continue

            assert holding.closed_at == add_months(holding.opened_at, holding.term)
            assert holding.is_open_at(holding.opened_at)
            assert not holding.is_open_at(holding.closed_at)


def test_blocked_prevents_duplicate_contract(states):
    for state in states.values():
        for holding in state.holdings:

            if holding.closed_at is None:
                assert state.blocked(holding.product_type, FEATURE_END)


def test_reopen_allowed_after_term():
    client_id = next(
        c
        for c in range(POPULATION)
        for h in ProductState(c).holdings
        if h.product_type == "cash_loan" and h.closed_at is not None
    )

    state = ProductState(client_id)

    holding = next(
        h for h in state.holdings if h.product_type == "cash_loan" and h.closed_at
    )

    assert state.blocked("cash_loan", holding.closed_at - timedelta(days=1))
    assert not state.blocked("cash_loan", holding.closed_at + timedelta(days=1))


def test_counters_are_monotonic(states):
    for state in states.values():

        before = state.contracts_count(HISTORY_START)
        after = state.contracts_count(FEATURE_END)

        assert after >= before
        assert state.active_contracts(FEATURE_END) <= after


def test_credit_limit_only_with_credit_card(states):
    for state in states.values():

        limit = state.credit_limit(FEATURE_END)

        if "credit_card" in state.owned_at(FEATURE_END):
            assert limit is not None and limit > 0
        else:
            assert limit is None


# ============================================================
# ДЕТЕРМИНИЗМ
# ============================================================


def test_prehistory_is_deterministic():
    assert [e for e, _ in prehistory_contracts(17)] == [
        e for e, _ in prehistory_contracts(17)
    ]


def test_open_contract_is_deterministic():
    ts = datetime(2025, 4, 10, 15, 20)

    first, _ = open_contract(5, "deposit", ts)
    second, _ = open_contract(5, "deposit", ts)

    assert first == second


def test_add_months():
    assert add_months(datetime(2024, 1, 31), 1) == datetime(2024, 2, 28)
    assert add_months(datetime(2024, 12, 15), 1) == datetime(2025, 1, 15)
    assert add_months(datetime(2024, 6, 10), 24) == datetime(2026, 6, 10)
