"""
Профиль: помесячные снимки, as-of cutoff, три вида пропусков.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from datetime import timedelta

import pandas as pd
import pytest

from src.generator.config import (
    FEATURE_END,
    HISTORY_START,
    LABEL_END,
    PROFILE_FIELDS,
)
from src.generator.chains import derive_lifecycle
from src.generator.history import generate_client_history
from src.generator.persona import draw_persona
from src.generator.products import ProductState
from src.generator.profile import (
    ALWAYS_PRESENT,
    FIELD_GROUPS,
    month_end,
    profile_as_of,
    profile_snapshots,
    snapshot_row,
)
from src.generator.world import INDUSTRY_INCOME_TYPES


POPULATION = 150


@pytest.fixture(scope="module")
def snapshots():
    return {
        client_id: profile_snapshots(client_id, HISTORY_START, FEATURE_END)
        for client_id in range(POPULATION)
    }


# ============================================================
# СНИМКИ
# ============================================================


def test_one_snapshot_per_month(snapshots):
    for client_id, client_snapshots in snapshots.items():

        assert len(client_snapshots) == 24, client_id

        months = [snapshot.snapshot_month for snapshot in client_snapshots]

        assert months == sorted(months)
        assert len(set(months)) == len(months)


def test_snapshot_is_computed_at_month_end(snapshots):
    for client_snapshots in snapshots.values():
        for snapshot in client_snapshots:
            assert snapshot.ts == month_end(snapshot.snapshot_month)
            assert snapshot.ts.day >= 28


def test_all_twenty_fields_present(snapshots):
    for snapshot in snapshots[0]:
        assert set(snapshot.values) == set(PROFILE_FIELDS)
        assert len(PROFILE_FIELDS) == 20


def test_profile_as_of_takes_last_snapshot_before_cutoff(snapshots):
    client_snapshots = snapshots[0]

    asof = profile_as_of(client_snapshots, FEATURE_END)

    assert asof is not None
    assert asof.ts < FEATURE_END
    assert asof is client_snapshots[-1]

    early = profile_as_of(client_snapshots, HISTORY_START)

    assert early is None


def test_profile_never_looks_ahead(snapshots):
    for client_snapshots in snapshots.values():
        for snapshot in client_snapshots:
            assert snapshot.ts < FEATURE_END


# ============================================================
# ЗНАЧЕНИЯ
# ============================================================


def test_age_and_relationship_grow(snapshots):
    for client_id, client_snapshots in snapshots.items():

        ages = [s.values["age"] for s in client_snapshots]
        months = [s.values["relationship_months"] for s in client_snapshots]

        present_ages = [value for value in ages if value is not None]

        assert present_ages == sorted(present_ages)

        # Стаж растёт ровно на число прошедших месяцев;
        # часть снимков может быть пустой из-за сбоя источника.
        present = [
            (snapshot.snapshot_month, snapshot.values["relationship_months"])
            for snapshot in client_snapshots
            if snapshot.values["relationship_months"] is not None
        ]

        assert present

        for (month_a, value_a), (month_b, value_b) in zip(present, present[1:]):

            elapsed = (month_b.year - month_a.year) * 12 + (month_b.month - month_a.month)

            assert value_b - value_a == elapsed, client_id


def test_ownership_matches_product_state(snapshots):
    for client_id, client_snapshots in snapshots.items():

        state = ProductState(client_id)

        for snapshot in client_snapshots:

            owned = state.owned_at(snapshot.ts)

            for field, product in (
                ("holds_credit_card", "credit_card"),
                ("holds_debit_card", "debit_card"),
                ("holds_deposit", "deposit"),
            ):
                value = snapshot.values[field]

                if value is None:
                    continue

                assert value == (product in owned), (client_id, field)


def test_credit_fields_require_a_card(snapshots):
    for client_id, client_snapshots in snapshots.items():

        state = ProductState(client_id)

        for snapshot in client_snapshots:

            if "credit_card" in state.owned_at(snapshot.ts):
                continue

            assert snapshot.values["credit_limit"] is None
            assert snapshot.values["credit_utilization"] is None


def test_utilization_within_bounds(snapshots):
    values = [
        snapshot.values["credit_utilization"]
        for client_snapshots in snapshots.values()
        for snapshot in client_snapshots
        if snapshot.values["credit_utilization"] is not None
    ]

    assert values
    assert all(0.0 <= value <= 1.2 for value in values)


# ============================================================
# УТЕЧКА БУДУЩЕГО
# ============================================================


# Клиент с кредитом, открытым ПОСЛЕ среза (26.06.2026).
LEAK_CLIENT = 673


def test_profile_ignores_contracts_opened_later():
    """
    Регресс: договор, открытый после среза, не должен менять
    ни один снимок до среза.

    Раньше сегментные пропуски считались по всем договорам сразу,
    и кредит из окна метки задним числом заполнял доход во всех
    снимках окна признаков.
    """

    state = derive_lifecycle(LEAK_CLIENT, HISTORY_START, LABEL_END).state

    future = [h for h in state.holdings if h.opened_at >= FEATURE_END]

    assert future, "у клиента нет договора после среза, тест бессмысленен"

    with_future = profile_snapshots(LEAK_CLIENT, HISTORY_START, LABEL_END, state)

    past_only = copy.copy(state)
    past_only.holdings = [h for h in state.holdings if h.opened_at < FEATURE_END]

    without_future = profile_snapshots(LEAK_CLIENT, HISTORY_START, LABEL_END, past_only)

    for a, b in zip(with_future, without_future):
        if a.ts < FEATURE_END:
            assert a.values == b.values, a.snapshot_month


def test_feature_profile_does_not_depend_on_the_horizon():
    """
    То же end-to-end: генерация до конца окна метки и до среза
    обязана давать одинаковые снимки окна признаков.
    """

    for client_id in (0, 1, LEAK_CLIENT):

        full = generate_client_history(client_id, end=LABEL_END).before(FEATURE_END)
        short = generate_client_history(client_id, end=FEATURE_END)

        assert full.profile == short.profile, client_id


def test_credit_history_flag_is_as_of():
    state = derive_lifecycle(LEAK_CLIENT, HISTORY_START, LABEL_END).state

    opened = min(
        h.opened_at for h in state.holdings if h.product_type in ("credit_card", "cash_loan")
    )

    assert not state.has_credit_history_at(opened - timedelta(seconds=1))
    assert state.has_credit_history_at(opened)


# ============================================================
# ПРОПУСКИ
# ============================================================


def test_key_fields_are_never_missing(snapshots):
    for client_snapshots in snapshots.values():
        for snapshot in client_snapshots:
            for field in ALWAYS_PRESENT:
                assert snapshot.values[field] is not None


def test_industry_is_structurally_missing(snapshots):
    for client_id, client_snapshots in snapshots.items():

        persona = draw_persona(client_id)

        if persona.income_type in INDUSTRY_INCOME_TYPES:
            continue

        for snapshot in client_snapshots:
            assert snapshot.values["industry"] is None


def test_missing_rate_varies_by_month(snapshots):
    by_month: dict[str, list[bool]] = defaultdict(list)

    for client_snapshots in snapshots.values():
        for snapshot in client_snapshots:
            key = snapshot.snapshot_month.strftime("%Y-%m")
            by_month[key].append(snapshot.values["declared_income"] is None)

    rates = {month: sum(values) / len(values) for month, values in by_month.items()}

    assert max(rates.values()) - min(rates.values()) > 0.10, rates


def test_group_fields_go_missing_together(snapshots):
    """
    Пропуск приходит блоком: отваливается источник, а не поле.
    """

    hits = 0

    for client_snapshots in snapshots.values():
        for snapshot in client_snapshots:
            for group, fields in FIELD_GROUPS.items():

                present = [snapshot.values[field] is not None for field in fields]

                if group == "employment":
                    # industry имеет собственный структурный пропуск.
                    present = [
                        snapshot.values[field] is not None
                        for field in fields
                        if field != "industry"
                    ]

                if group == "credit":
                    continue

                assert len(set(present)) == 1, (group, snapshot.snapshot_month)

                hits += 1

    assert hits > 0


def test_some_clients_have_a_permanently_empty_block(snapshots):
    empty = 0

    for client_snapshots in snapshots.values():

        if all(snapshot.values["income_type"] is None for snapshot in client_snapshots):
            empty += 1

    assert 0 < empty < POPULATION * 0.4


# ============================================================
# RAW
# ============================================================


def test_raw_profile_shape(raw_tables, emit_clients):
    profile = raw_tables["profile"]

    assert set(PROFILE_FIELDS) <= set(profile.columns)
    assert {"client_id", "ts", "snapshot_month"} <= set(profile.columns)

    per_client = profile.groupby("client_id").size()

    assert per_client.max() == 24
    assert profile.ts.max() < pd.Timestamp(FEATURE_END)


def test_snapshot_row_matches_values():
    snapshot = profile_snapshots(2, HISTORY_START, FEATURE_END)[-1]

    row = snapshot_row(snapshot)

    assert row["client_id"] == 2
    assert row["ts"] == snapshot.ts

    for field in PROFILE_FIELDS:
        assert row[field] == snapshot.values[field]
