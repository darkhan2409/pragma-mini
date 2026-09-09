"""
Приложение: сессии, экраны, воронка заявки, операции, баннеры.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pytest

from src.generator.app import (
    adopted_domains,
    application_session,
    sessions_for_day,
)
from src.generator.config import FEATURE_END, HISTORY_START, LABEL_END
from src.generator.coverage import app_adoption
from src.generator.history import generate_client_history
from src.generator.world import (
    ACTION_CLICKED,
    ACTION_SHOWN,
    BANNER_OFFERS,
    BANNER_SLOTS,
    BROWSE_SCREENS,
    DOMAIN_OPERATIONS,
    FUNNEL_SCREENS,
    FUNNEL_STAGES,
    OPERATION_STATUSES,
    REJECT_REASONS,
    SCREEN_HOME,
    SCREEN_OFFERS,
)


ALL_SCREENS = (
    {screen for screens in BROWSE_SCREENS.values() for screen in screens}
    | set(FUNNEL_SCREENS.values())
    | {SCREEN_HOME, SCREEN_OFFERS}
)

ALL_OPERATIONS = {
    operation for operations in DOMAIN_OPERATIONS.values() for operation in operations
}


@pytest.fixture(scope="module")
def app_client() -> int:
    """
    Клиент, который пользуется приложением и хотя бы раз
    подавал в нём заявку: иначе воронку не на чем проверять.
    """

    for client_id in range(200):

        if app_adoption(client_id) is None:
            continue

        history = generate_client_history(client_id)

        if any(screen.funnel_stage for screen in history.app_screens):
            return client_id

    raise AssertionError("не нашлось клиента с заявкой в приложении")


@pytest.fixture(scope="module")
def history(app_client):
    return generate_client_history(app_client)


# ============================================================
# СЕССИИ
# ============================================================


def test_sessions_are_deterministic(app_client):
    day = datetime(2025, 9, 10)

    first = sessions_for_day(app_client, day)
    second = sessions_for_day(app_client, day)

    assert [s.session_id for s in first] == [s.session_id for s in second]


def test_screens_belong_to_a_session(history):
    assert history.app_screens

    for screen in history.app_screens:
        assert screen.session_id
        assert screen.session_id.isdigit()
        assert screen.firebase_screen in ALL_SCREENS or screen.firebase_screen == "(not set)"


def test_screens_within_session_are_ordered(history):
    by_session: dict[str, list] = {}

    for screen in history.app_screens:
        by_session.setdefault(screen.session_id, []).append(screen)

    assert by_session

    for screens in by_session.values():
        timestamps = [screen.ts for screen in screens]
        assert timestamps == sorted(timestamps)

        # Сессия не может длиться сутками.
        assert timestamps[-1] - timestamps[0] < timedelta(hours=6)


def test_client_uses_only_adopted_domains(app_client, history):
    domains = set(adopted_domains(app_client))

    screen_to_domain = {
        screen: domain
        for domain, screens in BROWSE_SCREENS.items()
        for screen in screens
    }

    for screen in history.app_screens:

        if screen.funnel_stage is not None:
            continue

        domain = screen_to_domain.get(screen.firebase_screen)

        if domain is None:
            continue

        assert domain in domains


# ============================================================
# ВОРОНКА
# ============================================================


def test_funnel_stage_order_and_reject_reason(history):
    funnels: dict[str, list] = {}

    for screen in history.app_screens:
        if screen.funnel_stage is not None:
            funnels.setdefault(screen.session_id, []).append(screen)

    assert funnels

    for screens in funnels.values():

        stages = [screen.funnel_stage for screen in screens]

        assert stages[:3] == ["view", "application", "kyc"]
        assert stages[3] in ("approved", "rejected")

        for screen in screens:
            assert screen.product is not None
            assert screen.firebase_screen == FUNNEL_SCREENS[screen.funnel_stage]

            if screen.funnel_stage == "rejected":
                assert screen.reject_reason in REJECT_REASONS
            else:
                assert screen.reject_reason is None


def test_application_session_shape():
    started = datetime(2025, 7, 3, 14, 5)

    approved = application_session(4, "cash_loan", started, approved=True)
    rejected = application_session(4, "cash_loan", started, approved=False)

    assert [s.funnel_stage for s in approved.screens] == [
        "view",
        "application",
        "kyc",
        "approved",
    ]
    assert [s.funnel_stage for s in rejected.screens][-1] == "rejected"

    assert all(s.session_id == approved.session_id for s in approved.screens)
    assert approved.screens[0].ts >= started


def test_funnel_stages_are_known(history):
    stages = {s.funnel_stage for s in history.app_screens if s.funnel_stage}

    assert stages <= set(FUNNEL_STAGES)


# ============================================================
# ОПЕРАЦИИ
# ============================================================


def test_operations_contract(history):
    assert history.app_operations

    for operation in history.app_operations:
        assert operation.domain in DOMAIN_OPERATIONS
        assert operation.operation in DOMAIN_OPERATIONS[operation.domain]
        assert operation.operation in ALL_OPERATIONS
        assert operation.status is None or operation.status in OPERATION_STATUSES


def test_operations_are_dominated_by_auth(history):
    domains = Counter(operation.domain for operation in history.app_operations)

    assert domains["auth"] > 0
    assert len(domains) > 1


# ============================================================
# БАННЕРЫ
# ============================================================


def test_banner_contract_single_action_per_row(history):
    assert history.banners

    for banner in history.banners:
        assert banner.slot in BANNER_SLOTS
        assert banner.offer in BANNER_OFFERS
        assert banner.action in (ACTION_SHOWN, ACTION_CLICKED)

        # Двух флагов быть не должно.
        assert not hasattr(banner, "shown")
        assert not hasattr(banner, "clicked")


def test_click_never_precedes_its_impression(history):
    shown: dict[tuple[str, str], list] = {}

    for banner in history.banners:
        if banner.action == ACTION_SHOWN:
            shown.setdefault((banner.slot, banner.offer), []).append(banner.ts)

    clicks = [b for b in history.banners if b.action == ACTION_CLICKED]

    assert clicks

    for click in clicks:

        impressions = shown.get((click.slot, click.offer), [])

        assert any(
            0 <= (click.ts - ts).total_seconds() <= 60 for ts in impressions
        ), click


def test_clicks_are_rarer_than_impressions(history):
    actions = Counter(banner.action for banner in history.banners)

    assert actions[ACTION_CLICKED] < 0.15 * actions[ACTION_SHOWN]


# ============================================================
# ЧАСТОТА ВОРОНКИ В ВЫБОРКЕ
# ============================================================


def test_funnel_events_are_common_enough(raw_tables, emit_clients):
    """
    У отдельного клиента заявок может не быть вовсе, это норма.
    Но в выборке воронка обязана встречаться часто, иначе
    учить на ней нечего.

    Ориентир, снятый на 10 000 клиентов: воронка есть у 65
    процентов клиентов, 1.17 заявки на клиента за 24 месяца.
    """

    screens = raw_tables["app_screens"]

    funnel = screens[screens.funnel_stage.notna()]

    assert not funnel.empty, "в выборке нет ни одной заявки"

    with_funnel = funnel.client_id.nunique()

    assert with_funnel / emit_clients > 0.25, (
        f"воронка слишком редкая: {with_funnel} из {emit_clients}"
    )

    applications = int((funnel.funnel_stage == "application").sum())

    assert applications / emit_clients > 0.3, (
        f"заявок на клиента слишком мало: {applications / emit_clients:.2f}"
    )

    decided = funnel.funnel_stage.isin(["approved", "rejected"]).sum()

    assert decided == applications, "каждая заявка обязана получить решение"

    approved = int((funnel.funnel_stage == "approved").sum())

    assert 0.4 < approved / decided < 0.95, "доля одобрений вне разумных границ"


# ============================================================
# КЛИЕНТ БЕЗ ПРИЛОЖЕНИЯ
# ============================================================


def test_client_without_app_has_no_app_streams():
    client_id = next(c for c in range(200) if app_adoption(c) is None)

    history = generate_client_history(client_id)

    assert history.app_screens == []
    assert history.app_operations == []
    assert history.banners == []

    # При этом обычная жизнь клиента продолжается.
    assert history.transactions
    assert history.profile
