from __future__ import annotations

from datetime import datetime

import pytest

from src.generator import config, emit
from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.finance import cards as card_rules
from src.generator.finance.entities import ACCOUNT_CARD, ACCOUNT_CURRENT
from src.generator.life import calendar as cal
from src.generator.simulate import CommunitySimulation
from src.generator.world import communities


# ============================================================
# ИДЕЯ
# ============================================================
#
# Долг по кредитной карте обслуживается выпиской: рассрочка
# покупок, проценты на наличные, минимальный платёж, пропуск.
# Всё это живёт в состоянии карты (ClientState.card_credits).
# Карта, открытая до окна, состояния не получала, и долг по ней
# копился без выписок, процентов и просрочки: остаток стоял на
# одном минусе полтора года.
#
# Выписку гасят со своего счёта в банке, а не хватает денег —
# подтягивают из другого банка. Клиенту, у которого в банке одна
# карта, подтягивать было некуда, и его выписка не проходила
# никогда. Теперь деньги приходят прямо на счёт карты.
# ============================================================


CLIENTS = 80


@pytest.fixture
def world():

    saved = (config.HISTORY_START, config.HISTORY_END, config.REGISTRATION_END)

    config.activate_horizon(datetime(2024, 1, 1), datetime(2026, 1, 1), datetime(2026, 1, 1))

    settings = emit._build_params(None, None, 4)
    params_module.activate(settings)
    rng_module.configure(7, settings.fingerprint(), 42)

    yield

    config.activate_horizon(*saved)
    rng_module.clear_caches()


def test_every_credit_card_open_at_the_window_start_has_its_debt_state(world):

    cards = 0

    for community in range(communities.community_count(CLIENTS)):

        sim = CommunitySimulation(community, communities.members(community, CLIENTS))

        for state in sim.clients.values():

            # Договоры до окна заводит движок перед первым днём
            # (engine.run_community); здесь — тем же вызовом.
            sim._prehistory(state)

            for contract in state.contracts.values():

                if contract.product_family != "credit_card" or contract.account_id is None:
                    continue

                if not contract.is_open_at(config.HISTORY_START):
                    continue

                cards += 1

                assert contract.contract_id in state.card_credits, (state.client_id, contract.contract_id)

    assert cards, "в выборке есть кредитные карты, открытые до окна"


def test_a_client_with_only_the_card_pays_the_statement_from_outside(world):

    # Модуль месяца импортируется через движок: они ссылаются друг
    # на друга, и прямой импорт первым упирается в цикл.
    from src.generator import engine  # noqa: F401
    from src.generator import engine_month

    found = None

    for community in range(communities.community_count(CLIENTS)):

        sim = CommunitySimulation(community, communities.members(community, CLIENTS))

        for state in sim.clients.values():
            sim._prehistory(state)
            if state.card_credits:
                found = sim, state
                break

        if found:
            break

    sim, state = found
    contract_id, credit = next(iter(state.card_credits.items()))

    # Своих счетов, кроме карты, у клиента больше нет.
    for account in state.ledger.accounts.values():
        if account.kind in (ACCOUNT_CARD, ACCOUNT_CURRENT):
            account.closed_at = config.HISTORY_START

    assert state.primary_card_account(datetime(2024, 3, 31)) is None

    card = state.ledger.accounts[credit.account_id]
    other = state.ledger.accounts[state.ledger.other_bank_id]
    cash = state.ledger.accounts[state.ledger.cash_id]

    card_rules.add_purchase(credit, 90_000, cal.month_index(datetime(2024, 2, 1)), 1)

    paid = False

    for day in (datetime(2024, 3, 31), datetime(2024, 4, 30), datetime(2024, 5, 31)):

        month = cal.month_index(day)
        minimum = card_rules.minimum_payment(credit, month)
        other.balance = 10 * minimum
        before = (card.balance, other.balance + cash.balance, credit.outstanding, len(state.events))

        if engine_month._pay_card(sim, state, day, credit, contract_id, minimum, None, month):
            paid = True
            break

    assert paid, "выписка оплачена хотя бы в одном из трёх месяцев"

    # Деньги пришли на карту из другого банка или наличными:
    # столько же ушло оттуда, сколько легло здесь, и ровно на
    # столько же меньше долг.
    assert card.balance - before[0] == minimum
    assert before[1] - (other.balance + cash.balance) == minimum
    assert before[2] - credit.outstanding == minimum

    kinds = [(event.event_type, event.payload.get("reason")) for event in state.events[before[3]:]]

    assert ("transfer_in", "card_statement") in kinds or ("cash_deposit", "card_statement") in kinds
    assert ("installment_paid", "payment") in kinds
    assert not any(kind == "loan_payment" for kind, _ in kinds)
