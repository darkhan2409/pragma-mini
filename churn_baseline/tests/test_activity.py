from __future__ import annotations

import pandas as pd
import pytest

from churn.activity import is_client_action, is_target_action, is_visit


def frame(**fields) -> pd.DataFrame:
    columns = [
        "type", "reason", "channel", "direction", "counterparty", "migration_reason", "change_source",
        "operation", "status",
    ]
    row = {name: fields.get(name) for name in columns}
    return pd.DataFrame([row])


CASES = [
    # действие клиента
    ({"type": "app_screen"}, True),
    ({"type": "app_operation", "channel": None}, True),
    ({"type": "purchase", "reason": "purchase", "channel": "pos"}, True),
    ({"type": "transfer_out", "reason": "own_transfer", "channel": "app"}, True),
    ({"type": "p2p_out"}, True),
    ({"type": "cash_withdrawal", "channel": "atm"}, True),
    ({"type": "cash_deposit", "channel": "atm"}, True),
    ({"type": "cash_deposit", "reason": "cash_deposit", "channel": "atm"}, True),
    ({"type": "bill_payment", "channel": "app"}, True),
    ({"type": "bill_payment", "channel": "branch"}, True),
    ({"type": "deposit_topup", "direction": "debit", "channel": "ecom"}, True),
    ({"type": "deposit_withdrawal", "reason": "early_closure", "direction": "debit"}, True),
    ({"type": "transfer_in", "reason": "transfer", "counterparty": "Own account"}, True),
    ({"type": "early_repayment"}, True),
    ({"type": "application_submitted"}, True),
    ({"type": "banner_clicked"}, True),
    ({"type": "case_opened"}, True),
    ({"type": "card_blocked", "reason": "client_freeze"}, True),
    ({"type": "card_blocked", "reason": "lost_or_stolen"}, True),
    ({"type": "card_unblocked", "reason": "client_request"}, True),
    ({"type": "product_closed", "reason": "early_closure"}, True),
    ({"type": "product_migrated", "migration_reason": "successor_offer"}, True),
    ({"type": "profile_change", "change_source": "client"}, True),
    # обязательства: взнос по кредиту и пополнение под платёж или выписку
    # (решение владельца 2026-10-06)
    ({"type": "loan_payment", "channel": "ecom"}, False),
    ({"type": "loan_payment", "channel": "app"}, False),
    ({"type": "cash_deposit", "reason": "payment_topup", "channel": "atm"}, False),
    ({"type": "cash_deposit", "reason": "card_statement", "channel": "atm"}, False),
    ({"type": "transfer_in", "reason": "payment_topup", "counterparty": "Own account"}, False),
    # банк, автоматика, третьи лица
    ({"type": "purchase", "reason": "subscription", "channel": "ecom"}, False),
    ({"type": "purchase", "reason": "insurance_premium", "channel": "system"}, False),
    ({"type": "bill_payment", "channel": "system"}, False),
    ({"type": "loan_payment", "channel": "system"}, False),
    ({"type": "deposit_topup", "direction": "credit", "channel": "system"}, False),
    ({"type": "deposit_withdrawal", "reason": "matured", "direction": "debit"}, False),
    ({"type": "deposit_withdrawal", "reason": "early_closure", "direction": "credit"}, False),
    ({"type": "transfer_in", "reason": "inbound", "counterparty": "A. Person"}, False),
    ({"type": "transfer_in", "reason": "own_transfer", "counterparty": "Own account"}, False),
    ({"type": "p2p_in"}, False),
    ({"type": "salary_credit"}, False),
    ({"type": "pension_credit"}, False),
    ({"type": "other_income_credit"}, False),
    ({"type": "cashback_credit"}, False),
    ({"type": "interest_credit"}, False),
    ({"type": "fee_charge"}, False),
    ({"type": "balance_snapshot"}, False),
    ({"type": "refund"}, False),
    ({"type": "reversal"}, False),
    ({"type": "chargeback"}, False),
    ({"type": "loan_disbursement"}, False),
    ({"type": "installment_paid"}, False),
    ({"type": "installment_due"}, False),
    ({"type": "installment_missed"}, False),
    ({"type": "schedule_created"}, False),
    ({"type": "loan_closed"}, False),
    ({"type": "loan_restructured"}, False),
    ({"type": "arrears_cleared"}, False),
    ({"type": "delinquency_registered"}, False),
    ({"type": "product_opened", "reason": "application_approved"}, False),
    ({"type": "account_opened", "reason": "opened"}, False),
    ({"type": "card_activated"}, False),
    ({"type": "card_reissued", "reason": "lost_or_stolen"}, False),
    ({"type": "card_blocked", "reason": "fraud_suspicion"}, False),
    ({"type": "card_unblocked", "reason": "fraud_check_closed"}, False),
    ({"type": "product_closed", "reason": "matured"}, False),
    ({"type": "product_migrated", "migration_reason": "forced_migration"}, False),
    ({"type": "product_renewed"}, False),
    ({"type": "product_repriced"}, False),
    ({"type": "contract_terms_changed"}, False),
    ({"type": "profile_change", "change_source": "application"}, False),
    ({"type": "application_decision"}, False),
    ({"type": "case_updated"}, False),
    ({"type": "case_resolved"}, False),
    ({"type": "communication_sent", "channel": "push"}, False),
    ({"type": "banner_shown"}, False),
    ({"type": "fraud_alert"}, False),
    ({"type": "fraud_decision"}, False),
]


@pytest.mark.parametrize(("fields", "expected"), CASES)
def test_client_action_rules(fields: dict, expected: bool) -> None:
    assert bool(is_client_action(frame(**fields))[0]) is expected


def test_every_generator_type_is_classified() -> None:
    # Все 55 типов выгрузки встречаются в таблице хотя бы раз.
    types = {fields["type"] for fields, _ in CASES}
    assert len(types) == 55


VISITS = [
    ({"type": "app_operation", "operation": "login", "status": "success"}, True),
    ({"type": "app_operation", "operation": "biometry_login", "status": "success"}, True),
    # неудачный вход — действие клиента, но не визит
    ({"type": "app_operation", "operation": "login", "status": "failed"}, False),
    ({"type": "app_operation", "operation": "login", "status": None}, False),
    # прочие события приложения — действие, но не визит
    ({"type": "app_operation", "operation": "card_view", "status": "success"}, False),
    ({"type": "app_screen"}, False),
    ({"type": "banner_clicked"}, False),
]


@pytest.mark.parametrize(("fields", "expected"), VISITS)
def test_a_visit_is_only_a_successful_login(fields: dict, expected: bool) -> None:
    assert bool(is_visit(frame(**fields))[0]) is expected


TARGETS = [
    ({"type": "purchase", "reason": "purchase", "channel": "pos"}, True),
    ({"type": "app_operation", "operation": "login", "status": "success"}, True),
    # продукт по заявке клиента — целевое действие, хотя не действие клиента
    ({"type": "product_opened", "reason": "application_approved"}, True),
    # карта при регистрации и автоматическая активация — дело банка
    ({"type": "product_opened", "reason": "opened"}, False),
    ({"type": "card_activated"}, False),
    ({"type": "loan_payment", "channel": "ecom"}, False),
]


@pytest.mark.parametrize(("fields", "expected"), TARGETS)
def test_target_action_is_a_client_action_or_an_applied_product(fields: dict, expected: bool) -> None:
    events = frame(**fields)
    assert bool(is_target_action(events, is_client_action(events))[0]) is expected


def test_visit_action_and_target_action_are_different_sets() -> None:
    rows = [fields for fields, _ in VISITS + TARGETS]
    events = pd.concat([frame(**fields) for fields in rows], ignore_index=True)
    action = is_client_action(events)
    visit = is_visit(events)
    target = is_target_action(events, action)

    # Визит — всегда действие; действие — не всегда визит; целевое
    # действие шире действия на открытие продукта по заявке.
    assert (action | ~visit).all()
    assert (action & ~visit).any()
    assert (target & ~action).any()
