from __future__ import annotations

import pandas as pd
import pytest

from churn.activity import is_client_action


def frame(**fields) -> pd.DataFrame:
    columns = ["type", "reason", "channel", "direction", "counterparty", "migration_reason", "change_source"]
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
    ({"type": "bill_payment", "channel": "app"}, True),
    ({"type": "bill_payment", "channel": "branch"}, True),
    ({"type": "loan_payment", "channel": "ecom"}, True),
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
