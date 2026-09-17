from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# ДЕТЕРМИНИРОВАННЫЕ СЦЕНАРИИ
# ============================================================
#
# Редкое состояние нельзя проверять надеждой на удачу: пресет
# поднимает интенсивность нужного механизма так, что он
# обязательно проявится на маленькой популяции, не меняя самой
# логики.
#
# Пресет это ТОЛЬКО переопределение параметров. Ни один пресет
# не добавляет и не убирает кода: он двигает вероятности.
# ============================================================


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    overrides: dict
    expect: tuple


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="stress_recovery",
        description="увольнение, нарастание стресса, просрочка и восстановление",
        overrides={
            "lifecycle": {
                "life_event_rate_per_year": {"job_loss": 2.5, "income_down": 1.2, "illness": 0.6},
                "pause_probability_per_year": {"rare": 0.1, "moderate": 0.1, "regular": 0.1,
                                               "high": 0.1, "extreme": 0.1},
            },
            "stress": {"random_shock_per_year": 1.2, "cooldown_days": 20, "max_episodes": 6},
            "products": {"autopay_share": 0.2, "cure_probability_per_day": {"low": 0.05, "mid": 0.12, "high": 0.2}},
        },
        expect=("delinquency_registered", "arrears_cleared"),
    ),
    Scenario(
        name="income_late_and_missed",
        description="поздняя, частичная и пропущенная выплата дохода",
        overrides={
            "income": {
                "payout_outcome": {"on_time": 0.15, "early": 0.15, "late": 0.35,
                                   "partial": 0.25, "skipped": 0.10},
            },
        },
        expect=("salary_credit",),
    ),
    Scenario(
        name="card_compromise",
        description="компрометация карты, блокировка, обращение и возврат",
        overrides={
            "fraud": {
                "base_rate_per_year": 4.0,
                "kind_weights": {"card_compromise": 1.0, "unusual_purchase": 0.0,
                                 "suspicious_transfer": 0.0, "social_engineering": 0.0,
                                 "account_takeover": 0.0, "false_positive": 0.0},
                "detection_probability": {"card_compromise": 1.0, "unusual_purchase": 1.0,
                                          "suspicious_transfer": 1.0, "social_engineering": 1.0,
                                          "account_takeover": 1.0, "false_positive": 1.0},
                "decision_weights": {"monitor": 0.0, "confirm_request": 0.0, "block": 1.0},
                "client_disputes_share": 1.0,
                "dispute_opens_case_share": 1.0,
                "chargeback_share_of_disputes": 1.0,
            },
        },
        expect=("fraud_alert", "fraud_decision", "card_blocked", "case_opened", "chargeback"),
    ),
    Scenario(
        name="false_positive",
        description="ложное срабатывание антифрода и подтверждение клиентом",
        overrides={
            "fraud": {
                "base_rate_per_year": 4.0,
                "kind_weights": {"card_compromise": 0.0, "unusual_purchase": 0.0,
                                 "suspicious_transfer": 0.0, "social_engineering": 0.0,
                                 "account_takeover": 0.0, "false_positive": 1.0},
                "decision_weights": {"monitor": 0.0, "confirm_request": 0.0, "block": 1.0},
                "false_positive_confirm_share": 1.0,
                "reissue_share_after_block": 0.0,
            },
        },
        expect=("fraud_alert", "card_blocked", "card_unblocked"),
    ),
    Scenario(
        name="pause_and_return",
        description="полная пауза и возвращение клиента",
        overrides={
            "lifecycle": {
                "pause_probability_per_year": {"rare": 4.0, "moderate": 4.0, "regular": 4.0,
                                               "high": 4.0, "extreme": 4.0},
                "pause_kind_weights": {"full": 1.0, "app_only": 0.0, "cards_only": 0.0,
                                       "other_bank": 0.0, "seasonal": 0.0},
                "pause_length_days": {"full": (60, 120), "app_only": (30, 60),
                                      "cards_only": (30, 60), "other_bank": (60, 120),
                                      "seasonal": (30, 60)},
            },
        },
        expect=(),
    ),
    Scenario(
        name="product_migration",
        description="закрытие продаж и переход на продукт-преемник",
        overrides={
            "products": {
                "adoption": {
                    "base_rate_per_year": {"debit_card": 1.2, "credit_card": 1.0, "cash_loan": 1.2,
                                           "refinance": 0.4, "installment": 1.0, "deposit": 1.0,
                                           "deposit_certificate": 0.4, "bonds": 0.2,
                                           "insurance": 0.6, "service": 0.0},
                    "ramp_days": 30,
                    "ramp_start_share": 0.9,
                    "early_adopter_digital_factor": 2.2,
                    "offer_factor": 2.6,
                    "organic_share": 1.0,
                    "migration_pull": 12.0,
                    "pilot_share_default": 0.6,
                    "trait_factor": {},
                    "first_use_delay_days": (0, 5),
                    "notice_days_default": 5,
                },
            },
        },
        expect=("product_opened", "application_submitted", "application_decision"),
    ),
    Scenario(
        name="transfer_pair",
        description="внутрибанковский перевод и нехватка средств",
        overrides={
            "relationships": {
                "community_size": 8,
                "household_pair_share": 0.9,
                "internal_share": {"spouse": 1.0, "relative": 1.0, "friend": 1.0,
                                   "colleague": 1.0, "regular_counterparty": 1.0,
                                   "random_counterparty": 1.0},
                "frequency_per_month": {"spouse": (6.0, 10.0), "relative": (2.0, 4.0),
                                        "friend": (2.0, 4.0), "colleague": (1.0, 2.0),
                                        "employer": (0.0, 0.0), "landlord": (1.0, 1.0),
                                        "own_account_other_bank": (1.0, 3.0),
                                        "regular_counterparty": (2.0, 4.0),
                                        "random_counterparty": (0.5, 1.0)},
            },
            "activity": {
                "transfers_per_month": {"rare": 6.0, "moderate": 8.0, "regular": 10.0,
                                        "high": 12.0, "extreme": 14.0},
            },
        },
        expect=("p2p_out", "p2p_in"),
    ),
    Scenario(
        name="habitual_client",
        description="устойчивые любимые точки и повторные контрагенты",
        overrides={
            "merchants": {
                "loyalty_to_favourite": (0.85, 0.95),
                "favourite_outlets_per_category": (1, 2),
                "catalog_scale": 0.05,
            },
            "geography": {"new_place_base": 0.05},
        },
        expect=("purchase",),
    ),
    Scenario(
        name="dsr_reject",
        description="запрос сверх долговой нагрузки отклоняется по правилу, а не по монетке",
        overrides={
            "products": {
                "loan_amount_income_multiple": {
                    "cash_loan": (7.0, 9.0),
                    "refinance": (7.0, 9.0),
                    "installment": (6.0, 8.0),
                    "credit_card": (6.0, 8.0),
                },
                "bank_rules": {
                    "min_age": 18,
                    "max_age": 79,
                    "max_active_contracts": 9,
                    "max_debit_cards_total": 4,
                    "max_active_cash_loans": 3,
                    "max_debt_service_ratio": 0.02,
                },
                "application_probability_cap": 0.4,
                "application_cooldown_days": 0,
            },
        },
        expect=("application_decision",),
    ),
    Scenario(
        name="credit_discipline",
        description="дисциплинированный заёмщик платит вовремя и без просрочки",
        overrides={
            "products": {
                "on_time_payment_probability": {"low": 1.0, "mid": 1.0, "high": 1.0},
                "autopay_share": 0.0,
                "loan_topup_from_other_bank_share": 1.0,
                "adoption": {
                    "base_rate_per_year": {
                        "debit_card": 0.4, "credit_card": 0.0, "cash_loan": 1.2,
                        "refinance": 0.0, "installment": 0.8, "deposit": 0.1,
                        "deposit_certificate": 0.0, "bonds": 0.0,
                        "insurance": 0.0, "service": 0.0,
                    },
                    "ramp_days": 30,
                    "ramp_start_share": 0.9,
                    "early_adopter_digital_factor": 1.0,
                    "offer_factor": 1.0,
                    "organic_share": 1.0,
                    "migration_pull": 1.0,
                    "pilot_share_default": 0.6,
                    "trait_factor": {},
                    "first_use_delay_days": (0, 5),
                    "notice_days_default": 30,
                },
            },
        },
        expect=("installment_paid",),
    ),
    Scenario(
        name="refinance_closes_debt",
        description="рефинансирование гасит прежние кредиты, а не добавляет ещё один",
        overrides={
            "products": {
                "adoption": {
                    "base_rate_per_year": {
                        "debit_card": 0.3, "credit_card": 0.0, "cash_loan": 1.5,
                        "refinance": 2.5, "installment": 0.0, "deposit": 0.0,
                        "deposit_certificate": 0.0, "bonds": 0.0,
                        "insurance": 0.0, "service": 0.0,
                    },
                    "ramp_days": 30,
                    "ramp_start_share": 0.9,
                    "early_adopter_digital_factor": 1.0,
                    "offer_factor": 1.0,
                    "organic_share": 1.0,
                    "migration_pull": 1.0,
                    "pilot_share_default": 0.6,
                    "trait_factor": {},
                    "first_use_delay_days": (0, 5),
                    "notice_days_default": 30,
                },
                "application_probability_cap": 0.25,
                "application_cooldown_days": 5,
            },
        },
        expect=("loan_closed",),
    ),
    Scenario(
        name="transfer_fraud",
        description="подозрительный перевод и социальная инженерия выглядят переводом",
        overrides={
            "fraud": {
                "base_rate_per_year": 4.0,
                "kind_weights": {
                    "card_compromise": 0.0, "unusual_purchase": 0.0,
                    "suspicious_transfer": 0.5, "social_engineering": 0.3,
                    "account_takeover": 0.2, "false_positive": 0.0,
                },
                "detection_probability": {
                    "card_compromise": 1.0, "unusual_purchase": 1.0,
                    "suspicious_transfer": 1.0, "social_engineering": 1.0,
                    "account_takeover": 1.0, "false_positive": 1.0,
                },
                "decision_weights": {"monitor": 0.0, "confirm_request": 0.0, "block": 1.0},
            },
        },
        expect=("transfer_out", "fraud_alert"),
    ),
    Scenario(
        name="card_installments",
        description="покупка по карте рассрочки становится долгом и гасится частями",
        overrides={
            "products": {
                "adoption": {
                    "base_rate_per_year": {
                        "debit_card": 0.3, "credit_card": 2.5, "cash_loan": 0.0,
                        "refinance": 0.0, "installment": 0.0, "deposit": 0.0,
                        "deposit_certificate": 0.0, "bonds": 0.0,
                        "insurance": 0.0, "service": 0.0,
                    },
                    "ramp_days": 30,
                    "ramp_start_share": 0.9,
                    "early_adopter_digital_factor": 1.0,
                    "offer_factor": 1.0,
                    "organic_share": 1.0,
                    "migration_pull": 1.0,
                    "pilot_share_default": 0.6,
                    "trait_factor": {},
                    "first_use_delay_days": (0, 5),
                    "notice_days_default": 30,
                },
                "application_probability_cap": 0.3,
                "application_cooldown_days": 5,
            },
        },
        expect=("installment_due", "fee_charge"),
    ),
    Scenario(
        name="fraud_chargeback",
        description="клиент оспаривает чужую операцию, хотя карту банк не блокировал",
        overrides={
            "fraud": {
                "base_rate_per_year": 4.0,
                "kind_weights": {"card_compromise": 1.0, "unusual_purchase": 0.0,
                                 "suspicious_transfer": 0.0, "social_engineering": 0.0,
                                 "account_takeover": 0.0, "false_positive": 0.0},
                "detection_probability": {"card_compromise": 1.0, "unusual_purchase": 1.0,
                                          "suspicious_transfer": 1.0, "social_engineering": 1.0,
                                          "account_takeover": 1.0, "false_positive": 1.0},
                # Банк только наблюдает: блокировки нет ни одной.
                "decision_weights": {"monitor": 1.0, "confirm_request": 0.0, "block": 0.0},
                "block_decision_boost_high_band": 1.0,
                "client_disputes_share": 1.0,
                "dispute_opens_case_share": 1.0,
                "chargeback_share_of_disputes": 1.0,
            },
            # Клиентская блокировка тут только мешала бы: пресет
            # проверяет, что спор идёт БЕЗ участия блокировки.
            "products": {"card_block_client_share_per_year": 0.0},
        },
        expect=("chargeback",),
    ),
    Scenario(
        name="card_freeze",
        description="клиент временно замораживает свою карту и сам её размораживает",
        overrides={
            "fraud": {"base_rate_per_year": 0.0},
            "products": {
                "card_block_client_share_per_year": 40.0,
                "card_block_lost_share": 0.0,
                "card_freeze_self_unblock_share": 1.0,
            },
        },
        expect=("card_blocked", "card_unblocked"),
    ),
    Scenario(
        name="card_lost",
        description="утраченную карту не размораживают: её место занимает перевыпущенная",
        overrides={
            "fraud": {"base_rate_per_year": 0.0},
            "products": {
                "card_block_client_share_per_year": 40.0,
                "card_block_lost_share": 1.0,
            },
        },
        expect=("card_blocked", "card_reissued"),
    ),
    Scenario(
        name="inbound_money",
        description="деньги приходят извне, а не только уходят",
        overrides={
            "relationships": {
                "community_size": 8,
                "inbound_frequency_per_month": {
                    "spouse": (4.0, 8.0),
                    "relative": (3.0, 6.0),
                    "friend": (2.0, 4.0),
                    "colleague": (1.0, 2.0),
                    "regular_counterparty": (2.0, 4.0),
                    "random_counterparty": (0.5, 1.5),
                },
            },
        },
        expect=("transfer_in",),
    ),
)

BY_NAME = {item.name: item for item in SCENARIOS}


def overrides_for(name: str) -> dict:
    return BY_NAME[name].overrides


__all__ = ["BY_NAME", "SCENARIOS", "Scenario", "overrides_for"]
