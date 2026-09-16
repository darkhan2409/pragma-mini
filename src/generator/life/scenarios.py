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
)

BY_NAME = {item.name: item for item in SCENARIOS}


def overrides_for(name: str) -> dict:
    return BY_NAME[name].overrides


__all__ = ["BY_NAME", "SCENARIOS", "Scenario", "overrides_for"]
