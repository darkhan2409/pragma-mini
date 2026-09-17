from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# ДОХОДЫ
# ============================================================
#
# Зарплата не приходит идеальным таймером: она бывает ранней,
# поздней, частичной, пропущенной, переносится с выходных,
# меняется в сумме и вовсе прекращается при увольнении.
#
# Часть дохода не видна этому банку: она уходит на счёт другого
# банка или приходит наличными.
# ============================================================


INCOME_KINDS = (
    "salary",
    "pension",
    "business",
    "freelance",
    "rent",
    "family_support",
    "social_benefit",
    "severance",
)

LANDINGS = ("hcb_account", "other_bank", "cash")


@dataclass(frozen=True)
class IncomeParams:

    kinds: tuple = INCOME_KINDS

    # Основной поток дохода по типу занятости.
    primary_kind_by_income_type: dict = field(
        default_factory=lambda: {
            "employed": "salary",
            "state_employee": "salary",
            "self_employed": "freelance",
            "business_owner": "business",
            "pensioner": "pension",
            "student": "family_support",
            "unemployed": "social_benefit",
        }
    )

    # Расписание выплат основного потока.
    schedule_weights: dict = field(
        default_factory=lambda: {
            "salary": {"monthly": 0.72, "twice_monthly": 0.28},
            "pension": {"monthly": 1.0},
            "business": {"irregular": 0.70, "monthly": 0.30},
            "freelance": {"irregular": 0.82, "monthly": 0.18},
            "rent": {"monthly": 1.0},
            "family_support": {"monthly": 0.65, "irregular": 0.35},
            "social_benefit": {"monthly": 1.0},
            "severance": {"irregular": 1.0},
        }
    )

    # Вероятность второго потока дохода.
    second_stream_share: float = 0.22

    second_kind_weights: dict = field(
        default_factory=lambda: {
            "freelance": 0.38,
            "rent": 0.22,
            "family_support": 0.24,
            "business": 0.16,
        }
    )

    second_stream_amount_share: tuple = (0.10, 0.45)

    # Куда приземляется поток. Скорректируется видимой долей роли HCB.
    landing_weights: dict = field(
        default_factory=lambda: {
            "salary": {"hcb_account": 0.55, "other_bank": 0.38, "cash": 0.07},
            "pension": {"hcb_account": 0.48, "other_bank": 0.44, "cash": 0.08},
            "business": {"hcb_account": 0.32, "other_bank": 0.50, "cash": 0.18},
            "freelance": {"hcb_account": 0.36, "other_bank": 0.40, "cash": 0.24},
            "rent": {"hcb_account": 0.30, "other_bank": 0.30, "cash": 0.40},
            "family_support": {"hcb_account": 0.42, "other_bank": 0.30, "cash": 0.28},
            "social_benefit": {"hcb_account": 0.50, "other_bank": 0.45, "cash": 0.05},
            "severance": {"hcb_account": 0.55, "other_bank": 0.40, "cash": 0.05},
        }
    )

    # Исходы конкретной выплаты.
    payout_outcome: dict = field(
        default_factory=lambda: {
            "on_time": 0.70,
            "early": 0.08,
            "late": 0.15,
            "partial": 0.05,
            "skipped": 0.02,
        }
    )

    early_days: tuple = (1, 4)
    late_days: tuple = (1, 9)
    partial_share: tuple = (0.35, 0.75)
    partial_topup_days: tuple = (2, 14)

    # Перенос с выходного или праздника.
    weekend_shift: str = "earlier"
    weekend_shift_share: float = 0.85

    # Стресс делает выплаты хуже.
    stress_late_boost: float = 0.55
    stress_skip_boost: float = 0.28
    stress_partial_boost: float = 0.30

    # Бонусы и отпускные.
    quarterly_bonus_share: float = 0.22
    annual_bonus_share: float = 0.35
    annual_bonus_month: int = 12
    bonus_share_of_income: tuple = (0.35, 1.60)
    vacation_pay_share: float = 0.55
    vacation_pay_of_income: tuple = (0.60, 1.25)

    # Индексация и изменения суммы.
    annual_indexation: tuple = (0.02, 0.12)
    indexation_month_weights: dict = field(
        default_factory=lambda: {1: 0.45, 4: 0.15, 7: 0.20, 9: 0.10, 10: 0.10}
    )
    raise_share_per_year: float = 0.24
    raise_factor: tuple = (1.06, 1.28)
    cut_share_per_year: float = 0.07
    cut_factor: tuple = (0.72, 0.94)

    # Смена работодателя.
    employer_change_share_per_year: float = 0.16
    employer_gap_days: tuple = (0, 45)

    # Нерегулярный доход.
    irregular_events_per_month: tuple = (0.6, 3.2)
    irregular_amount_sigma: float = 0.62

    # Увольнение.
    severance_months: tuple = (0, 2)
    unemployment_benefit_share: float = 0.35
    unemployment_benefit_of_income: tuple = (0.15, 0.40)

    # Зарплатный день.
    salary_day_range: tuple = (1, 29)

    # Государственная пенсия приходит в начале месяца, а не в
    # любой день, как зарплата.
    pension_day_range: tuple = (3, 11)
    second_payday_offset: int = 15
    payout_hour_range: tuple = (6, 13)
