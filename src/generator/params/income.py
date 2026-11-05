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
            # Безработного без работы до окна держит семья. Прежде это
            # было бессрочное «пособие» ГФСС до 500 тыс. ₸ (аудит
            # 2026-11-05, M9): пособие по потере работы — 1–6 месяцев
            # после увольнения (unemployment_benefit_months).
            "unemployed": "family_support",
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
            "on_time": 0.73,
            "early": 0.08,
            "late": 0.15,
            "partial": 0.02,
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
    raise_share_per_year: float = 0.24
    raise_factor: tuple = (1.06, 1.28)
    cut_share_per_year: float = 0.07
    cut_factor: tuple = (0.72, 0.94)

    # Нерегулярный доход.
    irregular_events_per_month: tuple = (0.6, 3.2)
    irregular_amount_sigma: float = 0.62

    # Увольнение.
    severance_months: tuple = (0, 2)
    # Пособие ГФСС по потере работы: 45% среднего дохода, 1–6 месяцев
    # по стажу участия (research.md); доход для расчёта не выше
    # social_income_cap_mzp МЗП. Доля обратившихся — синтетическое
    # допущение.
    unemployment_benefit_share: float = 0.35
    unemployment_benefit_of_income: float = 0.45
    unemployment_benefit_months: tuple = (1, 7)
    social_income_cap_mzp: int = 7
    mzp: int = 85_000

    # Рождение ребёнка (research.md). Работающей матери ГФСС платит
    # за 126 дней декрета разом (70 до родов и 56 после), затем по
    # уходу за ребёнком 40% дохода до полутора лет; зарплата на это
    # время останавливается. Пособие при рождении — 38 МРП на 1–3-го
    # ребёнка и 63 МРП на 4-го и следующих; неработающей матери по
    # уходу платит бюджет: 5.76 / 6.81 / 7.85 / 8.90 МРП по порядку
    # ребёнка.
    maternity_days_before: int = 70
    maternity_days: int = 126
    childcare_share: float = 0.40
    childcare_days: int = 548
    birth_benefit_mrp: tuple = (38, 63)
    childcare_budget_mrp: tuple = (5.76, 6.81, 7.85, 8.90)
    # Через сколько дней после права приходит выплата.
    benefit_delay_days: tuple = (7, 25)

    # Зарплатный день.
    salary_day_range: tuple = (1, 29)

    # Пенсия по возрасту приходит и тому, кто продолжает работать:
    # она не заменяет доход, а добавляется к нему. Доля от
    # собственного дохода человека, ограниченная снизу, потому что
    # базовая часть выплаты от заработка не зависит.
    state_pension_of_income: tuple = (0.18, 0.34)
    # Пенсия не ниже минимальной совокупной (солидарная и базовая) на
    # начало 2024 года: около 95 тыс. ₸ в 2025 году после индексации
    # (research.md, уверенность средняя). Было 60 и 75 тыс. — ниже
    # законного минимума (аудит 2026-11-05, M9).
    state_pension_floor: int = 88_000
    # Индексация пенсий с 1 января — одна на всех: средняя пенсия
    # 131 356 → 142 669 ₸ за 2025 год, в 2026 году +10% (research.md).
    # Прежде индексация была своей у каждого, и 13% пенсионеров в
    # 2025 году получали меньше, чем в 2024.
    pension_indexation: dict = field(default_factory=lambda: {2025: 0.086, 2026: 0.10})
    second_payday_offset: int = 15
    # Выплата не раньше стольких дней до срока считается выплатой этого
    # срока (engine._next_payday): ранний реестр и перенос с выходного
    # двигают её на 1–6 дней. Меньше расстояния между выплатами при
    # двух в месяц.
    early_payout_window_days: int = 10
    payout_hour_range: tuple = (6, 13)
