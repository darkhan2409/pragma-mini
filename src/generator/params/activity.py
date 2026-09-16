from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# АКТИВНОСТЬ
# ============================================================
#
# Интенсивности задаются режимом активности, состоянием
# жизненного цикла и ролью банка. Мягкие ограничители не дают
# хвосту превратиться в бесконечный цикл повторов.
# ============================================================


@dataclass(frozen=True)
class ActivityParams:

    # Покупок в день по режиму активности.
    purchases_per_day: dict = field(
        default_factory=lambda: {
            "rare": 0.14,
            "moderate": 1.70,
            "regular": 3.10,
            "high": 4.60,
            "extreme": 6.00,
        }
    )

    # Сессий приложения в день.
    sessions_per_day: dict = field(
        default_factory=lambda: {
            "rare": 0.005,
            "moderate": 0.018,
            "regular": 0.046,
            "high": 0.110,
            "extreme": 0.250,
        }
    )

    # Множитель по состоянию жизненного цикла.
    state_factor: dict = field(
        default_factory=lambda: {
            "prospect": 0.0,
            "onboarding": 0.55,
            "new_client": 0.85,
            "active": 1.0,
            "growing": 1.30,
            "stable": 1.0,
            "financial_stress": 0.80,
            "delinquent": 0.65,
            "dormant": 0.0,
            "churn_risk": 0.45,
            "churned": 0.0,
            "returned": 0.90,
            "closed_relationship": 0.0,
        }
    )

    # Множитель по роли банка: видимая доля оборота.
    role_factor: dict = field(
        default_factory=lambda: {
            "primary": 1.30,
            "secondary": 0.85,
            "credit_only": 0.45,
            "deposit_only": 0.35,
            "episodic": 0.40,
        }
    )

    # Пауза по видам: что именно замолкает.
    pause_silences: dict = field(
        default_factory=lambda: {
            "full": ("purchases", "sessions", "transfers", "cash", "bills"),
            "app_only": ("sessions",),
            "cards_only": ("purchases", "cash"),
            "other_bank": ("purchases", "transfers", "cash", "bills"),
            "seasonal": ("purchases", "sessions", "transfers", "cash", "bills"),
        }
    )

    other_bank_residual: float = 0.12

    # Нехватка денег на счёте в банке почти никогда не выглядит
    # как отказ: потребность закрывается наличными, деньгами в
    # другом банке или просто откладывается. Наблюдаемый отказ
    # это редкое событие, и после пары отказов за день клиент
    # перестаёт пробовать.
    hidden_purchase_share: float = 0.62
    decline_attempt_share: float = 0.10
    max_declines_per_day: int = 2
    autopay_attempt_share: float = 0.30

    weekend_factor_purchases: float = 1.10
    weekend_factor_sessions: float = 0.92

    # Часовые профили: будни и выходные различаются формой.
    hour_profile_weekday: tuple = (
        0.006, 0.003, 0.002, 0.002, 0.003, 0.008,
        0.022, 0.048, 0.062, 0.058, 0.052, 0.058,
        0.078, 0.070, 0.055, 0.052, 0.062, 0.085,
        0.098, 0.082, 0.055, 0.030, 0.014, 0.008,
    )

    hour_profile_weekend: tuple = (
        0.010, 0.006, 0.004, 0.003, 0.003, 0.005,
        0.010, 0.020, 0.035, 0.055, 0.072, 0.080,
        0.082, 0.078, 0.072, 0.068, 0.065, 0.068,
        0.072, 0.064, 0.050, 0.032, 0.020, 0.012,
    )

    session_hour_profile: tuple = (
        0.010, 0.005, 0.003, 0.003, 0.003, 0.008,
        0.025, 0.055, 0.080, 0.085, 0.075, 0.070,
        0.075, 0.075, 0.065, 0.065, 0.070, 0.085,
        0.100, 0.100, 0.085, 0.060, 0.035, 0.015,
    )

    # Ночной сегмент: кто вообще покупает ночью.
    night_segment_share: float = 0.08
    night_hours: tuple = (0, 1, 2, 3, 4, 5)
    night_boost: float = 6.0

    # Мягкие ограничители.
    max_sessions_per_day: int = 9
    max_screens_per_session: int = 20
    max_purchases_per_day: int = 18
    max_offline_settlements_per_day: int = 2
    max_active_contracts: int = 9
    max_retries_per_intent: int = 3
    max_intents_per_session: int = 3
    max_support_hops: int = 1

    # Внешние переводы и наличные.
    transfers_per_month: dict = field(
        default_factory=lambda: {
            "rare": 0.3,
            "moderate": 1.1,
            "regular": 2.4,
            "high": 4.5,
            "extreme": 7.5,
        }
    )

    # Вероятность показать баннеры на экране витрины.
    banner_screen_share: float = 0.35

    cash_withdrawals_per_month: dict = field(
        default_factory=lambda: {
            "rare": 0.5,
            "moderate": 1.1,
            "regular": 1.6,
            "high": 2.1,
            "extreme": 2.8,
        }
    )

    cash_withdrawal_share_of_income: tuple = (0.05, 0.35)

    # Продолжительность сессии и шага.
    session_step_seconds: tuple = (4, 95)
    session_max_seconds: int = 3600
    min_step_seconds: int = 1

    # Сбой сервиса: общий для всех клиентов, в RAW не пишется.
    outage_day_probability: float = 0.05
    outage_domains: tuple = ("transfers", "payments", "auth", "cards")
    outage_hours: tuple = (1, 4)
    outage_failure_boost: float = 0.35
