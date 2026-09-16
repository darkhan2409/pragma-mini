from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# ФИНАНСОВЫЙ СТРЕСС
# ============================================================
#
# Стресс это ПЕРИОД, а не тип клиента. У него есть причина,
# нарастание, пик, спад и исход. Он влияет на траты, остатки,
# платёжную дисциплину, обращения в поддержку и интерес к
# кредитным продуктам постепенно, а не рубильником.
# ============================================================


TRIGGERS = (
    "job_loss",
    "job_change",
    "income_delay",
    "illness",
    "big_purchase",
    "obligation_growth",
    "divorce",
    "move",
    "random_shock",
)

RESOLUTIONS = (
    "income_restored",
    "refinanced",
    "restructured",
    "savings_spent",
    "family_help",
    "faded",
    "unresolved",
)


@dataclass(frozen=True)
class StressParams:

    triggers: tuple = TRIGGERS

    # Базовая интенсивность спонтанного шока в год.
    random_shock_per_year: float = 0.22

    # Вероятность, что жизненное событие породит стресс.
    trigger_probability: dict = field(
        default_factory=lambda: {
            "job_loss": 0.92,
            "job_change": 0.30,
            "income_delay": 0.45,
            "illness": 0.55,
            "big_purchase": 0.35,
            "obligation_growth": 0.40,
            "divorce": 0.70,
            "move": 0.32,
            "random_shock": 1.0,
        }
    )

    # Сила эпизода по причине.
    intensity_range: dict = field(
        default_factory=lambda: {
            "job_loss": (0.65, 1.00),
            "job_change": (0.20, 0.50),
            "income_delay": (0.25, 0.55),
            "illness": (0.35, 0.75),
            "big_purchase": (0.25, 0.60),
            "obligation_growth": (0.30, 0.70),
            "divorce": (0.45, 0.85),
            "move": (0.20, 0.55),
            "random_shock": (0.25, 0.80),
        }
    )

    length_days: dict = field(
        default_factory=lambda: {
            "job_loss": (60, 320),
            "job_change": (20, 90),
            "income_delay": (15, 70),
            "illness": (30, 160),
            "big_purchase": (25, 120),
            "obligation_growth": (45, 260),
            "divorce": (60, 300),
            "move": (20, 110),
            "random_shock": (20, 150),
        }
    )

    # Доля эпизода, занятая нарастанием.
    onset_share: tuple = (0.10, 0.35)
    decay_share: tuple = (0.20, 0.50)

    # Дисциплина укорачивает и смягчает эпизод.
    discipline_intensity_factor: float = 0.45
    discipline_length_factor: float = 0.40
    savings_buffer_factor: float = 0.35

    # Эффекты при интенсивности 1.0.
    discretionary_cut: float = 0.55
    grocery_cut: float = 0.12
    utilization_rise: float = 0.65
    missed_payment_boost: float = 0.55
    support_contact_boost: float = 2.6
    refinance_interest_boost: float = 3.2
    loan_interest_boost: float = 2.1
    offer_response_boost: float = 1.7
    app_checks_boost_early: float = 1.8
    app_checks_drop_late: float = 0.55
    deposit_close_boost: float = 2.4

    # Исходы эпизода.
    resolution_weights: dict = field(
        default_factory=lambda: {
            "income_restored": 0.34,
            "refinanced": 0.14,
            "restructured": 0.06,
            "savings_spent": 0.16,
            "family_help": 0.12,
            "faded": 0.14,
            "unresolved": 0.04,
        }
    )

    # Повторный стресс возможен, но не сразу.
    cooldown_days: int = 60
    repeat_boost: float = 1.45
    max_episodes: int = 4
