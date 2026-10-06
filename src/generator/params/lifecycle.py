from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# ЖИЗНЕННЫЙ ЦИКЛ
# ============================================================
#
# Стадия клиента на день и жизненные события. Неактивность
# заранее не планируется: она вырастает из отношений клиента с
# банком (params/engagement.py, behaviour/engagement.py).
# ============================================================


@dataclass(frozen=True)
class LifecycleParams:

    # Стадия клиента на день (behaviour/engagement.stage): сколько
    # месяцев он новый после прихода, с какого уровня стресса и с
    # какой просрочки поведение сбавляет частоты.
    onboarding_months: int = 1
    new_client_months: int = 3
    stress_stage_level: float = 0.45
    delinquent_dpd: int = 30

    # Сколько дней молчания банк ждёт, прежде чем звать клиента
    # назад. Скрытое состояние банк не видит, он видит только
    # отсутствие операций, и замечает его не в первый же день.
    winback_after_silence_days: int = 30

    # ------------------------------------------------------------
    # ЖИЗНЕННЫЕ СОБЫТИЯ
    # ------------------------------------------------------------
    #
    # Событие меняет несколько потоков сразу: доход, бюджет,
    # категории трат, привычки, интерес к продуктам и вероятность
    # просрочки. Банк узнаёт о нём позже и не всегда.
    # ------------------------------------------------------------

    life_event_rate_per_year: dict = field(
        default_factory=lambda: {
            "job_change": 0.16,
            "job_loss": 0.055,
            "income_up": 0.24,
            "income_down": 0.07,
            "child_birth": 0.045,
            "move": 0.09,
            "big_purchase": 0.20,
            "illness": 0.13,
            "vacation": 0.55,
            "renovation": 0.07,
            "education": 0.06,
            "wedding": 0.03,
            "divorce": 0.018,
        }
    )

    life_event_stage_factor: dict = field(
        default_factory=lambda: {
            "young_adult": {"job_change": 2.1, "education": 3.2, "wedding": 2.4, "child_birth": 1.3,
                            "big_purchase": 0.7, "renovation": 0.4, "divorce": 0.3, "illness": 0.6},
            "early_career": {"job_change": 1.6, "child_birth": 2.6, "wedding": 2.2, "big_purchase": 1.2,
                             "education": 1.4, "move": 1.5},
            "family": {"child_birth": 1.1, "renovation": 1.6, "big_purchase": 1.3, "divorce": 1.5,
                       "education": 0.8, "job_change": 0.8},
            "mature": {"child_birth": 0.12, "job_change": 0.5, "illness": 1.6, "renovation": 1.2,
                       "wedding": 0.3, "education": 0.4, "move": 0.7},
            "retired": {"child_birth": 0.0, "job_change": 0.15, "job_loss": 0.2, "illness": 2.6,
                        "income_up": 0.3, "education": 0.1, "wedding": 0.1, "move": 0.5,
                        "big_purchase": 0.6, "vacation": 0.8},
        }
    )

    # Событие требует состояния: свадьба только у несемейных,
    # развод только у семейных.
    life_event_requires: dict = field(
        default_factory=lambda: {
            "wedding": ("single", "divorced", "widow"),
            "divorce": ("married", "civil_marriage"),
        }
    )

    # Сколько раз событие может повториться за горизонт.
    life_event_max: dict = field(
        default_factory=lambda: {
            "job_loss": 2,
            "child_birth": 2,
            "move": 2,
            "wedding": 1,
            "divorce": 1,
            "job_change": 3,
        }
    )

    # Переезд: в другое поселение или внутри своего.
    move_to_other_settlement_share: float = 0.42

    # Банк узнаёт об изменении атрибута с задержкой и не всегда.
    profile_change_known_share: dict = field(
        default_factory=lambda: {
            "move": 0.62,
            "job_change": 0.48,
            "job_loss": 0.22,
            "income_up": 0.35,
            "income_down": 0.20,
            "child_birth": 0.34,
            "wedding": 0.40,
            "divorce": 0.30,
        }
    )

    profile_change_delay_days: tuple = (3, 240)

    # Изменение подтверждается документом не всегда.
    profile_change_confirmed_share: float = 0.58
