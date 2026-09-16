from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# МОШЕННИЧЕСТВО И АНТИФРОД
# ============================================================
#
# Мошенничество это ЭПИЗОД и риск, а не пожизненный класс
# клиента. Уязвимость повышает вероятность эпизода, но не
# делает клиента помеченным: в RAW нет ни флага, ни персоны.
# ============================================================


EPISODE_KINDS = (
    "card_compromise",
    "unusual_purchase",
    "suspicious_transfer",
    "social_engineering",
    "account_takeover",
    "false_positive",
)


@dataclass(frozen=True)
class FraudParams:

    kinds: tuple = EPISODE_KINDS

    # Базовая интенсивность эпизода в год на клиента.
    base_rate_per_year: float = 0.085

    kind_weights: dict = field(
        default_factory=lambda: {
            "card_compromise": 0.24,
            "unusual_purchase": 0.14,
            "suspicious_transfer": 0.14,
            "social_engineering": 0.12,
            "account_takeover": 0.06,
            "false_positive": 0.30,
        }
    )

    # Уязвимость и экспозиция.
    vulnerability_factor: float = 2.4
    online_exposure_factor: float = 1.6
    travel_exposure_factor: float = 1.9
    new_device_factor: float = 1.5

    # Шаги эпизода.
    probe_purchases: dict = field(
        default_factory=lambda: {"card_compromise": (2, 5), "account_takeover": (0, 2)}
    )
    probe_amount: tuple = (200, 3_000)
    strike_amount_share_of_limit: tuple = (0.25, 0.95)
    strike_count: dict = field(
        default_factory=lambda: {
            "card_compromise": (1, 3),
            "unusual_purchase": (1, 1),
            "suspicious_transfer": (1, 4),
            "social_engineering": (1, 3),
            "account_takeover": (2, 5),
        }
    )
    step_gap_minutes: tuple = (2, 180)

    # Антифрод.
    detection_probability: dict = field(
        default_factory=lambda: {
            "card_compromise": 0.72,
            "unusual_purchase": 0.55,
            "suspicious_transfer": 0.62,
            "social_engineering": 0.35,
            "account_takeover": 0.68,
            "false_positive": 1.0,
        }
    )

    decision_weights: dict = field(
        default_factory=lambda: {
            "monitor": 0.34,
            "confirm_request": 0.34,
            "block": 0.32,
        }
    )

    block_decision_boost_high_band: float = 2.6

    score_band_thresholds: tuple = (0.35, 0.70)

    detection_delay_minutes: tuple = (1, 240)

    # Реакция клиента.
    client_confirms_share: float = 0.45
    client_disputes_share: float = 0.42
    dispute_opens_case_share: float = 0.88
    chargeback_share_of_disputes: float = 0.66
    chargeback_delay_days: tuple = (3, 35)
    reissue_share_after_block: float = 0.72
    reissue_delay_days: tuple = (1, 12)
    unblock_delay_hours: tuple = (1, 96)

    # Ложное срабатывание: обычная поездка или крупная покупка.
    false_positive_confirm_share: float = 0.86
    false_positive_unblock_hours: tuple = (0, 24)

    # Восстановление активности после инцидента.
    recovery_days: tuple = (7, 45)
    recovery_activity_factor: float = 0.62
