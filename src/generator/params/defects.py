from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# ДЕФЕКТЫ НАБЛЮДАЕМОСТИ
# ============================================================
#
# Дефект принадлежит ИСТОЧНИКУ, а не равномерному шуму по всем
# данным: POS-витрина приходит батчами, GA4 отстаёт на сутки,
# кредитное обслуживание вовсе теряет время.
# ============================================================


MISSING_REASONS = (
    "not_applicable",
    "not_collected",
    "source_unavailable",
    "redacted",
    "unknown",
)


@dataclass(frozen=True)
class DefectParams:

    missing_reasons: tuple = MISSING_REASONS

    # Времени поступления записи в хранилище у выгрузки нет:
    # задержки, опоздания и порядок доставки не моделируются.

    # Дубли одной и той же записи.
    # Отмена уже проведённой операции.
    reversal_share: float = 0.0035
    reversal_delay_hours: tuple = (1, 96)

    # Возврат покупки по инициативе клиента или магазина.
    refund_share: float = 0.012
    refund_delay_days: tuple = (1, 21)
    partial_refund_share: float = 0.32

    # Пропуски полей с причиной.
    field_missing: dict = field(
        default_factory=lambda: {
            "transactions": {"merchant_city": (0.012, "source_unavailable"),
                             "merchant_name": (0.004, "not_collected"),
                             "mcc": (0.002, "unknown")},
            "app_screens": {"firebase_screen": (0.006, "not_collected")},
            "app_operations": {"status": (0.004, "source_unavailable")},
            "communications": {"delivered": (0.003, "source_unavailable")},
        }
    )

    ga4_not_set: str = "(not set)"

    # Сбой источника: на сутки данные не доходят вовсе.
    outage_days_per_year: dict = field(
        default_factory=lambda: {
            "app_screens": 3.0,
            "banners": 2.0,
            "communications": 1.2,
            "app_operations": 1.0,
            "antifraud": 0.8,
            "support": 0.6,
            "transactions": 0.25,
        }
    )

    outage_recovers_share: float = 0.55

    # Смена схемы в середине истории: поле начинает собираться.
    schema_changes: tuple = (
        {"source": "app_screens", "field": "product_id", "from": "2025-06-01",
         "reason": "not_collected"},
        {"source": "app_operations", "field": "device_new", "from": "2025-03-01",
         "reason": "not_collected"},
        {"source": "banners", "field": "campaign_code", "from": "2025-01-15",
         "reason": "not_collected"},
    )

    # Доступность источника клиенту.
    app_adoption_share: float = 0.93
    consent_share: float = 0.92
