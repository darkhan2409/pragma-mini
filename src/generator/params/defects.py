from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SCHEMA_CHANGES


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

    # Смена схемы в середине истории: поле начинает собираться.
    # Таблица статична и объявлена в config: препроцессинг читает
    # оттуда же, поэтому переопределять её параметрами нельзя.
    schema_changes: tuple = SCHEMA_CHANGES

    # Доступность источника клиенту.
    app_adoption_share: float = 0.93
    consent_share: float = 0.92
