from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# МЕРЧАНТЫ
# ============================================================
#
# Сеть, торговая точка и MCC это РАЗНЫЕ сущности. Популярность
# сетей и точек имеет тяжёлый хвост: несколько сетей собирают
# большую часть оборота, а десятки тысяч точек встречаются
# считанные разы.
# ============================================================


@dataclass(frozen=True)
class MerchantParams:

    # Масштаб каталога. 1.0 это рабочий ориентир V1 в сотни
    # тысяч точек, 0.05 достаточно для smoke.
    catalog_scale: float = 1.0

    # Тяжесть хвоста популярности сетей и точек. Это параметр
    # симуляции: из справочника популярность не выводится.
    brand_zipf_alpha: float = 1.05
    outlet_zipf_alpha: float = 0.85

    # Ценовые сегменты сетей.
    price_segment_weights: dict = field(
        default_factory=lambda: {
            "budget": 0.33,
            "mid": 0.47,
            "premium": 0.16,
            "luxury": 0.04,
        }
    )

    # Сегмент доступен не в каждом типе поселения.
    segment_availability: dict = field(
        default_factory=lambda: {
            "metropolis": ("budget", "mid", "premium", "luxury"),
            "major_city": ("budget", "mid", "premium", "luxury"),
            "regional_centre": ("budget", "mid", "premium"),
            "industrial_town": ("budget", "mid", "premium"),
            "small_town": ("budget", "mid"),
            "district_centre": ("budget", "mid"),
            "rural": ("budget",),
        }
    )

    # Имя в терминальной строке.
    name_variants: dict = field(
        default_factory=lambda: {
            "upper": 0.46,
            "title": 0.24,
            "translit": 0.16,
            "with_branch": 0.10,
            "facilitator": 0.04,
        }
    )

    facilitators: tuple = ("QRPAY", "IOKA", "EPAY", "CLOUDPAY", "PAYBOX", "ROBOKASSA")

    terminal_noise_share: float = 0.22

    branch_number_share: float = 0.35

    # Привычки клиента.
    favourite_outlets_per_category: tuple = (1, 3)
    favourite_categories: tuple = (4, 10)

    # Доля покупок в любимой точке при высокой лояльности.
    # Закрытая точка не продаёт. Ноль означает жёсткий запрет.
    closed_outlet_weight: float = 0.0

    loyalty_to_favourite: tuple = (0.35, 0.88)

    # Смена привычек после жизненного события.
    habit_reset_on_move: float = 0.85
    habit_reset_on_job_change: float = 0.40
    habit_reset_on_child_birth: float = 0.30

    # Бытовые последовательности.
    routine_share_weekday: float = 0.55
    routine_share_weekend: float = 0.35
