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

    # Сколько сетей на категорию: национальных и локальных.
    national_brands_per_category: tuple = (2, 9)
    regional_brands_per_category: tuple = (1, 6)
    local_brands_per_settlement: tuple = (0, 4)

    # Тяжесть хвоста популярности сетей и точек.
    brand_zipf_alpha: float = 1.05
    outlet_zipf_alpha: float = 0.85

    # Доля национальных сетей в обороте категории.
    national_share: dict = field(
        default_factory=lambda: {
            "grocery": 0.62, "market": 0.05, "convenience": 0.30,
            "fastfood": 0.70, "restaurant": 0.28, "coffee": 0.58, "delivery": 0.86,
            "pharmacy": 0.66, "medical": 0.30, "lab": 0.72,
            "fuel": 0.78, "car_service": 0.22, "parking": 0.40,
            "taxi": 0.92, "transit": 0.98, "car_rental": 0.55, "micromobility": 0.90,
            "airline": 0.95, "railway": 0.99, "hotel": 0.35, "travel": 0.60,
            "clothing": 0.60, "shoes": 0.58, "home_goods": 0.55, "furniture": 0.50,
            "electronics": 0.82, "appliances": 0.80,
            "telecom": 0.99, "internet": 0.95, "subscription": 0.97,
            "utilities": 0.99,
            "education": 0.35, "kids": 0.55, "books": 0.60,
            "entertainment": 0.45, "cinema": 0.80, "sports": 0.42, "beauty": 0.18, "cosmetics": 0.72,
            "marketplace": 0.96, "ecom": 0.70,
            "government": 1.0, "fines": 1.0, "taxes": 1.0,
            "financial": 0.85, "charity": 0.70,
            "pets": 0.40, "tobacco": 0.35, "gambling": 0.75,
        }
    )

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

    aggregators: tuple = ("WOLT", "GLOVO", "CHOCOFOOD", "YANDEX EDA", "AIRBA", "KASPI SHOP")

    terminal_noise_share: float = 0.22

    branch_number_share: float = 0.35

    # Привычки клиента.
    favourite_outlets_per_category: tuple = (1, 3)
    favourite_categories: tuple = (4, 10)

    # Доля покупок в любимой точке при высокой лояльности.
    loyalty_to_favourite: tuple = (0.35, 0.88)

    # Смена привычек после жизненного события.
    habit_reset_on_move: float = 0.85
    habit_reset_on_job_change: float = 0.40
    habit_reset_on_child_birth: float = 0.30

    # Бытовые последовательности.
    routine_share_weekday: float = 0.55
    routine_share_weekend: float = 0.35
