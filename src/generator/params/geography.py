from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# ГЕОГРАФИЯ
# ============================================================
#
# Тип населённого пункта задаёт доступность категорий, плотность
# точек, уровень цен, долю наличных и e-commerce, транспорт,
# часы работы и вероятность поездок.
# ============================================================


SETTLEMENT_TYPES = (
    "metropolis",
    "major_city",
    "regional_centre",
    "industrial_town",
    "small_town",
    "district_centre",
    "rural",
)


@dataclass(frozen=True)
class GeographyParams:

    settlement_types: tuple = SETTLEMENT_TYPES

    # Сколько торговых зон (районов) в поселении.
    districts: dict = field(
        default_factory=lambda: {
            "metropolis": (7, 12),
            "major_city": (5, 8),
            "regional_centre": (3, 6),
            "industrial_town": (2, 4),
            "small_town": (2, 3),
            "district_centre": (1, 2),
            "rural": (1, 1),
        }
    )

    # Плотность точек: сколько outlets на категорию в поселении.
    outlets_per_category: dict = field(
        default_factory=lambda: {
            "metropolis": (14, 120),
            "major_city": (8, 55),
            "regional_centre": (4, 26),
            "industrial_town": (3, 16),
            "small_town": (2, 10),
            "district_centre": (1, 5),
            "rural": (1, 3),
        }
    )

    # Уровень цен относительно среднего по стране.
    price_level: dict = field(
        default_factory=lambda: {
            "metropolis": 1.22,
            "major_city": 1.08,
            "regional_centre": 1.00,
            "industrial_town": 1.02,
            "small_town": 0.90,
            "district_centre": 0.85,
            "rural": 0.82,
        }
    )

    # Доля наличных операций.
    cash_share: dict = field(
        default_factory=lambda: {
            "metropolis": 0.11,
            "major_city": 0.16,
            "regional_centre": 0.21,
            "industrial_town": 0.23,
            "small_town": 0.30,
            "district_centre": 0.38,
            "rural": 0.46,
        }
    )

    # Доля e-commerce и доставки.
    ecom_share: dict = field(
        default_factory=lambda: {
            "metropolis": 0.30,
            "major_city": 0.23,
            "regional_centre": 0.18,
            "industrial_town": 0.15,
            "small_town": 0.12,
            "district_centre": 0.09,
            "rural": 0.07,
        }
    )

    # Транспортная структура: чем добираются.
    transport_mix: dict = field(
        default_factory=lambda: {
            "metropolis": {"transit": 0.36, "taxi": 0.26, "car": 0.30, "micromobility": 0.08},
            "major_city": {"transit": 0.32, "taxi": 0.24, "car": 0.38, "micromobility": 0.06},
            "regional_centre": {"transit": 0.28, "taxi": 0.22, "car": 0.46, "micromobility": 0.04},
            "industrial_town": {"transit": 0.24, "taxi": 0.18, "car": 0.56, "micromobility": 0.02},
            "small_town": {"transit": 0.16, "taxi": 0.18, "car": 0.64, "micromobility": 0.02},
            "district_centre": {"transit": 0.10, "taxi": 0.16, "car": 0.73, "micromobility": 0.01},
            "rural": {"transit": 0.05, "taxi": 0.10, "car": 0.85, "micromobility": 0.00},
        }
    )

    # Часы работы точек: обычный и расширенный график.
    opening_hours: dict = field(
        default_factory=lambda: {
            "metropolis": {"standard": (8, 23), "extended_share": 0.30},
            "major_city": {"standard": (8, 22), "extended_share": 0.22},
            "regional_centre": {"standard": (9, 21), "extended_share": 0.16},
            "industrial_town": {"standard": (9, 21), "extended_share": 0.12},
            "small_town": {"standard": (9, 20), "extended_share": 0.08},
            "district_centre": {"standard": (9, 19), "extended_share": 0.05},
            "rural": {"standard": (9, 18), "extended_share": 0.02},
        }
    )

    # Круглосуточные точки: аптеки, АЗС, магазины у дома.
    around_the_clock_categories: tuple = ("pharmacy", "fuel", "convenience")

    foreign_countries: dict = field(
        default_factory=lambda: {
            "RU": 0.19, "TR": 0.16, "AE": 0.13, "CN": 0.09, "GE": 0.07,
            "UZ": 0.06, "KG": 0.05, "DE": 0.05, "US": 0.04, "TH": 0.04,
            "EG": 0.03, "IT": 0.02, "ES": 0.02, "GB": 0.02, "AZ": 0.02, "PL": 0.01,
        }
    )

    home_country: str = "KZ"

    # Доля покупок в домашней зоне, рабочей и прочих.
    zone_share: dict = field(
        default_factory=lambda: {"home": 0.56, "work": 0.26, "other": 0.18}
    )

    # Расстояние наказывает выбор точки.
    distance_decay: float = 0.55

    # Вероятность зайти в новое место вместо любимого.
    new_place_base: float = 0.30
