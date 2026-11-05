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
            # Доля машины — до поправки на доход (population.car_income_elasticity):
            # около 250 легковых на 1000 жителей (research.md), в селе без
            # машины ездят попутками и маршрутками (taxi, transit).
            "metropolis": {"transit": 0.36, "taxi": 0.26, "car": 0.30, "micromobility": 0.08},
            "major_city": {"transit": 0.34, "taxi": 0.24, "car": 0.36, "micromobility": 0.06},
            "regional_centre": {"transit": 0.32, "taxi": 0.24, "car": 0.40, "micromobility": 0.04},
            "industrial_town": {"transit": 0.30, "taxi": 0.23, "car": 0.45, "micromobility": 0.02},
            "small_town": {"transit": 0.24, "taxi": 0.25, "car": 0.49, "micromobility": 0.02},
            "district_centre": {"transit": 0.20, "taxi": 0.27, "car": 0.52, "micromobility": 0.01},
            "rural": {"transit": 0.12, "taxi": 0.33, "car": 0.55, "micromobility": 0.00},
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

    # Куда едут внутри страны (синтетическое допущение по
    # популярным направлениям: курорты Бурабай — Щучинск, Капшагай —
    # Конаев, Каспий — Актау, паломничество — Туркестан).
    domestic_destinations: dict = field(
        default_factory=lambda: {
            "Almaty": 0.22, "Astana": 0.16, "Shchuchinsk": 0.14, "Konaev": 0.12,
            "Aktau": 0.09, "Shymkent": 0.09, "Turkistan": 0.10, "Taldykorgan": 0.08,
        }
    )

    # Билет на поездку покупают за столько дней до неё; внутри
    # страны часть летит самолётом, остальные едут поездом.
    trip_ticket_lead_days: tuple = (7, 60)
    domestic_flight_share: float = 0.45

    # Категории, которые живут только дома: в поездке их нет
    # (аудит 2026-11-05, context-03 — налоги и дорога на работу за
    # границей).
    home_only_categories: tuple = (
        "government", "taxes", "fines", "utilities", "telecom", "internet", "education",
        "car_service", "parking", "transit", "fuel", "kids", "furniture", "appliances",
        "home_goods", "pets", "medical", "lab", "market", "micromobility",
    )

    # Доля покупок в домашней зоне, рабочей и прочих.
    zone_share: dict = field(
        default_factory=lambda: {"home": 0.56, "work": 0.26, "other": 0.18}
    )

    # Расстояние наказывает выбор точки.
    distance_decay: float = 0.55

    # Вероятность зайти в новое место вместо любимого.
    new_place_base: float = 0.30
