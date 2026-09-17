from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# КАЛЕНДАРЬ И СЕЗОННОСТЬ
# ============================================================
#
# Праздники Казахстана заданы явно: фиксированные даты и
# плавающий Курбан айт. Даты за пределами таблицы не
# выдумываются.
# ============================================================


FIXED_HOLIDAYS = (
    (1, 1, "new_year"),
    (1, 2, "new_year"),
    (1, 7, "orthodox_christmas"),
    (3, 8, "womens_day"),
    (3, 21, "nauryz"),
    (3, 22, "nauryz"),
    (3, 23, "nauryz"),
    (5, 1, "unity_day"),
    (5, 7, "defender_day"),
    (5, 9, "victory_day"),
    (7, 6, "capital_day"),
    (8, 30, "constitution_day"),
    (10, 25, "republic_day"),
    (12, 16, "independence_day"),
    (12, 17, "independence_day"),
)

# Курбан айт по годам окна наблюдения.
KURBAN_AIT = {
    2024: (6, 16),
    2025: (6, 6),
    2026: (5, 27),
}


@dataclass(frozen=True)
class SeasonalityParams:

    fixed_holidays: tuple = FIXED_HOLIDAYS
    kurban_ait: dict = field(default_factory=lambda: dict(KURBAN_AIT))

    # Множители трат в праздничные окна по категориям.
    holiday_factors: dict = field(
        default_factory=lambda: {
            "new_year": {"grocery": 1.55, "marketplace": 1.60, "electronics": 1.45, "restaurant": 1.40,
                         "clothing": 1.30, "cosmetics": 1.50, "entertainment": 1.35, "kids": 1.60,
                         "travel": 1.25, "delivery": 1.30},
            "womens_day": {"cosmetics": 1.75, "beauty": 1.55, "restaurant": 1.45, "clothing": 1.25,
                           "marketplace": 1.30, "coffee": 1.20},
            "nauryz": {"grocery": 1.40, "market": 1.45, "restaurant": 1.35, "clothing": 1.25,
                       "entertainment": 1.30, "travel": 1.20, "fuel": 1.15},
            "kurban_ait": {"market": 1.65, "grocery": 1.35, "charity": 2.40, "clothing": 1.20, "fuel": 1.15},
            "victory_day": {"grocery": 1.15, "restaurant": 1.20, "travel": 1.15, "fuel": 1.12},
            "unity_day": {"grocery": 1.12, "travel": 1.18, "fuel": 1.12},
            "capital_day": {"entertainment": 1.20, "restaurant": 1.15},
            "constitution_day": {"travel": 1.15, "fuel": 1.10},
            "republic_day": {"entertainment": 1.15, "restaurant": 1.12},
            "independence_day": {"marketplace": 1.25, "grocery": 1.20, "restaurant": 1.25},
            "orthodox_christmas": {"grocery": 1.15, "restaurant": 1.15},
            "defender_day": {"restaurant": 1.15, "marketplace": 1.10},
        }
    )

    # Дней до праздника, когда начинается подготовка.
    pre_holiday_days: dict = field(
        default_factory=lambda: {
            "new_year": 30,
            "womens_day": 5,
            "nauryz": 6,
            "kurban_ait": 4,
            "independence_day": 4,
        }
    )

    post_holiday_dip_days: int = 12
    post_holiday_dip: float = 0.80

    # Месячные множители категорий.
    month_factors: dict = field(
        default_factory=lambda: {
            "utilities": {1: 1.42, 2: 1.38, 3: 1.20, 4: 1.00, 5: 0.86, 6: 0.80,
                          7: 0.78, 8: 0.80, 9: 0.88, 10: 1.05, 11: 1.25, 12: 1.38},
            "travel": {1: 0.80, 2: 0.78, 3: 0.95, 4: 1.00, 5: 1.15, 6: 1.35,
                       7: 1.45, 8: 1.40, 9: 1.05, 10: 0.90, 11: 0.85, 12: 1.10},
            "hotel": {1: 0.85, 2: 0.85, 3: 0.95, 4: 1.00, 5: 1.10, 6: 1.30,
                      7: 1.40, 8: 1.35, 9: 1.05, 10: 0.95, 11: 0.90, 12: 1.05},
            "airline": {1: 0.85, 2: 0.80, 3: 0.95, 4: 1.00, 5: 1.10, 6: 1.35,
                        7: 1.45, 8: 1.35, 9: 1.00, 10: 0.90, 11: 0.85, 12: 1.15},
            "education": {1: 0.85, 2: 1.05, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.72,
                          7: 0.70, 8: 1.30, 9: 1.45, 10: 1.10, 11: 1.00, 12: 0.95},
            "kids": {1: 0.85, 2: 0.95, 3: 1.00, 4: 0.95, 5: 0.95, 6: 0.90,
                     7: 0.95, 8: 1.45, 9: 1.25, 10: 1.00, 11: 1.00, 12: 1.35},
            "books": {1: 0.90, 2: 1.00, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.80,
                      7: 0.85, 8: 1.40, 9: 1.25, 10: 1.00, 11: 1.00, 12: 1.15},
            "electronics": {1: 0.82, 2: 0.90, 3: 1.00, 4: 0.95, 5: 0.95, 6: 0.95,
                            7: 0.95, 8: 1.10, 9: 1.05, 10: 1.00, 11: 1.30, 12: 1.50},
            "marketplace": {1: 0.88, 2: 0.95, 3: 1.05, 4: 1.00, 5: 1.00, 6: 0.98,
                            7: 0.95, 8: 1.05, 9: 1.05, 10: 1.05, 11: 1.35, 12: 1.45},
            "clothing": {1: 0.85, 2: 0.90, 3: 1.10, 4: 1.05, 5: 1.00, 6: 0.95,
                         7: 0.90, 8: 1.20, 9: 1.15, 10: 1.10, 11: 1.10, 12: 1.30},
            "pharmacy": {1: 1.30, 2: 1.25, 3: 1.15, 4: 1.00, 5: 0.92, 6: 0.85,
                         7: 0.82, 8: 0.85, 9: 0.98, 10: 1.12, 11: 1.25, 12: 1.30},
            "fuel": {1: 0.92, 2: 0.92, 3: 1.00, 4: 1.02, 5: 1.05, 6: 1.10,
                     7: 1.12, 8: 1.12, 9: 1.05, 10: 1.00, 11: 0.96, 12: 0.96},
            "restaurant": {1: 0.90, 2: 0.95, 3: 1.05, 4: 1.00, 5: 1.05, 6: 1.08,
                           7: 1.08, 8: 1.05, 9: 1.02, 10: 1.00, 11: 0.98, 12: 1.25},
            "micromobility": {1: 0.15, 2: 0.15, 3: 0.40, 4: 0.85, 5: 1.25, 6: 1.50,
                              7: 1.55, 8: 1.50, 9: 1.20, 10: 0.70, 11: 0.25, 12: 0.15},
            "sports": {1: 1.30, 2: 1.15, 3: 1.05, 4: 1.00, 5: 1.00, 6: 0.90,
                       7: 0.82, 8: 0.85, 9: 1.15, 10: 1.10, 11: 1.05, 12: 0.90},
        }
    )

    weekend_factors: dict = field(
        default_factory=lambda: {
            "cinema": 1.45, "entertainment": 1.40, "restaurant": 1.32, "market": 1.35,
            "delivery": 1.25, "grocery": 1.18, "fastfood": 1.15, "clothing": 1.22,
            "marketplace": 1.10, "furniture": 1.30, "home_goods": 1.25, "sports": 1.20,
            "beauty": 1.15, "travel": 1.10, "fuel": 1.05,
            "transit": 0.55, "government": 0.15, "medical": 0.55, "education": 0.60,
            "car_service": 0.80, "lab": 0.50, "taxes": 0.10, "fines": 0.30,
        }
    )

    # Зарплатный эффект: первые дни после выплаты тратят больше.
    # Общий уровень месяца поверх категорийной сезонности.
    month_factor: dict = field(
        default_factory=lambda: {
            1: 0.88, 2: 0.92, 3: 1.06, 4: 0.98, 5: 1.02, 6: 1.00,
            7: 1.02, 8: 1.06, 9: 1.04, 10: 0.98, 11: 0.98, 12: 1.22,
        }
    )

    payday_window_days: int = 4
    payday_boost: float = 1.45
    pre_payday_days: int = 5
    pre_payday_factor: float = 0.82

    # Начало и конец месяца.
    month_start_days: int = 3
    month_start_boost: dict = field(
        default_factory=lambda: {"utilities": 1.6, "telecom": 1.5, "internet": 1.5}
    )

    # Зимние регионы: коммуналка дороже.
    winter_months: tuple = (11, 12, 1, 2, 3)
    winter_region_factor: dict = field(
        default_factory=lambda: {
            "Astana": 1.22, "Petropavl": 1.22, "Kostanay": 1.18, "Pavlodar": 1.16,
            "Oskemen": 1.14, "Karaganda": 1.14, "Aktobe": 1.10, "Oral": 1.10,
            "Semey": 1.12, "Kokshetau": 1.18, "Taldykorgan": 1.04, "Atyrau": 1.02,
            "Aktau": 0.96, "Almaty": 1.00, "Taraz": 0.94, "Kyzylorda": 0.94,
            "Shymkent": 0.88, "Turkistan": 0.88, "Konaev": 1.00, "Zhezkazgan": 1.16,
        }
    )

    # Учебный год и отпуска.
    school_year_months: tuple = (9, 10, 11, 12, 1, 2, 3, 4, 5)
    vacation_months: tuple = (6, 7, 8)
