from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# СУММЫ
# ============================================================
#
# Сумма покупки НЕ определяется одним MCC. Она складывается из
# потребности, размера корзины, ценового уровня точки, дохода
# домохозяйства, чувствительности к цене и сезона.
# ============================================================


@dataclass(frozen=True)
class AmountParams:

    # Базовая корзина категории в тенге для среднего дохода.
    basket_median: dict = field(
        default_factory=lambda: {
            "grocery": 6_500, "market": 4_200, "convenience": 1_900,
            "fastfood": 3_200, "restaurant": 9_500, "coffee": 1_600, "delivery": 5_200,
            "pharmacy": 4_100, "medical": 18_000, "lab": 12_000,
            "fuel": 12_000, "car_service": 32_000, "parking": 700,
            "taxi": 1_800, "transit": 200, "car_rental": 18_000, "micromobility": 500,
            "airline": 95_000, "railway": 12_000, "hotel": 42_000, "travel": 180_000,
            "clothing": 18_000, "shoes": 22_000, "home_goods": 14_000, "furniture": 85_000,
            "electronics": 95_000, "appliances": 120_000,
            "telecom": 3_500, "internet": 6_000, "subscription": 2_500,
            "utilities": 14_000,
            "education": 35_000, "kids": 9_000, "books": 4_500,
            "entertainment": 6_500, "cinema": 3_000, "sports": 12_000, "beauty": 9_500, "cosmetics": 7_000,
            "marketplace": 12_000, "ecom": 16_000,
            "government": 9_000, "fines": 15_000, "taxes": 30_000,
            "financial": 8_000, "charity": 3_000,
            "pets": 6_000, "tobacco": 2_400, "gambling": 8_000,
        }
    )

    basket_sigma: dict = field(
        default_factory=lambda: {
            "grocery": 0.55, "market": 0.60, "convenience": 0.50,
            "fastfood": 0.45, "restaurant": 0.65, "coffee": 0.35, "delivery": 0.50,
            "pharmacy": 0.70, "medical": 0.85, "lab": 0.60,
            "fuel": 0.40, "car_service": 0.95, "parking": 0.45,
            "taxi": 0.55, "transit": 0.20, "car_rental": 0.55, "micromobility": 0.40,
            "airline": 0.55, "railway": 0.50, "hotel": 0.60, "travel": 0.60,
            "clothing": 0.75, "shoes": 0.60, "home_goods": 0.80, "furniture": 0.85,
            "electronics": 0.90, "appliances": 0.75,
            "telecom": 0.25, "internet": 0.15, "subscription": 0.40,
            "utilities": 0.35,
            "education": 0.80, "kids": 0.70, "books": 0.55,
            "entertainment": 0.70, "cinema": 0.35, "sports": 0.55, "beauty": 0.55, "cosmetics": 0.65,
            "marketplace": 0.85, "ecom": 0.85,
            "government": 0.65, "fines": 0.60, "taxes": 0.75,
            "financial": 0.70, "charity": 0.75,
            "pets": 0.60, "tobacco": 0.35, "gambling": 0.95,
        }
    )

    # Как доход сдвигает сумму: степень эластичности.
    income_elasticity: float = 0.28

    # Ценовой сегмент точки.
    price_segment_factor: dict = field(
        default_factory=lambda: {
            "budget": 0.62,
            "mid": 1.00,
            "premium": 1.85,
            "luxury": 3.40,
        }
    )

    # Чувствительность к цене смещает выбор сегмента и корзину.
    price_sensitivity_factor: float = 0.30

    # Размер домохозяйства влияет на продуктовую корзину.
    household_size_categories: tuple = ("grocery", "market", "utilities", "kids", "delivery", "pharmacy")
    household_size_elasticity: float = 0.45

    # Округление суммы.
    round_to: int = 10

    # Какая доля денег месяца проходит именно покупками по
    # картам этого банка: остальное уходит счетами, наличными,
    # переводами и мимо банка.
    card_share_of_spend: float = 0.62

    # Насколько резко клиент сбавляет траты, когда деньги
    # месяца заканчиваются раньше времени.
    budget_pressure_strength: float = 2.4
    budget_pressure_floor: float = 0.12

    # Сколько денег на счёте клиент считает нормальным запасом:
    # ниже этого он заметно сбавляет траты.
    comfortable_balance_share: float = 0.10
    empty_wallet_floor: float = 0.06

    # Дешёвое покупают чаще дорогого. Вес категории в выборе
    # уменьшается с размером её корзины: кофе берут каждый день,
    # холодильник раз в несколько лет.
    reference_basket: int = 5_000
    frequency_from_basket: float = 0.62
    amount_bounds: tuple = (100, 8_000_000)

    # Регулярные счета.
    bill_medians: dict = field(
        default_factory=lambda: {
            "utilities": 14_000,
            "telecom": 3_500,
            "internet": 6_000,
            "kindergarten": 35_000,
            "fines": 15_000,
            "taxes": 30_000,
        }
    )

    bill_sigma: dict = field(
        default_factory=lambda: {
            "utilities": 0.30,
            "telecom": 0.18,
            "internet": 0.10,
            "kindergarten": 0.20,
            "fines": 0.55,
            "taxes": 0.70,
        }
    )

    utilities_winter_factor: float = 1.42

    # Подписки.
    subscription_count: tuple = (0, 5)
    subscription_median: int = 2_500
    subscription_sigma: float = 0.50
    subscription_stop_share: float = 0.30
    subscription_price_change_share: float = 0.28
    subscription_price_change: tuple = (0.08, 0.30)

    # Разовые редкие счета в месяц.
    fine_rate_per_month: float = 0.16
    tax_per_year: tuple = (1, 2)
