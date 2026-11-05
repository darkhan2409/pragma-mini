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
            "taxi": 1_400, "transit": 200, "car_rental": 18_000, "micromobility": 500,
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
            "bars": 9_000, "flowers": 12_000, "gifts": 9_000,
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
            "bars": 0.60, "flowers": 0.55, "gifts": 0.70,
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
    # Память о тратах: вклад покупки в «уже потрачено» затухает с этим
    # временем, дней. Прежде считались ровно 30 дней: всплеск после
    # зарплаты держал траты 30 дней и выпадал из окна точно к
    # следующей зарплате — цикл повторялся (аудит 2026-11-05, C3).
    budget_memory_days: float = 10.0
    # Во сколько раз привычный темп покупок может превышать бюджет
    # покупок месяца (engine._money_factor): запас на крупные дни.
    money_headroom: float = 1.2

    # Пол частоты при пустом счёте: человек без денег на карте всё же
    # изредка платит ею (мелочь, остаток). Прежде 0.06 вместе с ценой
    # давал провал трат перед зарплатой (аудит 2026-11-05, C3).
    empty_wallet_floor: float = 0.15

    # Дешёвое покупают чаще дорогого. Вес категории в выборе
    # уменьшается с размером её корзины: кофе берут каждый день,
    # холодильник раз в несколько лет.
    # Курсы к тенге. Нужны и для зарубежных покупок, и для
    # облигаций в долларах: раньше курс 500 был зашит в код.
    fx_rates: dict = field(
        default_factory=lambda: {
            "USD": 500.0,
            "EUR": 545.0,
            "RUB": 5.6,
            "TRY": 14.5,
            "AED": 136.0,
            "GEL": 186.0,
            "KGS": 5.7,
            "UZS": 0.039,
            "CNY": 69.0,
            "THB": 14.2,
            "GBP": 640.0,
            "PLN": 126.0,
            "AZN": 294.0,
            "CZK": 22.0,
            "EGP": 10.4,
            "VND": 0.02,
        }
    )

    country_currency: dict = field(
        default_factory=lambda: {
            "TR": "TRY", "AE": "AED", "GE": "GEL", "KG": "KGS", "UZ": "UZS",
            "RU": "RUB", "CN": "CNY", "TH": "THB", "GB": "GBP", "PL": "PLN",
            "AZ": "AZN", "CZ": "CZK", "EG": "EGP", "VN": "VND", "DE": "EUR",
            "IT": "EUR", "ES": "EUR", "FR": "EUR", "US": "USD",
        }
    )

    reference_basket: int = 5_000
    frequency_from_basket: float = 0.62
    amount_bounds: tuple = (100, 8_000_000)

    # --- Цена отдельно от частоты (аудит 2026-11-05, C3) ---
    #
    # Сумма покупки зависит от точки, корзины и клиента, но не от дня:
    # зарплатный цикл, остаток и праздник меняют частоту покупок, а не
    # цену чашки кофе. Прежде дневной множитель бюджета умножал и
    # цену: пик ровно 100 ₸ (8% покупок), кофе 220–970 ₸.

    # Меню точки: цена позиции (база в начале 2024 г.), её разброс по
    # сетям и число дополнительных позиций в чеке (Пуассон). Позиции —
    # общие у сети. Ориентиры research.md: кофе около 1 310 ₸, комбо
    # фастфуда 3 400–4 000 ₸, кино 2 132 → 2 365 ₸.
    menu_items: dict = field(
        default_factory=lambda: {
            "coffee": (1_100, 0.30, 0.30),
            "fastfood": (1_500, 0.35, 0.90),
            "cinema": (2_200, 0.18, 0.80),
            # Бар: напитки и закуски, обычно не одна позиция.
            "bars": (2_400, 0.40, 1.60),
        }
    )

    # Шаг цены по категориям: товары — «красивые» цены …90 / …990,
    # корзины и счета-итоги — любое целое, услуги — по прейскуранту.
    charm_categories: tuple = (
        "clothing", "shoes", "electronics", "appliances", "furniture", "home_goods", "books",
        "sports", "kids", "cosmetics", "marketplace", "ecom", "flowers", "gifts",
    )
    price_step: dict = field(
        default_factory=lambda: {
            "gambling": 500, "charity": 500, "beauty": 500, "education": 1_000, "travel": 1_000,
            "medical": 100, "lab": 100, "hotel": 100, "car_rental": 100, "entertainment": 100,
            "financial": 100, "government": 10, "airline": 10, "railway": 10, "parking": 50,
            "car_service": 100,
        }
    )
    min_amount: dict = field(
        default_factory=lambda: {"coffee": 400, "fastfood": 700, "cinema": 1_200, "parking": 100,
                                 "micromobility": 150, "taxi": 600, "convenience": 150}
    )

    # Заправка: круглой суммой («на 5 000») или литрами по цене АИ-92
    # на дату (research.md: 202 → 231–239 ₸/л в 2024–2025).
    fuel_round_share: float = 0.55
    fuel_round_amounts: dict = field(
        default_factory=lambda: {3_000: 0.10, 5_000: 0.30, 10_000: 0.33, 15_000: 0.12, 20_000: 0.15}
    )
    fuel_liters: tuple = (28.0, 0.45)
    fuel_price: tuple = (
        ("2018-01-01", 180), ("2024-01-01", 205), ("2024-07-01", 210), ("2025-01-01", 216),
        ("2025-07-01", 226), ("2025-10-01", 235), ("2026-01-01", 240),
    )

    # Инфляция — индекс цен по группам ИПЦ, годовой темп по годам,
    # месячное начисление; база — январь 2024 (research.md: 2024 —
    # продукты 5.5%, услуги 13.3%, ЖКУ 14.2%; 2025 — 12.3%, продукты
    # 13.5%; 2026 — около 9%). Прочие темпы — синтетическое
    # приближение. Прежде цены за два года не менялись (аудит, M7).
    inflation: dict = field(
        default_factory=lambda: {
            "food": {2023: 0.10, 2024: 0.055, 2025: 0.135, 2026: 0.09},
            "services": {2023: 0.12, 2024: 0.133, 2025: 0.12, 2026: 0.10},
            "goods": {2023: 0.09, 2024: 0.075, 2025: 0.10, 2026: 0.08},
            "utilities": {2023: 0.12, 2024: 0.142, 2025: 0.15, 2026: 0.10},
        }
    )
    inflation_groups: dict = field(
        default_factory=lambda: {
            "food": ("grocery", "market", "convenience", "coffee", "fastfood", "restaurant", "delivery", "tobacco",
                     "bars"),
            "services": ("taxi", "parking", "car_rental", "micromobility", "airline", "railway", "hotel",
                         "travel", "education", "entertainment", "cinema", "sports", "beauty", "medical",
                         "lab", "car_service", "financial", "kindergarten"),
            "utilities": ("utilities",),
        }
    )

    # Комиссия банка за конвертацию валютной покупки к курсу НБК.
    fx_margin: float = 0.015

    # Проезд — тариф города по датам, ₸ за поездку при оплате картой:
    # Алматы 120 с 03.08.2024, Астана 110 с 01.08.2024 (research.md);
    # прежние тарифы и остальные города — синтетическое приближение.
    # Прежде проезд стоил непрерывно от 110 до 2 770 ₸ (аудит
    # 2026-11-05, M3).
    transit_fares: dict = field(
        default_factory=lambda: {
            "Almaty": (("2018-01-01", 80), ("2024-08-03", 120)),
            "Astana": (("2018-01-01", 90), ("2024-08-01", 110)),
            "": (("2018-01-01", 90), ("2025-01-01", 110)),
        }
    )

    # Штрафы — в МРП года (административный кодекс РК): размер в
    # МРП и его вес (синтетическое допущение по частым нарушениям).
    mrp_by_year: dict = field(default_factory=lambda: {2024: 3692, 2025: 3932, 2026: 4325})
    fine_mrp: dict = field(default_factory=lambda: {1: 0.10, 2: 0.10, 3: 0.20, 5: 0.25, 10: 0.20, 15: 0.08, 20: 0.07})

    # Связь и интернет — фиксированный тариф клиента (оканчивается
    # на 90) с индексацией провайдера в январе. Прежде сумма была
    # новой каждый месяц, CV 0.25 и 0.19 (аудит 2026-11-05, M3).
    fixed_tariff_bills: tuple = ("telecom", "internet")
    tariff_january_indexation: float = 0.08

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

    # Подписки: регулярные списания, которых у клиента может не
    # быть вовсе, а может быть с десяток.
    subscription_count: tuple = (1, 9)

    # Доля подписок, продлеваемых ночью (0–5 ч): ночное окно
    # биллинга сервиса. Остальные — в час оформления, днём.
    subscription_night_share: float = 0.30
    subscription_median: int = 2_500
    subscription_sigma: float = 0.50
    subscription_stop_share: float = 0.30
    subscription_price_change_share: float = 0.28

    # Через сколько месяцев в среднем подписка начинается (у той,
    # что появляется позже начала окна), кончается и меняет цену:
    # задержки экспоненциальные и от горизонта планирования не
    # зависят (синтетическое допущение; средние близки к прежним
    # равномерным срокам на окне 2024-01 … 2026-09).
    subscription_mean_months: dict = field(
        default_factory=lambda: {"start": 12, "stop": 14, "change": 12}
    )
    subscription_price_change: tuple = (0.08, 0.30)

    # Прейскурант сервисов подписки, ₸ в месяц: тарифы одни для всех
    # клиентов, повышение — одной датой у сервиса (дата, множитель).
    # Цены — синтетическое приближение к казахстанским тарифам
    # 2024–2025 (уверенность низкая: официальных прайсов research не
    # нашёл). Прежде цена разыгрывалась у каждого клиента: у сервиса
    # было 114–151 разная цена, NETFLIX от 550 до 10 650 ₸ (аудит
    # 2026-11-05, C1).
    subscription_tariffs: dict = field(
        default_factory=lambda: {
            "ЯНДЕКС ПЛЮС": {"tiers": (1290, 1990), "raise": ("2025-04-01", 1.16)},
            "КИНОПОИСК": {"tiers": (1290, 1990), "raise": ("2025-04-01", 1.16)},
            "ЯНДЕКС МУЗЫКА": {"tiers": (990,), "raise": ("2025-04-01", 1.10)},
            "SPOTIFY": {"tiers": (1290, 1690, 2090), "raise": ("2025-07-01", 1.08)},
            "YOUTUBE PREMIUM": {"tiers": (1690, 2490), "raise": None},
            "NETFLIX": {"tiers": (3490, 4990, 6490), "raise": ("2025-01-01", 1.10)},
            "APPLE MUSIC": {"tiers": (1190, 1790), "raise": None},
            "APPLE TV": {"tiers": (1790,), "raise": None},
            "MEGOGO": {"tiers": (1490, 2990), "raise": ("2025-02-01", 1.10)},
            "IVI": {"tiers": (1190,), "raise": None},
            "APP STORE": {"tiers": (490, 990, 1990, 4990), "raise": None},
            "GOOGLE PLAY": {"tiers": (490, 990, 1990, 4990), "raise": None},
        }
    )
