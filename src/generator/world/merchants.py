from __future__ import annotations

from dataclasses import dataclass

from .. import params as params_module
from ..rng import NS_MERCHANT, keyed_rng, stable_hash, state_cache
from . import geography
from .dictionaries import CATEGORY_BY_NAME, CATEGORY_NAMES, Category


# ============================================================
# МЕРЧАНТЫ
# ============================================================
#
# Три РАЗНЫЕ сущности:
#
#   сеть (merchant)  бренд, юридическое имя, ценовой сегмент
#   точка (outlet)   конкретный адрес, район, часы, канал
#   MCC              код категории точки
#
# Популярность сетей и точек имеет тяжёлый хвост: несколько
# сетей собирают большую часть оборота, а десятки тысяч точек
# встречаются считанные разы.
#
# Каталог процедурный: атрибуты точки считаются из её индекса,
# поэтому в памяти лежат только диапазоны, а не сотни тысяч
# объектов.
# ============================================================


_SYL_A = (
    "Ak", "Kok", "Zhas", "Bay", "Tan", "Nur", "Ala", "Sar", "Kaz", "Dan",
    "Er", "Sun", "Arg", "Tem", "Zhet", "Ulan", "Bek", "Mer", "Kus", "Tar",
    "Shan", "Alt", "Kyz", "Tal", "Bur", "Esil", "Ora", "Sil", "Zhar", "Ken",
)

_SYL_B = (
    "tau", "su", "kol", "bek", "zhan", "gul", "bay", "dar", "man", "sar",
    "ken", "ata", "asyl", "kent", "aral", "ray", "dan", "shy", "burg", "nai",
)

_BRAND_SUFFIX = {
    "retail_food": ("Market", "Mart", "Food", "Store", "Bazar"),
    "food_service": ("Cafe", "Kitchen", "House", "Grill", "Coffee"),
    "health": ("Pharm", "Clinic", "Med", "Lab", "Health"),
    "transport": ("Oil", "Auto", "Service", "Drive", "Motors"),
    "travel": ("Travel", "Tour", "Air", "Voyage", "Trip"),
    "retail_goods": ("Shop", "Style", "Home", "Tech", "Trade"),
    "services": ("Telecom", "Net", "Connect", "Service", "Digital"),
    "education": ("School", "Academy", "Books", "Kids", "Learn"),
    "leisure": ("Club", "Fit", "Beauty", "Cinema", "Play"),
    "ecommerce": ("Shop", "Online", "Market", "Store", "Express"),
    "government": ("Service", "Centre", "Portal"),
    "financial": ("Finance", "Invest", "Assist", "Fund"),
}

_LEGAL_FORMS = ("TOO", "IP", "AO", "KH")

_SEGMENT_ORDER = {"budget": 0, "mid": 1, "premium": 2, "luxury": 3}


@dataclass(frozen=True)
class Brand:
    merchant_id: str
    name: str
    legal_name: str
    category: str
    sector: str
    scope: str
    region: str | None
    price_segment: str
    popularity: float
    is_aggregator: bool


@dataclass(frozen=True)
class Outlet:
    outlet_id: str
    merchant_id: str
    brand: str
    merchant_name: str
    category: str
    subcategory: str
    mcc: str
    sector: str
    settlement: str
    region: str
    settlement_type: str
    district: str
    channel: str
    price_segment: str
    opening_hour: int
    closing_hour: int
    popularity: float
    country: str
    is_online: bool


# ============================================================
# СЕТИ
# ============================================================


def _brand_name(category: Category, index: int, scope: str, region: str | None) -> str:

    rng = keyed_rng(NS_MERCHANT, stable_hash("brand", category.name, scope, region or "", index) % (2 ** 31))

    first = _SYL_A[rng.integers(0, len(_SYL_A))]
    second = _SYL_B[rng.integers(0, len(_SYL_B))]

    suffixes = _BRAND_SUFFIX.get(category.sector, ("Group",))
    suffix = suffixes[rng.integers(0, len(suffixes))]

    return f"{first}{second} {suffix}"


def _zipf_weights(count: int, alpha: float) -> tuple:
    return tuple(1.0 / ((index + 1) ** alpha) for index in range(count))


@state_cache
def _national_brands(category_name: str) -> tuple:

    settings = params_module.active().merchants
    category = CATEGORY_BY_NAME[category_name]

    rng = keyed_rng(NS_MERCHANT, stable_hash("national", category_name) % (2 ** 31))

    low, high = settings.national_brands_per_category
    count = max(1, rng.integers(low, high + 1))

    weights = _zipf_weights(count, settings.brand_zipf_alpha)

    segments = list(settings.price_segment_weights)
    segment_weights = [settings.price_segment_weights[item] for item in segments]

    brands = []

    for index in range(count):

        item_rng = keyed_rng(NS_MERCHANT, stable_hash("national_brand", category_name, index) % (2 ** 31))

        name = _brand_name(category, index, "national", None)
        form = _LEGAL_FORMS[item_rng.integers(0, len(_LEGAL_FORMS))]

        brands.append(
            Brand(
                merchant_id=f"mc_{stable_hash('national', category_name, index) % 10 ** 10:010d}",
                name=name,
                legal_name=f'{form} "{name}"',
                category=category_name,
                sector=category.sector,
                scope="national",
                region=None,
                price_segment=str(item_rng.choice(segments, p=segment_weights)),
                popularity=weights[index],
                is_aggregator=category_name in ("delivery", "marketplace", "taxi", "micromobility"),
            )
        )

    return tuple(brands)


@state_cache
def _regional_brands(category_name: str, region: str) -> tuple:

    settings = params_module.active().merchants
    category = CATEGORY_BY_NAME[category_name]

    rng = keyed_rng(NS_MERCHANT, stable_hash("regional", category_name, region) % (2 ** 31))

    low, high = settings.regional_brands_per_category
    count = max(1, rng.integers(low, high + 1))

    weights = _zipf_weights(count, settings.brand_zipf_alpha)

    segments = list(settings.price_segment_weights)
    segment_weights = [settings.price_segment_weights[item] for item in segments]

    brands = []

    for index in range(count):

        item_rng = keyed_rng(
            NS_MERCHANT, stable_hash("regional_brand", category_name, region, index) % (2 ** 31)
        )

        name = _brand_name(category, index, "regional", region)
        form = _LEGAL_FORMS[item_rng.integers(0, len(_LEGAL_FORMS))]

        brands.append(
            Brand(
                merchant_id=f"mc_{stable_hash('regional', category_name, region, index) % 10 ** 10:010d}",
                name=name,
                legal_name=f'{form} "{name}"',
                category=category_name,
                sector=category.sector,
                scope="regional",
                region=region,
                price_segment=str(item_rng.choice(segments, p=segment_weights)),
                popularity=weights[index] * 0.45,
                is_aggregator=False,
            )
        )

    return tuple(brands)


@state_cache
def _local_brands(category_name: str, settlement_name: str) -> tuple:

    settings = params_module.active().merchants
    category = CATEGORY_BY_NAME[category_name]

    rng = keyed_rng(NS_MERCHANT, stable_hash("local", category_name, settlement_name) % (2 ** 31))

    low, high = settings.local_brands_per_settlement
    count = rng.integers(low, high + 1)

    if count <= 0:
        return ()

    brands = []

    for index in range(count):

        item_rng = keyed_rng(
            NS_MERCHANT, stable_hash("local_brand", category_name, settlement_name, index) % (2 ** 31)
        )

        name = _brand_name(category, index, f"local:{settlement_name}", settlement_name)

        brands.append(
            Brand(
                merchant_id=f"mc_{stable_hash('local', category_name, settlement_name, index) % 10 ** 10:010d}",
                name=name,
                legal_name=f'IP "{name}"',
                category=category_name,
                sector=category.sector,
                scope="local",
                region=None,
                price_segment="budget" if item_rng.random() < 0.7 else "mid",
                popularity=0.20 / (index + 1),
                is_aggregator=False,
            )
        )

    return tuple(brands)


def brands_for(category_name: str, settlement: geography.Settlement) -> tuple:
    """
    Сети, доступные в поселении: национальные, региональные
    и локальные.
    """

    settings = params_module.active().merchants

    allowed = set(settings.segment_availability[settlement.settlement_type])

    pool = (
        _national_brands(category_name)
        + _regional_brands(category_name, settlement.region)
        + _local_brands(category_name, settlement.name)
    )

    filtered = tuple(brand for brand in pool if brand.price_segment in allowed)

    if filtered:
        return filtered

    # В маленьком поселении доступных сегментов может не
    # оказаться вовсе. Тогда остаётся самый дешёвый вариант:
    # без единой точки категория просто не существует.
    cheapest = min(pool, key=lambda brand: _SEGMENT_ORDER.get(brand.price_segment, 9)) if pool else None

    return (cheapest,) if cheapest is not None else ()


# ============================================================
# ТОЧКИ
# ============================================================


def available_categories(settlement: geography.Settlement) -> tuple:
    return tuple(
        name
        for name in CATEGORY_NAMES
        if CATEGORY_BY_NAME[name].rural_available or settlement.settlement_type != "rural"
    )


@state_cache
def outlet_count(settlement_name: str, category_name: str) -> int:

    settings = params_module.active()

    settlement = geography.by_name(settlement_name)

    if category_name not in available_categories(settlement):
        return 0

    if not brands_for(category_name, settlement):
        return 0

    low, high = settings.geography.outlets_per_category[settlement.settlement_type]

    rng = keyed_rng(NS_MERCHANT, stable_hash("outlets", settlement_name, category_name) % (2 ** 31))

    base = rng.integers(low, high + 1)

    scaled = max(1, int(round(base * settings.merchants.catalog_scale)))

    return scaled


def _terminal_name(brand: Brand, settlement: geography.Settlement, index: int, rng) -> str:
    """
    Имя в терминальной строке: регистр, транслитерация, номер
    филиала, агрегатор. Шум воспроизводим и привязан к сети.
    """

    settings = params_module.active().merchants

    variants = list(settings.name_variants)
    weights = [settings.name_variants[item] for item in variants]

    variant = str(rng.choice(variants, p=weights))

    base = brand.name

    if variant == "upper":
        text = base.upper()
    elif variant == "title":
        text = base
    elif variant == "translit":
        text = base.upper().replace("Z", "J").replace("KH", "H")
    elif variant == "with_branch":
        text = f"{base.upper()} #{index + 1}"
    else:
        facilitator = settings.facilitators[rng.integers(0, len(settings.facilitators))]
        text = f"{facilitator}*{base.upper()}"

    if rng.random() < settings.branch_number_share and variant != "with_branch":
        text = f"{text} {settlement.name.upper()[:6]}"

    if rng.random() < settings.terminal_noise_share:
        text = text.replace(" ", "")

    return text[:40]


def outlet(settlement_name: str, category_name: str, index: int) -> Outlet:
    """
    Точка по индексу. Атрибуты считаются, а не хранятся.
    """

    settings = params_module.active()

    settlement = geography.by_name(settlement_name)
    category = CATEGORY_BY_NAME[category_name]

    pool = brands_for(category_name, settlement)

    rng = keyed_rng(
        NS_MERCHANT, stable_hash("outlet", settlement_name, category_name, index) % (2 ** 31)
    )

    national = tuple(brand for brand in pool if brand.scope == "national")
    others = tuple(brand for brand in pool if brand.scope != "national")

    national_share = settings.merchants.national_share.get(category_name, 0.4)

    if national and (not others or rng.random() < national_share):
        chosen = national
    else:
        chosen = others or national

    brand = chosen[
        int(rng.choice(len(chosen), p=[item.popularity for item in chosen]))
    ]

    district = settlement.districts[rng.integers(0, len(settlement.districts))]

    hours = settings.geography.opening_hours[settlement.settlement_type]
    opening, closing = hours["standard"]

    if category_name in settings.geography.around_the_clock_categories and rng.random() < 0.35:
        opening, closing = 0, 24
    elif rng.random() < hours["extended_share"]:
        closing = min(24, closing + 2)

    is_online = rng.random() < category.online_share and brand.scope == "national"

    subcategory = category.subcategories[rng.integers(0, len(category.subcategories))]

    mcc = str(rng.choice(category.mccs, p=category.mcc_weights))

    popularity = brand.popularity / ((index + 1) ** settings.merchants.outlet_zipf_alpha)

    return Outlet(
        outlet_id=f"ot_{stable_hash('outlet', settlement_name, category_name, index) % 10 ** 12:012d}",
        merchant_id=brand.merchant_id,
        brand=brand.name,
        merchant_name=_terminal_name(brand, settlement, index, rng),
        category=category_name,
        subcategory=subcategory,
        mcc=mcc,
        sector=category.sector,
        settlement=settlement.name,
        region=settlement.region,
        settlement_type=settlement.settlement_type,
        district="online" if is_online else district,
        channel="ecom" if is_online else "pos",
        price_segment=brand.price_segment,
        opening_hour=0 if is_online else opening,
        closing_hour=24 if is_online else closing,
        popularity=popularity,
        country=settings.geography.home_country,
        is_online=is_online,
    )


@state_cache
def outlets_of(settlement_name: str, category_name: str) -> tuple:
    """
    Все точки категории в поселении.
    """

    count = outlet_count(settlement_name, category_name)

    return tuple(outlet(settlement_name, category_name, index) for index in range(count))


def foreign_outlet(country: str, category_name: str, index: int) -> Outlet:
    """
    Точка за границей: страна известна, город нет.
    """

    category = CATEGORY_BY_NAME[category_name]

    rng = keyed_rng(NS_MERCHANT, stable_hash("foreign", country, category_name, index) % (2 ** 31))

    name = _brand_name(category, index, f"foreign:{country}", country)

    return Outlet(
        outlet_id=f"ot_{stable_hash('foreign', country, category_name, index) % 10 ** 12:012d}",
        merchant_id=f"mc_{stable_hash('foreign_brand', country, category_name, index) % 10 ** 10:010d}",
        brand=name,
        merchant_name=name.upper(),
        category=category_name,
        subcategory=category.subcategories[0],
        mcc=str(rng.choice(category.mccs, p=category.mcc_weights)),
        sector=category.sector,
        settlement="",
        region="",
        settlement_type="foreign",
        district="",
        channel="pos",
        price_segment="mid",
        opening_hour=8,
        closing_hour=23,
        popularity=1.0 / (index + 1),
        country=country,
        is_online=False,
    )


def iter_catalog():
    """
    Полный каталог точек для записи на диск.
    """

    for settlement in geography.settlements():
        for category_name in available_categories(settlement):
            for item in outlets_of(settlement.name, category_name):
                yield item


def catalog_size() -> int:
    return sum(
        outlet_count(settlement.name, category_name)
        for settlement in geography.settlements()
        for category_name in available_categories(settlement)
    )


__all__ = [
    "Brand",
    "Outlet",
    "available_categories",
    "brands_for",
    "catalog_size",
    "foreign_outlet",
    "iter_catalog",
    "outlet",
    "outlet_count",
    "outlets_of",
]
