from __future__ import annotations

from dataclasses import dataclass

from .. import params as params_module
from ..rng import NS_MERCHANT, keyed_rng, stable_hash, state_cache
from . import geography, reference
from .dictionaries import CATEGORY_BY_NAME, CATEGORY_NAMES


# ============================================================
# МЕРЧАНТЫ
# ============================================================
#
# Три РАЗНЫЕ сущности:
#
#   сеть (merchant)  бренд и ценовой сегмент
#   точка (outlet)   конкретный адрес, район, часы, канал
#   MCC              код категории точки
#
# ЕДИНСТВЕННЫЙ источник названий — справочник 2ГИС и
# OpenStreetMap (world.reference). Выдуманных брендов здесь нет
# ни одного: ни по слогам, ни по суффиксам, ни запасным путём.
# Терминальную строку разрешено оформлять по-разному — регистр,
# номер филиала, город, префикс агрегатора, — но название в ней
# всегда принадлежит существующей сети.
#
# Справочник покрывает 19 категорий из 49, поэтому у каждой
# внутренней категории объявлено соответствие. Где разумного
# соответствия нет, точка остаётся БЕЗЫМЯННОЙ: merchant_id и
# merchant_name пусты, а категория, MCC, город и цена на месте.
# Операция при этом происходит: банк видел трату, но имени
# продавца в этой ленте нет.
#
# Популярность сетей и точек имеет тяжёлый хвост: несколько
# сетей собирают большую часть оборота, а десятки тысяч точек
# встречаются считанные разы.
#
# MCC справочник не даёт ни в одном источнике, поэтому код
# категории по-прежнему назначает внутренняя категория.
#
# Точки остаются процедурными: атрибуты считаются из индекса,
# поэтому в памяти лежат диапазоны, а не сотни тысяч объектов.
# ============================================================


# Внутренняя категория -> категория справочника.
#
# Соответствие ставится по смыслу торговли, а не ради
# заполнения: ремонтная мастерская не становится такси, а
# магазин техники — сотовым оператором. None значит, что
# подходящей категории в справочнике нет.
REFERENCE_CATEGORY: dict[str, str | None] = {
    # --- еда ---
    "grocery": "grocery",
    "market": "grocery",            # базар это та же торговля продуктами
    "convenience": "convenience",
    "fastfood": "fastfood",
    "restaurant": "restaurant",
    "coffee": "coffee",
    "delivery": "restaurant",       # доставку везёт ресторан под своим именем
    # --- здоровье ---
    "pharmacy": "pharmacy",
    "medical": "medical",
    "lab": "medical",               # лаборатория это медицинское учреждение
    # --- транспорт ---
    "fuel": "fuel",
    "car_service": "car_service",
    "parking": None,                # парковка не автосервис, и в справочнике её нет
    "taxi": None,                   # перевозка это не ремонт машин
    "transit": None,                # проезд продаёт перевозчик, а не магазин
    "car_rental": None,             # аренда машины это не мастерская
    "micromobility": None,          # самокаты это не спортивный магазин
    # --- путешествия ---
    "airline": None,
    "railway": None,
    "hotel": "hotel",
    "travel": None,                 # турагентство это не гостиница
    # --- товары ---
    "clothing": "clothing",
    "shoes": "clothing",            # обувь продают те же сети одежды
    "home_goods": "home_goods",
    "furniture": "home_goods",      # мебельные магазины лежат в справочнике здесь
    "electronics": "electronics",
    "appliances": "electronics",    # бытовую технику продают те же сети
    # --- услуги ---
    "telecom": None,                # оператор связи это не магазин техники
    "internet": None,               # провайдер это не магазин техники
    "subscription": None,           # подписка на сервис это не розница
    "utilities": None,              # коммунальные платежи принимает не магазин
    # --- образование ---
    "education": "education",
    "kids": None,                   # детские товары это не книжный
    "books": "books",
    # --- досуг ---
    "entertainment": None,          # парк или боулинг это не кинотеатр
    "cinema": "cinema",
    "sports": "sports",
    "beauty": "beauty",
    "cosmetics": "beauty",          # магазины косметики справочник относит сюда
    # --- онлайн ---
    "marketplace": None,            # маркетплейс это не магазин техники
    "ecom": None,                   # интернет-магазин своей категории не имеет
    # --- государство ---
    "government": None,
    "fines": None,
    "taxes": None,
    # --- финансы ---
    "financial": None,
    "charity": None,                # благотворительность это не финансовая сеть
    # --- прочее ---
    "pets": "pets",
    "tobacco": "convenience",       # табачный киоск это тот же киоск
    "gambling": None,
}

assert set(REFERENCE_CATEGORY) == set(CATEGORY_NAMES), (
    "соответствие категорий справочнику разошлось с категориями генератора: "
    f"{sorted(set(REFERENCE_CATEGORY) ^ set(CATEGORY_NAMES))}"
)


_SEGMENT_ORDER = {"budget": 0, "mid": 1, "premium": 2, "luxury": 3}


@dataclass(frozen=True)
class Brand:
    merchant_id: str
    name: str
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
    # Сети у точки может не быть вовсе: категории нет в
    # справочнике, и выдумывать ей имя нечем.
    merchant_id: str | None
    brand: str | None
    merchant_name: str | None
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


@state_cache
def _region_names(source_category: str, region: str) -> tuple:
    """
    Имена категории справочника, встреченные в поселениях
    региона.
    """

    seen: list[str] = []

    for settlement in geography.settlements():

        if settlement.region != region:
            continue

        for name in reference.names_in(settlement.name, source_category):
            if name not in seen:
                seen.append(name)

    return tuple(seen)


def _reference_pool(category_name: str, scope: str, key: str) -> tuple:
    """
    Имена справочника, подходящие сети этого охвата.

    Голова списка по частоте — это и есть национальные сети:
    «Magnum» встречается в справочнике сотни раз. Региональной
    и местной сети достаётся то, что встречено именно там, а
    если там не встречено ничего — хвост списка категории со
    сдвигом по ключу, чтобы соседние места не получили одно имя.
    """

    source = REFERENCE_CATEGORY[category_name]

    if source is None:
        return ()

    catalog = reference.names_of_category(source)

    if not catalog:
        return ()

    head_size = min(len(catalog), params_module.active().merchants.national_brands_per_category[1])

    head = catalog[:head_size]

    if scope == "national":
        return head

    nearby = (
        reference.names_in(key, source)
        if scope == "local"
        else _region_names(source, key)
    )

    local = tuple(name for name in nearby if name not in set(head))

    if local:
        return local

    tail = catalog[head_size:] or catalog

    offset = stable_hash(scope, category_name, key) % len(tail)

    return tail[offset:] + tail[:offset]


def _zipf_weights(count: int, alpha: float) -> tuple:
    return tuple(1.0 / ((index + 1) ** alpha) for index in range(count))


@state_cache
def _national_brands(category_name: str) -> tuple:

    settings = params_module.active().merchants
    category = CATEGORY_BY_NAME[category_name]

    pool = _reference_pool(category_name, "national", "")

    # Категории нет в справочнике: сетей у неё не будет, а точки
    # останутся безымянными.
    if not pool:
        return ()

    rng = keyed_rng(NS_MERCHANT, stable_hash("national", category_name) % (2 ** 31))

    low, high = settings.national_brands_per_category
    count = min(max(1, rng.integers(low, high + 1)), len(pool))

    weights = _zipf_weights(count, settings.brand_zipf_alpha)

    segments = list(settings.price_segment_weights)
    segment_weights = [settings.price_segment_weights[item] for item in segments]

    brands = []

    for index in range(count):

        item_rng = keyed_rng(NS_MERCHANT, stable_hash("national_brand", category_name, index) % (2 ** 31))

        name = pool[index % len(pool)]

        brands.append(
            Brand(
                merchant_id=f"mc_{stable_hash('national', category_name, index) % 10 ** 10:010d}",
                name=name,
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

    pool = _reference_pool(category_name, "regional", region)

    if not pool:
        return ()

    rng = keyed_rng(NS_MERCHANT, stable_hash("regional", category_name, region) % (2 ** 31))

    low, high = settings.regional_brands_per_category
    count = min(max(1, rng.integers(low, high + 1)), len(pool))

    weights = _zipf_weights(count, settings.brand_zipf_alpha)

    segments = list(settings.price_segment_weights)
    segment_weights = [settings.price_segment_weights[item] for item in segments]

    brands = []

    for index in range(count):

        item_rng = keyed_rng(
            NS_MERCHANT, stable_hash("regional_brand", category_name, region, index) % (2 ** 31)
        )

        name = pool[index % len(pool)]

        brands.append(
            Brand(
                merchant_id=f"mc_{stable_hash('regional', category_name, region, index) % 10 ** 10:010d}",
                name=name,
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

    pool = _reference_pool(category_name, "local", settlement_name)

    if not pool:
        return ()

    rng = keyed_rng(NS_MERCHANT, stable_hash("local", category_name, settlement_name) % (2 ** 31))

    low, high = settings.local_brands_per_settlement
    count = min(rng.integers(low, high + 1), len(pool))

    if count <= 0:
        return ()

    brands = []

    for index in range(count):

        item_rng = keyed_rng(
            NS_MERCHANT, stable_hash("local_brand", category_name, settlement_name, index) % (2 ** 31)
        )

        name = pool[index % len(pool)]

        brands.append(
            Brand(
                merchant_id=f"mc_{stable_hash('local', category_name, settlement_name, index) % 10 ** 10:010d}",
                name=name,
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

    low, high = settings.geography.outlets_per_category[settlement.settlement_type]

    rng = keyed_rng(NS_MERCHANT, stable_hash("outlets", settlement_name, category_name) % (2 ** 31))

    base = rng.integers(low, high + 1)

    scaled = max(1, int(round(base * settings.merchants.catalog_scale)))

    return scaled


def _terminal_name(brand: Brand, settlement: geography.Settlement, index: int, rng) -> str:
    """
    Имя в терминальной строке: регистр, транслитерация, номер
    филиала, город, префикс агрегатора. Шум воспроизводим и
    привязан к сети.

    Оформляется ТОЛЬКО существующее название справочника: новых
    брендов здесь не появляется.
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


def _anonymous_segment(settlement: geography.Settlement, rng) -> str:
    """
    Ценовой сегмент точки без сети.

    Сегмент задаёт сумму покупки, поэтому он нужен и безымянной
    точке. Берётся из тех же весов, что у сетей, но только среди
    сегментов, доступных этому типу поселения.
    """

    settings = params_module.active().merchants

    allowed = [
        name
        for name in settings.price_segment_weights
        if name in settings.segment_availability[settlement.settlement_type]
    ]

    if not allowed:
        return min(settings.price_segment_weights, key=lambda name: _SEGMENT_ORDER.get(name, 9))

    weights = [settings.price_segment_weights[name] for name in allowed]
    total = sum(weights)

    return str(rng.choice(allowed, p=[value / total for value in weights]))


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

    # Сетей может не быть: категории нет в справочнике. Тогда
    # точка безымянная — торговля состоялась, а имени продавца
    # в ленте нет.
    brand: Brand | None = None

    if pool:

        national = tuple(item for item in pool if item.scope == "national")
        others = tuple(item for item in pool if item.scope != "national")

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

    # Без сети масштаба у точки нет, и онлайн-витриной её делает
    # только сама категория.
    is_online = rng.random() < category.online_share and (
        brand is None or brand.scope == "national"
    )

    subcategory = category.subcategories[rng.integers(0, len(category.subcategories))]

    mcc = str(rng.choice(category.mccs, p=category.mcc_weights))

    base_popularity = brand.popularity if brand is not None else 1.0

    popularity = base_popularity / ((index + 1) ** settings.merchants.outlet_zipf_alpha)

    return Outlet(
        outlet_id=f"ot_{stable_hash('outlet', settlement_name, category_name, index) % 10 ** 12:012d}",
        merchant_id=brand.merchant_id if brand is not None else None,
        brand=brand.name if brand is not None else None,
        merchant_name=_terminal_name(brand, settlement, index, rng) if brand is not None else None,
        category=category_name,
        subcategory=subcategory,
        mcc=mcc,
        sector=category.sector,
        settlement=settlement.name,
        region=settlement.region,
        settlement_type=settlement.settlement_type,
        district="online" if is_online else district,
        channel="ecom" if is_online else "pos",
        price_segment=brand.price_segment if brand is not None else _anonymous_segment(settlement, rng),
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
    Точка за границей: страна известна, город и сеть нет.

    Справочник описывает только Казахстан, поэтому заграничная
    покупка остаётся без имени продавца: придумывать его нечем.
    Категория, MCC и страна при этом известны.
    """

    category = CATEGORY_BY_NAME[category_name]

    rng = keyed_rng(NS_MERCHANT, stable_hash("foreign", country, category_name, index) % (2 ** 31))

    return Outlet(
        outlet_id=f"ot_{stable_hash('foreign', country, category_name, index) % 10 ** 12:012d}",
        merchant_id=None,
        brand=None,
        merchant_name=None,
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


# ============================================================
# ТОЧКА В СОБЫТИИ
# ============================================================
#
# Наружу выходят шесть полей выбранной точки и только они.
# Остальное — сектор, подкатегория, район, часы, ценовой
# сегмент, идентификатор самой точки — остаётся внутри
# генератора: справочник мерчантов не выгружается.
# ============================================================

MERCHANT_PAYLOAD_FIELDS: tuple[str, ...] = (
    "merchant_id",
    "merchant_name",
    "merchant_category",
    "merchant_city",
    "merchant_country",
    "mcc",
)


def counterpart(item: Outlet | None) -> str:
    """
    Сторона проводки для точки.

    У безымянной точки сети нет, и назвать контрагента нечем:
    проводка уходит на общую внешнюю сторону.
    """

    if item is None or not item.merchant_id:
        return "external:merchant"

    return f"merchant:{item.merchant_id}"


def payload_fields(item: Outlet | None) -> dict:
    """
    Поля выбранной точки для payload. Точки нет — все шесть
    пусты, выдумывать нечего.
    """

    if item is None:
        return {name: None for name in MERCHANT_PAYLOAD_FIELDS}

    return {
        "merchant_id": item.merchant_id,
        "merchant_name": item.merchant_name,
        "merchant_category": item.category,
        # Точка сельской территории города не называет: корзина
        # области это не населённый пункт. За границей город
        # неизвестен по той же причине — его там никто не писал.
        "merchant_city": _public_city(item),
        "merchant_country": item.country,
        "mcc": item.mcc,
    }


def _public_city(item: Outlet) -> str | None:

    if not item.settlement or item.settlement_type == "rural":
        return None

    return item.settlement


__all__ = [
    "MERCHANT_PAYLOAD_FIELDS",
    "REFERENCE_CATEGORY",
    "Brand",
    "Outlet",
    "available_categories",
    "brands_for",
    "counterpart",
    "foreign_outlet",
    "outlet",
    "outlet_count",
    "outlets_of",
    "payload_fields",
]
