from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from ..config import MERCHANT_REFERENCE_PATH


# ============================================================
# СПРАВОЧНИК РЕАЛЬНЫХ МЕРЧАНТОВ
# ============================================================
#
# reference/merchants_2gis.json — ВХОД генератора: названия
# точек берутся оттуда, а не собираются из слогов. Наружу
# справочник не выгружается: в событие попадают только поля
# выбранной точки.
#
# Что берётся из записи справочника:
#
#   name                имя точки
#   generator_category  категория; имена совпадают с внутренними
#   city_resolved       город; сопоставляется с поселением
#   source              2GIS или OpenStreetMap
#
# MCC в справочнике нет намеренно: ни 2ГИС, ни OpenStreetMap
# его не публикуют, и он остаётся за внутренней категорией.
#
# Частота имени в справочнике — это и есть масштаб сети:
# «Magnum» встречается сотни раз, сельский магазин один раз.
# Поэтому голова списка по частоте идёт национальным сетям,
# хвост — местным.
# ============================================================


# Кириллица в латиницу. Нужна, чтобы «Алматы» встретилось с
# поселением Almaty: справочник на русском, география генератора
# на латинице.
_TRANSLIT: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ә": "a", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
    "һ": "h", "і": "i",
}

# Города, которые генератор называет иначе, чем справочник:
# транслитерация тут не поможет, потому что различаются сами
# названия, а не их запись. Список закрытый и проверяемый —
# обе стороны каждой пары есть в своих справочниках.
_CITY_ALIASES: dict[str, str] = {
    "Уральск": "Oral",
    "Туркестан": "Turkistan",
    "Петропавловск": "Petropavl",
    "Рудный": "Rudny",
    "Аральск": "Aral",
}


@dataclass(frozen=True)
class ReferenceOutlet:
    name: str
    category: str
    # Поселение генератора; None, если город записи в географии
    # генератора не назван.
    settlement: str | None
    source: str


def _transliterate(text: str) -> str:
    return "".join(_TRANSLIT.get(letter, letter) for letter in text.lower())


def _base_name(name: str, source: str) -> str:
    """
    Имя сети без пояснения. У 2ГИС после запятой идёт тип
    организации («Magnum, супермаркет»), у OpenStreetMap имя
    приходит как есть.
    """

    if source == "2GIS":
        name = name.split(",")[0]

    return " ".join(name.split()).strip()


@lru_cache(maxsize=1)
def _settlement_by_city() -> dict[str, str]:
    """
    Город справочника -> поселение генератора.
    """

    from . import geography

    known = {item.name.lower(): item.name for item in geography.settlements()}

    mapping: dict[str, str] = {}

    for city in _CITY_ALIASES:
        settlement = _CITY_ALIASES[city]
        if settlement.lower() in known:
            mapping[city] = known[settlement.lower()]

    return mapping


def _settlement_of(city: str | None) -> str | None:

    if not city:
        return None

    aliases = _settlement_by_city()

    if city in aliases:
        return aliases[city]

    from . import geography

    known = {item.name.lower(): item.name for item in geography.settlements()}

    return known.get(_transliterate(city))


@lru_cache(maxsize=1)
def outlets() -> tuple[ReferenceOutlet, ...]:
    """
    Справочник целиком. Читается один раз на процесс.
    """

    if not MERCHANT_REFERENCE_PATH.exists():
        return ()

    payload = json.loads(MERCHANT_REFERENCE_PATH.read_text(encoding="utf-8"))

    items: list[ReferenceOutlet] = []

    for row in payload.get("merchants", ()):

        source = row.get("source") or ""
        name = _base_name(row.get("name") or "", source)

        category = row.get("generator_category")

        if not name or not category:
            continue

        items.append(
            ReferenceOutlet(
                name=name,
                category=category,
                settlement=_settlement_of(row.get("city_resolved")),
                source=source,
            )
        )

    return tuple(items)


@lru_cache(maxsize=1)
def _names_by_category() -> dict[str, tuple[str, ...]]:
    """
    Категория -> имена по убыванию частоты в справочнике.
    """

    counts: dict[str, Counter] = defaultdict(Counter)

    for item in outlets():
        counts[item.category][item.name] += 1

    return {
        category: tuple(
            name for name, _ in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
        )
        for category, counter in counts.items()
    }


@lru_cache(maxsize=1)
def _names_by_place() -> dict[tuple[str, str], tuple[str, ...]]:
    """
    (поселение, категория) -> имена по убыванию частоты.
    """

    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for item in outlets():
        if item.settlement is not None:
            counts[(item.settlement, item.category)][item.name] += 1

    return {
        place: tuple(
            name for name, _ in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
        )
        for place, counter in counts.items()
    }


def names_of_category(category: str) -> tuple[str, ...]:
    """
    Все имена категории по всей стране, частые первыми.
    """

    return _names_by_category().get(category, ())


def names_in(settlement: str, category: str) -> tuple[str, ...]:
    """
    Имена категории, встреченные именно в этом поселении.
    """

    return _names_by_place().get((settlement, category), ())


def covered_categories() -> frozenset[str]:
    """
    Категории, для которых в справочнике есть хоть одно имя.
    Для остальных название точки остаётся процедурным.
    """

    return frozenset(_names_by_category())


__all__ = [
    "ReferenceOutlet",
    "covered_categories",
    "names_in",
    "names_of_category",
    "outlets",
]
