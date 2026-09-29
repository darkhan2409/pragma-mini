from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from ..config import MERCHANT_REFERENCE_PATH, NATIONAL_MERCHANTS_PATH


# ============================================================
# СПРАВОЧНИК РЕАЛЬНЫХ НАЗВАНИЙ
# ============================================================
#
# reference/merchants.json — ВХОД генератора: названия точек
# берутся оттуда, а не собираются из слогов. Наружу справочник
# не выгружается: в событие попадает только само название.
#
# Что лежит в записи:
#
#   name             название точки, как его дал источник
#   mapped_category  НАШЕ сопоставление категории генератора;
#                    ни рубрикой источника, ни MCC оно не является
#   city             город, но только если его назвал сам
#                    источник; вычисленный по координатам город
#                    в справочник не попал
#
# Источники названы в шапке файла целиком; у отдельной записи
# ни источника, ни его идентификатора нет — генератору нужно
# само название, а не ссылка на карточку в чужой базе.
#
# Справочник отвечает на вопрос: какие названия ПОДТВЕРЖДЕНЫ в
# этом городе. Ни масштаба сети, ни популярности, ни доли рынка
# из него не выводится: число записей с одним названием — это
# результат выборочного поиска, а не факт о компании.
#
# Запись без города (64% merchants.json) города не подтверждает и
# служит общенациональным списком своей категории: им названы
# точки там, где ни город, ни его область подтверждённых названий
# этой категории не дали (world.merchants.brands_for). Запись,
# чей город в географии генератора не назван, не годится ни туда,
# ни туда: присутствие подтверждено, но в другом месте.
#
# reference/national_merchants.json — второй вход того же формата:
# сервисы для категорий, которых в 2ГИС и OSM нет (такси, связь,
# авиакомпании, маркетплейсы, коммунальные…), собранные вручную с
# источником на каждую компанию.
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
    "Усть-Каменогорск": "Oskemen",
    "Сатпаев": "Satpayev",
    "Байконур": "Baikonur",
    "Тайынша": "Taiynsha",
    "Алтай": "Altai",
    "Аксай": "Aksai",
    # Города, которые генератор отличает от одноимённой области
    # или села словом town.
    "Абай": "Abai town",
    "Аксу": "Aksu town",
    "Есиль": "Esil town",
}


@dataclass(frozen=True)
class ReferenceName:
    """
    Одно подтверждённое название.
    """

    name: str
    category: str
    # Поселение генератора; None, если города у записи нет или
    # он не назван в географии генератора.
    settlement: str | None
    # Города у записи нет вовсе: название общенациональное.
    national: bool = False
    # Запись из national_merchants.json: сервис (такси, связь,
    # подписка…), а не точка из 2ГИС или OSM.
    service: bool = False


def _transliterate(text: str) -> str:
    return "".join(_TRANSLIT.get(letter, letter) for letter in text.lower())


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
def entries() -> tuple[ReferenceName, ...]:
    """
    Справочник целиком. Читается один раз на процесс.
    """

    items: list[ReferenceName] = []

    for path in (MERCHANT_REFERENCE_PATH, NATIONAL_MERCHANTS_PATH):

        if not path.exists():
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))

        for row in payload.get("merchants", ()):

            name = row.get("name") or ""
            category = row.get("mapped_category")

            if not name or not category:
                continue

            items.append(
                ReferenceName(
                    name=name,
                    category=category,
                    settlement=_settlement_of(row.get("city")),
                    national=not row.get("city"),
                    service=path == NATIONAL_MERCHANTS_PATH,
                )
            )

    return tuple(items)


@lru_cache(maxsize=1)
def _names_by_place() -> dict[tuple[str, str], tuple[str, ...]]:
    """
    (поселение, категория) -> подтверждённые там названия.

    Порядок алфавитный и ничего не утверждает: справочник не
    знает, какая сеть крупнее. Какое название достанется какой
    точке, решает генератор своим ключом.
    """

    names: dict[tuple[str, str], set[str]] = defaultdict(set)

    for item in entries():
        if item.settlement is not None:
            names[(item.settlement, item.category)].add(item.name)

    return {place: tuple(sorted(found)) for place, found in names.items()}


@lru_cache(maxsize=None)
def _national_names(service: bool) -> dict[str, tuple[str, ...]]:
    """
    Категория -> названия записей без города: сервисов или точек.
    """

    names: dict[str, set[str]] = defaultdict(set)

    for item in entries():
        if item.national and item.service == service:
            names[item.category].add(item.name)

    return {category: tuple(sorted(found)) for category, found in names.items()}


def cityless_names(category: str) -> tuple[str, ...]:
    """
    Точки 2ГИС и OSM без города: запасной уровень названий там, где
    ни поселение, ни его область подтверждённых названий не дали.
    """

    return _national_names(False).get(category, ())


def service_names(category: str) -> tuple[str, ...]:
    """
    Общенациональные сервисы категории (national_merchants.json без
    города): такси-агрегатор или стриминг есть в любом городе, в том
    числе там, где подтверждены и местные названия.
    """

    return _national_names(True).get(category, ())


def names_in(settlement: str, category: str) -> tuple[str, ...]:
    """
    Названия категории, ПОДТВЕРЖДЁННЫЕ в этом поселении.

    Пустой ответ значит, что подтверждения здесь нет; куда идти
    дальше, решает world.merchants.brands_for.
    """

    return _names_by_place().get((settlement, category), ())


__all__ = [
    "ReferenceName",
    "entries",
    "cityless_names",
    "names_in",
    "service_names",
]
