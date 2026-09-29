from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

import pytest

from src.generator import config, emit
from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.world import geography, merchants, reference


# ============================================================
# ИДЕЯ
# ============================================================
#
# Точка получает название из ближайшего места, где справочник его
# подтвердил: само поселение, крупнейший город области,
# общенациональный список (записи без города и сервисы
# reference/national_merchants.json). Проверяется, что:
#
#   - уровни не смешиваются: есть свои названия — только они;
#   - сельская корзина берёт названия города своей области;
#   - где ни город, ни область их не дали — общенациональные;
#   - запись с городом вне географии генератора в общенациональный
#     список не попадает: её присутствие подтверждено в другом месте;
#   - категории без соответствия и сервисы без записи для этого
#     места остаются безымянными;
#   - название запасного уровня не сдвигает прочие атрибуты точки:
#     район, часы, онлайн и MCC те же, что у неё безымянной.
# ============================================================


@pytest.fixture(autouse=True)
def world():

    saved = (config.HISTORY_START, config.HISTORY_END, config.REGISTRATION_END)

    config.activate_horizon(datetime(2024, 1, 1), datetime(2024, 7, 1))

    settings = emit._build_params(None, None, 4)
    params_module.activate(settings)
    rng_module.configure(1, settings.fingerprint(), 42)

    yield

    config.activate_horizon(*saved)
    rng_module.clear_caches()


def names(category: str, settlement: str) -> set[str]:
    return {brand.name for brand in merchants.brands_for(category, settlement)}


def test_confirmed_local_names_are_the_only_ones_used():

    local = set(reference.names_in("Almaty", "coffee"))

    assert local
    assert names("coffee", "Almaty") <= local


def test_a_rural_basket_takes_the_names_of_its_regions_city():

    rural = geography.by_name("Aktobe rural")

    assert not reference.names_in(rural.name, "coffee")
    assert names("coffee", rural.name) <= set(reference.names_in("Aktobe", "coffee"))


def national_only() -> tuple[str, str]:
    """
    Пара (категория генератора, поселение), где ни само поселение,
    ни его область названий не дали, а общенациональные есть.
    """

    for category, source in sorted(merchants.REFERENCE_CATEGORY.items()):

        if source is None or not reference.national_names(source):
            continue

        for item in geography.settlements():
            if not reference.names_in(item.name, source) and not merchants._regional_names(item, source):
                if merchants.outlet_count(item.name, category):
                    return category, item.name

    raise AssertionError("нет места, где работал бы только общенациональный список")


def test_without_a_city_or_region_the_national_list_is_used():

    category, place = national_only()

    national = set(reference.national_names(merchants.REFERENCE_CATEGORY[category]))

    assert names(category, place) and names(category, place) <= national


def test_a_record_confirmed_in_an_unknown_town_is_not_national():

    entries = reference.entries()

    national = {(item.category, item.name) for item in entries if item.national}
    unknown = [item for item in entries if item.settlement is None and not item.national]

    assert unknown, "в справочнике есть города вне географии генератора (Зачаганск и др.)"

    # Название такой записи попадает в список, только если то же
    # название есть и у записи без города.
    for item in unknown:
        if (item.category, item.name) not in national:
            assert item.name not in reference.national_names(item.category), item


def test_new_categories_are_named_from_the_national_services():

    # Общенациональные агрегаторы такси есть в списке без города.
    assert {"Yandex Go", "inDrive"} <= set(reference.national_names("taxi"))

    # Где своих названий нет ни в городе, ни в области, — только они.
    bare = [
        item for item in geography.settlements()
        if not reference.names_in(item.name, "taxi") and not merchants._regional_names(item, "taxi")
    ]
    assert bare
    assert names("taxi", bare[0].name) == set(reference.national_names("taxi"))

    assert {"АлматыЭнергоСбыт", "Алматы Су"} <= names("utilities", "Almaty")


def test_a_city_service_does_not_leak_into_another_region():

    # Коммунальные предприятия городские и общенационального списка
    # у них нет: там, где ни город, ни область записи не дали, точка
    # безымянная, а не с чужим поставщиком.
    assert not reference.national_names("utilities")

    bare = [
        item for item in geography.settlements()
        if not reference.names_in(item.name, "utilities")
        and not merchants._regional_names(item, "utilities")
    ]

    assert bare
    assert all(names("utilities", item.name) == set() for item in bare)


def test_categories_without_a_counterpart_stay_unnamed():

    for category in ("fines", "taxes", "gambling"):
        assert merchants.REFERENCE_CATEGORY[category] is None
        assert names(category, "Almaty") == set()


def test_every_national_service_belongs_to_a_mapped_category():

    payload = json.loads(config.NATIONAL_MERCHANTS_PATH.read_text(encoding="utf-8"))

    mapped = {value for value in merchants.REFERENCE_CATEGORY.values() if value}

    assert payload["sources"]
    assert {row["mapped_category"] for row in payload["merchants"]} <= mapped


def test_a_fallback_name_does_not_shift_the_other_outlet_attributes(monkeypatch):

    category, place = national_only()

    named = [merchants.outlet(place, category, index) for index in range(6)]

    monkeypatch.setattr(reference, "national_names", lambda category: ())
    monkeypatch.setattr(merchants, "_regional_names", lambda settlement, source: ())
    rng_module.clear_caches()

    anonymous = [merchants.outlet(place, category, index) for index in range(6)]

    assert all(item.merchant_name for item in named)
    assert not any(item.merchant_name for item in anonymous)

    kept = ("outlet_id", "district", "channel", "opening_hour", "closing_hour", "is_online", "mcc", "subcategory")

    for one, other in zip(named, anonymous):
        assert {name: asdict(one)[name] for name in kept} == {name: asdict(other)[name] for name in kept}
