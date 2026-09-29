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
# подтвердил: само поселение, крупнейший город области, точки без
# города. К этому уровню всегда добавляются общенациональные сервисы
# (reference/national_merchants.json без города). Проверяется, что:
#
#   - уровни мест не смешиваются: есть свои названия — только они
#     (и сервисы);
#   - сельская корзина берёт названия города своей области;
#   - где ни город, ни область их не дали — точки без города;
#   - запись с городом вне географии генератора в запасной уровень
#     не попадает: её присутствие подтверждено в другом месте;
#   - сервис есть и рядом с местными названиями (Yandex Go в городе
#     с местным такси);
#   - где пул из одних сервисов, у каждой точки поселения свой
#     сервис: одна подписка не заводится дважды;
#   - у онлайн-точки нет города ни в поле, ни в терминальной строке;
#   - категории без соответствия и городские сервисы вне своего
#     места остаются безымянными;
#   - название не из самого поселения не сдвигает прочие атрибуты
#     точки: район, часы, онлайн и MCC те же, что у неё безымянной.
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


def cityless_only() -> tuple[str, str]:
    """
    Пара (категория генератора, поселение), где ни само поселение,
    ни его область названий не дали, а точки без города есть.
    """

    for category, source in sorted(merchants.REFERENCE_CATEGORY.items()):

        if source is None or not reference.cityless_names(source):
            continue

        for item in geography.settlements():
            if not reference.names_in(item.name, source) and not merchants._regional_names(item, source):
                if merchants.outlet_count(item.name, category):
                    return category, item.name

    raise AssertionError("нет места, где работал бы только запасной уровень")


def test_confirmed_local_names_are_the_only_places_used():

    local = set(reference.names_in("Almaty", "coffee"))

    assert local and not reference.service_names("coffee")
    assert names("coffee", "Almaty") <= local


def test_a_rural_basket_takes_the_names_of_its_regions_city():

    rural = geography.by_name("Aktobe rural")

    assert not reference.names_in(rural.name, "coffee")
    assert names("coffee", rural.name) <= set(reference.names_in("Aktobe", "coffee"))


def test_without_a_city_or_region_the_cityless_points_are_used():

    category, place = cityless_only()
    source = merchants.REFERENCE_CATEGORY[category]

    allowed = set(reference.cityless_names(source)) | set(reference.service_names(source))

    assert names(category, place) and names(category, place) <= allowed


def test_a_record_confirmed_in_an_unknown_town_is_not_a_fallback():

    entries = reference.entries()

    cityless = {(item.category, item.name) for item in entries if item.national and not item.service}
    unknown = [item for item in entries if item.settlement is None and not item.national]

    assert unknown, "в справочнике есть города вне географии генератора (Зачаганск и др.)"

    for item in unknown:
        if (item.category, item.name) not in cityless:
            assert item.name not in reference.cityless_names(item.category), item


def test_national_services_stand_next_to_local_ones():

    services = set(reference.service_names("taxi"))

    assert {"Yandex Go", "inDrive"} <= services

    cities = [item.name for item in geography.settlements() if reference.names_in(item.name, "taxi")]

    assert cities, "где-то подтверждено местное такси (Maxim, APARU…)"

    for city in cities:
        assert names("taxi", city) == set(reference.names_in(city, "taxi")) | services


def test_where_only_services_exist_every_outlet_is_another_service():

    for place in ("Almaty", "Aktobe", "Akkol"):

        pool = names("subscription", place)
        outlets = merchants.outlets_of(place, "subscription")

        assert pool and not merchants._place_names(geography.by_name(place), "subscription")

        taken = [outlet.merchant_id for outlet in outlets]
        assert len(set(taken)) == min(len(outlets), len(pool)), place


def test_an_online_outlet_names_no_city():

    online = [
        outlet
        for place in ("Almaty", "Aktobe", "Taraz")
        for category in ("subscription", "marketplace", "taxi", "grocery")
        for outlet in merchants.outlets_of(place, category)
        if outlet.is_online and outlet.merchant_name
    ]

    assert online

    for outlet in online:
        assert merchants.payload_fields(outlet)["merchant_city"] is None
        assert "#" not in outlet.merchant_name
        assert not outlet.merchant_name.endswith(outlet.settlement.upper()[:6])


def test_a_city_service_does_not_leak_into_another_region():

    # Коммунальные предприятия городские: общенационального у них
    # нет, и там, где ни город, ни область записи не дали, точка
    # безымянная, а не с чужим поставщиком.
    assert not reference.service_names("utilities")

    bare = [
        item for item in geography.settlements()
        if not merchants._place_names(item, "utilities")
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


def test_a_name_from_elsewhere_does_not_shift_the_other_outlet_attributes(monkeypatch):

    category, place = cityless_only()

    named = [merchants.outlet(place, category, index) for index in range(6)]

    monkeypatch.setattr(reference, "cityless_names", lambda category: ())
    monkeypatch.setattr(reference, "service_names", lambda category: ())
    monkeypatch.setattr(merchants, "_regional_names", lambda settlement, source: ())
    rng_module.clear_caches()

    anonymous = [merchants.outlet(place, category, index) for index in range(6)]

    assert all(item.merchant_name for item in named)
    assert not any(item.merchant_name for item in anonymous)

    kept = ("outlet_id", "district", "channel", "opening_hour", "closing_hour", "is_online", "mcc", "subcategory")

    for one, other in zip(named, anonymous):
        assert {name: asdict(one)[name] for name in kept} == {name: asdict(other)[name] for name in kept}
