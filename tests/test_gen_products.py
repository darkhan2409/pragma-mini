from __future__ import annotations

import copy
from datetime import date, datetime

import pytest
import yaml

from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.config import HISTORY_END, REGISTRY_START
from src.generator.world import hcb_timeline, products as products_module
from src.generator.world.hcb_timeline import Period, TimelineError


# ============================================================
# ПРОДУКТОВЫЙ КАТАЛОГ ВО ВРЕМЕНИ
# ============================================================
#
# Хронология Home Credit лежит в data/reference и считается
# фактом. Python её только загружает и проверяет; ни одной даты
# и ни одного тарифа он не придумывает.
# ============================================================


@pytest.fixture(scope="module")
def timeline():
    return hcb_timeline.load()


@pytest.fixture(scope="module")
def catalog():
    params_module.activate(params_module.DEFAULT)
    rng_module.configure(42, params_module.DEFAULT.fingerprint())
    return products_module.catalog()


@pytest.fixture(scope="module")
def raw_yaml():
    return yaml.safe_load(hcb_timeline.PRODUCT_TIMELINE_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# ЗАГРУЗКА И ВАЛИДАЦИЯ
# ------------------------------------------------------------


def test_timeline_loads(timeline):

    assert timeline.products
    assert timeline.sha256

    codes = [record.product_code for record in timeline.products]

    assert len(codes) == len(set(codes))

    for record in timeline.products:
        assert record.versions, record.product_code


def test_no_synthetic_codes_in_confirmed_timeline(timeline):

    for record in timeline.products:
        assert not record.product_code.startswith("SYNTH_")


def test_month_is_never_turned_into_a_day(timeline):
    """
    Если известен только месяц или год, период остаётся
    периодом. Точная дата берётся отдельной колонкой.
    """

    for record in timeline.products:
        for milestone in record.milestones:

            period = milestone.period

            if period.precision == "day":
                assert period.start == period.end
                continue

            if period.precision == "month":
                assert period.start is not None and period.end is not None
                assert period.start != period.end
                assert period.start.day == 1

            if period.precision == "year":
                assert period.start is not None and period.end is not None
                assert (period.start.month, period.start.day) == (1, 1)


def test_ozen_pilot_is_a_month_and_home_card_is_a_year(timeline):

    ozen = timeline.by_code("OZEN")

    pilot = ozen.milestone("pilot_start")

    assert pilot.period.precision == "month"
    assert pilot.period.start == date(2023, 9, 1)
    assert pilot.period.end == date(2023, 9, 30)

    home_card = timeline.by_code("HOME_CARD")

    start = home_card.milestone("sales_start")

    assert start.period.precision == "year"
    assert start.period.start == date(2019, 1, 1)


def test_missing_source_requires_low_confidence(raw_yaml):

    raw = copy.deepcopy(raw_yaml)

    victim = raw["products"][0]["milestones"][0]

    victim["source"] = None
    victim["confidence"] = "high"
    victim.pop("unresolved_source", None)

    with pytest.raises(TimelineError, match="source"):
        hcb_timeline.parse(raw, "sha")


def test_filled_source_forbids_unresolved_flag(raw_yaml):

    raw = copy.deepcopy(raw_yaml)

    victim = raw["products"][0]["milestones"][0]

    victim["source"] = "https://home.kz/about"
    victim["unresolved_source"] = True

    with pytest.raises(TimelineError, match="unresolved_source"):
        hcb_timeline.parse(raw, "sha")


def test_unresolved_sources_are_listed(timeline):

    rows = timeline.unresolved_sources()

    for row in rows:
        assert row["confidence"] == "low"
        assert row["product_code"]

    # Отчёт реализма обязан показать их как требующие
    # подтверждения, а не спрятать.
    assert isinstance(rows, list)


def test_bad_precision_is_rejected(raw_yaml):

    raw = copy.deepcopy(raw_yaml)

    raw["products"][0]["milestones"][0]["date_precision"] = "quarter"

    with pytest.raises(TimelineError, match="date_precision"):
        hcb_timeline.parse(raw, "sha")


def test_reversed_period_is_rejected(raw_yaml):

    raw = copy.deepcopy(raw_yaml)

    victim = raw["products"][0]["milestones"][0]

    victim["date_precision"] = "interval"
    victim["known_period_start"] = "2025-01-01"
    victim["known_period_end"] = "2024-01-01"

    with pytest.raises(TimelineError, match="позже"):
        hcb_timeline.parse(raw, "sha")


def test_synthetic_code_in_yaml_is_rejected(raw_yaml):

    raw = copy.deepcopy(raw_yaml)

    raw["products"][0]["product_code"] = "SYNTH_SOMETHING"

    with pytest.raises(TimelineError, match="SYNTH_"):
        hcb_timeline.parse(raw, "sha")


def test_duplicate_codes_are_rejected(raw_yaml):

    raw = copy.deepcopy(raw_yaml)

    raw["products"].append(copy.deepcopy(raw["products"][0]))

    with pytest.raises(TimelineError):
        hcb_timeline.parse(raw, "sha")


def test_unknown_successor_is_rejected(raw_yaml):

    raw = copy.deepcopy(raw_yaml)

    raw["products"][0]["successor"] = "NO_SUCH_PRODUCT"

    with pytest.raises(TimelineError, match="successor"):
        hcb_timeline.parse(raw, "sha")


# ------------------------------------------------------------
# ДАТА ДЛЯ СИМУЛЯЦИИ
# ------------------------------------------------------------


def test_simulation_date_follows_the_policy():

    period = Period(start=date(2023, 9, 1), end=date(2023, 9, 30), precision="month")

    assert products_module.simulation_date(period, "period_start", REGISTRY_START) == datetime(2023, 9, 1)
    assert products_module.simulation_date(period, "period_end", REGISTRY_START) == datetime(2023, 9, 30)

    middle = products_module.simulation_date(period, "period_middle", REGISTRY_START)

    assert datetime(2023, 9, 1) < middle < datetime(2023, 9, 30)


def test_unknown_period_falls_back_to_registry_start():

    period = Period(start=None, end=date(2026, 9, 16), precision="unknown")

    assert products_module.simulation_date(period, "period_start", REGISTRY_START) == REGISTRY_START


def test_period_stays_next_to_the_chosen_day(catalog):

    for row in catalog.rows:
        if row.sales_start is None:
            continue
        assert row.sales_start_at is not None
        if row.sales_start.precision == "day":
            assert row.sales_start_at.date() == row.sales_start.start


# ------------------------------------------------------------
# СТАТУСЫ
# ------------------------------------------------------------


def test_statuses_come_from_dates_only(catalog):

    ozen = catalog.view("OZEN")

    assert ozen.status_at(datetime(2023, 1, 1)) == "planned"
    assert ozen.status_at(datetime(2023, 9, 15)) == "pilot"
    assert ozen.status_at(datetime(2024, 6, 1)) == "active"
    assert ozen.sellable_at(datetime(2024, 6, 1))


def test_sales_end_does_not_close_contracts(catalog):
    """
    Закрытие продаж снимает продукт с полки и со временем
    переводит его на обслуживание. Действующие договоры живут
    в обоих статусах.
    """

    view = catalog.view("DEPOSIT_AKCIONNY")

    right_after = datetime(2025, 8, 1)

    assert view.status_at(right_after) == "closed_to_new_clients"
    assert view.serviced_at(right_after)
    assert not view.sellable_at(right_after)

    later = datetime(2026, 8, 1)

    assert view.status_at(later) == "servicing_existing_contracts"
    assert view.serviced_at(later)
    assert not view.sellable_at(later)


def test_archived_is_terminal(catalog):

    legacy = catalog.view("APP_LEGACY")

    assert legacy.status_at(datetime(2025, 2, 1)) == "archived"
    assert legacy.status_at(HISTORY_END) == "archived"
    assert not legacy.serviced_at(HISTORY_END)


def test_status_does_not_depend_on_contracts_in_the_sample(catalog):
    """
    Статус вычисляется из вех продукта. Ни одна выборка
    клиентов в него не входит.
    """

    for view in catalog.views.values():

        spans = view.status_spans

        assert spans

        starts = [start for start, _ in spans]

        assert starts == sorted(starts)

        if any(status == "archived" for _, status in spans):
            assert spans[-1][1] == "archived"


def test_suspension_and_resumption_are_visible(catalog):

    view = catalog.view("DEPOSIT_AKCIONNY")

    assert view.sellable_at(datetime(2024, 8, 1))
    assert not view.sellable_at(datetime(2024, 11, 15))
    assert view.sellable_at(datetime(2025, 1, 15))
    assert not view.sellable_at(datetime(2025, 8, 1))


# ------------------------------------------------------------
# ВЕРСИИ
# ------------------------------------------------------------


def test_version_at_is_the_version_on_that_day(catalog):

    view = catalog.view("DEPOSIT_HOOM")

    early = view.version_at(datetime(2024, 3, 1))
    late = view.version_at(datetime(2025, 1, 1))

    assert early.tariff_version <= late.tariff_version
    assert early.terms != late.terms or early.tariff_version != late.tariff_version


def test_version_is_monotone_in_time(catalog):

    for view in catalog.views.values():

        starts = [start for start, _ in view.version_starts]

        assert starts == sorted(starts), view.code

        versions = [item.tariff_version for _, item in view.version_starts]

        assert versions == sorted(versions), view.code


def test_later_versions_are_only_in_the_future(catalog):

    view = catalog.view("DEPOSIT_HOOM")

    moment = datetime(2024, 9, 1)

    for version in view.later_versions(moment):
        assert view.version_start(version) > moment


def test_applies_to_is_declared(catalog):

    for row in catalog.rows:
        assert row.applies_to in hcb_timeline.APPLIES_TO


# ------------------------------------------------------------
# ПРАВИЛА ВЛАДЕНИЯ
# ------------------------------------------------------------


def test_holding_rules_are_present(catalog):

    for row in catalog.rows:
        assert isinstance(row.allow_multiple, bool)
        assert row.max_active_holdings >= 1
        assert isinstance(row.compatibility_rules, dict)
        assert isinstance(row.replacement_rules, dict)


def test_exclusive_cards_know_each_other(catalog):

    aspan = catalog.view("ASPAN")

    exclusive = aspan.rows[0].compatibility_rules.get("exclusive_with", ())

    assert "ALEM" in tuple(exclusive)


# ------------------------------------------------------------
# СИНТЕТИЧЕСКИЕ ПРОДУКТЫ
# ------------------------------------------------------------


def test_synthetic_products_are_separate(catalog):

    synthetic = [row for row in catalog.rows if row.is_synthetic]

    assert synthetic

    for row in synthetic:
        assert row.product_code.startswith("SYNTH_")
        assert row.confidence == "synthetic"

    for row in catalog.rows:
        if not row.is_synthetic:
            assert not row.product_code.startswith("SYNTH_")
            assert row.confidence in hcb_timeline.CONFIDENCES


def test_synthetic_dates_do_not_touch_real_products(catalog):
    """
    Вымышленный запуск не имеет права переписать дату или тариф
    настоящего продукта.
    """

    params_module.activate(params_module.DEFAULT)

    reference = {
        (row.product_code, row.valid_from): (row.status, row.tariff_version)
        for row in products_module.catalog().rows
        if not row.is_synthetic
    }

    changed = params_module.DEFAULT.with_overrides(
        {"products": {"synthetic_products": ()}}
    )

    params_module.activate(changed)
    rng_module.configure(42, changed.fingerprint())

    without = products_module.catalog()

    assert not [row for row in without.rows if row.is_synthetic]

    after = {
        (row.product_code, row.valid_from): (row.status, row.tariff_version)
        for row in without.rows
    }

    assert after == reference

    params_module.activate(params_module.DEFAULT)
    rng_module.configure(42, params_module.DEFAULT.fingerprint())


def test_synthetic_successor_chain(catalog):

    codes = {row.product_code for row in catalog.rows if row.is_synthetic}

    assert "SYNTH_LOAN_GREEN" in codes

    view = catalog.view("SYNTH_LOAN_GREEN")

    assert view.record.successor == "SYNTH_LOAN_GREEN_2"
    assert view.record.migration_policy in hcb_timeline.MIGRATION_POLICIES


# ------------------------------------------------------------
# КАТАЛОГ КАК ТАБЛИЦА
# ------------------------------------------------------------


def test_rows_are_contiguous_in_time(catalog):

    by_code: dict[str, list] = {}

    for row in catalog.rows:
        by_code.setdefault(row.product_code, []).append(row)

    for code, rows in by_code.items():

        rows.sort(key=lambda item: item.valid_from)

        for left, right in zip(rows, rows[1:]):
            assert left.valid_to == right.valid_from, code
            assert left.valid_from < left.valid_to, code


def test_horizon_rows_intersect_the_window():

    rows = products_module.horizon_rows()

    assert rows

    for row in rows:
        assert row.valid_to > REGISTRY_START
        assert row.valid_from < HISTORY_END
