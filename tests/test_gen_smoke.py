from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.config import (
    CHANGE_INITIATORS,
    REGISTRY_START,
    EVENT_TYPES,
    FORBIDDEN_RAW_FIELDS,
    HISTORY_END,
    HISTORY_START,
    LINK_TYPES,
    PAYLOAD_FIELDS,
    SOURCES,
    TIME_PRECISIONS,
)
from src.generator.emit import generate_dataset
from src.generator.finance import invariants as invariants_module


# ============================================================
# SMOKE
# ============================================================
#
# Маленькая популяция из НЕСКОЛЬКИХ сообществ: иначе сравнение
# одного и трёх воркеров не проверяло бы параллельное
# исполнение вовсе.
# ============================================================

CLIENTS = 24
COMMUNITY_SIZE = 8
CATALOG_SCALE = 0.05


def _emit(out: Path, workers: int, chunk_clients: int, seed: int = 42) -> dict:

    generate_dataset(
        total_clients=CLIENTS,
        out_dir=out,
        seed=seed,
        workers=workers,
        chunk_clients=chunk_clients,
        catalog_scale=CATALOG_SCALE,
        community_size=COMMUNITY_SIZE,
        quiet=True,
    )

    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> dict:

    out = tmp_path_factory.mktemp("raw_smoke")

    manifest = _emit(out, workers=1, chunk_clients=COMMUNITY_SIZE)

    events = pq.read_table(out / "events.parquet").to_pylist()

    for row in events:
        row["payload"] = json.loads(row["payload"])

    return {
        "dir": out,
        "manifest": manifest,
        "events": events,
        "profile": pq.read_table(out / "profile.parquet").to_pylist(),
        "coverage": pq.read_table(out / "source_coverage.parquet").to_pylist(),
        "products": pq.read_table(out / "catalog" / "products.parquet").to_pylist(),
        "merchants": pq.read_table(out / "catalog" / "merchants.parquet").to_pylist(),
        "truth_clients": pq.read_table(out / "truth" / "clients.parquet").to_pylist(),
        "truth_events": pq.read_table(out / "truth" / "events.parquet").to_pylist(),
        "truth_relationships": pq.read_table(out / "truth" / "relationships.parquet").to_pylist(),
    }


# ------------------------------------------------------------
# КОНТРАКТ
# ------------------------------------------------------------


def test_layout_is_complete(dataset):

    out = dataset["dir"]

    for relative in (
        "manifest.json",
        "events.parquet",
        "profile.parquet",
        "source_coverage.parquet",
        "catalog/products.parquet",
        "catalog/merchants.parquet",
        "catalog/geography.parquet",
        "truth/clients.parquet",
        "truth/events.parquet",
        "truth/relationships.parquet",
    ):
        assert (out / relative).exists(), relative


def test_payload_keys_match_the_catalogue(dataset):
    """
    Ключи payload обязаны совпадать с каталогом ключей манифеста.
    """

    catalogue = dataset["manifest"]["key_catalogue"]

    for row in dataset["events"]:

        expected = [item["name"] for item in catalogue[row["event_type"]]["fields"]]

        assert list(row["payload"]) == expected, row["event_type"]
        assert tuple(expected) == PAYLOAD_FIELDS[row["event_type"]]


def test_envelope_is_filled(dataset):

    for row in dataset["events"]:

        assert row["event_id"]
        assert row["event_type"] in EVENT_TYPES
        assert row["source"] in SOURCES
        # Реестр договоров тянется дальше окна наблюдения,
        # остальные источники начинаются вместе с ним.
        floor = REGISTRY_START if row["source"] == "product_events" else HISTORY_START

        assert floor <= row["event_time"] < HISTORY_END
        assert row["record_time"] is not None
        assert row["record_time"] >= row["event_time"]
        assert row["time_precision"] in TIME_PRECISIONS
        assert row["event_version"] >= 1
        assert row["change_initiator"] in CHANGE_INITIATORS
        assert row["link_type"] is None or row["link_type"] in LINK_TYPES


def test_sequence_numbers_are_dense(dataset):

    by_client = defaultdict(list)

    for row in dataset["events"]:
        by_client[row["client_id"]].append(row["sequence_number"])

    for client_id, numbers in by_client.items():
        assert sorted(numbers) == list(range(len(numbers))), client_id


def test_corrections_keep_the_event_id(dataset):
    """
    Исправление сохраняет event_id, повышает версию и получает
    более поздний record_time.
    """

    versions = defaultdict(list)

    for row in dataset["events"]:
        versions[row["event_id"]].append(row)

    corrected = 0

    for rows in versions.values():

        if len(rows) < 2:
            continue

        rows.sort(key=lambda item: (item["event_version"], item["record_time"]))

        for left, right in zip(rows, rows[1:]):
            assert right["record_time"] >= left["record_time"]
            assert right["event_version"] >= left["event_version"]
            if right["event_version"] > left["event_version"]:
                corrected += 1

    assert corrected > 0, "исправлений не оказалось вовсе"


def test_references_resolve(dataset):

    outlets = {row["outlet_id"] for row in dataset["merchants"]}
    merchants = {row["merchant_id"] for row in dataset["merchants"]}

    products = defaultdict(list)

    for row in dataset["products"]:
        products[row["product_code"]].append(row)

    event_ids = {row["event_id"] for row in dataset["events"]}

    for row in dataset["events"]:

        payload = row["payload"]

        if payload.get("outlet_id") and not payload["outlet_id"].startswith("ot_"):
            pytest.fail("странный outlet_id")

        if payload.get("outlet_id") and payload.get("merchant_country") == "KZ":
            assert payload["outlet_id"] in outlets or payload.get("merchant_id") not in merchants

        if payload.get("cause_event_id"):
            assert payload["cause_event_id"] in event_ids

        code = payload.get("product_code")

        if code:
            assert code in products, code


def test_contract_version_is_valid_at_opening(dataset):
    """
    Версия продукта на договоре обязана существовать в каталоге
    и действовать на дату открытия.
    """

    rows = defaultdict(list)

    for row in dataset["products"]:
        rows[row["product_code"]].append(row)

    checked = 0

    for row in dataset["events"]:

        if row["event_type"] != "product_opened":
            continue

        payload = row["payload"]

        matches = [
            item
            for item in rows[payload["product_code"]]
            if item["product_version"] == payload["product_version"]
            and item["tariff_version"] == payload["tariff_version"]
        ]

        assert matches, payload["product_code"]

        assert any(
            item["valid_from"] <= row["event_time"]
            and (item["valid_to"] is None or row["event_time"] < item["valid_to"])
            for item in matches
        ), f"{payload['product_code']} версия не действует на {row['event_time']}"

        checked += 1

    assert checked > 0


def test_coverage_covers_every_client_and_source(dataset):

    clients = {row["client_id"] for row in dataset["truth_clients"]}

    pairs = {(row["client_id"], row["source"]) for row in dataset["coverage"]}

    assert len(pairs) == len(clients) * len(SOURCES)

    seen = defaultdict(set)

    for row in dataset["events"]:
        seen[(row["client_id"], row["source"])].add(row["event_time"])

    for row in dataset["coverage"]:

        key = (row["client_id"], row["source"])

        if row["first_seen"] is None:
            assert not seen.get(key), f"события есть, а источник клиента не видел: {key}"
            assert row["coverage_status"] == "none"
            assert row["coverage_reason"]
        else:
            assert row["first_available_at"] <= row["first_seen"]


def test_profile_is_versioned(dataset):

    by_client = defaultdict(list)

    for row in dataset["profile"]:
        by_client[row["client_id"]].append(row)

    for client_id, rows in by_client.items():

        rows.sort(key=lambda item: item["profile_version"])

        assert [item["profile_version"] for item in rows] == list(range(1, len(rows) + 1))

        for left, right in zip(rows, rows[1:]):
            assert left["valid_to"] == right["valid_from"]

        assert rows[-1]["valid_to"] is None


# ------------------------------------------------------------
# УТЕЧКИ
# ------------------------------------------------------------


def test_hidden_fields_never_reach_raw(dataset):
    """
    Ни скрытые характеристики, ни household_id, ни готовые
    ответы не появляются в наблюдаемых данных.
    """

    columns = set(dataset["events"][0]) | set(dataset["profile"][0]) | set(dataset["coverage"][0])

    payload_keys = {name for row in dataset["events"] for name in row["payload"]}

    assert not (columns & FORBIDDEN_RAW_FIELDS)
    assert not (payload_keys & FORBIDDEN_RAW_FIELDS)

    for row in dataset["products"] + dataset["merchants"]:
        assert not (set(row) & FORBIDDEN_RAW_FIELDS)


def test_truth_keeps_what_raw_must_not(dataset):

    client = dataset["truth_clients"][0]

    assert any(name.startswith("trait_") for name in client)
    assert "client_ordinal" in client

    kinds = {row["kind"] for row in dataset["truth_events"]}

    assert "life_event" in kinds


def test_test_accounts_are_marked(dataset):

    flags = {row["client_id"]: row["is_test_account"] for row in dataset["truth_clients"]}

    for row in dataset["events"]:
        assert row["is_test_account"] == flags[row["client_id"]]


# ------------------------------------------------------------
# ФИНАНСЫ
# ------------------------------------------------------------


def test_financial_invariants_hold(dataset):

    by_client = defaultdict(list)

    for row in dataset["events"]:
        by_client[row["client_id"]].append(row)

    problems = invariants_module.check_all(by_client)

    assert not problems, [str(item) for item in problems[:5]]


def test_declined_operations_do_not_move_money(dataset):

    declined = [
        row
        for row in dataset["events"]
        if row["payload"].get("status") == "declined"
    ]

    assert declined, "отклонённых операций не оказалось вовсе"

    for row in declined:
        assert row["payload"].get("balance_after") is None
        assert row["payload"].get("decline_reason")


def test_periodic_charges_have_no_cause_but_have_a_period(dataset):
    """
    У периодической комиссии и начисления процентов нет
    отдельного события-причины: основание это договор и период.
    """

    checked = 0

    for row in dataset["events"]:

        if row["event_type"] not in ("fee_charge", "interest_credit", "cashback_credit"):
            continue

        payload = row["payload"]

        if payload.get("reason") != "periodic_contract_rule":
            continue

        assert payload.get("cause_event_id") is None
        assert payload.get("accrual_period")
        assert payload.get("contract_id")

        checked += 1

    assert checked > 0


def test_refunds_reference_their_purchase(dataset):

    amounts = {
        row["event_id"]: row["payload"]["amount"]
        for row in dataset["events"]
        if row["event_type"] == "purchase"
    }

    refunds = [
        row
        for row in dataset["events"]
        if row["event_type"] in ("refund", "reversal", "chargeback")
    ]

    assert refunds

    for row in refunds:
        cause = row["payload"].get("cause_event_id")
        assert cause is not None
        if cause in amounts:
            assert row["payload"]["amount"] <= amounts[cause]


# ------------------------------------------------------------
# ПРИЧИННОСТЬ
# ------------------------------------------------------------


def test_application_precedes_decision_and_contract(dataset):

    submitted = {}
    decided = {}

    for row in dataset["events"]:
        if row["event_type"] == "application_submitted":
            submitted[row["payload"]["application_id"]] = row["event_time"]
        elif row["event_type"] == "application_decision":
            decided[row["payload"]["application_id"]] = row["event_time"]

    assert submitted and decided

    for application_id, moment in decided.items():
        assert application_id in submitted
        assert submitted[application_id] <= moment

    for row in dataset["events"]:
        if row["event_type"] != "product_opened":
            continue
        correlation = row["correlation_id"]
        if correlation in decided:
            assert decided[correlation] <= row["event_time"]


def test_schedule_follows_disbursement(dataset):

    schedules = {}

    for row in dataset["events"]:
        if row["event_type"] == "schedule_created":
            schedules[row["payload"]["contract_id"]] = row["event_time"]

    assert schedules

    for row in dataset["events"]:
        if row["event_type"] == "installment_due":
            contract = row["payload"]["contract_id"]
            if contract in schedules:
                assert schedules[contract] <= row["event_time"]


def test_delinquency_follows_a_missed_installment(dataset):

    missed = defaultdict(list)
    registered = defaultdict(list)

    for row in dataset["events"]:
        if row["event_type"] == "installment_missed":
            missed[row["payload"]["contract_id"]].append(row["event_time"])
        elif row["event_type"] == "delinquency_registered":
            registered[row["payload"]["contract_id"]].append(row["event_time"])

    assert registered

    for contract, moments in registered.items():
        assert missed.get(contract), contract
        assert min(missed[contract]) <= max(moments)


# ------------------------------------------------------------
# ДЕТЕРМИНИЗМ
# ------------------------------------------------------------


def test_same_seed_same_content_across_workers_and_chunks(dataset, tmp_path):
    """
    Содержимое не зависит ни от числа воркеров, ни от размера
    чанка: они лишь распределяют сообщества.
    """

    parallel = _emit(tmp_path / "parallel", workers=3, chunk_clients=CLIENTS)

    assert parallel["rows"] == dataset["manifest"]["rows"]
    assert parallel["content_sha256"] == dataset["manifest"]["content_sha256"]

    other_chunk = _emit(tmp_path / "chunked", workers=2, chunk_clients=16)

    assert other_chunk["content_sha256"] == dataset["manifest"]["content_sha256"]


def test_other_seed_changes_content(dataset, tmp_path):

    other = _emit(tmp_path / "seed7", workers=1, chunk_clients=COMMUNITY_SIZE, seed=7)

    assert other["content_sha256"]["events"] != dataset["manifest"]["content_sha256"]["events"]


def test_hash_seed_does_not_change_content(dataset, tmp_path):
    """
    PYTHONHASHSEED не должен влиять ни на одну строку.
    """

    out = tmp_path / "hashseed"

    script = (
        "import sys; sys.path.insert(0, %r);"
        "from pathlib import Path;"
        "from src.generator.emit import generate_dataset;"
        "generate_dataset(total_clients=%d, out_dir=Path(%r), seed=42, workers=1,"
        " chunk_clients=%d, catalog_scale=%r, community_size=%d, quiet=True)"
        % (str(Path.cwd()), CLIENTS, str(out), COMMUNITY_SIZE, CATALOG_SCALE, COMMUNITY_SIZE)
    )

    environment = dict(os.environ, PYTHONHASHSEED="12345")

    subprocess.run([sys.executable, "-c", script], check=True, cwd=str(Path.cwd()), env=environment)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["content_sha256"] == dataset["manifest"]["content_sha256"]


# ------------------------------------------------------------
# ПРИСУТСТВИЕ ЯВЛЕНИЙ
# ------------------------------------------------------------


def test_all_sources_and_most_event_types_appear(dataset):

    types = Counter(row["event_type"] for row in dataset["events"])

    sources = {row["source"] for row in dataset["events"]}

    assert len(sources) >= 9
    assert len(types) >= 40


def test_defects_are_present(dataset):

    delays = [
        (row["record_time"] - row["event_time"]).total_seconds()
        for row in dataset["events"]
    ]

    assert max(delays) > 3600, "поздних записей нет"

    precisions = {row["time_precision"] for row in dataset["events"]}

    assert len(precisions) >= 2

    missing = sum(
        1
        for row in dataset["events"]
        for value in row["payload"].values()
        if value is None
    )

    assert missing > 0


# ============================================================
# ИТОГОВЫЕ ВЕРСИИ ЗАПИСЕЙ
# ============================================================
#
# Препроцессинг читает ПОСЛЕДНЮЮ версию каждой записи. Значит
# сходиться обязана именно она, а не та, что банк записал
# сначала и потом исправил.
# ============================================================


def test_invariants_hold_on_the_last_versions(dataset):

    by_client = defaultdict(list)

    for row in dataset["events"]:
        by_client[row["client_id"]].append(row)

    problems = []

    for events in by_client.values():

        latest = invariants_module.authoritative(events)

        assert latest, "после отбора последних версий не осталось записей"

        problems.extend(invariants_module.check_client(latest))
        problems.extend(invariants_module.check_money_conservation(latest))

    assert not problems, [str(item) for item in problems[:5]]


def test_correction_restores_the_true_value(dataset):
    """
    Исправление возвращает согласованное значение, а расходится
    с проводками ПЕРВАЯ версия. Наоборот быть не должно: иначе
    итоговая лента противоречила бы остаткам.
    """

    versions = defaultdict(list)

    for row in dataset["events"]:
        versions[row["event_id"]].append(row)

    checked = 0

    for rows in versions.values():

        if len({row["event_version"] for row in rows}) < 2:
            continue

        rows.sort(key=lambda item: item["event_version"])

        first, last = rows[0], rows[-1]

        if first["event_type"] not in invariants_module.MONEY_EVENTS:
            continue

        if first["payload"].get("status") != "approved":
            continue

        if first["payload"].get("amount") == last["payload"].get("amount"):
            continue

        checked += 1

        # Остаток принадлежит настоящей сумме, а её несёт
        # последняя версия.
        assert first["payload"].get("balance_after") == last["payload"].get("balance_after")

    assert checked > 0, "денежных исправлений в выборке не оказалось"


def test_corrections_touch_only_declared_fields(dataset):

    by_client = defaultdict(list)

    for row in dataset["events"]:
        by_client[row["client_id"]].append(row)

    problems = []

    for events in by_client.values():
        problems.extend(invariants_module.check_corrections(events))

    assert not problems, [str(item) for item in problems[:5]]


# ============================================================
# ПРОФИЛЬ НА ИСТОРИЧЕСКУЮ ДАТУ
# ============================================================


def test_profile_never_disappears_between_versions(dataset):
    """
    Новая версия поступает в витрину позже, чем начинает
    действовать. В этом промежутке банк знал предыдущую версию,
    и профиль обязан находиться.
    """

    from src.generator import profile as profile_module

    by_client = defaultdict(list)

    for row in dataset["profile"]:
        by_client[row["client_id"]].append(row)

    holes = 0
    checked = 0

    for client_id, versions in by_client.items():

        versions.sort(key=lambda item: item["profile_version"])

        if len(versions) < 2:
            continue

        for previous, following in zip(versions, versions[1:]):

            # Момент, когда предыдущая версия уже закрыта, а
            # новая ещё не дошла до витрины.
            moment = following["valid_from"]

            if following["record_time"] <= moment:
                continue

            checked += 1

            known = profile_module.as_of(versions, moment, record_time=moment)

            if known is None:
                holes += 1
                continue

            assert known["profile_version"] == previous["profile_version"]

    assert checked > 0, "границ версий с задержкой поступления не нашлось"
    assert holes == 0, f"профиль пропадал {holes} раз"


def test_profile_switches_once_the_new_version_is_known(dataset):

    from src.generator import profile as profile_module

    by_client = defaultdict(list)

    for row in dataset["profile"]:
        by_client[row["client_id"]].append(row)

    switched = 0

    for versions in by_client.values():

        versions.sort(key=lambda item: item["profile_version"])

        for following in versions[1:]:

            known = profile_module.known_at(versions, following["record_time"])

            assert known is not None
            assert known["profile_version"] >= following["profile_version"]

            switched += 1

    assert switched > 0
