from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator.emit import PRODUCTS_SCHEMA
from src.preprocessing.canonical.build import EVENTS_FILE, build_group
from src.preprocessing.history import (
    COVERAGE_AVAILABLE,
    COVERAGE_ENDED,
    COVERAGE_NOT_LAUNCHED,
    COVERAGE_NOT_SEEN,
    INTERNAL_COLUMNS,
    PAIR_NOT_VISIBLE,
    PAIR_VISIBLE,
    CanonicalStore,
    HistoryError,
    history_as_of,
)
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.run import main as run_main
from src.preprocessing.settings import PreprocessingConfig
from src.preprocessing.temporal import check_single, choose_cutoffs, temporal_report

from tests.prep_fixtures import MiniRaw, purchase_payload, tiny_raw  # noqa: F401


CONFIG = PreprocessingConfig()

FULL_HORIZON = datetime(2023, 1, 1)


def _cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        run_main(list(args))
    return int(result.value.code)


def _store(raw_dir: Path, out_dir: Path, products: pa.Table | None = None) -> CanonicalStore:
    build_group(raw_dir, out_dir, CONFIG, "train")
    return CanonicalStore(out_dir, products=products)


def _corrected_raw(tmp_path: Path, name: str = "raw") -> Path:
    """
    Покупка на 12 500, исправленная до 13 000, и ещё одна покупка
    днём позже.
    """

    mini = MiniRaw(tmp_path / name, history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    original = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=13000), event_id=original, version=2,
    )

    mini.event("c1", "purchase", "2023-03-06 10:00:00", payload=purchase_payload(amount=700))

    return mini.write()


# ============================================================
# ГРАНИЦА CUTOFF
# ============================================================


def test_row_exactly_at_cutoff_is_not_visible_yet(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    # Произошло ровно в cutoff.
    mini.event("c1", "purchase", "2023-03-05 12:00:00", payload=purchase_payload(amount=1))

    # Произошло раньше.
    mini.event("c1", "purchase", "2023-03-05 09:00:00", payload=purchase_payload(amount=2))
    mini.event("c1", "purchase", "2023-03-05 08:00:00", payload=purchase_payload(amount=3))

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 3, 5, 12, 0))

    assert history.events.column("amount").to_pylist() == [3, 2]
    assert history.counts["event_not_happened"] == 1

    # Через микросекунду после границы видны все три.
    later = history_as_of(store, "c1", datetime(2023, 3, 5, 12, 0, 0, 1))
    assert later.events.column("amount").to_pylist() == [3, 2, 1]


def test_cutoff_beyond_the_extract_is_refused(tmp_path):

    store = _store(_corrected_raw(tmp_path), tmp_path / "canonical")

    with pytest.raises(HistoryError, match="позже границы выгрузки"):
        history_as_of(store, "c1", datetime(2030, 1, 1))


# ============================================================
# ВЕРСИИ
# ============================================================


def test_highest_version_acts_as_soon_as_the_event_is_visible(tmp_path):
    """
    Времени поступления у выгрузки нет: как только событие
    произошло, действует его наибольшая версия, и стоит она на
    месте события, а не в конце истории.
    """

    store = _store(_corrected_raw(tmp_path), tmp_path / "canonical")

    before = history_as_of(store, "c1", datetime(2023, 3, 5, 9, 0))
    assert before.events.num_rows == 0

    after = history_as_of(store, "c1", datetime(2023, 3, 10))

    assert after.events.column("amount").to_pylist() == [13000, 700]
    assert after.events.column("event_version").to_pylist() == [2, 1]
    assert after.events.column("event_time").to_pylist()[0] == datetime(2023, 3, 5, 10, 0)
    assert after.counts["superseded"] == 1
    assert after.counts["visible"] == 2


def test_highest_version_wins_regardless_of_tape_order(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    first = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=100))
    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=300), event_id=first, version=3,
    )
    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=200), event_id=first, version=2,
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 4, 1))

    assert history.events.num_rows == 1
    assert history.events.column("event_version").to_pylist() == [3]
    assert history.events.column("amount").to_pylist() == [300]
    assert history.counts["superseded"] == 2


def test_redelivery_does_not_create_a_second_action(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    original = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=500))
    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=500), event_id=original, version=1,
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 3, 10))

    assert history.events.num_rows == 1
    assert history.counts["duplicate"] == 1


def test_conflicting_rows_of_one_version_keep_the_first_in_the_tape(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    first = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=100))
    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=555), event_id=first, version=1,
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 4, 1))

    assert history.events.column("amount").to_pylist() == [100]
    assert history.counts["conflict_dropped"] == 1
    assert any("конфликт версий" in note for note in history.limitations)


def test_future_events_do_not_change_earlier_history(tmp_path):
    """
    Добавление более поздних событий не меняет того, что история
    уже говорила о прошлом.
    """

    first = _store(_corrected_raw(tmp_path, "raw_a"), tmp_path / "canonical_a")

    before = history_as_of(first, "c1", datetime(2023, 3, 10))

    # Та же лента плюс будущее.
    mini = MiniRaw(tmp_path / "raw_b", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    original = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))
    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=13000), event_id=original, version=2,
    )
    mini.event("c1", "purchase", "2023-03-06 10:00:00", payload=purchase_payload(amount=700))
    mini.event("c1", "purchase", "2023-09-01 10:00:00", payload=purchase_payload(amount=9999))
    mini.event("c1", "purchase", "2023-10-01 10:00:00", payload=purchase_payload(amount=777))

    second = _store(mini.write(), tmp_path / "canonical_b")

    after = history_as_of(second, "c1", datetime(2023, 3, 10))

    columns = [name for name in before.events.column_names if name != "raw_row"]

    assert before.events.select(columns).to_pylist() == after.events.select(columns).to_pylist()


# ============================================================
# СЛУЖЕБНЫЕ ПОЛЯ
# ============================================================


def test_internal_columns_never_leave(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.cover_all("c2", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    store = _store(mini.write(), tmp_path / "canonical")

    with_events = history_as_of(store, "c1", datetime(2023, 4, 1))
    empty = history_as_of(store, "c2", datetime(2023, 4, 1))
    before_anything = history_as_of(store, "c1", datetime(2023, 1, 2))

    for history in (with_events, empty, before_anything):
        assert not set(INTERNAL_COLUMNS) & set(history.events.column_names)
        assert check_single(history) == []

    assert empty.events.num_rows == 0
    assert before_anything.events.num_rows == 0


# ============================================================
# ПРОФИЛЬ
# ============================================================


def test_profile_version_acts_from_its_valid_from(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    mini.profile_version("c1", 1, "2023-01-31", age=40, declared_income=100000)
    mini.profile_version("c1", 2, "2023-02-28", age=40, declared_income=150000)

    store = _store(mini.write(), tmp_path / "canonical")

    early = history_as_of(store, "c1", datetime(2023, 1, 5))
    assert early.profile is None
    assert early.profile_meta["state"] == "no_version_known_yet"

    between = history_as_of(store, "c1", datetime(2023, 2, 10))
    assert between.profile["profile_version"] == 1
    assert between.profile["declared_income"] == 100000
    assert between.profile_meta["versions_known"] == 1

    # Ровно в valid_from новая версия ещё не действует: граница исключительная.
    at_switch = history_as_of(store, "c1", datetime(2023, 2, 28))
    assert at_switch.profile["profile_version"] == 1

    after = history_as_of(store, "c1", datetime(2023, 3, 10))
    assert after.profile["profile_version"] == 2
    assert after.profile["declared_income"] == 150000


# ============================================================
# СОСТОЯНИЯ
# ============================================================


def _card_payload(**overrides) -> dict:
    payload = {
        "product_id": "prd_card",
        "product_code": "CARD",
        "product_version": 1,
        "tariff_version": 1,
        "product_family": "debit_card",
        "contract_id": "ctr_1",
        "account_id": "acc_1",
        "card_id": "crd_1",
        "offer_id": None,
        "previous_product_id": None,
        "migration_reason": None,
        "amount_or_limit": None,
        "term": None,
        "rate": None,
        "reason": "opened",
        "timestamp_quality": "exact",
    }
    payload.update(overrides)
    return payload


def test_planned_change_is_not_in_force_before_effective_at(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "card_activated", "2023-02-01 10:00:00", payload=_card_payload())

    # Блокировка объявлена первого марта, но вступает в силу первого апреля.
    mini.event(
        "c1", "card_blocked", "2023-03-01 10:00:00",
        payload=_card_payload(reason="blocked"), effective_at="2023-04-01 00:00:00",
    )

    store = _store(mini.write(), tmp_path / "canonical")

    announced = history_as_of(store, "c1", datetime(2023, 3, 15))

    card = next(item for item in announced.entities if item.kind == "card")

    assert card.state == "active"
    assert card.pending == (("blocked", datetime(2023, 4, 1)),)
    assert any("не вступивших в силу" in note for note in announced.limitations)

    in_force = history_as_of(store, "c1", datetime(2023, 4, 2))

    card = next(item for item in in_force.entities if item.kind == "card")

    assert card.state == "blocked"
    assert card.pending == ()


def test_state_follows_effective_at_order(tmp_path):
    """
    Порядок применения переходов задаёт effective_at, а не время
    записи.

    Блокировка, объявленная в феврале с первого апреля, действует
    ПОЗЖЕ мартовской разблокировки, хотя записана раньше неё.
    Считая по времени записи, слой возвращал бы карту рабочей там,
    где она заблокирована.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "card_activated", "2023-02-01 10:00:00", payload=_card_payload())

    # Объявлено раньше, действует позже.
    mini.event(
        "c1", "card_blocked", "2023-02-10 10:00:00",
        payload=_card_payload(reason="blocked"), effective_at="2023-04-01 00:00:00",
    )

    # Записано позже объявления, но действует раньше него.
    mini.event(
        "c1", "card_unblocked", "2023-03-05 10:00:00",
        payload=_card_payload(reason="unblocked"),
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 5, 1))

    card = next(item for item in history.entities if item.kind == "card")

    assert card.state == "blocked"
    assert card.since == datetime(2023, 4, 1)
    assert card.last_transition == "blocked"
    assert card.last_transition_at == datetime(2023, 2, 10, 10, 0)
    assert card.pending == ()

    # До первого апреля действует мартовская разблокировка.
    earlier = history_as_of(store, "c1", datetime(2023, 3, 20))

    card = next(item for item in earlier.entities if item.kind == "card")

    assert card.state == "active"
    assert card.pending == (("blocked", datetime(2023, 4, 1)),)


def test_coverage_state_uses_only_dated_fields(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)

    mini.cover("c1", "transactions", first_seen="2023-02-01")
    # Конец объявлен, но он ещё впереди относительно первого среза.
    mini.cover("c1", "loans", first_seen="2023-02-01", last_available_at="2023-06-01")
    # Источник работает, но клиент в нём не появлялся; причина не датирована.
    mini.cover("c1", "applications", first_seen=None, status="none", reason="no_consent")

    for source in ("profile", "product_events", "app_operations"):
        mini.cover("c1", source, first_seen="2023-02-01")

    # Эти источники банк запускает позже обоих срезов.
    for source in ("communications", "banners", "app_screens", "support", "antifraud"):
        mini.cover("c1", source, first_seen=None)

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 3, 10))

    states = {item.source: item.state for item in history.coverage}

    assert states["transactions"] == COVERAGE_AVAILABLE
    # Объявленный конец ещё впереди: источник доступен.
    assert states["loans"] == COVERAGE_AVAILABLE
    # Недатированная причина в состояние не входит: сказано только,
    # что клиент в источнике пока не замечен.
    assert states["applications"] == COVERAGE_NOT_SEEN
    # Запуск источника в банке позже среза.
    assert states["antifraud"] == COVERAGE_NOT_LAUNCHED
    assert states["support"] == COVERAGE_NOT_LAUNCHED

    assert any("не датированы" in note for note in history.limitations)

    later = history_as_of(store, "c1", datetime(2023, 7, 1))
    assert {item.source: item.state for item in later.coverage}["loans"] == COVERAGE_ENDED


# ============================================================
# ПЕРЕВОДЫ
# ============================================================


def test_counterpart_becomes_visible_only_after_its_own_event(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.cover_all("c2", first_seen="2023-01-01")

    mini.event(
        "c1", "p2p_out", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=5000, direction="debit"),
        correlation_id="trf_1", link_type="transfer",
    )
    # Получатель провёл приход на следующий день.
    mini.event(
        "c2", "p2p_in", "2023-03-06 09:00:00",
        payload=purchase_payload(amount=5000, direction="credit"),
        correlation_id="trf_1", link_type="transfer",
    )

    store = _store(mini.write(), tmp_path / "canonical")

    early = history_as_of(store, "c1", datetime(2023, 3, 5, 12, 0))

    side = early.transfers[0]
    assert side.pair_state == PAIR_NOT_VISIBLE
    assert side.counterpart_client_id is None

    late = history_as_of(store, "c1", datetime(2023, 3, 7))

    side = late.transfers[0]
    assert side.pair_state == PAIR_VISIBLE
    assert side.counterpart_client_id == "c2"


# ============================================================
# ОТНОШЕНИЯ И СПРАВОЧНИК
# ============================================================


def test_transfers_follow_the_same_version_rule(tmp_path):
    """
    У исправленного перевода действует последняя версия, и
    представление переводов обязано показывать её же.

    Раньше стороны отбирались по метке доставки: исправление
    получало link_type correction, выпадало из индекса, и история
    событий показывала 100, а история переводов — 99.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    original = mini.event(
        "c1", "p2p_out", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=99, counterparty="A. Serikov"),
        correlation_id="trf_1", link_type="transfer",
    )

    mini.event(
        "c1", "p2p_out", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=100, counterparty="A. Serikov"),
        event_id=original, version=2, correlation_id="trf_1", link_type="transfer",
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 4, 1))

    rows = history.events.to_pylist()

    assert [row["amount"] for row in rows] == [100]

    assert [item.amount for item in history.transfers] == [100]
    assert [item.side for item in history.transfers] == ["out"]


def test_transfer_index_selects_by_event_type(tmp_path):
    """
    Сторона перевода это тип события, а не метка связи.

    Метка связи описывает связь: комиссия за перевод носит ту же
    метку и тот же correlation_id, но стороной перевода не
    является. Индекс отбирал по метке и делал из комиссии сторону
    без направления; canonical при этом отбирал по типу, и два
    представления одного понятия расходились.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event(
        "c1", "p2p_out", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=100, counterparty="A. Serikov"),
        correlation_id="trf_1", link_type="transfer",
    )

    mini.event(
        "c1", "fee_charge", "2023-03-05 10:00:01",
        payload=purchase_payload(amount=50, reason="transfer_fee"),
        correlation_id="trf_1", link_type="transfer",
    )

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 4, 1))

    assert [(item.side, item.amount) for item in history.transfers] == [("out", 100)]


def test_relationship_reports_incomplete_history(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)

    for source in ("transactions", "profile", "applications", "loans", "communications", "banners",
                   "app_screens", "app_operations", "support", "antifraud"):
        mini.cover("c1", source, first_seen="2023-01-01")

    mini.cover(
        "c1", "product_events", first_seen="2023-01-01",
        opening_state='{"contracts_before_window": 2, "open_contracts": 1}',
    )

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    store = _store(mini.write(), tmp_path / "canonical")

    history = history_as_of(store, "c1", datetime(2023, 3, 10))

    assert history.relationship.history_incomplete is True
    assert any("opening_state" in reason for reason in history.relationship.incomplete_reasons)
    assert history.relationship.observed_start == datetime(2023, 1, 1)
    assert history.relationship.closed_at is None


def _products_table() -> pa.Table:

    rows = []

    for version, valid_from, status in ((1, datetime(2023, 1, 1), "active"), (2, datetime(2024, 1, 1), "active")):
        row = {name: None for name in PRODUCTS_SCHEMA.names}
        row.update(
            {
                "product_id": "prd_card",
                "product_code": "CARD",
                "product_family": "debit_card",
                "group": "cards",
                "product_version": version,
                "tariff_version": version,
                "status": status,
                "valid_from": valid_from,
                "unresolved_source": False,
                "is_synthetic": False,
                "allow_multiple": True,
                "max_active_holdings": 1,
                "notice_days": 0,
            }
        )
        rows.append(row)

    return pa.Table.from_pylist(rows, schema=PRODUCTS_SCHEMA)


def test_catalogue_version_is_taken_as_of_cutoff(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "card_activated", "2023-02-01 10:00:00", payload=_card_payload())
    mini.event("c1", "card_reissued", "2024-06-01 10:00:00", payload=_card_payload(product_version=2, tariff_version=2))

    store = _store(mini.write(), tmp_path / "canonical", products=_products_table())

    early = history_as_of(store, "c1", datetime(2023, 3, 1))

    # Событие названо одной версией и не размножается справочником.
    assert len(early.products) == 1
    assert early.products["prd_card|v1|t1"]["state"] == "known"
    assert early.products["prd_card|v1|t1"]["product_version"] == 1

    late = history_as_of(store, "c1", datetime(2024, 7, 1))

    assert late.products["prd_card|v2|t2"]["state"] == "known"
    assert late.products["prd_card|v1|t1"]["state"] == "known"


# ============================================================
# ОТЧЁТ И CLI
# ============================================================


def test_temporal_report_checks_rules_on_a_sample(tiny_raw, tmp_path):

    store = _store(tiny_raw, tmp_path / "canonical")

    cutoffs = [datetime(2025, 1, 1), datetime(2026, 1, 1), datetime(2026, 9, 1)]

    clients = [row["client_id"] for row in store.clients[:5]]

    report = temporal_report(store, clients, cutoffs, "train")

    assert report["status"] == "ok"
    assert report["problem_count"] == 0
    assert report["clients_checked"] == 5

    # Набор известных событий растёт от среза к срезу.
    counts = [report["per_cutoff"][moment.isoformat()]["events"] for moment in cutoffs]
    assert counts == sorted(counts)

    assert set(report["coverage_states"]) <= {
        COVERAGE_AVAILABLE,
        COVERAGE_ENDED,
        COVERAGE_NOT_LAUNCHED,
        COVERAGE_NOT_SEEN,
    }


def test_choose_cutoffs_stays_inside_the_extract():

    cutoffs = choose_cutoffs(datetime(2023, 1, 1), datetime(2026, 9, 1), datetime(2026, 3, 1), 4)

    assert len(cutoffs) == 4
    assert cutoffs[0] == datetime(2023, 1, 1)
    assert cutoffs[-1] == datetime(2026, 3, 1)
    assert cutoffs == sorted(cutoffs)


def test_history_cli_requires_canonical_and_writes_examples(tmp_path, capsys):

    raw_dir = _corrected_raw(tmp_path)
    out = tmp_path / "processed"

    # Без canonical этап не запускается.
    assert _cli("history", "--name", "x", "--group", "train", "--out", str(out)) == EXIT_BLOCKED
    assert "canonical" in capsys.readouterr().out

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    capsys.readouterr()

    assert (
        _cli(
            "history", "--name", "x", "--group", "train", "--out", str(out),
            "--client", "c1", "--cutoff", "2023-03-05T09:00:00", "--cutoff", "2023-03-10T00:00:00",
        )
        == EXIT_OK
    )

    printed = capsys.readouterr().out
    assert "c1 на 2023-03-05" in printed and "c1 на 2023-03-10" in printed

    assert (out / "history" / "train" / "examples" / "c1__2023-03-05.md").exists()
    assert (out / "history" / "train" / "temporal_report.json").exists()

    report = json.loads((out / "history" / "train" / "temporal_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"

    entry = json.loads((out / "preprocessing_manifest.json").read_text(encoding="utf-8"))["stages"]["history"]["train"]
    assert entry["problems"] == 0

    # Правленый canonical закрывает этап.
    events = out / "canonical" / "train" / EVENTS_FILE
    events.write_bytes(events.read_bytes() + b"tail")

    capsys.readouterr()

    assert _cli("history", "--name", "x", "--group", "train", "--out", str(out)) == EXIT_BLOCKED
    assert "изменились после сборки" in capsys.readouterr().out


def test_stage_version_tracks_sources():

    import hashlib

    from src.preprocessing import history as history_module
    from src.preprocessing import run as run_module
    from src.preprocessing import settings as settings_module
    from src.preprocessing import temporal as temporal_module

    modules = {
        "history": history_module,
        "temporal": temporal_module,
        "settings": settings_module,
        "run": run_module,
    }

    stored = json.loads(Path("tests/prep_stage_sources.json").read_text(encoding="utf-8"))["history"]

    assert stored["version"] == history_module.STAGE_VERSION

    actual = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in modules.items()
    }

    assert stored["modules"] == actual, (
        "модули этапа изменены: поднимите STAGE_VERSION и обновите tests/prep_stage_sources.json"
    )
