from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.preprocessing.artifacts import dumps_json
from src.preprocessing.canonical.build import (
    CLIENT_INDEX_FILE,
    COVERAGE_FILE,
    DEDUPE_FILE,
    EVENTS_FILE,
    MENTIONS_FILE,
    PROFILE_FILE,
    REGISTRY_FILE,
    REJECTS_FILE,
    TRANSFERS_FILE,
    build_group,
)
from src.preprocessing.canonical.events import normalize_text
from src.preprocessing.canonical.schema import (
    VERSION_ROLE_CONFLICT,
    VERSION_ROLE_CORRECTION,
    VERSION_ROLE_ORIGINAL,
    VERSION_ROLE_REDELIVERY,
)
from src.preprocessing.canonical.events import CanonicalError
from src.preprocessing.rawdata import RawDataset
from src.preprocessing.run import EXIT_BLOCKED, EXIT_CONTRACT_MISMATCH, EXIT_OK
from src.preprocessing.run import main as run_main
from src.preprocessing.settings import PreprocessingConfig

from tests.prep_fixtures import MiniRaw, check_raw, purchase_payload, tiny_raw  # noqa: F401


CONFIG = PreprocessingConfig()

FULL_HORIZON = datetime(2023, 1, 1)


def _cli(*args: str) -> int:
    with pytest.raises(SystemExit) as result:
        run_main(list(args))
    return int(result.value.code)


def _build(raw_dir: Path, out_dir: Path, config: PreprocessingConfig = CONFIG) -> tuple[pa.Table, dict]:
    result = build_group(raw_dir, out_dir, config, "train")
    return pq.read_table(out_dir / EVENTS_FILE), result.report


def _rows(table: pa.Table, **where) -> list[dict]:
    rows = table.to_pylist()
    for key, value in where.items():
        rows = [row for row in rows if row[key] == value]
    return rows


# ============================================================
# ПОЛНОТА И ТРАССИРОВКА
# ============================================================


def test_every_raw_row_becomes_exactly_one_canonical_row(tiny_raw, tmp_path):

    table, report = _build(tiny_raw, tmp_path / "canonical")

    raw = RawDataset(tiny_raw)

    assert table.num_rows == raw.manifest.rows["events"]
    assert report["rows"]["difference"] == 0
    assert report["status"] == "ok"

    traced = sorted(table.column("raw_row").to_pylist())
    assert traced == list(range(table.num_rows))

    assert set(table.column("raw_file").to_pylist()) == {"events.parquet"}
    assert set(table.column("raw_row_group").to_pylist()) <= set(range(raw.metadata("events").num_row_groups))


def test_values_survive_the_transform(tiny_raw, tmp_path):
    """
    Суммы, валюты и тексты обязаны совпасть с RAW побайтово: слой
    canonical перекладывает, а не пересчитывает.
    """

    table, _ = _build(tiny_raw, tmp_path / "canonical")

    raw_rows = pq.read_table(RawDataset(tiny_raw).path("events")).to_pylist()

    canonical = {row["raw_row"]: row for row in table.to_pylist()}

    checked = 0

    for index, raw_row in enumerate(raw_rows):

        payload = json.loads(raw_row["payload"])
        row = canonical[index]

        for name in ("amount", "currency", "merchant_name", "balance_after", "mcc", "original_currency"):
            if name in payload:
                assert row[name] == payload[name], f"{name} в строке {index}"
                checked += 1

        assert row["event_id"] == raw_row["event_id"]
        assert row["event_time"] == raw_row["event_time"]
        assert row["event_version"] == raw_row["event_version"]

    assert checked > 1000


def test_profile_and_coverage_are_kept_whole(tiny_raw, tmp_path):

    out = tmp_path / "canonical"
    _build(tiny_raw, out)

    raw = RawDataset(tiny_raw)

    profile = pq.read_table(out / PROFILE_FILE)
    coverage = pq.read_table(out / COVERAGE_FILE)

    assert profile.num_rows == raw.manifest.rows["profile"]
    assert coverage.num_rows == raw.manifest.rows["source_coverage"]

    # opening_state разобран, но остаётся заявленным состоянием, и
    # ни один объявленный ключ не теряется: форма зависит от источника.
    statuses = set(coverage.column("opening_state_status").to_pylist())
    assert statuses <= {"parsed", "absent", "unparseable", "unexpected_values"}

    for row in coverage.to_pylist():

        if row["opening_state"] is None:
            assert row["opening_state_status"] == "absent"
            assert row["opening_state_values"] is None
            continue

        declared = json.loads(row["opening_state"])
        stored = dict(row["opening_state_values"] or [])

        assert stored == declared, row["source"]


# ============================================================
# ВЕРСИИ, ДУБЛИ, КОНФЛИКТЫ
# ============================================================


def _versions_raw(tmp_path: Path) -> Path:
    """
    Четыре случая в одной ленте: исходная запись, её повторная
    доставка, исправление и отдельная похожая покупка.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    original = mini.event(
        "c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500)
    )

    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=12500), event_id=original, version=1,
    )

    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=13000), event_id=original, version=2,
    )

    # Другое событие с теми же полями: не дубль.
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=12500))

    # Ещё одно событие позже, чтобы проверить порядок.
    mini.event("c1", "purchase", "2023-03-06 10:00:00", payload=purchase_payload(amount=900))

    return mini.write()


def test_duplicate_correction_and_similar_event_are_distinguished(tmp_path):

    raw_dir = _versions_raw(tmp_path)

    out = tmp_path / "canonical"
    table, report = _build(raw_dir, out)

    rows = sorted(table.to_pylist(), key=lambda row: row["raw_row"])

    original, duplicate, correction, similar, later = rows

    assert original["version_role"] == VERSION_ROLE_ORIGINAL
    assert duplicate["version_role"] == VERSION_ROLE_REDELIVERY
    assert duplicate["is_exact_duplicate"] is True
    assert correction["version_role"] == VERSION_ROLE_CORRECTION
    assert correction["amount"] == 13000

    # Похожая покупка это отдельное событие, а не дубль.
    assert similar["version_role"] == VERSION_ROLE_ORIGINAL
    assert similar["is_exact_duplicate"] is False
    assert similar["event_id"] != original["event_id"]

    assert report["versions"]["corrections"] == 1
    assert report["versions"]["redeliveries"] == 1
    assert report["versions"]["conflicts"] == 0

    log = pq.read_table(out / DEDUPE_FILE).to_pylist()
    assert len(log) == 1
    assert log[0]["verdict"] == VERSION_ROLE_REDELIVERY
    assert log[0]["differing_fields"] is None
    assert log[0]["same_version_row"] == original["raw_row"]


def test_versions_share_one_place_and_order_is_deterministic(tmp_path):
    """
    Номера записи в сохранённых данных нет вовсе. Место события
    задают время, приоритет типа и event_id, а все версии и дубли
    одного события делят один номер логического события.
    """

    raw_dir = _versions_raw(tmp_path)

    table, _ = _build(raw_dir, tmp_path / "canonical")

    assert "sequence_number" not in table.column_names
    assert "original_sequence_number" not in table.column_names
    assert "sequence_number" not in pq.read_schema(raw_dir / "events.parquet").names

    rows = sorted(table.to_pylist(), key=lambda row: row["raw_row"])
    original, duplicate, correction, similar, later = rows

    # Оригинал, его повторная доставка и исправление это одно
    # событие и одно место в истории.
    assert duplicate["stable_event_index"] == original["stable_event_index"]
    assert correction["stable_event_index"] == original["stable_event_index"]

    # Похожая покупка в ту же секунду это другое событие: место
    # своё, и порядок между ними задан приоритетом типа и event_id.
    assert similar["stable_event_index"] != original["stable_event_index"]
    assert later["stable_event_index"] == max(row["stable_event_index"] for row in rows)

    # Тот же вход даёт тот же порядок.
    again, _ = _build(raw_dir, tmp_path / "canonical_again")
    assert again.column("stable_event_index").to_pylist() == table.column("stable_event_index").to_pylist()


def test_versions_are_recognised_regardless_of_tape_order(tmp_path):
    """
    Версии могут стоять в ленте не по номеру. Роль строки задаёт
    номер версии, а не место в файле; какая версия действует,
    решает этап истории.
    """

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

    table, report = _build(mini.write(), tmp_path / "canonical")

    rows = {row["event_version"]: row for row in table.to_pylist()}

    assert rows[1]["version_role"] == VERSION_ROLE_ORIGINAL
    assert rows[2]["version_role"] == VERSION_ROLE_CORRECTION
    assert rows[3]["version_role"] == VERSION_ROLE_CORRECTION

    # Все три версии это одно событие и одно место в истории.
    assert len({row["stable_event_index"] for row in rows.values()}) == 1

    assert report["versions"]["corrections"] == 2


def test_same_version_with_different_content_is_a_conflict(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    first = mini.event(
        "c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=100)
    )
    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=555), event_id=first, version=1,
    )

    out = tmp_path / "canonical"
    table, report = _build(mini.write(), out)

    roles = [row["version_role"] for row in sorted(table.to_pylist(), key=lambda row: row["raw_row"])]
    assert roles == [VERSION_ROLE_ORIGINAL, VERSION_ROLE_CONFLICT]

    assert report["versions"]["conflicts"] == 1

    log = pq.read_table(out / DEDUPE_FILE).to_pylist()
    assert log[0]["verdict"] == VERSION_ROLE_CONFLICT
    assert log[0]["differing_fields"] == ["amount"]

    # Конфликтная строка остаётся в canonical со своим содержимым.
    assert sorted(row["amount"] for row in table.to_pylist()) == [100, 555]


# ============================================================
# НАБЛЮДАЕМОСТЬ
# ============================================================


def test_null_in_required_envelope_field_stops_canonical_with_a_reason(tmp_path):
    """
    Если повреждённая выгрузка каким-то образом обошла паспорт,
    canonical останавливается сам, называя поле и строку, а не
    падает TypeError глубже по коду и не теряет строку молча.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    mini.event("c1", "purchase", None, payload=purchase_payload(amount=999))

    raw_dir = mini.write()

    with pytest.raises(CanonicalError) as failure:
        build_group(raw_dir, tmp_path / "canonical", CONFIG, "train")

    message = str(failure.value)

    assert "event_time" in message and "строка 1" in message


def test_unparseable_and_unknown_rows_are_kept_with_a_reason(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    mini.event("c1", "purchase", "2023-03-06 10:00:00", raw_payload="{это не json")

    out = tmp_path / "canonical"
    table, report = _build(mini.write(), out)

    assert table.num_rows == 2

    broken = _rows(table, payload_status="unparseable")
    assert len(broken) == 1
    assert broken[0]["amount"] is None
    assert broken[0]["payload_violations"][0].startswith("unparseable")

    rejects = pq.read_table(out / REJECTS_FILE).to_pylist()
    assert len(rejects) == 1
    assert rejects[0]["reason"] == "unparseable"
    assert rejects[0]["payload"] == "{это не json"
    assert report["rows"]["rejects"] == 1


def test_dated_schema_rule_explains_the_missing_value(tmp_path):
    """
    Правило схемы говорит, С КАКОЙ даты источник начал собирать
    поле. Пусто ДО этой даты — известная причина; пусто после —
    обычный пропуск, объяснять который нечем.

    Раньше знак сравнения стоял наоборот, и причина приписывалась
    ровно тем строкам, у которых её нет.
    """

    rule = {"source": "app_screens", "field": "product_id", "from": "2025-06-01", "reason": "not_collected"}

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON, schema_changes=(rule,))
    mini.cover_all("c1", first_seen="2023-01-01")

    payload = {"firebase_screen": "s_000_home", "domain": "home", "product_id": None, "funnel_stage": None, "reject_reason": None}

    mini.event("c1", "app_screen", "2025-05-01 10:00:00", payload=payload)
    mini.event("c1", "app_screen", "2025-07-01 10:00:00", payload=payload)

    table, _ = _build(mini.write(), tmp_path / "canonical")

    rows = sorted(table.to_pylist(), key=lambda row: row["event_time"])

    # До 1 июня поле не собиралось вовсе.
    assert rows[0]["known_missing"] == ["product_id=not_collected"]

    # После — собиралось, и пустое значение причиной не объяснено.
    assert rows[1]["known_missing"] is None


def test_non_contiguous_client_is_refused(tmp_path):
    """
    Canonical не полагается на то, что паспорт запускали: пачку
    целых клиентов из разорванной ленты собрать нельзя.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.cover_all("c2", first_seen="2023-01-01")

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    mini.event("c2", "purchase", "2023-03-06 10:00:00", payload=purchase_payload())
    mini.event("c1", "purchase", "2023-03-07 10:00:00", payload=purchase_payload())

    with pytest.raises(CanonicalError) as error:
        _build(mini.write(), tmp_path / "canonical")

    assert "не подряд" in str(error.value)


def test_eventless_test_client_keeps_its_flag(tmp_path):
    """
    У клиента без событий признак тестового счёта брать неоткуда,
    кроме покрытия. Проставленный False означал бы «настоящий
    клиент», и тестовые счета попадали бы в обучение.
    """

    from src.generator.config import SOURCES

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)

    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    # Клиент есть в покрытии, событий у него нет вовсе.
    for source in SOURCES:
        mini.cover("c_test", source, first_seen=None, status="none", reason="test_account")

    out = tmp_path / "canonical"

    _build(mini.write(), out)

    index = {
        row["client_id"]: row
        for row in pq.read_table(out / CLIENT_INDEX_FILE).to_pylist()
    }

    assert index["c_test"]["row_count"] == 0
    assert index["c_test"]["is_test_account"] is True
    assert index["c1"]["is_test_account"] is False


def test_balance_chain_gap_is_flagged_from_raw_only(tmp_path):
    """
    Остаток обязан продолжать предыдущий остаток того же счёта.
    Если не продолжает, между строками потеряно движение денег, и
    строка получает признак качества.

    Считается это по самой выгрузке: скрытую истину canonical не
    читает, а другого способа увидеть пропажу у него нет.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    def purchase(day: str, amount: int, balance: int) -> None:
        mini.event(
            "c1",
            "purchase",
            f"2023-03-{day} 10:00:00",
            payload=purchase_payload(amount=amount, balance_after=balance),
        )

    purchase("05", 1_000, 99_000)
    # Между строками потеряно зачисление: 99 000 − 500 это 98 500.
    purchase("06", 500, 88_500)
    purchase("07", 500, 88_000)

    table, report = _build(mini.write(), tmp_path / "canonical")

    rows = sorted(table.to_pylist(), key=lambda row: row["event_time"])

    assert [row["balance_chain_gap"] for row in rows] == [False, True, False]

    assert report["flags"]["balance_chain_gap"] == 1

    # Скрытая истина в наборе есть, но этап её не открывал.
    assert not (tmp_path / "canonical" / "truth").exists()


def test_snapshot_confirms_the_chain_without_moving_money(tmp_path):
    """
    Снимок остатка денег не двигает: его сумма это сам остаток.
    Разрывом считается расхождение снимка с цепочкой.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=1_000, balance_after=99_000),
    )

    mini.event(
        "c1", "balance_snapshot", "2023-03-31 23:55:00",
        payload=purchase_payload(
            amount=99_000, direction="credit", balance_after=99_000, reason="periodic_contract_rule",
        ),
    )

    mini.event(
        "c1", "balance_snapshot", "2023-04-30 23:55:00",
        payload=purchase_payload(
            amount=70_000, direction="credit", balance_after=70_000, reason="periodic_contract_rule",
        ),
    )

    table, _ = _build(mini.write(), tmp_path / "canonical")

    rows = sorted(table.to_pylist(), key=lambda row: row["event_time"])

    assert [row["balance_chain_gap"] for row in rows] == [False, False, True]


def test_time_precision_and_ambiguous_local_time_are_flagged(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    # Точность источника day, а во времени остались часы.
    mini.event(
        "c1", "installment_paid", "2023-03-05 14:30:00",
        payload={"contract_id": "ctr_1", "installment_no": 1, "amount_due": 100, "amount_paid": 100,
                 "principal_outstanding": 0, "days_past_due": 0, "due_date": "2023-03-05",
                 "cause_event_id": None, "reason": "payment"},
    )

    # Полночь того же источника: расхождения нет.
    mini.event(
        "c1", "installment_paid", "2023-03-06 00:00:00",
        payload={"contract_id": "ctr_1", "installment_no": 2, "amount_due": 100, "amount_paid": 100,
                 "principal_outstanding": 0, "days_past_due": 0, "due_date": "2023-03-06",
                 "cause_event_id": None, "reason": "payment"},
    )

    # Час, прожитый дважды при переходе на UTC+5.
    mini.event("c1", "purchase", "2024-02-29 23:30:00", payload=purchase_payload())

    table, report = _build(mini.write(), tmp_path / "canonical")

    rows = sorted(table.to_pylist(), key=lambda row: row["event_time"])

    assert rows[0]["time_finer_than_precision"] is True
    assert rows[1]["time_finer_than_precision"] is False
    assert rows[2]["ambiguous_local_time"] is True

    assert report["timezone"]["contract"] == "Asia/Almaty"
    assert report["timezone"]["ambiguous_rows"] == 1


def test_text_normalization_keeps_the_original(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload=purchase_payload(merchant_name="  EVRO PHARMA   24  "),
    )

    table, _ = _build(mini.write(), tmp_path / "canonical")

    row = table.to_pylist()[0]

    assert row["merchant_name"] == "  EVRO PHARMA   24  "
    assert row["merchant_name_norm"] == "evro pharma 24"
    assert normalize_text(None) is None


# ============================================================
# СУЩНОСТИ И СВЯЗИ
# ============================================================


def test_mentions_and_transitions(tiny_raw, tmp_path):

    out = tmp_path / "canonical"
    _build(tiny_raw, out)

    mentions = pq.read_table(out / MENTIONS_FILE)

    kinds = set(mentions.column("entity_kind").to_pylist())
    assert {"account", "card", "contract"} <= kinds

    rows = mentions.to_pylist()

    # Открытие продукта это переход договора, покупка по договору — нет.
    opened = [row for row in rows if row["entity_kind"] == "contract" and row["transition"] == "opened"]
    assert opened and all(row["event_type"] in ("product_opened", "account_opened") for row in opened)

    purchases = [row for row in rows if row["event_type"] == "purchase"]
    assert all(row["is_transition"] is False for row in purchases)

    # Каждое упоминание указывает на существующую строку ленты.
    events = pq.read_table(out / EVENTS_FILE, columns=["raw_row", "event_id"])
    known = set(events.column("raw_row").to_pylist())
    assert {row["raw_row"] for row in rows} <= known


def test_canonical_keeps_only_observed_transfer_sides(tmp_path):
    """
    Встречная сторона живёт у другого клиента и может быть
    проведена позже. Пометить исходящий перевод парным уже в
    canonical значит выдать знание из будущего: на раннем cutoff
    второй стороны ещё нет.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.cover_all("c2", first_seen="2023-01-01")

    mini.event(
        "c1", "p2p_out", "2023-03-05 10:00:00",
        payload=purchase_payload(amount=5000, direction="debit"),
        correlation_id="trf_1", link_type="transfer",
    )

    mini.event(
        "c1", "transfer_out", "2023-03-06 10:00:00", payload=purchase_payload(amount=7000, direction="debit"),
        correlation_id="trf_2", link_type="transfer",
    )

    # Та же операция глазами получателя: проведена на день позже.
    # В ленте она лежит своим блоком — строки клиента в выгрузке
    # идут подряд.
    mini.event(
        "c2", "p2p_in", "2023-03-06 09:00:00",
        payload=purchase_payload(amount=5000, direction="credit"),
        correlation_id="trf_1", link_type="transfer",
    )

    out = tmp_path / "canonical"
    _, report = _build(mini.write(), out)

    transfers = pq.read_table(out / TRANSFERS_FILE)

    # В схеме нет полей, раскрывающих встречную сторону.
    assert "counterpart_client_id" not in transfers.column_names
    assert "pair_status" not in transfers.column_names

    rows = transfers.to_pylist()

    outgoing = next(row for row in rows if row["transfer_id"] == "trf_1" and row["side"] == "out")
    incoming = next(row for row in rows if row["transfer_id"] == "trf_1" and row["side"] == "in")

    # Ни одно поле исходящей стороны не называет получателя.
    assert "c2" not in {value for value in outgoing.values() if isinstance(value, str)}

    # Обе стороны лежат отдельными строками со своим временем события,
    # поэтому этап истории соберёт пару только когда произошли обе.
    assert outgoing["event_time"] == datetime(2023, 3, 5, 10, 0)
    assert incoming["event_time"] == datetime(2023, 3, 6, 9, 0)

    # Статистика по всей выгрузке остаётся в отчёте качества и
    # прямо названа статистикой по выгрузке, а не признаком строки.
    stats = report["links"]["transfers"]
    assert stats["both_sides_in_full_extract"] == 1
    assert stats["one_side_in_full_extract"] == 1
    assert "этап истории" in stats["rule"]


def test_missing_cause_is_reported_not_invented(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    original = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    mini.event("c1", "refund", "2023-03-06 10:00:00", payload=purchase_payload(cause_event_id=original))
    mini.event("c1", "refund", "2023-03-07 10:00:00", payload=purchase_payload(cause_event_id="ev_нет_такого"))

    _, report = _build(mini.write(), tmp_path / "canonical")

    causes = report["links"]["causes"]

    assert causes["references"] == 2
    assert causes["resolved"] == 1
    assert causes["unresolved"] == {"target_not_in_dataset": 1}


def test_client_index_addresses_rows_inside_their_row_group(tiny_raw, tmp_path):
    """
    row_offset отсчитывается внутри своего row group, а не от начала
    файла: адрес обязан работать для групп после первой.
    """

    out = tmp_path / "canonical"
    _build(tiny_raw, out, replace(CONFIG, batch_clients=4))

    parquet = pq.ParquetFile(out / EVENTS_FILE)

    assert parquet.num_row_groups > 2, "нужен файл из нескольких row group"

    clients = pq.read_table(out / CLIENT_INDEX_FILE).to_pylist()

    later = [row for row in clients if row["row_count"] and row["row_group"] > 0]
    assert later, "все клиенты попали в первую группу: тест ничего не проверяет"

    for row in later[:6]:

        group = parquet.read_row_group(row["row_group"])

        assert row["row_offset"] < group.num_rows
        assert row["spans_row_groups"] is False

        block = group.slice(row["row_offset"], row["row_count"])

        assert set(block.column("client_id").to_pylist()) == {row["client_id"]}
        assert block.num_rows == row["row_count"]

        # Глобальная позиция адресует ту же самую строку.
        whole = pq.read_table(out / EVENTS_FILE, columns=["client_id"])
        assert whole.column("client_id")[row["global_row_start"]].as_py() == row["client_id"]


def test_client_without_events_stays_in_the_index(tmp_path):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.cover_all("c2", first_seen="2023-01-01")

    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())

    out = tmp_path / "canonical"
    _, report = _build(mini.write(), out)

    clients = pq.read_table(out / CLIENT_INDEX_FILE).to_pylist()

    assert [row["client_id"] for row in clients] == ["c1", "c2"]
    assert [row["client_idx"] for row in clients] == [0, 1]

    silent = clients[1]
    assert silent["row_count"] == 0 and silent["event_time_min"] is None
    assert silent["row_group"] is None and silent["row_offset"] is None

    assert report["rows"]["clients_without_events"] == 1


# ============================================================
# РЕЕСТР
# ============================================================


def test_registry_keeps_field_identity_per_event_type(tiny_raw, tmp_path):

    out = tmp_path / "canonical"
    _, report = _build(tiny_raw, out)

    registry = json.loads((out / REGISTRY_FILE).read_text(encoding="utf-8"))

    counts = registry["counts"]
    assert counts["by_owner_kind"]["payload"] == 872
    assert counts["distinct_payload_names"] == 72

    fields = {(item["owner"], item["name"]): item for item in registry["fields"]}

    # Одно имя, разные правила: у заявки канал обязателен, у операции нет.
    assert fields[("application_submitted", "channel")]["nullable"] is False
    assert fields[("purchase", "channel")]["nullable"] is True

    # Единицы взяты из описаний каталога, а не угаданы.
    assert fields[("purchase", "amount")]["unit"] == "KZT"
    assert fields[("product_opened", "term")]["unit"] == "months"
    assert fields[("purchase", "mcc")]["unit"] is None

    # Ссылочные поля помечены видом сущности.
    assert fields[("purchase", "account_id")]["reference_to"] == "account"

    assert report["registry"]["counts"] == counts


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ И ВОРОТА ЭТАПА
# ============================================================


def test_stage_version_tracks_sources():
    """
    Правка любого модуля этапа без поднятия STAGE_VERSION не
    проходит: иначе прежний canonical считался бы актуальным.
    """

    import hashlib

    from src.preprocessing import manifest as manifest_module
    from src.preprocessing import projection as projection_module
    from src.preprocessing import rawdata as rawdata_module
    from src.preprocessing import run as run_module
    from src.preprocessing import settings as settings_module
    from src.preprocessing.canonical import build as build_module
    from src.preprocessing.canonical import entities, events, links, registry, schema, sidecars

    modules = {
        "build": build_module,
        "events": events,
        "schema": schema,
        "registry": registry,
        "entities": entities,
        "links": links,
        "sidecars": sidecars,
        "projection": projection_module,
        "rawdata": rawdata_module,
        "settings": settings_module,
        "run": run_module,
    }

    stored = json.loads(Path("tests/prep_stage_sources.json").read_text(encoding="utf-8"))["canonical"]

    assert stored["version"] == build_module.STAGE_VERSION

    actual = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in modules.items()
    }

    assert stored["modules"] == actual, (
        "модули этапа изменены: поднимите STAGE_VERSION и обновите tests/prep_stage_sources.json"
    )


def test_canonical_is_deterministic(tmp_path):

    raw_dir = _versions_raw(tmp_path)

    first = build_group(raw_dir, tmp_path / "one", CONFIG, "train")
    second = build_group(raw_dir, tmp_path / "two", CONFIG, "train")

    assert dumps_json(first.report) == dumps_json(second.report)

    for path in first.outputs:
        other = tmp_path / "two" / path.relative_to(tmp_path / "one")
        assert path.read_bytes() == other.read_bytes(), path.name


def test_canonical_requires_passport_and_explicit_diagnostic_mode(tmp_path, capsys):

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload={**purchase_payload(), "zzz": 1})
    raw_dir = mini.write()

    out = tmp_path / "processed"

    # Без паспорта этап не запускается.
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_BLOCKED
    assert "passport" in capsys.readouterr().out

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_CONTRACT_MISMATCH
    capsys.readouterr()

    # Паспорт contract_mismatch: нужен явный режим диагностики.
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_CONTRACT_MISMATCH
    assert "--allow-contract-mismatch" in capsys.readouterr().out
    assert not (out / "canonical").exists()

    assert (
        _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out), "--allow-contract-mismatch")
        == EXIT_OK
    )

    report = json.loads((out / "canonical" / "train" / "canonical_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"

    entry = json.loads((out / "preprocessing_manifest.json").read_text(encoding="utf-8"))["stages"]["canonical"]["train"]
    assert entry["diagnostic_mode"] is True
    assert entry["passport_status"] == "contract_mismatch"

    # Повторный запуск пропускается по отпечатку.
    capsys.readouterr()
    assert (
        _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out), "--allow-contract-mismatch")
        == EXIT_OK
    )
    assert "пропущен" in capsys.readouterr().out


def test_canonical_refuses_a_stale_passport(tmp_path, capsys):
    """
    Паспорт, снятый с прежней выгрузки, ничего не говорит о нынешней.
    Если RAW изменился после этапа 1, canonical не собирается.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")
    mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload())
    raw_dir = mini.write()

    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    # Выгрузка переписана: та же схема, другое содержимое.
    other = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    other.cover_all("c1", first_seen="2023-01-01")
    other.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase_payload(amount=99999))
    other.write()

    capsys.readouterr()

    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_BLOCKED

    printed = capsys.readouterr().out
    assert "RAW изменился после паспорта" in printed
    assert not (out / "canonical").exists()

    # После повторного паспорта этап проходит.
    assert _cli("passport", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK
    assert _cli("canonical", "--raw", str(raw_dir), "--group", "train", "--out", str(out)) == EXIT_OK

    table = pq.read_table(out / "canonical" / "train" / EVENTS_FILE)
    assert table.column("amount").to_pylist() == [99999]


def test_canonical_refuses_a_blocked_passport(tiny_raw, tmp_path, capsys):

    import shutil

    copy = tmp_path / "raw"
    shutil.copytree(tiny_raw, copy)

    out = tmp_path / "processed"

    assert _cli("passport", "--raw", str(copy), "--group", "train", "--out", str(out)) in (
        EXIT_OK,
        EXIT_CONTRACT_MISMATCH,
    )

    (copy / "events.parquet").write_bytes(b"broken")

    assert _cli("passport", "--raw", str(copy), "--group", "train", "--out", str(out)) == EXIT_BLOCKED
    capsys.readouterr()

    assert (
        _cli("canonical", "--raw", str(copy), "--group", "train", "--out", str(out), "--allow-contract-mismatch")
        == EXIT_BLOCKED
    )
    assert "заблокирован" in capsys.readouterr().out
