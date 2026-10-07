from __future__ import annotations

import json
from datetime import date, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.preprocessing.canonical.build import build_group
from src.preprocessing.canonical.events import CanonicalError
from src.preprocessing.rawdata import RawContractError
from src.preprocessing.settings import PreprocessingConfig

from tests.test_profile_state import AS_OF, RAW_CLIENT, raw_event, when, write_raw


# ============================================================
# ИДЕЯ
# ============================================================
#
# Препроцессинг обязан не только перевести годную выгрузку, но и
# остановиться на негодной — с ошибкой, которая называет поломку.
# Каждый случай ниже — одна поломка контракта RAW поверх годной
# ленты; этап обязан её отвергнуть, а не пропустить молча или
# упасть чужим исключением.
#
# Ускорение этапа (один разбор type и времени, проверка вместе с
# переводом) не имеет права ослабить ни одну из этих проверок.
# ============================================================


UTC = timezone.utc

T1 = "2025-06-10T10:00:00"


def purchase(moment: str = "2025-03-01T09:00:00", **fields) -> dict:
    return raw_event(RAW_CLIENT, moment, {
        "type": "purchase", "amount": 700, "direction": "debit", "status": "approved", **fields})


def card(moment: str, card_id: str) -> dict:
    return raw_event(RAW_CLIENT, moment, {
        "type": "card_activated", "product_id": "prd_card", "card_id": card_id,
        "reason": "application_approved"})


GOOD = [purchase("2025-03-01T09:00:00"), purchase("2025-03-02T09:00:00")]


def edited(index: int, **envelope) -> list[dict]:
    """Годная лента, у строки index заменены поля конверта."""
    events = [dict(item) for item in GOOD]
    events[index].update(envelope)
    return events


def with_payload(payload: object) -> list[dict]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return edited(1, payload=text)


def milestone(kind: str, moment: str, source_id: str | None) -> dict:
    return {"type": kind, "event_time": when(moment).astimezone(UTC), "source_id": source_id}


def job(start: date, record: str) -> dict:
    return {"start_date": start, "record_time": when(record).astimezone(UTC)}


def duplicate_profile(raw) -> None:
    """Вторая строка профиля того же клиента; манифест согласован."""
    table = pq.read_table(raw / "profile.parquet")
    pq.write_table(pa.concat_tables([table, table]), raw / "profile.parquet", compression="zstd")
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    manifest["profile_rows"] = 2
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# (имя, события, снимок, правка выгрузки, класс ошибки, фрагмент сообщения)
CASES = [
    ("null client_id", edited(1, client_id=None), {}, None, RawContractError, "пустой client_id"),
    ("empty client_id", edited(1, client_id="  "), {}, None, RawContractError, "пустой client_id"),
    ("invalid timestamp", edited(1, event_time="2025-13-40T09:00:00+05:00"), {}, None,
     RawContractError, "не читается как ISO 8601"),
    ("timezone missing", edited(1, event_time="2025-03-02T09:00:00"), {}, None,
     RawContractError, "без часового пояса"),
    ("bad source", edited(1, source="telepathy"), {}, None, RawContractError, "источники вне контракта"),
    ("malformed JSON", with_payload('{"type": "purchase", "amount": '), {}, None,
     RawContractError, "не разбирается как JSON"),
    ("empty payload", with_payload(""), {}, None, RawContractError, "payload пуст"),
    ("missing type", with_payload({"amount": 700, "direction": "debit", "status": "approved"}), {}, None,
     RawContractError, "нет ключа 'type'"),
    ("unknown event type", with_payload({"type": "teleport"}), {}, None,
     CanonicalError, "не объявлен каталогом ключей"),
    ("unexpected key", with_payload({"type": "purchase", "amount": 700, "direction": "debit",
                                     "status": "approved", "colour": "red"}), {}, None,
     CanonicalError, "unexpected_key:colour"),
    ("wrong field type", with_payload({"type": "purchase", "amount": "семьсот", "direction": "debit",
                                       "status": "approved"}), {}, None,
     CanonicalError, "type_mismatch:amount"),
    ("missing required field", with_payload({"type": "purchase", "direction": "debit",
                                             "status": "approved"}), {}, None,
     CanonicalError, "missing_required:amount"),
    ("split client", GOOD[:1] + [raw_event("c000000000002", "2025-03-01T10:00:00", {
        "type": "purchase", "amount": 1, "direction": "debit", "status": "approved"})] + GOOD[1:], {}, None,
     CanonicalError, "не подряд"),
    ("duplicate profile", GOOD, {}, duplicate_profile, RawContractError, "больше одной строки профиля"),
    ("bad as_of", GOOD, {"as_of": AS_OF.replace(day=2)}, None, RawContractError, "as_of"),
    ("milestone without source_id", GOOD + [card(T1, "crd_1")],
     {"lifelong": [milestone("first_card_activated", T1, None)]}, None, RawContractError, "нет source_id"),
    ("milestone source missing in tape", GOOD,
     {"lifelong": [milestone("first_card_activated", T1, "crd_1")]}, None, CanonicalError, "в ленте нет"),
    ("milestone time differs from its source", GOOD + [card(T1, "crd_1")],
     {"lifelong": [milestone("first_card_activated", "2025-06-11T10:00:00", "crd_1")]}, None,
     CanonicalError, "а веха — в"),
    ("employment not in time order", GOOD,
     {"employment": [job(date(2023, 1, 1), "2025-02-01T10:00:00"), job(date(2020, 1, 1), "2025-01-01T10:00:00")]},
     None, RawContractError, "не по времени"),
    ("employment after as_of", GOOD, {"employment": [job(date(2023, 1, 1), "2026-08-01T10:00:00")]}, None,
     RawContractError, "не раньше as_of"),
]


def run_case(directory, events: list[dict], snapshot: dict, edit) -> None:

    raw = directory / "raw"

    write_raw(raw, events, snapshot)

    if edit is not None:
        edit(raw)

    build_group(raw, directory / "out", PreprocessingConfig.load(None), "val")


def test_the_untouched_tape_passes(tmp_path):

    run_case(tmp_path, GOOD + [card(T1, "crd_1")], {"lifelong": [milestone("first_card_activated", T1, "crd_1")]},
             None)

    table = pq.read_table(tmp_path / "out" / "events.parquet")

    assert table.num_rows == 3
    assert table.column("lifelong_source").to_pylist().count("first_card_activated") == 1


@pytest.mark.parametrize("name, events, snapshot, edit, error, fragment", CASES, ids=[case[0] for case in CASES])
def test_a_broken_raw_is_refused(tmp_path, name, events, snapshot, edit, error, fragment):

    with pytest.raises(error, match=fragment):
        run_case(tmp_path, events, snapshot, edit)

