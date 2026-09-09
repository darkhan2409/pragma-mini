"""
Представление события: тройки (key_id, value_id, позиция).

Golden-векторы зафиксированы литералами и не перегенерируются
кодом. Ручной мини-словарь проверяет правила, реальные данные
проверяют, что правила применяются к настоящему processed.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.tokenizer.config import EVT_ID, MISSING_ID, UNK_ID, USR_ID
from src.tokenizer.encode import (
    EVENT_FIELDS,
    EVENT_WIDTH,
    PROFILE_WIDTH,
    encode_event,
    encode_events_table,
    encode_pairs,
    encode_profile,
    encode_profile_table,
    event_pairs,
    profile_pairs,
)
from src.tokenizer.vocab import KeyEntry, ValueEntry, Vocab, fit_vocab


# ============================================================
# РУЧНОЙ МИНИ-СЛОВАРЬ
# ============================================================


def toy_vocab() -> Vocab:
    """
    Три ключа, шесть значений. Константы заданы здесь, а не
    прочитаны из файлов: тест обязан ломаться от изменения
    правил, а не от изменения данных.
    """

    keys = [
        KeyEntry(6, "toy__color", "toy", "color", "categorical", True, "string", 9, 11),
        KeyEntry(7, "toy__size", "toy", "size", "numeric", True, "int64", 11, 13),
        KeyEntry(8, "toy__flag", "toy", "flag", "boolean", False, "bool", 13, 15),
    ]

    values = [
        ValueEntry(9, 6, "toy__color", "blue", 5),
        ValueEntry(10, 6, "toy__color", "red", 3),
        ValueEntry(11, 7, "toy__size", "0", 4),
        ValueEntry(12, 7, "toy__size", "1", 4),
        ValueEntry(13, 8, "toy__flag", "false", 2),
        ValueEntry(14, 8, "toy__flag", "true", 6),
    ]

    return Vocab(keys, values)


@pytest.fixture()
def toy() -> Vocab:
    return toy_vocab()


# ============================================================
# ПРАВИЛА НА МИНИ-СЛОВАРЕ
# ============================================================


def test_event_starts_with_evt_in_both_arrays(toy):
    record = encode_pairs(toy, [("toy__color", "red")], EVT_ID)

    assert record.key_ids[0] == EVT_ID
    assert record.value_ids[0] == EVT_ID
    assert record.positions[0] == 0


def test_profile_starts_with_usr(toy):
    record = encode_pairs(toy, [("toy__color", "red")], USR_ID)

    assert record.key_ids[0] == USR_ID
    assert record.value_ids[0] == USR_ID


def test_three_arrays_have_one_length(toy):
    record = encode_pairs(toy, [("toy__color", "red"), ("toy__size", 1)], EVT_ID)

    assert len(record.key_ids) == len(record.value_ids) == len(record.positions) == 3


def test_positions_start_at_one_for_fields(toy):
    record = encode_pairs(toy, [("toy__color", "blue"), ("toy__size", 0), ("toy__flag", True)], EVT_ID)

    assert list(record.positions) == [0, 1, 2, 3]


def test_fields_follow_registry_order_not_input_order(toy):
    record = encode_pairs(
        toy,
        [("toy__flag", True), ("toy__size", 1), ("toy__color", "red")],
        EVT_ID,
    )

    assert list(record.key_ids) == [EVT_ID, 6, 7, 8]
    assert list(record.value_ids) == [EVT_ID, 10, 12, 14]


def test_missing_value_becomes_missing_token(toy):
    record = encode_pairs(toy, [("toy__color", None)], EVT_ID)

    assert list(record.value_ids) == [EVT_ID, MISSING_ID]

    # Ключ остаётся настоящим: [MISSING] это значение, не позиция.
    assert list(record.key_ids) == [EVT_ID, 6]


def test_unknown_value_becomes_unk(toy):
    record = encode_pairs(toy, [("toy__color", "green")], EVT_ID)

    assert list(record.value_ids) == [EVT_ID, UNK_ID]
    assert list(record.key_ids) == [EVT_ID, 6]


def test_unknown_key_is_unk_in_both_arrays_and_goes_last(toy):
    record = encode_pairs(
        toy,
        [("toy__mystery", 1), ("toy__color", "red")],
        EVT_ID,
    )

    assert list(record.key_ids) == [EVT_ID, 6, UNK_ID]
    assert list(record.value_ids) == [EVT_ID, 10, UNK_ID]


def test_unknown_keys_keep_input_order(toy):
    record = encode_pairs(toy, [("b__second", 1), ("a__first", 2)], EVT_ID)

    assert len(record) == 3
    assert list(record.key_ids) == [EVT_ID, UNK_ID, UNK_ID]


def test_repeated_key_keeps_every_value_in_order(toy):
    """
    Список пар, а не dict: повторяющийся ключ не схлопывается.
    """

    record = encode_pairs(
        toy,
        [("toy__color", "red"), ("toy__color", "blue"), ("toy__size", 1)],
        EVT_ID,
    )

    assert list(record.key_ids) == [EVT_ID, 6, 6, 7]
    assert list(record.value_ids) == [EVT_ID, 10, 9, 12]
    assert list(record.positions) == [0, 1, 2, 3]


def test_numeric_value_is_a_bucket(toy):
    record = encode_pairs(toy, [("toy__size", 1)], EVT_ID)

    assert list(record.value_ids) == [EVT_ID, 12]


def test_boolean_values_are_separate_ids(toy):
    assert toy.value_id(8, "true") != toy.value_id(8, "false")


def test_same_string_in_two_fields_would_be_two_ids(toy):
    assert toy.value_id(6, "blue") != toy.value_id(7, "0")


# ============================================================
# GOLDEN НА РЕАЛЬНЫХ ДАННЫХ
# ============================================================
#
# Литералы сняты вручную один раз и привязаны к:
#   PREP_CLIENTS = 100, PREP_CHUNK = 25 (tests/conftest.py),
#   Settings() по умолчанию, замороженный генератор.
# Изменение любого из них требует ОСОЗНАННОГО ручного
# обновления литералов; автоматически они не перегенерируются.

GOLDEN_REF = {
    "client_id": 0,
    "cutoff": "2024-09-01T00:00:00",
    "seq_end": 116,
    "snapshot_ts": "2024-08-31T23:59:59",
    "n_events": 116,
    "n_tokens": 1065,
}

GOLDEN_PROFILE = [
    (0, "[USR]", "[USR]"),
    (1, "profile__age", "9"),
    (2, "profile__gender", "M"),
    (3, "profile__family_status", "single"),
    (4, "profile__children", "2"),
    (5, "profile__education", "higher"),
    (6, "profile__region", "Shymkent"),
    (7, "profile__housing_type", "with_parents"),
    (8, "profile__pensioner", "false"),
    (9, "profile__income_type", "self_employed"),
    (10, "profile__declared_income", "12"),
    (11, "profile__industry", "[MISSING]"),
    (12, "profile__salary_day", "1"),
    (13, "profile__relationship_months", "0"),
    (14, "profile__contracts_count", "2"),
    (15, "profile__active_contracts", "2"),
    (16, "profile__holds_credit_card", "true"),
    (17, "profile__holds_debit_card", "true"),
    (18, "profile__holds_deposit", "false"),
    (19, "profile__credit_limit", "14"),
    (20, "profile__credit_utilization", "6"),
]

GOLDEN_EVENT_0 = {"seq": 0, "ts": "2024-04-02T11:30:00", "event_type": "product_event"}

GOLDEN_EVENT_0_TOKENS = [
    (0, "[EVT]", "[EVT]"),
    (1, "timeline__event_type", "product_event"),
    (2, "product_event__product_type", "debit_card"),
    (3, "product_event__amount_or_limit", "[MISSING]"),
    (4, "product_event__term", "[MISSING]"),
    (5, "product_event__product_subtype", "arna"),
    (6, "product_event__timestamp_quality", "exact"),
]

GOLDEN_EVENT_0_KEY_IDS = [3, 6, 16, 17, 18, 19, 20]
GOLDEN_EVENT_0_VALUE_IDS = [3, 67, 149, 5, 5, 174, 189]

GOLDEN_EVENT_1 = {"seq": 1, "ts": "2024-04-08T00:00:00", "event_type": "product_event"}

GOLDEN_EVENT_1_TOKENS = [
    (0, "[EVT]", "[EVT]"),
    (1, "timeline__event_type", "product_event"),
    (2, "product_event__product_type", "cash_loan"),
    (3, "product_event__amount_or_limit", "13"),
    (4, "product_event__term", "3"),
    (5, "product_event__product_subtype", "standard"),
    (6, "product_event__timestamp_quality", "date_only"),
]


def golden(tok_run) -> dict:
    import json

    return json.loads((tok_run["vocab"] / "golden_examples.json").read_text(encoding="utf-8"))["examples"]["train"]


def test_golden_example_reference_is_fixed(tok_run):
    example = golden(tok_run)

    assert {key: example[key] for key in GOLDEN_REF} == GOLDEN_REF


def test_golden_profile_is_fixed(tok_run):
    tokens = [(row[0], row[1], row[2]) for row in golden(tok_run)["profile"]["tokens"]]

    assert tokens == GOLDEN_PROFILE
    assert len(tokens) == PROFILE_WIDTH


def test_golden_first_events_are_fixed(tok_run):
    events = golden(tok_run)["events"]

    for expected, expected_tokens, event in (
        (GOLDEN_EVENT_0, GOLDEN_EVENT_0_TOKENS, events[0]),
        (GOLDEN_EVENT_1, GOLDEN_EVENT_1_TOKENS, events[1]),
    ):
        assert {key: event[key] for key in expected} == expected
        assert [(row[0], row[1], row[2]) for row in event["tokens"]] == expected_tokens


def test_golden_ids_are_fixed(tok_run):
    tokens = golden(tok_run)["events"][0]["tokens"]

    assert [row[3] for row in tokens] == GOLDEN_EVENT_0_KEY_IDS
    assert [row[4] for row in tokens] == GOLDEN_EVENT_0_VALUE_IDS


# ============================================================
# ВЕКТОРНЫЙ ПУТЬ РАВЕН ЭТАЛОНУ
# ============================================================


@pytest.fixture(scope="module")
def fitted(prep_run):
    vocab, _ = fit_vocab(prep_run["processed"], prep_run["artifacts"])
    return vocab


def test_vectorised_events_match_reference(prep_run, fitted):
    events = pq.read_table(prep_run["processed"] / "clients" / "train_clients" / "events.parquet")

    step = max(1, events.num_rows // 2000)

    sample = events.take(list(range(0, events.num_rows, step))[:2000])

    tokenized = encode_events_table(sample, fitted)

    rows = sample.to_pylist()
    produced = tokenized.to_pylist()

    for row, made in zip(rows, produced):

        reference = encode_event(fitted, row["event_type"], event_pairs(row["event_type"], row))

        assert made["key_ids"] == list(reference.key_ids)
        assert made["value_ids"] == list(reference.value_ids)
        assert made["positions"] == list(reference.positions)


def test_vectorised_profile_matches_reference(prep_run, fitted):
    profile = pq.read_table(prep_run["processed"] / "clients" / "train_clients" / "profile.parquet")

    tokenized = encode_profile_table(profile, fitted)

    for row, made in zip(profile.to_pylist(), tokenized.to_pylist()):

        reference = encode_profile(fitted, profile_pairs(row))

        assert made["key_ids"] == list(reference.key_ids)
        assert made["value_ids"] == list(reference.value_ids)


# ============================================================
# ПОЗИЦИИ И ШИРИНА
# ============================================================


def test_positions_reset_inside_every_event(tok_run):
    events = pq.read_table(tok_run["tokenized"] / "clients" / "train_clients" / "events.parquet")

    for row in events.slice(0, 5000).to_pylist():
        assert row["positions"] == list(range(row["n_tokens"]))


def test_width_includes_evt_and_event_type(tok_run):
    events = pq.read_table(tok_run["tokenized"] / "clients" / "train_clients" / "events.parquet")

    widths = {}

    for row in events.slice(0, 20000).to_pylist():
        widths.setdefault(row["event_type"], set()).add(row["n_tokens"])

    for event_type, seen in widths.items():
        assert seen == {EVENT_WIDTH[event_type]}
        assert EVENT_WIDTH[event_type] == 2 + len(EVENT_FIELDS[event_type])


def test_metadata_is_not_a_token(tok_run):
    events = pq.read_table(tok_run["tokenized"] / "clients" / "train_clients" / "events.parquet")

    row = next(
        item
        for item in events.slice(0, 20000).to_pylist()
        if item["event_type"] == "app_screen"
    )

    from src.tokenizer.vocab import Vocab

    vocab = Vocab.load(tok_run["vocab"])

    names = [vocab.decode(key) for key in row["key_ids"]]

    assert "app_screen__session_id" not in names
    assert len(names) == EVENT_WIDTH["app_screen"]


def test_profile_width_is_fixed(tok_run):
    profile = pq.read_table(tok_run["tokenized"] / "clients" / "train_clients" / "profile.parquet")

    assert set(profile.column("n_tokens").to_pylist()) == {PROFILE_WIDTH}

    lead = np.asarray(profile.column("value_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False))

    assert (lead[::PROFILE_WIDTH] == USR_ID).all()
