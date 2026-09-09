"""
Единая лента событий: контракт, порядок, tie-break, payload.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import pytest

from src.generator.config import (
    EVENT_TYPE_BY_SOURCE,
    EVENT_TYPE_PRIORITY,
    EVENT_TYPES,
    FEATURE_END,
    MAX_EVENTS_PER_HISTORY,
    MAX_TOKENS_PER_EVENT,
    PROFILE_DYNAMIC_FIELDS,
)
from src.generator.history import STREAM_FIELDS, generate_client_history, observed
from src.generator.timeline import (
    PAYLOAD_BUILDERS,
    build_timeline,
    payload_json,
    timeline_rows,
)


CLIENT = 1


@pytest.fixture(scope="module")
def history():
    return observed(generate_client_history(CLIENT).before(FEATURE_END))


@pytest.fixture(scope="module")
def timeline(history):
    return build_timeline(history)


# ============================================================
# КОНТРАКТ
# ============================================================


def test_timeline_covers_all_streams(timeline, history):
    assert len(timeline) == sum(len(history.events(source)) for source in STREAM_FIELDS)

    types = {event.event_type for event in timeline}

    assert types <= set(EVENT_TYPES)
    assert "transaction" in types
    assert "profile_snapshot" in types


def test_event_contract_fields(timeline):
    for event in timeline:
        assert event.client_id == CLIENT
        assert event.event_type in EVENT_TYPE_PRIORITY
        assert isinstance(event.payload, dict)
        assert event.payload, "payload не может быть пустым"


def test_payload_fits_token_budget(timeline):
    for event in timeline:
        # event_type и ts тоже занимают место в бюджете события.
        assert len(event.payload) + 2 <= MAX_TOKENS_PER_EVENT


def test_profile_payload_is_dynamic_subset(timeline):
    snapshots = [e for e in timeline if e.event_type == "profile_snapshot"]

    assert snapshots

    for event in snapshots:
        assert set(event.payload) == set(PROFILE_DYNAMIC_FIELDS)


# ============================================================
# ПОРЯДОК И TIE-BREAK
# ============================================================


def test_timeline_is_sorted_and_seq_is_dense(timeline):
    assert [event.seq for event in timeline] == list(range(len(timeline)))

    timestamps = [event.ts for event in timeline]

    assert timestamps == sorted(timestamps)


def test_equal_timestamps_exist(timeline):
    """
    Совпадающие ts обязаны встречаться: баннер делит момент
    с экраном, договоры без времени падают в полночь.
    """

    equal = sum(1 for a, b in zip(timeline, timeline[1:]) if a.ts == b.ts)

    assert equal > 0


def test_tie_break_follows_event_type_priority(timeline):
    for previous, current in zip(timeline, timeline[1:]):

        if previous.ts != current.ts:
            continue

        assert (
            EVENT_TYPE_PRIORITY[previous.event_type]
            <= EVENT_TYPE_PRIORITY[current.event_type]
        )


def test_tie_break_is_stable_under_input_permutation(history):
    """
    Порядок ленты не должен зависеть от порядка, в котором
    потоки пришли в сборку.
    """

    baseline = build_timeline(history)

    shuffled = history

    for source in STREAM_FIELDS.values():
        events = list(getattr(shuffled, source))
        events.reverse()
        object.__setattr__(shuffled, source, events)

    permuted = build_timeline(shuffled)

    # Возвращаем порядок, чтобы не портить фикстуру модуля.
    for source in STREAM_FIELDS.values():
        events = list(getattr(shuffled, source))
        events.reverse()
        object.__setattr__(shuffled, source, events)

    assert [(e.ts, e.event_type) for e in permuted] == [
        (e.ts, e.event_type) for e in baseline
    ]


def test_banner_shares_timestamp_with_screen(history):
    screen_times = {event.ts for event in history.app_screens}

    shown = [event for event in history.banners if event.action == "shown"]

    assert shown
    assert any(event.ts in screen_times for event in shown)


# ============================================================
# СЕРИАЛИЗАЦИЯ
# ============================================================


def test_payload_json_roundtrip(timeline):
    for event in timeline[:200]:

        text = payload_json(event.payload)

        parsed = json.loads(text)

        assert set(parsed) == set(event.payload)


def test_timeline_rows_shape(history):
    rows = timeline_rows(history)

    assert rows

    for row in rows[:50]:
        assert set(row) == {"client_id", "ts", "seq", "event_type", "payload"}
        assert isinstance(row["payload"], str)


# ============================================================
# RAW
# ============================================================


def test_raw_timeline_matches_streams(raw_tables, emit_clients):
    timeline = raw_tables["timeline"]

    total = sum(len(raw_tables[source]) for source in STREAM_FIELDS)

    assert len(timeline) == total

    counts = timeline.event_type.value_counts().to_dict()

    for source, event_type in EVENT_TYPE_BY_SOURCE.items():
        assert counts.get(event_type, 0) == len(raw_tables[source]), source


def test_raw_timeline_sorted_per_client(raw_tables, emit_clients):
    timeline = raw_tables["timeline"]

    for client_id in range(emit_clients):

        rows = timeline[timeline.client_id == client_id]

        if rows.empty:
            continue

        assert rows.seq.tolist() == list(range(len(rows)))
        assert rows.ts.is_monotonic_increasing


def test_raw_timeline_length_within_budget(raw_tables):
    lengths = raw_tables["timeline"].groupby("client_id").size()

    assert lengths.max() <= MAX_EVENTS_PER_HISTORY


def test_raw_timeline_payload_parses(raw_tables):
    payloads = raw_tables["timeline"].payload.head(500)

    for text in payloads:
        assert isinstance(json.loads(text), dict)


# ============================================================
# ЛЕНТА И ТАБЛИЦЫ ЭТО ОДНИ И ТЕ ЖЕ ДАННЫЕ
# ============================================================


def test_raw_timeline_payload_matches_tables(raw_tables):
    """
    Регресс: раньше шум применялся только к таблицам, и одно
    событие выглядело в ленте иначе, чем в своей таблице.
    """

    timeline = raw_tables["timeline"]

    for source, event_type in EVENT_TYPE_BY_SOURCE.items():

        if source == "profile":
            continue

        table = raw_tables[source].sort_values(["client_id", "ts"]).reset_index(drop=True)

        rows = timeline[timeline.event_type == event_type]
        rows = rows.sort_values(["client_id", "ts", "seq"]).reset_index(drop=True)

        assert len(rows) == len(table), source

        payloads = [json.loads(text) for text in rows.payload]

        for index, payload in enumerate(payloads):

            for field, value in payload.items():

                actual = table.at[index, field]

                if actual is None or (isinstance(actual, float) and pd.isna(actual)):
                    actual = None
                elif hasattr(actual, "item"):
                    actual = actual.item()

                assert actual == value, (source, field, index, actual, value)


def test_noise_is_visible_in_both_representations(raw_tables):
    """
    Шум обязан быть виден и в таблице, и в ленте одновременно.
    """

    timeline = raw_tables["timeline"]

    screens = raw_tables["app_screens"]

    lost = screens[screens.firebase_screen == "(not set)"]

    assert not lost.empty, "шум GA4 не сработал, проверять нечего"

    payloads = [
        json.loads(text)
        for text in timeline[timeline.event_type == "app_screen"].payload
    ]

    assert sum(1 for p in payloads if p["firebase_screen"] == "(not set)") == len(lost)
