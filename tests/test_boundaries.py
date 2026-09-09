"""
Границы окон и инвариантность префикса.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.generator.config import (
    FEATURE_END,
    HISTORY_START,
    LABEL_END,
    SOURCE_AVAILABILITY,
)
from src.generator.history import (
    EVENT_TABLES,
    STREAM_FIELDS,
    generate_client_history,
)
from src.generator.transactions import generate_transaction_history


CLIENTS = [0, 1, 3, 5]


@pytest.fixture(scope="module")
def histories():
    return {client_id: generate_client_history(client_id) for client_id in CLIENTS}


# ============================================================
# ГРАНИЦЫ
# ============================================================


def test_streams_inside_horizon(histories):
    for client_id, history in histories.items():

        assert history.start == HISTORY_START
        assert history.end == LABEL_END

        for source in STREAM_FIELDS:

            events = history.events(source)

            timestamps = [event.ts for event in events]

            assert timestamps == sorted(timestamps), source
            assert all(event.client_id == client_id for event in events), source

            if source == "product_events":
                # Реестр договоров старше окна наблюдения.
                assert all(ts < LABEL_END for ts in timestamps)
                assert all(
                    ts >= SOURCE_AVAILABILITY["product_events"] for ts in timestamps
                )
            else:
                assert all(HISTORY_START <= ts < LABEL_END for ts in timestamps), source


def test_reversed_interval_raises():
    with pytest.raises(ValueError):
        generate_client_history(0, start=LABEL_END, end=HISTORY_START)

    with pytest.raises(ValueError):
        generate_transaction_history(0, LABEL_END, HISTORY_START)


# ============================================================
# СРЕЗЫ
# ============================================================


def test_before_and_since_split_exactly(histories):
    for history in histories.values():

        feature = history.before(FEATURE_END)
        future = history.since(FEATURE_END)

        assert feature.end == FEATURE_END
        assert future.start == FEATURE_END

        for source in STREAM_FIELDS:

            assert all(e.ts < FEATURE_END for e in feature.events(source)), source
            assert all(e.ts >= FEATURE_END for e in future.events(source)), source

            assert (
                feature.events(source) + future.events(source)
                == history.events(source)
            ), source


def test_before_beyond_end_returns_same_object(histories):
    history = histories[0]

    assert history.before(LABEL_END) is history
    assert history.since(HISTORY_START) is history


# ============================================================
# ИНВАРИАНТНОСТЬ ПРЕФИКСА
# ============================================================


def test_extending_horizon_keeps_feature_prefix():
    """
    Сгенерировать до LABEL_END и отрезать по FEATURE_END это
    то же самое, что сгенерировать сразу до FEATURE_END.
    """

    for client_id in CLIENTS:

        long = generate_client_history(client_id, start=HISTORY_START, end=LABEL_END)
        short = generate_client_history(client_id, start=HISTORY_START, end=FEATURE_END)

        prefix = long.before(FEATURE_END)

        assert prefix.transactions == short.transactions, client_id
        assert prefix.profile == short.profile, client_id


def test_transactions_prefix_is_stable():
    for client_id in CLIENTS:

        long = generate_transaction_history(client_id, HISTORY_START, LABEL_END)
        short = generate_transaction_history(client_id, HISTORY_START, FEATURE_END)

        assert [e for e in long if e.ts < FEATURE_END] == short


# ============================================================
# RAW
# ============================================================


def test_raw_never_leaks_the_future(raw_tables):
    for source in STREAM_FIELDS:

        table = raw_tables[source]

        if table.empty:
            continue

        assert table.ts.max() < pd.Timestamp(FEATURE_END), source


def test_raw_timeline_stops_at_cutoff(raw_tables):
    assert raw_tables["timeline"].ts.max() < pd.Timestamp(FEATURE_END)


def test_labels_window(raw_tables, emit_clients):
    labels = raw_tables["labels"]

    assert len(labels) == emit_clients
    assert (labels.label_start == pd.Timestamp(FEATURE_END)).all()
    assert (labels.label_end == pd.Timestamp(LABEL_END)).all()
