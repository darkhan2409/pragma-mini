"""
Дефекты покрытия источников.

1. availability_start: до этой даты в источнике нет ничего.
2. client_first_seen_in_source: клиент появляется позже,
   а часть клиентов не появляется никогда.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.generator.config import (
    FEATURE_END,
    HISTORY_START,
    SOURCE_AVAILABILITY,
    SOURCES,
)
from src.generator.coverage import (
    app_adoption,
    consent_date,
    coverage_rows,
    first_seen,
    is_visible,
)
from src.generator.history import STREAM_FIELDS, generate_client_history


POPULATION = 600


# ============================================================
# FIRST SEEN
# ============================================================


def test_first_seen_never_before_availability():
    for client_id in range(POPULATION):
        for source in SOURCES:

            seen = first_seen(client_id, source)

            if seen is None:
                continue

            assert seen >= SOURCE_AVAILABILITY[source], (client_id, source)


def test_first_seen_is_deterministic():
    for source in SOURCES:
        assert first_seen(11, source) == first_seen(11, source)


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        first_seen(0, "nonexistent")


def test_some_clients_never_appear_in_optional_sources():
    """
    Приложение и согласие на связь есть не у всех.
    """

    for source in ("app_screens", "app_operations", "banners"):

        missing = sum(
            1 for client_id in range(POPULATION) if first_seen(client_id, source) is None
        )

        assert 0.05 < missing / POPULATION < 0.45, (source, missing)

    missing_comms = sum(
        1 for client_id in range(POPULATION) if first_seen(client_id, "communications") is None
    )

    assert 0.01 < missing_comms / POPULATION < 0.25


def test_mandatory_sources_cover_everyone():
    for source in ("profile", "transactions", "product_events"):
        for client_id in range(POPULATION):
            assert first_seen(client_id, source) is not None


def test_app_sources_share_adoption_date():
    for client_id in range(200):

        adoption = app_adoption(client_id)

        if adoption is None:
            assert first_seen(client_id, "app_screens") is None
            continue

        for source in ("app_screens", "app_operations", "banners"):
            assert first_seen(client_id, source) == max(
                SOURCE_AVAILABILITY[source], adoption
            )


def test_communications_follow_consent():
    for client_id in range(200):

        granted = consent_date(client_id)

        if granted is None:
            assert first_seen(client_id, "communications") is None
        else:
            assert first_seen(client_id, "communications") == max(
                SOURCE_AVAILABILITY["communications"], granted
            )


def test_is_visible_matches_first_seen():
    seen = first_seen(5, "transactions")

    assert seen is not None
    assert not is_visible(5, "transactions", seen - pd.Timedelta(seconds=1).to_pytimedelta())
    assert is_visible(5, "transactions", seen)


# ============================================================
# ФИЛЬТРАЦИЯ ПОТОКОВ
# ============================================================


@pytest.mark.parametrize("client_id", [0, 1, 2, 3, 4, 5])
def test_history_respects_coverage(client_id):
    history = generate_client_history(client_id)

    for source in STREAM_FIELDS:

        seen = first_seen(client_id, source)

        events = history.events(source)

        if seen is None:
            assert events == [], source
            continue

        assert all(event.ts >= seen for event in events), source


def test_raw_respects_coverage(raw_tables, emit_clients):
    for source in STREAM_FIELDS:

        table = raw_tables[source]

        for client_id in range(emit_clients):

            seen = first_seen(client_id, source)

            client_rows = table[table.client_id == client_id]

            if seen is None:
                assert client_rows.empty, (source, client_id)
                continue

            if client_rows.empty:
                continue

            assert client_rows.ts.min() >= pd.Timestamp(seen), (source, client_id)


def test_late_sources_have_no_early_rows(raw_tables):
    """
    У витрин с поздним подключением нет строк за первые месяцы,
    хотя транзакции за те же месяцы есть.
    """

    for source in ("communications", "app_screens"):

        table = raw_tables[source]

        assert table.ts.min() >= pd.Timestamp(SOURCE_AVAILABILITY[source]), source

    assert raw_tables["transactions"].ts.min() < pd.Timestamp(
        SOURCE_AVAILABILITY["communications"]
    )


# ============================================================
# ТАБЛИЦА ПОКРЫТИЯ
# ============================================================


def test_coverage_table(raw_tables, emit_clients):
    coverage = raw_tables["source_coverage"]

    assert len(coverage) == emit_clients * len(SOURCES)
    assert set(coverage.source) == set(SOURCES)

    for source in SOURCES:
        rows = coverage[coverage.source == source]
        assert (rows.availability_start == pd.Timestamp(SOURCE_AVAILABILITY[source])).all()

    # Есть и клиенты без источника, и клиенты с поздним появлением.
    assert coverage.first_seen.isna().any()
    assert (coverage.first_seen > pd.Timestamp(HISTORY_START)).any()


def test_coverage_rows_match_first_seen():
    for row in coverage_rows(9):
        assert row["first_seen"] == first_seen(9, row["source"])
        assert row["availability_start"] == SOURCE_AVAILABILITY[row["source"]]


def test_product_registry_reaches_before_history(raw_tables):
    """
    Реестр договоров старше окна наблюдения: часть открытий
    произошла до HISTORY_START.
    """

    products = raw_tables["product_events"]

    assert (products.ts < pd.Timestamp(HISTORY_START)).any()
    assert products.ts.min() >= pd.Timestamp(SOURCE_AVAILABILITY["product_events"])
