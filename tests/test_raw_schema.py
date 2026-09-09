"""
Схема RAW: состав таблиц, типы, отсутствие latent-полей, манифест.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.generator.categories import ALL_MCCS, DIRECTIONS
from src.generator.config import (
    EVENT_TYPE_PRIORITY,
    EVENT_TYPES,
    FEATURE_END,
    HISTORY_START,
    LABEL_END,
    MAX_EVENTS_PER_HISTORY,
    MAX_TOKENS_PER_EVENT,
    PROFILE_FIELDS,
    SEED,
    SOURCE_AVAILABILITY,
    SOURCES,
)
from src.generator.emit import SCHEMAS
from src.generator.products import PRODUCT_TYPES, TIMESTAMP_QUALITIES
from src.generator.world import (
    ACTION_CLICKED,
    ACTION_SHOWN,
    APP_DOMAINS,
    BANNER_OFFERS,
    BANNER_SLOTS,
    COMM_CHANNELS,
    FUNNEL_STAGES,
    OPERATION_STATUSES,
    REJECT_REASONS,
)


# Скрытые поля генератора: в RAW их быть не должно нигде.
LATENT_COLUMNS = {
    "activity",
    "digital_affinity",
    "mobility",
    "credit_need",
    "risk",
    "volatility",
    "push_reachable",
    "campaign",
    "clicked",
    "outcome",
    "activity_scenario",
    "stress_scenario",
    "credit_stress",
    "browsed_products",
}


# ============================================================
# СОСТАВ
# ============================================================


def test_all_tables_written(raw_dir, raw_tables):
    for name, schema in SCHEMAS.items():
        assert (raw_dir / f"{name}.parquet").exists(), name
        assert list(raw_tables[name].columns) == schema.names, name


@pytest.mark.parametrize("name", list(SCHEMAS))
def test_no_latent_columns(name, raw_tables):
    leaked = LATENT_COLUMNS & set(raw_tables[name].columns)

    assert not leaked, f"{name}: latent-колонки в RAW: {sorted(leaked)}"


def test_one_label_per_client(raw_tables, emit_clients):
    assert sorted(raw_tables["labels"].client_id) == list(range(emit_clients))


@pytest.mark.parametrize(
    "name",
    [
        "transactions",
        "product_events",
        "communications",
        "app_screens",
        "app_operations",
        "banners",
        "profile",
        "timeline",
    ],
)
def test_client_ids_are_known(name, raw_tables, emit_clients):
    table = raw_tables[name]

    if table.empty:
        return

    assert table.client_id.between(0, emit_clients - 1).all()
    assert table.ts.notna().all()


# ============================================================
# ЗНАЧЕНИЯ
# ============================================================


def test_transaction_values(raw_tables):
    transactions = raw_tables["transactions"]

    assert set(transactions.direction) <= set(DIRECTIONS)
    assert set(transactions.mcc) <= set(ALL_MCCS)
    assert (transactions.amount > 0).all()
    assert transactions.is_online.dtype == bool
    assert transactions.is_subscription.dtype == bool

    # Валюты в схеме нет сознательно.
    assert "currency" not in transactions.columns


def test_product_values(raw_tables):
    products = raw_tables["product_events"]

    assert set(products.product_type) <= set(PRODUCT_TYPES)
    assert set(products.timestamp_quality) <= set(TIMESTAMP_QUALITIES)

    # У карт нет срока, у дебетовой карты нет суммы.
    cards = products[products.product_type.isin(["debit_card", "credit_card"])]
    assert cards.term.isna().all()

    debit = products[products.product_type == "debit_card"]
    assert debit.amount_or_limit.isna().all()

    assert "action" not in products.columns


def test_communication_values(raw_tables):
    communications = raw_tables["communications"]

    assert set(communications.channel) <= set(COMM_CHANNELS)
    assert communications.delivered.dtype == bool
    assert communications.day_of_week.between(0, 6).all()
    assert communications.hour.between(0, 23).all()

    assert "campaign" not in communications.columns


def test_app_screen_values(raw_tables):
    screens = raw_tables["app_screens"]

    stages = set(screens.funnel_stage.dropna())
    assert stages <= set(FUNNEL_STAGES)

    reasons = set(screens.reject_reason.dropna())
    assert reasons <= set(REJECT_REASONS)

    # reject_reason только на экране отказа.
    with_reason = screens[screens.reject_reason.notna()]
    assert (with_reason.funnel_stage == "rejected").all()

    assert screens.session_id.notna().all()


def test_app_operation_values(raw_tables):
    operations = raw_tables["app_operations"]

    assert set(operations.domain) <= set(APP_DOMAINS)
    assert set(operations.status.dropna()) <= set(OPERATION_STATUSES)


def test_banner_values(raw_tables):
    banners = raw_tables["banners"]

    assert set(banners.slot) <= set(BANNER_SLOTS)
    assert set(banners.offer) <= set(BANNER_OFFERS)
    assert set(banners.action) == {ACTION_SHOWN, ACTION_CLICKED}

    assert "shown" not in banners.columns
    assert "clicked" not in banners.columns


def test_timeline_values(raw_tables):
    timeline = raw_tables["timeline"]

    assert set(timeline.event_type) <= set(EVENT_TYPES)
    assert (timeline.seq >= 0).all()


def test_profile_types(raw_tables):
    profile = raw_tables["profile"]

    assert list(profile.columns)[:3] == ["client_id", "ts", "snapshot_month"]
    assert list(profile.columns)[3:] == list(PROFILE_FIELDS)


def test_label_values(raw_tables):
    labels = raw_tables["labels"]

    assert labels.product_open_90d.dtype == bool


# ============================================================
# МАНИФЕСТ
# ============================================================


def test_manifest(raw_dir, raw_tables, emit_clients):
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["seed"] == SEED
    assert manifest["total_clients"] == emit_clients

    assert manifest["history_start"] == HISTORY_START.isoformat()
    assert manifest["feature_end"] == FEATURE_END.isoformat()
    assert manifest["label_end"] == LABEL_END.isoformat()

    assert manifest["max_tokens_per_event"] == MAX_TOKENS_PER_EVENT
    assert manifest["max_events_per_history"] == MAX_EVENTS_PER_HISTORY
    assert manifest["max_tokens_per_event"] != manifest["max_events_per_history"]

    assert set(manifest["source_availability"]) == set(SOURCES)

    for source, value in manifest["source_availability"].items():
        assert value == SOURCE_AVAILABILITY[source].isoformat()

    assert manifest["event_type_priority"] == EVENT_TYPE_PRIORITY

    for name, table in raw_tables.items():
        assert manifest["rows"][name] == len(table), name
