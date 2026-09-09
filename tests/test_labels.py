"""
Метка: product_open_90d выводится только из будущих событий.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.generator.config import FEATURE_END, HISTORY_START, LABEL_END
from src.generator.derive import derive_labels
from src.generator.history import ClientHistory, generate_client_history
from src.generator.products import ProductEvent


def history_with(product_events: list[ProductEvent]) -> ClientHistory:
    return ClientHistory(
        client_id=1,
        start=HISTORY_START,
        end=LABEL_END,
        profile=[],
        transactions=[],
        product_events=product_events,
        communications=[],
        app_screens=[],
        app_operations=[],
        banners=[],
    )


def contract(offset_days: int) -> ProductEvent:
    return ProductEvent(
        client_id=1,
        ts=FEATURE_END + timedelta(days=offset_days),
        product_type="deposit",
        amount_or_limit=500_000.0,
        term=12,
        product_subtype="standard",
        timestamp_quality="date_only",
    )


# ============================================================
# ОКНО
# ============================================================


def test_opening_inside_window_sets_label():
    assert derive_labels(history_with([contract(10)])).product_open_90d


def test_opening_before_cutoff_does_not_count():
    assert not derive_labels(history_with([contract(-10)])).product_open_90d


def test_opening_after_window_does_not_count():
    assert not derive_labels(history_with([contract(120)])).product_open_90d


def test_no_products_no_label():
    assert not derive_labels(history_with([])).product_open_90d


def test_window_bounds():
    labels = derive_labels(history_with([]))

    assert labels.label_start == FEATURE_END
    assert labels.label_end == LABEL_END


def test_label_is_deterministic_and_rng_free():
    history = generate_client_history(3)

    assert derive_labels(history) == derive_labels(history)


# ============================================================
# ТОЛЬКО ОДНА МЕТКА
# ============================================================


def test_only_product_open_label():
    labels = derive_labels(history_with([]))

    assert not hasattr(labels, "default_90d")
    assert not hasattr(labels, "churn_90d")


def test_label_rate_is_reasonable(raw_tables):
    rate = raw_tables["labels"].product_open_90d.mean()

    assert 0.0 < rate < 0.35
