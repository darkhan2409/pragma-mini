"""
Дефекты данных, ради которых существует эта синтетика.

Если какой-то из них исчезнет, preprocessing, написанный
на синтетике, сломается на реальных данных.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from src.generator.config import (
    FEATURE_END,
    HISTORY_START,
    SOURCE_AVAILABILITY,
)
from src.generator.noise import GA4_NOT_SET
from src.generator.products import QUALITY_DATE_ONLY


# ============================================================
# 1. РАЗНЫЕ ДАТЫ СТАРТА ИСТОЧНИКОВ
# ============================================================


def test_sources_start_at_different_dates():
    starts = set(SOURCE_AVAILABILITY.values())

    assert len(starts) >= 4, "источники обязаны стартовать в разные даты"


def test_late_sources_are_empty_at_the_beginning(raw_tables):
    early = pd.Timestamp(HISTORY_START) + pd.Timedelta(days=30)

    transactions = raw_tables["transactions"]

    assert (transactions.ts < early).any(), "транзакции идут с начала истории"

    for source in ("communications", "app_screens"):
        table = raw_tables[source]
        assert not (table.ts < early).any(), source


# ============================================================
# 2. КЛИЕНТ ПОЯВЛЯЕТСЯ В ИСТОЧНИКЕ НЕ СРАЗУ
# ============================================================


def test_clients_enter_sources_at_different_times(raw_tables):
    coverage = raw_tables["source_coverage"]

    app = coverage[coverage.source == "app_screens"]

    assert app.first_seen.isna().any(), "часть клиентов не пользуется приложением"
    assert app.first_seen.nunique() > 1, "клиенты приходят в источник в разные даты"


def test_stream_is_empty_for_uncovered_clients(raw_tables):
    coverage = raw_tables["source_coverage"]

    never = coverage[(coverage.source == "app_screens") & coverage.first_seen.isna()]

    screens = raw_tables["app_screens"]

    for client_id in never.client_id:
        assert screens[screens.client_id == client_id].empty


# ============================================================
# 3. TIMESTAMP_QUALITY
# ============================================================


def test_date_only_contracts_sit_at_midnight(raw_tables):
    products = raw_tables["product_events"]

    date_only = products[products.timestamp_quality == QUALITY_DATE_ONLY]

    assert not date_only.empty

    assert (date_only.ts.dt.hour == 0).all()
    assert (date_only.ts.dt.minute == 0).all()
    assert (date_only.ts.dt.second == 0).all()

    exact = products[products.timestamp_quality != QUALITY_DATE_ONLY]

    assert not exact.empty
    assert (exact.ts.dt.hour > 0).any()


def test_both_qualities_present(raw_tables):
    qualities = set(raw_tables["product_events"].timestamp_quality)

    assert len(qualities) == 2


# ============================================================
# 4. НЕРАВНОМЕРНЫЕ ПРОПУСКИ ПРОФИЛЯ
# ============================================================


def test_profile_missing_rate_differs_across_months(raw_tables):
    profile = raw_tables["profile"]

    by_month = profile.groupby(profile.snapshot_month.dt.to_period("M")).apply(
        lambda frame: frame.declared_income.isna().mean()
    )

    assert by_month.max() - by_month.min() > 0.10


def test_profile_has_structural_and_random_missing(raw_tables):
    profile = raw_tables["profile"]

    # Структурный: у неработающих нет отрасли.
    assert profile.industry.isna().mean() > 0.2

    # Ключевые поля не теряются никогда.
    assert profile.age.notna().all()
    assert profile.region.notna().all()


# ============================================================
# 5. РЕДКИЕ КАТЕГОРИИ
# ============================================================


@pytest.mark.parametrize(
    "table, column, minimum, skew",
    [
        ("transactions", "mcc", 25, 5.0),
        ("communications", "template", 15, 5.0),
        ("app_screens", "firebase_screen", 15, 5.0),
        ("app_operations", "operation", 15, 5.0),
        # Офферов в баннерах немного и они распределены ровнее:
        # банк крутит ограниченный набор кампаний.
        ("banners", "offer", 6, 2.5),
    ],
)
def test_long_tail_categories(raw_tables, table, column, minimum, skew):
    counts = raw_tables[table][column].value_counts()

    assert len(counts) >= minimum, (table, column, len(counts))

    top = counts.iloc[0]
    tail = counts.iloc[-1]

    assert top >= skew * tail, (table, column, top, tail)


def test_rare_values_are_actually_rare(raw_tables):
    mcc = raw_tables["transactions"].mcc.value_counts(normalize=True)

    assert (mcc < 0.005).any(), "должны быть MCC с долей меньше половины процента"


def test_ga4_not_set_appears(raw_tables):
    screens = raw_tables["app_screens"]

    share = (screens.firebase_screen == GA4_NOT_SET).mean()

    assert 0.0 < share < 0.05


# ============================================================
# 6. ОДИНАКОВЫЕ TS И TIE-BREAK
# ============================================================


def test_duplicate_timestamps_are_present_but_not_dominant(raw_tables):
    """
    Совпадения ts нужны, иначе tie-break ничем не проверяется.

    Но их доля не должна быть завышенной: в реальных данных
    события разных систем редко попадают в одну секунду.
    Ориентир, снятый на 10 000 клиентов: около 4 процентов.
    """

    timeline = raw_tables["timeline"]

    share = timeline.duplicated(subset=["client_id", "ts"], keep=False).mean()

    assert 0.01 < share < 0.12, f"доля совпадающих ts вышла за ориентир: {share:.3f}"


def test_duplicate_timestamps_are_not_concentrated_in_one_client(raw_tables):
    timeline = raw_tables["timeline"]

    per_client = timeline.groupby("client_id").apply(
        lambda group: group.duplicated(subset=["ts"], keep=False).mean(),
        include_groups=False,
    )

    assert per_client.max() < 0.25, f"у клиента слишком много совпадений: {per_client.max():.3f}"


def test_duplicate_timestamps_span_different_streams(raw_tables):
    timeline = raw_tables["timeline"]

    grouped = timeline[timeline.duplicated(subset=["client_id", "ts"], keep=False)]

    types = grouped.groupby(["client_id", "ts"]).event_type.nunique()

    assert (types > 1).any(), "должны быть совпадения между разными потоками"


def test_seq_resolves_ties_deterministically(raw_tables):
    timeline = raw_tables["timeline"]

    for _, group in timeline.groupby(["client_id", "ts"]):

        if len(group) < 2:
            continue

        assert group.seq.is_monotonic_increasing
        assert group.seq.nunique() == len(group)


# ============================================================
# 7. ЗАПРЕЩЁННОЕ
# ============================================================


def test_no_p2p_or_repayment_tables(raw_dir):
    for name in ("transfers", "repayments", "p2p"):
        assert not (raw_dir / f"{name}.parquet").exists()


def test_only_one_label(raw_tables):
    labels = raw_tables["labels"]

    assert "default_90d" not in labels.columns
    assert "churn_90d" not in labels.columns
    assert "product_open_90d" in labels.columns
