"""
Месячные as-of примеры: сетка, причины пропуска, префикс истории.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from src.preprocessing.config import SKIP_REASONS, Settings
from src.preprocessing.cutoffs import (
    CUTOFF_INDEX_SCHEMA,
    build_cutoff_index,
    cutoff_of,
    month_grid,
    skip_summary,
)
from src.preprocessing.raw import RawDataset
from src.preprocessing.splits import assign_groups, month_roles


SETTINGS = Settings()


@pytest.fixture(scope="module")
def raw(prep_raw_dir):
    return RawDataset(prep_raw_dir)


@pytest.fixture(scope="module")
def index(raw):
    groups = assign_groups(raw.client_ids(), SETTINGS.split_seed, SETTINGS.split_shares)
    roles = month_roles(month_grid(raw.manifest.history_start, raw.manifest.feature_end))
    return build_cutoff_index(raw, groups, roles, SETTINGS)


# ============================================================
# СЕТКА МЕСЯЦЕВ
# ============================================================


def test_grid_covers_full_months_only(raw):
    months = month_grid(raw.manifest.history_start, raw.manifest.feature_end)

    assert len(months) == 24
    assert months[0] == datetime(2024, 6, 1)
    assert months[-1] == datetime(2026, 5, 1)


def test_last_cutoff_equals_feature_end(raw):
    months = month_grid(raw.manifest.history_start, raw.manifest.feature_end)

    assert cutoff_of(months[-1]) == raw.manifest.feature_end


def test_partial_month_at_the_end_is_dropped():
    months = month_grid(datetime(2024, 6, 1), datetime(2026, 6, 15))

    assert months[-1] == datetime(2026, 5, 1)


def test_partial_month_at_the_start_is_dropped():
    months = month_grid(datetime(2024, 6, 10), datetime(2026, 6, 1))

    assert months[0] == datetime(2024, 7, 1)


def test_cutoff_is_next_month_start():
    assert cutoff_of(datetime(2026, 5, 1)) == datetime(2026, 6, 1)
    assert cutoff_of(datetime(2025, 12, 1)) == datetime(2026, 1, 1)


# ============================================================
# КАНДИДАТЫ
# ============================================================


def test_index_covers_every_client_and_month(index, raw, prep_clients):
    assert index.num_rows == prep_clients * 24
    assert index.schema.equals(CUTOFF_INDEX_SCHEMA)


def test_first_cutoffs_lack_observation(index):
    """
    До 90 дней наблюдения примеров быть не может.
    """

    for cutoff in (datetime(2024, 7, 1), datetime(2024, 8, 1)):

        rows = index.filter(pc.equal(index.column("cutoff"), pa.scalar(cutoff, pa.timestamp("us"))))

        assert not any(rows.column("valid").to_pylist())
        assert set(rows.column("skip_reason").to_pylist()) == {"insufficient_observation"}


def test_observation_window_reaches_ninety_days(index):
    rows = index.filter(index.column("valid"))

    days = rows.column("observation_days").to_numpy()

    assert days.min() >= SETTINGS.min_observation_days


def test_skip_reasons_are_known(index):
    reasons = set(pc.drop_null(index.column("skip_reason")).to_pylist())

    assert reasons <= set(SKIP_REASONS)


def test_validity_is_monotone_in_cutoff(index):
    """
    Если клиент валиден на каком-то cutoff, он валиден и позже.
    На этом держится определение fit-набора.
    """

    client_id = index.column("client_id").to_numpy()
    cutoff = index.column("cutoff").to_numpy()
    valid = np.array(index.column("valid").to_pylist(), dtype=bool)

    order = np.lexsort((cutoff, client_id))

    client_id, valid = client_id[order], valid[order]

    for cid in np.unique(client_id):
        flags = valid[client_id == cid]
        first_true = np.argmax(flags) if flags.any() else len(flags)
        assert flags[first_true:].all(), cid


def test_dataset_assignment_follows_matrix(index):
    for group, role, dataset in (
        ("train", "train_period", "train"),
        ("val", "train_period", "val_client"),
        ("test", "train_period", "test_client"),
        ("train", "val_month", "val_time"),
        ("train", "test_month", "test_time"),
    ):
        rows = index.filter(
            pc.and_(
                pc.and_(pc.equal(index.column("client_group"), group), pc.equal(index.column("month_role"), role)),
                index.column("valid"),
            )
        )

        assert set(rows.column("dataset").to_pylist()) == {dataset}


def test_unused_combinations_are_marked(index):
    rows = index.filter(
        pc.and_(
            pc.is_in(index.column("client_group"), pa.array(["val", "test"])),
            pc.is_in(index.column("month_role"), pa.array(["val_month", "test_month"])),
        )
    )

    valid = rows.filter(rows.column("valid"))

    assert valid.num_rows > 0
    assert set(valid.column("dataset").to_pylist()) == {None}
    assert set(valid.column("skip_reason").to_pylist()) == {"unused_combination"}


# ============================================================
# ПРЕФИКС ИСТОРИИ
# ============================================================


def test_seq_end_equals_events_before_cutoff(index, raw):
    """
    seq_end это ровно число событий строго раньше cutoff.
    """

    timeline = raw.read("timeline", ["client_id", "ts"])

    client_id = timeline.column("client_id").to_numpy()
    ts = timeline.column("ts").to_numpy()

    order = np.lexsort((ts, client_id))
    client_id, ts = client_id[order], ts[order]

    rows = index.filter(index.column("valid")).slice(0, 400)

    for cid, cutoff, seq_end in zip(
        rows.column("client_id").to_pylist(),
        rows.column("cutoff").to_pylist(),
        rows.column("seq_end").to_pylist(),
    ):
        own = ts[client_id == cid]
        expected = int((own < np.datetime64(cutoff, "us")).sum())
        assert seq_end == expected, (cid, cutoff)


def test_events_exactly_at_cutoff_are_excluded(raw, index):
    """
    Договоры без времени лежат в полночь. Событие в полночь дня
    cutoff принадлежит следующему месяцу, а не текущему.
    """

    products = raw.read("product_events", ["client_id", "ts"])

    ts = products.column("ts").to_numpy()

    at_month_start = ts == ts.astype("datetime64[M]").astype("datetime64[us]")

    assert at_month_start.any(), "в выборке нет договора в полночь первого числа"

    cid = int(products.column("client_id").to_numpy()[at_month_start][0])
    stamp = ts[at_month_start][0].astype("datetime64[us]").astype(datetime)

    row = index.filter(
        pc.and_(
            pc.equal(index.column("client_id"), cid),
            pc.equal(index.column("cutoff"), pa.scalar(stamp, pa.timestamp("us"))),
        )
    )

    if row.num_rows == 0:
        pytest.skip("событие пришлось на границу вне сетки cutoff")

    timeline = raw.read("timeline", ["client_id", "ts", "seq"])
    own = timeline.filter(pc.equal(timeline.column("client_id"), cid))

    seq_end = row.column("seq_end").to_pylist()[0]

    included = own.filter(pc.less(own.column("seq"), seq_end))

    assert all(value < stamp for value in included.column("ts").to_pylist())


def test_snapshot_is_the_last_before_cutoff(index, raw):
    profile = raw.read("profile", ["client_id", "ts"])

    client_id = profile.column("client_id").to_numpy()
    ts = profile.column("ts").to_numpy()

    rows = index.filter(index.column("valid")).slice(0, 200)

    for cid, cutoff, snapshot in zip(
        rows.column("client_id").to_pylist(),
        rows.column("cutoff").to_pylist(),
        rows.column("snapshot_ts").to_pylist(),
    ):
        own = np.sort(ts[client_id == cid])
        expected = own[own < np.datetime64(cutoff, "us")].max()

        assert snapshot == expected.astype("datetime64[us]").astype(datetime)
        assert snapshot < cutoff


# ============================================================
# СИНТЕТИЧЕСКИЕ ПРИЧИНЫ
# ============================================================


def test_missing_transaction_coverage_skips_client(raw):
    """
    Клиент, которого источник транзакций не видит никогда,
    не даёт ни одного примера.
    """

    groups = assign_groups(raw.client_ids(), SETTINGS.split_seed, SETTINGS.split_shares)
    roles = month_roles(month_grid(raw.manifest.history_start, raw.manifest.feature_end))

    class Blind(RawDataset):
        def read(self, name, columns=None):
            table = super().read(name, columns)
            if name != "source_coverage":
                return table
            mask = pc.and_(pc.equal(table.column("client_id"), 0), pc.equal(table.column("source"), "transactions"))
            first_seen = pc.if_else(mask, pa.scalar(None, pa.timestamp("us")), table.column("first_seen"))
            return table.set_column(table.schema.get_field_index("first_seen"), "first_seen", first_seen)

    blind = Blind(raw.raw_dir)

    index = build_cutoff_index(blind, groups, roles, SETTINGS)

    rows = index.filter(pc.equal(index.column("client_id"), 0))

    assert not any(rows.column("valid").to_pylist())
    assert set(rows.column("skip_reason").to_pylist()) == {"no_transactions_coverage"}


def test_missing_profile_skips_example(raw):
    groups = assign_groups(raw.client_ids(), SETTINGS.split_seed, SETTINGS.split_shares)
    roles = month_roles(month_grid(raw.manifest.history_start, raw.manifest.feature_end))

    class NoProfile(RawDataset):
        def read(self, name, columns=None):
            table = super().read(name, columns)
            if name != "profile":
                return table
            return table.filter(pc.not_equal(table.column("client_id"), 0))

    index = build_cutoff_index(NoProfile(raw.raw_dir), groups, roles, SETTINGS)

    rows = index.filter(pc.equal(index.column("client_id"), 0))

    reasons = set(rows.column("skip_reason").to_pylist())

    assert "no_profile_snapshot" in reasons
    assert not any(rows.column("valid").to_pylist())


def test_skip_summary_counts_every_reason(index):
    summary = skip_summary(index)

    assert set(summary) <= set(SKIP_REASONS)
    assert summary["insufficient_observation"] > 0
