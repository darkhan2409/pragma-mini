from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from fraud.build import prepared_profile
from fraud.features import compute, eligible
from fraud.raw import client_blocks
from world import LOCAL, UTC, alert, event, login, profile, profile_change, purchase, salary, transfer, write_group

END = datetime(2026, 4, 1, tzinfo=LOCAL)
NOW = datetime(2025, 6, 10, 14, 0, tzinfo=LOCAL)

FORBIDDEN = ("second", "minute", "status", "decline_reason", "transfer_id", "card_id", "account_id", "session", "round", "balance_after", "counterparty_name", "episode", "fraud_label")


def features(tmp_path, events: list[dict], profiles: list[dict] | None = None, name: str = "w") -> pd.DataFrame:
    clients = sorted({row["client_id"] for row in events})
    profiles = profiles or [profile(client, END) for client in clients]
    out = write_group(tmp_path / name, "val", END, events, profiles)
    table = prepared_profile(out / "profile.parquet")
    frames = []
    for block in client_blocks(out / "events.parquet"):
        rows = np.flatnonzero(eligible(block))
        columns, _ = compute(block, rows, table)
        frame = pd.DataFrame(columns)
        frame.index = pd.MultiIndex.from_arrays([block["client_id"].to_numpy()[rows], block["t"].iloc[rows].to_numpy(), block["type"].to_numpy()[rows]])
        frames.append(frame)
    return pd.concat(frames).sort_index()


def one(table: pd.DataFrame, moment: datetime, kind: str) -> pd.Series:
    rows = table.loc[[("c", pd.Timestamp(moment).tz_convert("UTC"), kind)]]
    assert len(rows) == 1
    return rows.iloc[0]


def history(client: str = "c") -> list[dict]:
    return [
        purchase(client, NOW - timedelta(days=40), amount=3000, mcc="5411", merchant_name="Magnum", merchant_country="KZ"),
        purchase(client, NOW - timedelta(days=3), amount=4000, mcc="5411", merchant_name="Magnum", merchant_country="KZ"),
        salary(client, NOW - timedelta(days=12)),
        login(client, NOW - timedelta(hours=5)),
        transfer(client, NOW - timedelta(days=20), counterparty="B. Friend"),
    ]


def test_row_outcome_and_the_future_do_not_change_features(tmp_path) -> None:
    row = purchase("c", NOW, amount=90000, mcc="5732", merchant_name="Sulpak", merchant_country="AE", status="approved", balance_after=10000)
    quiet = features(tmp_path, history() + [row], name="quiet")

    declined = purchase("c", NOW, amount=90000, mcc="5732", merchant_name="Sulpak", merchant_country="AE", status="declined", decline_reason="insufficient_funds")
    future = [
        alert("c", NOW + timedelta(minutes=40)),
        event("c", NOW + timedelta(minutes=41), "product_events", type="card_blocked", reason="fraud_suspicion"),
        event("c", NOW + timedelta(hours=3), "support", type="case_opened", topic="fraud_report", channel="chat", status="open"),
        purchase("c", NOW + timedelta(days=1), amount=100),
        # В тот же момент: порядок внутри момента банк не знает.
        login("c", NOW),
        transfer("c", NOW, amount=7, counterparty="S. Same"),
    ]
    busy = features(tmp_path, history() + [declined] + future, name="busy")
    pd.testing.assert_series_equal(one(quiet, NOW, "purchase"), one(busy, NOW, "purchase"), check_names=False)


def test_generator_traces_do_not_change_features(tmp_path) -> None:
    # Шаг мошенника: целые секунды от целого часа, без transfer_id, со счётом.
    plain = transfer("c", NOW, amount=25000, counterparty="X. Stranger", transfer_id="trf_9", account_id=None)
    traced = transfer("c", NOW + timedelta(minutes=17, seconds=43), amount=25000, counterparty="X. Stranger", transfer_id=None, account_id="acc_1")
    first = features(tmp_path, history() + [plain], name="plain")
    second = features(tmp_path, history() + [traced], name="traced")
    a = first.xs("transfer_out", level=2).iloc[-1]
    b = second.xs("transfer_out", level=2).iloc[-1]
    changed = a.index[~((a == b) | (a.isna() & b.isna()))]
    # Разница в 17 минут видна только в давности прошлых событий, а следы
    # идентификаторов и секунды — нигде.
    assert len(changed) and all(name.endswith("_hours_since_prev") or name.startswith("days_since_") for name in changed)
    assert a["txn_hours_since_prev"] < b["txn_hours_since_prev"]


def test_feature_names_have_no_forbidden_fields(tmp_path) -> None:
    table = features(tmp_path, history() + [purchase("c", NOW)])
    for name in table.columns:
        assert not any(part in name for part in FORBIDDEN), name


def test_windows_are_strictly_before_the_row(tmp_path) -> None:
    events = [
        purchase("c", NOW - timedelta(hours=1), amount=100),
        purchase("c", NOW - timedelta(hours=1, milliseconds=1), amount=200),
        purchase("c", NOW - timedelta(days=1), amount=400),
        purchase("c", NOW, amount=800),
        transfer("c", NOW, amount=1600),
    ]
    table = features(tmp_path, events)
    row = one(table, NOW, "purchase")
    assert row["txn_count_1h"] == 1 and row["txn_sum_1h"] == 100
    assert row["txn_count_1d"] == 3 and row["txn_sum_1d"] == 700
    assert row["txn_hours_since_prev"] == pytest.approx(1.0)


def test_novelty_of_merchant_and_counterparty(tmp_path) -> None:
    events = history() + [
        purchase("c", NOW, mcc="5411", merchant_name="Magnum", merchant_country="KZ"),
        purchase("c", NOW + timedelta(hours=1), mcc="7995", merchant_name="Casino", merchant_country="GE"),
        transfer("c", NOW + timedelta(hours=2), counterparty="B. Friend"),
        transfer("c", NOW + timedelta(hours=3), counterparty="Z. Unknown"),
    ]
    table = features(tmp_path, events)
    at = lambda hours, kind: one(table, NOW + timedelta(hours=hours), kind)
    known, new = at(0, "purchase"), at(1, "purchase")
    assert known["mcc_is_new"] == 0 and known["mcc_hours_since_prev"] == pytest.approx(72.0)
    assert new["mcc_is_new"] == 1 and new["country_is_new"] == 1 and new["merchant_is_new"] == 1
    assert at(2, "transfer_out")["counterparty_is_new"] == 0
    assert at(3, "transfer_out")["counterparty_is_new"] == 1


def test_new_device_login_before_the_row_is_seen(tmp_path) -> None:
    login_new = event("c", NOW - timedelta(minutes=4), "app_operations", type="app_operation", operation="login", status="success", device_new=True, session_id="s9")
    table = features(tmp_path, history() + [login_new, transfer("c", NOW, amount=50000, counterparty="Q. Mule")])
    row = table.xs("transfer_out", level=2).iloc[-1]
    assert row["new_device_login_1h"] == 1
    assert row["new_device_hours_since_prev"] == pytest.approx(4 / 60)


def test_profile_is_taken_at_the_moment_of_the_row(tmp_path) -> None:
    events = history() + [
        purchase("c", NOW),
        profile_change("c", NOW + timedelta(days=5), "region", "Astana", "Almaty"),
        profile_change("c", NOW + timedelta(days=6), "declared_income", "250000", "300000", source="application"),
    ]
    lifelong = [
        {"type": "bank_registered", "event_time": (NOW - timedelta(days=10)).astimezone(UTC), "source_id": None},
        {"type": "app_registered", "event_time": (NOW + timedelta(days=1)).astimezone(UTC), "source_id": None},
    ]
    table = features(tmp_path, events, [profile("c", END, region="Almaty", declared_income=300000, lifelong=lifelong)])
    row = one(table, NOW, "purchase")
    assert row["region"] == "Astana"
    assert row["declared_income"] == 250000
    assert row["days_since_bank_registered"] == pytest.approx(10.0)
    assert np.isnan(row["days_since_app_registered"])
    assert row["age"] == 34
