from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fraud.build import KEYS, build, sampled
from fraud.config import NEGATIVE_ONE_IN
from fraud.train import ROW_KEYS, train
from world import LOCAL, login, profile, purchase, transfer, write_group

ENDS = {
    "train": datetime(2026, 1, 1, tzinfo=LOCAL),
    "val": datetime(2026, 4, 1, tzinfo=LOCAL),
    "test": datetime(2026, 8, 1, tzinfo=LOCAL),
}
START = datetime(2025, 3, 1, 9, tzinfo=LOCAL)


def world(tmp_path):
    raw, data = tmp_path / "raw", tmp_path / "data"
    for group, end in ENDS.items():
        events, profiles, frauds = [], [], []
        for index in range(12):
            client = f"{group}_{index:02d}"
            for day in range(40):
                events.append(purchase(client, START + timedelta(days=day, hours=index % 5), amount=1000 + 37 * day, mcc="5411"))
            events.append(login(client, START + timedelta(days=41)))
            if index % 3 == 0:
                moment = START + timedelta(days=42, hours=3)
                events.append(transfer(client, moment, amount=200000, counterparty=f"Z. Mule{index}"))
                frauds.append((client, moment))
            profiles.append(profile(client, end))
        write_group(raw, group, end, events, profiles, row_group_size=64)
        table = pq.read_table(raw / group / "events.parquet").to_pandas()
        table["t"] = pd.to_datetime(table["event_time"], format="ISO8601", utc=True)
        rows = []
        for client, moment in frauds:
            hit = table[(table["client_id"] == client) & (table["t"] == pd.Timestamp(moment))]
            rows.append({"client_id": client, "event_time": pd.Timestamp(moment).tz_convert("UTC"), "raw_row": int(hit.index[0]),
                         "episode_kind": "suspicious_transfer", "step_kind": "strike", "type": "transfer_out", "amount": 200000, "fraud": 1})
        (data / group).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(data / group / "labels.parquet", index=False)
        build(group, raw_dir=raw, data_dir=data, reports_dir=tmp_path / "reports")
    return raw, data


def test_sampling_is_deterministic_and_one_in_n() -> None:
    rows = np.arange(200_000)
    first, second = sampled(rows), sampled(rows)
    assert np.array_equal(first, second)
    assert abs(first.mean() - 1 / NEGATIVE_ONE_IN) < 0.005


def test_train_keeps_every_marked_row_and_weights_negatives(tmp_path) -> None:
    raw, data = world(tmp_path)
    train_rows = pd.read_parquet(data / "train" / "features.parquet")
    assert train_rows["fraud"].sum() == 4
    assert (train_rows.loc[train_rows["fraud"] == 1, "weight"] == 1).all()
    assert (train_rows.loc[train_rows["fraud"] == 0, "weight"] == NEGATIVE_ONE_IN).all()
    assert sampled(train_rows.loc[train_rows["fraud"] == 0, "raw_row"].to_numpy()).all()
    val_rows = pd.read_parquet(data / "val" / "features.parquet")
    assert (val_rows["weight"] == 1).all() and val_rows["fraud"].sum() == 4


def test_rows_for_comparison_are_exactly_the_evaluated_objects(tmp_path) -> None:
    raw, data = world(tmp_path)
    reports = tmp_path / "reports"
    metrics = train(data_dir=data, models_dir=tmp_path / "models", reports_dir=reports, raw_dir=raw)

    evaluated = pd.read_parquet(reports / "eval_rows.parquet")
    assert list(evaluated.columns) == list(ROW_KEYS) + ["score"]
    assert not evaluated.duplicated(["group", "raw_row"]).any()
    for group in ("val", "test"):
        features = pd.read_parquet(data / group / "features.parquet")
        rows = evaluated[evaluated["group"] == group].reset_index(drop=True)
        pd.testing.assert_frame_equal(rows[list(ROW_KEYS)], features[list(ROW_KEYS)])
        assert metrics["groups"][group]["all"]["rows"] == len(features)
    for name in ("eval_rows", "train_rows"):
        assert metrics[f"{name}_sha256"] == hashlib.sha256((reports / f"{name}.parquet").read_bytes()).hexdigest()
    assert set(KEYS) & set(metrics["categorical"]) == set()
