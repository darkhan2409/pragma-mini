from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from churn.build import KEYS, build
from churn.config import GROUPS, HORIZON
from churn.train import check_groups, threshold_max_f1, train
from world import LOCAL, login, profile, purchase, push, salary, write_group

ENDS = {
    "train": datetime(2026, 1, 1, tzinfo=LOCAL),
    "val": datetime(2026, 4, 1, tzinfo=LOCAL),
    "test": datetime(2026, 8, 1, tzinfo=LOCAL),
}


def world(tmp_path):
    raw = tmp_path / "raw"
    for group, end in ENDS.items():
        cutoff = end - HORIZON
        events, profiles = [], []
        for index in range(16):
            client = f"{group}_{index:02d}"
            leaving = index % 2 == 1
            last = cutoff - timedelta(days=45 if leaving else 2, hours=index)
            events += [purchase(client, last - timedelta(days=d), amount=500 + 10 * d) for d in range(0, 30, 3)]
            events += [login(client, last, session=f"{client}_s"), salary(client, cutoff - timedelta(days=12)), push(client, cutoff - timedelta(days=4))]
            if not leaving:
                events.append(login(client, cutoff + timedelta(days=7), session=f"{client}_f"))
            profiles.append(profile(client, end))
        write_group(raw, group, end, events, profiles)
    for group in GROUPS:
        build(group, raw_dir=raw, data_dir=tmp_path / "data", reports_dir=tmp_path / "reports")
    return raw


def test_rows_for_comparison_are_exactly_the_evaluated_objects(tmp_path) -> None:
    raw = world(tmp_path)
    metrics = train(data_dir=tmp_path / "data", models_dir=tmp_path / "models", reports_dir=tmp_path / "reports", raw_dir=raw)

    evaluated = pd.read_parquet(tmp_path / "reports" / "eval_rows.parquet")
    trained = pd.read_parquet(tmp_path / "reports" / "train_rows.parquet")
    assert list(evaluated.columns) == list(KEYS) + ["score"]
    assert set(evaluated["group"]) == {"val", "test"} and set(trained["group"]) == {"train"}
    assert not evaluated.duplicated(["group", "client_id"]).any()

    for group in ("val", "test"):
        features = pd.read_parquet(tmp_path / "data" / group / "features.parquet")
        rows = evaluated[evaluated["group"] == group].reset_index(drop=True)
        pd.testing.assert_frame_equal(rows[list(KEYS)], features[list(KEYS)])
        assert metrics["groups"][group]["all"]["rows"] == len(features)

    for name in ("eval_rows", "train_rows"):
        digest = hashlib.sha256((tmp_path / "reports" / f"{name}.parquet").read_bytes()).hexdigest()
        assert metrics[f"{name}_sha256"] == digest
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text())["threshold"] == metrics["threshold"]
    assert (tmp_path / "models" / "catboost.cbm").exists()


def test_groups_must_not_share_clients() -> None:
    frames = {
        "train": pd.DataFrame({"client_id": ["a", "b"], "group": "train"}),
        "val": pd.DataFrame({"client_id": ["c"], "group": "val"}),
        "test": pd.DataFrame({"client_id": ["b"], "group": "test"}),
    }
    with pytest.raises(ValueError):
        check_groups(frames)


def test_threshold_maximises_f1() -> None:
    y = pd.Series([0, 0, 1, 1, 0, 1]).to_numpy()
    score = pd.Series([0.1, 0.4, 0.35, 0.8, 0.2, 0.9]).to_numpy()
    assert threshold_max_f1(y, score) == pytest.approx(0.35)
