from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from churn.build import KEYS
from churn.plus_usr import train_plus_usr
from churn.profile import CATEGORICAL
from churn.target import TASKS
from churn.train import train
from test_train import fitted, spoil, world

DIM = 4
TAG = "w-test"


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    world(root)
    return root


def fresh(tmp_path, template):
    shutil.copytree(template, tmp_path, dirs_exist_ok=True)
    return tmp_path / "raw"


def write_vectors(tmp_path, groups=("train", "val"), seed=0, damage=None) -> None:
    """
    Векторы в формате src.downstream.embed для строк групп: по одному на
    клиента, на T группы, с отпечатками выгрузки из meta признаков.
    Лишний клиент без строки churn — как у настоящего embed.
    """
    rng = np.random.default_rng(seed)
    directory = tmp_path / "embeddings"
    directory.mkdir(exist_ok=True)
    meta = {"tag": TAG, "checkpoint": "runs/test/best_checkpoint.pt", "epoch": 1, "groups": {}}
    for group in groups:
        built = json.loads((tmp_path / "data" / group / "meta.json").read_text())
        clients = list(pd.read_parquet(tmp_path / "data" / group / "features.parquet")["client_id"]) + [f"{group}_extra"]
        moment = pd.Timestamp(built["T"]).tz_convert("UTC")
        frame = pd.DataFrame({
            "client_id": clients,
            "cutoff": moment,
            "usr": list(rng.normal(size=(len(clients), DIM)).astype(np.float32)),
        })
        record = {
            "cutoff": moment.isoformat(),
            "raw_events_sha256": built["feature_history_events_sha256"],
            "raw_profile_sha256": built["feature_profile_sha256"],
        }
        if damage == ("missing", group):
            frame = frame.iloc[1:]
        elif damage == ("duplicate", group):
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        elif damage == ("row_cutoff", group):
            frame.loc[0, "cutoff"] = moment - pd.Timedelta(days=1)
        elif damage == ("meta_cutoff", group):
            record["cutoff"] = (moment - pd.Timedelta(days=1)).isoformat()
        elif damage == ("export", group):
            record["raw_events_sha256"] = "previous-generation"
        frame.to_parquet(directory / f"{group}.parquet", index=False)
        meta["groups"][group] = record
    (directory / "meta.json").write_text(json.dumps(meta))


def run(tmp_path, final_test=False) -> dict:
    return train_plus_usr(
        tmp_path / "embeddings",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        raw_dir=tmp_path / "raw",
        future_dir=tmp_path / "future",
        final_test=final_test,
    )


def rows(tmp_path, name: str) -> pd.DataFrame:
    return pd.read_parquet(tmp_path / "reports" / "plus_usr" / TAG / f"{name}.parquet")


def test_the_model_takes_the_full_features_plus_usr_on_the_same_rows(tmp_path, template) -> None:
    """
    catboost_plus_usr — полный X и usr_*, категориальные те же. Строки и
    метки — строки CatBoost-бейзлайна.
    """
    raw = fresh(tmp_path, template)
    write_vectors(tmp_path)
    metrics = run(tmp_path)

    features = pd.read_parquet(tmp_path / "data" / "train" / "features.parquet")
    columns = [name for name in features.columns if name not in KEYS]
    usr = [f"usr_{index}" for index in range(DIM)]
    assert metrics["features"] == {"handcrafted": len(columns), "usr": DIM, "total": len(columns) + DIM}

    for task in TASKS:
        model = CatBoostClassifier()
        model.load_model(tmp_path / "models" / "plus_usr" / TAG / f"catboost_{task}.cbm")
        assert model.feature_names_ == columns + usr
        categorical = [model.feature_names_[index] for index in model.get_cat_feature_indices()]
        assert categorical == [name for name in columns if name in CATEGORICAL] and categorical
        block = metrics["tasks"][task]
        assert block["trees"] == block["best_iteration"] + 1
        assert block["threshold_rule"] == "max F1 on train inner holdout"

    # Строки и метки — ровно те, на которых оценивается CatBoost-бейзлайн.
    train(data_dir=tmp_path / "data", models_dir=tmp_path / "models", reports_dir=tmp_path / "reports",
          raw_dir=raw, future_dir=tmp_path / "future")
    for name in ("eval_rows", "train_rows"):
        baseline = pd.read_parquet(tmp_path / "reports" / f"{name}.parquet")
        plus = rows(tmp_path, name)
        pd.testing.assert_frame_equal(plus[["task", *KEYS]], baseline[["task", *KEYS]])
    assert set(rows(tmp_path, "eval_rows")["group"]) == {"val"}


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (("missing", "val"), "нет вектора"),
        (("duplicate", "train"), "повторяется"),
        (("row_cutoff", "val"), "не на T"),
        (("meta_cutoff", "train"), "сняты на"),
        (("export", "val"), "не с той выгрузки"),
    ],
)
def test_vectors_that_do_not_match_the_rows_are_refused(tmp_path, template, damage, message) -> None:
    fresh(tmp_path, template)
    write_vectors(tmp_path, damage=damage)
    with pytest.raises(ValueError, match=message):
        run(tmp_path)


def test_val_does_not_move_the_model(tmp_path, template) -> None:
    """
    Порог, число деревьев и сама модель — только от train: испорченные
    метки, признаки и векторы val их не меняют.
    """
    fresh(tmp_path, template)
    write_vectors(tmp_path)
    first = run(tmp_path)
    trained = rows(tmp_path, "train_rows")

    spoil(tmp_path, "val")
    vectors = tmp_path / "embeddings" / "val.parquet"
    frame = pd.read_parquet(vectors)
    frame["usr"] = [np.asarray(item) * 100 for item in frame["usr"]]
    frame.to_parquet(vectors, index=False)

    second = run(tmp_path)
    assert fitted(second) == fitted(first)
    pd.testing.assert_frame_equal(rows(tmp_path, "train_rows"), trained)


def test_test_is_used_only_in_the_final_evaluation(tmp_path, template) -> None:
    fresh(tmp_path, template)
    write_vectors(tmp_path)
    default = run(tmp_path)
    assert all(set(block["groups"]) == {"val"} for block in default["tasks"].values())
    assert set(default["embeddings"]["groups"]) == {"train", "val"}

    with pytest.raises(ValueError, match="нет векторов группы test"):
        run(tmp_path, final_test=True)

    write_vectors(tmp_path, groups=("train", "val", "test"))
    final = run(tmp_path, final_test=True)
    assert all(set(block["groups"]) == {"val", "test"} for block in final["tasks"].values())
