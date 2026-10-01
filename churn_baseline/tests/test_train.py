from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from churn.build import KEYS, build
from churn.config import FUTURE_LABEL_GROUPS, GROUPS, HORIZON
from churn.target import TASKS, task_rows
from churn.train import THRESHOLD_RULE, check_groups, evaluate, holdout, threshold_max_f1, train
from world import LOCAL, login, profile, purchase, push, salary, write_future, write_group

ENDS = {
    "train": datetime(2026, 1, 1, tzinfo=LOCAL),
    "val": datetime(2026, 4, 1, tzinfo=LOCAL),
    "test": datetime(2026, 8, 1, tzinfo=LOCAL),
}


def world(tmp_path):
    """
    По 20 клиентов в группе: 16 активных (половина уходит) и 4 давно
    замолчавших — они в популяции, но не в churn_active90. У train T —
    конец выгрузки, и действия после T лежат в её продолжении.
    """
    raw = tmp_path / "raw"
    for group, end in ENDS.items():
        future_labels = group in FUTURE_LABEL_GROUPS
        cutoff = end if future_labels else end - HORIZON
        events, profiles, future = [], [], []
        for index in range(20):
            client = f"{group}_{index:02d}"
            if index >= 16:
                last = cutoff - timedelta(days=120 + index)
                events += [purchase(client, last - timedelta(days=d), amount=300 + d) for d in range(0, 20, 4)]
                events += [salary(client, cutoff - timedelta(days=12))]
                profiles.append(profile(client, end))
                continue
            leaving = index % 2 == 1
            last = cutoff - timedelta(days=45 if leaving else 2, hours=index)
            events += [purchase(client, last - timedelta(days=d), amount=500 + 10 * d) for d in range(0, 30, 3)]
            events += [login(client, last, session=f"{client}_s"), salary(client, cutoff - timedelta(days=12)), push(client, cutoff - timedelta(days=4))]
            if not leaving:
                (future if future_labels else events).append(login(client, cutoff + timedelta(days=7), session=f"{client}_f"))
            profiles.append(profile(client, end))
        write_group(raw, group, end, events, profiles)
        if future_labels:
            write_future(tmp_path / "future", raw, group, future, end + HORIZON + timedelta(days=1))
    for group in GROUPS:
        build(group, raw_dir=raw, data_dir=tmp_path / "data", reports_dir=tmp_path / "reports", future_dir=tmp_path / "future")
    return raw


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """
    Мир строится один раз на модуль; каждый тест получает свою копию.
    """
    root = tmp_path_factory.mktemp("world")
    world(root)
    return root


def fresh(tmp_path, template):
    shutil.copytree(template, tmp_path, dirs_exist_ok=True)
    return tmp_path / "raw"


def run(tmp_path, raw, final_test: bool = False) -> dict:
    return train(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        raw_dir=raw,
        final_test=final_test,
        future_dir=tmp_path / "future",
    )


def rows(tmp_path, name: str) -> pd.DataFrame:
    return pd.read_parquet(tmp_path / "reports" / f"{name}.parquet")


def test_rows_for_comparison_are_exactly_the_evaluated_objects(tmp_path, template) -> None:
    raw = fresh(tmp_path, template)
    metrics = run(tmp_path, raw)

    evaluated, trained = rows(tmp_path, "eval_rows"), rows(tmp_path, "train_rows")
    assert list(evaluated.columns) == ["task", *KEYS, "score"]
    assert set(evaluated["group"]) == {"val"} and set(trained["group"]) == {"train"}
    assert not evaluated.duplicated(["task", "group", "client_id"]).any()

    for group, table in (("val", evaluated), ("train", trained)):
        features = pd.read_parquet(tmp_path / "data" / group / "features.parquet")
        for task in TASKS:
            expected = task_rows(task, features)[list(KEYS)].reset_index(drop=True)
            found = table[table["task"] == task][list(KEYS)].reset_index(drop=True)
            pd.testing.assert_frame_equal(found, expected)
            if group == "val":
                assert metrics["tasks"][task]["groups"]["val"]["rows"] == len(expected)

        # churn_active90 — строки популяции с действием за 90 дней до T.
        active = table[table["task"] == "churn_active90"]
        assert set(active["client_id"]) == set(features.loc[features["active90"], "client_id"])
        assert len(active) == 16 and len(features) == 20

    for name in ("eval_rows", "train_rows"):
        digest = hashlib.sha256((tmp_path / "reports" / f"{name}.parquet").read_bytes()).hexdigest()
        assert metrics[f"{name}_sha256"] == digest

    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert set(saved["tasks"]) == set(TASKS)
    for task in TASKS:
        block = saved["tasks"][task]
        result = block["groups"]["val"]
        assert result["positive_rate"] == pytest.approx(result["positives"] / result["rows"])
        inner = [block["inner_train"], block["inner_holdout"]]
        assert sum(part["rows"] for part in inner) == len(trained[trained["task"] == task])
        # sklearn округляет отложенную долю вверх.
        assert block["inner_holdout"]["rows"] == math.ceil(0.2 * len(trained[trained["task"] == task]))
        for part in inner:
            assert part["positive_rate"] == pytest.approx(part["positives"] / part["rows"])
        assert block["threshold_rule"] == THRESHOLD_RULE == "max F1 on train inner holdout"
        # На всём train учится модель с числом деревьев лучшей итерации.
        assert block["trees"] == block["best_iteration"] + 1
        model = CatBoostClassifier()
        model.load_model(tmp_path / "models" / f"catboost_{task}.cbm")
        assert model.tree_count_ == block["trees"]


def spoil(tmp_path, group: str) -> None:
    """
    Метки группы перевёрнуты, признаки сдвинуты на строку: от данных
    группы, которые видела бы модель, не остаётся ничего.
    """
    path = tmp_path / "data" / group / "features.parquet"
    frame = pd.read_parquet(path)
    frame["churn"] = (1 - frame["churn"]).astype(frame["churn"].dtype)
    for name in frame.columns:
        if name not in KEYS:
            frame[name] = np.roll(frame[name].to_numpy(), 1)
    frame.to_parquet(path, index=False)


def fitted(metrics: dict) -> dict:
    """
    Всё, что выбирается при обучении, по задачам.
    """
    keys = ("threshold", "threshold_rule", "best_iteration", "trees", "inner_train", "inner_holdout")
    return {task: {key: block[key] for key in keys} for task, block in metrics["tasks"].items()}


def test_threshold_is_chosen_on_the_train_inner_holdout_only(tmp_path, template) -> None:
    """
    Порог и число деревьев воспроизводимы при seed 42 и зависят только
    от train: испорченный val не меняет ни их, ни модель (её прогнозы
    на train те же).
    """
    raw = fresh(tmp_path, template)

    first = run(tmp_path, raw)
    trained, evaluated = rows(tmp_path, "train_rows"), rows(tmp_path, "eval_rows")

    again = run(tmp_path, raw)
    assert fitted(again) == fitted(first)
    pd.testing.assert_frame_equal(rows(tmp_path, "eval_rows"), evaluated)

    spoil(tmp_path, "val")
    spoiled = run(tmp_path, raw)

    assert fitted(spoiled) == fitted(first)
    pd.testing.assert_frame_equal(rows(tmp_path, "train_rows"), trained)
    assert not rows(tmp_path, "eval_rows")["score"].equals(evaluated["score"])


def test_inner_holdout_is_a_stratified_fifth_of_train() -> None:
    y = np.array([1] * 30 + [0] * 70)
    fit, held = holdout(y)
    assert len(held) == 20 and len(fit) == 80
    assert y[held].sum() == 6 and y[fit].sum() == 24
    assert sorted(np.concatenate([fit, held])) == list(range(100))
    np.testing.assert_array_equal(holdout(y)[1], held)


def test_test_is_used_only_in_the_final_evaluation(tmp_path, template) -> None:
    raw = fresh(tmp_path, template)

    # Без --final-test test не читается вовсе: его признаков может не быть.
    kept = tmp_path / "kept_test"
    shutil.move(tmp_path / "data" / "test", kept)
    default = run(tmp_path, raw)
    assert all(set(block["groups"]) == {"val"} for block in default["tasks"].values())
    assert set(default["sources"]) == {"train", "val"}
    assert set(rows(tmp_path, "eval_rows")["group"]) == {"val"}
    val_scores = rows(tmp_path, "eval_rows")

    shutil.move(kept, tmp_path / "data" / "test")
    final = run(tmp_path, raw, final_test=True)
    assert all(set(block["groups"]) == {"val", "test"} for block in final["tasks"].values())
    evaluated = rows(tmp_path, "eval_rows")
    assert set(evaluated["group"]) == {"val", "test"}

    # test модель и порог не меняет: прогнозы val те же, что без него.
    assert fitted(final) == fitted(default)
    again = evaluated[evaluated["group"] == "val"].reset_index(drop=True)
    pd.testing.assert_frame_equal(again, val_scores)

    # Метрики val и test — при пороге, зафиксированном на train.
    for task, block in final["tasks"].items():
        for group in ("val", "test"):
            part = evaluated[(evaluated["task"] == task) & (evaluated["group"] == group)]
            expected = evaluate(part["churn"].to_numpy(), part["score"].to_numpy(), block["threshold"])
            found = block["groups"][group]
            assert found["confusion_matrix"] == expected.pop("confusion_matrix")
            assert {key: found[key] for key in expected} == pytest.approx(expected)

    # Ни val, ни test не могут сдвинуть порог.
    spoil(tmp_path, "val")
    spoil(tmp_path, "test")
    assert fitted(run(tmp_path, raw, final_test=True)) == fitted(default)


@pytest.mark.parametrize(
    ("group", "source", "message"),
    [
        ("val", "events_sha256", "feature_history_events_sha256"),
        ("val", "profile_sha256", "feature_profile_sha256"),
        ("train", "events_sha256", "не этой выгрузки"),
        ("train", "profile_sha256", "не этой выгрузки"),
        ("train", "continuation", "target_events_sha256"),
    ],
)
def test_rows_of_other_sources_are_refused(tmp_path, template, group, source, message) -> None:
    """
    Строки собраны на одних источниках, а сейчас другие: другая история
    или анкета выгрузки, другое продолжение. Каждый источник проверяется
    отдельно.
    """
    raw = fresh(tmp_path, template)
    if source == "continuation":
        write_future(tmp_path / "future", raw, group, [login(f"{group}_00", ENDS[group] + timedelta(days=3))],
                     ENDS[group] + HORIZON + timedelta(days=1))
    else:
        path = raw / group / "manifest.json"
        exported = json.loads(path.read_text())
        exported[source] = "previous-generation"
        path.write_text(json.dumps(exported))
    with pytest.raises(ValueError, match=message):
        run(tmp_path, raw)


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


def test_log_loss_is_the_mean_cross_entropy_of_the_probability() -> None:
    """
    log_loss — средняя кросс-энтропия вероятности без порога: от
    порога не зависит, у уверенно верного прогноза меньше, чем у
    константы.
    """
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    expected = -np.mean(y * np.log(score) + (1 - y) * np.log(1 - score))

    assert evaluate(y, score, 0.35)["log_loss"] == pytest.approx(expected)
    assert evaluate(y, score, 0.9)["log_loss"] == evaluate(y, score, 0.35)["log_loss"]
    assert evaluate(y, score, 0.5)["log_loss"] < evaluate(y, np.full(6, 0.5), 0.5)["log_loss"]
