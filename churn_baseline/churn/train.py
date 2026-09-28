from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from .build import KEYS
from .config import DATA_DIR, GROUPS, MODELS_DIR, RAW_DIR, REPORTS_DIR, SEED, manifest
from .profile import CATEGORICAL


# ============================================================
# ОБУЧЕНИЕ И ОЦЕНКА
# ============================================================
#
#   train  обучение;
#   val    ранняя остановка и выбор порога по наибольшему F1;
#   test   одна финальная оценка при пороге, выбранном на val.
#
# Объекты оценки сохраняются поимённо (eval_rows.parquet): модель
# PRAGMA + Head оценивается ровно на этих строках, и по score baseline
# сравнение можно делать парным.
# ============================================================


PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "od_type": "Iter",
    "od_wait": 300,
    "use_best_model": True,
    "random_seed": SEED,
    "thread_count": 8,
    "allow_writing_files": False,
}

# Срез: клиенты с действием за 90 дней до T. Показывает, сколько качества
# дают давно замолчавшие клиенты.
RECENT_SLICE = "act_count_90"


def load(group: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / group / "features.parquet")


def check_groups(frames: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for group, frame in frames.items():
        if (frame["group"] != group).any() or frame["client_id"].duplicated().any():
            raise ValueError(f"строки группы {group} повреждены")
        for client in frame["client_id"]:
            if client in seen:
                raise ValueError(f"клиент {client} в группах {seen[client]} и {group}")
            seen[client] = group


def threshold_max_f1(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    # Последняя точка кривой порога не имеет.
    return float(thresholds[int(np.argmax(f1[:-1]))])


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(data_dir: Path = DATA_DIR, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR, raw_dir: Path = RAW_DIR) -> dict:
    frames = {group: load(group, data_dir) for group in GROUPS}
    check_groups(frames)

    columns = [name for name in frames["train"].columns if name not in KEYS]
    categorical = [name for name in columns if name in CATEGORICAL]
    for group, frame in frames.items():
        if [name for name in frame.columns if name not in KEYS] != columns:
            raise ValueError(f"признаки группы {group} не совпадают с train")

    def pool(frame: pd.DataFrame) -> Pool:
        return Pool(frame[columns], label=frame["churn"].to_numpy(), cat_features=categorical)

    model = CatBoostClassifier(**PARAMS)
    model.fit(pool(frames["train"]), eval_set=pool(frames["val"]), verbose=200)

    scores = {group: model.predict_proba(pool(frame))[:, 1] for group, frame in frames.items()}
    threshold = threshold_max_f1(frames["val"]["churn"].to_numpy(), scores["val"])

    metrics: dict = {"threshold": threshold, "threshold_rule": "max F1 on val", "groups": {}}
    for group in ("val", "test"):
        frame = frames[group]
        y = frame["churn"].to_numpy()
        recent = frame[RECENT_SLICE].to_numpy() > 0
        metrics["groups"][group] = {
            "all": evaluate(y, scores[group], threshold),
            "active_90_days": evaluate(y[recent], scores[group][recent], threshold),
        }

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / "catboost.cbm")

    importance = pd.DataFrame(
        {"feature": columns, "importance": model.get_feature_importance(type="PredictionValuesChange")}
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    def rows(group_names: tuple[str, ...]) -> pd.DataFrame:
        return pd.concat(
            [frames[g][list(KEYS)].assign(score=scores[g]) for g in group_names], ignore_index=True
        )

    rows(("val", "test")).to_parquet(reports_dir / "eval_rows.parquet", index=False)
    rows(("train",)).to_parquet(reports_dir / "train_rows.parquet", index=False)

    distribution = {
        group: {
            "rows": int(len(frame)),
            "churn_1": int(frame["churn"].sum()),
            "churn_0": int((frame["churn"] == 0).sum()),
            "churn_rate": float(frame["churn"].mean()),
        }
        for group, frame in frames.items()
    }
    (reports_dir / "target_distribution.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=2))

    metrics.update(
        {
            "best_iteration": int(model.get_best_iteration()),
            "params": PARAMS,
            "features": len(columns),
            "categorical": categorical,
            "eval_rows_sha256": sha256(reports_dir / "eval_rows.parquet"),
            "train_rows_sha256": sha256(reports_dir / "train_rows.parquet"),
            "raw_events_sha256": {group: manifest(group, raw_dir)["events_sha256"] for group in GROUPS},
        }
    )
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="CatBoost churn: обучение на train, порог на val, оценка на test").parse_args(argv)
    metrics = train()
    print(json.dumps({key: metrics[key] for key in ("threshold", "best_iteration", "groups")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
