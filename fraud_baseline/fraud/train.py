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
from .features import CATEGORICAL


# ============================================================
# ОБУЧЕНИЕ И ОЦЕНКА
# ============================================================
#
#   train  обучение на всех позитивах и прореженных отрицательных с весом;
#   val    все строки: ранняя остановка и порог по наибольшему F1;
#   test   все строки: одна финальная оценка при пороге с val.
#
# Объекты оценки сохраняются поимённо (eval_rows.parquet): модель
# PRAGMA + Fraud Head оценивается ровно на этих строках, и по score
# baseline сравнение можно делать парным.
# ============================================================


PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 6,
    "od_type": "Iter",
    "od_wait": 300,
    "use_best_model": True,
    "random_seed": SEED,
    "thread_count": 8,
    "allow_writing_files": False,
}

ROW_KEYS: tuple[str, ...] = ("group", "client_id", "event_time", "raw_row", "type", "fraud")


def check_groups(frames: dict[str, pd.DataFrame]) -> None:
    owner: dict[str, str] = {}
    for group, frame in frames.items():
        if (frame["group"] != group).any() or frame["raw_row"].duplicated().any():
            raise ValueError(f"строки группы {group} повреждены")
        for client in pd.unique(frame["client_id"]):
            if client in owner:
                raise ValueError(f"клиент {client} в группах {owner[client]} и {group}")
            owner[client] = group


def threshold_max_f1(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = 0 < y.sum() < len(y)
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, score)) if both else None,
        "pr_auc": float(average_precision_score(y, score)) if both else None,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(data_dir: Path = DATA_DIR, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR, raw_dir: Path = RAW_DIR) -> dict:
    frames = {group: pd.read_parquet(data_dir / group / "features.parquet") for group in GROUPS}
    check_groups(frames)

    columns = [name for name in frames["train"].columns if name not in KEYS]
    categorical = [name for name in columns if name in CATEGORICAL]
    for group, frame in frames.items():
        if [name for name in frame.columns if name not in KEYS] != columns:
            raise ValueError(f"признаки группы {group} не совпадают с train")

    def pool(frame: pd.DataFrame, weighted: bool) -> Pool:
        weight = frame["weight"].to_numpy() if weighted else None
        return Pool(frame[columns], label=frame["fraud"].to_numpy(), weight=weight, cat_features=categorical)

    model = CatBoostClassifier(**PARAMS)
    model.fit(pool(frames["train"], True), eval_set=pool(frames["val"], False), verbose=100, metric_period=5)

    scores = {group: model.predict_proba(pool(frame, False))[:, 1] for group, frame in frames.items()}
    threshold = threshold_max_f1(frames["val"]["fraud"].to_numpy(), scores["val"])

    metrics: dict = {"threshold": threshold, "threshold_rule": "max F1 on val", "groups": {}}
    for group in ("val", "test"):
        frame = frames[group]
        y = frame["fraud"].to_numpy()
        report = {"all": evaluate(y, scores[group], threshold)}
        for kind in ("purchase", "transfer_out"):
            mask = (frame["type"] == kind).to_numpy()
            report[kind] = evaluate(y[mask], scores[group][mask], threshold)
        hard = (frame["episode_kind"] == "false_positive").to_numpy()
        report["false_positive_flagged"] = {
            "rows": int(hard.sum()),
            "above_threshold": int((scores[group][hard] >= threshold).sum()),
        }
        metrics["groups"][group] = report

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(models_dir / "catboost.cbm")

    importance = pd.DataFrame(
        {"feature": columns, "importance": model.get_feature_importance(type="PredictionValuesChange")}
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    def rows(names: tuple[str, ...], extra: tuple[str, ...] = ()) -> pd.DataFrame:
        return pd.concat(
            [frames[g][list(ROW_KEYS + extra)].assign(score=scores[g]) for g in names], ignore_index=True
        )

    rows(("val", "test")).to_parquet(reports_dir / "eval_rows.parquet", index=False)
    rows(("train",), ("weight",)).to_parquet(reports_dir / "train_rows.parquet", index=False)

    distribution = {}
    for group, frame in frames.items():
        meta = json.loads((data_dir / group / "meta.json").read_text())
        distribution[group] = {
            "rows": int(len(frame)),
            "fraud_1": int(frame["fraud"].sum()),
            "fraud_0": int((frame["fraud"] == 0).sum()),
            "fraud_rate": float(frame["fraud"].mean()),
            "false_positive_rows": int((frame["episode_kind"] == "false_positive").sum()),
            "by_type": {
                kind: {"rows": int(len(part)), "fraud_1": int(part["fraud"].sum())}
                for kind, part in frame.groupby("type")
            },
            "by_episode_kind": frame[frame["episode_kind"].notna()].groupby("episode_kind").size().to_dict(),
            "eligible_before_sampling": meta["eligible"],
            "eligible_fraud_before_sampling": meta["eligible_fraud"],
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
    argparse.ArgumentParser(description="CatBoost fraud: обучение на train, порог на val, оценка на test").parse_args(argv)
    metrics = train()
    print(json.dumps({key: metrics[key] for key in ("threshold", "best_iteration", "groups")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
