from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from .build import KEYS
from .config import DATA_DIR, FINAL_GROUP, FUTURE_DIR, MODELS_DIR, RAW_DIR, REPORTS_DIR, SEED, groups
from .profile import CATEGORICAL
from .sources import provenance
from .target import TASKS, task_rows


# ============================================================
# ОБУЧЕНИЕ И ОЦЕНКА
# ============================================================
#
# Для задачи churn_active90 — свой CatBoost, и учится он только
# на train:
#
#   1. строки train задачи делятся на inner_train (80%) и
#      inner_holdout (20%), стратифицированно по target, seed 42.
#      CatBoost учится на inner_train с ранней остановкой по
#      inner_holdout; по прогнозам на том же inner_holdout выбирается
#      порог — наибольший F1;
#   2. CatBoost учится заново на всём train с числом деревьев лучшей
#      модели шага 1 (best_iteration + 1: итерации считаются с нуля).
#      Порог фиксируется с шага 1 и больше не пересчитывается.
#
# val только оценивается — одним инференсом при фиксированном пороге:
# по нему выбирают эксперименты PRAGMA, и ни метки, ни признаки val в
# обучение и выбор порога не попадают. test строится и оценивается
# только в финальной оценке (--final-test) при том же пороге. ROC-AUC
# и PR-AUC считаются по вероятности, без порога.
#
# Объекты оценки сохраняются поимённо (eval_rows.parquet): модель
# PRAGMA + Head оценивается ровно на этих строках, и по score
# baseline сравнение можно делать парным.
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

# Обучение на всём train: те же параметры без ранней остановки,
# число деревьев задаётся явно.
EARLY_STOPPING = ("eval_metric", "od_type", "od_wait", "use_best_model")

# Доля строк train, отложенная для ранней остановки и порога.
HOLDOUT = 0.2

THRESHOLD_RULE = "max F1 on train inner holdout"


def load(group: str, data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / group / "features.parquet")


def check_fresh(group: str, data_dir: Path, raw_dir: Path, future_dir: Path = FUTURE_DIR) -> dict:
    """
    Строки группы обязаны быть собраны из текущих источников: история и
    анкета признаков, источник метки и T. Отчёт по прежней генерации
    или по другому продолжению для сравнений не годится.
    """
    built = json.loads((data_dir / group / "meta.json").read_text())
    current = provenance(group, raw_dir, future_dir)
    changed = sorted(key for key, value in current.items() if built.get(key) != value)
    if changed:
        flag = " --final-test" if group == FINAL_GROUP else ""
        raise ValueError(
            f"строки группы {group} собраны на других источниках ({', '.join(changed)}): "
            f"python -m churn.build {group}{flag}"
        )
    return current


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
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def holdout(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Номера строк inner_train и inner_holdout, стратифицированно по target.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT, random_state=SEED)
    fit, held = next(split.split(np.zeros(len(y)), y))
    return np.sort(fit), np.sort(held)


def fit_task(rows: pd.DataFrame, columns: list[str], categorical: list[str]) -> tuple[CatBoostClassifier, dict]:
    """
    CatBoost задачи по её строкам train: ранняя остановка и порог по
    inner_holdout, затем обучение на всех строках train с числом
    деревьев лучшей модели. Других данных функция не получает.
    """

    def pool(frame: pd.DataFrame) -> Pool:
        return Pool(frame[columns], label=frame["churn"].to_numpy(), cat_features=categorical)

    y = rows["churn"].to_numpy()
    fit, held = holdout(y)
    held_pool = pool(rows.iloc[held])

    stopped = CatBoostClassifier(**PARAMS)
    stopped.fit(pool(rows.iloc[fit]), eval_set=held_pool, verbose=False)

    best = int(stopped.get_best_iteration())
    threshold = threshold_max_f1(y[held], stopped.predict_proba(held_pool)[:, 1])

    params = {key: value for key, value in PARAMS.items() if key not in EARLY_STOPPING}
    model = CatBoostClassifier(**{**params, "iterations": best + 1})
    model.fit(pool(rows), verbose=False)

    def part(index: np.ndarray) -> dict:
        return {"rows": int(len(index)), "positives": int(y[index].sum()), "positive_rate": float(y[index].mean())}

    return model, {
        "inner_train": part(fit),
        "inner_holdout": part(held),
        "threshold": threshold,
        "threshold_rule": THRESHOLD_RULE,
        # Итерации считаются с нуля: у лучшей модели best + 1 деревьев,
        # столько же у модели на всём train.
        "best_iteration": best,
        "trees": int(model.tree_count_),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    raw_dir: Path = RAW_DIR,
    final_test: bool = False,
    future_dir: Path = FUTURE_DIR,
) -> dict:
    used = groups(final_test)

    sources = {group: check_fresh(group, data_dir, raw_dir, future_dir) for group in used}

    frames = {group: load(group, data_dir) for group in used}
    check_groups(frames)

    columns = [name for name in frames["train"].columns if name not in KEYS]
    categorical = [name for name in columns if name in CATEGORICAL]
    for group, frame in frames.items():
        if [name for name in frame.columns if name not in KEYS] != columns:
            raise ValueError(f"признаки группы {group} не совпадают с train")

    evaluated = tuple(group for group in used if group != "train")

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"final_test": final_test, "tasks": {}}
    train_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    importance: list[pd.DataFrame] = []
    distribution: dict = {}

    for task in TASKS:
        rows = {group: task_rows(task, frame) for group, frame in frames.items()}

        model, fitted = fit_task(rows["train"], columns, categorical)
        model.save_model(models_dir / f"catboost_{task}.cbm")

        scores = {
            group: model.predict_proba(Pool(frame[columns], cat_features=categorical))[:, 1]
            for group, frame in rows.items()
        }

        fitted["groups"] = {
            group: evaluate(rows[group]["churn"].to_numpy(), scores[group], fitted["threshold"])
            for group in evaluated
        }
        metrics["tasks"][task] = fitted

        importance.append(
            pd.DataFrame(
                {
                    "task": task,
                    "feature": columns,
                    "importance": model.get_feature_importance(type="PredictionValuesChange"),
                }
            ).sort_values("importance", ascending=False)
        )

        def part(group: str) -> pd.DataFrame:
            frame = rows[group][list(KEYS)].assign(score=scores[group])
            frame.insert(0, "task", task)
            return frame

        train_parts.append(part("train"))
        eval_parts += [part(group) for group in evaluated]

        distribution[task] = {
            group: {
                "rows": int(len(frame)),
                "churn_1": int(frame["churn"].sum()),
                "churn_0": int((frame["churn"] == 0).sum()),
                "churn_rate": float(frame["churn"].mean()),
            }
            for group, frame in rows.items()
        }

    pd.concat(importance, ignore_index=True).to_csv(reports_dir / "feature_importance.csv", index=False)
    pd.concat(eval_parts, ignore_index=True).to_parquet(reports_dir / "eval_rows.parquet", index=False)
    pd.concat(train_parts, ignore_index=True).to_parquet(reports_dir / "train_rows.parquet", index=False)
    (reports_dir / "target_distribution.json").write_text(json.dumps(distribution, ensure_ascii=False, indent=2))

    metrics.update(
        {
            "params": PARAMS,
            "holdout_share": HOLDOUT,
            "features": len(columns),
            "categorical": categorical,
            "eval_rows_sha256": sha256(reports_dir / "eval_rows.parquet"),
            "train_rows_sha256": sha256(reports_dir / "train_rows.parquet"),
            # Отпечатки источников по группам: история и анкета
            # признаков, источник метки, T и окно метки.
            "sources": sources,
        }
    )
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def show(metrics: dict) -> str:
    lines = []
    for task, block in metrics["tasks"].items():
        lines.append(
            f"\n{task}: inner_train {block['inner_train']['rows']}, inner_holdout {block['inner_holdout']['rows']} "
            f"(доля churn {block['inner_holdout']['positive_rate']:.1%}); best_iteration {block['best_iteration']}, "
            f"деревьев {block['trees']}; порог {block['threshold']:.3f} ({block['threshold_rule']})"
        )
        lines.append(
            f"  {'группа':<6} {'строк':>6} {'churn=1':>8} {'доля':>7} {'ROC-AUC':>8} {'PR-AUC':>7} "
            f"{'Precision':>9} {'Recall':>7} {'F1':>6}   TN / FP / FN / TP"
        )
        for group, result in block["groups"].items():
            matrix = result["confusion_matrix"]
            lines.append(
                f"  {group:<6} {result['rows']:>6} {result['positives']:>8} {result['positive_rate']:>7.1%} "
                f"{result['roc_auc']:>8.3f} {result['pr_auc']:>7.3f} {result['precision']:>9.3f} "
                f"{result['recall']:>7.3f} {result['f1']:>6.3f}   "
                f"{matrix['tn']} / {matrix['fp']} / {matrix['fn']} / {matrix['tp']}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CatBoost churn: обучение только на train, оценка на val; test — только с --final-test"
    )
    parser.add_argument("--final-test", action="store_true", help="финальная оценка: оценить и test")
    args = parser.parse_args(argv)
    print(show(train(final_test=args.final_test)))


if __name__ == "__main__":
    main()
