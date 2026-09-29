from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn — на уровне модуля: threadpool_limits в run ограничивает
# только уже загруженные библиотеки потоков.
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from .settings import EMBEDDINGS_META, REPORT_FILE, downstream_dir


# ============================================================
# ПРОБЫ
# ============================================================
#
#   python -m src.downstream.probe --tag best --control init
#
# Над векторами клиентов на момент T учится простая голова —
# логистическая регрессия со стандартизацией, сила регуляризации
# выбирается 3-кратной кросс-валидацией на train по log-loss.
# Train учит, val и test только оцениваются. Главные метрики —
# ROC-AUC и PR-AUC, без порога.
#
# Наборы признаков:
#
#   recency          давность последнего события до T и число событий
#   counts           счётчики типов событий за 30/90/365 дней + recency
#   usr              [USR] модели
#   usr+recency      [USR] и давность: модель давности не видит
#   readouts         [USR], анкета, среднее и последнее событие
#   readouts+recency
#   counts+usr       гибрид: агрегаты и вектор модели
#   init:…           те же векторы необученной модели (--control)
#   catboost         прогноз churn-бейзлайна, без обучения (только churn)
#
# Сравнение парное: bootstrap по клиентам test (одни и те же
# выборки для набора и эталона). Эталон churn — CatBoost, у
# остальных задач — counts. Разница с доверительным интервалом,
# а не два числа рядом: на ~140 положительных в test точечная
# разница в 0.02 PR-AUC — ещё шум.
# ============================================================


TASKS = ("churn", "ndq", "a1")

# Наборы, по которым модели сравниваются друг с другом (--baseline):
# вектор с давностью, все векторы с давностью и гибрид с агрегатами.
COMPARED = ("usr+recency", "readouts+recency", "counts+usr")

PREDICTIONS_FILE = "predictions.parquet"

REFERENCE = {"churn": "catboost", "ndq": "counts", "a1": "counts"}

GROUPS = ("train", "val", "test")

# Сетка силы регуляризации по декадам и 3 фолда: пробы идут после
# каждого эксперимента.
CS = np.logspace(-4, 2, 7)

FOLDS = 3

# Потоков BLAS: пробы идут рядом с обучением, и 12 потоков OpenBLAS,
# деля ядра с ним, замедляли lbfgs в 7–10 раз против 4.
BLAS_THREADS = 4


def load_embeddings(tag: str) -> dict[str, pd.DataFrame]:
    """
    Векторы групп тега: client_id -> колонки векторов как массивы.
    """

    directory = downstream_dir(tag)

    if not (directory / EMBEDDINGS_META).exists():
        raise FileNotFoundError(
            f"нет {directory / EMBEDDINGS_META}: выполните python -m src.downstream.embed"
        )

    return {
        group: pd.read_parquet(directory / f"{group}.parquet").set_index("client_id")
        for group in GROUPS
    }


def stacked(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(frame[column].to_numpy()).astype(np.float64)


def recency(frame: pd.DataFrame) -> np.ndarray:
    """
    Давность последнего события в сутках и число событий, в логарифме.
    """

    return np.column_stack([
        np.log1p(frame["gap_seconds"].to_numpy() / 86_400.0),
        np.log1p(frame["n_events"].to_numpy()),
    ])


def features(
    name: str, group: str, rows: pd.DataFrame, vectors: dict[str, dict[str, pd.DataFrame]], counts: list[str]
) -> np.ndarray:
    """
    Матрица набора name для строк rows группы group (индекс — client_id).
    """

    def vector(tag: str, column: str) -> np.ndarray:
        return stacked(vectors[tag][group].loc[rows.index], column)

    def readouts(tag: str) -> np.ndarray:
        return np.hstack([vector(tag, column) for column in ("usr", "profile", "mean_event", "last_event")])

    tag, _, kind = name.rpartition(":")
    tag = tag or "model"

    parts = {
        "recency": lambda: recency(rows),
        "counts": lambda: np.hstack([np.log1p(rows[counts].to_numpy(dtype=np.float64)), recency(rows)]),
        "usr": lambda: vector(tag, "usr"),
        "usr+recency": lambda: np.hstack([vector(tag, "usr"), recency(rows)]),
        "readouts": lambda: readouts(tag),
        "readouts+recency": lambda: np.hstack([readouts(tag), recency(rows)]),
        "counts+usr": lambda: np.hstack([
            np.log1p(rows[counts].to_numpy(dtype=np.float64)), recency(rows), vector(tag, "usr"),
        ]),
    }

    return parts[kind]()


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, others: list[np.ndarray], seed: int):
    """
    Логистическая регрессия: C — 3-кратной CV на train по log-loss.
    """

    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=CS, cv=StratifiedKFold(FOLDS, shuffle=True, random_state=seed),
            scoring="neg_log_loss", max_iter=5000, l1_ratios=(0.0,), use_legacy_attributes=False,
        ),
    )

    model.fit(train_x, train_y)

    chosen = float(np.ravel(model[-1].C_)[0])

    return [model.predict_proba(x)[:, 1] for x in others], chosen


def metrics(y: np.ndarray, score: np.ndarray) -> dict:

    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


def paired(y: np.ndarray, score: np.ndarray, reference: np.ndarray, draws: int, seed: int) -> dict:
    """
    Разница метрик набора и эталона на одних и тех же bootstrap-
    выборках клиентов: среднее, 95% интервал и доля выборок, где
    набор не лучше эталона.
    """

    rng = np.random.default_rng(seed)

    deltas = {"roc_auc": [], "pr_auc": []}

    for _ in range(draws):

        take = rng.integers(0, len(y), len(y))

        if not 0 < y[take].sum() < len(take):
            continue

        deltas["roc_auc"].append(roc_auc_score(y[take], score[take]) - roc_auc_score(y[take], reference[take]))
        deltas["pr_auc"].append(
            average_precision_score(y[take], score[take]) - average_precision_score(y[take], reference[take])
        )

    return {
        name: {
            "mean": float(np.mean(values)),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "not_better": float(np.mean(np.asarray(values) <= 0.0)),
        }
        for name, values in deltas.items()
    }


def task_rows(task: str, tables: dict[str, pd.DataFrame], churn: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame]:
    """
    Строки задачи по группам: признаки таблицы задач и метка y.
    """

    rows = {}

    for group in GROUPS:

        table = tables[group]

        if task == "churn":
            base = churn[group]
            frame = pd.concat(
                [table.loc[base.index], pd.DataFrame(
                    {"y": base["churn"].astype(int), "catboost": base["score"]}, index=base.index
                )],
                axis=1,
            )
        elif task == "ndq":
            frame = table[table["ndq_population"]].assign(y=lambda part: part["ndq"].astype(int))
        else:
            frame = table.assign(y=table["a1"].astype(int))

        rows[group] = frame

    return rows


def run_probe(tag: str, control: str | None, draws: int, seed: int, baseline: str | None = None) -> dict:

    from .tasks import COUNT_WINDOWS, churn_rows, table

    vectors = {"model": load_embeddings(tag)}

    if control:
        vectors[control] = load_embeddings(control)

    tables = {group: table(group) for group in GROUPS}
    churn = {group: churn_rows(group) for group in GROUPS}

    # Признаки-счётчики — по train: тип, которого в train нет, голова
    # не выучит. В других группах недостающий тип — нули.
    windows = tuple(f"n_{days}d_" for days in COUNT_WINDOWS)
    counts = [name for name in tables["train"].columns if name.startswith(windows)]

    for group in ("val", "test"):
        missing = [name for name in counts if name not in tables[group]]
        tables[group] = tables[group].assign(**{name: 0 for name in missing})

    names = ["recency", "counts", "usr", "usr+recency", "readouts", "readouts+recency", "counts+usr"]

    if control:
        names += [f"{control}:usr", f"{control}:readouts+recency"]

    report: dict = {"tag": tag, "control": control, "baseline": baseline, "draws": draws, "tasks": {}}

    # Прогнозы по строкам: по ним сравниваются модели между собой.
    predictions: list[pd.DataFrame] = []

    before = (
        pd.read_parquet(downstream_dir(baseline) / PREDICTIONS_FILE) if baseline else None
    )

    for task in TASKS:

        rows = task_rows(task, tables, churn)

        # Строки без вектора — ошибка, а не пропуск: сравнение с
        # бейзлайном обязано идти на тех же клиентах.
        for tag_name, groups in vectors.items():
            for group, frame in rows.items():
                missing = frame.index.difference(groups[group].index)
                if len(missing):
                    raise ValueError(
                        f"{task}/{group}: у {len(missing)} строк нет вектора ({tag_name}), "
                        f"например {missing[0]}"
                    )

        y = {group: rows[group]["y"].to_numpy() for group in GROUPS}

        scores: dict[str, dict[str, np.ndarray]] = {}
        chosen: dict[str, float] = {}

        for name in names:

            matrices = {
                group: features(name if ":" in name else f"model:{name}", group, rows[group], vectors, counts)
                for group in GROUPS
            }

            (val_score, test_score), chosen[name] = fit_predict(
                matrices["train"], y["train"], [matrices["val"], matrices["test"]], seed
            )

            scores[name] = {"val": val_score, "test": test_score}

        if task == "churn":
            scores["catboost"] = {group: rows[group]["catboost"].to_numpy() for group in ("val", "test")}

        for name, by_group in scores.items():
            for group in ("val", "test"):
                predictions.append(pd.DataFrame({
                    "task": task, "set": name, "group": group, "client_id": rows[group].index,
                    "y": y[group], "score": by_group[group],
                }))

        reference = REFERENCE[task]

        results = {}

        for name, by_group in scores.items():
            results[name] = {
                "val": metrics(y["val"], by_group["val"]),
                "test": metrics(y["test"], by_group["test"]),
                "C": chosen.get(name),
            }
            if name != reference:
                results[name]["vs_reference_test"] = paired(
                    y["test"], by_group["test"], scores[reference]["test"], draws, seed
                )
            if before is not None and name in COMPARED:
                results[name]["vs_baseline"] = {
                    group: vs_baseline(before, task, name, group, rows[group].index, y[group],
                                       by_group[group], draws, seed)
                    for group in ("val", "test")
                }

        report["tasks"][task] = {
            "reference": reference,
            "rows": {group: int(len(rows[group])) for group in GROUPS},
            "positives": {group: int(y[group].sum()) for group in GROUPS},
            "results": results,
        }

    directory = downstream_dir(tag)
    pd.concat(predictions, ignore_index=True).to_parquet(directory / PREDICTIONS_FILE)

    return report


def vs_baseline(before: pd.DataFrame, task: str, name: str, group: str, index: pd.Index,
                y: np.ndarray, score: np.ndarray, draws: int, seed: int) -> dict:
    """
    Разница с прогнозом другой модели на тех же клиентах и той же
    метке — парный bootstrap.
    """

    old = before[(before["task"] == task) & (before["set"] == name) & (before["group"] == group)]
    old = old.set_index("client_id").reindex(index)

    if old["score"].isna().any() or not np.array_equal(old["y"].to_numpy(), y):
        raise ValueError(f"{task}/{name}/{group}: строки или метки базовой модели другие")

    return paired(y, score, old["score"].to_numpy(), draws, seed)


def show(report: dict) -> str:

    lines = []

    for task, block in report["tasks"].items():

        lines.append(
            f"\n{task}: строк train/val/test {block['rows']['train']}/{block['rows']['val']}/"
            f"{block['rows']['test']}, положительных {block['positives']['train']}/"
            f"{block['positives']['val']}/{block['positives']['test']}; эталон {block['reference']}"
        )
        lines.append(f"  {'набор':<24} {'val ROC':>8} {'val PR':>7} {'test ROC':>9} {'test PR':>8}   Δ test PR к эталону [95%]")

        for name, result in block["results"].items():

            delta = result.get("vs_reference_test")
            shown = (
                f"{delta['pr_auc']['mean']:+.3f} [{delta['pr_auc']['low']:+.3f}, {delta['pr_auc']['high']:+.3f}]"
                if delta else "эталон"
            )

            versus = result.get("vs_baseline")
            if versus:
                shown += (
                    f"   Δ к {report['baseline']}: val PR {versus['val']['pr_auc']['mean']:+.3f} "
                    f"[{versus['val']['pr_auc']['low']:+.3f}, {versus['val']['pr_auc']['high']:+.3f}], "
                    f"val ROC {versus['val']['roc_auc']['mean']:+.3f}"
                )

            lines.append(
                f"  {name:<24} {result['val']['roc_auc']:8.3f} {result['val']['pr_auc']:7.3f} "
                f"{result['test']['roc_auc']:9.3f} {result['test']['pr_auc']:8.3f}   {shown}"
            )

    return "\n".join(lines)


def run(args) -> int:

    try:
        with threadpool_limits(BLAS_THREADS):
            report = run_probe(args.tag, args.control, args.draws, args.seed, args.baseline)
    except (FileNotFoundError, ValueError) as error:
        print(f"[probe] {error}")
        return EXIT_BLOCKED

    path = downstream_dir(args.tag) / REPORT_FILE
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(show(report))
    print(f"\n[probe] → {path}")

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.downstream.probe")
    parser.add_argument("--tag", required=True, help="каталог векторов в data/15_downstream")
    parser.add_argument("--control", default=None, help="тег векторов-контроля, например init")
    parser.add_argument("--baseline", default=None, help="тег модели для парного сравнения")
    parser.add_argument("--draws", type=int, default=1000, help="bootstrap-выборок")
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(handler=run)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()


__all__ = ["COMPARED", "REFERENCE", "TASKS", "features", "fit_predict", "metrics", "paired", "run_probe",
           "show", "vs_baseline"]
