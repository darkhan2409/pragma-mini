from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

# sklearn — на уровне модуля: threadpool_limits в run ограничивает
# только уже загруженные библиотеки потоков.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from .settings import EMBEDDINGS_META, REPORT_FILE, cutoff, downstream_dir, groups


# ============================================================
# ПРОБЫ
# ============================================================
#
#   python -m src.downstream.probe --tag w4-b0
#   python -m src.downstream.probe --tag w4-b0 --final-test   # финальная оценка
#
# Три сценария на одних и тех же клиентах задачи churn_active90:
#
#   catboost           handcrafted-признаки → CatBoost (churn_baseline,
#                      готовый прогноз)
#   usr                [USR] после энкодера истории → логистическая
#                      регрессия (учится здесь)
#   catboost_plus_usr  handcrafted-признаки и [USR] → CatBoost
#                      (python -m churn.plus_usr, готовый прогноз; если
#                      для этих векторов он не обучен — его нет)
#
# Голова над [USR] — логистическая регрессия со стандартизацией, сила
# регуляризации выбирается 3-кратной кросс-валидацией на train по
# log-loss. Стандартизация — часть pipeline и учится внутри каждого
# фолда; после выбора C pipeline учится заново на всём train.
# Train учит, val только оценивается — по нему выбирают
# эксперименты. test не считается вовсе, пока не задан --final-test:
# тогда к val добавляются метрики и сравнения на test. Главные
# метрики — ROC-AUC и PR-AUC, без порога; рядом log-loss — средняя
# кросс-энтропия вероятности, она видит ещё и калибровку.
#
# Порог (Precision, Recall, F1, матрица ошибок) — только от train: у
# регрессии max F1 по out-of-fold вероятностям train тех же фолдов, у
# CatBoost — его порог с inner_holdout train. val порога не касается.
#
# Сравнение парное: bootstrap по клиентам val (одни и те же выборки
# для сценария и эталона). Эталон — CatBoost на handcrafted-признаках.
# С --baseline сценарии с векторами сравниваются так же с прогнозами
# другой модели PRAGMA на тех же клиентах и той же метке. Разница с
# доверительным интервалом, а не два числа рядом: на ~50 положительных
# точечная разница в 0.02 PR-AUC — ещё шум.
# ============================================================


TASKS = ("churn_active90",)

# Сценарии в порядке отчёта.
SCENARIOS = ("catboost", "usr", "catboost_plus_usr")

# Сценарии, по которым модели PRAGMA сравниваются друг с другом
# (--baseline): те, что зависят от векторов модели.
COMPARED = ("usr", "catboost_plus_usr")

PREDICTIONS_FILE = "predictions.parquet"

REFERENCE = {"churn_active90": "catboost"}

# Сетка силы регуляризации по декадам и 3 фолда: пробы идут после
# каждого эксперимента.
CS = np.logspace(-4, 2, 7)

FOLDS = 3

# Потоков BLAS: пробы идут рядом с обучением, и 12 потоков OpenBLAS,
# деля ядра с ним, замедляли lbfgs в 7–10 раз против 4.
BLAS_THREADS = 4


def load_embeddings(tag: str, used: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """
    Векторы групп тега: client_id -> колонки векторов как массивы.
    """

    directory = downstream_dir(tag)

    if not (directory / EMBEDDINGS_META).exists():
        raise FileNotFoundError(
            f"нет {directory / EMBEDDINGS_META}: выполните python -m src.downstream.embed"
        )

    embedded = json.loads((directory / EMBEDDINGS_META).read_text(encoding="utf-8")).get("groups", {})
    missing = [group for group in used if group not in embedded]

    if missing:
        raise FileNotFoundError(
            f"{tag}: нет векторов {', '.join(missing)} — python -m src.downstream.embed"
            + (" --final-test" if "test" in missing else "")
        )

    # Векторы — из текущей выгрузки, той же, из которой строятся
    # признаки churn-бейзлайна: иначе строки задач и векторы были бы
    # про разные истории одних клиентов.
    from .embed import raw_record

    current = {group: raw_record(group) for group in used}
    stale = [group for group in used if {key: embedded[group].get(key) for key in current[group]} != current[group]]

    if stale:
        raise ValueError(
            f"{tag}: векторы {', '.join(stale)} сняты с другой выгрузки (или прежним кодом без её "
            "отпечатков) — python -m src.downstream.embed заново"
        )

    # Векторы сняты ровно на T группы — тот же, что у строк churn
    # (churn_rows): иначе вектор и метка были бы про разные моменты.
    frames = {}

    for group in used:

        moment = pd.Timestamp(cutoff(group))

        if pd.Timestamp(embedded[group].get("cutoff")) != moment:
            raise ValueError(
                f"{tag}: векторы {group} сняты на {embedded[group].get('cutoff')}, а T группы — "
                f"{moment.isoformat()}"
            )

        frame = pd.read_parquet(directory / f"{group}.parquet")

        if frame["client_id"].duplicated().any():
            raise ValueError(f"{tag}: в векторах {group} клиент повторяется")

        if not (pd.to_datetime(frame["cutoff"], utc=True) == moment).all():
            raise ValueError(f"{tag}: в векторах {group} есть строки не на T {moment.isoformat()}")

        frames[group] = frame.set_index("client_id")

    return frames


def vectors_meta(tag: str) -> dict:

    return json.loads((downstream_dir(tag) / EMBEDDINGS_META).read_text(encoding="utf-8"))


def usr_matrix(vectors: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """
    [USR] строк rows (индекс — client_id) в их порядке.
    """

    return np.stack(vectors.loc[rows.index, "usr"].to_numpy()).astype(np.float64)


def probe_model(seed: int) -> GridSearchCV:
    """
    Стандартизация и логистическая регрессия одной pipeline; C —
    3-кратной CV на train по log-loss. Стандартизация учится внутри
    каждого фолда на его обучающей части, а не на всём train до CV:
    иначе проверочная часть фолда заранее знала бы свои среднее и
    разброс. После выбора C pipeline учится на всём train.
    """

    return GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {"logisticregression__C": CS},
        cv=StratifiedKFold(FOLDS, shuffle=True, random_state=seed),
        scoring="neg_log_loss",
        refit=True,
    )


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, others: list[np.ndarray], seed: int):
    """
    Голова по train и её вероятности на others; выбранный C и кривая
    перебора — средний log-loss фолдов при каждом C сетки.
    """

    model = probe_model(seed)

    model.fit(train_x, train_y)

    chosen = float(model.best_params_["logisticregression__C"])

    cv = [
        {"C": float(c), "log_loss": float(-score)}
        for c, score in zip(model.cv_results_["param_logisticregression__C"], model.cv_results_["mean_test_score"])
    ]

    return [model.predict_proba(x)[:, 1] for x in others], chosen, cv


def threshold_max_f1(y: np.ndarray, score: np.ndarray) -> float:
    """
    Порог наибольшего F1 — то же правило, что у churn-бейзлайна.
    """

    precision, recall, thresholds = precision_recall_curve(y, score)

    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    # Последняя точка кривой порога не имеет.
    return float(thresholds[int(np.argmax(f1[:-1]))])


def oof_threshold(train_x: np.ndarray, train_y: np.ndarray, chosen: float, seed: int) -> float:
    """
    Порог пробы: max F1 по out-of-fold вероятностям train. Каждую строку
    предсказывает голова с выбранным C, её не видевшая, на тех же
    фолдах, что выбирали C. val порога не касается.
    """

    head = make_pipeline(StandardScaler(), LogisticRegression(C=chosen, max_iter=5000))
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(head, train_x, train_y, cv=folds, method="predict_proba")[:, 1]

    return threshold_max_f1(train_y, oof)


def at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """
    Precision, Recall, F1 и матрица ошибок при фиксированном пороге.
    """

    predicted = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def metrics(y: np.ndarray, score: np.ndarray) -> dict:

    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
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


def task_rows(task: str, churn: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """
    Строки задачи по группам — строки churn-бейзлайна: метка y и
    прогноз CatBoost.
    """

    return {
        group: pd.DataFrame({"y": base["churn"].astype(int), "catboost": base["score"]}, index=base.index)
        for group, base in churn[task].items()
    }


def run_probe(tag: str, draws: int, seed: int, baseline: str | None = None, final_test: bool = False) -> dict:

    from .tasks import CHURN_TASKS, churn_rows, churn_thresholds, plus_usr

    used = groups(final_test)

    # Оцениваются все группы, кроме train: val, а в финальной оценке
    # и test.
    evaluated = used[1:]

    vectors = load_embeddings(tag, used)
    churn = {task: {group: churn_rows(task, group) for group in used} for task in CHURN_TASKS}
    thresholds = churn_thresholds()
    plus = plus_usr(tag, used, vectors_meta(tag))

    report: dict = {
        "tag": tag, "baseline": baseline, "draws": draws,
        "final_test": final_test, "groups": list(used), "plus_usr": plus is not None, "tasks": {},
    }

    # Прогнозы по строкам: по ним сравниваются модели между собой.
    predictions: list[pd.DataFrame] = []

    before = (
        pd.read_parquet(downstream_dir(baseline) / PREDICTIONS_FILE) if baseline else None
    )

    for task in TASKS:

        rows = task_rows(task, churn)

        # Строки без вектора — ошибка, а не пропуск: сравнение с
        # бейзлайном обязано идти на тех же клиентах.
        for group, frame in rows.items():
            missing = frame.index.difference(vectors[group].index)
            if len(missing):
                raise ValueError(
                    f"{task}/{group}: у {len(missing)} строк нет вектора, например {missing[0]}"
                )

        y = {group: rows[group]["y"].to_numpy() for group in used}
        usr = {group: usr_matrix(vectors[group], rows[group]) for group in used}

        predicted, chosen, cv = fit_predict(usr["train"], y["train"], [usr[group] for group in evaluated], seed)

        scores: dict[str, dict[str, np.ndarray]] = {
            "catboost": {group: rows[group]["catboost"].to_numpy() for group in evaluated},
            "usr": dict(zip(evaluated, predicted)),
        }
        cut = {"catboost": thresholds[task], "usr": oof_threshold(usr["train"], y["train"], chosen, seed)}

        if plus is not None:
            for group in used:
                other = plus["rows"].get((task, group))
                if other is None or not other.index.equals(rows[group].index) or not np.array_equal(
                    other["churn"].to_numpy(), y[group]
                ):
                    raise ValueError(f"catboost_plus_usr/{task}/{group}: другие клиенты или метки, чем у catboost")
            scores["catboost_plus_usr"] = {
                group: plus["rows"][(task, group)]["score"].to_numpy() for group in evaluated
            }
            cut["catboost_plus_usr"] = plus["thresholds"][task]

        for name, by_group in scores.items():
            for group in evaluated:
                predictions.append(pd.DataFrame({
                    "task": task, "set": name, "group": group, "client_id": rows[group].index,
                    "y": y[group], "score": by_group[group],
                }))

        reference = REFERENCE[task]

        results = {}

        for name, by_group in scores.items():
            results[name] = {
                group: {**metrics(y[group], by_group[group]), **at_threshold(y[group], by_group[group], cut[name])}
                for group in evaluated
            }
            results[name]["C"] = chosen if name == "usr" else None
            results[name]["threshold"] = cut[name]
            if name == "usr":
                results[name]["cv"] = cv
            if name != reference:
                results[name]["vs_reference"] = {
                    group: paired(y[group], by_group[group], scores[reference][group], draws, seed)
                    for group in evaluated
                }
            if before is not None and name in COMPARED:
                results[name]["vs_baseline"] = {
                    group: vs_baseline(before, task, name, group, rows[group].index, y[group],
                                       by_group[group], draws, seed)
                    for group in evaluated
                }

        report["tasks"][task] = {
            "reference": reference,
            "rows": {group: int(len(rows[group])) for group in used},
            "positives": {group: int(y[group].sum()) for group in used},
            "positive_rate": {group: float(y[group].mean()) for group in used},
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

    if old.empty:
        raise ValueError(
            f"{task}/{name}/{group}: у базовой модели нет таких прогнозов — её пробы запущены "
            "другим кодом или без --final-test"
        )

    old = old.set_index("client_id").reindex(index)

    if old["score"].isna().any() or not np.array_equal(old["y"].to_numpy(), y):
        raise ValueError(f"{task}/{name}/{group}: строки или метки базовой модели другие")

    return paired(y, score, old["score"].to_numpy(), draws, seed)


def verdict(delta: dict) -> str:
    """
    Что говорит интервал разницы catboost_plus_usr − catboost по PR-AUC.
    """

    if delta["low"] > 0:
        return "интервал выше нуля: [USR] даёт сигнал сверх handcrafted-признаков"
    if delta["high"] < 0:
        return "интервал ниже нуля: добавление [USR] ухудшает модель"
    return "интервал захватывает ноль: [USR] почти ничего не добавляет"


def show(report: dict) -> str:

    lines = []

    # Оцениваемые группы: val, в финальной оценке ещё test.
    evaluated = report["groups"][1:]

    def interval(delta: dict) -> str:
        return f"{delta['mean']:+.3f} [{delta['low']:+.3f}, {delta['high']:+.3f}]"

    for task, block in report["tasks"].items():

        lines.append(f"\n{task}: эталон {block['reference']}")

        for group in report["groups"]:
            lines.append(
                f"  {group:<5} строк {block['rows'][group]:>6}, положительных {block['positives'][group]:>5}, "
                f"доля {block['positive_rate'][group]:.1%}"
            )

        for group in evaluated:

            lines.append(
                f"  {group}: {'сценарий':<20} {'PR-AUC':>7} {'ROC-AUC':>8} {'LogLoss':>8} {'F1':>6}   "
                "Δ PR-AUC к эталону [95%]"
            )

            for name in SCENARIOS:
                result = block["results"].get(name)
                if result is None:
                    continue
                delta = result.get("vs_reference")
                shown = interval(delta[group]["pr_auc"]) if delta else "эталон"
                versus = result.get("vs_baseline")
                if versus:
                    shown += (
                        f"   Δ к {report['baseline']}: PR {interval(versus[group]['pr_auc'])}, "
                        f"ROC {versus[group]['roc_auc']['mean']:+.3f}"
                    )
                cells = result[group]
                lines.append(
                    f"  {'':<{len(group) + 1}} {name:<20} {cells['pr_auc']:7.3f} {cells['roc_auc']:8.3f} "
                    f"{cells['log_loss']:8.4f} {cells['f1']:6.3f}   {shown}"
                )

            plus = block["results"].get("catboost_plus_usr")

            if plus is None:
                lines.append("  catboost_plus_usr нет: cd churn_baseline && python -m churn.plus_usr --embeddings <векторы>")
            else:
                pr, roc = plus["vs_reference"][group]["pr_auc"], plus["vs_reference"][group]["roc_auc"]
                lines.append(
                    f"  Δ catboost_plus_usr − catboost ({group}): PR-AUC {interval(pr)}, ROC-AUC {interval(roc)} — "
                    f"{verdict(pr)}"
                )

    return "\n".join(lines)


def run(args) -> int:

    try:
        with threadpool_limits(BLAS_THREADS):
            report = run_probe(args.tag, args.draws, args.seed, args.baseline, args.final_test)
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
    parser.add_argument("--tag", required=True, help="каталог векторов в data/13_downstream")
    parser.add_argument("--baseline", default=None, help="тег модели для парного сравнения")
    parser.add_argument("--draws", type=int, default=1000, help="bootstrap-выборок")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--final-test", action="store_true",
        help="финальная оценка: метрики и сравнения ещё и на test (один раз, после выбора модели)",
    )
    parser.set_defaults(handler=run)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()


__all__ = ["COMPARED", "REFERENCE", "SCENARIOS", "TASKS", "at_threshold", "fit_predict", "metrics", "oof_threshold",
           "paired", "probe_model", "run_probe", "show", "threshold_max_f1", "usr_matrix", "verdict", "vectors_meta",
           "vs_baseline"]
