from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn — на уровне модуля: threadpool_limits в run ограничивает
# только уже загруженные библиотеки потоков.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
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
#   python -m src.downstream.probe --tag best --control init
#   python -m src.downstream.probe --tag best --final-test   # финальная оценка
#
# Над векторами клиентов на момент T учится простая голова —
# логистическая регрессия со стандартизацией, сила регуляризации
# выбирается 3-кратной кросс-валидацией на train по log-loss.
# Стандартизация — часть pipeline и учится внутри каждого фолда;
# после выбора C pipeline учится заново на всём train.
# Train учит, val только оценивается — по нему выбирают
# эксперименты. test не считается вовсе, пока не задан --final-test:
# тогда к val добавляются метрики и сравнения на test. Главные
# метрики — ROC-AUC и PR-AUC, без порога.
#
# Задача: churn_active90 (строки и CatBoost — churn_baseline). Все
# наборы оцениваются на одних и тех же клиентах.
#
# Наборы признаков:
#
#   recency          давность последнего события до T и число событий
#   counts           счётчики типов событий за 30/90/365 дней + recency
#   usr              [USR] модели
#   usr+recency      [USR] и давность: модель давности не видит
#   usr+last_event   [USR] и последнее событие после истории —
#                    комбинация пробы из статьи PRAGMA (§3.1.1)
#   readouts         [USR], анкета, среднее и последнее событие
#   readouts+recency
#   counts+usr       гибрид: агрегаты и вектор модели
#   init:…           те же векторы необученной модели (--control)
#   catboost         прогноз CatBoost-бейзлайна на полном X, без обучения
#   catboost_plus_usr  прогноз CatBoost на полном X бейзлайна и [USR]
#                    этих векторов (python -m churn.plus_usr), без
#                    обучения; если он для векторов не обучен — нет
#   catboost_usr     прогноз CatBoost только на [USR] (churn.plus_usr
#                    --usr-only) — так же; у контроля — по его векторам
#
# Диагностика [USR] (--control init): 2×2 — векторы модели и
# контроля, логистическая регрессия и CatBoost на одном [USR]. Парные
# разницы: модель − контроль при той же голове (что дало обучение) и
# CatBoost − регрессия на тех же векторах (что даёт нелинейная голова).
#
# Порог (Precision, Recall, F1, матрица ошибок) — только от train: у
# проб max F1 по out-of-fold вероятностям train тех же фолдов, у
# CatBoost — его порог с inner_holdout train. val порога не касается.
#
# Сравнение парное: bootstrap по клиентам val (одни и те же выборки
# для набора и эталона). Эталон — CatBoost задачи. С --baseline наборы векторов сравниваются
# так же с прогнозами другой модели PRAGMA на тех же клиентах и той
# же метке. Разница с доверительным интервалом, а не два числа
# рядом: на ~100 положительных точечная разница в 0.02 PR-AUC — ещё
# шум.
# ============================================================


TASKS = ("churn_active90",)

# Наборы, по которым модели PRAGMA сравниваются друг с другом
# (--baseline): все наборы с векторами модели.
COMPARED = ("usr", "usr+recency", "usr+last_event", "readouts", "readouts+recency", "counts+usr")

PREDICTIONS_FILE = "predictions.parquet"

# Нелинейная голова и линейная на тех же векторах: набор → его
# линейная пара.
HEADS = {"catboost_usr": "usr"}

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
        "usr+last_event": lambda: np.hstack([vector(tag, "usr"), vector(tag, "last_event")]),
        "readouts": lambda: readouts(tag),
        "readouts+recency": lambda: np.hstack([readouts(tag), recency(rows)]),
        "counts+usr": lambda: np.hstack([
            np.log1p(rows[counts].to_numpy(dtype=np.float64)), recency(rows), vector(tag, "usr"),
        ]),
    }

    return parts[kind]()


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
    Голова по train и её вероятности на others; выбранный C.
    """

    model = probe_model(seed)

    model.fit(train_x, train_y)

    chosen = float(model.best_params_["logisticregression__C"])

    return [model.predict_proba(x)[:, 1] for x in others], chosen


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


def task_rows(
    task: str, tables: dict[str, pd.DataFrame], churn: dict[str, dict[str, pd.DataFrame]]
) -> dict[str, pd.DataFrame]:
    """
    Строки задачи по группам — строки churn-бейзлайна: признаки таблицы
    задач, метка y и прогноз CatBoost.
    """

    rows = {}

    for group, table in tables.items():
        base = churn[task][group]
        rows[group] = pd.concat(
            [table.loc[base.index], pd.DataFrame(
                {"y": base["churn"].astype(int), "catboost": base["score"]}, index=base.index
            )],
            axis=1,
        )

    return rows


def run_probe(
    tag: str, control: str | None, draws: int, seed: int, baseline: str | None = None, final_test: bool = False
) -> dict:

    from .tasks import CHURN_TASKS, COUNT_WINDOWS, USR_CATBOOST, churn_rows, churn_thresholds, table, usr_catboost

    used = groups(final_test)

    # Оцениваются все группы, кроме train: val, а в финальной оценке
    # и test.
    evaluated = used[1:]

    vectors = {"model": load_embeddings(tag, used)}

    if control:
        vectors[control] = load_embeddings(control, used)

    tables = {group: table(group) for group in used}
    churn = {task: {group: churn_rows(task, group) for group in used} for task in CHURN_TASKS}
    thresholds = churn_thresholds()

    # Готовые прогнозы CatBoost на [USR]: по векторам модели и, с
    # --control, по векторам контроля (набор с приставкой контроля).
    ready = {}

    for prefix, owner in [("", tag)] + ([(f"{control}:", control)] if control else []):
        for name in USR_CATBOOST:
            found = usr_catboost(name, owner, used, vectors_meta(owner))
            if found is not None:
                ready[prefix + name] = found

    # Признаки-счётчики — по train: тип, которого в train нет, голова
    # не выучит. В других группах недостающий тип — нули.
    windows = tuple(f"n_{days}d_" for days in COUNT_WINDOWS)
    counts = [name for name in tables["train"].columns if name.startswith(windows)]

    for group in evaluated:
        missing = [name for name in counts if name not in tables[group]]
        zeros = pd.DataFrame(0, index=tables[group].index, columns=missing)
        tables[group] = pd.concat([tables[group], zeros], axis=1)

    names = [
        "recency", "counts", "usr", "usr+recency", "usr+last_event", "readouts", "readouts+recency", "counts+usr",
    ]

    if control:
        names += [f"{control}:usr", f"{control}:readouts+recency"]

    report: dict = {
        "tag": tag, "control": control, "baseline": baseline, "draws": draws,
        "final_test": final_test, "groups": list(used), "plus_usr": "catboost_plus_usr" in ready, "tasks": {},
    }

    # Прогнозы по строкам: по ним сравниваются модели между собой.
    predictions: list[pd.DataFrame] = []

    before = (
        pd.read_parquet(downstream_dir(baseline) / PREDICTIONS_FILE) if baseline else None
    )

    for task in TASKS:

        rows = task_rows(task, tables, churn)

        # Строки без вектора — ошибка, а не пропуск: сравнение с
        # бейзлайном обязано идти на тех же клиентах.
        for tag_name, embedded in vectors.items():
            for group, frame in rows.items():
                missing = frame.index.difference(embedded[group].index)
                if len(missing):
                    raise ValueError(
                        f"{task}/{group}: у {len(missing)} строк нет вектора ({tag_name}), "
                        f"например {missing[0]}"
                    )

        y = {group: rows[group]["y"].to_numpy() for group in used}

        scores: dict[str, dict[str, np.ndarray]] = {}
        chosen: dict[str, float] = {}
        cut: dict[str, float] = {"catboost": thresholds[task]}

        for name in names:

            matrices = {
                group: features(name if ":" in name else f"model:{name}", group, rows[group], vectors, counts)
                for group in used
            }

            predicted, chosen[name] = fit_predict(
                matrices["train"], y["train"], [matrices[group] for group in evaluated], seed
            )

            scores[name] = dict(zip(evaluated, predicted))
            cut[name] = oof_threshold(matrices["train"], y["train"], chosen[name], seed)

        scores["catboost"] = {group: rows[group]["catboost"].to_numpy() for group in evaluated}

        for name, found in ready.items():
            for group in used:
                other = found["rows"].get((task, group))
                if other is None or not other.index.equals(rows[group].index) or not np.array_equal(
                    other["churn"].to_numpy(), y[group]
                ):
                    raise ValueError(f"{name}/{task}/{group}: другие клиенты или метки, чем у catboost")
            scores[name] = {group: found["rows"][(task, group)]["score"].to_numpy() for group in evaluated}
            cut[name] = found["thresholds"][task]

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
            results[name]["C"] = chosen.get(name)
            results[name]["threshold"] = cut[name]
            if name != reference:
                results[name]["vs_reference"] = {
                    group: paired(y[group], by_group[group], scores[reference][group], draws, seed)
                    for group in evaluated
                }
            # Что дало обучение: та же голова на векторах контроля.
            if control and f"{control}:{name}" in scores:
                results[name]["vs_control"] = {
                    group: paired(y[group], by_group[group], scores[f"{control}:{name}"][group], draws, seed)
                    for group in evaluated
                }
            # Что даёт нелинейная голова: линейная на тех же векторах.
            prefix, _, kind = name.rpartition(":")
            if kind in HEADS:
                linear = f"{prefix}:{HEADS[kind]}" if prefix else HEADS[kind]
                results[name]["vs_lr"] = {
                    group: paired(y[group], by_group[group], scores[linear][group], draws, seed)
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


# Главное сравнение: CatBoost на полном X, только [USR] и полный X
# вместе с [USR].
MAIN = ("catboost", "usr", "catboost_plus_usr")


def main_comparison(block: dict) -> list[str]:
    """
    Компактная таблица задачи на val и разница catboost_plus_usr с
    catboost: даёт ли [USR] сигнал сверх handcrafted-признаков.
    """

    lines = ["  главное сравнение на val:", f"  {'модель':<20} {'PR-AUC':>7} {'ROC-AUC':>8} {'F1':>6}"]

    for name in MAIN:
        if name in block["results"]:
            result = block["results"][name]["val"]
            lines.append(f"  {name:<20} {result['pr_auc']:7.3f} {result['roc_auc']:8.3f} {result['f1']:6.3f}")

    plus = block["results"].get("catboost_plus_usr")

    if plus is None:
        lines.append("  catboost_plus_usr нет: cd churn_baseline && python -m churn.plus_usr --embeddings <векторы>")
        return lines

    pr, roc = plus["vs_reference"]["val"]["pr_auc"], plus["vs_reference"]["val"]["roc_auc"]

    if pr["low"] > 0:
        verdict = "интервал выше нуля: [USR] даёт сигнал сверх handcrafted-признаков"
    elif pr["high"] < 0:
        verdict = "интервал ниже нуля: добавление [USR] ухудшает модель"
    else:
        verdict = "интервал захватывает ноль: [USR] почти ничего не добавляет"

    lines.append(
        f"  Δ catboost_plus_usr − catboost: PR-AUC {pr['mean']:+.3f} [{pr['low']:+.3f}, {pr['high']:+.3f}], "
        f"ROC-AUC {roc['mean']:+.3f} [{roc['low']:+.3f}, {roc['high']:+.3f}] — {verdict}"
    )

    return lines


def usr_diagnostic(block: dict, tag: str, control: str | None) -> list[str]:
    """
    2×2 на val: векторы модели и контроля × регрессия и CatBoost на
    одном [USR], и парные разницы, какие посчитаны.
    """

    results = block["results"]

    cells = [
        (f"{control} USR + LR", f"{control}:usr"),
        (f"{tag} USR + LR", "usr"),
        (f"{tag} USR + CatBoost", "catboost_usr"),
        (f"{control} USR + CatBoost", f"{control}:catboost_usr"),
    ]
    cells = [(label, name) for label, name in cells if name in results]

    if len(cells) < 2:
        return []

    lines = ["  диагностика [USR] на val:", f"  {'модель':<32} {'PR-AUC':>7} {'ROC-AUC':>8} {'F1':>6}"]

    for label, name in cells:
        result = results[name]["val"]
        lines.append(f"  {label:<32} {result['pr_auc']:7.3f} {result['roc_auc']:8.3f} {result['f1']:6.3f}")

    deltas = [
        (f"{tag} LR − {control} LR", "usr", "vs_control"),
        (f"{tag} CatBoost − {tag} LR", "catboost_usr", "vs_lr"),
        (f"{tag} CatBoost − {control} CatBoost", "catboost_usr", "vs_control"),
        (f"{control} CatBoost − {control} LR", f"{control}:catboost_usr", "vs_lr"),
    ]

    for label, name, key in deltas:
        delta = results.get(name, {}).get(key)
        if delta:
            pr, roc = delta["val"]["pr_auc"], delta["val"]["roc_auc"]
            lines.append(
                f"  {label + ':':<40} Δ PR-AUC {pr['mean']:+.3f} [{pr['low']:+.3f}, {pr['high']:+.3f}], "
                f"Δ ROC-AUC {roc['mean']:+.3f} [{roc['low']:+.3f}, {roc['high']:+.3f}]"
            )

    return lines


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

        lines += main_comparison(block)
        lines += usr_diagnostic(block, report["tag"], report["control"])

        lines.append(
            f"  {'набор':<24}" + "".join(f" {group + ' ROC':>9} {group + ' PR':>8}" for group in evaluated)
            + "".join(f"   Δ {group} PR к эталону [95%]" for group in evaluated)
        )

        for name, result in block["results"].items():

            cells = "".join(f" {result[group]['roc_auc']:9.3f} {result[group]['pr_auc']:8.3f}" for group in evaluated)

            delta = result.get("vs_reference")
            shown = "   ".join(interval(delta[group]["pr_auc"]) for group in evaluated) if delta else "эталон"

            versus = result.get("vs_baseline")
            if versus:
                shown += "".join(
                    f"   Δ к {report['baseline']} ({group}): PR {interval(versus[group]['pr_auc'])}, "
                    f"ROC {versus[group]['roc_auc']['mean']:+.3f}"
                    for group in evaluated
                )

            lines.append(f"  {name:<24}{cells}   {shown}")

    return "\n".join(lines)


def run(args) -> int:

    try:
        with threadpool_limits(BLAS_THREADS):
            report = run_probe(args.tag, args.control, args.draws, args.seed, args.baseline, args.final_test)
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
    parser.add_argument("--control", default=None, help="тег векторов-контроля, например init")
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


__all__ = ["COMPARED", "HEADS", "MAIN", "REFERENCE", "TASKS", "at_threshold", "features", "fit_predict", "metrics",
           "oof_threshold", "paired", "probe_model", "run_probe", "show", "threshold_max_f1", "usr_diagnostic",
           "vectors_meta", "vs_baseline"]
