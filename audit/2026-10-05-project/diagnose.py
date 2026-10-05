"""
Диагностика аудита 2026-10-05: почему [USR] → LR проигрывает CatBoost на
churn_active90.

Только читает и печатает; ничего не пишет. Запуск из корня репозитория:

    python audit/2026-10-05-project/diagnose.py --tag w4-b0

Читает строки и признаки churn_baseline (train, val), векторы
data/13_downstream/<тег>, ленту data/02_preprocessed/<группа> и манифесты
data/01_raw/<группа>. Скрытый план пауз не выгружается генератором: он
восстанавливается в памяти теми же draw_persona и plan_pauses по seed группы
из манифеста. Это оценка потолка задачи, во вход модели он не идёт.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.generator import config, emit
from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.life import events as life_events
from src.generator.life import lifecycle
from src.generator.life.persona import draw_persona
from src.generator.world import communities


REPO = Path(__file__).resolve().parents[2]
CHURN = REPO / "churn_baseline"
LOCAL = timedelta(hours=5)
HORIZON = timedelta(days=60)

# Давность и счётчики действий клиента — признаки churn_baseline.
RECENCY = [
    "act_days_since_last", "act_days_7", "act_count_7", "act_days_30", "act_count_30",
    "act_count_90", "act_gap_max_90", "act_gap_mean_90", "purchase_days_since_last",
    "app_days_since_last", "act_count_all", "transfer_days_since_last", "cash_out_days_since_last",
]

# Виды пауз, при которых молчат все потоки клиента.
SILENT_KINDS = ("full", "seasonal")


def load_activity():
    """
    Определение действия клиента — то же, что у метки churn_baseline.
    """
    spec = importlib.util.spec_from_file_location("activity", CHURN / "churn" / "activity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(group: str, tag: str) -> pd.DataFrame:
    """
    Строки задачи группы: метка и прогноз CatBoost, признаки, векторы на T.
    """
    name = "train_rows.parquet" if group == "train" else "eval_rows.parquet"
    frame = pd.read_parquet(CHURN / "reports" / name)
    frame = frame[frame["group"] == group][["client_id", "T", "churn", "score"]]
    features = pd.read_parquet(CHURN / "data" / group / "features.parquet")
    features = features.drop(columns=["churn", "group", "T", "active90"])
    vectors = pd.read_parquet(REPO / "data" / "13_downstream" / tag / f"{group}.parquet")
    return frame.merge(features, on="client_id").merge(vectors, on="client_id")


def matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.vstack(frame[column].to_numpy())


def log_columns(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.log1p(frame[columns].fillna(-1).clip(lower=0).to_numpy(dtype=float))


def logistic(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> np.ndarray:
    """
    Голова пробы: стандартизация и логистическая регрессия, C по log-loss
    на трёх фолдах train.
    """
    model = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {"logisticregression__C": np.logspace(-4, 2, 7)},
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        scoring="neg_log_loss",
    )
    return model.fit(train_x, train_y).predict_proba(val_x)[:, 1]


def boosting(train: pd.DataFrame, val: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """
    Градиентный бустинг sklearn, среднее пяти seed: CatBoost здесь не нужен,
    сравниваются входы, а не головы.
    """
    categorical = [column.endswith("_code") for column in columns]
    scores = []
    for seed in range(5):
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=seed, categorical_features=categorical,
        )
        model.fit(train[columns].astype(float), train["churn"])
        scores.append(model.predict_proba(val[columns].astype(float))[:, 1])
    return np.mean(scores, axis=0)


def line(name: str, y: np.ndarray, score: np.ndarray) -> None:
    print(f"  {name:66s} PR-AUC {average_precision_score(y, score):.3f}  ROC-AUC {roc_auc_score(y, score):.3f}")


def manifest(group: str) -> dict:
    return json.loads((REPO / "data" / "01_raw" / group / "manifest.json").read_text())


def local(moment: str) -> datetime:
    return datetime.fromisoformat(moment).replace(tzinfo=None)


def activate(group: str) -> int:
    """
    Генератор в состоянии выгрузки группы: окно, параметры и seed манифеста.
    """
    meta = manifest(group)
    config.activate_horizon(local(meta["period_start"]), local(meta["period_end"]), local(meta["registration_end"]))
    settings = emit._build_params(None, None, int(meta["community_size"]))
    if settings.fingerprint() != meta["generation_config_sha256"]:
        raise SystemExit(f"{group}: параметры генератора не те, что у выгрузки — план пауз не восстановить")
    params_module.activate(settings)
    rng_module.configure(int(meta["seed"]), settings.fingerprint(), int(meta["world_seed"]))
    return int(meta["total_clients"])


def pause_plans(group: str) -> dict[str, tuple]:
    """
    Скрытый план пауз каждого клиента группы: client_id → (persona, паузы).
    """
    total = activate(group)
    plans = {}
    for ordinal in range(1, total + 1):
        persona = draw_persona(ordinal)
        plans[communities.client_id(ordinal)] = (persona, lifecycle.plan_pauses(persona, life_events.plan_events(persona)))
    return plans


def hidden_state(plans: dict[str, tuple], cutoff: datetime) -> pd.DataFrame:
    """
    Скрытое состояние на T. Прошлое: режим, идёт ли пауза, её вид и возраст.
    Будущее: сколько паузе осталось, начнётся ли новая в окне метки.
    """
    rows = []
    end = cutoff + HORIZON
    for client_id, (persona, pauses) in plans.items():
        current = lifecycle.pause_at(pauses, cutoff)
        upcoming = [pause for pause in pauses if cutoff < pause.start <= end]
        covered = sum(
            max(0.0, (min(pause.actual_end, end) - max(pause.start, cutoff)).total_seconds())
            for pause in pauses if pause.kind in SILENT_KINDS
        )
        rows.append({
            "client_id": client_id,
            "mode": persona.activity_mode,
            "vanished": bool(persona.vanished_after_registration),
            "in_pause": current is not None,
            "pause_kind": current.kind if current else "none",
            "pause_age": (cutoff - current.start).days if current else -1,
            "pause_left": (current.actual_end - cutoff).days if current else -1,
            "next_kind": upcoming[0].kind if upcoming else "none",
            "next_in": (upcoming[0].start - cutoff).days if upcoming else -1,
            "silent_cover": covered / HORIZON.total_seconds(),
        })
    return pd.DataFrame(rows)


def encode(frames: list[pd.DataFrame], columns: list[str]) -> None:
    for column in columns:
        values = sorted(set().union(*(set(frame[column]) for frame in frames)))
        for frame in frames:
            frame[f"{column}_code"] = pd.Categorical(frame[column], categories=values).codes


def bootstrap_delta(y: np.ndarray, new: np.ndarray, old: np.ndarray, draws: int = 2000) -> np.ndarray:
    generator = np.random.default_rng(0)
    deltas = []
    for _ in range(draws):
        index = generator.integers(0, len(y), len(y))
        if y[index].sum() == 0:
            continue
        deltas.append(average_precision_score(y[index], new[index]) - average_precision_score(y[index], old[index]))
    return np.percentile(deltas, [2.5, 50, 97.5])


def missed_expected(frame: pd.DataFrame) -> list[str]:
    """
    «Пропущено ожидаемых действий» по потоку: давность × личная частота за 90 дней.
    """
    pairs = [
        ("act", "act_days_since_last", "act_days_90"),
        ("purchase", "purchase_days_since_last", "purchase_count_90"),
        ("app", "app_days_since_last", "app_sessions_90"),
        ("transfer", "transfer_days_since_last", "transfer_count_90"),
        ("cash", "cash_out_days_since_last", "cash_out_count_90"),
    ]
    names = []
    for name, since, count in pairs:
        frame[f"missed_{name}"] = frame[since].fillna(999) * frame[count].fillna(0) / 90.0
        names.append(f"missed_{name}")
    frame["silent_ratio"] = frame["act_days_since_last"] / frame["act_gap_mean_90"].replace(0, np.nan)
    return names + ["silent_ratio"]


def actions_in_silent_pauses(plans: dict[str, tuple], activity) -> None:
    """
    Действия клиента (по определению метки) внутри полных пауз val.
    """
    columns = ["client_id", "event_time", "type", "reason", "channel", "direction", "counterparty",
               "migration_reason", "change_source"]
    events = pd.read_parquet(REPO / "data" / "02_preprocessed" / "val" / "events.parquet", columns=columns)
    events["local"] = events["event_time"].dt.tz_convert(None) + LOCAL
    events = events[activity.is_client_action(events)]
    end = pd.Timestamp(local(manifest("val")["period_end"]))
    by_client = dict(tuple(events.groupby("client_id")))

    days = 0
    inside = []
    for client_id, (_, pauses) in plans.items():
        mine = by_client.get(client_id)
        for pause in pauses:
            if pause.kind != "full":
                continue
            start, finish = pd.Timestamp(pause.start), min(pd.Timestamp(pause.actual_end), end)
            if finish <= start:
                continue
            days += (finish - start).days
            if mine is not None:
                inside.append(mine[(mine["local"] >= start) & (mine["local"] < finish)])

    found = pd.concat(inside)
    print(f"  клиент-дней в полных паузах: {days}, действий клиента внутри: {len(found)}")
    top = found.groupby(["type", "reason"], dropna=False).size().sort_values(ascending=False).head(8)
    for (kind, reason), count in top.items():
        print(f"    {kind:20s} {str(reason):16s} {count}")


def phase_of_cutoff(group: str, cutoff: pd.Timestamp) -> None:
    """
    Что лежит перед T: тип и давность последнего события клиента.
    """
    events = pd.read_parquet(REPO / "data" / "02_preprocessed" / group / "events.parquet", columns=["client_id", "event_time", "type"])
    events = events[events["event_time"] < cutoff]
    last = events.groupby("client_id").tail(1)
    minutes = (cutoff - last["event_time"]).dt.total_seconds() / 60
    kinds = last["type"].value_counts(normalize=True).head(3).round(3).to_dict()
    print(f"  {group}: T {cutoff.isoformat()}; до последнего события, мин: p10 {minutes.quantile(0.1):.0f}, "
          f"p50 {minutes.quantile(0.5):.0f}, p90 {minutes.quantile(0.9):.0f}; его тип: {kinds}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python audit/2026-10-05-project/diagnose.py")
    parser.add_argument("--tag", default="w4-b0", help="векторы data/13_downstream/<тег>")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    train, val = rows("train", args.tag), rows("val", args.tag)
    y_train, y_val = train["churn"].to_numpy(), val["churn"].to_numpy()
    print(f"churn_active90: train {len(train)} строк ({y_train.sum()} ушли), val {len(val)} ({y_val.sum()} ушли)\n")

    print("1. Вход → голова, val")
    line("116 признаков → CatBoost (эталон churn_baseline)", y_val, val["score"].to_numpy())
    line("13 признаков давности и счётчиков → HGB", y_val, boosting(train, val, RECENCY))
    one = ["act_days_since_last"]
    line("log act_days_since_last → LR", y_val, logistic(log_columns(train, one), y_train, log_columns(val, one)))

    def gap(frame: pd.DataFrame) -> np.ndarray:
        return np.c_[np.log1p(frame["gap_seconds"].to_numpy() / 86400), np.log1p(frame["n_events"].to_numpy())]

    line("log gap_seconds + log n_events (из embed) → LR", y_val, logistic(gap(train), y_train, gap(val)))
    line("[USR] → LR", y_val, logistic(matrix(train, "usr"), y_train, matrix(val, "usr")))
    line("[USR] + log act_days_since_last → LR", y_val, logistic(
        np.c_[matrix(train, "usr"), log_columns(train, one)], y_train, np.c_[matrix(val, "usr"), log_columns(val, one)]))
    line("выход энкодера анкеты → LR", y_val, logistic(matrix(train, "profile"), y_train, matrix(val, "profile")))

    print("\n2. Что знает [USR]: R² Ridge, учится на train, проверка на val")
    targets = {
        "log давность до последнего события": lambda frame: np.log1p(frame["gap_seconds"].to_numpy() / 86400),
        "log n_events": lambda frame: np.log1p(frame["n_events"].to_numpy()),
    }
    for column in ["act_days_since_last", "act_gap_max_90", "act_count_7", "act_count_90"]:
        targets[f"log1p {column}"] = lambda frame, column=column: np.log1p(frame[column].fillna(0).clip(lower=0).to_numpy(dtype=float))
    for name, target in targets.items():
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(matrix(train, "usr"), target(train))
        print(f"  {name:40s} R² {r2_score(target(val), model.predict(matrix(val, 'usr'))):.3f}")
    profile_fit = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(matrix(train, "profile"), matrix(train, "usr"))
    explained = r2_score(matrix(val, "usr"), profile_fit.predict(matrix(val, "profile")), multioutput="uniform_average")
    print(f"  [USR] из выхода энкодера анкеты, средний R² по координатам: {explained:.3f}")

    print("\n3. Потолок задачи: скрытый план пауз генератора")
    plans = {group: pause_plans(group) for group in ("train", "val")}
    cutoffs = {
        group: frame["T"].iloc[0].tz_convert(None).to_pydatetime() + LOCAL
        for group, frame in (("train", train), ("val", val))
    }
    states = {group: hidden_state(plans[group], cutoffs[group]) for group in plans}
    train, val = train.merge(states["train"], on="client_id"), val.merge(states["val"], on="client_id")
    encode([train, val], ["mode", "pause_kind", "next_kind"])
    numeric = [
        column for column in pd.read_parquet(CHURN / "data" / "train" / "features.parquet").columns
        if column not in ("client_id", "group", "T", "churn", "active90") and train[column].dtype.kind in "fiub"
    ]
    past = ["mode_code", "vanished", "in_pause", "pause_kind_code", "pause_age"]
    future = ["pause_left", "next_kind_code", "next_in", "silent_cover"]
    base = boosting(train, val, numeric)
    line("числовые признаки churn_baseline → HGB", y_val, base)
    line("только режим активности (скрытый) → HGB", y_val, boosting(train, val, ["mode_code"]))
    line("признаки + скрытое состояние на T (только прошлое) → HGB", y_val, boosting(train, val, numeric + past))
    line("признаки + весь план пауз, включая будущее → HGB", y_val, boosting(train, val, numeric + past + future))
    engineered = missed_expected(train)
    missed_expected(val)
    richer = boosting(train, val, numeric + engineered)
    line("признаки + «пропущено ожидаемых» по потокам → HGB", y_val, richer)
    low, middle, high = bootstrap_delta(y_val, richer, base)
    print(f"    Δ PR-AUC к признакам: {middle:+.3f} [{low:+.3f}, {high:+.3f}]")
    in_pause = val["in_pause"].to_numpy()
    print(f"  ушедших val не в паузе на T: {int(y_val[~in_pause].sum())} из {int(y_val.sum())}")

    print("\n4. Полная пауза и метка: действия клиента внутри пауз val")
    actions_in_silent_pauses(plans["val"], load_activity())

    print("\n5. Фаза T: что модель видит последним")
    for group, frame in (("train", train), ("val", val)):
        phase_of_cutoff(group, frame["T"].iloc[0])


if __name__ == "__main__":
    main()
