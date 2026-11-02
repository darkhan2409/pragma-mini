"""
Диагностика аудита 2026-11-02: из чего складывается разрыв [USR] → LR и
CatBoost на churn_active90 и где стоит бейзлайн pragmatiq из счётчиков.

Только читает и печатает; ничего не пишет и test не читает. CatBoost нужен из
окружения churn_baseline, поэтому запуск из корня репозитория так:

    churn_baseline/.venv/bin/python audit/2026-11-02-usr-gap/diagnose.py --tag ref-2026-10-09

Читает строки и признаки churn_baseline (train, val), прогноз эталонного
CatBoost (reports/eval_rows.parquet), важности его признаков, векторы
data/13_downstream/<тег> и ленту data/02_preprocessed/<группа>. CatBoost
учится тем же churn.train.fit_task, что бейзлайн; LR — той же pipeline, что
src.downstream.probe. Сравнения парные: bootstrap по клиентам val, как
probe.paired (seed 0, 1000 выборок). Полный прогон — около 15 минут на CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "churn_baseline"))

from catboost import CatBoostClassifier, Pool  # noqa: E402
from churn.build import KEYS  # noqa: E402
from churn.profile import CATEGORICAL  # noqa: E402
from churn.train import EARLY_STOPPING, PARAMS, fit_task  # noqa: E402

GROUPS = ("train", "val")
CS = np.logspace(-4, 2, 7)  # сетка C — как в probe
ALPHAS = np.logspace(-2, 4, 13)
DRAWS = 1000

# Цели E1′ — как RecentTypes (src/mlm/model.py): окна и предел давности.
RECENT_DAYS = (7, 30, 90)
RECENCY_CAP_DAYS = 365.0
DAY_US = 86_400 * 1_000_000

# Группы handcrafted-признаков: по смыслу драйвера ухода, а не по источнику.
PROFILE_NUMERIC = {"declared_income", "income_day", "children", "age", "job_tenure_months"}
FRICTION = {"purchase_declined_share_90", "app_login_failed_30", "application_rejected_90", "insufficient_funds_90",
            "card_blocked_90", "support_case_count_90", "fee_count_90", "installment_missed_90", "delinquency_90",
            "loan_dpd_max_90", "fraud_alert_count_90"}
MONEY_EXTRA = {"purchase_mean_90", "balance_last", "balance_30d_ago", "balance_change_30"}
ATTACHMENTS = {"salary_count_90", "salary_days_since_last", "pension_count_90", "bill_auto_count_90",
               "subscription_count_90", "loan_pay_auto_count_90", "loan_pay_manual_count_90", "insurance_premium_count",
               "deposit_contracts_90", "loan_contracts_90", "products_nunique_90", "contracts_nunique_90",
               "cards_nunique_90", "transfer_in_count_90", "product_opened_90", "product_closed_90",
               "application_count_90"}
TYPE_ACTIVITY = {"purchase_count_7", "purchase_count_30", "purchase_count_90", "purchase_days_since_last",
                 "purchase_online_share_90", "purchase_mcc_nunique_90", "purchase_merchant_nunique_90",
                 "transfer_count_7", "transfer_count_30", "transfer_count_90", "transfer_days_since_last",
                 "cash_out_count_7", "cash_out_count_30", "cash_out_count_90", "cash_out_days_since_last",
                 "cash_in_count_7", "cash_in_count_30", "cash_in_count_90", "cash_in_days_since_last",
                 "bill_manual_count_90", "events_count_90", "bank_events_count_30"}

# Предупреждения о сходимости LR на сетке C — шум; процессы GridSearchCV
# берут настройку из окружения.
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")


# ------------------------------------------------------------ данные


def load(tag: str) -> dict:
    """Строки задачи, эталонный прогноз CatBoost и векторы на T."""

    features = {g: pd.read_parquet(ROOT / f"churn_baseline/data/{g}/features.parquet") for g in GROUPS}
    rows = {g: frame[frame["active90"]].set_index("client_id") for g, frame in features.items()}
    columns = [c for c in features["train"].columns if c not in KEYS]

    ref = pd.read_parquet(ROOT / "churn_baseline/reports/eval_rows.parquet")
    ref = ref[ref["task"] == "churn_active90"].set_index("client_id").loc[rows["val"].index]
    y = {g: rows[g]["churn"].astype(int).to_numpy() for g in GROUPS}
    if not (ref["churn"].astype(int).to_numpy() == y["val"]).all():
        raise SystemExit("метки eval_rows не совпадают со строками val: отчёт churn_baseline устарел")

    vectors = {g: pd.read_parquet(ROOT / f"data/13_downstream/{tag}/{g}.parquet").set_index("client_id").loc[rows[g].index]
               for g in GROUPS}
    return {"rows": rows, "columns": columns, "categorical": [c for c in columns if c in CATEGORICAL],
            "y": y, "ref": ref["score"].to_numpy(), "vectors": vectors}


def stack(frame: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack(frame[name].to_numpy()).astype(np.float64)


def oracle_targets(rows: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Точные цели E1′ на T по всей ленте 02_preprocessed: log1p числа событий
    каждого типа за 7/30/90 дней до T и log1p давности последнего (не больше
    365 дней). Типы — только из train. Модель видит отобранную историю, а
    здесь — всю: это потолок того, что цель просит хранить.
    """

    types: set[str] = set()
    events = pq.ParquetFile(ROOT / "data/02_preprocessed/train/events.parquet")
    for i in range(events.metadata.num_row_groups):
        types |= set(pc.unique(events.read_row_group(i, columns=["type"])["type"]).to_pylist())
    kinds = sorted(types)

    out = {}
    for g in GROUPS:
        clients = rows[g].index
        moments = pd.to_datetime(rows[g]["T"], utc=True).unique()
        if len(moments) != 1:
            raise SystemExit(f"у строк {g} не один T: {moments}")
        t_us = pd.Timestamp(moments[0]).value // 1000
        n_c, n_t = len(clients), len(kinds)
        counts = np.zeros((len(RECENT_DAYS), n_c * n_t), dtype=np.int64)
        recency = np.full(n_c * n_t, RECENCY_CAP_DAYS)
        client_set, type_set = pa.array(clients.to_numpy()), pa.array(kinds)

        events = pq.ParquetFile(ROOT / f"data/02_preprocessed/{g}/events.parquet")
        for i in range(events.metadata.num_row_groups):
            table = events.read_row_group(i, columns=["client_id", "event_time", "type"])
            moment = table["event_time"].cast(pa.int64())
            table = table.filter(pc.and_(pc.less(moment, t_us), pc.greater_equal(moment, t_us - int(RECENCY_CAP_DAYS * DAY_US))))
            ci = pc.index_in(table["client_id"], value_set=client_set)
            ti = pc.index_in(table["type"], value_set=type_set)
            valid = pc.and_(pc.is_valid(ci), pc.is_valid(ti))
            table, ci, ti = table.filter(valid), ci.filter(valid), ti.filter(valid)
            if table.num_rows == 0:
                continue
            days = (t_us - table["event_time"].cast(pa.int64()).to_numpy()) / DAY_US
            flat = ci.to_numpy().astype(np.int64) * n_t + ti.to_numpy().astype(np.int64)
            for w, window in enumerate(RECENT_DAYS):
                inside = days <= window
                counts[w] += np.bincount(flat[inside], minlength=n_c * n_t)
            latest = pd.Series(days).groupby(flat).min()
            recency[latest.index] = np.minimum(recency[latest.index], latest.to_numpy())

        columns: dict[str, np.ndarray] = {}
        for w, window in enumerate(RECENT_DAYS):
            block = np.log1p(counts[w].reshape(n_c, n_t))
            columns.update({f"n{window}_{name}": block[:, k] for k, name in enumerate(kinds)})
        block = np.log1p(recency.reshape(n_c, n_t))
        columns.update({f"rec_{name}": block[:, k] for k, name in enumerate(kinds)})
        out[g] = pd.DataFrame(columns, index=clients)
    return out


def group_of(column: str, categorical: list[str]) -> str:
    if column in categorical or column in PROFILE_NUMERIC:
        return "profile"
    if column.startswith("days_since_") or column == "act_days_since_first":
        return "tenure"
    if column.startswith("act_"):
        return "act"
    if column in FRICTION:
        return "friction"
    if column.startswith("app_"):
        return "app"
    if column.startswith("comm_") or column in {"consent_withdrawn", "banner_click_count_90"}:
        return "comm"
    if "_sum_" in column or column in MONEY_EXTRA:
        return "money"
    if column in ATTACHMENTS:
        return "attachments"
    if column in TYPE_ACTIVITY:
        return "type_activity"
    raise SystemExit(f"признак без группы: {column}")


# ------------------------------------------------------------ головы и сравнение


class Bench:
    """Головы на train, оценка на val и парный bootstrap к выбранным базам."""

    def __init__(self, data: dict) -> None:
        self.rows, self.y = data["rows"], data["y"]
        self.columns, self.categorical = data["columns"], data["categorical"]
        self.numeric = [c for c in self.columns if c not in self.categorical]
        self.yv = self.y["val"]
        rng = np.random.default_rng(0)
        draws = (rng.integers(0, len(self.yv), len(self.yv)) for _ in range(DRAWS))
        self.draws = [t for t in draws if 0 < self.yv[t].sum() < len(t)]
        self.base: dict[str, np.ndarray] = {"catboost": self.draw_ap(data["ref"])}
        self.results: dict[str, dict] = {}

    def draw_ap(self, score: np.ndarray) -> np.ndarray:
        return np.array([average_precision_score(self.yv[t], score[t]) for t in self.draws])

    def record(self, name: str, score: np.ndarray, info: str = "", vs: tuple[str, ...] = ("catboost",)) -> None:
        aps = self.draw_ap(score)
        run = {"pr_auc": average_precision_score(self.yv, score), "aps": aps, "score": score}
        parts = [f"PR-AUC {run['pr_auc']:.3f}", f"ROC {roc_auc_score(self.yv, score):.3f}",
                 f"LL {log_loss(self.yv, score):.4f}"]
        for base in vs:
            d = aps - self.base[base]
            parts.append(f"Δ к {base} {d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
        self.results[name] = run
        print(f"  {name:<44} " + " | ".join(parts) + (f"  ({info})" if info else ""), flush=True)

    def as_base(self, name: str, alias: str) -> None:
        """Прогон name становится базой парных сравнений под коротким именем alias."""
        self.base[alias] = self.results[name]["aps"]

    def hand(self, cols: list[str]) -> dict[str, np.ndarray]:
        """Handcrafted для LR: signed log1p, пропуск — медиана train и флаг, категории — one-hot по train."""

        def slog(frame: pd.DataFrame) -> pd.DataFrame:
            x = frame.astype(float)
            return np.sign(x) * np.log1p(np.abs(x))

        num = [c for c in cols if c in self.numeric]
        cat = [c for c in cols if c in self.categorical]
        train = slog(self.rows["train"][num])
        median = train.median()
        missing = [c for c in num if train[c].isna().any()]
        encoder = None
        if cat:
            encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)
            encoder.fit(self.rows["train"][cat].astype(str))
        out = {}
        for g in GROUPS:
            x = slog(self.rows[g][num])
            parts = [x.fillna(median).fillna(0.0).to_numpy(), x[missing].isna().to_numpy(float)]
            if encoder is not None:
                parts.append(encoder.transform(self.rows[g][cat].astype(str)))
            out[g] = np.hstack(parts)
        return out

    def lr(self, x: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        model = GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": CS},
            cv=StratifiedKFold(3, shuffle=True, random_state=0),
            scoring="neg_log_loss", refit=True, n_jobs=4,
        )
        model.fit(x["train"], self.y["train"])
        return model.predict_proba(x["val"])[:, 1], float(model.best_params_["logisticregression__C"])

    def cb(self, frames: dict[str, pd.DataFrame], cols: list[str]) -> tuple[np.ndarray, int]:
        cats = [c for c in cols if c in self.categorical]
        model, info = fit_task(frames["train"].assign(churn=self.y["train"]), cols, cats)
        return model.predict_proba(Pool(frames["val"][cols], cat_features=cats))[:, 1], int(info["trees"])

    def frames(self, x: dict[str, np.ndarray], prefix: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
        names = [f"{prefix}{i}" for i in range(x["train"].shape[1])]
        return {g: pd.DataFrame(x[g], columns=names, index=self.rows[g].index) for g in GROUPS}, names

    def run_lr(self, name: str, x: dict[str, np.ndarray], vs: tuple[str, ...] = ("catboost",)) -> float:
        score, c = self.lr(x)
        self.record(name, score, f"C={c:g}", vs)
        return c

    def run_cb(self, name: str, x: dict[str, np.ndarray] | dict[str, pd.DataFrame], cols: list[str] | None = None,
               vs: tuple[str, ...] = ("catboost",)) -> int:
        frames = x if cols is not None else None
        if cols is None:
            frames, cols = self.frames(x, "f")  # type: ignore[arg-type]
        score, trees = self.cb(frames, cols)  # type: ignore[arg-type]
        self.record(name, score, f"деревьев {trees}", vs)
        return trees


def r2_pooled(truth: pd.DataFrame, pred: np.ndarray) -> float:
    return float(1 - ((truth - pred) ** 2).to_numpy().sum() / ((truth - truth.mean()) ** 2).to_numpy().sum())


# ------------------------------------------------------------ разделы


def main() -> None:
    parser = argparse.ArgumentParser(prog="churn_baseline/.venv/bin/python audit/2026-11-02-usr-gap/diagnose.py")
    parser.add_argument("--tag", default="ref-2026-10-09", help="векторы data/13_downstream/<тег>")
    args = parser.parse_args()

    data = load(args.tag)
    bench = Bench(data)
    rows, y, vec = data["rows"], data["y"], data["vectors"]
    usr = {g: stack(vec[g], "usr") for g in GROUPS}
    scalars = {g: np.log1p(vec[g][["n_events", "gap_seconds"]].to_numpy(np.float64)) for g in GROUPS}
    readouts = {g: np.hstack([stack(vec[g], n) for n in ("usr", "profile", "mean_event", "last_event")] + [scalars[g]])
                for g in GROUPS}
    oracle = oracle_targets(rows)
    ora = {g: oracle[g].to_numpy() for g in GROUPS}

    print(f"== 0. Строки и контроль (векторы {args.tag})")
    print(f"  строк train {len(y['train'])} (ушедших {y['train'].sum()}), val {len(y['val'])} (ушедших {y['val'].sum()})")
    total = np.expm1(oracle["val"].filter(like="n90_")).sum(axis=1).round()
    print(f"  оракул E1′: сумма событий за 90 дней = events_count_90 бейзлайна у доли val "
          f"{float((total == rows['val']['events_count_90']).mean()):.3f}; типов {oracle['val'].shape[1] // 4}")
    trees_hand = bench.run_cb("CatBoost × handcrafted (контроль 0.570)", {g: rows[g] for g in GROUPS}, data["columns"])
    c_usr = bench.run_lr("LR × [USR] (контроль 0.452)", usr)
    bench.as_base("LR × [USR] (контроль 0.452)", "lr_usr")

    print("== 1. Голова × вход")
    hand = bench.hand(data["columns"])
    c_hand = bench.run_lr("LR × handcrafted", hand)
    trees_usr = bench.run_cb("CatBoost × [USR]", usr, vs=("catboost", "lr_usr"))
    scaler = StandardScaler().fit(usr["train"])
    for hidden, alpha in (((64,), 1e-3), ((128, 64), 1e-3), ((64,), 1e-1)):
        probs = []
        for seed in range(5):
            mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=20, max_iter=500, random_state=seed)
            mlp.fit(scaler.transform(usr["train"]), y["train"])
            probs.append(mlp.predict_proba(scaler.transform(usr["val"]))[:, 1])
        bench.record(f"MLP {hidden} alpha={alpha:g} × [USR], 5 seed", np.mean(probs, axis=0), vs=("catboost", "lr_usr"))
    bench.run_lr("LR × все векторы модели", readouts, vs=("catboost", "lr_usr"))
    bench.run_cb("CatBoost × все векторы модели", readouts, vs=("catboost", "lr_usr"))

    print("== 2. Оракул E1′: что цель просит хранить и как [USR] это хранит")
    c_ora = bench.run_lr("LR × точные цели E1′", ora, vs=("catboost", "lr_usr"))
    bench.run_cb("CatBoost × точные цели E1′", {g: oracle[g] for g in GROUPS}, list(oracle["train"].columns),
                 vs=("catboost", "lr_usr"))
    bench.as_base("CatBoost × точные цели E1′", "cb_oracle")
    bench.run_lr("LR × [USR] + точные цели E1′", {g: np.hstack([usr[g], ora[g]]) for g in GROUPS}, vs=("catboost", "lr_usr"))
    keep = ((oracle["train"].std() > 0) & (oracle["val"].std() > 0)).to_numpy()
    ridge = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(scaler.transform(usr["train"]), oracle["train"].loc[:, keep])
    decoded = {g: pd.DataFrame(ridge.predict(scaler.transform(usr[g])), index=rows[g].index,
                               columns=oracle["train"].columns[keep]) for g in GROUPS}
    truth = oracle["val"].loc[:, keep]
    blocks = {b: r2_pooled(truth.filter(like=b), decoded["val"].filter(like=b).to_numpy()) for b in ("n7_", "n30_", "n90_", "rec_")}
    print(f"  R² целей E1′ из [USR] на val (ridge, train → val): всего {r2_pooled(truth, decoded['val'].to_numpy()):.3f}; "
          + ", ".join(f"{b.rstrip('_')} {v:.3f}" for b, v in blocks.items()))
    bench.run_cb("CatBoost × E1′, декодированные из [USR]", decoded, list(decoded["train"].columns),
                 vs=("catboost", "lr_usr", "cb_oracle"))
    spread = (truth - decoded["val"]).std().to_numpy()
    noise = np.random.default_rng(1)
    noisy = {g: oracle[g].loc[:, keep] + noise.normal(0.0, 1.0, oracle[g].loc[:, keep].shape) * spread for g in GROUPS}
    bench.run_cb("CatBoost × точные E1′ + шум ошибки декодирования", noisy, list(noisy["train"].columns),
                 vs=("catboost", "cb_oracle"))

    print("== 3. Что CatBoost берёт из векторов модели")
    bench.run_cb("CatBoost × [USR] + log n_events + log gap_seconds",
                 {g: np.hstack([usr[g], scalars[g]]) for g in GROUPS}, vs=("catboost", "lr_usr"))
    bench.run_cb("CatBoost × log n_events + log gap_seconds", scalars, vs=("catboost", "lr_usr"))
    for name in ("mean_event", "last_event", "profile"):
        bench.run_cb(f"CatBoost × {name}", {g: stack(vec[g], name) for g in GROUPS}, vs=("catboost", "lr_usr"))

    print("== 4. Группы handcrafted-признаков")
    groups: dict[str, list[str]] = {}
    for c in data["columns"]:
        groups.setdefault(group_of(c, data["categorical"]), []).append(c)
    print("  " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()))
    c_act = 1.0
    for name, cols in groups.items():
        part = bench.hand(cols)
        c = bench.run_lr(f"LR × {name}", part, vs=("catboost", "lr_usr"))
        c_act = c if name == "act" else c_act
        bench.run_lr(f"LR × [USR] + {name}", {g: np.hstack([usr[g], part[g]]) for g in GROUPS}, vs=("catboost", "lr_usr"))
    full = {g: rows[g] for g in GROUPS}
    activity = groups["act"] + groups["type_activity"] + groups["app"]
    bench.run_cb("CatBoost × только активность (act, типы, app)", full, activity)
    bench.run_cb("CatBoost × handcrafted без act", full, [c for c in data["columns"] if c not in groups["act"]])
    drop = ("money", "tenure", "attachments", "profile")
    bench.run_cb("CatBoost × без денег, стажа, привязок, анкеты", full,
                 [c for c in data["columns"] if group_of(c, data["categorical"]) not in drop])

    print("== 5. Линейные потолки сводки по типам")
    kinds = [c[4:] for c in oracle["train"].columns if c.startswith("rec_")]

    def linearized(frame: pd.DataFrame) -> np.ndarray:
        days = np.expm1(frame[[f"rec_{k}" for k in kinds]].to_numpy())
        silence = np.hstack([(days >= d).astype(float) for d in (3, 7, 14, 30, 60)])
        n30 = np.expm1(frame[[f"n30_{k}" for k in kinds]].to_numpy())
        n90 = np.expm1(frame[[f"n90_{k}" for k in kinds]].to_numpy())
        return np.hstack([frame.to_numpy(), silence, np.log((n30 + 1) / (n90 + 1))])

    bench.run_lr("LR × E1′ + пороги тишины + доли 30/90", {g: linearized(oracle[g]) for g in GROUPS}, vs=("catboost", "lr_usr"))
    splines = SplineTransformer(n_knots=5, degree=3).fit(ora["train"])
    bench.run_lr("LR × сплайны E1′", {g: splines.transform(ora[g]) for g in GROUPS}, vs=("catboost", "lr_usr"))

    print("== 6. Восстановимость топ-40 признаков CatBoost (R² на val, ridge train → val)")
    importance = pd.read_csv(ROOT / "churn_baseline/reports/feature_importance.csv").set_index("feature")["importance"]
    top = [c for c in importance.index if c in bench.numeric][:40]

    def recover(source: dict[str, np.ndarray]) -> dict[str, float]:
        sc = StandardScaler().fit(source["train"])
        a, b = sc.transform(source["train"]), sc.transform(source["val"])
        out = {}
        for c in top:
            ttr = rows["train"][c].astype(float)
            tva = rows["val"][c].astype(float)
            ttr, tva = np.sign(ttr) * np.log1p(ttr.abs()), np.sign(tva) * np.log1p(tva.abs())
            mtr, mva = ttr.notna().to_numpy(), tva.notna().to_numpy()
            pred = RidgeCV(alphas=ALPHAS).fit(a[mtr], ttr[mtr]).predict(b[mva])
            out[c] = float(1 - ((tva[mva] - pred) ** 2).sum() / ((tva[mva] - tva[mva].mean()) ** 2).sum())
        return out

    from_usr, from_read, from_ora = recover(usr), recover(readouts), recover(ora)
    print(f"  {'признак':<34} {'важн.':>5} {'[USR]':>6} {'все':>6} {'E1′':>6}  группа")
    for c in top:
        print(f"  {c:<34} {importance[c]:5.2f} {from_usr[c]:6.3f} {from_read[c]:6.3f} {from_ora[c]:6.3f}  "
              f"{group_of(c, data['categorical'])}")

    print("== 7. Состав ушедших и out-of-fold train")
    for g in GROUPS:
        frame = rows[g]
        churned = frame["churn"].astype(bool)
        silence = frame.loc[churned, "act_days_since_last"]
        print(f"  {g}: доля ушедших {churned.mean():.4f}; молчание ушедших на T — медиана {silence.median():.1f} д, "
              f">=14 д {np.mean(silence >= 14):.2f}, >=30 д {np.mean(silence >= 30):.2f}, >=60 д {np.mean(silence >= 60):.2f}; "
              f"PR-AUC одного act_days_since_last {average_precision_score(churned.astype(int), frame['act_days_since_last']):.3f}")
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    yt = y["train"]
    act = bench.hand(groups["act"])
    plain = {k: v for k, v in PARAMS.items() if k not in EARLY_STOPPING}
    # OOF-прогон → его прогон на val: те же C и число деревьев.
    on_val = {"CatBoost × handcrafted": "CatBoost × handcrafted (контроль 0.570)",
              "LR × [USR]": "LR × [USR] (контроль 0.452)", "LR × handcrafted": "LR × handcrafted",
              "LR × act": "LR × act", "LR × точные цели E1′": "LR × точные цели E1′",
              "CatBoost × [USR]": "CatBoost × [USR]"}
    oof = {name: np.zeros(len(yt)) for name in on_val}
    linear = {"LR × [USR]": (usr["train"], c_usr), "LR × handcrafted": (hand["train"], c_hand),
              "LR × act": (act["train"], c_act), "LR × точные цели E1′": (ora["train"], c_ora)}
    usr_frame = pd.DataFrame(usr["train"], index=rows["train"].index, columns=[f"u{i}" for i in range(usr["train"].shape[1])])
    for fit, held in folds.split(usr["train"], yt):
        for name, (x, c) in linear.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(x[fit], yt[fit])
            oof[name][held] = model.predict_proba(x[held])[:, 1]
        for name, frame, cols, cats, trees in (
            ("CatBoost × handcrafted", rows["train"], data["columns"], data["categorical"], trees_hand),
            ("CatBoost × [USR]", usr_frame, list(usr_frame.columns), [], trees_usr),
        ):
            model = CatBoostClassifier(**{**plain, "iterations": trees})
            model.fit(Pool(frame.iloc[fit][cols], label=yt[fit], cat_features=cats), verbose=False)
            oof[name][held] = model.predict_proba(Pool(frame.iloc[held][cols], cat_features=cats))[:, 1]
    rng = np.random.default_rng(0)
    draws = [t for t in (rng.integers(0, len(yt), len(yt)) for _ in range(DRAWS)) if 0 < yt[t].sum() < len(t)]
    base = np.array([average_precision_score(yt[t], oof["CatBoost × handcrafted"][t]) for t in draws])
    for name, score in oof.items():
        d = np.array([average_precision_score(yt[t], score[t]) for t in draws]) - base
        print(f"  OOF train {name:<22} PR-AUC {average_precision_score(yt, score):.3f} (val {bench.results[on_val[name]]['pr_auc']:.3f})"
              f"  Δ к CatBoost × handcrafted {d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

    print("== 8. Бейзлайн pragmatiq: счётчики (n_events, n_tokens, log1p n_events) на T")
    counts = {g: np.column_stack([vec[g]["n_events"], vec[g]["n_tokens"], np.log1p(vec[g]["n_events"])]).astype(np.float64)
              for g in GROUPS}
    bench.run_lr("LR × счётчики", counts, vs=("catboost", "lr_usr"))
    bench.run_cb("CatBoost × счётчики", counts, vs=("catboost", "lr_usr"))


if __name__ == "__main__":
    main()
