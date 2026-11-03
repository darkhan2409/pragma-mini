import hashlib, json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
R = "/home/darkhan/projects/mini_pragma_v2"
rep = f"{R}/churn_baseline/reports"
ref = f"{R}/data/13_downstream/ref-2026-10-09"
m = json.load(open(f"{rep}/metrics.json"))
pm = json.load(open(f"{rep}/plus_usr/ref-2026-10-09/metrics.json"))
sha = lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
print("eval sha ok", sha(f"{rep}/eval_rows.parquet")==m["eval_rows_sha256"], "train sha ok", sha(f"{rep}/train_rows.parquet")==m["train_rows_sha256"])
print("plus eval sha ok", sha(f"{rep}/plus_usr/ref-2026-10-09/eval_rows.parquet")==pm["eval_rows_sha256"], sha(f"{rep}/plus_usr/ref-2026-10-09/train_rows.parquet")==pm["train_rows_sha256"])
cols = ["task","client_id","group","T","churn","active90","score"]
ev = pd.read_parquet(f"{rep}/eval_rows.parquet", columns=cols)
tr = pd.read_parquet(f"{rep}/train_rows.parquet", columns=cols)
pev = pd.read_parquet(f"{rep}/plus_usr/ref-2026-10-09/eval_rows.parquet", columns=cols)
ptr = pd.read_parquet(f"{rep}/plus_usr/ref-2026-10-09/train_rows.parquet", columns=cols)
for name, a, b in [("val", ev, pev), ("train", tr, ptr)]:
    print(name, "rows", len(a), len(b), "T", a["T"].unique(), b["T"].unique(), "active90 all", a["active90"].all(), b["active90"].all())
    print("  same ids+order", a["client_id"].equals(b["client_id"]), "same labels", (a["churn"].values==b["churn"].values).all(), "sorted", a["client_id"].is_monotonic_increasing, "dups", a["client_id"].duplicated().sum())
    print("  identical score?", np.allclose(a["score"], b["score"]))
pr = pd.read_parquet(f"{ref}/predictions.parquet")
print(pr.groupby("set").size().to_dict(), pr["group"].unique())
base = ev.set_index("client_id")
for s, part in pr.groupby("set"):
    part = part.set_index("client_id")
    same = part.index.equals(base.index)
    lab = (part["y"].values == base.loc[part.index, "churn"].values).all()
    print(s, "ids == eval_rows ids (order)", same, "labels", lab, "pr_auc", round(average_precision_score(part["y"], part["score"]),4), "roc", round(roc_auc_score(part["y"], part["score"]),4))
cb = pr[pr.set=="catboost"].set_index("client_id")["score"]
print("catboost score == eval_rows score", np.allclose(cb.values, base.loc[cb.index,"score"].values))
pp = pr[pr.set=="catboost_plus_usr"].set_index("client_id")["score"]
print("plus score == plus eval_rows score", np.allclose(pp.values, pev.set_index("client_id").loc[pp.index,"score"].values))
# vectors
for g, rows in [("train", tr), ("val", ev)]:
    v = pd.read_parquet(f"{ref}/{g}.parquet", columns=["client_id","cutoff","n_events","gap_seconds"])
    print(g, "vectors", len(v), "cutoff", v["cutoff"].unique(), "rows missing vector", len(set(rows.client_id)-set(v.client_id)), "dups", v.client_id.duplicated().sum())
    # alignment sanity: usr score vs gap_seconds (val only)
    if g == "val":
        u = pr[pr.set=="usr"].set_index("client_id")
        j = u.join(v.set_index("client_id"))
        from scipy.stats import spearmanr
        print("  spearman(usr score, gap_seconds) aligned", round(spearmanr(j["score"], j["gap_seconds"]).correlation,3),
              "shuffled", round(spearmanr(j["score"].values, np.random.default_rng(0).permutation(j["gap_seconds"].values)).correlation,3))
        jc = pr[pr.set=="catboost"].set_index("client_id").join(v.set_index("client_id"))
        print("  spearman(catboost score, gap_seconds)", round(spearmanr(jc["score"], jc["gap_seconds"]).correlation,3))
# population check from features
f = pd.read_parquet(f"{R}/churn_baseline/data/val/features.parquet", columns=["client_id","T","churn","active90","act_days_since_last","act_count_90"])
print("val features rows", len(f), "active90", f.active90.sum(), "act_count_90>0 == active90", ((f.act_count_90>0)==f.active90).all())
ft = pd.read_parquet(f"{R}/churn_baseline/data/train/features.parquet", columns=["client_id","T","churn","active90","act_days_since_last","act_count_90"])
print("train features rows", len(ft), "act_count_90>0 == active90", ((ft.act_count_90>0)==ft.active90).all())
print("val act_days_since_last max on active90", f.loc[f.active90,"act_days_since_last"].max(), "train", ft.loc[ft.active90,"act_days_since_last"].max())
