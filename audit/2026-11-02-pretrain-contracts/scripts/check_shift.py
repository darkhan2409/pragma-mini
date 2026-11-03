import json, numpy as np, pandas as pd
R = "/home/darkhan/projects/mini_pragma_v2"
ref = f"{R}/data/13_downstream/ref-2026-10-09"
rep = json.load(open(f"{ref}/report.json"))["tasks"]["churn_active90"]["results"]
print("usr cv (train OOF logloss by C):", [(p["C"], round(p["log_loss"],4)) for p in rep["usr"]["cv"]])
print("usr val logloss", round(rep["usr"]["val"]["log_loss"],4))
m = json.load(open(f"{R}/churn_baseline/reports/metrics.json"))["tasks"]["churn_active90"]
print("catboost inner_holdout logloss at best", m["holdout_logloss"][m["best_iteration"]], "val", round(m["groups"]["val"]["log_loss"],4))
pm = json.load(open(f"{R}/churn_baseline/reports/plus_usr/ref-2026-10-09/metrics.json"))["tasks"]["churn_active90"]
print("plus inner_holdout logloss at best", pm["holdout_logloss"][pm["best_iteration"]], "val", round(pm["groups"]["val"]["log_loss"],4), "best_it", pm["best_iteration"])
print("catboost holdout PRAUC at best", m["holdout_curve"][m["best_iteration"]], "plus", pm["holdout_curve"][pm["best_iteration"]])
pr = pd.read_parquet(f"{ref}/predictions.parquet")
for s, part in pr.groupby("set"):
    print(s, "mean score val", round(part.score.mean(),4), "actual rate", round(part.y.mean(),4))
tr_rows = pd.read_parquet(f"{R}/churn_baseline/reports/train_rows.parquet", columns=["client_id","churn"])
ev_rows = pd.read_parquet(f"{R}/churn_baseline/reports/eval_rows.parquet", columns=["client_id","churn"])
tv = pd.read_parquet(f"{ref}/train.parquet", columns=["client_id","usr","gap_seconds","n_events"]).set_index("client_id").loc[tr_rows.client_id]
vv = pd.read_parquet(f"{ref}/val.parquet", columns=["client_id","usr","gap_seconds","n_events"]).set_index("client_id").loc[ev_rows.client_id]
X = np.stack(tv.usr.to_numpy()); V = np.stack(vv.usr.to_numpy())
print("dim", X.shape[1])
mu, sd = X.mean(0), X.std(0) + 1e-12
d = (V.mean(0) - mu) / sd
print("std mean shift val vs train: mean|d|", round(np.abs(d).mean(),3), "max|d|", round(np.abs(d).max(),3), "share |d|>0.2", round((np.abs(d)>0.2).mean(),3), "share |d|>0.5", round((np.abs(d)>0.5).mean(),3))
print("sd ratio val/train median", round(np.median(V.std(0)/sd),3))
print("norm train", np.percentile(np.linalg.norm(X,axis=1),[5,50,95]).round(3), "val", np.percentile(np.linalg.norm(V,axis=1),[5,50,95]).round(3))
# same for gap_seconds/n_events as reference of covariate shift
print("gap days median train", round(tv.gap_seconds.median()/86400,3), "val", round(vv.gap_seconds.median()/86400,3))
print("n_events median train", tv.n_events.median(), "val", vv.n_events.median())
# shift within negatives only
neg_t = X[tr_rows.churn.values==0]; neg_v = V[ev_rows.churn.values==0]
d2 = (neg_v.mean(0)-neg_t.mean(0))/(neg_t.std(0)+1e-12)
print("negatives only: mean|d|", round(np.abs(d2).mean(),3), "share |d|>0.2", round((np.abs(d2)>0.2).mean(),3))
