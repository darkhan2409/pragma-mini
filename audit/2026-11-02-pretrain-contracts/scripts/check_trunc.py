import pyarrow.parquet as pq, pyarrow.compute as pc, pandas as pd, numpy as np
R = "/home/darkhan/projects/mini_pragma_v2"
f = pq.ParquetFile(f"{R}/data/05_dataset/train/samples.parquet")
ids, first = [], []
for i in range(f.metadata.num_row_groups):
    t = f.read_row_group(i, columns=["client_id", "event_time"])
    ids += t.column("client_id").to_pylist()
    col = t.column("event_time").combine_chunks()
    off = col.offsets.to_numpy(); vals = col.values.cast("int64").to_numpy()
    first += [vals[a] if b > a else np.iinfo("int64").min for a, b in zip(off[:-1], off[1:])]
s = pd.Series(pd.to_datetime(np.array(first), unit="us", utc=True, errors="coerce"), index=ids)
T = pd.Timestamp("2025-12-31T19:00:00Z")
rows = pd.read_parquet(f"{R}/churn_baseline/reports/train_rows.parquet", columns=["client_id","churn"]).set_index("client_id")
start = pd.Timestamp("2023-12-31T19:00:00Z")
feat = pd.read_parquet(f"{R}/churn_baseline/data/train/features.parquet", columns=["client_id","act_days_since_first"]).set_index("client_id")
x = s.reindex(rows.index)
span = (T - x).dt.total_seconds()/86400
print("train task rows", len(rows))
# truncated = first kept event later than the client's first event; approximate by span < (T - first action)
trunc = span + 1e-6 < feat.loc[rows.index, "act_days_since_first"]
print("rows whose kept history starts after their first client action:", int(trunc.sum()), "of", len(rows), "churn among them", int(rows.churn[trunc].sum()))
print("kept span days among those: min", round(span[trunc].min(),1), "p5", round(span[trunc].quantile(0.05),1), "median", round(span[trunc].median(),1))
print("rows with kept span < 90d:", int((span < 90).sum()))
