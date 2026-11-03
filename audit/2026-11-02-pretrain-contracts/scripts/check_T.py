import pyarrow.parquet as pq, pyarrow.compute as pc, json
R = "/home/darkhan/projects/mini_pragma_v2"
for g in ("train", "val"):
    f = pq.ParquetFile(f"{R}/data/02_preprocessed/{g}/events.parquet")
    md = f.schema_arrow.metadata or {}
    man = json.load(open(f"{R}/data/01_raw/{g}/manifest.json"))
    print(g, "02 rows", f.metadata.num_rows, "raw rows", man["events_rows"], "02 period", md.get(b"period_start"), md.get(b"period_end"))
# events exactly at val T in RAW val; and client-action candidates
T = "2026-03-01T00:00:00"
t = pq.read_table(f"{R}/data/01_raw/val/events.parquet", columns=["event_time"]).column("event_time")
at = pc.starts_with(t, T)
print("raw val events with event_time starting", T, ":", pc.sum(at).as_py())
# events at exactly T+60d (window closed end)
for s in ("2026-04-30T00:00:00", "2026-02-28T23:55", "2026-02-28T23:5", "2026-03-01T00:0"):
    print(s, pc.sum(pc.starts_with(t, s)).as_py())
ft = pq.read_table(f"{R}/churn_baseline/data/future/train/events.parquet", columns=["event_time"]).column("event_time")
print("continuation min/max", pc.min(ft).as_py(), pc.max(ft).as_py())
print("continuation events at T", pc.sum(pc.starts_with(ft, "2026-01-01T00:00:00")).as_py(), "at T+60d", pc.sum(pc.starts_with(ft, "2026-03-02T00:00:00")).as_py())
