import sys
import numpy as np, torch
sys.path.insert(0, "/home/darkhan/projects/mini_pragma_v2")
from src.temporal.samples import TemporalGroup
from src.dataset.settings import dataset_dir
from src.temporal.position import TIME_SCALE
g = TemporalGroup("train", dataset_dir("train"))
cutoff = np.datetime64(g.cutoff.replace(tzinfo=None), "us")
n = int(sys.argv[1])
hits = []
total = 0
for i in range(n):
    t = g.row_group(i, columns=["client_id", "event_time", "event_time_log"]).to_pylist()
    for row in t:
        times = np.asarray([x.replace(tzinfo=None) for x in row["event_time"]], dtype="datetime64[us]")
        d_us = (cutoff - times).astype("timedelta64[us]").astype(np.int64)
        etl = torch.tensor(row["event_time_log"], dtype=torch.float32)
        days_f = (TIME_SCALE * torch.expm1(etl / TIME_SCALE) / 86400.0).numpy()
        total += times.size
        for lim in (7, 30, 90):
            exact = d_us == lim * 86_400_000_000
            near = np.abs(d_us - lim * 86_400_000_000) <= 1_000_000
            for j in np.flatnonzero(near):
                hits.append((lim, int(d_us[j]) - lim * 86_400_000_000, bool(exact[j]), float(days_f[j]), bool(days_f[j] <= lim), str(times[j])))
print("events", total, "within 1s of an edge", len(hits))
for h in hits[:20]:
    print(h)
