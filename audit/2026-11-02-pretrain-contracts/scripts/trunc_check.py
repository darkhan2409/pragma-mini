import sys, json
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
from datetime import datetime
base = "/home/darkhan/projects/mini_pragma_v2/data/05_dataset/"
for group in sys.argv[1:]:
    meta = json.load(open(base + group + "/meta.json"))
    T = np.datetime64(datetime.fromisoformat(meta["events_cutoff"]).replace(tzinfo=None), "us").astype(np.int64)
    f = pq.ParquetFile(base + group + "/samples.parquet")
    ne, nt, oldest = [], [], []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["event_time", "key_ids"])
        et = t.column("event_time").combine_chunks()
        ne += pc.list_value_length(et).to_pylist()
        nt += pc.list_value_length(t.column("key_ids").combine_chunks()).to_pylist()
        flat = et.flatten().cast(pa.timestamp("us")).to_numpy().astype(np.int64) if True else None
        offs = et.offsets.to_numpy()
        for a, b in zip(offs[:-1], offs[1:]):
            oldest.append((T - flat[a]) / 86400e6 if b > a else np.nan)
    ne, nt, oldest = map(np.asarray, (ne, nt, oldest))
    capped = (ne >= meta["context"]["max_events"]) | (nt >= meta["context"]["max_tokens"] - 200)
    print(group, "samples", ne.size, "truncated(meta)", meta["truncated_samples"], "at event/token cap", int(capped.sum()))
    if capped.any():
        o = oldest[capped]
        print("  oldest kept event age (days) among capped: min %.1f p10 %.1f median %.1f" % (np.nanmin(o), np.nanpercentile(o, 10), np.nanmedian(o)))
        print("  capped samples whose visible history is shorter than 90 days:", int((o < 90).sum()), "; shorter than 365:", int((o < 365).sum()))
