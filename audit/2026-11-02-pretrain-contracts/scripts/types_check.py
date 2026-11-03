import sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/darkhan/projects/mini_pragma_v2")
from src.tokenization.finalvocab import FrozenArtifacts
from src.mlm.model import recent_types
a = FrozenArtifacts.load(); r = recent_types(a, 128, 43); tov = r.type_of_value.numpy(); key = r.event_type_key
for g in ("train", "val", "test"):
    f = pq.ParquetFile(f"/home/darkhan/projects/mini_pragma_v2/data/05_dataset/{g}/samples.parquet")
    ev = no_type = multi = unmapped = 0
    for i in range(min(10, f.num_row_groups)):
        t = f.read_row_group(i, columns=["key_ids", "value_ids", "positions", "event_starts"]).to_pydict()
        for k, v, p, s in zip(t["key_ids"], t["value_ids"], t["positions"], t["event_starts"]):
            k, v, p, s = map(np.asarray, (k, v, p, s))
            owner = np.searchsorted(s, np.arange(k.size), side="right") - 1
            typed = (k == key) & (p == 0)
            per = np.bincount(owner[typed], minlength=s.size)
            ev += s.size; no_type += int((per == 0).sum()); multi += int((per > 1).sum())
            unmapped += int((tov[v[typed]] < 0).sum())
    print(g, "events", ev, "without event_type", no_type, "with >1 event_type", multi, "event_type value not in the type map", unmapped)
