import sys, json, collections
sys.path.insert(0, "/home/darkhan/projects/mini_pragma_v2/churn_baseline")
import pyarrow.parquet as pq, pyarrow.compute as pc, pandas as pd
from churn.raw import decode
from churn.activity import is_client_action
R = "/home/darkhan/projects/mini_pragma_v2"
def at(path, prefixes):
    tab = pq.read_table(path, columns=["client_id","event_time","source","payload"])
    mask = None
    for p in prefixes:
        m = pc.starts_with(tab.column("event_time"), p)
        mask = m if mask is None else pc.or_(mask, m)
    sub = tab.filter(mask)
    fr = decode(sub, 0)
    fr["action"] = is_client_action(fr)
    return fr
for name, path, pre in [("val@T", f"{R}/data/01_raw/val/events.parquet", ["2026-03-01T00:00:00"]),
                        ("cont@T", f"{R}/churn_baseline/data/future/train/events.parquet", ["2026-01-01T00:00:00"]),
                        ("cont@T+60", f"{R}/churn_baseline/data/future/train/events.parquet", ["2026-03-02T00:00:00"]),
                        ]:
    fr = at(path, pre)
    print(name, len(fr), "client actions", int(fr.action.sum()), "clients w/ action", fr.loc[fr.action,"client_id"].nunique())
    print("  types", collections.Counter(fr["type"]).most_common(8))
    print("  action types", collections.Counter(fr.loc[fr.action,"type"]).most_common(8))
