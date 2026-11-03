"""
Read-only: aux targets of RecentTypes vs an independent count from the
unmasked source and the raw event_time (train group, first row groups).
"""
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, "/home/darkhan/projects/mini_pragma_v2")

from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source
from src.mlm.model import RECENCY_CAP_DAYS, RECENT_DAYS, pack, recent_types
from src.mlm.train import for_epoch
from src.temporal.samples import TemporalGroup
from src.dataset.settings import dataset_dir
from src.tokenization.finalvocab import FrozenArtifacts

CPU = torch.device("cpu")
GROUPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EPOCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1

state = torch.load("/home/darkhan/projects/mini_pragma_v2/data/12_train/best_checkpoint.pt",
                   map_location="cpu", weights_only=False)
masking = for_epoch(MaskingConfig.from_dict(state["masking"]), EPOCH)

artifacts = FrozenArtifacts.load()
recent = recent_types(artifacts, 128, 43)
key = recent.event_type_key
tov = recent.type_of_value.numpy()

source = Source("train", masking=masking)
raw = TemporalGroup("train", dataset_dir("train"))
cutoff = np.datetime64(raw.cutoff.replace(tzinfo=None), "us")

tot_events = dropped = 0
diff_cells = cells = 0
max_abs = 0.0
cap_never = cap_old = rec_cells = 0
count_mismatch_vs_truth = 0
boundary_hits = 0
sq = 0.0
all_got = []
false_cap = rec_diff = cnt_diff = clients_n = clients_false_cap = 0
win_mismatch_time = 0

for index in range(GROUPS):
    clients = source.batch(index)
    table = raw._file.read_row_group(index, columns=["client_id", "value_ids", "key_ids", "positions",
                                                     "event_starts", "event_time"]).to_pylist()
    by_id = {row["client_id"]: row for row in table}

    for client in clients:
        data = pack([client], CPU)
        got = recent.targets(data).numpy()[0]  # [4, types]

        row = by_id[client.client_id]
        src_vals = np.asarray(row["value_ids"])
        keys = np.asarray(row["key_ids"])
        pos = np.asarray(row["positions"])
        starts = np.asarray(row["event_starts"])
        times = np.asarray([t.replace(tzinfo=None) for t in row["event_time"]], dtype="datetime64[us]")
        age_days = (cutoff - times).astype("timedelta64[us]").astype(np.int64) / 86_400e6

        owner = np.searchsorted(starts, np.arange(keys.size), side="right") - 1
        typed = (keys == key) & (pos == 0)
        kind_true = np.full(starts.size, -1)
        kind_true[owner[typed]] = tov[src_vals[typed]]

        # Events that the target counted (masked labels + visible)
        vis = client.value_ids
        lab = client.labels
        original = np.where(lab != -100, lab, vis)
        kind_used = np.full(starts.size, -1)
        kind_used[owner[typed]] = tov[original[typed]]

        tot_events += starts.size
        dropped += int(((kind_used < 0) & (kind_true >= 0)).sum())

        # truth from source + datetimes
        truth = np.zeros((len(RECENT_DAYS) + 1, recent.types))
        latest = np.full(recent.types, np.inf)
        for e in range(starts.size):
            k = kind_true[e]
            if k < 0:
                continue
            for w, lim in enumerate(RECENT_DAYS):
                if age_days[e] <= lim:
                    truth[w, k] += 1
                if abs(age_days[e] - lim) < 1.0 / 86400:
                    boundary_hits += 1
            latest[k] = min(latest[k], age_days[e])
        never = ~np.isfinite(latest)
        old = np.isfinite(latest) & (latest > RECENCY_CAP_DAYS)
        cap_never += int(never.sum()); cap_old += int(old.sum()); rec_cells += recent.types
        truth[len(RECENT_DAYS)] = np.minimum(np.where(never, RECENCY_CAP_DAYS, latest), RECENCY_CAP_DAYS)
        truth = np.log1p(truth)

        delta = np.abs(got - truth)
        capv = np.log1p(RECENCY_CAP_DAYS)
        false_cap += int(((np.abs(got[-1] - capv) < 1e-6) & (truth[-1] < capv - 1e-3)).sum())
        rec_diff += int((delta[-1] > 1e-4).sum())
        cnt_diff += int((delta[:-1] > 1e-4).sum())
        clients_n += 1
        clients_false_cap += int(((np.abs(got[-1] - capv) < 1e-6) & (truth[-1] < capv - 1e-3)).any())
        sq += float((delta ** 2).sum()); all_got.append(got)
        cells += delta.size
        diff_cells += int((delta > 1e-4).sum())
        max_abs = max(max_abs, float(delta.max()))

        # same kinds (kind_used) but windows from datetimes: isolates the time path
        same = np.zeros((len(RECENT_DAYS), recent.types))
        for e in range(starts.size):
            k = kind_used[e]
            if k < 0:
                continue
            for w, lim in enumerate(RECENT_DAYS):
                if age_days[e] <= lim:
                    same[w, k] += 1
        win_mismatch_time += int((np.abs(np.log1p(same) - got[:len(RECENT_DAYS)]) > 1e-5).sum())

print(f"epoch {EPOCH}, groups {GROUPS}")
print("events", tot_events, "dropped from target (type -> [UNK] without label)", dropped,
      f"({dropped / max(tot_events, 1):.4%})")
print("cells", cells, "cells differing from source truth", diff_cells, "max |log1p diff|", max_abs)
print("window cells differing when only time path differs (datetime vs event_time_log):", win_mismatch_time)
print("events within 1s of a window edge:", boundary_hits)
print("recency cells", rec_cells, "never seen -> cap", cap_never, f"({cap_never / rec_cells:.2%})",
      "seen but >365d -> cap", cap_old, f"({cap_old / rec_cells:.2%})")
print("count cells differing", cnt_diff, "recency cells differing", rec_diff,
      "recency cells set to cap although the type occurred <365d (all its events [UNK])", false_cap,
      "clients with >=1 such cell", clients_false_cap, "of", clients_n)
G = np.stack(all_got)
print("MSE(target, truth) =", sq / cells)
print("MSE of per-cell mean predictor =", float(((G - G.mean(0)) ** 2).mean()),
      "counts part", float(((G[:, :3] - G[:, :3].mean(0)) ** 2).mean()), "recency part", float(((G[:, 3] - G[:, 3].mean(0)) ** 2).mean()))
print("share of recency cells at cap in targets", float((np.abs(G[:, 3] - np.log1p(RECENCY_CAP_DAYS)) < 1e-6).mean()))
