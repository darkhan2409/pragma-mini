"""
Read-only, CPU: reference checkpoint, fp32 pass vs bf16 autocast pass
(same sdpa path) on short val clients; logit range; calendar vs [EVT]
scale in the event vector.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/darkhan/projects/mini_pragma_v2")

from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source
from src.mlm.model import pack
from src.mlm.train import load_trained

torch.set_num_threads(8)
CPU = torch.device("cpu")
PATH = Path("/home/darkhan/projects/mini_pragma_v2/data/12_train/best_checkpoint.pt")

model, state = load_trained(PATH, CPU, attention_backend="sdpa")
model.eval()

source = Source("val", masking=MaskingConfig.from_dict(state["masking"]))
clients = []
for index in range(source.count):
    for client in source.batch(index):
        if 1000 <= client.n_tokens <= 6000 and client.n_targets > 0:
            clients.append(client)
    if len(clients) >= 24:
        break
clients = clients[:24]
print("clients", len(clients), "max event_time_log", max(float(c.event_time_log.max()) for c in clients))

nll32, nll16, nll_round, cos_usr, aux32, aux16 = [], [], [], [], [], []
lmin, lmax = 1e9, -1e9
evt_norm, cal_norm = [], []
with torch.no_grad():
    for i in range(0, len(clients), 4):
        data = pack(clients[i:i + 4], CPU)
        a = model(data)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            b = model(data)
        la, lb = a.logits.float(), b.logits.float()
        lmin, lmax = min(lmin, float(la.min())), max(lmax, float(la.max()))
        nll32.append(F.cross_entropy(la, a.targets, reduction="none"))
        nll16.append(F.cross_entropy(lb, b.targets, reduction="none"))
        nll_round.append(F.cross_entropy(la.bfloat16().float(), a.targets, reduction="none"))
        u32 = model.client_embeddings(data).float()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            u16 = model.client_embeddings(data).float()
        cos_usr.append(F.cosine_similarity(u32, u16, dim=-1))
        aux32.append(float(a.aux)); aux16.append(float(b.aux))

        # event vector parts: [EVT] after event encoder norm vs calendar term
        dated, _ = model._events(data)
        calendar = model.event.calendar(data.calendar)
        evt_norm.append((dated - calendar).norm(dim=-1))
        cal_norm.append(calendar.norm(dim=-1))

n32, n16, nr = torch.cat(nll32), torch.cat(nll16), torch.cat(nll_round)
print("targets", n32.numel(), "logit range [%.2f, %.2f]" % (lmin, lmax))
print("mean nll fp32 %.5f | full bf16 pass %.5f | fp32 logits rounded to bf16 %.5f" % (n32.mean(), n16.mean(), nr.mean()))
print("per-target |nll bf16 - fp32|: mean %.4f p99 %.4f" % ((n16 - n32).abs().mean(), (n16 - n32).abs().quantile(0.99)))
print("[USR] cosine fp32 vs bf16: min %.5f mean %.5f" % (torch.cat(cos_usr).min(), torch.cat(cos_usr).mean()))
print("aux fp32 %.5f bf16 %.5f" % (np.mean(aux32), np.mean(aux16)))
e, c = torch.cat(evt_norm), torch.cat(cal_norm)
print("event vector: |[EVT] after norm| mean %.2f, |calendar term| mean %.2f (ratio %.2f)" % (e.mean(), c.mean(), float(c.mean() / e.mean())))
