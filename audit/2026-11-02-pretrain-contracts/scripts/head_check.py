"""
Read-only, CPU: reference checkpoint on a few short val clients.
- softmax mass on token ids that cannot be the value of the target's key;
- labels equal to [UNK];
- relative change of every parameter from its initial value.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/darkhan/projects/mini_pragma_v2")

from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source
from src.mlm.model import Mlm, load_model, pack, recent_types
from src.mlm.train import load_trained
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import load_special_tokens

torch.set_num_threads(8)
CPU = torch.device("cpu")
PATH = Path("/home/darkhan/projects/mini_pragma_v2/data/12_train/best_checkpoint.pt")
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

model, state = load_trained(PATH, CPU, attention_backend="sdpa")
model.eval()

art = FrozenArtifacts.load()
spec = load_special_tokens()
V = art.size

allowed: dict[int, np.ndarray] = {}
for key, kid in art.keys.items():
    kind = art.kind(key)
    if kind == "categorical":
        ids = list(art.values[key].values())
    elif kind == "numeric":
        ids = [b.token_id for b in art.buckets[key]]
    else:
        ids = list(art.bpe_ids)
    allowed[kid] = np.asarray(sorted(set(ids)), dtype=np.int64)

masking = MaskingConfig.from_dict(state["masking"])
source = Source("val", masking=masking)

clients = []
for index in range(source.count):
    for client in source.batch(index):
        if client.n_tokens <= MAXTOK and client.n_targets > 0:
            clients.append(client)
    if len(clients) >= WANT:
        break
clients = clients[:WANT]
print("clients", len(clients), "tokens", sum(c.n_tokens for c in clients))

mass_out = []
top1_out = 0
lab_unk = 0
lab_outside = 0
total = 0
with torch.no_grad():
    for i in range(0, len(clients), 8):
        data = pack(clients[i:i + 8], CPU)
        out = model(data)
        prob = out.logits.float().softmax(-1).numpy()
        keys = data.key_ids[data.target_token].numpy()
        labels = out.targets.numpy()
        top = prob.argmax(-1)
        for r in range(labels.size):
            ok = allowed[int(keys[r])]
            m = prob[r, ok].sum()
            mass_out.append(1.0 - m)
            top1_out += int(top[r] not in set(ok.tolist()))
            lab_unk += int(labels[r] == spec["[UNK]"])
            lab_outside += int(labels[r] not in set(ok.tolist()))
            total += 1

mass_out = np.asarray(mass_out)
print("targets", total)
print("softmax mass outside the target key's own values: mean %.4f median %.4f p90 %.4f" % (
    mass_out.mean(), np.median(mass_out), np.percentile(mass_out, 90)))
print("top-1 of an impossible kind:", top1_out, f"({top1_out / total:.4%})")
print("labels == [UNK]:", lab_unk, "; labels outside the key's own values:", lab_outside)

# --- parameters vs init ---
init = load_model(seed=int(state["config"]["seed"]), events_per_chunk=512, label_smoothing=0.1,
                  device=CPU, attention_backend="sdpa", backbone=state["backbone"])
init_state = {k: v for k, v in init.state_dict().items()}
init_state.update({f"head.{k}": v for k, v in Mlm(128, int(state["config"]["seed"])).state_dict().items()})
rec = recent_types(art, 128, int(state["config"]["seed"]) + 1)
init_state.update({f"recent.{k}": v for k, v in rec.state_dict().items()})

trained = state["model_state_dict"]
rows = []
for name, value in trained.items():
    if name not in init_state:
        print("no init for", name)
        continue
    w0 = init_state[name].float()
    change = float((value.float() - w0).norm() / (w0.norm() + 1e-12))
    rows.append((change, name))
rows.sort()
print("smallest relative changes from init:")
for change, name in rows[:8]:
    print(f"  {name}: {change:.4f}")
print("largest:")
for change, name in rows[-4:]:
    print(f"  {name}: {change:.4f}")

# embedding rows that are never targets: specials and keys
table = trained["embedding.table.weight"]
table0 = init_state["embedding.table.weight"]
key_ids = sorted(art.keys.values())
for label, ids in (("specials", list(spec.values())), ("keys", key_ids)):
    ids = torch.tensor(ids)
    print(label, "row norm init %.3f -> trained %.3f" % (float(table0[ids].norm(dim=1).mean()), float(table[ids].norm(dim=1).mean())))
for name, tid in spec.items():
    print(f"  {name}: init {float(table0[tid].norm()):.3f} trained {float(table[tid].norm()):.3f}")
