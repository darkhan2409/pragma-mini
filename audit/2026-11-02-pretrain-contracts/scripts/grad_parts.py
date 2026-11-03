"""
Read-only: доли MLM и [USR]-aux в норме градиента шага (accum=1, как в
ref-2026-10-09) на весах init / epoch_01 / epoch_05, CPU, SDPA, fp32.
Ничего не пишет.
"""
import sys
import time

import torch

from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source, micro_batches
from src.mlm.model import load_model, pack
from src.mlm.settings import MlmConfig
from src.mlm.train import attach_recent, for_epoch, load_trained

torch.set_num_threads(4)
cpu = torch.device("cpu")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else None

ref = torch.load("data/12_train/checkpoint.pt", map_location="cpu", weights_only=True)
config = MlmConfig.from_dict(ref["config"])
masking = MaskingConfig.from_dict(ref["masking"])


def models():
    m = load_model(seed=config.seed, events_per_chunk=config.events_per_chunk,
                   label_smoothing=config.label_smoothing, device=cpu, attention_backend="sdpa")
    attach_recent(m, config)
    yield "init", m
    for name in ("epoch_05",):
        m, _ = load_trained(__import__("pathlib").Path(f"data/12_train/epochs/{name}.pt"), cpu, "sdpa")
        yield name, m


def norm(model):
    return float(torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]).norm())


batches = []
short = (c for c in Source("train", masking=for_epoch(masking, 1)).clients() if c.n_events <= 400)
print("source open", flush=True)
for clients in micro_batches(short, BUDGET or config.token_budget):
    batches.append(clients)
    if len(batches) == N:
        break

for name, model in models():
    model.train()
    for i, clients in enumerate(batches):
        data = pack(clients, cpu)
        res = {}
        for part in ("mlm", "aux", "both"):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(0)  # одинаковый dropout у трёх проходов
            out = model(data, logits=False)
            obj = {"mlm": out.loss, "aux": config.usr_aux_weight * out.aux,
                   "both": out.loss + config.usr_aux_weight * out.aux}[part]
            obj.backward()
            res[part] = norm(model)
            res["loss"], res["auxv"] = float(out.loss), float(out.aux)
        print(f"{name} mb{i} clients={len(clients)} targets={out.count} mlm_loss={res['loss']:.3f} "
              f"aux={res['auxv']:.3f} |g_mlm|={res['mlm']:.3f} |g_aux|={res['aux']:.3f} "
              f"|g_both|={res['both']:.3f}", flush=True)
