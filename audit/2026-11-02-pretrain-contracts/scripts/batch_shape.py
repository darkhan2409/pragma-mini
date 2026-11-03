"""
Read-only: сколько клиентов в micro-batch'ах train ref-2026-10-09 и какой
вес получает клиент в aux (1/B при accum=1) против равного веса клиента.
"""
import numpy as np
import torch

from src.masking.settings import MaskingConfig
from src.mlm.inputs import micro_batches
from src.mlm.settings import MlmConfig
from src.mlm.train import train_source

ref = torch.load("data/12_train/checkpoint.pt", map_location="cpu", weights_only=True)
config = MlmConfig.from_dict(ref["config"])
masking = MaskingConfig.from_dict(ref["masking"])

sizes = list(train_source(config, masking, 1).sizes())
tokens = np.array([s.n_tokens + s.profile_n_tokens + 1 for s in sizes])
batches = list(micro_batches(sizes, config.token_budget))
B = np.array([len(b) for b in batches])
print("clients", len(sizes), "micro-batches", len(batches), "mean B", B.mean())
print("B distribution: B=1", (B == 1).mean(), "B<=2", (B <= 2).mean(), "B>=5", (B >= 5).mean(), "max B", B.max())
weight = np.concatenate([[1.0 / len(b)] * len(b) for b in batches])  # вес клиента в aux за эпоху
tok = tokens  # тот же порядок
w = weight / weight.sum()
order = np.argsort(tok)
dec = np.array_split(order, 10)
print("decile by cost: share of clients 0.1 each; share of aux weight:",
      [round(float(w[d].sum()), 3) for d in dec])
print("decile median cost:", [int(np.median(tok[d])) for d in dec])
print("max/min client aux weight", weight.max() / weight.min())
