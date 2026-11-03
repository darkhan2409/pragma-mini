from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from src.mlm.model import Model


# ============================================================
# ЗАПЛАТКИ LoRA И ГОЛОВА
# ============================================================
#
# Заплатка W + (alpha/rank)·B·A ставится параметризацией на веса
# эталона, поэтому и обычный путь, и varlen/flash читают уже
# заплатанный вес без правок кода модели. B стартует нулями: до первого
# шага модель совпадает с эталоном побитово.
# ============================================================


TARGETS_EVENT = ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight")
TARGETS_BLOCK = ("qkv.weight", "out.weight", "ffn.0.weight", "ffn.3.weight")


class LoraDelta(nn.Module):

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()

        self.scale = alpha / rank
        # A на CPU заданным генератором: одинаковый seed — одинаковые A на любом устройстве.
        self.A = nn.Parameter(torch.randn(rank, in_features, generator=generator) / math.sqrt(in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, weight: Tensor) -> Tensor:

        # Добавка в fp32: в bf16 малая добавка к большому весу проглатывается.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            delta = (self.B.float() @ self.A.float()) * self.scale
            return (weight.float() + delta).to(weight.dtype)


class ChurnHead(nn.Module):

    def __init__(self, dim: int, base_rate: float):
        super().__init__()

        self.linear = nn.Linear(dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, math.log(base_rate / (1.0 - base_rate)))

    def forward(self, usr: Tensor) -> Tensor:

        # Логит около -2.9 в bf16 теряет точность: голова всегда в fp32.
        with torch.autocast(device_type=usr.device.type, enabled=False):
            return self.linear(usr.float()).squeeze(-1)


def target_names(model: Model) -> list[str]:

    names = []
    for part, targets in (("event", TARGETS_EVENT), ("profile", TARGETS_BLOCK), ("history", TARGETS_BLOCK)):
        for i in range(len(getattr(model, part).layers)):
            names += [f"{part}.layers.{i}.{target}" for target in targets]

    return names


def locate(model: Model, name: str) -> tuple[nn.Module, str]:

    owner, _, leaf = name.rpartition(".")

    return model.get_submodule(owner), leaf


def delta_of(model: Model, name: str) -> LoraDelta:

    owner, leaf = locate(model, name)

    return owner.parametrizations[leaf][0]


def inject(model: Model, rank: int, alpha: float, seed: int) -> list[str]:

    names = target_names(model)
    if any(parametrize.is_parametrized(*locate(model, name)) for name in names):
        raise ValueError("заплатки LoRA уже поставлены")

    for p in model.parameters():
        p.requires_grad_(False)

    generator = torch.Generator().manual_seed(seed)
    for name in names:
        owner, leaf = locate(model, name)
        weight = getattr(owner, leaf)
        out_features, in_features = weight.shape
        delta = LoraDelta(out_features, in_features, rank, alpha, generator).to(weight.device)
        parametrize.register_parametrization(owner, leaf, delta)

    learning = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    expected = sorted(f"{locate_path(name)}.{x}" for name in names for x in "AB")
    if learning != expected:
        raise ValueError("обучаемыми должны быть ровно A и B заплаток")

    return names


def locate_path(name: str) -> str:

    owner, _, leaf = name.rpartition(".")

    return f"{owner}.parametrizations.{leaf}.0"


def trainable(model: Model, head: ChurnHead) -> list[nn.Parameter]:

    return [p for p in (*model.parameters(), *head.parameters()) if p.requires_grad]


def adapter_state(model: Model, head: ChurnHead) -> dict[str, Tensor]:

    state = {}
    for name in target_names(model):
        delta = delta_of(model, name)
        state[f"{name}.A"] = delta.A.detach().cpu().clone()
        state[f"{name}.B"] = delta.B.detach().cpu().clone()
    state["head.weight"] = head.linear.weight.detach().cpu().clone()
    state["head.bias"] = head.linear.bias.detach().cpu().clone()

    return state


def load_adapters(model: Model, head: ChurnHead, state: dict[str, Tensor]) -> None:

    slots: dict[str, nn.Parameter] = {}
    for name in target_names(model):
        delta = delta_of(model, name)
        slots[f"{name}.A"] = delta.A
        slots[f"{name}.B"] = delta.B
    slots["head.weight"] = head.linear.weight
    slots["head.bias"] = head.linear.bias

    if set(state) != set(slots):
        lost, alien = sorted(set(slots) - set(state)), sorted(set(state) - set(slots))
        raise ValueError(f"ключи заплаток не совпадают: нет {lost[:3]}, лишние {alien[:3]}")
    for key, slot in slots.items():
        if tuple(state[key].shape) != tuple(slot.shape):
            raise ValueError(f"форма {key}: {tuple(state[key].shape)}, нужна {tuple(slot.shape)}")

    with torch.no_grad():
        for key, slot in slots.items():
            slot.copy_(state[key])


def delta_ratio(model: Model) -> float:

    delta_sq = weight_sq = 0.0
    with torch.no_grad():
        for name in target_names(model):
            owner, leaf = locate(model, name)
            original = owner.parametrizations[leaf].original.float()
            delta = delta_of(model, name)
            change = (delta.B.float() @ delta.A.float()) * delta.scale
            delta_sq += float(change.pow(2).sum())
            weight_sq += float(original.pow(2).sum())

    return math.sqrt(delta_sq / weight_sq)
