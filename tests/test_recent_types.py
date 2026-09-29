from __future__ import annotations

import math

import numpy as np
import torch

from src.mlm.inputs import IGNORE
from src.mlm.model import RECENT_DAYS, RecentTypes, pack
from src.mlm.settings import checkpoint_path
from src.mlm.train import load_trained, train

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вспомогательная цель [USR] — доли типов событий за 7/30/90 дней
# до последнего события. Проверяется, что:
#
#   - потеря — ровно кросс-энтропия с долями, посчитанными руками
#     по ленте клиента;
#   - тип закрытого маской события берётся из его метки: маска MLM
#     не прячет от вспомогательной цели то, что было на самом деле;
#   - с весом больше нуля голова учится и едет в чекпойнте.
#
# В синтетическом мире ключа event_type нет: его роль играет key_a,
# а типы — его значения.
# ============================================================


def head(dim: int = world.DIM) -> RecentTypes:

    type_of_value = torch.full((world.VOCAB,), -1, dtype=torch.long)
    values = list(range(10, 40))
    type_of_value[values] = torch.arange(len(values))

    return RecentTypes(dim, type_of_value, world.KEY_A, seed=1)


def expected(recent: RecentTypes, data, usr: torch.Tensor) -> torch.Tensor:
    """
    Та же потеря циклами по клиентам и событиям.
    """

    key_ids, labels, values = data.key_ids.tolist(), data.labels.tolist(), data.value_ids.tolist()
    positions, owner = data.positions.tolist(), data.event_of_token.tolist()
    client_of = data.user_of_event.tolist()
    ages = data.event_time_log.tolist()

    kinds = {}
    for index, key in enumerate(key_ids):
        if key == world.KEY_A and positions[index] == 0:
            value = labels[index] if labels[index] != IGNORE else values[index]
            kinds[owner[index]] = int(recent.type_of_value[value])

    logq = recent.proj(usr).view(data.clients, len(RECENT_DAYS), recent.types).log_softmax(-1)

    losses = []
    for client in range(data.clients):
        for window, limit in enumerate(RECENT_DAYS):
            counts = np.zeros(recent.types)
            for event, kind in kinds.items():
                days = 8 * math.expm1(ages[event] / 8) / 86_400
                if client_of[event] == client and kind >= 0 and days <= limit:
                    counts[kind] += 1
            if counts.sum():
                share = torch.as_tensor(counts / counts.sum(), dtype=torch.float32)
                losses.append(-(share * logq[client, window]).sum())

    return torch.stack(losses).mean()


def test_loss_is_the_cross_entropy_with_the_recent_type_shares():

    recent = head()
    data = pack(world.clients(), CPU)
    usr = torch.randn(data.clients, world.DIM)

    assert torch.allclose(recent(data, usr), expected(recent, data, usr), atol=1e-6)


def test_a_masked_event_type_is_read_from_its_label():
    """
    Метка у закрытого маской значения — настоящий тип: доли те же,
    что без маски.
    """

    recent = head()
    clients = world.clients()
    data = pack(clients, CPU)
    usr = torch.randn(data.clients, world.DIM)

    first = next(index for index, key in enumerate(data.key_ids.tolist())
                 if key == world.KEY_A and int(data.positions[index]) == 0)

    hidden = data.value_ids.clone()
    labels = data.labels.clone()
    labels[first] = hidden[first]
    hidden[first] = 2  # [MASK]

    from dataclasses import replace

    masked = replace(data, value_ids=hidden, labels=labels)

    assert torch.equal(recent(masked, usr), recent(data, usr))


def test_the_auxiliary_head_learns_and_travels_in_the_checkpoint(stage, monkeypatch):

    import src.mlm.model as model_module

    monkeypatch.setattr(model_module, "recent_types", lambda artifacts, dim, seed: head(dim))

    settle(stage, train_people=many())

    train(tiny(token_budget=6, usr_aux_weight=0.5), epochs=1, max_steps=None, masking=every_value())

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)

    assert state["config"]["usr_aux_weight"] == 0.5
    assert not torch.equal(state["model_state_dict"]["recent.proj.weight"], head().proj.weight)

    model, _ = load_trained(checkpoint_path(), CPU)

    assert model.recent is not None
    assert torch.equal(model.recent.proj.weight, state["model_state_dict"]["recent.proj.weight"])
