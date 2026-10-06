from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from src.mlm.inputs import IGNORE
from src.mlm.model import RECENCY_CAP_DAYS, RECENT_DAYS, RecentTypes, pack
from src.mlm.settings import MlmConfig, checkpoint_path
from src.mlm.train import load_trained, train

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вспомогательная цель [USR] — log1p числа событий каждого типа за
# 7/30/90 дней до точки отсчёта и log1p давности последнего события
# каждого типа (не больше RECENCY_CAP_DAYS). Проверяется, что:
#
#   - цели ровно те, что посчитаны руками по ленте клиента, включая
#     пустые окна (ноль) и типы без событий (предел давности);
#   - потеря — средний квадрат ошибки по этим целям;
#   - тип закрытого маской события берётся из его метки: маска MLM
#     не прячет от цели то, что было на самом деле;
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


def expected_targets(recent: RecentTypes, data) -> torch.Tensor:
    """
    Те же цели циклами по клиентам и событиям.
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

    out = np.zeros((data.clients, len(RECENT_DAYS) + 1, recent.types))

    for client in range(data.clients):
        latest = np.full(recent.types, RECENCY_CAP_DAYS)
        for event, kind in kinds.items():
            if client_of[event] != client or kind < 0:
                continue
            days = 8 * math.expm1(ages[event] / 8) / 86_400
            for window, limit in enumerate(RECENT_DAYS):
                if days <= limit:
                    out[client, window, kind] += 1
            latest[kind] = min(latest[kind], days, RECENCY_CAP_DAYS)
        out[client, len(RECENT_DAYS)] = latest

    return torch.as_tensor(np.log1p(out), dtype=torch.float32)


def test_targets_are_the_counts_and_the_recency_of_each_type():

    recent = head()
    data = pack(world.clients(), CPU)

    assert torch.allclose(recent.targets(data), expected_targets(recent, data), atol=1e-5)


def test_empty_windows_and_unseen_types_are_targets_too():
    """
    Пустое окно учится нулём, тип без событий — пределом давности: «за
    7 дней ничего» — тоже ответ.
    """

    recent = head()
    data = pack(world.clients(), CPU)
    targets = recent.targets(data)

    assert (targets[:, : len(RECENT_DAYS)] == 0).any()
    assert torch.isclose(targets[:, len(RECENT_DAYS)], torch.tensor(math.log1p(RECENCY_CAP_DAYS))).any()


def test_loss_is_the_mean_squared_error_against_the_targets():

    recent = head()
    data = pack(world.clients(), CPU)
    usr = torch.randn(data.clients, world.DIM)

    predicted = recent.proj(usr).view(data.clients, len(RECENT_DAYS) + 1, recent.types)

    assert torch.allclose(recent(data, usr), F.mse_loss(predicted, expected_targets(recent, data)), atol=1e-6)


def test_a_masked_event_type_is_read_from_its_label():
    """
    Метка у закрытого маской значения — настоящий тип: цели те же,
    что без маски.
    """

    recent = head()
    data = pack(world.clients(), CPU)

    first = next(index for index, key in enumerate(data.key_ids.tolist())
                 if key == world.KEY_A and int(data.positions[index]) == 0)

    hidden = data.value_ids.clone()
    labels = data.labels.clone()
    labels[first] = hidden[first]
    hidden[first] = 2  # [MASK]

    masked = replace(data, value_ids=hidden, labels=labels)

    assert torch.equal(recent.targets(masked), recent.targets(data))


def test_the_target_is_on_by_default_and_old_checkpoints_keep_their_weight():
    """
    С 2026-10-06 цель [USR] — часть эталона. Чекпойнт хранит свой вес,
    и B0 с весом 0 загружается без головы.
    """

    assert MlmConfig().usr_aux_weight == 1.0
    assert MlmConfig.from_dict({}).usr_aux_weight == 1.0
    assert MlmConfig.from_dict({"usr_aux_weight": 0.0}).usr_aux_weight == 0.0


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
