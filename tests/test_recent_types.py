from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import src.mlm.model as model_module
from src.mlm.inputs import IGNORE
from src.mlm.model import RECENCY_CAP_DAYS, RECENT_DAYS, RecentTypes, pack
from src.mlm.settings import RETIRED, TELEMETRY_FILE, ConfigError, MlmConfig, checkpoint_path, train_dir
from src.mlm.train import aux_loss, load_trained, train

from tests import world
from tests.test_checkpoint_resume import compare, read
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, fresh, settle, tiny, uneven, windows


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
    """
    Проход отдаёт ошибки по ячейкам; свёртка эталона — их средний
    квадрат по всем клиентам, типам и целям.
    """

    recent = head()
    data = pack(world.clients(), CPU)
    usr = torch.randn(data.clients, world.DIM)

    predicted = recent.proj(usr).view(data.clients, len(RECENT_DAYS) + 1, recent.types)
    expected = expected_targets(recent, data)

    errors = recent(data, usr)

    assert errors.shape == predicted.shape and errors.dtype == torch.float32
    assert torch.allclose(errors, predicted - expected, atol=1e-5)
    assert torch.allclose(aux_loss(errors), F.mse_loss(predicted, expected), atol=1e-6)


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


def test_retired_aux_per_client_key_is_read_only_as_false():
    """
    Переключатель aux_per_client удалён (кандидат 1 аудита 2026-11-03 не
    прошёл правило принятия), но ключ лежит в конфиге каждого чекпойнта кампании со
    значением false и обязан читаться. true — ошибка: чекпойнт,
    обученный с ним, этим кодом не воспроизвести.
    """

    assert RETIRED == {"aux_per_client": False}
    assert MlmConfig.from_dict({"aux_per_client": False}) == MlmConfig.from_dict({})

    with pytest.raises(ConfigError, match="aux_per_client"):
        MlmConfig.from_dict({"aux_per_client": True})

    assert "aux_per_client" not in MlmConfig().as_dict()
    assert MlmConfig.from_dict(MlmConfig().as_dict()) == MlmConfig()


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


# ============================================================
# ШАГ С ЦЕЛЬЮ [USR]: ФОРМУЛА ЭТАЛОНА
# ============================================================
#
# micro-batch с целями MLM делает backward((loss + w · aux) · count),
# aux — средний квадрат ошибки по всем ячейкам micro-batch'а, и окно
# делится на число своих целей. Формула прежнего кода повторяется
# здесь руками и сравнивается через torch.equal.
# ============================================================


WEIGHT = 0.5


def old_recent_forward(self: RecentTypes, data, usr: torch.Tensor) -> torch.Tensor:
    """
    RecentTypes.forward до 2026-11-03: среднее по всем ячейкам
    micro-batch'а.
    """

    predicted = self.proj(usr).float().view(data.clients, len(RECENT_DAYS) + 1, self.types)

    return F.mse_loss(predicted, self.targets(data))


def test_window_step_is_the_reference_formula_bitwise(stage, monkeypatch):
    """
    Градиент шага окна из нескольких micro-batch'ей побитно равен прежней
    формуле: (loss + w · mse) · count на micro-batch с целями и деление на
    число целей окна.
    """

    monkeypatch.setattr(model_module, "recent_types", lambda artifacts, dim, seed: head(dim))

    masking = every_value()

    settle(stage, train_people=uneven(stage))

    probe = tiny(token_budget=14)
    config = tiny(token_budget=14, grad_accum_steps=len(windows(probe, masking)), usr_aux_weight=WEIGHT)

    assert config.grad_accum_steps > 1

    taken: list[list] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        taken.append([p.grad.clone() if p.grad is not None else None for p in parameters])
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    train(config, epochs=1, max_steps=None, masking=masking)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", original)

    assert len(taken) == 1, "тест рассчитан ровно на один шаг оптимизатора"

    model = fresh(None, config)
    model.attach_recent(head(int(model.embedding.dim)))
    model.train()

    targets = 0

    with monkeypatch.context() as patch:

        patch.setattr(RecentTypes, "forward", old_recent_forward)

        for clients in windows(config, masking):
            out = model(pack(clients, CPU), logits=False)
            if out.count > 0:
                ((out.loss + WEIGHT * out.aux) * out.count).backward()
                targets += out.count

    torch._foreach_div_([p.grad for p in model.parameters() if p.grad is not None], targets)

    expected = list(model.parameters())

    assert len(expected) == len(taken[0])
    assert float(model.recent.proj.weight.grad.abs().max()) > 0

    for parameter, grad in zip(expected, taken[0]):
        assert torch.equal(grad, parameter.grad)


def steps_telemetry() -> list[dict]:
    """
    Строки шагов без часов: время и ожидание у прогонов разные.
    """

    rows = [json.loads(line) for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()]

    return [
        {key: value for key, value in row.items() if key not in ("time", "seconds", "wait_seconds")}
        for row in rows if row["kind"] == "step"
    ]


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_resume_mid_epoch_with_the_auxiliary_head(stage, monkeypatch, dropout: float):
    """
    grad_accum_steps=3, семь micro-batch'ей на эпоху: окна 3 + 3 + 1.
    Остановка после первого окна второй эпохи и --resume: веса, AdamW
    (с головой [USR]), расписание и телеметрия шагов те же, что без
    остановки.
    """

    monkeypatch.setattr(model_module, "recent_types", lambda artifacts, dim, seed: head(dim))

    people = many(7)
    masking = every_value()

    config = tiny(token_budget=6, grad_accum_steps=3, warmup_steps=2, usr_aux_weight=WEIGHT)

    settle(stage, train_people=people, dropout=dropout)

    assert len(windows(config, masking)) == 7

    train(config, epochs=2, max_steps=None, masking=masking)

    straight, straight_steps = read(checkpoint_path()), steps_telemetry()

    assert straight["step"] == 6
    assert [row["micro_batches"] for row in straight_steps] == [3, 3, 1] * 2

    settle(stage, train_people=people, dropout=dropout)

    train(config, epochs=2, max_steps=4, masking=masking)

    paused = read(checkpoint_path())

    assert paused["epoch_complete"] is False
    assert paused["epoch"] == 2
    assert paused["micro_batches_done"] == 3

    train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    compare(straight, read(checkpoint_path()))

    assert steps_telemetry() == straight_steps
