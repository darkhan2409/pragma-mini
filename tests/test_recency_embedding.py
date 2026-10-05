from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.mlm.model import RECENCY_PERIODS, RecencyEmbedding, pack
from src.mlm.settings import MlmConfig, checkpoint_path
from src.mlm.train import load_trained, train

from tests import world
from tests.test_attention_backends import reference_attend
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# RecencyEmbedding (recency_embedding в конфиге) кладёт давность
# события до точки отсчёта в сам вектор события перед энкодером
# истории, а слоту [USR] даёт свой вектор. Проверяется, что:
#
#   - при инициализации модель бит в бит та же, что без него:
#     выходной слой и вектор [USR] начинаются с нуля;
#   - давность входит слагаемым ровно в векторы событий, вектор
#     [USR] — в слот анкеты, и без этой части вход истории от
#     времени не зависит вовсе;
#   - плоский путь flash и корзины считают одно и то же;
#   - слой учится вместе с моделью и едет в чекпойнте.
# ============================================================


def lively(seed: int = 11) -> RecencyEmbedding:
    """
    Слой с ненулевым выходом — как после нескольких шагов обучения.
    """

    layer = RecencyEmbedding(world.DIM, seed=seed)
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        layer.outer.weight.copy_(torch.randn(layer.outer.weight.shape, generator=generator) * 0.1)
        layer.outer.bias.copy_(torch.randn(layer.outer.bias.shape, generator=generator) * 0.1)
        layer.usr.copy_(torch.randn(world.DIM, generator=generator))

    return layer


def test_periods_cover_the_whole_scale_of_positions():
    """
    Позиция — 8·log1p(секунды / 8): минута ≈ 17, сутки ≈ 74, два года
    ≈ 127. Периоды идут от двух единиц до всей шкалы.
    """

    assert min(RECENCY_PERIODS) == pytest.approx(2.0)
    assert max(RECENCY_PERIODS) == pytest.approx(256.0)
    assert list(RECENCY_PERIODS) == sorted(RECENCY_PERIODS)


def test_at_initialisation_the_model_is_bit_for_bit_the_same(clients):

    plain = world.model()
    dated = world.model()
    dated.attach_recency(RecencyEmbedding(world.DIM, seed=11))

    plain.eval()
    dated.eval()

    data = pack(clients, CPU)

    with torch.no_grad():
        assert torch.equal(plain(data).logits, dated(data).logits)
        assert torch.equal(plain.client_embeddings(data), dated.client_embeddings(data))


def test_recency_is_added_to_events_and_the_usr_slot_gets_its_own_vector(clients):

    plain = world.model()
    dated = world.model()
    dated.attach_recency(lively())

    data = pack(clients, CPU)

    generator = torch.Generator().manual_seed(3)
    profile = torch.randn(data.clients, world.DIM, generator=generator)
    events = torch.randn(data.events.segments, world.DIM, generator=generator)

    with torch.no_grad():

        base = plain._history_input(data, profile, events)
        flat = dated._history_input(data, profile, events)

        shift = dated.recency(data.event_time_log)

        assert torch.allclose(flat[data.history_event_slot] - base[data.history_event_slot], shift, atol=1e-6)
        assert torch.allclose(
            flat[data.history_profile_slot] - base[data.history_profile_slot],
            dated.recency.usr.expand(data.clients, -1),
            atol=1e-6,
        )

        # Те же события в другое время: без слоя вход истории тот же —
        # время там только в углах TimeRoPE, — со слоем другой.
        later = replace(data, event_time_log=data.event_time_log + 10.0)

        assert torch.equal(plain._history_input(later, profile, events), base)
        assert not torch.allclose(dated._history_input(later, profile, events), flat)


def test_flash_path_and_buckets_agree_with_recency(monkeypatch, clients):

    monkeypatch.setattr("src.mlm.varlen.attend", reference_attend)

    built = world.model(attention="flash")
    built.attach_recency(lively())
    built.eval()

    data = pack(clients, CPU)

    with torch.no_grad():

        buckets = built(data)

        monkeypatch.setattr(built, "_flash", lambda: True)

        flat = built(data)

    assert torch.allclose(flat.logits, buckets.logits, atol=1e-5, rtol=1e-5)
    assert float(flat.loss) == pytest.approx(float(buckets.loss), rel=1e-5)


def test_the_switch_is_off_by_default():

    assert MlmConfig().recency_embedding is False
    assert MlmConfig.from_dict({}).recency_embedding is False


def test_the_recency_layer_learns_and_travels_in_the_checkpoint(stage):

    settle(stage, train_people=many())

    train(tiny(token_budget=6, recency_embedding=True), epochs=1, max_steps=None, masking=every_value())

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    weights = state["model_state_dict"]

    assert state["config"]["recency_embedding"] is True

    # Выход и вектор [USR] стартовали с нуля и сдвинулись.
    assert float(weights["recency.outer.weight"].abs().sum()) > 0.0
    assert float(weights["recency.usr"].abs().sum()) > 0.0

    model, _ = load_trained(checkpoint_path(), CPU)

    assert model.recency is not None

    for name, value in model.recency.named_parameters():
        assert torch.equal(value, weights[f"recency.{name}"]), name
