"""
Время во внимании: поворот q и k вместо слагаемого к вектору.

Проверяется, что поворот действительно применяется и зависит
только от разности времён, что признаки времени считаются верно,
что batch с сессиями и batch без единой сессии проходят forward
и backward, и что база частот доезжает до checkpoint.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.tokenizer.config import IncompatibleArtifactsError
from src.tokenizer.dataset import collate
from src.tokenizer.masking import Masker, MaskingConfig
from src.model.backbone import build_backbone
from src.model.checkpoint import load_checkpoint
from src.model.config import ModelConfig
from src.model.history_batching import (
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
)
from src.model.history_encoder import HistoryEncoder
from src.model.losses import mlm_loss
from src.model.mlm_batching import build_targets, representations
from src.model.mlm_head import FieldTable, MLMHead
from src.model.rotary import RotaryHistoryLayer, apply_rotary, rotary_tables
from src.model.time_features import (
    INACTIVITY_NORM_HOURS,
    calendar_features,
    squash_np,
)
from src.model.trainer import Trainer, run_training

from tests.helpers_data import examples as session_examples, synthetic_batch, toy_vocab
from tests.helpers_model import mlm_config, prepared as mlm_prepared, full_config, small_config, toy_config



# ============================================================
# ХЕЛПЕРЫ
# ============================================================


def rope_config(vocab=None) -> ModelConfig:
    return toy_config(vocab)


def standalone_case(vocab):
    """
    Batch без единой сессии: каждое событие своим элементом.
    """

    history, targets, table = mlm_prepared([[0, 5, 20], [0, 2]], vocab=vocab)

    return history, targets, table, mlm_config(vocab)


def ledger_case(vocab):
    """
    Лента из разнородных событий одного клиента.
    """

    items = list(session_examples(vocab).values())

    masker = Masker(vocab, MaskingConfig(mode="token", seed=3, token_rate=1.0))

    history = prepare_history_batch(
        collate(items),
        metadata_from_examples(items),
        None,
        masker=masker,
    )

    config = full_config(vocab)

    return history, build_targets(history, FieldTable(vocab)), FieldTable(vocab), config


CASES = {"standalone": standalone_case, "ledger": ledger_case}


# ============================================================
# 1. ПОВОРОТ
# ============================================================


def test_rotary_layer_matches_the_plain_layer_and_rotation_is_relative():
    """
    Слой обязан совпасть с обычным при нулевых координатах и
    зависеть только от РАЗНОСТИ координат при ненулевых.

    Совпадение при нуле это проверка собственной реализации
    внимания на SDPA, включая полярность маски: у src_mask True
    означает запрет, у SDPA — разрешение.
    """

    torch.manual_seed(0)

    config = ModelConfig(
        vocab_size=64, d_model=32, n_heads=4, dim_feedforward=64, dropout=0.0
    )

    rope = RotaryHistoryLayer(config).eval()

    plain = nn.TransformerEncoderLayer(
        d_model=32, nhead=4, dim_feedforward=64, dropout=0.0,
        activation="gelu", layer_norm_eps=1e-5, batch_first=True, norm_first=True,
    ).eval()

    # Имена весов совпадают, поэтому переносятся как есть.
    plain.load_state_dict(rope.state_dict())

    batch, length = 2, 5

    x = torch.randn(batch, length, 32)

    padding = torch.zeros(batch, length, dtype=torch.bool)
    padding[1, 3:] = True

    real = ~padding

    encoder = HistoryEncoder(config).eval()

    zero = torch.zeros(batch, length)

    cos, sin = rotary_tables(zero, config.head_dim, config.rope_base)

    with torch.no_grad():

        ours = rope(x, cos, sin, encoder.allowed_keys(padding))

        # Полярность обратная: у nn.Transformer True это запрет,
        # у SDPA — разрешение. Инверсия живёт в allowed_keys.
        reference = plain(x, src_key_padding_mask=padding)

    torch.testing.assert_close(ours[real], reference[real], rtol=1e-5, atol=1e-5)

    # --- поворот относителен --------------------------------

    coords = torch.tensor([[3.0, 2.0, 1.0, 0.0, 0.0], [5.0, 2.0, 0.0, 0.0, 0.0]])

    allowed = encoder.allowed_keys(padding)

    with torch.no_grad():
        rotated = rope(x, *rotary_tables(coords, config.head_dim), allowed)
        shifted = rope(x, *rotary_tables(coords + 7.5, config.head_dim), allowed)
        stretched = rope(x, *rotary_tables(coords * 2.0, config.head_dim), allowed)

    torch.testing.assert_close(rotated[real], shifted[real], rtol=1e-5, atol=1e-5)

    # Иначе поворот не нёс бы вообще никакой информации.
    assert (rotated[real] - stretched[real]).abs().max() > 1e-3

    # Fast path не мог обойти поворот: родительский forward не
    # вызывается вовсе, и при ненулевых координатах результат
    # обязан отличаться от обычного слоя.
    assert (rotated[real] - reference[real]).abs().max() > 1e-3

    # --- норма не меняется ----------------------------------

    vectors = torch.randn(batch, 4, length, config.head_dim)

    turned = apply_rotary(vectors, *rotary_tables(coords, config.head_dim))

    torch.testing.assert_close(vectors.norm(dim=-1), turned.norm(dim=-1), rtol=1e-5, atol=1e-5)


# ============================================================
# 2. ПРИЗНАКИ ВРЕМЕНИ
# ============================================================


def test_temporal_inputs_are_built():
    """
    Координата это часы до последнего элемента истории, а не
    возраст: у самого свежего элемента она ноль.
    """

    vocab = toy_vocab()

    batch, meta = synthetic_batch([[0, 10, 20], [0, 5]], cutoff_hours=24)

    history = prepare_history_batch(batch, meta, max_events=None)

    temporal = to_model_inputs(history, rope_config(vocab)).temporal

    # Пример 0: события в 0, 10, 20 ч при cutoff 24 ч.
    expected = squash_np([20.0, 10.0, 0.0, 5.0, 0.0]).astype(np.float32)

    torch.testing.assert_close(temporal.event_coords, torch.from_numpy(expected))

    # Простой это часы от последнего события до cutoff: у первого
    # примера 4 ч, у второго 19 ч.
    idle = (squash_np([4.0, 19.0]) / squash_np(INACTIVITY_NORM_HOURS)).astype(np.float32)

    torch.testing.assert_close(temporal.inactivity, torch.from_numpy(idle).reshape(2, 1))

    # --- календарь ------------------------------------------

    circle = temporal.calendar[:, 0::2] ** 2 + temporal.calendar[:, 1::2] ** 2

    torch.testing.assert_close(circle, torch.ones_like(circle), rtol=1e-6, atol=1e-6)

    # Понедельник 6 января 2025 года, 06:00.
    known = calendar_features(np.array([np.datetime64(datetime(2025, 1, 6, 6, 0), "us")]))

    assert known[0, 0] == pytest.approx(1.0, abs=1e-6)      # sin(2π·6/24)
    assert known[0, 1] == pytest.approx(0.0, abs=1e-6)      # cos того же
    assert known[0, 2] == pytest.approx(0.0, abs=1e-6)      # понедельник это ноль
    assert known[0, 3] == pytest.approx(1.0, abs=1e-6)
    assert known[0, 4] == pytest.approx(np.sin(2 * np.pi * 5 / 31), abs=1e-6)

    assert calendar_features(np.array([], dtype="datetime64[us]")).shape == (0, 6)

    # --- разнородная лента ---------------------------------

    items = list(session_examples(vocab).values())

    grouped = prepare_history_batch(
        collate(items),
        metadata_from_examples(items),
        None,
    )

    config = full_config(vocab)

    inputs = to_model_inputs(grouped, config)

    assert inputs.temporal.calendar.shape == (grouped.n_events, 6)

    backbone = build_backbone(config, seed=2)

    coords = backbone.history_coords(inputs)

    used = inputs.used_history_length

    for index in range(inputs.n_examples):

        row = coords[index, : int(used[index]) + 1]

        # Профиль стоит на cutoff, как и самый свежий элемент.
        assert float(row[0]) == 0.0

        # Ровно один элемент истории самый свежий.
        assert int((row[1:] == 0.0).sum()) == 1

        # Padding координат не получает.
        assert float(coords[index, int(used[index]) + 1 :].abs().sum()) == 0.0

    moved = inputs.temporal.to("cpu")

    torch.testing.assert_close(moved.event_coords, inputs.temporal.event_coords)


# ============================================================
# 3. BACKBONE
# ============================================================


@pytest.mark.parametrize("structure", tuple(CASES))
def test_rope_backbone_forward_backward(structure):
    """
    Полный проход: вход, голова, loss, backward.
    """

    vocab = toy_vocab()

    history, targets, table, config = CASES[structure](vocab)

    inputs = to_model_inputs(history, config)

    torch.manual_seed(11)

    backbone = build_backbone(config, seed=11)
    head = MLMHead(config, table)

    # Временного слагаемого нет вовсе: время живёт во внимании.
    assert not any(name.startswith("time.") for name in backbone.state_dict())

    # Нулевой последний слой: на старте признаки не прибавляют
    # ничего, и вход History Encoder это чистый вектор события.
    with torch.no_grad():
        assert float(backbone.calendar(inputs.temporal.calendar).abs().max()) == 0.0
        assert float(backbone.inactivity(inputs.temporal.inactivity).abs().max()) == 0.0

    tensors = targets.tensors()

    out = backbone(inputs, gather=targets.gather())

    assert bool(torch.isfinite(out.contextualized).all())

    # Padded позиции зануляются и здесь.
    assert float(out.contextualized.detach()[out.padding_mask].abs().max()) == 0.0

    local, event, user = representations(out, tensors)

    result = mlm_loss(head(local, event, user, tensors["field_ids"]), tensors["local_targets"])

    assert bool(torch.isfinite(result.field_balanced))

    result.field_balanced.backward()

    # Градиент доходит и до внимания, и до обоих новых признаков.
    for name, parameter in (
        ("history.layers.0.self_attn.in_proj_weight", backbone.history.layers[0].self_attn.in_proj_weight),
        ("calendar.output.weight", backbone.calendar.output.weight),
        ("inactivity.output.weight", backbone.inactivity.output.weight),
    ):
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().sum()) > 0.0, name

    # --- правило full это тот же путь -----------------------

    backbone.eval()

def test_rope_result_does_not_depend_on_batch_neighbours():
    """
    Сосед по batch и его padding не должны менять чужой вектор.
    """

    vocab = toy_vocab()

    config = rope_config(vocab)

    backbone = build_backbone(config, seed=4).eval()

    alone, _ = synthetic_batch([[0, 3, 9]], cutoff_hours=24)
    pair, _ = synthetic_batch([[0, 3, 9], [0, 1, 2, 5, 7, 11]], cutoff_hours=24)

    _, meta_alone = synthetic_batch([[0, 3, 9]], cutoff_hours=24)
    _, meta_pair = synthetic_batch([[0, 3, 9], [0, 1, 2, 5, 7, 11]], cutoff_hours=24)

    first = to_model_inputs(prepare_history_batch(alone, meta_alone, None), config)
    both = to_model_inputs(prepare_history_batch(pair, meta_pair, None), config)

    with torch.no_grad():
        single = backbone(first).client_embedding[0]
        together = backbone(both).client_embedding[0]

    torch.testing.assert_close(single, together, rtol=1e-5, atol=1e-5)


# ============================================================
# 4. КОНФИГУРАЦИЯ И CHECKPOINT
# ============================================================


def test_rope_base_travels_to_checkpoint(env, tmp_path):
    """
    База частот поворота это часть архитектуры: она обязана
    доехать до checkpoint и попасть в отчёт.
    """

    config = small_config(max_steps=2, eval_every=100)

    out = tmp_path / "run"

    report = run_training(env, config, out, device="cpu", quiet=True)

    assert report["model"]["rope_base"] == 10_000.0

    same = Trainer(config, env.tokenizer, env.table, env.unigram, "cpu")

    load_checkpoint(
        out / "last.pt",
        backbone=same.backbone,
        head=same.head,
        model_config=same.model_config.as_dict(),
        restore_random=False,
    )

    # Другая база это другая архитектура, и веса в неё не идут.
    other = Trainer(config, env.tokenizer, env.table, env.unigram, "cpu")

    changed = dict(other.model_config.as_dict())
    changed["rope_base"] = 5_000.0

    with pytest.raises(IncompatibleArtifactsError, match="rope_base"):
        load_checkpoint(
            out / "last.pt",
            backbone=other.backbone,
            head=other.head,
            model_config=changed,
            restore_random=False,
        )
