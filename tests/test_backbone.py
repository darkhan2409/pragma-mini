"""
Backbone: три энкодера в один проход.

Проверяется и склейка последовательности, и то, что время и
содержимое действительно влияют на результат.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tokenizer.artifacts import Tokenizer
from src.tokenizer.config import EVT_ID, MASK_ID, PAD_ID, USR_ID
from src.tokenizer.dataset import TokenizedDataset, collate
from src.model.batching import BatchError
from src.model.config import ModelConfig, config_from_tokenizer
from src.model.history_batching import (
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
)
from src.model.backbone import build_backbone
from src.model.history_encoder import HistoryEncoder
from src.model.time_encoding import sinusoidal_positions

from tests.test_history_batching import build_example, synthetic_batch, toy_config
from tests.test_model_encoders import random_names
from tests.test_tok_encode import toy_vocab


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA недоступна")

# Допуск задаётся Event и Profile Encoder: они идут fast path,
# и там измерено расхождение CPU и CUDA 1.3e-4. Сам History
# Encoder на обычном пути сходится до ~1e-6.
CROSS_DEVICE = {"rtol": 1e-3, "atol": 5e-4}


# ============================================================
# ХЕЛПЕРЫ
# ============================================================


@pytest.fixture(scope="module")
def config():
    return toy_config()


@pytest.fixture(scope="module")
def backbone(config):
    return build_backbone(config, seed=1).eval()


def prepared(histories, cutoff_hours, max_events, config, device=None):
    batch, meta = synthetic_batch(histories, cutoff_hours)

    history = prepare_history_batch(batch, meta, max_events)

    return to_model_inputs(history, config, device)


def run(backbone, inputs):
    with torch.no_grad():
        return backbone(inputs)


# ============================================================
# ФОРМЫ
# ============================================================


def test_output_shapes(config, backbone):
    inputs = prepared([[0, 1, 2, 3], [0, 1]], 24, 10, config)

    out = run(backbone, inputs)

    assert out.client_embedding.shape == (2, config.d_model)
    assert out.contextualized.shape == (2, 1 + 4, config.d_model)
    assert out.padding_mask.shape == (2, 5)
    assert out.event_embeddings.shape == (6, config.d_model)

    assert out.event_example.tolist() == [0, 0, 0, 0, 1, 1]
    assert out.event_slot.tolist() == [1, 2, 3, 4, 1, 2]
    assert out.kept_events.tolist() == list(range(6))


def test_padding_mask_marks_the_tail(config, backbone):
    inputs = prepared([[0, 1, 2, 3], [0, 1]], 24, 10, config)

    out = run(backbone, inputs)

    assert out.padding_mask[0].tolist() == [False] * 5
    assert out.padding_mask[1].tolist() == [False, False, False, True, True]


def test_padded_rows_are_zero(config, backbone):
    inputs = prepared([[0, 1, 2, 3], [0]], 24, 10, config)

    out = run(backbone, inputs)

    assert (out.contextualized[out.padding_mask] == 0).all()
    assert out.contextualized[0].abs().sum() > 0


def test_truncation_shows_up_in_the_shapes(config, backbone):
    inputs = prepared([[0, 1, 2, 3, 4, 5]], 24, 2, config)

    out = run(backbone, inputs)

    assert out.contextualized.shape[1] == 3
    assert out.event_embeddings.shape[0] == 2
    assert out.kept_events.tolist() == [4, 5]


# ============================================================
# POOLING ИЗ ПРОФИЛЯ
# ============================================================


def test_client_embedding_is_position_zero(config, backbone):
    inputs = prepared([[0, 1, 2]], 24, 10, config)

    out = run(backbone, inputs)

    torch.testing.assert_close(out.client_embedding, out.contextualized[:, 0])


def test_profile_sits_at_position_zero_without_a_time_term(config, backbone):
    """
    Первая позиция это выход Profile Encoder плюс позиционное
    кодирование. Временного слагаемого у неё нет.
    """

    inputs = prepared([[0, 1, 2]], 24, 10, config)

    with torch.no_grad():

        events = backbone.pair.encode_events(inputs.events)
        profiles = backbone.pair.encode_profiles(inputs.profiles)

        x, mask = backbone.assemble(inputs, events, profiles)

    expected = profiles + sinusoidal_positions(x.shape[1], config.d_model)[0]

    torch.testing.assert_close(x[:, 0], expected)


def test_usr_is_not_re_embedded(config, backbone):
    """
    Профиль входит уже как вектор Profile Encoder, а не как
    токен [USR] из таблицы.
    """

    inputs = prepared([[0, 1]], 24, 10, config)

    with torch.no_grad():

        profiles = backbone.pair.encode_profiles(inputs.profiles)
        events = backbone.pair.encode_events(inputs.events)

        x, _ = backbone.assemble(inputs, events, profiles)

        token = backbone.pair.embeddings.token.weight[USR_ID]

    assert not torch.allclose(x[0, 0], token)
    torch.testing.assert_close(x[0, 0] - sinusoidal_positions(x.shape[1], config.d_model)[0], profiles[0])


# ============================================================
# ВРЕМЯ ВЛИЯЕТ
# ============================================================


def test_age_changes_the_output(config, backbone):
    """
    Те же интервалы между событиями, но история кончилась
    сутки назад: клиент обязан выглядеть иначе.
    """

    recent = prepared([[20, 21, 22]], 24, 10, config)
    stale = prepared([[0, 1, 2]], 24, 10, config)

    a = run(backbone, recent)
    b = run(backbone, stale)

    assert not torch.allclose(a.client_embedding, b.client_embedding)


def test_gap_changes_the_output(config, backbone):
    dense = prepared([[0, 1, 2]], 24, 10, config)
    sparse = prepared([[0, 1, 20]], 24, 10, config)

    a = run(backbone, dense)
    b = run(backbone, sparse)

    assert not torch.allclose(a.client_embedding, b.client_embedding)


def test_time_projection_is_actually_used(config, backbone):
    inputs = prepared([[0, 5, 10]], 24, 10, config)

    with torch.no_grad():

        events = backbone.pair.encode_events(inputs.events)
        profiles = backbone.pair.encode_profiles(inputs.profiles)

        x, _ = backbone.assemble(inputs, events, profiles)

        zeroed = replace(inputs, time_hours=torch.zeros_like(inputs.time_hours))

        y, _ = backbone.assemble(zeroed, events, profiles)

    assert not torch.allclose(x[:, 1:], y[:, 1:])
    torch.testing.assert_close(x[:, 0], y[:, 0])


# ============================================================
# СОДЕРЖИМОЕ ВЛИЯЕТ
# ============================================================


def test_content_changes_the_output_at_fixed_times(config, backbone):
    """
    Те же timestamps, другое содержимое событий.
    """

    vocab = toy_vocab()

    left = collate([build_example(vocab, 0, [0, 1, 2], 24, colors=["red", "red", "red"])])
    right = collate([build_example(vocab, 0, [0, 1, 2], 24, colors=["blue", "blue", "blue"])])

    meta = metadata_from_examples([build_example(vocab, 0, [0, 1, 2], 24)])

    a = run(backbone, to_model_inputs(prepare_history_batch(left, meta, 10), config))
    b = run(backbone, to_model_inputs(prepare_history_batch(right, meta, 10), config))

    assert not torch.allclose(a.client_embedding, b.client_embedding)


def test_order_of_content_matters(config, backbone):
    vocab = toy_vocab()

    meta = metadata_from_examples([build_example(vocab, 0, [0, 1, 2], 24)])

    first = collate([build_example(vocab, 0, [0, 1, 2], 24, colors=["red", "blue", "blue"])])
    second = collate([build_example(vocab, 0, [0, 1, 2], 24, colors=["blue", "blue", "red"])])

    a = run(backbone, to_model_inputs(prepare_history_batch(first, meta, 10), config))
    b = run(backbone, to_model_inputs(prepare_history_batch(second, meta, 10), config))

    assert not torch.allclose(a.client_embedding, b.client_embedding)


# ============================================================
# СОСЕДИ И PADDING
# ============================================================


def test_neighbours_do_not_change_the_result(config, backbone):
    alone = prepared([[0, 1, 2]], 24, 10, config)
    crowded = prepared([[0, 1, 2], [0, 1, 2, 3, 4, 5, 6]], 24, 10, config)

    a = run(backbone, alone)
    b = run(backbone, crowded)

    torch.testing.assert_close(a.client_embedding[0], b.client_embedding[0])
    torch.testing.assert_close(a.event_embeddings, b.event_embeddings[:3])


# ============================================================
# ПОВТОРНОЕ СОБЫТИЕ ПОД РАЗНЫМИ МАСКАМИ
# ============================================================


def test_repeated_event_under_different_masks_differs(config, backbone):
    """
    Одно и то же событие входит дважды. Если у одного вхождения
    значение замаскировано, их векторы обязаны разойтись: кэш
    embeddings здесь был бы ошибкой.
    """

    vocab = toy_vocab()

    batch = collate([build_example(vocab, 0, [0, 1, 2], 24, colors=["red", "red", "red"])])
    meta = metadata_from_examples([build_example(vocab, 0, [0, 1, 2], 24)])

    history = prepare_history_batch(batch, meta, 10)

    values = np.asarray(history.tokens.value_ids).copy()

    # Второе событие: маскируем значение поля.
    offsets = np.asarray(history.tokens.event_offsets)
    values[offsets[1] + 1] = MASK_ID

    masked = replace(history, tokens=replace(history.tokens, value_ids=values))

    inputs = to_model_inputs(masked, config)

    with torch.no_grad():
        vectors = backbone.pair.encode_events(inputs.events)

    # Одинаковые токены дают одинаковый вектор, замаскированное
    # вхождение расходится: кэш по содержимому был бы неверен.
    torch.testing.assert_close(vectors[0], vectors[2])
    assert not torch.allclose(vectors[0], vectors[1])

    # После History Encoder расходятся и незамаскированные:
    # у них разные позиция в истории и время.
    out = run(backbone, inputs)

    assert not torch.allclose(out.event_embeddings[0], out.event_embeddings[2])


# ============================================================
# СЛОИ
# ============================================================


def test_history_has_its_own_layers(backbone, config):
    assert len(backbone.history.layers) == config.n_history_layers == 2

    history_ids = {id(p) for p in backbone.history.parameters()}
    event_ids = {id(p) for p in backbone.pair.event.layers.parameters()}
    profile_ids = {id(p) for p in backbone.pair.profile.layers.parameters()}

    assert not (history_ids & event_ids)
    assert not (history_ids & profile_ids)


def test_history_layers_are_independently_initialised(backbone):
    first = dict(backbone.history.layers[0].named_parameters())
    second = dict(backbone.history.layers[1].named_parameters())

    names = random_names(first)

    assert {"self_attn.in_proj_weight", "linear1.weight", "linear2.weight"} <= set(names)

    same = [name for name in names if torch.allclose(first[name], second[name])]

    assert not same, f"слои History Encoder совпадают: {same}"


def test_history_differs_from_event_layers(backbone):
    left = dict(backbone.history.layers.named_parameters())
    right = dict(backbone.pair.event.layers.named_parameters())

    names = random_names(left)

    same = [name for name in names if torch.allclose(left[name], right[name])]

    assert not same


# ============================================================
# FAST PATH
# ============================================================


def test_history_layers_carry_a_hook_and_event_layers_do_not(backbone):
    for layer in backbone.history.layers:
        assert layer._forward_hooks, "у слоя History Encoder нет хука, fast path не отключён"

    for layer in backbone.pair.event.layers:
        assert not layer._forward_hooks


def test_global_fastpath_flag_is_untouched(config, backbone):
    before = torch.backends.mha.get_fastpath_enabled()

    run(backbone, prepared([[0, 1, 2]], 24, 10, config))

    assert torch.backends.mha.get_fastpath_enabled() == before


@CUDA
def test_long_history_stays_within_memory(config):
    """
    Регресс на материализацию матрицы внимания: с fast path
    здесь было бы больше трёх гигабайт.
    """

    encoder = HistoryEncoder(config).cuda().eval()

    x = torch.randn(4, 3637, config.d_model, device="cuda")
    mask = torch.zeros(4, 3637, dtype=torch.bool, device="cuda")
    mask[:, 3000:] = True

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        out = encoder(x, mask)

    peak = torch.cuda.max_memory_allocated() / (1 << 20)

    assert torch.isfinite(out[~mask]).all()
    assert peak < 200, f"пик {peak:.0f} МБ: похоже, fast path не отключён"


# ============================================================
# ГРАДИЕНТЫ
# ============================================================


def test_gradients_reach_every_encoder(config):
    backbone = build_backbone(config, seed=3).train()

    inputs = prepared([[0, 1, 2], [0, 5]], 24, 10, config)

    out = backbone(inputs)

    loss = out.client_embedding.sum() + out.event_embeddings.sum()

    loss.backward()

    checked = {
        "embeddings": backbone.pair.embeddings.token.weight,
        "positions": backbone.pair.embeddings.position.weight,
        "event": backbone.pair.event.layers[0].linear1.weight,
        "profile": backbone.pair.profile.layers[0].linear1.weight,
        "history": backbone.history.layers[0].linear1.weight,
        "time": backbone.time.linear.weight,
    }

    for name, parameter in checked.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name

    assert (backbone.pair.embeddings.token.weight.grad[PAD_ID] == 0).all()


def test_outputs_are_finite(config, backbone):
    out = run(backbone, prepared([[0, 1, 2, 8, 100]], 200, 10, config))

    assert torch.isfinite(out.contextualized).all()
    assert torch.isfinite(out.client_embedding).all()
    assert torch.isfinite(out.event_embeddings).all()


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================


def test_same_seed_reproduces(config):
    left = build_backbone(config, seed=11).eval()
    right = build_backbone(config, seed=11).eval()

    inputs = prepared([[0, 1, 2]], 24, 10, config)

    torch.testing.assert_close(run(left, inputs).client_embedding, run(right, inputs).client_embedding)


def test_other_seed_differs(config):
    left = build_backbone(config, seed=11).eval()
    right = build_backbone(config, seed=12).eval()

    inputs = prepared([[0, 1, 2]], 24, 10, config)

    assert not torch.allclose(run(left, inputs).client_embedding, run(right, inputs).client_embedding)


def test_eval_repeats_itself(config, backbone):
    inputs = prepared([[0, 1, 2]], 24, 10, config)

    torch.testing.assert_close(run(backbone, inputs).client_embedding, run(backbone, inputs).client_embedding)


def test_microbatch_does_not_change_the_result(config, backbone):
    inputs = prepared([[0, 1, 2, 3, 4, 5]], 24, 10, config)

    with torch.no_grad():
        whole = backbone(inputs, event_microbatch=None)
        pieces = backbone(inputs, event_microbatch=2)

    torch.testing.assert_close(whole.client_embedding, pieces.client_embedding)
    torch.testing.assert_close(whole.event_embeddings, pieces.event_embeddings)


# ============================================================
# ОШИБКИ
# ============================================================


def test_history_rejects_a_wrong_mask(config, backbone):
    with pytest.raises(BatchError, match="не совпадает"):
        backbone.history(torch.zeros(2, 3, config.d_model), torch.zeros(2, 4, dtype=torch.bool))


def test_history_rejects_a_fully_padded_example(config, backbone):
    with pytest.raises(BatchError, match="целиком из padding"):
        backbone.history(torch.zeros(1, 3, config.d_model), torch.ones(1, 3, dtype=torch.bool))


# ============================================================
# CUDA
# ============================================================


@CUDA
def test_cuda_matches_cpu(config):
    inputs = prepared([[0, 1, 2, 3], [0, 5]], 24, 10, config)

    on_cpu = build_backbone(config, seed=21, device="cpu").eval()
    on_gpu = build_backbone(config, seed=21, device="cuda").eval()

    with torch.no_grad():
        a = on_cpu(inputs)
        b = on_gpu(inputs.to("cuda"))

    torch.testing.assert_close(a.client_embedding, b.client_embedding.cpu(), **CROSS_DEVICE)


@CUDA
def test_history_encoder_alone_matches_closely(config):
    torch.manual_seed(5)

    encoder = HistoryEncoder(config).eval()

    x = torch.randn(3, 200, config.d_model)
    mask = torch.zeros(3, 200, dtype=torch.bool)
    mask[:, 150:] = True

    with torch.no_grad():
        a = encoder(x, mask)
        b = encoder.cuda()(x.cuda(), mask.cuda()).cpu()

    torch.testing.assert_close(a, b)


# ============================================================
# РЕАЛЬНЫЕ ДАННЫЕ
# ============================================================


def test_real_batch_goes_through(tok_run):
    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    config = config_from_tokenizer(tokenizer)

    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    examples = [data.load(index) for index in range(4)]

    history = prepare_history_batch(collate(examples), metadata_from_examples(examples), 50)

    inputs = to_model_inputs(history, config)

    backbone = build_backbone(config, seed=7).eval()

    out = run(backbone, inputs)

    assert out.client_embedding.shape == (4, config.d_model)
    assert out.contextualized.shape[1] <= 51
    assert int(history.info.used_history_length.max()) <= 50
    assert torch.isfinite(out.contextualized).all()
