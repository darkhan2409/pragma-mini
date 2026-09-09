"""
Event и Profile Encoder: pooling, padding, общие таблицы,
независимые блоки, воспроизводимость.

Все сравнения через assert_close: fast path PyTorch и обычный
путь дают разные последние биты, побайтового равенства требовать
нельзя.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tokenizer.artifacts import Tokenizer
from src.tokenizer.config import EVT_ID, PAD_ID, USR_ID
from src.tokenizer.dataset import Record, TokenizedDataset, collate
from src.tokenizer.encode import PROFILE_WIDTH, encode_pairs
from src.model.batching import BatchError, events_from_batch, pad_records, profiles_from_batch
from src.model.config import ModelConfig, config_from_tokenizer
from src.model.encoders import build_encoders, encode_events, encode_profiles

from tests.test_tok_encode import toy_vocab


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA недоступна")

# Расхождение CPU и CUDA измерено на этой машине: 1.3e-4.
CROSS_DEVICE = {"rtol": 1e-3, "atol": 5e-4}


# ============================================================
# СИНТЕТИКА
# ============================================================


@pytest.fixture(scope="module")
def toy():
    return toy_vocab()


@pytest.fixture(scope="module")
def toy_config(toy):
    return ModelConfig(vocab_size=toy.size, max_position_embeddings=16, dropout=0.1)


def event(vocab, pairs) -> Record:
    return encode_pairs(vocab, pairs, EVT_ID)


def profile(vocab, pairs) -> Record:
    return encode_pairs(vocab, pairs, USR_ID)


def three_fields(vocab) -> Record:
    return event(vocab, [("toy__color", "red"), ("toy__size", 1), ("toy__flag", True)])


@pytest.fixture(scope="module")
def toy_pair(toy_config):
    return build_encoders(toy_config, seed=1).eval()


# ============================================================
# РЕАЛЬНЫЕ ДАННЫЕ
# ============================================================


@pytest.fixture(scope="module")
def real(tok_run):
    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    config = config_from_tokenizer(tokenizer)

    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    batch = collate([data.load(index) for index in range(3)])

    return {
        "tokenizer": tokenizer,
        "config": config,
        "batch": batch,
        "pair": build_encoders(config, seed=7).eval(),
    }


# ============================================================
# ФОРМЫ
# ============================================================


def test_output_shapes_on_real_data(real):
    with torch.no_grad():
        events = encode_events(real["pair"], real["batch"], microbatch_size=512)
        profiles = encode_profiles(real["pair"], real["batch"])

    assert events.shape == (real["batch"].n_events, real["config"].d_model)
    assert profiles.shape == (real["batch"].n_examples, real["config"].d_model)

    assert torch.isfinite(events).all()
    assert torch.isfinite(profiles).all()


def test_records_of_different_width_are_handled(real):
    records = events_from_batch(real["batch"], real["config"])

    widths = set(records.lengths.tolist())

    # В ленте есть события всех типов: от 5 до 11 токенов.
    assert min(widths) >= 5
    assert max(widths) <= 11
    assert len(widths) > 1

    profiles = profiles_from_batch(real["batch"], real["config"])

    assert profiles.max_length == PROFILE_WIDTH


def test_config_matches_the_tokenizer(real):
    assert real["config"].vocab_size == real["tokenizer"].vocab.size
    assert real["config"].pad_id == PAD_ID


# ============================================================
# PADDING НЕ ВЛИЯЕТ
# ============================================================


def test_padding_does_not_change_the_vector(toy, toy_config, toy_pair):
    """
    Одна и та же запись, посчитанная в одиночку и рядом с более
    длинным соседом, обязана дать один вектор.
    """

    single = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    padded = pad_records(
        [three_fields(toy), event(toy, [("toy__color", "blue"), ("toy__size", 0), ("toy__flag", False)])],
        lead_id=EVT_ID,
        config=toy_config,
    )

    short = pad_records(
        [event(toy, [("toy__color", "red")]), three_fields(toy)],
        lead_id=EVT_ID,
        config=toy_config,
    )

    with torch.no_grad():
        alone = toy_pair.encode_events(single)
        together = toy_pair.encode_events(padded)
        after_padding = toy_pair.encode_events(short)

    torch.testing.assert_close(alone[0], together[0])
    torch.testing.assert_close(alone[0], after_padding[1])


def test_neighbours_do_not_leak(toy, toy_config, toy_pair):
    first = three_fields(toy)

    left = pad_records([first, event(toy, [("toy__color", "blue")])], lead_id=EVT_ID, config=toy_config)
    right = pad_records([first, event(toy, [("toy__flag", False)])], lead_id=EVT_ID, config=toy_config)

    with torch.no_grad():
        a = toy_pair.encode_events(left)
        b = toy_pair.encode_events(right)

    torch.testing.assert_close(a[0], b[0])

    assert not torch.allclose(a[1], b[1])


# ============================================================
# POOLING
# ============================================================


def test_pooling_is_position_zero_after_final_norm(toy, toy_config, toy_pair):
    records = pad_records(
        [three_fields(toy), event(toy, [("toy__color", "blue")])],
        lead_id=EVT_ID,
        config=toy_config,
    )

    with torch.no_grad():

        out = toy_pair.event(
            records.key_ids, records.value_ids, records.positions, records.padding_mask, return_hidden=True
        )

        x = toy_pair.embeddings(
            records.key_ids, records.value_ids, records.positions, records.padding_mask
        )

        manual = toy_pair.event.final_norm(toy_pair.event.contextualize(x, records.padding_mask))

    torch.testing.assert_close(out.pooled, out.hidden[:, 0])
    torch.testing.assert_close(out.pooled, manual[:, 0])


def test_padded_hidden_states_are_zero(toy, toy_config, toy_pair):
    records = pad_records(
        [three_fields(toy), event(toy, [("toy__color", "blue")])],
        lead_id=EVT_ID,
        config=toy_config,
    )

    with torch.no_grad():
        out = toy_pair.event(
            records.key_ids, records.value_ids, records.positions, records.padding_mask, return_hidden=True
        )

    assert (out.hidden[records.padding_mask] == 0).all()

    # Без зануления LayerNorm дал бы здесь bias, а не ноль.
    assert out.hidden[0].abs().sum() > 0


def test_pooling_is_not_a_mean(toy, toy_config, toy_pair):
    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    with torch.no_grad():
        out = toy_pair.event(
            records.key_ids, records.value_ids, records.positions, records.padding_mask, return_hidden=True
        )

    assert not torch.allclose(out.pooled[0], out.hidden[0].mean(dim=0))


def test_profile_pools_from_usr(toy, toy_config, toy_pair):
    records = pad_records(
        [profile(toy, [("toy__color", "red"), ("toy__size", 0)])], lead_id=USR_ID, config=toy_config
    )

    with torch.no_grad():
        out = toy_pair.profile(
            records.key_ids, records.value_ids, records.positions, records.padding_mask, return_hidden=True
        )

    torch.testing.assert_close(out.pooled, out.hidden[:, 0])
    assert records.key_ids[0, 0].item() == USR_ID


# ============================================================
# ПОЗИЦИИ
# ============================================================


def test_content_is_bound_to_its_position(toy, toy_config, toy_pair):
    """
    Те же (key, value), но переставленные positions: результат
    обязан измениться.
    """

    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    swapped = records.positions.clone()
    swapped[0, 1], swapped[0, 2] = swapped[0, 2].clone(), swapped[0, 1].clone()

    with torch.no_grad():
        original = toy_pair.event(
            records.key_ids, records.value_ids, records.positions, records.padding_mask
        )
        changed = toy_pair.event(records.key_ids, records.value_ids, swapped, records.padding_mask)

    assert not torch.allclose(original, changed)


def test_permuting_whole_triples_keeps_the_vector(toy, toy_config, toy_pair):
    """
    Перестановка целых троек (key, value, position) при том же
    ведущем токене ничего не меняет: порядок несёт position, а
    не место в массиве.
    """

    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    order = torch.tensor([0, 3, 1, 2])

    with torch.no_grad():
        original = toy_pair.event(
            records.key_ids, records.value_ids, records.positions, records.padding_mask
        )
        permuted = toy_pair.event(
            records.key_ids[:, order],
            records.value_ids[:, order],
            records.positions[:, order],
            records.padding_mask[:, order],
        )

    torch.testing.assert_close(original, permuted)


def test_position_beyond_the_table_is_rejected_by_embeddings(toy, toy_config, toy_pair):
    """
    Проверка живёт и в самих embeddings: под python -O assert
    исчез бы, и таблица читалась бы за границей.
    """

    keys = torch.tensor([[EVT_ID, 6]], dtype=torch.long)
    values = torch.tensor([[EVT_ID, 9]], dtype=torch.long)
    positions = torch.tensor([[0, toy_config.max_position_embeddings]], dtype=torch.long)
    mask = torch.zeros(1, 2, dtype=torch.bool)

    with pytest.raises(BatchError, match="вне таблицы"):
        toy_pair.embeddings(keys, values, positions, mask)


def test_id_beyond_the_vocabulary_is_rejected_by_embeddings(toy_config, toy_pair):
    keys = torch.tensor([[EVT_ID, toy_config.vocab_size]], dtype=torch.long)
    values = torch.tensor([[EVT_ID, 9]], dtype=torch.long)
    positions = torch.tensor([[0, 1]], dtype=torch.long)
    mask = torch.zeros(1, 2, dtype=torch.bool)

    with pytest.raises(BatchError, match="вне словаря"):
        toy_pair.embeddings(keys, values, positions, mask)


def random_names(parameters: dict) -> list[str]:
    """
    Имена тензоров со случайной инициализацией.

    Биасы внимания и веса LayerNorm задаются константами, они
    совпадают у любых двух блоков по построению, а не потому,
    что блок скопирован. Сравнивать имеет смысл только то, что
    действительно разыгрывается.
    """

    return [name for name, tensor in parameters.items() if tensor.detach().unique().numel() > 2]


# ============================================================
# ОБЩИЕ ТАБЛИЦЫ, НЕЗАВИСИМЫЕ БЛОКИ
# ============================================================


def test_embeddings_are_shared(toy_pair):
    assert toy_pair.event.embeddings is toy_pair.profile.embeddings
    assert toy_pair.event.embeddings is toy_pair.embeddings

    assert toy_pair.event.embeddings.token.weight is toy_pair.profile.embeddings.token.weight
    assert toy_pair.event.embeddings.position.weight is toy_pair.profile.embeddings.position.weight


def test_state_dict_holds_one_copy_of_the_tables(toy_pair):
    keys = [name for name in toy_pair.state_dict() if "token.weight" in name or "position.weight" in name]

    assert keys == ["embeddings.token.weight", "embeddings.position.weight"]

    unique = {id(parameter) for parameter in toy_pair.parameters()}

    assert len(unique) == len(list(toy_pair.parameters()))


def test_transformer_parameters_are_independent(toy_pair):
    event_ids = {id(p) for p in toy_pair.event.layers.parameters()} | {
        id(p) for p in toy_pair.event.final_norm.parameters()
    }
    profile_ids = {id(p) for p in toy_pair.profile.layers.parameters()} | {
        id(p) for p in toy_pair.profile.final_norm.parameters()
    }

    assert not (event_ids & profile_ids)

    left = dict(toy_pair.event.layers.named_parameters())
    right = dict(toy_pair.profile.layers.named_parameters())

    assert set(left) == set(right)

    random = random_names(left)

    assert random, "не нашлось ни одного случайно инициализированного тензора"

    same = [name for name in random if torch.allclose(left[name], right[name])]

    assert not same, f"одинаковые начальные веса у блоков: {same}"


def test_layers_inside_one_encoder_differ(toy_pair):
    """
    Регресс на nn.TransformerEncoder: он копирует один слой, и
    все блоки стартуют одинаковыми.
    """

    first = dict(toy_pair.event.layers[0].named_parameters())
    second = dict(toy_pair.event.layers[1].named_parameters())

    random = random_names(first)

    assert {"self_attn.in_proj_weight", "linear1.weight", "linear2.weight"} <= set(random)

    same = [name for name in random if torch.allclose(first[name], second[name])]

    assert not same, f"слои одного энкодера совпадают: {same}"


def test_embedding_table_matches_the_vocabulary(real):
    table = real["pair"].embeddings.token

    assert table.num_embeddings == real["tokenizer"].vocab.size
    assert table.padding_idx == PAD_ID
    assert (table.weight[PAD_ID] == 0).all()


# ============================================================
# ГРАДИЕНТЫ
# ============================================================


def test_backward_produces_finite_gradients(toy, toy_config):
    pair = build_encoders(toy_config, seed=3).train()

    records = pad_records(
        [three_fields(toy), event(toy, [("toy__color", "blue")])],
        lead_id=EVT_ID,
        config=toy_config,
    )

    profiles = pad_records(
        [profile(toy, [("toy__color", "red"), ("toy__size", 1)])], lead_id=USR_ID, config=toy_config
    )

    loss = pair.encode_events(records).sum() + pair.encode_profiles(profiles).sum()

    loss.backward()

    touched = 0

    for name, parameter in pair.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        touched += int(parameter.grad.abs().sum() > 0)

    assert touched > 0

    assert (pair.embeddings.token.weight.grad[PAD_ID] == 0).all()


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================


def test_same_seed_gives_same_weights_and_outputs(toy, toy_config):
    left = build_encoders(toy_config, seed=11).eval()
    right = build_encoders(toy_config, seed=11).eval()

    for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters()):
        torch.testing.assert_close(a, b, msg=name)

    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    with torch.no_grad():
        torch.testing.assert_close(left.encode_events(records), right.encode_events(records))


def test_other_seed_gives_other_weights(toy, toy_config):
    left = build_encoders(toy_config, seed=11).eval()
    right = build_encoders(toy_config, seed=12).eval()

    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    with torch.no_grad():
        assert not torch.allclose(left.encode_events(records), right.encode_events(records))


def test_dropout_is_active_only_in_train(toy, toy_config):
    pair = build_encoders(toy_config, seed=5)

    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    pair.train()

    torch.manual_seed(0)
    first = pair.encode_events(records)
    second = pair.encode_events(records)

    assert not torch.allclose(first, second)

    pair.eval()

    with torch.no_grad():
        torch.testing.assert_close(pair.encode_events(records), pair.encode_events(records))


# ============================================================
# MICROBATCH
# ============================================================


def test_microbatch_matches_the_whole_batch(real):
    records = events_from_batch(real["batch"], real["config"])

    with torch.no_grad():
        whole = real["pair"].encode_events(records)
        pieces = real["pair"].encode_events(records, microbatch_size=7)

    assert pieces.shape == whole.shape

    torch.testing.assert_close(whole, pieces)


def test_microbatch_keeps_the_order(toy, toy_config, toy_pair):
    records = pad_records(
        [
            three_fields(toy),
            event(toy, [("toy__color", "blue")]),
            event(toy, [("toy__size", 0), ("toy__flag", False)]),
            event(toy, [("toy__flag", True)]),
        ],
        lead_id=EVT_ID,
        config=toy_config,
    )

    with torch.no_grad():
        whole = toy_pair.encode_events(records)
        pieces = toy_pair.encode_events(records, microbatch_size=1)

    for index in range(len(records)):
        torch.testing.assert_close(whole[index], pieces[index])


def test_microbatch_must_be_positive(toy, toy_config, toy_pair):
    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    with pytest.raises(ValueError, match="положительным"):
        toy_pair.encode_events(records, microbatch_size=0)


# ============================================================
# КОНФИГ И ОШИБКИ
# ============================================================


def test_head_count_must_divide_the_width():
    with pytest.raises(ValueError, match="не делится"):
        ModelConfig(vocab_size=100, d_model=65, n_heads=4)


def test_tiny_vocabulary_is_rejected():
    with pytest.raises(ValueError, match="special"):
        ModelConfig(vocab_size=3)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="activation"):
        ModelConfig(vocab_size=100, activation="swish")


def test_wrong_lead_is_rejected_by_the_encoder(toy, toy_config, toy_pair):
    records = pad_records(
        [profile(toy, [("toy__color", "red")])], lead_id=USR_ID, config=toy_config
    )

    with pytest.raises(BatchError, match="ведущего токена"):
        toy_pair.encode_events(records)


def test_fully_padded_row_is_rejected(toy, toy_config, toy_pair):
    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config)

    mask = torch.ones_like(records.padding_mask)

    with pytest.raises(BatchError, match="целиком из padding"):
        toy_pair.event(records.key_ids, records.value_ids, records.positions, mask)


# ============================================================
# CUDA
# ============================================================


@CUDA
def test_cuda_matches_cpu(toy, toy_config):
    records = pad_records(
        [three_fields(toy), event(toy, [("toy__color", "blue")])],
        lead_id=EVT_ID,
        config=toy_config,
    )

    on_cpu = build_encoders(toy_config, seed=21, device="cpu").eval()
    on_gpu = build_encoders(toy_config, seed=21, device="cuda").eval()

    # Сборка всегда на CPU, поэтому веса совпадают.
    for (name, a), (_, b) in zip(on_cpu.named_parameters(), on_gpu.named_parameters()):
        torch.testing.assert_close(a, b.cpu(), msg=name)

    with torch.no_grad():
        cpu_out = on_cpu.encode_events(records)
        gpu_out = on_gpu.encode_events(records.to("cuda"))

    torch.testing.assert_close(cpu_out, gpu_out.cpu(), **CROSS_DEVICE)


@CUDA
def test_cuda_repeats_itself(toy, toy_config):
    records = pad_records([three_fields(toy)], lead_id=EVT_ID, config=toy_config).to("cuda")

    pair = build_encoders(toy_config, seed=21, device="cuda").eval()

    with torch.no_grad():
        torch.testing.assert_close(pair.encode_events(records), pair.encode_events(records))
