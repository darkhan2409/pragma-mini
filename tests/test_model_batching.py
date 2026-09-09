"""
Адаптер входа: плоский TokenBatch в прямоугольный batch модели.

Ошибка формата обязана называть запись и причину, а не всплывать
индексом за границей внутри слоя.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tokenizer.config import EVT_ID, PAD_ID, USR_ID
from src.tokenizer.dataset import Record, TokenBatch, TokenizedDataset, collate
from src.tokenizer.encode import PROFILE_WIDTH, encode_pairs
from src.model.batching import (
    BatchError,
    check_batch,
    events_from_batch,
    pad_records,
    profiles_from_batch,
    split_events,
    split_profiles,
)
from src.model.config import ModelConfig

from tests.test_tok_encode import toy_vocab


# ============================================================
# ХЕЛПЕРЫ
# ============================================================


@pytest.fixture(scope="module")
def toy():
    return toy_vocab()


@pytest.fixture(scope="module")
def toy_config(toy):
    return ModelConfig(vocab_size=toy.size, max_position_embeddings=16)


def event(vocab, pairs) -> Record:
    return encode_pairs(vocab, pairs, EVT_ID)


def profile(vocab, pairs) -> Record:
    return encode_pairs(vocab, pairs, USR_ID)


@pytest.fixture(scope="module")
def real_batch(tok_run) -> TokenBatch:
    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    return collate([data.load(index) for index in range(4)])


@pytest.fixture(scope="module")
def real_config(tok_run):
    from src.model.config import config_from_tokenizer
    from src.tokenizer.artifacts import Tokenizer

    return config_from_tokenizer(Tokenizer.load(tok_run["vocab"], tok_run["artifacts"]))


# ============================================================
# НАРЕЗКА
# ============================================================


def test_events_are_split_by_offsets(real_batch):
    records = split_events(real_batch)

    assert len(records) == real_batch.n_events

    widths = np.diff(real_batch.event_offsets)

    assert [len(record.key_ids) for record in records] == list(widths)


def test_profiles_are_split_by_example(real_batch):
    records = split_profiles(real_batch)

    assert len(records) == real_batch.n_examples
    assert {len(record.key_ids) for record in records} == {PROFILE_WIDTH}


def test_every_event_keeps_its_lead(real_batch):
    for record in split_events(real_batch):
        assert record.key_ids[0] == EVT_ID
        assert record.value_ids[0] == EVT_ID
        assert record.positions[0] == 0


def test_broken_offsets_are_caught(real_batch):
    broken = TokenBatch(**{**real_batch.__dict__, "event_offsets": real_batch.event_offsets[:-1]})

    with pytest.raises(BatchError, match="event_offsets"):
        check_batch(broken)


def test_mismatched_flat_arrays_are_caught(real_batch):
    broken = TokenBatch(**{**real_batch.__dict__, "positions": real_batch.positions[:-1]})

    with pytest.raises(BatchError, match="разной длины"):
        check_batch(broken)


def test_profile_arrays_must_agree(real_batch):
    broken = TokenBatch(
        **{**real_batch.__dict__, "profile_positions": real_batch.profile_positions[:-1]}
    )

    with pytest.raises(BatchError, match="профиля разной длины"):
        check_batch(broken)


def test_empty_batch_is_rejected(real_batch):
    broken = TokenBatch(**{**real_batch.__dict__, "n_examples": 0})

    with pytest.raises(BatchError, match="пустой TokenBatch"):
        check_batch(broken)


# ============================================================
# ВЫРАВНИВАНИЕ
# ============================================================


def test_padding_fills_the_tail_only(toy, toy_config):
    records = [
        event(toy, [("toy__color", "red"), ("toy__size", 1), ("toy__flag", True)]),
        event(toy, [("toy__color", "blue")]),
    ]

    padded = pad_records(records, lead_id=EVT_ID, config=toy_config)

    assert padded.key_ids.shape == (2, 4)
    assert padded.lengths.tolist() == [4, 2]

    assert padded.padding_mask.tolist() == [
        [False, False, False, False],
        [False, False, True, True],
    ]

    assert padded.key_ids[1, 2:].tolist() == [PAD_ID, PAD_ID]
    assert padded.value_ids[1, 2:].tolist() == [PAD_ID, PAD_ID]
    assert padded.positions[1, 2:].tolist() == [0, 0]


def test_tensors_have_model_dtypes(toy, toy_config):
    padded = pad_records([event(toy, [("toy__color", "red")])], lead_id=EVT_ID, config=toy_config)

    assert padded.key_ids.dtype == torch.long
    assert padded.value_ids.dtype == torch.long
    assert padded.positions.dtype == torch.long
    assert padded.padding_mask.dtype == torch.bool
    assert padded.lengths.dtype == torch.long


def test_slice_trims_width_to_its_own_records(toy, toy_config):
    records = [
        event(toy, [("toy__color", "red"), ("toy__size", 1), ("toy__flag", True)]),
        event(toy, [("toy__color", "blue")]),
    ]

    padded = pad_records(records, lead_id=EVT_ID, config=toy_config)

    piece = padded.slice(1, 2)

    assert piece.key_ids.shape == (1, 2)
    assert not piece.padding_mask.any()


def test_to_moves_every_tensor(toy, toy_config):
    padded = pad_records([event(toy, [("toy__color", "red")])], lead_id=EVT_ID, config=toy_config)

    moved = padded.to("cpu")

    for name in ("key_ids", "value_ids", "positions", "padding_mask", "lengths"):
        assert getattr(moved, name).device.type == "cpu"


def test_batch_helpers_produce_expected_counts(real_batch, real_config):
    events = events_from_batch(real_batch, real_config)
    profiles = profiles_from_batch(real_batch, real_config)

    assert len(events) == real_batch.n_events
    assert len(profiles) == real_batch.n_examples
    assert profiles.max_length == PROFILE_WIDTH


# ============================================================
# ОШИБКИ ФОРМАТА
# ============================================================


def test_arrays_of_different_length_are_caught(toy_config):
    """
    Record сам следит за длинами, поэтому рассогласование
    подсовывается в обход конструктора: адаптер обязан ловить
    и такое, а не полагаться на чужую проверку.
    """

    record = Record(
        key_ids=np.array([EVT_ID, 6], np.int32),
        value_ids=np.array([EVT_ID, 9], np.int32),
        positions=np.array([0, 1], np.int16),
    )

    object.__setattr__(record, "value_ids", np.array([EVT_ID], np.int32))

    with pytest.raises(BatchError, match="разной длины"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)


def test_empty_record_is_caught(toy_config):
    record = Record(
        key_ids=np.array([], np.int32),
        value_ids=np.array([], np.int32),
        positions=np.array([], np.int16),
    )


    with pytest.raises(BatchError, match="пустая"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)


def test_empty_list_is_caught(toy_config):
    with pytest.raises(BatchError, match="нечего кодировать"):
        pad_records([], lead_id=EVT_ID, config=toy_config)


def test_pad_inside_content_is_caught(toy_config):
    record = Record(
        key_ids=np.array([EVT_ID, PAD_ID], np.int32),
        value_ids=np.array([EVT_ID, 9], np.int32),
        positions=np.array([0, 1], np.int16),
    )

    with pytest.raises(BatchError, match=r"\[PAD\]"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)


def test_id_outside_the_vocabulary_is_caught(toy_config):
    record = Record(
        key_ids=np.array([EVT_ID, 6], np.int32),
        value_ids=np.array([EVT_ID, toy_config.vocab_size], np.int32),
        positions=np.array([0, 1], np.int16),
    )

    with pytest.raises(BatchError, match="выходит за словарь"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)


def test_negative_id_is_caught(toy_config):
    record = Record(
        key_ids=np.array([EVT_ID, -1], np.int32),
        value_ids=np.array([EVT_ID, 9], np.int32),
        positions=np.array([0, 1], np.int16),
    )

    with pytest.raises(BatchError, match="выходит за словарь"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)


def test_position_beyond_the_table_is_caught(toy_config):
    record = Record(
        key_ids=np.array([EVT_ID, 6], np.int32),
        value_ids=np.array([EVT_ID, 9], np.int32),
        positions=np.array([0, toy_config.max_position_embeddings], np.int16),
    )

    with pytest.raises(BatchError, match="не помещается в таблицу позиций"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)


def test_wrong_lead_for_events_is_caught(toy, toy_config):
    record = profile(toy, [("toy__color", "red")])

    with pytest.raises(BatchError, match="ведущий токен"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)


def test_wrong_lead_for_profile_is_caught(toy, toy_config):
    record = event(toy, [("toy__color", "red")])

    with pytest.raises(BatchError, match="ведущий токен"):
        pad_records([record], lead_id=USR_ID, config=toy_config)


def test_lead_not_at_position_zero_is_caught(toy_config):
    record = Record(
        key_ids=np.array([EVT_ID, 6], np.int32),
        value_ids=np.array([EVT_ID, 9], np.int32),
        positions=np.array([1, 2], np.int16),
    )

    with pytest.raises(BatchError, match="на позиции"):
        pad_records([record], lead_id=EVT_ID, config=toy_config)
