"""
Подготовка истории: metadata, проверка порядка, обрезка recent.

Обрезка обязана строить корректный TokenBatch: на нём потом
работают и masker, и адаптер модели.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from src.tokenizer.config import EVT_ID, MASK_ID, USR_ID
from src.tokenizer.dataset import Events, Example, Record, TokenizedDataset, collate
from src.tokenizer.encode import encode_pairs
from src.tokenizer.masking import Masker, MaskingConfig
from src.model.batching import BatchError, check_batch
from src.model.config import ModelConfig
from src.model.history_batching import (
    HistoryMeta,
    example_lengths,
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
    truncate_recent,
    validate_history_batch,
)

from tests.test_tok_encode import toy_vocab


BASE = datetime(2025, 1, 1)

COLORS = ("red", "blue")


# ============================================================
# СИНТЕТИЧЕСКАЯ ИСТОРИЯ
# ============================================================


def toy_config(vocab=None) -> ModelConfig:
    vocab = vocab or toy_vocab()
    return ModelConfig(vocab_size=vocab.size, max_position_embeddings=16)


def build_example(
    vocab,
    client_id: int,
    hours: list[float],
    cutoff_hours: float,
    colors: list[str] | None = None,
) -> Example:
    """
    Пример с событиями в заданные часы от базы.
    """

    colors = colors or [COLORS[index % len(COLORS)] for index in range(len(hours))]

    records = [encode_pairs(vocab, [("toy__color", color)], EVT_ID) for color in colors]

    widths = np.array([len(record.key_ids) for record in records], dtype=np.int64)

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    events = Events(
        key_ids=np.concatenate([record.key_ids for record in records]),
        value_ids=np.concatenate([record.value_ids for record in records]),
        positions=np.concatenate([record.positions for record in records]),
        offsets=offsets,
        event_type=np.array(["toy"] * len(records), dtype=object),
        ts=np.array([np.datetime64(BASE + timedelta(hours=h), "us") for h in hours]),
        seq=np.arange(len(records), dtype=np.int64),
    )

    # Ключи профиля в реальном словаре никогда не predictable:
    # берём toy__flag, иначе masker справедливо ругается.
    profile = encode_pairs(vocab, [("toy__flag", True)], USR_ID)

    return Example(
        client_id=client_id,
        cutoff=BASE + timedelta(hours=cutoff_hours),
        dataset="toy",
        client_group="train",
        seq_end=len(records),
        snapshot_ts=BASE - timedelta(hours=1),
        profile=profile,
        events=events,
    )


def synthetic_batch(histories: list[list[float]], cutoff_hours: float, sort: bool = True):
    """
    TokenBatch и metadata из списка историй, заданных часами.
    """

    vocab = toy_vocab()

    examples = [
        build_example(vocab, index, sorted(hours) if sort else hours, cutoff_hours)
        for index, hours in enumerate(histories)
    ]

    return collate(examples), metadata_from_examples(examples)


# ============================================================
# METADATA
# ============================================================


def test_metadata_follows_the_example_order():
    vocab = toy_vocab()

    examples = [
        build_example(vocab, 7, [0, 1], 10),
        build_example(vocab, 3, [0], 20),
    ]

    meta = metadata_from_examples(examples)

    assert meta.client_ids.tolist() == [7, 3]
    assert meta.cutoffs.dtype == np.dtype("datetime64[us]")
    assert meta.snapshot_ts.dtype == np.dtype("datetime64[us]")
    assert len(meta) == 2


def test_metadata_needs_examples():
    with pytest.raises(BatchError, match="нет примеров"):
        metadata_from_examples([])


def test_metadata_from_real_examples(tok_run):
    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    examples = [data.load(index) for index in range(3)]

    meta = metadata_from_examples(examples)

    assert meta.client_ids.tolist() == [example.client_id for example in examples]
    assert (meta.snapshot_ts < meta.cutoffs).all()


# ============================================================
# ПРОВЕРКА
# ============================================================


def test_clean_batch_passes():
    batch, meta = synthetic_batch([[0, 1, 2], [0, 5]], cutoff_hours=24)

    validate_history_batch(batch, meta)


def test_equal_timestamps_with_growing_seq_pass():
    batch, meta = synthetic_batch([[0, 1, 1, 2]], cutoff_hours=24)

    validate_history_batch(batch, meta)


def test_backwards_timestamps_are_caught():
    batch, meta = synthetic_batch([[0, 5, 2]], cutoff_hours=24, sort=False)

    with pytest.raises(BatchError, match="не упорядочена"):
        validate_history_batch(batch, meta)


def test_equal_timestamps_with_falling_seq_are_caught():
    batch, meta = synthetic_batch([[0, 1, 1]], cutoff_hours=24)

    seq = np.asarray(batch.seq).copy()
    seq[1], seq[2] = seq[2], seq[1]

    with pytest.raises(BatchError, match="не упорядочена"):
        validate_history_batch(replace(batch, seq=seq), meta)


def test_event_at_or_after_cutoff_is_caught():
    batch, meta = synthetic_batch([[0, 5]], cutoff_hours=5)

    with pytest.raises(BatchError, match="не раньше cutoff"):
        validate_history_batch(batch, meta)


def test_snapshot_after_cutoff_is_caught():
    batch, meta = synthetic_batch([[0, 1]], cutoff_hours=10)

    broken = HistoryMeta(meta.client_ids, meta.cutoffs, meta.cutoffs.copy())

    with pytest.raises(BatchError, match="снимок профиля"):
        validate_history_batch(batch, broken)


def test_metadata_of_wrong_length_is_caught():
    batch, meta = synthetic_batch([[0, 1], [0, 2]], cutoff_hours=10)

    short = HistoryMeta(meta.client_ids[:1], meta.cutoffs[:1], meta.snapshot_ts[:1])

    with pytest.raises(BatchError, match="metadata"):
        validate_history_batch(batch, short)


def test_missing_profile_is_caught():
    batch, meta = synthetic_batch([[0, 1], [0, 2]], cutoff_hours=10)

    owner = np.zeros_like(batch.profile_example_ids)

    with pytest.raises(BatchError, match="профиль"):
        validate_history_batch(replace(batch, profile_example_ids=owner), meta)


# ============================================================
# ОБРЕЗКА
# ============================================================


def test_recent_keeps_the_last_events():
    batch, meta = synthetic_batch([[0, 1, 2, 3, 4]], cutoff_hours=24)

    truncated, info = truncate_recent(batch, 3)

    assert info.kept_events.tolist() == [2, 3, 4]
    assert info.original_history_length.tolist() == [5]
    assert info.used_history_length.tolist() == [3]
    assert info.truncated.tolist() == [True]
    assert info.slot_of_event.tolist() == [1, 2, 3]

    assert truncated.n_events == 3
    assert np.asarray(truncated.seq).tolist() == [2, 3, 4]


def test_kept_tokens_point_at_the_original_batch():
    batch, meta = synthetic_batch([[0, 1, 2, 3, 4]], cutoff_hours=24)

    truncated, info = truncate_recent(batch, 2)

    assert np.array_equal(truncated.key_ids, batch.key_ids[info.kept_tokens])
    assert np.array_equal(truncated.value_ids, batch.value_ids[info.kept_tokens])
    assert np.array_equal(truncated.positions, batch.positions[info.kept_tokens])


def test_short_history_is_kept_whole():
    batch, meta = synthetic_batch([[0, 1]], cutoff_hours=24)

    truncated, info = truncate_recent(batch, 3)

    assert info.truncated.tolist() == [False]
    assert info.used_history_length.tolist() == [2]
    assert truncated.n_events == batch.n_events


def test_truncation_is_per_example():
    batch, meta = synthetic_batch([[0, 1, 2, 3], [0, 1]], cutoff_hours=24)

    truncated, info = truncate_recent(batch, 2)

    assert info.original_history_length.tolist() == [4, 2]
    assert info.used_history_length.tolist() == [2, 2]
    assert info.truncated.tolist() == [True, False]
    assert info.slot_of_event.tolist() == [1, 2, 1, 2]
    assert np.asarray(truncated.example_of_event).tolist() == [0, 0, 1, 1]


def test_profile_is_untouched_and_usr_is_outside_the_limit():
    batch, meta = synthetic_batch([[0, 1, 2, 3]], cutoff_hours=24)

    truncated, info = truncate_recent(batch, 2)

    assert np.array_equal(truncated.profile_key_ids, batch.profile_key_ids)
    assert np.array_equal(truncated.profile_value_ids, batch.profile_value_ids)
    assert truncated.n_examples == batch.n_examples

    # Профиль занимает позицию 0 и в лимит N не входит.
    assert int(info.used_history_length.max()) == 2


def test_truncated_batch_is_a_valid_batch():
    batch, meta = synthetic_batch([[0, 1, 2, 3, 4], [0, 1, 2]], cutoff_hours=24)

    truncated, _ = truncate_recent(batch, 2)

    check_batch(truncated)
    validate_history_batch(truncated, meta)

    assert truncated.event_offsets[-1] == truncated.key_ids.size
    assert np.array_equal(
        truncated.event_ids, np.repeat(np.arange(truncated.n_events), np.diff(truncated.event_offsets))
    )


def test_zero_limit_is_rejected():
    batch, meta = synthetic_batch([[0, 1]], cutoff_hours=24)

    with pytest.raises(BatchError, match="положительным"):
        truncate_recent(batch, 0)


def test_example_lengths_counts_every_example():
    batch, meta = synthetic_batch([[0, 1, 2], [0]], cutoff_hours=24)

    assert example_lengths(np.asarray(batch.example_of_event), 2).tolist() == [3, 1]


# ============================================================
# ПОДГОТОВКА ЦЕЛИКОМ
# ============================================================


def test_gap_of_the_first_kept_event_is_real():
    """
    Временные признаки считаются по полной истории, поэтому у
    первого оставшегося события gap не ноль.
    """

    batch, meta = synthetic_batch([[0, 2, 3, 8]], cutoff_hours=24)

    history = prepare_history_batch(batch, meta, max_events=2)

    assert history.info.kept_events.tolist() == [2, 3]
    assert history.gap_hours.tolist() == [1.0, 5.0]
    assert history.age_hours.tolist() == [21.0, 16.0]


def test_preparation_without_masker_has_no_targets():
    batch, meta = synthetic_batch([[0, 1, 2]], cutoff_hours=24)

    history = prepare_history_batch(batch, meta, max_events=3)

    assert history.targets is None
    assert history.mask is None


def test_masker_runs_on_the_truncated_batch():
    batch, meta = synthetic_batch([[0, 1, 2, 3, 4, 5]], cutoff_hours=24)

    vocab = toy_vocab()

    masker = Masker(vocab, MaskingConfig(mode="token", seed=3, token_rate=1.0))

    history = prepare_history_batch(batch, meta, max_events=3, masker=masker)

    assert history.tokens.n_events == 3
    assert history.targets.size == history.tokens.key_ids.size
    assert history.mask.size == history.tokens.key_ids.size

    masked = history.tokens.value_ids == MASK_ID

    assert np.array_equal(masked, history.mask)
    assert masked.any()


def test_targets_do_not_reach_the_model():
    batch, meta = synthetic_batch([[0, 1, 2]], cutoff_hours=24)

    masker = Masker(toy_vocab(), MaskingConfig(mode="token", seed=1, token_rate=1.0))

    history = prepare_history_batch(batch, meta, max_events=3, masker=masker)

    inputs = to_model_inputs(history, toy_config())

    assert not hasattr(inputs, "targets")
    assert set(vars(inputs)) == {
        "events",
        "profiles",
        "example_of_event",
        "slot_of_event",
        "time_hours",
        "used_history_length",
        "n_examples",
        "kept_events",
        "kept_tokens",
        # Раскладка сессий: в прежней структуре обе пустые.
        "standalone_rows",
        "sessions",
    }

    assert inputs.standalone_rows is None
    assert inputs.sessions is None


# ============================================================
# ТЕНЗОРЫ
# ============================================================


def test_model_inputs_shapes_and_types():
    batch, meta = synthetic_batch([[0, 1, 2, 3], [0, 1]], cutoff_hours=24)

    history = prepare_history_batch(batch, meta, max_events=3)

    inputs = to_model_inputs(history, toy_config())

    assert inputs.time_hours.shape == (history.tokens.n_events, 2)
    assert inputs.time_hours.dtype == torch.float32
    assert inputs.example_of_event.dtype == torch.long
    assert inputs.slot_of_event.dtype == torch.long

    assert inputs.used_history_length.tolist() == history.info.used_history_length.tolist()
    assert inputs.max_length == 1 + int(history.info.used_history_length.max())

    slots = inputs.slot_of_event.tolist()

    assert min(slots) == 1
    assert max(slots) <= int(history.info.used_history_length.max())


def test_model_inputs_move_to_device():
    batch, meta = synthetic_batch([[0, 1]], cutoff_hours=24)

    inputs = to_model_inputs(prepare_history_batch(batch, meta, 2), toy_config(), device="cpu")

    assert inputs.example_of_event.device.type == "cpu"
    assert inputs.events.key_ids.device.type == "cpu"


def test_real_batch_prepares(tok_run):
    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    examples = [data.load(index) for index in range(4)]

    batch = collate(examples)
    meta = metadata_from_examples(examples)

    history = prepare_history_batch(batch, meta, max_events=50)

    assert int(history.info.used_history_length.max()) <= 50
    assert history.gap_hours.size == history.tokens.n_events
    assert (history.gap_hours >= 0).all()
    assert (history.age_hours > 0).all()
