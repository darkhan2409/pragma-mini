"""
Runtime masking: воспроизводимость, чистота и четыре режима.

Masker обязан менять только копию value_ids и только там, где
значение известно, непусто и принадлежит predictable-полю.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tokenizer.config import MASK_ID, MISSING_ID, UNK_ID
from src.tokenizer.dataset import TokenBatch, TokenizedDataset, collate
from src.tokenizer.masking import (
    IGNORE_INDEX,
    MODE_EVENT,
    MODE_FIELD_BALANCED,
    MODE_KEY,
    MODE_TOKEN,
    MODES,
    Masker,
    MaskingConfig,
)
from src.tokenizer.vocab import KeyEntry, ValueEntry, Vocab

from tests.test_tok_encode import toy_vocab


# ============================================================
# СИНТЕТИЧЕСКИЙ BATCH
# ============================================================


def unbalanced_vocab() -> Vocab:
    """
    Два predictable-поля и одно нет: A широкое, B узкое.
    """

    keys = [
        KeyEntry(6, "s__a", "s", "a", "categorical", True, "string", 9, 12),
        KeyEntry(7, "s__b", "s", "b", "categorical", True, "string", 12, 15),
        KeyEntry(8, "s__quiet", "s", "quiet", "categorical", False, "string", 15, 18),
    ]

    values = [
        ValueEntry(9 + offset, 6 + offset // 3, f"s__{name}", f"v{offset % 3}", 1)
        for offset, name in enumerate(["a"] * 3 + ["b"] * 3 + ["quiet"] * 3)
    ]

    return Vocab(keys, values)


def make_batch(key_ids, value_ids, event_ids=None, example_ids=None, profile=None) -> TokenBatch:
    """
    Batch из готовых массивов: маскирование не зависит от того,
    откуда пришли токены.
    """

    key_ids = np.asarray(key_ids, dtype=np.int32)
    value_ids = np.asarray(value_ids, dtype=np.int32)

    size = key_ids.size

    event_ids = np.arange(size, dtype=np.int64) if event_ids is None else np.asarray(event_ids, np.int64)
    example_ids = np.zeros(size, dtype=np.int64) if example_ids is None else np.asarray(example_ids, np.int64)

    n_events = int(event_ids.max()) + 1 if size else 0

    widths = np.bincount(event_ids, minlength=n_events)

    offsets = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    profile = np.zeros(0, dtype=np.int32) if profile is None else np.asarray(profile, np.int32)

    return TokenBatch(
        key_ids=key_ids,
        value_ids=value_ids,
        positions=np.zeros(size, dtype=np.int16),
        event_ids=event_ids,
        example_ids=example_ids,
        event_offsets=offsets,
        example_of_event=np.zeros(n_events, dtype=np.int64),
        event_type=np.array(["synthetic"] * n_events, dtype=object),
        ts=np.zeros(n_events, dtype="datetime64[us]"),
        seq=np.arange(n_events, dtype=np.int64),
        profile_key_ids=np.zeros(profile.size, dtype=np.int32),
        profile_value_ids=profile,
        profile_positions=np.zeros(profile.size, dtype=np.int16),
        profile_example_ids=np.zeros(profile.size, dtype=np.int64),
        n_examples=int(example_ids.max()) + 1 if size else 0,
    )


# ============================================================
# ЧТО ВООБЩЕ МОЖНО МАСКИРОВАТЬ
# ============================================================


def test_special_and_unknown_values_are_never_eligible():
    vocab = toy_vocab()

    masker = Masker(vocab)

    keys = np.array([6, 6, 6, 6], dtype=np.int32)
    values = np.array([9, UNK_ID, MISSING_ID, MASK_ID], dtype=np.int32)

    assert list(masker.eligible(keys, values)) == [True, False, False, False]


def test_non_predictable_key_is_never_eligible():
    vocab = toy_vocab()

    masker = Masker(vocab)

    # toy__flag объявлен predictable=False.
    assert not masker.eligible(np.array([8]), np.array([13]))[0]
    assert masker.eligible(np.array([6]), np.array([9]))[0]


def test_profile_positions_are_rejected():
    """
    Профиль это контекст: если бы в нём оказалась маскируемая
    позиция, masker обязан упасть, а не молча её замаскировать.
    """

    vocab = toy_vocab()

    batch = make_batch([6], [9], profile=[9])

    batch = TokenBatch(**{**batch.__dict__, "profile_key_ids": np.array([6], dtype=np.int32)})

    with pytest.raises(AssertionError):
        Masker(vocab).apply(batch)


# ============================================================
# ЧИСТОТА И ВОСПРОИЗВОДИМОСТЬ
# ============================================================


@pytest.fixture(scope="module")
def real_batch(tok_run):
    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    return collate([data.load(index) for index in range(12)])


@pytest.fixture(scope="module")
def real_vocab(tok_run):
    return Vocab.load(tok_run["vocab"])


@pytest.mark.parametrize("mode", MODES)
def test_same_seed_and_step_repeat(real_vocab, real_batch, mode):
    masker = Masker(real_vocab, MaskingConfig(mode=mode, seed=11))

    first = masker.apply(real_batch, step=5)
    second = masker.apply(real_batch, step=5)

    assert np.array_equal(first.value_ids, second.value_ids)
    assert np.array_equal(first.targets, second.targets)
    assert np.array_equal(first.mask, second.mask)


@pytest.mark.parametrize("mode", MODES)
def test_other_seed_or_step_differs(real_vocab, real_batch, mode):
    base = Masker(real_vocab, MaskingConfig(mode=mode, seed=11)).apply(real_batch, step=5)

    other_step = Masker(real_vocab, MaskingConfig(mode=mode, seed=11)).apply(real_batch, step=6)
    other_seed = Masker(real_vocab, MaskingConfig(mode=mode, seed=12)).apply(real_batch, step=5)

    assert not np.array_equal(base.mask, other_step.mask)
    assert not np.array_equal(base.mask, other_seed.mask)


@pytest.mark.parametrize("mode", MODES)
def test_input_arrays_are_not_touched(real_vocab, real_batch, mode):
    before = {
        "key_ids": real_batch.key_ids.copy(),
        "value_ids": real_batch.value_ids.copy(),
        "positions": real_batch.positions.copy(),
        "ts": real_batch.ts.copy(),
        "profile_value_ids": real_batch.profile_value_ids.copy(),
    }

    result = Masker(real_vocab, MaskingConfig(mode=mode, seed=3)).apply(real_batch, step=1)

    for name, snapshot in before.items():
        assert np.array_equal(getattr(real_batch, name), snapshot), name

    assert result.value_ids is not real_batch.value_ids


@pytest.mark.parametrize("mode", MODES)
def test_targets_are_correct(real_vocab, real_batch, mode):
    result = Masker(real_vocab, MaskingConfig(mode=mode, seed=3)).apply(real_batch, step=1)

    assert result.n_masked > 0

    assert (result.value_ids[result.mask] == MASK_ID).all()
    assert np.array_equal(result.targets[result.mask], real_batch.value_ids[result.mask])

    assert (result.targets[~result.mask] == IGNORE_INDEX).all()
    assert np.array_equal(result.value_ids[~result.mask], real_batch.value_ids[~result.mask])

    assert (result.profile_targets == IGNORE_INDEX).all()
    assert np.array_equal(result.profile_value_ids, real_batch.profile_value_ids)


@pytest.mark.parametrize("mode", MODES)
def test_only_eligible_positions_are_masked(real_vocab, real_batch, mode):
    masker = Masker(real_vocab, MaskingConfig(mode=mode, seed=3))

    eligible = masker.eligible(real_batch.key_ids, real_batch.value_ids)

    result = masker.apply(real_batch, step=1)

    assert not (result.mask & ~eligible).any()


@pytest.mark.parametrize("mode", MODES)
def test_derived_and_quality_fields_are_never_masked(real_vocab, real_batch, mode):
    result = Masker(real_vocab, MaskingConfig(mode=mode, seed=3)).apply(real_batch, step=1)

    forbidden = {
        real_vocab.key_id("communication", "day_of_week"),
        real_vocab.key_id("communication", "hour"),
        real_vocab.key_id("product_event", "timestamp_quality"),
    }

    masked_keys = set(real_batch.key_ids[result.mask].tolist())

    assert not (masked_keys & forbidden)


# ============================================================
# РЕЖИМЫ
# ============================================================


def test_token_mode_hits_about_the_configured_share(real_vocab, real_batch):
    result = Masker(real_vocab, MaskingConfig(mode=MODE_TOKEN, seed=1, token_rate=0.25)).apply(real_batch)

    share = result.n_masked / result.n_eligible

    assert 0.22 < share < 0.28


def test_key_mode_masks_whole_fields_inside_an_example(real_vocab, real_batch):
    masker = Masker(real_vocab, MaskingConfig(mode=MODE_KEY, seed=1, keys_per_example=2))

    eligible = masker.eligible(real_batch.key_ids, real_batch.value_ids)

    result = masker.apply(real_batch)

    for example in range(real_batch.n_examples):

        inside = real_batch.example_ids == example

        masked_keys = set(real_batch.key_ids[inside & result.mask].tolist())

        assert len(masked_keys) <= 2

        for key in masked_keys:
            available = inside & eligible & (real_batch.key_ids == key)
            assert np.array_equal(result.mask[available], np.ones(available.sum(), dtype=bool))


def test_event_mode_masks_whole_events(real_vocab, real_batch):
    masker = Masker(real_vocab, MaskingConfig(mode=MODE_EVENT, seed=1, event_rate=0.3))

    eligible = masker.eligible(real_batch.key_ids, real_batch.value_ids)

    result = masker.apply(real_batch)

    touched = set(real_batch.event_ids[result.mask].tolist())

    assert touched

    for event in list(touched)[:50]:
        inside = real_batch.event_ids == event
        assert result.mask[inside & eligible].all()

    untouched = set(range(real_batch.n_events)) - touched

    for event in list(untouched)[:50]:
        inside = real_batch.event_ids == event
        assert not result.mask[inside].any()


# ============================================================
# FIELD BALANCED
# ============================================================


def unbalanced_batch(wide: int = 1000, narrow: int = 20) -> TokenBatch:
    keys = np.array([6] * wide + [7] * narrow, dtype=np.int32)
    values = np.array([9] * wide + [12] * narrow, dtype=np.int32)

    return make_batch(keys, values)


def test_field_balanced_does_not_copy_field_frequencies():
    """
    Поле, которого в batch в 50 раз меньше, обязано получить
    сопоставимое число масок, а не пропорциональное частоте.
    """

    vocab = unbalanced_vocab()

    batch = unbalanced_batch()

    balanced = Masker(vocab, MaskingConfig(mode=MODE_FIELD_BALANCED, seed=5, balanced_share=0.15))
    plain = Masker(vocab, MaskingConfig(mode=MODE_TOKEN, seed=5, token_rate=0.15))

    left = balanced.apply(batch)
    right = plain.apply(batch)

    narrow_balanced = int((batch.key_ids[left.mask] == 7).sum())
    narrow_plain = int((batch.key_ids[right.mask] == 7).sum())

    assert narrow_plain < 8
    assert narrow_balanced >= 15
    assert narrow_balanced > 3 * narrow_plain


def test_field_balanced_respects_the_budget():
    vocab = unbalanced_vocab()

    batch = unbalanced_batch()

    result = Masker(vocab, MaskingConfig(mode=MODE_FIELD_BALANCED, seed=5, balanced_share=0.2)).apply(batch)

    assert result.n_masked == round(0.2 * result.n_eligible)


def test_field_balanced_never_picks_a_position_twice():
    vocab = unbalanced_vocab()

    batch = unbalanced_batch()

    result = Masker(vocab, MaskingConfig(mode=MODE_FIELD_BALANCED, seed=5, balanced_share=0.3)).apply(batch)

    positions = result.masked_positions

    assert positions.size == result.n_masked
    assert np.unique(positions).size == result.n_masked


def test_field_balanced_exhausts_everything_when_the_budget_is_large():
    vocab = unbalanced_vocab()

    batch = unbalanced_batch(wide=30, narrow=10)

    result = Masker(vocab, MaskingConfig(mode=MODE_FIELD_BALANCED, seed=5, balanced_share=1.0)).apply(batch)

    assert result.n_masked == result.n_eligible


# ============================================================
# КОНФИГ
# ============================================================


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        MaskingConfig(mode="everything")


def test_rates_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError):
        MaskingConfig(token_rate=1.5)


def test_modes_are_not_mixed():
    from src.tokenizer.masking import MODE_COMBINED

    assert set(MODES) == {MODE_TOKEN, MODE_KEY, MODE_EVENT, MODE_FIELD_BALANCED, MODE_COMBINED}

    config = MaskingConfig(mode=MODE_EVENT)

    assert isinstance(config.mode, str)
