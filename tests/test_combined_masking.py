"""
Combined masking: три независимые выборки и их объединение.

Проверяется не «маска непустая», а что каждая стратегия делает
ровно своё: token берёт отдельные позиции, event закрывает
событие целиком, key закрывает ключ целиком, но только внутри
своего примера. Три последовательных masker дали бы не это.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tokenizer.config import MASK_ID, MISSING_ID, UNK_ID
from src.tokenizer.dataset import TokenizedDataset, collate
from src.tokenizer.masking import (
    IGNORE_INDEX,
    MODE_COMBINED,
    MODES,
    Masker,
    MaskingConfig,
)
from src.model.data import ClientStore, EpochSampler, FixedSplit
from src.model.history_batching import metadata_from_examples, prepare_history_batch
from src.model.mlm_batching import build_targets
from src.model.mlm_head import FieldTable
from src.model.trainer import TrainConfig, Trainer

from tests.test_tok_masking import make_batch, unbalanced_vocab
from tests.test_trainer import env, small_config  # noqa: F401


# ============================================================
# СИНТЕТИКА
# ============================================================
#
# Два примера по три события, в каждом событии оба
# предсказуемых поля и одно нет. Такой batch различает все три
# стратегии: у него есть повторяющиеся ключи внутри примера,
# несколько событий и несколько примеров.


KEY_A, KEY_B, KEY_QUIET = 6, 7, 8

VALUE_A, VALUE_B, VALUE_QUIET = 9, 12, 15


def grid_batch(n_examples: int = 2, n_events: int = 3):
    """
    Batch, где ключ встречается в каждом событии каждого примера.
    """

    keys: list[int] = []
    values: list[int] = []
    event_ids: list[int] = []
    example_ids: list[int] = []

    event = 0

    for example in range(n_examples):
        for _ in range(n_events):
            for key, value in ((KEY_A, VALUE_A), (KEY_B, VALUE_B), (KEY_QUIET, VALUE_QUIET)):
                keys.append(key)
                values.append(value)
                event_ids.append(event)
                example_ids.append(example)
            event += 1

    batch = make_batch(keys, values, event_ids, example_ids)

    # make_batch кладёт все события в пример 0: поправляем.
    owner = np.array(
        [example_ids[event_ids.index(index)] for index in range(event)], dtype=np.int64
    )

    return replace(batch, example_of_event=owner)


def rates(**overrides) -> MaskingConfig:
    base = {"mode": MODE_COMBINED, "seed": 5, "token_rate": 0.0, "event_rate": 0.0, "key_rate": 0.0}
    return MaskingConfig(**{**base, **overrides})


def apply(config: MaskingConfig, batch=None, step: int = 0):
    batch = grid_batch() if batch is None else batch
    return Masker(unbalanced_vocab(), config).apply(batch, step), batch


# ============================================================
# СТРАТЕГИИ ПО ОТДЕЛЬНОСТИ
# ============================================================


def test_all_rates_zero_masks_nothing():
    result, _ = apply(rates())

    assert result.n_masked == 0
    assert not result.mask.any()
    assert (result.targets == IGNORE_INDEX).all()

    assert result.selection["unique"] == 0
    assert set(result.selection["strategies"].values()) == {0}

    assert result.masked_fraction == 0.0


@pytest.mark.parametrize("strategy", ["token_rate", "event_rate", "key_rate"])
def test_each_strategy_alone_can_take_everything(strategy):
    result, batch = apply(rates(**{strategy: 1.0}))

    eligible = Masker(unbalanced_vocab()).eligible(batch.key_ids, batch.value_ids)

    assert result.n_masked == int(eligible.sum())
    assert np.array_equal(result.mask, eligible)

    assert result.selection["strategies"][strategy.replace("_rate", "")] == result.n_masked


def test_event_masking_takes_whole_events():
    result, batch = apply(rates(event_rate=0.5), step=3)

    eligible = Masker(unbalanced_vocab()).eligible(batch.key_ids, batch.value_ids)

    events = np.asarray(batch.event_ids)

    partial = 0

    for event in np.unique(events[eligible]):
        inside = eligible & (events == event)
        if len(set(result.mask[inside].tolist())) != 1:
            partial += 1

    assert partial == 0

    # И это не «всё или ничего» на весь batch.
    assert 0 < result.n_masked < int(eligible.sum())


def test_key_masking_takes_whole_keys_inside_one_example():
    result, batch = apply(rates(key_rate=0.5), step=11)

    eligible = Masker(unbalanced_vocab()).eligible(batch.key_ids, batch.value_ids)

    keys = np.asarray(batch.key_ids, dtype=np.int64)
    owners = np.asarray(batch.example_ids, dtype=np.int64)

    partial = 0

    for key in np.unique(keys[eligible]):
        for owner in np.unique(owners[eligible & (keys == key)]):
            inside = eligible & (keys == key) & (owners == owner)
            if len(set(result.mask[inside].tolist())) != 1:
                partial += 1

    assert partial == 0
    assert 0 < result.n_masked < int(eligible.sum())


def test_key_masking_is_per_example_not_per_batch():
    """
    Один и тот же ключ обязан закрываться в одном примере и
    оставаться открытым в другом.
    """

    vocab = unbalanced_vocab()

    batch = grid_batch(n_examples=6, n_events=2)

    eligible = Masker(vocab).eligible(batch.key_ids, batch.value_ids)

    keys = np.asarray(batch.key_ids, dtype=np.int64)
    owners = np.asarray(batch.example_ids, dtype=np.int64)

    mixed = 0

    for step in range(20):

        result = Masker(vocab, rates(key_rate=0.5)).apply(batch, step)

        for key in np.unique(keys[eligible]):

            states = {
                bool(result.mask[eligible & (keys == key) & (owners == owner)][0])
                for owner in np.unique(owners[eligible & (keys == key)])
            }

            if len(states) > 1:
                mixed += 1

    assert mixed > 0, "ключ всегда закрывался во всех примерах сразу: это выбор на batch, а не на пример"


def test_token_masking_can_split_an_event():
    """
    Token берёт отдельные позиции: событие остаётся частично видимым.
    """

    vocab = unbalanced_vocab()

    batch = grid_batch(n_examples=4, n_events=4)

    eligible = Masker(vocab).eligible(batch.key_ids, batch.value_ids)

    events = np.asarray(batch.event_ids)

    split = 0

    for step in range(10):

        result = Masker(vocab, rates(token_rate=0.5)).apply(batch, step)

        for event in np.unique(events[eligible]):
            inside = eligible & (events == event)
            if len(set(result.mask[inside].tolist())) > 1:
                split += 1

    assert split > 0


# ============================================================
# ОБЪЕДИНЕНИЕ
# ============================================================


def test_union_counts_intersections_once():
    result, batch = apply(rates(token_rate=1.0, event_rate=1.0, key_rate=1.0))

    eligible = Masker(unbalanced_vocab()).eligible(batch.key_ids, batch.value_ids)

    total = int(eligible.sum())

    assert result.n_masked == total
    assert result.selection["unique"] == total

    # Каждая стратегия взяла всё, но объединение не утроилось.
    assert result.selection["strategies"] == {"token": total, "event": total, "key": total}

    assert sum(result.selection["strategies"].values()) == 3 * total


def test_selection_is_never_smaller_than_its_parts():
    result, _ = apply(rates(token_rate=0.2, event_rate=0.2, key_rate=0.2), step=4)

    strategies = result.selection["strategies"]

    assert result.n_masked >= max(strategies.values())
    assert result.n_masked <= sum(strategies.values())


def test_expected_share_is_not_the_sum_of_rates():
    config = rates(token_rate=0.15, event_rate=0.10, key_rate=0.10)

    assert config.expected_share == pytest.approx(0.3115)
    assert config.expected_share < 0.35


def test_observed_share_matches_the_expectation(tok_run):
    """
    На большом реальном batch доля скрытых значений сходится к
    1 - 0.85*0.9*0.9, а не к 0.35.
    """

    from src.tokenizer.vocab import Vocab

    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    batch = collate([data.load(index) for index in range(12)])

    vocab = Vocab.load(tok_run["vocab"])

    config = rates(token_rate=0.15, event_rate=0.10, key_rate=0.10, seed=1)

    masker = Masker(vocab, config)

    assert int(masker.eligible(batch.key_ids, batch.value_ids).sum()) > 3000

    shares = [masker.apply(batch, step).masked_fraction for step in range(20)]

    assert float(np.mean(shares)) == pytest.approx(config.expected_share, abs=0.04)


# ============================================================
# ЧИСТОТА
# ============================================================


def test_targets_hold_the_original_values():
    result, batch = apply(rates(token_rate=0.5, event_rate=0.2, key_rate=0.2), step=2)

    assert result.n_masked > 0

    assert (result.value_ids[result.mask] == MASK_ID).all()
    assert np.array_equal(result.targets[result.mask], batch.value_ids[result.mask])

    assert (result.targets[~result.mask] == IGNORE_INDEX).all()
    assert np.array_equal(result.value_ids[~result.mask], batch.value_ids[~result.mask])


def test_specials_and_non_predictable_keys_are_left_alone():
    keys = [KEY_A, KEY_A, KEY_A, KEY_QUIET]
    values = [VALUE_A, UNK_ID, MISSING_ID, VALUE_QUIET]

    batch = make_batch(keys, values, event_ids=[0, 0, 0, 0], example_ids=[0, 0, 0, 0])

    result, _ = apply(rates(token_rate=1.0, event_rate=1.0, key_rate=1.0), batch)

    assert result.mask.tolist() == [True, False, False, False]


def test_profile_is_never_masked():
    batch = make_batch([KEY_A], [VALUE_A], profile=[VALUE_A])

    result, _ = apply(rates(token_rate=1.0), batch)

    assert (result.profile_targets == IGNORE_INDEX).all()
    assert np.array_equal(result.profile_value_ids, batch.profile_value_ids)


def test_input_batch_is_untouched():
    batch = grid_batch()

    before = {name: getattr(batch, name).copy() for name in ("key_ids", "value_ids", "event_ids")}

    apply(rates(token_rate=0.5, event_rate=0.5, key_rate=0.5), batch)

    for name, snapshot in before.items():
        assert np.array_equal(getattr(batch, name), snapshot), name


def test_same_seed_and_step_repeat():
    config = rates(token_rate=0.3, event_rate=0.3, key_rate=0.3)

    first, batch = apply(config, step=9)
    second, _ = apply(config, batch, step=9)
    other, _ = apply(config, batch, step=10)

    assert np.array_equal(first.mask, second.mask)
    assert not np.array_equal(first.mask, other.mask)


def test_the_mode_is_registered():
    assert MODE_COMBINED in MODES


def test_key_rate_is_validated():
    with pytest.raises(ValueError, match="key_rate"):
        MaskingConfig(mode=MODE_COMBINED, key_rate=1.5)


# ============================================================
# БЕЗ ЦЕЛЕЙ
# ============================================================


def test_empty_mask_gives_no_targets(tok_run):
    """
    Ноль масок это явный случай, а не повод добавить маску молча.
    """

    from src.tokenizer.artifacts import Tokenizer

    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    examples = [data.load(index) for index in range(2)]

    history = prepare_history_batch(
        collate(examples),
        metadata_from_examples(examples),
        32,
        masker=Masker(tokenizer.vocab, rates()),
        step=0,
    )

    targets = build_targets(history, FieldTable.load(tokenizer.vocab, tok_run["vocab"]))

    assert history.masking["n_masked"] == 0
    assert targets.n == 0
    assert targets.n_masked == 0


# ============================================================
# ИНТЕГРАЦИЯ
# ============================================================


def combined_config(**overrides) -> TrainConfig:
    return replace(
        small_config(),
        masking_mode=MODE_COMBINED,
        token_rate=0.15,
        event_rate=0.10,
        key_rate=0.10,
        **overrides,
    )


def test_config_round_trips_the_key_rate():
    config = combined_config()

    assert TrainConfig.from_dict(config.as_dict()) == config
    assert config.masking().key_rate == 0.10
    assert config.as_dict()["key_rate"] == 0.10


def test_initialisation_does_not_depend_on_the_objective(env):
    """
    Вторая модель обязана стартовать из тех же весов: иначе
    сравнение мерило бы и разницу инициализации.
    """

    left = Trainer(small_config(), env.tokenizer, env.table, env.unigram, "cpu")
    right = Trainer(combined_config(), env.tokenizer, env.table, env.unigram, "cpu")

    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b)


def test_training_step_works_and_stays_finite(env):
    trainer = Trainer(combined_config(), env.tokenizer, env.table, env.unigram, "cpu")

    store = ClientStore(env.root, "train", env.vocab_dir, max_clients=4)

    sampler = EpochSampler(len(store), 2, seed=1)

    trainer.train_mode()

    before = trainer.backbone.pair.embeddings.token.weight.detach().clone()

    for _ in range(3):

        result = trainer.train_step(store.examples(sampler.next_batch()))

        assert not result.skipped
        assert np.isfinite(result.field_balanced)
        assert np.isfinite(result.grad_norm)

        assert result.masking["mode"] == MODE_COMBINED
        assert set(result.masking["selection"]["strategies"]) == {"token", "event", "key"}

    assert not torch.allclose(before, trainer.backbone.pair.embeddings.token.weight)

    for parameter in trainer.parameters():
        assert torch.isfinite(parameter).all()


def test_fixed_split_records_the_diagnostics(env):
    config = combined_config()

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, "cpu")

    store = ClientStore(env.root, "val_time", env.vocab_dir, config.max_val_clients)

    split = FixedSplit.build(
        "val_time",
        store,
        env.vocab,
        env.table,
        trainer.model_config,
        config.masking(seed=config.val_seed),
        config.max_events_per_history,
        config.eval_batch_size,
    )

    description = split.description()

    assert description["n_masked"] == split.n_targets + split.n_degenerate
    assert 0.0 < description["masked_fraction"] < 1.0
    assert set(description["selected_by"]) == {"token", "event", "key"}

    again = FixedSplit.build(
        "val_time",
        store,
        env.vocab,
        env.table,
        trainer.model_config,
        config.masking(seed=config.val_seed),
        config.max_events_per_history,
        config.eval_batch_size,
    )

    assert again.digest == split.digest
