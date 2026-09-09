"""
Полные истории без обрезки и бюджет в одну эпоху.

Проверяется шесть вещей: отключение лимита это тождество, а не
большое число; белый список клиентов; ленивое построение входа;
эпоха покрывает каждый пример ровно один раз; накопление
градиента даёт один шаг оптимизатора.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tokenizer.dataset import TokenizedDataset, collate
from src.model.batching import check_batch
from src.model.data import ClientStore, EpochSampler, FixedSplit, select_clients
from src.model.history_batching import (
    POLICY_NONE,
    metadata_from_examples,
    prepare_history_batch,
    to_model_inputs,
    truncate_recent,
)
from src.model.trainer import Trainer, run_training, store_for, truncation_check

from tests.test_combined_masking import combined_config, grid_batch
from tests.test_history_batching import synthetic_batch
from tests.test_trainer import env, small_config  # noqa: F401


# ============================================================
# ЛИМИТ ОТКЛЮЧЁН
# ============================================================


def test_no_limit_is_an_identity():
    batch, meta = synthetic_batch([[0, 1, 2, 3, 4], [0, 1]], cutoff_hours=24)

    kept, info = truncate_recent(batch, None)

    assert info.policy == POLICY_NONE
    assert info.max_events is None

    assert info.original_history_length.tolist() == [5, 2]
    assert info.used_history_length.tolist() == [5, 2]
    assert info.truncated.tolist() == [False, False]

    assert info.kept_events.tolist() == list(range(batch.n_events))
    assert info.kept_tokens.tolist() == list(range(batch.n_tokens))

    # Слоты нумеруются внутри примера, а не сквозь batch.
    assert info.slot_of_event.tolist() == [1, 2, 3, 4, 5, 1, 2]

    assert kept is batch

    assert np.array_equal(batch.key_ids[info.kept_tokens], batch.key_ids)

    check_batch(kept)


def test_no_limit_keeps_every_event_on_real_data(tok_run):
    from src.tokenizer.artifacts import Tokenizer
    from src.tokenizer.masking import Masker

    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    examples = [data.load(index) for index in range(6)]

    masker = Masker(tokenizer.vocab, combined_config().masking())

    history = prepare_history_batch(
        collate(examples), metadata_from_examples(examples), None, masker=masker, step=0
    )

    info = history.info

    assert np.array_equal(info.original_history_length, info.used_history_length)
    assert not info.truncated.any()

    assert history.tokens.n_events == sum(example.events.n_events for example in examples)
    assert history.masking["n_masked"] > 0

    # Обрезка с лимитом выше самой длинной истории даёт то же.
    limited = prepare_history_batch(
        collate(examples), metadata_from_examples(examples), 10_000, masker=masker, step=0
    )

    assert np.array_equal(limited.info.used_history_length, info.used_history_length)


# ============================================================
# БЕЛЫЙ СПИСОК КЛИЕНТОВ
# ============================================================


def test_client_whitelist_selects_exactly_those_clients(env):
    data = TokenizedDataset(env.root, "train", vocab_dir=env.vocab_dir)

    every = select_clients(data.examples)

    wanted = set(every[:3])

    store = ClientStore(env.root, "train", env.vocab_dir, clients=wanted)

    assert set(store.client_ids) == wanted
    assert store.client_ids == sorted(wanted)

    # Порядок примеров прежний: (client_id, cutoff).
    ids = [int(row["client_id"]) for row in store.rows]

    assert ids == sorted(ids)


def test_universe_is_shared_between_splits(env):
    config = replace(small_config(), client_universe=6, max_train_clients=None, max_val_clients=None)

    train = store_for(env, config, "train")
    validation = store_for(env, config, "val_time")

    assert all(value < 6 for value in train.client_ids)
    assert all(value < 6 for value in validation.client_ids)

    # Распределение по сплитам не переопределяется: val_time
    # берёт train-клиентов, val_client других.
    assert set(validation.client_ids) <= set(train.client_ids)


def test_unknown_client_is_refused(env):
    with pytest.raises(ValueError, match="не нашлось ни одного клиента"):
        ClientStore(env.root, "train", env.vocab_dir, clients={10 ** 9})


# ============================================================
# ЛЕНИВЫЙ ВХОД
# ============================================================


def test_iter_batches_is_stable_and_matches_the_stored_batches(env):
    config = replace(small_config(), max_events_per_history=None)

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, "cpu")

    store = store_for(env, config, "val_time")

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

    first = list(split.iter_batches(trainer.model_config))
    second = list(split.iter_batches(trainer.model_config))

    assert len(first) == split.n_batches

    for (left_inputs, left_targets), (right_inputs, right_targets) in zip(first, second):

        torch.testing.assert_close(left_inputs.events.key_ids, right_inputs.events.key_ids)
        torch.testing.assert_close(left_inputs.events.value_ids, right_inputs.events.value_ids)

        assert left_targets.digest() == right_targets.digest()

    # Тот же вход, что строит to_model_inputs напрямую.
    history, targets = split.prepared[0]

    direct = to_model_inputs(history, trainer.model_config)

    torch.testing.assert_close(first[0][0].events.key_ids, direct.events.key_ids)

    assert split.description()["settings"]["max_events_per_history"] is None


# ============================================================
# ЭПОХА
# ============================================================


def smallest_universe(env) -> int:
    """
    Наименьшая вселенная, при которой непусты все нужные сплиты.
    """

    limits = [
        min(select_clients(TokenizedDataset(env.root, name, vocab_dir=env.vocab_dir).examples))
        for name in ("train", "val_client", "val_time")
    ]

    return max(limits) + 1


def test_one_epoch_covers_every_example_once(env, tmp_path):
    config = replace(
        small_config(),
        client_universe=smallest_universe(env),
        max_train_clients=2,
        max_val_clients=1,
        max_events_per_history=None,
        epochs=1,
        eval_every=10 ** 6,
        log_every=10 ** 6,
        eval_batch_size=2,
    )

    report = run_training(env, config, tmp_path / "epoch", device="cpu", quiet=True, preflight=True)

    store = store_for(env, config, "train")

    expected = math.ceil(len(store) / config.batch_size)

    budget = report["budget"]

    assert budget["epoch_examples"] == len(store)
    assert budget["micro_per_epoch"] == expected
    assert budget["micro_batches_done"] == expected

    # Пропуски эпоху не удлиняют: batch'ей ровно столько, сколько
    # нужно, чтобы каждый пример попал один раз.
    assert report["counters"]["n_batches"] == expected
    assert report["counters"]["n_steps"] + report["counters"]["n_skipped"] == expected

    assert report["truncation_check"]["passed"]
    assert report["truncation_check"]["n_truncated"] == 0


def test_the_last_partial_batch_is_not_lost():
    sampler = EpochSampler(7, 2, seed=1)

    seen: list[int] = []

    for _ in range(math.ceil(7 / 2)):
        seen.extend(sampler.next_batch().tolist())

    assert sorted(seen) == list(range(7))
    assert sampler.epoch == 0
    assert sampler.position == 7


def test_truncation_check_notices_a_limit(env):
    config = replace(small_config(), client_universe=3, max_train_clients=None)

    limited = truncation_check(store_for(env, config, "train"), replace(config, max_events_per_history=4))

    assert not limited["passed"]
    assert limited["n_truncated"] > 0

    full = truncation_check(store_for(env, config, "train"), replace(config, max_events_per_history=None))

    assert full["passed"]
    assert full["n_truncated"] == 0
    assert full["lengths"]["max"] >= full["lengths"]["p50"]


# ============================================================
# НАКОПЛЕНИЕ ГРАДИЕНТА
# ============================================================


def test_accumulation_makes_one_optimizer_step(env):
    config = replace(small_config(), accumulation_steps=2, max_events_per_history=None)

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, "cpu")

    trainer.train_mode()

    store = store_for(env, config, "train")

    groups = [store.examples([0, 1]), store.examples([2, 3])]

    before = trainer.scheduler.last_epoch

    result = trainer.train_group(groups)

    assert not result.skipped
    assert trainer.n_steps == 1
    assert trainer.n_batches == 2
    assert trainer.scheduler.last_epoch == before + 1

    # Диагностика маскирования сложена по группе.
    assert result.masking["n_masked"] > 0


def test_accumulation_matches_one_larger_batch(env):
    """
    Два микро-batch'а по два примера и один batch из четырёх
    обязаны дать один и тот же градиентный шаг.
    """

    base = replace(small_config(), max_events_per_history=None, dropout=0.0, warmup_steps=1)

    store = store_for(env, base, "train")

    indices = [0, 1, 2, 3]

    split = Trainer(replace(base, accumulation_steps=2), env.tokenizer, env.table, env.unigram, "cpu")
    whole = Trainer(replace(base, batch_size=4), env.tokenizer, env.table, env.unigram, "cpu")

    split.train_mode()
    whole.train_mode()

    # Один и тот же шаг masker: маски у обоих одинаковые.
    prepared = [split.prepare(store.examples(indices), 0)]

    torch.manual_seed(0)
    whole.optimize_group([(prepared[0][0], prepared[0][1])])

    torch.manual_seed(0)
    split.optimize_group([(prepared[0][0], prepared[0][1])])

    for left, right in zip(whole.parameters(), split.parameters()):
        torch.testing.assert_close(left, right)
