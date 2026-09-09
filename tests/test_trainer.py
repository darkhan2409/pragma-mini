"""
Обучение: шаг, пропуски, бюджет, фиксированная validation.

Проверяется не «loss падает», а что механика честная: шаг
меняет веса, оценка ничего не трогает, бюджет соблюдается, а
batch без целей не считается шагом.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tokenizer.dataset import TokenizedDataset
from src.model.data import ClientStore, EpochSampler, FixedSplit
from src.model.trainer import (
    TrainConfig,
    Trainer,
    TrainingAborted,
    build_validation,
    load_environment,
    resolve_precision,
    run_training,
    tiny_overfit,
)


# ============================================================
# ОКРУЖЕНИЕ
# ============================================================


@pytest.fixture(scope="module")
def env(tok_run):
    return load_environment(tok_run["tokenized"], tok_run["vocab"], tok_run["artifacts"])


def small_config(**overrides) -> TrainConfig:
    """
    Конфигурация, на которой тест идёт секунды, а не минуты.
    """

    base = TrainConfig(
        max_train_clients=4,
        max_val_clients=2,
        batch_size=2,
        eval_batch_size=2,
        max_events_per_history=24,
        max_steps=3,
        warmup_steps=1,
        eval_every=2,
        log_every=1,
        precision="float32",
    )

    return replace(base, **overrides) if overrides else base


@pytest.fixture(scope="module")
def store(env):
    return ClientStore(env.root, "train", env.vocab_dir, max_clients=4)


def trainer_for(env, config=None) -> Trainer:
    return Trainer(config or small_config(), env.tokenizer, env.table, env.unigram, "cpu")


# ============================================================
# PRECISION
# ============================================================


def test_cpu_is_float32():
    assert resolve_precision("auto", torch.device("cpu")) == "float32"


def test_bf16_is_refused_on_cpu():
    with pytest.raises(ValueError, match="только на CUDA"):
        resolve_precision("bf16", torch.device("cpu"))


def test_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp8")


# ============================================================
# ХРАНИЛИЩЕ
# ============================================================


def test_store_matches_the_dataset_reader(env, store, tok_run):
    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    for index in (0, 1, len(store) - 1):

        ours = store.example(index)
        theirs = data.load(int(store.row_index[index]))

        assert ours.client_id == theirs.client_id
        assert ours.cutoff == theirs.cutoff
        assert ours.seq_end == theirs.seq_end
        assert ours.events.n_events == theirs.events.n_events

        assert np.array_equal(ours.events.key_ids, theirs.events.key_ids)
        assert np.array_equal(ours.events.value_ids, theirs.events.value_ids)
        assert np.array_equal(ours.profile.value_ids, theirs.profile.value_ids)


def test_store_selects_clients_deterministically(env):
    left = ClientStore(env.root, "train", env.vocab_dir, max_clients=3)
    right = ClientStore(env.root, "train", env.vocab_dir, max_clients=3)

    assert left.client_ids == right.client_ids
    assert left.client_ids == sorted(left.client_ids)
    assert np.array_equal(left.row_index, right.row_index)


# ============================================================
# SAMPLER
# ============================================================


def test_epoch_covers_every_example_once():
    sampler = EpochSampler(7, 2, seed=1)

    seen: list[int] = []

    while sampler.position < sampler.n_examples:
        seen.extend(sampler.next_batch().tolist())

    assert sorted(seen) == list(range(7))
    assert sampler.epoch == 0


def test_new_epoch_reshuffles():
    sampler = EpochSampler(6, 3, seed=1)

    first = [sampler.next_batch().tolist() for _ in range(2)]

    assert sampler.epoch == 0

    sampler.next_batch()

    assert sampler.epoch == 1

    assert first != [sampler.order[:3].tolist(), sampler.order[3:].tolist()]


def test_sampler_state_round_trips():
    left = EpochSampler(9, 2, seed=3)

    for _ in range(3):
        left.next_batch()

    right = EpochSampler(9, 2, seed=3)
    right.load_state(left.state())

    assert right.epoch == left.epoch
    assert right.position == left.position
    assert np.array_equal(right.next_batch(), left.next_batch())


def test_sampler_refuses_a_different_dataset():
    sampler = EpochSampler(5, 2, seed=1)

    with pytest.raises(ValueError, match="примеров"):
        EpochSampler(4, 2, seed=1).load_state(sampler.state())


# ============================================================
# ШАГ
# ============================================================


def test_optimizer_step_changes_the_weights(env, store):
    trainer = trainer_for(env)

    trainer.train_mode()

    sampler = EpochSampler(len(store), 2, seed=1)

    before = {
        "embeddings": trainer.backbone.pair.embeddings.token.weight.detach().clone(),
        "history": trainer.backbone.history.layers[0].linear1.weight.detach().clone(),
        "fuse": trainer.head.fuse[0].weight.detach().clone(),
    }

    result = trainer.train_step(store.examples(sampler.next_batch()))

    assert not result.skipped
    assert result.n_targets > 0
    assert result.grad_norm > 0
    assert trainer.n_steps == 1

    assert not torch.allclose(before["embeddings"], trainer.backbone.pair.embeddings.token.weight)
    assert not torch.allclose(before["history"], trainer.backbone.history.layers[0].linear1.weight)
    assert not torch.allclose(before["fuse"], trainer.head.fuse[0].weight)


def test_five_steps_stay_finite(env, store):
    trainer = trainer_for(env)

    trainer.train_mode()

    sampler = EpochSampler(len(store), 2, seed=2)

    for _ in range(5):

        result = trainer.train_step(store.examples(sampler.next_batch()))

        assert np.isfinite(result.field_balanced)
        assert np.isfinite(result.grad_norm)

    for parameter in trainer.parameters():
        assert torch.isfinite(parameter).all()


def test_learning_rate_warms_up(env, store):
    trainer = trainer_for(env, small_config(warmup_steps=4))

    trainer.train_mode()

    sampler = EpochSampler(len(store), 2, seed=3)

    rates = [trainer.train_step(store.examples(sampler.next_batch())).learning_rate for _ in range(3)]

    assert rates[0] < rates[1] < rates[2]
    assert rates[0] == pytest.approx(trainer.config.lr / 4)


# ============================================================
# ПРОПУСКИ
# ============================================================


def test_batch_without_targets_is_not_a_step(env, store):
    trainer = trainer_for(env, small_config(balanced_share=0.0))

    trainer.train_mode()

    sampler = EpochSampler(len(store), 2, seed=4)

    result = trainer.train_step(store.examples(sampler.next_batch()))

    assert result.skipped
    assert result.reason
    assert trainer.n_steps == 0
    assert trainer.n_batches == 1
    assert trainer.n_skipped == 1

    # Scheduler тоже не двигался.
    assert trainer.scheduler.last_epoch == 0


def test_too_many_skips_stop_the_run(env, store):
    trainer = trainer_for(env, small_config(balanced_share=0.0, max_consecutive_skips=2))

    trainer.train_mode()

    sampler = EpochSampler(len(store), 2, seed=5)

    trainer.train_step(store.examples(sampler.next_batch()))
    trainer.train_step(store.examples(sampler.next_batch()))

    with pytest.raises(TrainingAborted, match="подряд без целей"):
        trainer.train_step(store.examples(sampler.next_batch()))


# ============================================================
# VALIDATION
# ============================================================


@pytest.fixture(scope="module")
def splits(env):
    config = small_config()

    trainer = trainer_for(env, config)

    made, _ = build_validation(env, config, trainer.model_config)

    return made


def test_fixed_split_is_reproducible(env, splits):
    config = small_config()

    trainer = trainer_for(env, config)

    again, _ = build_validation(env, config, trainer.model_config)

    for name, split in splits.items():
        assert again[name].digest == split.digest
        assert again[name].description() == split.description()


def test_fixed_split_covers_the_examples(env, splits):
    for split in splits.values():
        assert split.n_examples > 0
        assert split.n_batches > 0
        assert split.n_targets > 0
        assert int(split.used_lengths.max()) <= small_config().max_events_per_history


def test_different_masking_gives_a_different_set(env):
    config = small_config()

    trainer = trainer_for(env, config)

    left, _ = build_validation(env, config, trainer.model_config, names=("val_time",))
    right, _ = build_validation(env, replace(config, val_seed=99), trainer.model_config, names=("val_time",))

    assert left["val_time"].digest != right["val_time"].digest


def test_evaluation_changes_neither_weights_nor_random_state(env, store, splits):
    trainer = trainer_for(env)

    trainer.train_mode()

    sampler = EpochSampler(len(store), 2, seed=6)

    trainer.train_step(store.examples(sampler.next_batch()))

    before = [parameter.detach().clone() for parameter in trainer.parameters()]

    torch.manual_seed(123)
    np.random.seed(123)

    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()

    trainer.evaluate(splits)

    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])

    for old, new in zip(before, trainer.parameters()):
        assert torch.equal(old, new)

    # Модель вернулась в режим обучения.
    assert trainer.backbone.training
    assert trainer.head.training


def test_evaluation_reports_every_split(env, splits):
    trainer = trainer_for(env)

    reports = trainer.evaluate(splits)

    assert reports.pop("_seconds") >= 0

    for name, report in reports.items():
        assert report["n_targets"] > 0
        assert report["field_balanced_ce"] > 0
        assert report["n_fields_with_targets"] > 1
        assert len(report["fields"]) == len(env.table.trainable_key_ids) + len(
            env.table.degenerate_key_ids
        )


def test_evaluation_is_repeatable(env, splits):
    trainer = trainer_for(env)

    first = trainer.evaluate(splits)
    second = trainer.evaluate(splits)

    for name in splits:
        assert first[name]["field_balanced_ce"] == second[name]["field_balanced_ce"]
        assert first[name]["accuracy"] == second[name]["accuracy"]


# ============================================================
# КОРОТКИЙ RUN
# ============================================================


@pytest.fixture(scope="module")
def short_run(env, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")

    report = run_training(env, small_config(), out, device="cpu", quiet=True)

    return {"report": report, "out": out}


def test_run_respects_the_step_budget(short_run):
    report = short_run["report"]

    assert report["counters"]["n_steps"] == 3
    assert report["config"]["max_steps"] == 3


def test_run_evaluates_before_and_after(short_run):
    steps = [item["step"] for item in short_run["report"]["evaluations"]]

    assert steps[0] == 0
    assert steps[-1] == 3
    assert 2 in steps


def test_run_writes_its_artefacts(short_run):
    out = short_run["out"]

    for name in ("report.json", "report.md", "log.jsonl", "last.pt"):
        assert (out / name).exists(), name

    lines = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]

    assert lines[0]["event"] == "start"
    assert any(item["event"] == "eval" for item in lines)
    assert any(item["event"] == "train" for item in lines)


def test_run_reports_the_actual_sizes(short_run):
    data = short_run["report"]["data"]

    assert data["train"]["clients"] == 4
    assert data["val_client"]["clients"] == 2
    assert data["val_time"]["clients"] == 2

    for item in data.values():
        assert item["examples"] > 0


def test_run_compares_before_and_after_on_the_same_masks(short_run):
    report = short_run["report"]

    for name in report["splits"]:
        assert report["before"][name]["n_targets"] == report["after"][name]["n_targets"]
        assert report["before"][name]["field_balanced_ce"] is not None
        assert report["after"][name]["field_balanced_ce"] is not None


def test_run_selects_best_by_val_time(short_run):
    best = short_run["report"]["best"]

    assert best["step"] is not None
    assert best["value"] is not None

    assert (short_run["out"] / "best.pt").exists()


def test_report_markdown_is_readable(short_run):
    text = (short_run["out"] / "report.md").read_text(encoding="utf-8")

    assert "Короткий MLM-эксперимент" in text
    assert "val_time" in text
    assert "NCE gain" in text


def test_nan_stops_the_run_with_diagnostics(env, tmp_path):
    """
    Порча выхода головы обязана останавливать обучение, а не
    расходовать бюджет на NaN.
    """

    out = tmp_path / "nan"

    original = Trainer.compute

    def broken(self, inputs, targets, attention_rule=None):
        result, field_logits, local = original(self, inputs, targets, attention_rule)
        if result.field_balanced is not None and self.n_batches > 0:
            for item in field_logits:
                item.logits.data.fill_(float("nan"))
            from src.model.losses import mlm_loss

            return mlm_loss(field_logits, local), field_logits, local
        return result, field_logits, local

    Trainer.compute = broken

    try:
        with pytest.raises(TrainingAborted, match="остановлено"):
            run_training(env, small_config(), out, device="cpu", quiet=True)
    finally:
        Trainer.compute = original

    assert (out / "diagnostics.json").exists()

    diagnostics = json.loads((out / "diagnostics.json").read_text(encoding="utf-8"))

    assert diagnostics["batch"]["clients"]
    assert diagnostics["counters"]["n_steps"] >= 0


# ============================================================
# TINY OVERFIT
# ============================================================


def test_tiny_overfit_learns_a_fixed_batch(env, tmp_path):
    report = tiny_overfit(
        env,
        small_config(),
        tmp_path,
        steps=200,
        n_examples=2,
        lr=1e-2,
        max_events=8,
        balanced_share=0.5,
        device="cpu",
        quiet=True,
    )

    assert report["batch"]["n_fields"] >= 3
    assert report["batch"]["fields_with_varied_targets"] >= 1

    assert report["ce_drop"] >= 0.8, report["ce_drop"]
    assert report["after"]["accuracy"] >= 0.9, report["after"]["accuracy"]
    assert report["passed"]

    assert (tmp_path / "overfit.json").exists()
    assert (tmp_path / "overfit.txt").exists()


def test_tiny_overfit_refuses_a_silly_budget(env, tmp_path):
    with pytest.raises(ValueError, match="100"):
        tiny_overfit(env, small_config(), tmp_path, steps=10, device="cpu", quiet=True)
