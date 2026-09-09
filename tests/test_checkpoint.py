"""
Checkpoint: продолжение обучения, а не только веса.

Восстановиться должно всё, от чего зависит следующий шаг:
оптимизатор, scheduler, счётчики, sampler и генераторы. Иначе
«продолжил с того же места» ничем не проверяется.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tokenizer.config import IncompatibleArtifactsError
from src.model.checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint
from src.model.data import ClientStore, EpochSampler
from src.model.trainer import Trainer, build_validation, run_training

from tests.test_trainer import env, small_config, store, trainer_for  # noqa: F401


# ============================================================
# ХЕЛПЕРЫ
# ============================================================


def write(path, trainer, sampler, env, splits=None, metrics=None):
    return save_checkpoint(
        path,
        backbone=trainer.backbone,
        head=trainer.head,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        counters=trainer.counters(),
        train_config=trainer.config.as_dict(),
        model_config=trainer.model_config.as_dict(),
        masking_config=trainer.config.masking().as_dict(),
        sampler_state=None if sampler is None else sampler.state(),
        splits=splits or {},
        artifacts=env.hashes,
        metrics=metrics,
    )


def train(trainer, store, sampler, steps: int) -> None:
    for _ in range(steps):
        trainer.train_step(store.examples(sampler.next_batch()))


# ============================================================
# КРУГ
# ============================================================


def test_saved_model_reproduces_its_forward(env, store, tmp_path):
    config = small_config()

    trainer = trainer_for(env, config)

    trainer.train_mode()

    sampler = EpochSampler(len(store), config.batch_size, config.seed)

    train(trainer, store, sampler, 2)

    splits, _ = build_validation(env, config, trainer.model_config, names=("val_time",))

    before = trainer.evaluate(splits)

    path = write(tmp_path / "last.pt", trainer, sampler, env,
                 {name: split.description() for name, split in splits.items()})

    fresh = Trainer(config, env.tokenizer, env.table, env.unigram, "cpu")

    load_checkpoint(path, backbone=fresh.backbone, head=fresh.head)

    after = fresh.evaluate(splits)

    assert after["val_time"]["field_balanced_ce"] == before["val_time"]["field_balanced_ce"]
    assert after["val_time"]["accuracy"] == before["val_time"]["accuracy"]


def test_resumed_training_matches_uninterrupted(env, store, tmp_path):
    """
    Четыре шага подряд и два плюс два через checkpoint обязаны
    дать одни и те же веса.
    """

    config = small_config()

    straight = trainer_for(env, config)
    straight.train_mode()

    sampler = EpochSampler(len(store), config.batch_size, config.seed)

    torch.manual_seed(1000)

    train(straight, store, sampler, 4)

    # --------------------------------------------------------

    stopped = trainer_for(env, config)
    stopped.train_mode()

    early = EpochSampler(len(store), config.batch_size, config.seed)

    torch.manual_seed(1000)

    train(stopped, store, early, 2)

    path = write(tmp_path / "half.pt", stopped, early, env)

    # --------------------------------------------------------

    resumed = trainer_for(env, config)
    resumed.train_mode()

    late = EpochSampler(len(store), config.batch_size, config.seed)

    payload = load_checkpoint(
        path,
        backbone=resumed.backbone,
        head=resumed.head,
        optimizer=resumed.optimizer,
        scheduler=resumed.scheduler,
        sampler=late,
        model_config=resumed.model_config.as_dict(),
        artifacts=env.hashes,
    )

    resumed.load_counters(payload["counters"])

    assert resumed.n_steps == 2
    assert late.position == early.position

    train(resumed, store, late, 2)

    assert resumed.n_steps == straight.n_steps

    for left, right in zip(straight.parameters(), resumed.parameters()):
        torch.testing.assert_close(left, right)


def test_random_state_is_restored(env, store, tmp_path):
    config = small_config()

    trainer = trainer_for(env, config)

    sampler = EpochSampler(len(store), config.batch_size, config.seed)

    path = write(tmp_path / "rng.pt", trainer, sampler, env)

    expected_torch = torch.rand(4)
    expected_numpy = np.random.rand(4)

    fresh = trainer_for(env, config)

    load_checkpoint(path, backbone=fresh.backbone, head=fresh.head)

    torch.testing.assert_close(torch.rand(4), expected_torch)
    assert np.allclose(np.random.rand(4), expected_numpy)


def test_scheduler_and_optimizer_survive(env, store, tmp_path):
    config = small_config(warmup_steps=10)

    trainer = trainer_for(env, config)
    trainer.train_mode()

    sampler = EpochSampler(len(store), config.batch_size, config.seed)

    train(trainer, store, sampler, 3)

    path = write(tmp_path / "state.pt", trainer, sampler, env)

    fresh = trainer_for(env, config)

    load_checkpoint(
        path,
        backbone=fresh.backbone,
        head=fresh.head,
        optimizer=fresh.optimizer,
        scheduler=fresh.scheduler,
    )

    assert fresh.scheduler.last_epoch == trainer.scheduler.last_epoch

    assert fresh.optimizer.param_groups[0]["lr"] == pytest.approx(
        trainer.optimizer.param_groups[0]["lr"]
    )

    left = trainer.optimizer.state_dict()["state"]
    right = fresh.optimizer.state_dict()["state"]

    assert set(left) == set(right)
    assert left[0]["step"] == right[0]["step"]


# ============================================================
# СОВМЕСТИМОСТЬ
# ============================================================


def test_version_is_checked(env, store, tmp_path):
    config = small_config()

    trainer = trainer_for(env, config)

    path = write(tmp_path / "old.pt", trainer, None, env)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = CHECKPOINT_VERSION + 1
    torch.save(payload, path)

    with pytest.raises(IncompatibleArtifactsError, match="версии"):
        load_checkpoint(path, backbone=trainer.backbone, head=trainer.head)


def test_changed_artifacts_are_refused(env, store, tmp_path):
    config = small_config()

    trainer = trainer_for(env, config)

    path = write(tmp_path / "arts.pt", trainer, None, env)

    spoiled = {**env.hashes, "unigram_baselines": "0" * 64}

    with pytest.raises(IncompatibleArtifactsError, match="artifacts"):
        load_checkpoint(path, backbone=trainer.backbone, head=trainer.head, artifacts=spoiled)


def test_changed_model_config_is_refused(env, store, tmp_path):
    config = small_config()

    trainer = trainer_for(env, config)

    path = write(tmp_path / "model.pt", trainer, None, env)

    other = {**trainer.model_config.as_dict(), "d_model": 128}

    with pytest.raises(IncompatibleArtifactsError, match="модели"):
        load_checkpoint(path, backbone=trainer.backbone, head=trainer.head, model_config=other)


def test_changed_validation_masks_are_refused(env, tmp_path):
    config = small_config()

    trainer = trainer_for(env, config)

    splits, _ = build_validation(env, config, trainer.model_config, names=("val_time",))

    descriptions = {name: split.description() for name, split in splits.items()}

    path = write(tmp_path / "splits.pt", trainer, None, env, descriptions)

    other, _ = build_validation(
        env, replace(config, val_seed=1234), trainer.model_config, names=("val_time",)
    )

    with pytest.raises(IncompatibleArtifactsError, match="маски и цели"):
        load_checkpoint(
            path,
            backbone=trainer.backbone,
            head=trainer.head,
            splits={name: split.description() for name, split in other.items()},
        )


def test_missing_split_is_refused(env, tmp_path):
    config = small_config()

    trainer = trainer_for(env, config)

    path = write(tmp_path / "nosplit.pt", trainer, None, env, {})

    with pytest.raises(IncompatibleArtifactsError, match="описания набора"):
        load_checkpoint(
            path,
            backbone=trainer.backbone,
            head=trainer.head,
            splits={"val_time": {"targets_sha256": "abc"}},
        )


# ============================================================
# BEST
# ============================================================


def test_best_is_not_written_when_val_time_has_no_targets(env, tmp_path, monkeypatch):
    """
    Пустой критерий не заменяется другим молча: best просто нет,
    и отчёт говорит почему.
    """

    original = Trainer.evaluate

    def hollow(self, splits, *args, **kwargs):
        reports = original(self, splits, *args, **kwargs)
        if "val_time" in reports:
            reports["val_time"] = {**reports["val_time"], "field_balanced_ce": None, "n_targets": 0}
        return reports

    monkeypatch.setattr(Trainer, "evaluate", hollow)

    out = tmp_path / "hollow"

    report = run_training(env, small_config(), out, device="cpu", quiet=True)

    assert report["best"]["step"] is None
    assert "критерий best не применим" in report["best"]["reason"]

    assert not (out / "best.pt").exists()
    assert (out / "last.pt").exists()

    text = (out / "report.md").read_text(encoding="utf-8")

    assert "best` не выбран" in text
