"""
Отказоустойчивость долгого обучения.

Три вещи, без которых многочасовой прогон нельзя оставлять
одного: продолжение обязано давать то же самое обучение,
неудачное сохранение не должно уничтожать удачное, а
несовместимое продолжение обязано отказываться до первого шага,
а не после часа работы.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model import trainer as trainer_module
from src.model.data import EpochSampler
from src.model.trainer import (
    TrainConfig,
    TrainingInterrupted,
    run_training,
)
from src.tokenizer.config import IncompatibleArtifactsError

from tests.test_trainer import env  # noqa: F401


STOP_AFTER = 4
TOTAL_STEPS = 8


def resume_config(**overrides) -> TrainConfig:
    """
    Крошечный, но настоящий прогон: те же режимы, что у эпохи.
    """

    base = TrainConfig(
        max_train_clients=6,
        max_val_clients=2,
        batch_size=2,
        eval_batch_size=2,
        max_events_per_history=32,
        max_steps=TOTAL_STEPS,
        warmup_steps=1,
        eval_every=4,
        log_every=2,
        checkpoint_every=2,
        precision="float32",
        masking_mode="combined",
        mask_scheme="example",
        stream_validation=True,
        target_policy="history",
        best_metric="recent",
    )

    return trainer_module.replace(base, **overrides) if overrides else base


def payload_of(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def same_tensors(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[name].cpu(), right[name].cpu()) for name in left)


# ============================================================
# 1. ПРОДОЛЖЕНИЕ ЭТО ТО ЖЕ ОБУЧЕНИЕ
# ============================================================


def test_interrupted_and_resumed_training_matches_uninterrupted(env, tmp_path, monkeypatch):

    config = resume_config()

    # --- непрерывный прогон -------------------------------
    straight = tmp_path / "straight"

    run_training(env, config, straight, device="cpu", quiet=True)

    # --- прогон с остановкой посередине -------------------
    broken = tmp_path / "broken"

    original = trainer_module.stop_requested

    def stop_after(out_dir) -> bool:
        # Просьба остановиться приходит ровно после
        # STOP_AFTER шага: остальное как обычно.
        marker = Path(out_dir) / "steps.seen"
        seen = int(marker.read_text()) if marker.exists() else 0
        seen += 1
        marker.write_text(str(seen))
        return seen >= STOP_AFTER

    monkeypatch.setattr(trainer_module, "stop_requested", stop_after)

    with pytest.raises(TrainingInterrupted):
        run_training(env, config, broken, device="cpu", quiet=True)

    interrupted = broken / "interrupted.pt"

    assert interrupted.exists()
    assert (broken / "interrupted.json").exists()

    stopped = payload_of(interrupted)

    assert stopped["counters"]["n_steps"] == STOP_AFTER
    assert stopped["counters"]["n_steps"] < TOTAL_STEPS

    monkeypatch.setattr(trainer_module, "stop_requested", original)

    # --- продолжение --------------------------------------
    run_training(env, config, broken, device="cpu", quiet=True, resume=interrupted)

    left = payload_of(straight / "last.pt")
    right = payload_of(broken / "last.pt")

    # --- веса ---------------------------------------------
    assert same_tensors(left["backbone"], right["backbone"])
    assert same_tensors(left["head"], right["head"])

    # --- оптимизатор и scheduler --------------------------
    assert left["optimizer"]["param_groups"] == right["optimizer"]["param_groups"]

    for key in left["optimizer"]["state"]:
        for name, value in left["optimizer"]["state"][key].items():
            other = right["optimizer"]["state"][key][name]
            if torch.is_tensor(value):
                assert torch.equal(value.cpu(), other.cpu()), (key, name)
            else:
                assert value == other, (key, name)

    assert left["scheduler"] == right["scheduler"]

    # --- счётчики и прогресс ------------------------------
    assert left["counters"] == right["counters"]
    assert left["counters"]["n_steps"] == TOTAL_STEPS

    assert left["progress"]["micro_done"] == right["progress"]["micro_done"]
    assert left["progress"]["evaluated_steps"] == right["progress"]["evaluated_steps"]
    assert left["progress"]["best"] == right["progress"]["best"]

    # --- sampler и следующий batch ------------------------
    assert left["sampler"]["position"] == right["sampler"]["position"]
    assert left["sampler"]["epoch"] == right["sampler"]["epoch"]
    assert left["sampler"]["order"] == right["sampler"]["order"]

    def next_batch(state: dict) -> list[int]:
        sampler = EpochSampler(state["n_examples"], state["batch_size"], state["seed"])
        sampler.load_state(state)
        return sampler.next_batch().tolist()

    assert next_batch(left["sampler"]) == next_batch(right["sampler"])

    # --- генераторы случайных чисел -----------------------
    assert left["rng"]["python"] == right["rng"]["python"]
    assert torch.equal(
        torch.as_tensor(left["rng"]["torch_cpu"]), torch.as_tensor(right["rng"]["torch_cpu"])
    )

    for a, b in zip(left["rng"]["numpy"], right["rng"]["numpy"]):
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b

    # --- оценка шага 0 не повторялась ---------------------
    import json

    events = [
        json.loads(line)
        for line in (broken / "log.jsonl").read_text(encoding="utf-8").strip().split("\n")
    ]

    assert sum(1 for item in events if item["event"] == "resume") == 1

    zero = [item for item in events if item["event"] == "eval" and item["step"] == 0]

    assert len(zero) == 1

    # Лог продолжен, а не начат заново.
    assert events[0]["event"] == "start"
    assert any(item["event"] == "interrupted" for item in events)


# ============================================================
# 2. НЕУДАЧНОЕ СОХРАНЕНИЕ НЕ ТРОГАЕТ ПРЕЖНЕЕ
# ============================================================


def test_failed_save_leaves_the_previous_checkpoint_intact(env, tmp_path, monkeypatch):

    from src.model.checkpoint import save_checkpoint
    from src.model.mlm_head import MLMHead
    from src.model.trainer import Trainer

    config = resume_config()

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, "cpu")

    path = tmp_path / "last.pt"

    def save(**overrides):
        return save_checkpoint(
            path,
            backbone=trainer.backbone,
            head=trainer.head,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            counters=trainer.counters(),
            train_config=config.as_dict(),
            model_config=trainer.model_config.as_dict(),
            masking_config=config.masking().as_dict(),
            sampler_state=None,
            splits={},
            artifacts=env.hashes,
            **overrides,
        )

    save(metrics={"generation": 1})

    good = path.read_bytes()

    assert payload_of(path)["metrics"] == {"generation": 1}

    # --- падение во время записи --------------------------
    real_save = torch.save

    def half_written(payload, target, *args, **kwargs):
        # Файл создан и оборван: ровно то, что бывает при
        # выключении питания посреди сохранения.
        Path(target).write_bytes(b"\x80\x02broken")
        raise RuntimeError("диск кончился")

    monkeypatch.setattr(torch, "save", half_written)

    with pytest.raises(RuntimeError):
        save(metrics={"generation": 2})

    monkeypatch.setattr(torch, "save", real_save)

    assert path.read_bytes() == good
    assert payload_of(path)["metrics"] == {"generation": 1}

    # --- запись прошла, но файл не читается ---------------
    def corrupt_after_write(payload, target, *args, **kwargs):
        Path(target).write_bytes(b"not a checkpoint at all")

    monkeypatch.setattr(torch, "save", corrupt_after_write)

    with pytest.raises(Exception):
        save(metrics={"generation": 3})

    monkeypatch.setattr(torch, "save", real_save)

    assert path.read_bytes() == good
    assert payload_of(path)["metrics"] == {"generation": 1}

    # --- временные файлы не остаются ----------------------
    leftovers = [item.name for item in tmp_path.iterdir() if ".tmp-" in item.name]

    assert leftovers == []

    # --- удачное сохранение по-прежнему работает ----------
    save(metrics={"generation": 4})

    assert payload_of(path)["metrics"] == {"generation": 4}


# ============================================================
# 3. НЕСОВМЕСТИМОЕ ПРОДОЛЖЕНИЕ ОТКЛОНЯЕТСЯ
# ============================================================


def test_incompatible_resume_is_refused_before_the_first_step(env, tmp_path):

    config = resume_config(max_steps=2, eval_every=100)

    out = tmp_path / "run"

    run_training(env, config, out, device="cpu", quiet=True)

    source = out / "last.pt"

    before = source.read_bytes()

    steps_done = payload_of(source)["counters"]["n_steps"]

    def attempt(target: Path, other: TrainConfig, environment=None):
        return run_training(
            environment or env, other, target, device="cpu", quiet=True, resume=source
        )

    # --- другой seed --------------------------------------
    with pytest.raises(IncompatibleArtifactsError, match="seed"):
        attempt(tmp_path / "seed", resume_config(max_steps=2, eval_every=100, seed=777))

    # --- другая политика целей ----------------------------
    with pytest.raises(IncompatibleArtifactsError, match="target_policy"):
        attempt(
            tmp_path / "policy",
            resume_config(max_steps=2, eval_every=100, target_policy="all"),
        )

    # --- другая схема масок -------------------------------
    with pytest.raises(IncompatibleArtifactsError):
        attempt(
            tmp_path / "scheme",
            resume_config(
                max_steps=2, eval_every=100, mask_scheme="batch", stream_validation=False
            ),
        )

    # --- другая архитектура -------------------------------
    with pytest.raises(IncompatibleArtifactsError):
        attempt(tmp_path / "model", resume_config(max_steps=2, eval_every=100, d_model=32))

    # --- другие artifacts ---------------------------------
    spoiled = trainer_module.replace(env, hashes={**env.hashes, "unigram_baselines": "0" * 64})

    with pytest.raises(IncompatibleArtifactsError, match="artifacts"):
        attempt(tmp_path / "artifacts", config, spoiled)

    # --- ничего из этого не сдвинуло обучение -------------
    assert source.read_bytes() == before
    assert payload_of(source)["counters"]["n_steps"] == steps_done

    for name in ("seed", "policy", "scheme", "model", "artifacts"):
        rejected = tmp_path / name
        # Каталог мог появиться, но ни одного шага в нём нет.
        assert not (rejected / "last.pt").exists()
        assert not (rejected / "report.json").exists()

    # --- бюджет тоже часть эксперимента -------------------
    #
    # Продолжить «то же обучение» с другим числом шагов нельзя:
    # это другой эксперимент под тем же именем.
    with pytest.raises(IncompatibleArtifactsError, match="max_steps"):
        attempt(tmp_path / "budget", resume_config(max_steps=4, eval_every=100))

    # --- совместимое продолжение принимается --------------
    again = tmp_path / "again"

    shutil.copytree(out, again)

    report = run_training(
        env, config, again, device="cpu", quiet=True, resume=again / "last.pt"
    )

    assert report["counters"]["n_steps"] == steps_done
    assert report["resumed_from"].endswith("last.pt")
