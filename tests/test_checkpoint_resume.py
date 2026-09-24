from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.mlm.settings import best_checkpoint_path, checkpoint_path
from src.mlm.train import (
    CHECKPOINT_KEYS,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
    train,
)

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Продолжение обязано быть НЕОТЛИЧИМО от непрерывного обучения.
# Совпасть должны не только веса: состояние AdamW, положение
# расписания, счётчик шагов, лучший val_loss и терпение — всё, от
# чего зависит следующий шаг.
#
# Поэтому сравниваются целиком два чекпойнта, а не выборочные
# поля: забытое поле — это и есть тот способ, которым продолжение
# тихо расходится с непрерывным прогоном.
#
# Отдельно проверяется продолжение ВНУТРИ эпохи: пройденные
# micro-batch'и пропускаются без прохода модели, и если бы они
# учились второй раз, итог разошёлся бы.
# ============================================================


def read(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def alike(left, right, where: str = "") -> None:
    """
    Рекурсивное сравнение состояний: тензоры побитово, остальное
    по значению.
    """

    assert type(left) is type(right), where

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), where

    elif isinstance(left, dict):
        assert set(left) == set(right), where
        for key in left:
            alike(left[key], right[key], f"{where}.{key}")

    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), where
        for number, (one, other) in enumerate(zip(left, right)):
            alike(one, other, f"{where}[{number}]")

    else:
        assert left == right, where


def compare(left: dict, right: dict) -> None:

    assert set(left) == set(right) == set(CHECKPOINT_KEYS)

    for key in CHECKPOINT_KEYS:
        alike(left[key], right[key], key)


# ============================================================
# НЕПРЕРЫВНО ПРОТИВ ПРОДОЛЖЕНИЯ
# ============================================================


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_resume_after_a_whole_epoch_matches_straight_training(stage, dropout: float):
    """
    Две эпохи подряд против «остановились на границе эпох,
    сохранение, продолжение».

    Обе фазы планируются на ОДНИ И ТЕ ЖЕ две эпохи: план
    замораживается первым запуском, и обучение, начатое на одну
    эпоху, продолжается по своей кривой намеренно — это отдельная
    семантика, проверяемая в test_scheduler.

    При dropout больше нуля это ещё и проверка состояния
    генераторов: без него продолжение разыграло бы другие маски
    dropout и веса разошлись бы.
    """

    settle(stage, train_people=many(), dropout=dropout)

    config = tiny(token_budget=6, warmup_steps=2)
    masking = every_value()

    train(config, epochs=2, max_steps=None, masking=masking)

    straight = read(checkpoint_path())

    assert straight["step"] == 12

    settle(stage, train_people=many(), dropout=dropout)

    # Шесть шагов — ровно одна эпоха этого мира.
    train(config, epochs=2, max_steps=6, masking=masking)
    train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    compare(straight, read(checkpoint_path()))


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_resume_inside_an_unfinished_epoch_matches_straight_training(
    stage, dropout: float
):
    """
    Остановка посреди эпохи и продолжение с того же места.

    Пройденные micro-batch'и пропускаются без прохода модели; если
    бы они учились второй раз, состояние разошлось бы уже на
    первом шаге продолжения.
    """

    settle(stage, train_people=many(), dropout=dropout)

    config = tiny(token_budget=6, warmup_steps=2)
    masking = every_value()

    train(config, epochs=2, max_steps=8, masking=masking)

    straight = read(checkpoint_path())

    assert straight["epoch_complete"] is False
    assert straight["step"] == 8

    settle(stage, train_people=many(), dropout=dropout)

    train(config, epochs=2, max_steps=4, masking=masking)

    paused = read(checkpoint_path())

    assert paused["epoch_complete"] is False
    assert paused["micro_batches_done"] == 4

    train(config, epochs=2, max_steps=8, masking=masking, resume=True)

    compare(straight, read(checkpoint_path()))


def test_resume_keeps_the_frozen_schedule_plan(stage):
    """
    Горизонт и число эпох плана лежат в чекпойнте и при
    продолжении не пересчитываются.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, warmup_steps=1)
    masking = every_value()

    train(config, epochs=3, max_steps=2, masking=masking)

    first = read(checkpoint_path())

    train(config, epochs=3, max_steps=4, masking=masking, resume=True)

    second = read(checkpoint_path())

    assert first["scheduler_total"] == second["scheduler_total"]
    assert first["scheduler_epochs"] == second["scheduler_epochs"] == 3
    assert second["scheduler_state_dict"]["last_epoch"] == second["step"]


def test_nothing_left_to_do_is_reported_not_repeated(stage):
    """
    Продолжение исчерпанного обучения ничего не учит и говорит об
    этом причиной nothing.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, warmup_steps=1)
    masking = every_value()

    train(config, epochs=1, max_steps=None, masking=masking)

    before = read(checkpoint_path())

    result = train(config, epochs=1, max_steps=None, masking=masking, resume=True)

    assert result["reason"] == "nothing"

    compare(before, read(checkpoint_path()))


# ============================================================
# ЧТО ЛЕЖИТ В ЧЕКПОЙНТЕ
# ============================================================


def test_checkpoint_carries_everything_needed_to_continue(stage):

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    state = read(checkpoint_path())

    assert set(state) == set(CHECKPOINT_KEYS)
    assert state["model_state_dict"]
    assert state["optimizer_state_dict"]["state"]
    assert state["scheduler_state_dict"]["last_epoch"] == state["step"]
    assert state["config"]["token_budget"] == 6
    assert state["masking"]["value_probability"] == 1.0
    assert state["rng_state"].dtype == torch.uint8


def test_best_checkpoint_is_a_full_state_not_only_weights(stage):

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    best = read(best_checkpoint_path())

    assert set(best) == set(CHECKPOINT_KEYS)
    assert best["epoch_complete"] is True


# ============================================================
# НАДЁЖНОСТЬ
# ============================================================


def test_resume_without_a_checkpoint_says_so(stage):

    settle(stage, train_people=many())

    with pytest.raises(CheckpointError, match="продолжать нечего"):
        train(tiny(), epochs=1, max_steps=None, masking=every_value(), resume=True)


def test_broken_file_is_a_domain_error(stage):

    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a checkpoint at all")

    with pytest.raises(CheckpointError, match="не читается"):
        load_checkpoint(path)


def test_checkpoint_of_the_old_format_is_refused(stage):

    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"model_state_dict": {}, "step": 1}, path)

    with pytest.raises(CheckpointError, match="чекпойнт старого формата"):
        load_checkpoint(path)


@pytest.mark.parametrize("missing", ["scheduler_total", "scheduler_epochs", "rng_state"])
def test_every_required_field_is_checked(stage, missing: str):

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    path = checkpoint_path()

    state = read(path)
    state.pop(missing)

    torch.save(state, path)

    with pytest.raises(CheckpointError, match=missing):
        load_checkpoint(path)


def test_weights_of_another_shape_are_refused(stage):

    settle(stage, train_people=many())

    config = tiny(token_budget=6)

    train(config, epochs=1, max_steps=None, masking=every_value())

    path = checkpoint_path()

    state = read(path)
    state["model_state_dict"]["head.proj.bias"] = torch.zeros(7)

    torch.save(state, path)

    with pytest.raises(CheckpointError, match="не подходит к модели"):
        train(config, epochs=2, max_steps=None, masking=every_value(), resume=True)


def test_save_goes_through_a_temporary_file(stage, monkeypatch):
    """
    Прерванная запись не портит прежний чекпойнт: новый пишется
    рядом и переименовывается на место одним движением.
    """

    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint({"step": 1}, path)

    assert read(path) == {"step": 1}
    assert not path.with_name(path.name + ".tmp").exists()

    original = torch.save

    def fails(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("диск кончился")

    monkeypatch.setattr(torch, "save", fails)

    with pytest.raises(OSError):
        save_checkpoint({"step": 2}, path)

    monkeypatch.setattr(torch, "save", original)

    assert read(path) == {"step": 1}


def test_a_new_run_removes_the_checkpoints_of_the_previous_one(stage):
    """
    Прогон без --resume к прошлому обучению отношения не имеет:
    оба старых чекпойнта удаляются.

    Проверяется на мире, где у val целей нет вовсе: лучший
    чекпойнт тогда не пишется заново, и его отсутствие видно.
    """

    quiet = [
        world.make(
            "v-quiet",
            [[(world.KEY_A, [10], False)]],
            [(world.KEY_A, [20])],
            targetable=False,
        )
    ]

    settle(stage, train_people=many(), val_people=quiet)

    latest, best = checkpoint_path(), best_checkpoint_path()

    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_bytes(b"old")
    best.write_bytes(b"old")

    result = train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    assert result["best_val_loss"] is None
    assert not best.exists()
    assert set(read(latest)) == set(CHECKPOINT_KEYS)


def test_a_resumed_run_keeps_the_best_checkpoint(stage):

    settle(stage, train_people=many())

    config = tiny(token_budget=6)
    masking = every_value()

    train(config, epochs=1, max_steps=None, masking=masking)

    before = read(best_checkpoint_path())

    train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    assert best_checkpoint_path().exists()

    after = read(best_checkpoint_path())

    assert after["epoch"] >= before["epoch"]
