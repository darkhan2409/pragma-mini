from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from src.masking.settings import MaskingConfig
from src.mlm.settings import MlmConfig
from src.mlm.train import horizon, lr_factor, train

from tests import world
from tests.test_training_math import every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Расписание LR принадлежит ОБУЧЕНИЮ, а не прогону. Горизонт
# cosine считается по полному плану --epochs и замораживается
# первым запуском; --max-steps только останавливает прогон.
#
# Если бы --max-steps укорачивал горизонт, прогон с ним проехал
# бы весь cosine за сто шагов, а продолжение тех же эпох поехало
# бы по другой кривой — со скачком LR на первом же шаге. Это
# ровно тот класс ошибки, ради которого здесь два теста на
# непрерывность кривой.
#
# LR наблюдается там же, где его видит человек, — в строке лога
# шага: так проверяется то, с чем реально сделан шаг.
# ============================================================


LINE = re.compile(r"step=(\d+) .*lr=([0-9.e+-]+)")


def trail(text: str) -> list[tuple[int, float]]:
    """
    Пары (шаг, LR) из лога обучения.
    """

    return [(int(step), float(value)) for step, value in LINE.findall(text)]


def many(count: int = 6) -> list[world.Made]:

    return [
        world.make(
            f"c{number}",
            [[(world.KEY_A, [10 + number], True)]],
            [(world.KEY_A, [20])],
        )
        for number in range(count)
    ]


# ============================================================
# ФОРМУЛА
# ============================================================


def test_warmup_reaches_full_rate_exactly_at_its_last_step():
    """
    Разгон — (done + 1) / warmup: первый шаг уже учит, а шаг
    номер warmup идёт с полным LR. Здесь легче всего ошибиться на
    единицу в любую сторону.
    """

    values = [lr_factor(done, 4, 100, 0.0) for done in range(5)]

    assert values[:4] == [0.25, 0.5, 0.75, 1.0]

    # Первый шаг cosine тоже на полном LR: прогресс равен нулю.
    assert values[4] == pytest.approx(1.0)


def test_cosine_runs_from_one_to_the_floor():

    warmup, total, floor = 4, 24, 0.1

    middle = lr_factor(warmup + (total - warmup) // 2, warmup, total, floor)

    assert lr_factor(warmup, warmup, total, floor) == pytest.approx(1.0)
    assert middle == pytest.approx(floor + (1.0 - floor) * 0.5)
    assert lr_factor(total, warmup, total, floor) == pytest.approx(floor)


@pytest.mark.parametrize("done", [24, 25, 100, 10_000])
def test_rate_never_falls_below_the_floor(done: int):
    """
    За горизонтом множитель остаётся floor: обучение сверх плана
    идёт на min_learning_rate, а не уходит в ноль или в минус.
    """

    assert lr_factor(done, 4, 24, 0.1) == pytest.approx(0.1)


def test_floor_is_the_ratio_of_the_two_configured_rates():

    config = MlmConfig(learning_rate=1e-2, min_learning_rate=1e-3)

    floor = config.min_learning_rate / config.learning_rate

    assert config.learning_rate * lr_factor(1_000, 1, 10, floor) == pytest.approx(
        config.min_learning_rate
    )


# ============================================================
# ГОРИЗОНТ
# ============================================================


def test_horizon_counts_the_whole_plan(stage):
    """
    Горизонт — шаги всех эпох: micro-batch'и эпохи, делённые на
    окно накопления, умноженные на число эпох.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, grad_accum_steps=4)
    masking = every_value()

    assert horizon(config, masking, 1) == math.ceil(6 / 4)
    assert horizon(config, masking, 3) == math.ceil(6 / 4) * 3
    assert horizon(config, masking, 10) == math.ceil(6 / 4) * 10


def test_horizon_does_not_know_about_max_steps(stage):
    """
    У horizon нет и не должно быть параметра max_steps: предел
    прогона в план не входит.
    """

    import inspect

    assert list(inspect.signature(horizon).parameters) == ["config", "masking", "epochs"]


def test_horizon_does_not_depend_on_the_mask_of_the_epoch(stage):
    """
    Маскирование меняет значения, но не длины, поэтому число
    micro-batch'ей от эпохи не зависит.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6)

    assert horizon(config, every_value(1), 2) == horizon(config, every_value(2), 2)


# ============================================================
# РАСПИСАНИЕ ДВИГАЮТ ШАГИ, А НЕ MICRO-BATCH'И
# ============================================================


def test_only_optimizer_steps_advance_the_schedule(stage, capsys):
    """
    Шесть micro-batch'ей при окне 2 дают три шага, и номера шагов
    идут подряд: micro-batch сам по себе расписание не двигает.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, grad_accum_steps=2, warmup_steps=1)

    train(config, epochs=1, max_steps=None, masking=every_value())

    steps = [step for step, _ in trail(capsys.readouterr().out)]

    assert steps == [1, 2, 3]


def test_rate_follows_the_formula_step_by_step(stage, capsys):
    """
    Лог обязан показывать тот же LR, что даёт формула на этом
    горизонте: числа сверяются по одному.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, grad_accum_steps=2, warmup_steps=2)
    masking = every_value()

    total = horizon(config, masking, 2)
    floor = config.min_learning_rate / config.learning_rate

    train(config, epochs=2, max_steps=None, masking=masking)

    seen = trail(capsys.readouterr().out)

    assert len(seen) == total

    for step, value in seen:

        expected = config.learning_rate * lr_factor(
            step - 1, config.warmup_steps, total, floor
        )

        assert value == pytest.approx(expected, rel=5e-3)


# ============================================================
# MAX-STEPS НЕ УКОРАЧИВАЕТ КРИВУЮ
# ============================================================


def test_stopping_early_does_not_bend_the_curve(stage, capsys):
    """
    Regression. Прогон с --max-steps обязан идти по той же
    кривой, что и прогон без него: предел прогона — не план.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, warmup_steps=1)
    masking = every_value()

    train(config, epochs=2, max_steps=None, masking=masking)

    whole = trail(capsys.readouterr().out)

    settle(stage, train_people=many())

    train(config, epochs=2, max_steps=3, masking=masking)

    stopped = trail(capsys.readouterr().out)

    assert len(stopped) == 3
    assert len(whole) > 3

    for (step, value), (same_step, same_value) in zip(stopped, whole):
        assert step == same_step
        assert value == pytest.approx(same_value, rel=1e-9)


def test_resume_continues_the_same_curve_without_a_jump(stage, capsys):
    """
    Regression. Десять шагов подряд против «пять, сохранение,
    продолжение, ещё пять»: последовательность LR обязана совпасть
    шаг в шаг.

    Скачок на шестом шаге означал бы, что горизонт посчитан заново
    от места остановки.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, warmup_steps=2)
    masking = every_value()

    train(config, epochs=2, max_steps=10, masking=masking)

    straight = trail(capsys.readouterr().out)

    assert len(straight) == 10

    settle(stage, train_people=many())

    train(config, epochs=2, max_steps=5, masking=masking)

    first = trail(capsys.readouterr().out)

    train(config, epochs=2, max_steps=10, masking=masking, resume=True)

    second = trail(capsys.readouterr().out)

    assert [step for step, _ in first + second] == list(range(1, 11))

    for (step, value), (same_step, same_value) in zip(first + second, straight):
        assert step == same_step
        assert value == pytest.approx(same_value, rel=1e-9), step


def test_more_epochs_on_resume_keep_the_old_plan(stage, capsys):
    """
    План замораживается первым запуском. Увеличенный --epochs
    добавляет эпохи уже на min_learning_rate и прошлую часть
    кривой не пересчитывает — об этом обучение говорит строкой в
    логе.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, warmup_steps=1)
    masking = every_value()

    train(config, epochs=1, max_steps=None, masking=masking)

    planned = horizon(config, masking, 1)

    first = trail(capsys.readouterr().out)

    assert len(first) == planned

    train(config, epochs=3, max_steps=None, masking=masking, resume=True)

    text = capsys.readouterr().out

    assert "расписание рассчитано на 1 эпох" in text
    assert "остаётся прежним" in text

    later = trail(text)

    assert later, "продолжение обязано сделать хотя бы один шаг"

    # За горизонтом множитель равен floor, то есть LR равен
    # min_learning_rate.
    for _, value in later:
        assert value == pytest.approx(config.min_learning_rate, rel=5e-3)
