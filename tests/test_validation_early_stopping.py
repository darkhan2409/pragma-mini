from __future__ import annotations

import pytest
import torch

from src.mlm.inputs import Source, micro_batches
from src.mlm.model import pack
from src.mlm.settings import best_checkpoint_path, checkpoint_path
from src.mlm.train import Scores, train, validate

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import every_value, fresh, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Validation обязана быть измерением, а не продолжением обучения:
# та же модель, eval, no_grad, ни одного изменённого веса.
#
# Среднее берётся ПО ЦЕЛЯМ, а не по micro-batch'ам. Разница видна
# только на неравном числе целей, поэтому мир здесь такой.
#
# Early stopping и лучший чекпойнт зависят не от данных, а от
# последовательности val_loss. Чтобы проверять именно решение, а
# не случайность обучения, последовательность задаётся явно:
# validate подменяется сценарием.
# ============================================================


CPU = torch.device("cpu")


def uneven_val() -> list[world.Made]:
    """
    Клиенты с разным числом целей: одна, четыре и ни одной.
    """

    return [
        world.make("v-one", [[(world.KEY_A, [10], True)]], [(world.KEY_A, [20])]),
        world.make(
            "v-four",
            [[(world.KEY_B, [11, 12, 13, 14], True)]],
            [(world.KEY_B, [21])],
        ),
        world.make(
            "v-quiet",
            [[(world.KEY_C, [15], False)]],
            [(world.KEY_C, [22])],
            targetable=False,
        ),
    ]


def by_hand(model, budget: int) -> tuple[float, int, list[float]]:
    """
    То же среднее, посчитанное здесь: сумма потерь целей,
    делённая на их число.
    """

    total = 0.0
    targets = 0
    each: list[float] = []

    with torch.no_grad():
        for batch in micro_batches(Source("val").clients(), budget):
            out = model(pack(batch, CPU))
            each.append(float(out.loss) if out.count else float("nan"))
            if out.count:
                total += float(out.loss) * out.count
                targets += out.count

    return total / targets, targets, each


# ============================================================
# ИЗМЕРЕНИЕ, А НЕ ОБУЧЕНИЕ
# ============================================================


def test_validation_averages_over_targets_not_over_micro_batches(stage):

    settle(stage, train_people=many(), val_people=uneven_val())

    config = tiny(token_budget=8)

    model = fresh(stage, config)
    model.eval()

    scores = validate(model, Source("val"), CPU, config.token_budget)
    loss, targets = scores.loss, scores.targets

    expected, count, each = by_hand(model, config.token_budget)

    assert targets == count == 5
    assert loss == pytest.approx(expected, rel=1e-9)

    # Неправильное среднее — по micro-batch'ам — даёт другое число.
    alive = [value for value in each if value == value]

    assert len(alive) > 1
    assert loss != pytest.approx(sum(alive) / len(alive), rel=1e-4)


def test_validation_leaves_the_model_and_its_weights_alone(stage):

    settle(stage, train_people=many(), val_people=uneven_val())

    config = tiny(token_budget=8)

    model = fresh(stage, config)
    model.train()

    before = {name: value.clone() for name, value in model.state_dict().items()}

    validate(model, Source("val"), CPU, config.token_budget)

    assert model.training is False

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name

    for name, parameter in model.named_parameters():
        assert parameter.grad is None, name


def test_micro_batch_without_targets_does_not_move_the_average(stage):
    """
    Клиент без целей участвует в проходе, но в среднее не входит:
    иначе он тянул бы его к нулю.
    """

    settle(stage, train_people=many(), val_people=uneven_val())

    config = tiny(token_budget=8)

    model = fresh(stage, config)

    scores = validate(model, Source("val"), CPU, config.token_budget)
    with_quiet, targets = scores.loss, scores.targets

    settle(
        stage,
        train_people=many(),
        val_people=[made for made in uneven_val() if made.targetable],
    )

    scores = validate(model, Source("val"), CPU, config.token_budget)
    without, fewer = scores.loss, scores.targets

    assert targets == fewer
    assert with_quiet == pytest.approx(without, rel=1e-9)


def test_group_without_targets_gives_no_loss_at_all(stage):

    quiet = [made for made in uneven_val() if not made.targetable]

    settle(stage, train_people=many(), val_people=quiet)

    config = tiny(token_budget=8)

    scores = validate(fresh(stage, config), Source("val"), CPU, config.token_budget)

    assert scores.loss is None
    assert scores.targets == 0

    # Без целей нет и долей: ни деления на ноль, ни выдуманного нуля.
    assert scores.top1_accuracy is None and scores.top5_accuracy is None


def test_validation_uses_the_model_it_was_given(stage):
    """
    Своих весов validation не загружает: результат меняется
    вместе с переданной моделью.
    """

    settle(stage, train_people=many(), val_people=uneven_val())

    config = tiny(token_budget=8)

    model = fresh(stage, config)

    before = validate(model, Source("val"), CPU, config.token_budget).loss

    with torch.no_grad():
        model.head.proj.bias.add_(1.0)

    after = validate(model, Source("val"), CPU, config.token_budget).loss

    assert before != pytest.approx(after, rel=1e-6)


# ============================================================
# СЦЕНАРИЙ VAL_LOSS
# ============================================================


def scripted(monkeypatch, values: list[float | None], targets: int = 5):
    """
    validate по сценарию: решения об улучшении проверяются на
    заданной последовательности, а не на случайности обучения.
    """

    seen = iter(values)

    def fake(model, source, device, token_budget):
        value = next(seen)
        return Scores() if value is None else Scores(loss_sum=value * targets, targets=targets)

    monkeypatch.setattr("src.mlm.train.validate", fake)


def read(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_best_checkpoint_is_the_epoch_with_the_lowest_loss(stage, monkeypatch):
    """
    Сценарий 1.00, 0.90, 0.95, 0.80, 0.82: лучший — четвёртая
    эпоха, и именно её состояние лежит в best_checkpoint.
    """

    settle(stage, train_people=many())

    scripted(monkeypatch, [1.00, 0.90, 0.95, 0.80, 0.82])

    result = train(
        tiny(token_budget=6, early_stopping_patience=10),
        epochs=5, max_steps=None, masking=every_value(),
    )

    assert result["best_val_loss"] == 0.80
    assert result["epoch"] == 5

    assert read(best_checkpoint_path())["epoch"] == 4
    assert read(checkpoint_path())["epoch"] == 5


def test_equal_loss_is_not_an_improvement(stage, monkeypatch):

    settle(stage, train_people=many())

    scripted(monkeypatch, [1.00, 1.00, 1.00])

    result = train(
        tiny(token_budget=6, early_stopping_patience=10),
        epochs=3, max_steps=None, masking=every_value(),
    )

    assert result["best_val_loss"] == 1.00
    assert result["epochs_without_improvement"] == 2
    assert read(best_checkpoint_path())["epoch"] == 1


def test_improvement_must_beat_the_minimum_delta(stage, monkeypatch):
    """
    Граница: при min_delta = 0.1 значение ровно на 0.1 лучше
    улучшением НЕ считается — сравнение строгое.
    """

    settle(stage, train_people=many())

    scripted(monkeypatch, [1.00, 0.90, 0.89])

    result = train(
        tiny(token_budget=6, early_stopping_min_delta=0.1, early_stopping_patience=10),
        epochs=3, max_steps=None, masking=every_value(),
    )

    assert result["best_val_loss"] == 0.89
    assert read(best_checkpoint_path())["epoch"] == 3


# ============================================================
# EARLY STOPPING
# ============================================================


def test_training_stops_right_after_the_patience_runs_out(stage, monkeypatch):
    """
    Терпение 3, после лучшей эпохи три неулучшения подряд:
    обучение обязано остановиться ровно на третьем.
    """

    settle(stage, train_people=many())

    scripted(monkeypatch, [0.90, 0.91, 0.92, 0.93, 0.94])

    result = train(
        tiny(token_budget=6, early_stopping_patience=3),
        epochs=5, max_steps=None, masking=every_value(),
    )

    assert result["reason"] == "early_stopping"
    assert result["epoch"] == 4
    assert result["epochs_without_improvement"] == 3
    assert result["best_val_loss"] == 0.90


def test_an_improvement_resets_the_patience(stage, monkeypatch):

    settle(stage, train_people=many())

    scripted(monkeypatch, [0.90, 0.91, 0.92, 0.50, 0.51, 0.52, 0.53])

    result = train(
        tiny(token_budget=6, early_stopping_patience=3),
        epochs=7, max_steps=None, masking=every_value(),
    )

    assert result["reason"] == "early_stopping"
    assert result["epoch"] == 7
    assert result["best_val_loss"] == 0.50


def test_missing_val_loss_counts_neither_way(stage, monkeypatch):
    """
    Группа без целей сигнала не даёт: ни улучшением, ни
    ухудшением это не считается, и терпение не тратится.
    """

    settle(stage, train_people=many())

    scripted(monkeypatch, [0.90, None, None, None, 0.95])

    result = train(
        tiny(token_budget=6, early_stopping_patience=2),
        epochs=5, max_steps=None, masking=every_value(),
    )

    assert result["reason"] == "epochs"
    assert result["epoch"] == 5
    assert result["best_val_loss"] == 0.90
    assert result["epochs_without_improvement"] == 1


def test_a_paused_epoch_gets_no_validation(stage, monkeypatch):
    """
    Эпоха, прерванная --max-steps, целиком не пройдена: ни
    validation, ни лучшего чекпойнта она не получает.
    """

    settle(stage, train_people=many())

    called = 0

    def fake(model, source, device, token_budget):
        nonlocal called
        called += 1
        return Scores(loss_sum=2.5, targets=5)

    monkeypatch.setattr("src.mlm.train.validate", fake)

    result = train(
        tiny(token_budget=6), epochs=2, max_steps=3, masking=every_value()
    )

    assert called == 0
    assert result["reason"] == "max_steps"
    assert result["best_val_loss"] is None
    assert not best_checkpoint_path().exists()
