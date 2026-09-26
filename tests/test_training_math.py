from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source, micro_batches
from src.mlm.model import pack
from src.mlm.settings import MlmConfig, checkpoint_path
from src.mlm.train import for_epoch, train

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# Окно накопления обязано давать ГРАДИЕНТ СРЕДНЕГО ПО ЦЕЛЯМ, а не
# среднее по micro-batch'ам. Разница видна только тогда, когда
# целей в micro-batch'ах разное число, — поэтому мир здесь
# подобран так, чтобы их было именно разное.
#
# Половина проверок идёт на рецепте (два способа сложить одни и
# те же потери), половина — через настоящий train: шаги, окна и
# клип живут в нём, а не в отдельной функции.
# ============================================================


CPU = torch.device("cpu")


def every_value(seed: int = 5) -> MaskingConfig:
    """
    Маска, при которой цель — каждое значение.

    Так число целей известно заранее, а не зависит от розыгрыша.
    """

    return MaskingConfig(
        seed=seed,
        value_probability=1.0,
        event_probability=0.0,
        key_probability=0.0,
        unknown_probability=0.0,
    )


def tiny(**overrides) -> MlmConfig:
    """
    Конфиг, на котором обучение вообще что-то делает.

    device=cpu обязателен: auto взял бы CUDA. warmup_steps=1 даёт
    первому шагу полный LR, иначе разгон съел бы весь прогон.
    """

    base = dict(
        device="cpu",
        seed=3,
        learning_rate=1e-2,
        min_learning_rate=1e-3,
        warmup_steps=1,
        token_budget=1000,
        grad_accum_steps=1,
        attention_backend="sdpa",
        label_smoothing=0.0,
    )

    base.update(overrides)

    return MlmConfig(**base)


def settle(root: Path, *, train_people, val_people=None, dropout: float = 0.0) -> None:

    world.install(
        root,
        {
            "train": [train_people],
            "val": [val_people if val_people is not None else world.population("v")],
        },
        dropout=dropout,
    )


def weights_of(path: Path) -> dict:

    state = torch.load(path, map_location="cpu", weights_only=True)

    return {name: value.clone() for name, value in state["model_state_dict"].items()}


def fresh(root: Path, config: MlmConfig):

    from src.mlm.model import load_model

    return load_model(
        seed=config.seed,
        events_per_chunk=config.events_per_chunk,
        label_smoothing=config.label_smoothing,
        device=CPU,
        attention_backend="sdpa",
    )


def windows(config: MlmConfig, masking: MaskingConfig) -> list[list]:
    """
    Те же micro-batch'и, что увидит эпоха 1 обучения.
    """

    source = Source("train", masking=for_epoch(masking, 1))

    return list(micro_batches(source.clients(), config.token_budget))


# ============================================================
# НОРМИРОВКА ОКНА
# ============================================================


def uneven(root: Path) -> list[world.Made]:
    """
    Клиенты с заметно разным числом целей.

    При равном числе целей неправильное усреднение по
    micro-batch'ам совпало бы с правильным, и тест ничего бы не
    поймал.
    """

    return [
        world.make("few", [[(world.KEY_A, [10], True)]], [(world.KEY_A, [20])]),
        world.make(
            "many",
            [
                [(world.KEY_A, [11, 12, 13], True), (world.KEY_B, [14, 15], True)],
                [(world.KEY_C, [16, 17, 18, 19], True)],
            ],
            [(world.KEY_B, [21])],
        ),
        world.make(
            "some",
            [[(world.KEY_B, [22, 23], True)]],
            [(world.KEY_C, [24])],
        ),
    ]


def test_window_gradient_is_the_mean_over_targets_not_over_micro_batches(stage):
    """
    Рецепт окна: backward по СУММЕ потерь каждого micro-batch,
    затем деление градиентов на общее число целей.

    Сравнивается с одним backward по среднему сразу по всем целям
    — при тех же самых проходах модели, поэтому расхождение здесь
    может дать только арифметика нормировки.
    """

    settle(stage, train_people=uneven(stage))

    config = tiny(token_budget=14)
    masking = every_value()

    batches = windows(config, masking)

    counts = [
        int((pack(batch, CPU).labels != -100).sum()) for batch in batches
    ]

    assert len(batches) >= 2
    assert len(set(counts)) > 1, "нужны micro-batch'и с разным числом целей"

    # Так считает окно обучения.
    stepwise = fresh(stage, config)
    stepwise.train()

    total = 0

    for batch in batches:
        out = stepwise(pack(batch, CPU))
        if out.count:
            (out.loss * out.count).backward()
            total += out.count

    for parameter in stepwise.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(total)

    # Так считает один большой effective batch.
    at_once = fresh(stage, config)
    at_once.train()

    pieces = []

    for batch in batches:
        out = at_once(pack(batch, CPU))
        if out.count:
            pieces.append(out.loss * out.count)

    (torch.stack(pieces).sum() / total).backward()

    left = dict(stepwise.named_parameters())
    right = dict(at_once.named_parameters())

    for name, parameter in left.items():
        assert torch.allclose(
            parameter.grad, right[name].grad, atol=1e-7, rtol=1e-5
        ), name


def test_mean_over_micro_batches_is_a_different_number(stage):
    """
    Обратная сторона: если бы окно усредняло потери по
    micro-batch'ам, ответ отличался бы. Иначе предыдущий тест
    ничего не проверял.
    """

    settle(stage, train_people=uneven(stage))

    config = tiny(token_budget=14)

    batches = windows(config, every_value())

    model = fresh(stage, config)
    model.eval()

    with torch.no_grad():
        results = [model(pack(batch, CPU)) for batch in batches]

    losses = [out for out in results if out.count]

    by_target = sum(float(out.loss) * out.count for out in losses) / sum(
        out.count for out in losses
    )
    by_batch = sum(float(out.loss) for out in losses) / len(losses)

    assert by_target != pytest.approx(by_batch, rel=1e-4)


def captured(monkeypatch, config: MlmConfig, masking: MaskingConfig) -> tuple[dict, dict]:
    """
    Прогон train с перехватом градиента в момент клипа.

    Клип стоит ровно между делением на число целей и шагом
    оптимизатора, поэтому это и есть тот градиент, которым учится
    модель. Имена берутся из той же сборки модели: порядок
    parameters() и named_parameters() один и тот же.
    """

    taken: list[list[torch.Tensor]] = []

    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):

        parameters = list(parameters)

        taken.append(
            [
                p.grad.clone() if p.grad is not None else None
                for p in parameters
            ]
        )

        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    result = train(config, epochs=1, max_steps=None, masking=masking)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", original)

    assert len(taken) == 1, "тест рассчитан ровно на один шаг оптимизатора"

    names = [name for name, _ in fresh(None, config).named_parameters()]

    return dict(zip(names, taken[0])), weights_of(checkpoint_path()) | {
        "_step": result["step"]
    }


def test_one_big_batch_and_several_micro_batches_train_the_same(stage, monkeypatch):
    """
    Через настоящий train: одна эпоха, один шаг оптимизатора.

    Слева бюджет вмещает всех разом, справа — по клиенту за раз с
    накоплением. Градиент шага обязан совпасть.

    Веса сравниваются отдельно и с оговоркой: первый шаг AdamW
    это почти знак градиента (m / (sqrt(v) + eps)), поэтому
    компонента, у которой градиент около нуля, законно уезжает на
    другую сторону от разницы в последнем бите. Тест не ослабляет
    допуск, а проверяет ИМЕННО ЭТО: разойтись имеют право только
    те компоненты, у которых градиент пренебрежимо мал.
    """

    people = uneven(stage)

    masking = every_value()

    settle(stage, train_people=people)

    whole = tiny(token_budget=10_000, grad_accum_steps=1)

    one_grad, one_weights = captured(monkeypatch, whole, masking)

    settle(stage, train_people=people)

    split = tiny(token_budget=14, grad_accum_steps=len(windows(whole, masking)) + 5)

    assert len(windows(split, masking)) > 1

    split_grad, split_weights = captured(monkeypatch, split, masking)

    assert one_weights["_step"] == split_weights["_step"] == 1

    for name, value in one_grad.items():
        assert torch.allclose(value, split_grad[name], atol=1e-7, rtol=1e-4), name

    for name, value in one_weights.items():

        if name == "_step":
            continue

        apart = (value - split_weights[name]).abs() > 1e-6

        if not bool(apart.any()):
            continue

        assert bool(
            (one_grad[name][apart].abs() < 1e-6).all()
        ), f"{name}: веса разошлись там, где градиент не мал"


# ============================================================
# ОКНА БЕЗ ЦЕЛЕЙ
# ============================================================


def silent(prefix: str = "s") -> list[world.Made]:
    """
    Клиенты, у которых целей не бывает ни при какой маске.
    """

    return [
        world.make(
            f"{prefix}-{number}",
            [[(world.KEY_A, [10 + number], False)]],
            [(world.KEY_A, [20])],
            targetable=False,
        )
        for number in range(3)
    ]


def test_epoch_without_a_single_target_makes_no_step(stage):
    """
    Окно без целей не делает ни шага оптимизатора, ни шага
    расписания: weight decay AdamW иначе сдвинул бы веса без
    обучающего сигнала.
    """

    settle(stage, train_people=silent())

    config = tiny(token_budget=8, weight_decay=0.5)

    before = {
        name: value.clone() for name, value in fresh(stage, config).state_dict().items()
    }

    result = train(config, epochs=1, max_steps=None, masking=every_value())

    assert result["step"] == 0

    after = weights_of(checkpoint_path())

    for name, value in before.items():
        assert torch.equal(value, after[name]), name


def test_quiet_micro_batches_do_not_shift_the_window(stage):
    """
    Два пустых micro-batch'а подряд и нормальный третий: шаг
    делается один и по целям третьего.
    """

    people = silent() + [
        world.make(
            "loud",
            [[(world.KEY_A, [11, 12], True)]],
            [(world.KEY_B, [21])],
        )
    ]

    settle(stage, train_people=people)

    config = tiny(token_budget=8, grad_accum_steps=4)

    assert len(windows(config, every_value())) == 4

    result = train(config, epochs=1, max_steps=None, masking=every_value())

    assert result["step"] == 1


# ============================================================
# НЕПОЛНОЕ ОКНО
# ============================================================


def test_leftover_micro_batches_still_make_a_step(stage):
    """
    Шесть micro-batch'ей при grad_accum_steps = 4 дают два шага:
    полное окно и неполный остаток в конце эпохи.
    """

    people = [
        world.make(
            f"c{number}",
            [[(world.KEY_A, [10 + number], True)]],
            [(world.KEY_A, [20])],
        )
        for number in range(6)
    ]

    settle(stage, train_people=people)

    config = tiny(token_budget=6, grad_accum_steps=4)

    assert len(windows(config, every_value())) == 6

    result = train(config, epochs=1, max_steps=None, masking=every_value())

    assert result["step"] == 2


@pytest.mark.parametrize("accum, expected", [(1, 6), (2, 3), (3, 2), (4, 2), (6, 1), (7, 1)])
def test_step_count_follows_the_accumulation_window(stage, accum: int, expected: int):

    people = [
        world.make(
            f"c{number}",
            [[(world.KEY_A, [10 + number], True)]],
            [(world.KEY_A, [20])],
        )
        for number in range(6)
    ]

    settle(stage, train_people=people)

    config = tiny(token_budget=6, grad_accum_steps=accum)

    assert len(windows(config, every_value())) == 6

    assert train(config, epochs=1, max_steps=None, masking=every_value())["step"] == expected


# ============================================================
# КЛИП
# ============================================================


def test_clip_is_applied_after_the_division_by_target_count(stage, monkeypatch):
    """
    Порядок: накопление -> деление на число целей -> клип -> шаг.

    Проверяется тем, что норма, которую видит клип, равна норме
    УЖЕ ПОДЕЛЁННОГО градиента. Клип до деления дал бы норму ровно
    в число целей раз больше.
    """

    settle(stage, train_people=uneven(stage))

    config = tiny(token_budget=10_000, max_grad_norm=1e-4)
    masking = every_value()

    seen: list[tuple[float, float]] = []

    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):

        parameters = list(parameters)

        before = torch.cat(
            [p.grad.reshape(-1) for p in parameters if p.grad is not None]
        ).norm()

        result = original(parameters, max_norm, *args, **kwargs)

        after = torch.cat(
            [p.grad.reshape(-1) for p in parameters if p.grad is not None]
        ).norm()

        seen.append((float(before), float(after)))

        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    train(config, epochs=1, max_steps=None, masking=masking)

    assert len(seen) == 1

    before, after = seen[0]

    # То же самое, посчитанное отдельно: сумма потерь целей,
    # делённая на их число.
    model = fresh(stage, config)
    model.train()

    total = 0

    for batch in windows(config, masking):
        out = model(pack(batch, CPU))
        if out.count:
            (out.loss * out.count).backward()
            total += out.count

    expected = torch.cat(
        [p.grad.reshape(-1) / total for p in model.parameters() if p.grad is not None]
    ).norm()

    assert before == pytest.approx(float(expected), rel=1e-4)
    assert before > config.max_grad_norm
    assert after == pytest.approx(config.max_grad_norm, rel=1e-5)


def test_clip_leaves_a_small_gradient_alone(stage, monkeypatch):

    settle(stage, train_people=uneven(stage))

    config = tiny(token_budget=10_000, max_grad_norm=1e9)

    seen: list[tuple[float, float]] = []

    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        before = torch.cat(
            [p.grad.reshape(-1) for p in parameters if p.grad is not None]
        ).norm()
        result = original(parameters, max_norm, *args, **kwargs)
        after = torch.cat(
            [p.grad.reshape(-1) for p in parameters if p.grad is not None]
        ).norm()
        seen.append((float(before), float(after)))
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    train(config, epochs=1, max_steps=None, masking=every_value())

    before, after = seen[0]

    assert after == pytest.approx(before, rel=1e-6)


# ============================================================
# ЧИСЛЕННАЯ БЕЗОПАСНОСТЬ
# ============================================================


def test_nothing_turns_into_nan_after_several_steps(stage):
    """
    Несколько эпох подряд на крошечных данных: ни потери, ни веса,
    ни градиенты не уходят в NaN или бесконечность.
    """

    settle(stage, train_people=uneven(stage) + silent())

    config = tiny(token_budget=12, learning_rate=1e-1)

    result = train(config, epochs=3, max_steps=None, masking=every_value())

    assert result["step"] > 0
    assert result["best_val_loss"] is not None
    assert torch.isfinite(torch.tensor(result["best_val_loss"]))

    for name, value in weights_of(checkpoint_path()).items():
        assert bool(torch.isfinite(value).all()), name


@pytest.mark.parametrize("pieces", [1, 2, 5])
def test_extremely_short_and_long_values_stay_finite(stage, pieces: int):
    """
    Крайние формы: значение из одного куска и из пяти, событие из
    одного маркера рядом с ними.
    """

    people = [
        world.make(
            "edge",
            [
                [],
                [(world.KEY_A, list(range(10, 10 + pieces)), True)],
            ],
            [(world.KEY_B, [21])],
        ),
        world.make("plain", [[(world.KEY_C, [30], True)]], [(world.KEY_C, [31])]),
    ]

    settle(stage, train_people=people)

    config = tiny(token_budget=9)

    result = train(config, epochs=1, max_steps=None, masking=every_value())

    assert result["step"] >= 1

    for name, value in weights_of(checkpoint_path()).items():
        assert bool(torch.isfinite(value).all()), name


def test_training_starts_from_stage_09_weights_and_moves_the_table(stage):
    """
    Снимка векторов у этапа 09 нет: модель считает вход по номерам
    токенов и его weights.pt. Обучение начинает ровно с этой
    таблицы и двигает её градиентом, а сам файл этапа не трогает.
    """

    from src.embedding.settings import WEIGHTS_FILE, embeddings_dir

    settle(stage, train_people=world.population())

    path = embeddings_dir("train") / WEIGHTS_FILE

    table = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]["table.weight"]

    config = tiny()

    assert torch.equal(fresh(stage, config).embedding.weight.detach(), table)

    result = train(config, epochs=1, max_steps=1, masking=every_value())

    assert result["step"] == 1

    trained = weights_of(checkpoint_path())["embedding.table.weight"]

    assert not torch.equal(trained, table)

    on_disk = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]["table.weight"]

    assert torch.equal(on_disk, table)
