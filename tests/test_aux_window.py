from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

import src.mlm.model as model_module
import src.mlm.train as train_module
from src.mlm.model import RECENT_DAYS, RecentTypes, pack
from src.mlm.settings import TELEMETRY_FILE, MlmConfig, checkpoint_path, train_dir
from src.mlm.train import clients_per_window, train

from tests import world
from tests.test_checkpoint_resume import compare, read
from tests.test_recent_types import head
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, fresh, settle, tiny, uneven, windows


# ============================================================
# ИДЕЯ
# ============================================================
#
# aux_per_client делает вес клиента во вспомогательной цели [USR]
# (E1′) одинаковым за эпоху: член E1′ окна — w · Σ aux_клиента / K,
# где K — среднее число клиентов на окно эпохи (клиенты эпохи / окна
# эпохи), одно число на всю эпоху. MLM остаётся средним по ЦЕЛЯМ окна.
# Тогда шаг не зависит от того, как окно разбито на micro-batch'и, у
# каждого клиента эпохи вес w/K, и клиент без целей MLM всё равно
# учит голову [USR]. Деление на клиентов самого окна (C) этого не
# даёт: при grad_accum_steps=1 окно с одним длинным клиентом весило бы
# столько же, сколько окно с десятью короткими.
#
# Выключенный флаг — шаг эталона побитно: формула прежнего кода
# повторяется здесь руками и сравнивается через torch.equal.
#
# Градиент шага перехватывается у clip_grad_norm_: клип стоит между
# нормировкой окна и шагом оптимизатора.
# ============================================================


WEIGHT = 0.5


def with_head(monkeypatch) -> list[RecentTypes]:
    """
    Голова [USR] из мира с типами (значения key_a). Возвращает список,
    в который ложится каждая собранная голова.
    """

    built: list[RecentTypes] = []

    def make(artifacts, dim, seed):
        recent = head(dim)
        built.append(recent)
        return recent

    monkeypatch.setattr(model_module, "recent_types", make)

    return built


def names(config: MlmConfig) -> list[str]:
    """
    Имена параметров в порядке model.parameters() обучения: голова
    [USR] пристёгивается последней.
    """

    model = fresh(None, config)
    model.attach_recent(head(int(model.embedding.dim)))

    return [name for name, _ in model.named_parameters()]


def step_grads(monkeypatch, config: MlmConfig, masking) -> dict[str, torch.Tensor]:
    """
    Градиент единственного шага оптимизатора эпохи, по именам.
    """

    taken: list[list] = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        taken.append([p.grad.clone() if p.grad is not None else None for p in parameters])
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    train(config, epochs=1, max_steps=None, masking=masking)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", original)

    assert len(taken) == 1, "тест рассчитан ровно на один шаг оптимизатора"

    labels = names(config)

    assert len(labels) == len(taken[0])

    return dict(zip(labels, taken[0]))


def one_window(token_budget: int, masking, **overrides) -> MlmConfig:
    """
    Конфиг, у которого вся эпоха — одно окно оптимизатора.
    """

    probe = tiny(token_budget=token_budget)
    count = len(windows(probe, masking))

    return tiny(
        token_budget=token_budget, grad_accum_steps=count, usr_aux_weight=WEIGHT, **overrides
    )


# ============================================================
# ОКНО НЕ ЗАВИСИТ ОТ РАЗБИЕНИЯ
# ============================================================


def test_window_split_does_not_change_the_step(stage, monkeypatch):
    """
    Одно окно одним micro-batch'ем и то же окно из нескольких: градиент
    шага тот же — в допуске теста без цели [USR]
    (test_one_big_batch_and_several_micro_batches_train_the_same).
    """

    with_head(monkeypatch)

    people = uneven(stage)
    masking = every_value()

    settle(stage, train_people=people)
    whole = one_window(10_000, masking, aux_per_client=True)
    assert whole.grad_accum_steps == 1
    one = step_grads(monkeypatch, whole, masking)

    settle(stage, train_people=people)
    split = one_window(14, masking, aux_per_client=True)
    assert split.grad_accum_steps > 1
    many_grads = step_grads(monkeypatch, split, masking)

    assert float(one["recent.proj.weight"].abs().max()) > 0

    for name, value in one.items():
        assert torch.allclose(value, many_grads[name], atol=1e-7, rtol=1e-4), name


# ============================================================
# ВЕС КЛИЕНТА — 1/C
# ============================================================


def client_mean_aux(masking, config: MlmConfig) -> dict[str, torch.Tensor]:
    """
    Градиент головы [USR] от w · среднего по клиентам окна (среднего
    ячеек клиента), посчитанный по каждому клиенту отдельно.

    Проекция головы получает градиент только от E1′, поэтому её
    градиент шага и есть градиент члена E1′.
    """

    model = fresh(None, config)
    recent = head(int(model.embedding.dim))
    model.attach_recent(recent)
    model.train()

    seen: list[tuple] = []
    hook = recent.register_forward_hook(lambda module, inputs, output: seen.append(inputs))

    clients = [client for batch in windows(config, masking) for client in batch]

    values = []

    for client in clients:
        seen.clear()
        model(pack([client], CPU), logits=False)
        data, usr = seen[0]
        predicted = recent.proj(usr).float().view(1, len(RECENT_DAYS) + 1, recent.types)
        cells = F.mse_loss(predicted, recent.targets(data), reduction="none")
        values.append(cells.mean())

    hook.remove()

    objective = WEIGHT * torch.stack(values).mean()
    weight, bias = torch.autograd.grad(objective, [recent.proj.weight, recent.proj.bias])

    return {"recent.proj.weight": weight, "recent.proj.bias": bias}


@pytest.mark.parametrize("token_budget", [10_000, 14, 6])
def test_client_aux_weight_is_one_over_window_clients(stage, monkeypatch, token_budget: int):
    """
    Эпоха — одно окно, поэтому K = C, и член E1′ шага — w · среднее по
    клиентам окна, какими бы ни были micro-batch'и: один на всех,
    несколько разного размера или по клиенту. Вес клиента не зависит
    ни от соседей по micro-batch'у, ни от числа его целей MLM.
    """

    with_head(monkeypatch)

    people = uneven(stage)
    masking = every_value()

    settle(stage, train_people=people)
    config = one_window(token_budget, masking, aux_per_client=True)

    grads = step_grads(monkeypatch, config, masking)
    expected = client_mean_aux(masking, config)

    for name, value in expected.items():
        assert torch.allclose(grads[name], value, atol=1e-7, rtol=1e-4), name


# ============================================================
# MICRO-BATCH БЕЗ ЦЕЛЕЙ MLM
# ============================================================


def test_micro_batch_without_targets_still_trains_aux(stage, monkeypatch):
    """
    У клиента без целей MLM цели [USR] есть: его micro-batch делает
    backward по E1′, и проекция головы получает от него градиент.
    """

    built = with_head(monkeypatch)

    people = many(2) + [
        world.make("quiet", [[(world.KEY_A, [15], False)]], [(world.KEY_A, [20])], targetable=False)
    ]
    masking = every_value()

    settle(stage, train_people=people)
    config = one_window(6, masking, aux_per_client=True)

    batches = windows(config, masking)
    quiet = [number for number, batch in enumerate(batches)
             if sum(client.n_targets for client in batch) == 0]

    assert len(quiet) == 1 and len(batches) == 3, "нужен отдельный micro-batch без целей"

    passes: list[torch.Tensor] = []
    original = model_module.Model.attach_recent

    def attach(self, recent):
        original(self, recent)
        self.recent.proj.weight.register_hook(lambda grad: passes.append(grad.clone()))

    monkeypatch.setattr(model_module.Model, "attach_recent", attach)

    grads = step_grads(monkeypatch, config, masking)

    assert len(built) == 1
    assert len(passes) == len(batches), "у каждого micro-batch'а свой backward"
    assert float(passes[quiet[0]].abs().max()) > 0
    assert torch.allclose(sum(passes), grads["recent.proj.weight"], atol=1e-7, rtol=1e-5)


# ============================================================
# ФЛАГ ВЫКЛЮЧЕН — ПРЕЖНИЙ ШАГ
# ============================================================


def old_recent_forward(self: RecentTypes, data, usr: torch.Tensor) -> torch.Tensor:
    """
    RecentTypes.forward до 2026-11-03: среднее по всем ячейкам
    micro-batch'а.
    """

    predicted = self.proj(usr).float().view(data.clients, len(RECENT_DAYS) + 1, self.types)

    return F.mse_loss(predicted, self.targets(data))


def test_aux_off_is_the_previous_step(stage, monkeypatch):
    """
    Флаг выключен: градиент шага побитно равен прежней формуле —
    (loss + w · mse) · count на micro-batch с целями и деление на число
    целей окна.
    """

    with_head(monkeypatch)

    people = uneven(stage)
    masking = every_value()

    settle(stage, train_people=people)
    config = one_window(14, masking)

    assert config.aux_per_client is False
    assert config.grad_accum_steps > 1

    grads = step_grads(monkeypatch, config, masking)

    model = fresh(None, config)
    model.attach_recent(head(int(model.embedding.dim)))
    model.train()

    targets = 0

    with monkeypatch.context() as patch:

        patch.setattr(RecentTypes, "forward", old_recent_forward)

        for clients in windows(config, masking):
            out = model(pack(clients, CPU), logits=False)
            if out.count > 0:
                ((out.loss + WEIGHT * out.aux) * out.count).backward()
                targets += out.count

    torch._foreach_div_([p.grad for p in model.parameters() if p.grad is not None], targets)

    expected = dict(model.named_parameters())

    assert set(expected) == set(grads)

    for name, parameter in expected.items():
        assert torch.equal(grads[name], parameter.grad), name


def test_aux_per_client_is_off_by_default_and_travels_in_the_config():

    assert MlmConfig().aux_per_client is False
    assert MlmConfig.from_dict({}).aux_per_client is False
    assert MlmConfig.from_dict({"aux_per_client": True}).aux_per_client is True
    assert MlmConfig(aux_per_client=True).as_dict()["aux_per_client"] is True


# ============================================================
# RESUME ПОСРЕДИ ЭПОХИ
# ============================================================


def steps_telemetry() -> list[dict]:
    """
    Строки шагов без часов: время и ожидание у прогонов разные.
    """

    rows = [json.loads(line) for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()]

    return [
        {key: value for key, value in row.items() if key not in ("time", "seconds", "wait_seconds")}
        for row in rows if row["kind"] == "step"
    ]


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_resume_mid_epoch_with_aux_per_client(stage, monkeypatch, dropout: float):
    """
    grad_accum_steps=3, семь micro-batch'ей на эпоху: окна 3 + 3 + 1,
    K = 7/3. Остановка после первого окна второй эпохи и --resume:
    веса, AdamW, расписание и телеметрия шагов те же, что без
    остановки, и продолжение берёт для эпохи то же K.

    Совпадение окон здесь держится по построению: остановка по
    --max-steps всегда приходится на границу окна (шаг закрывает
    окно), а окна отсчитываются от начала эпохи. Тест ловит другое —
    повторное обучение пропущенных micro-batch'ей, другое K или другие
    T у продолжения.
    """

    with_head(monkeypatch)

    scales: list[tuple[int, float]] = []
    original = train_module.clients_per_window

    def spy(config, masking, epoch):
        value = original(config, masking, epoch)
        scales.append((epoch, value))
        return value

    monkeypatch.setattr(train_module, "clients_per_window", spy)

    people = many(7)
    masking = every_value()

    config = tiny(
        token_budget=6, grad_accum_steps=3, warmup_steps=2, usr_aux_weight=WEIGHT,
        aux_per_client=True,
    )

    settle(stage, train_people=people, dropout=dropout)

    assert len(windows(config, masking)) == 7

    train(config, epochs=2, max_steps=None, masking=masking)

    straight, straight_steps = read(checkpoint_path()), steps_telemetry()

    assert scales == [(1, 7 / 3), (2, 7 / 3)]

    assert straight["step"] == 6
    assert [row["micro_batches"] for row in straight_steps] == [3, 3, 1] * 2

    settle(stage, train_people=people, dropout=dropout)

    train(config, epochs=2, max_steps=4, masking=masking)

    paused = read(checkpoint_path())

    assert paused["epoch_complete"] is False
    assert paused["epoch"] == 2
    assert paused["micro_batches_done"] == 3

    scales.clear()

    train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    assert scales == [(2, 7 / 3)]

    compare(straight, read(checkpoint_path()))

    assert steps_telemetry() == straight_steps


# ============================================================
# ВЕС КЛИЕНТА ЗА ЭПОХУ — w/K
# ============================================================


def mixed(stage) -> list[world.Made]:
    """
    Клиенты, которых бюджет 18 собирает в micro-batch'и по 1, 1, 3 и
    2: окна эпохи разного состава.
    """

    return uneven(stage) + many(4)


def test_clients_per_window_is_clients_over_windows(stage):
    """
    K — клиенты эпохи на окна эпохи, по тем же длинам и той же упаковке,
    что горизонт расписания.
    """

    settle(stage, train_people=mixed(stage))

    masking = every_value()

    assert [len(batch) for batch in windows(tiny(token_budget=18), masking)] == [1, 1, 3, 2]

    for accum, expected in ((1, 7 / 4), (2, 7 / 2), (3, 7 / 2), (4, 7 / 1)):
        config = tiny(token_budget=18, grad_accum_steps=accum, aux_per_client=True)
        assert clients_per_window(config, masking, 1) == expected, accum

    settle(stage, train_people=many(7))

    config = tiny(token_budget=6, grad_accum_steps=3, aux_per_client=True)

    assert clients_per_window(config, masking, 1) == 7 / 3


@pytest.mark.parametrize("accum", [1, 2, 3])
def test_every_client_weighs_the_same_over_the_epoch(stage, monkeypatch, accum: int):
    """
    Коэффициент при aux каждого клиента в backward — производная потери
    шага по его aux — за эпоху равен w/K, сидит ли клиент один в
    micro-batch'е или с соседями, один в окне или в окне из многих.
    При делении на клиентов окна (C) у клиента окна из одного было бы
    w, а у клиента окна из трёх — w/3.
    """

    with_head(monkeypatch)

    masking = every_value()

    settle(stage, train_people=mixed(stage))

    config = tiny(token_budget=18, grad_accum_steps=accum, usr_aux_weight=WEIGHT, aux_per_client=True)

    batches = windows(config, masking)
    clients = sum(len(batch) for batch in batches)
    scale = clients / -(-len(batches) // accum)

    coefficients: list[torch.Tensor] = []
    original = train_module.client_aux

    def spy(errors):
        value = original(errors)
        if value.requires_grad:
            value.register_hook(lambda grad: coefficients.append(grad.clone()))
        return value

    monkeypatch.setattr(train_module, "client_aux", spy)

    train(config, epochs=1, max_steps=None, masking=masking)

    each = torch.cat(coefficients)

    assert [len(part) for part in coefficients] == [len(batch) for batch in batches]
    assert torch.allclose(each, torch.full((clients,), WEIGHT / scale), rtol=1e-6, atol=0)
