from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.mlm import model as mlm_model
from src.mlm.inputs import IGNORE
from src.mlm.model import Mlm, Model, mlm_loss, pack

from tests import world
from tests.test_isolation import long, short


# ============================================================
# ИДЕЯ
# ============================================================
#
# Корзины существуют ради скорости, а не ради смысла: результат
# обязан совпадать с тем, что даёт честный проход по одному
# сегменту за раз. Это и есть главный regression прохода —
# «те же данные, другой способ счёта, тот же результат».
#
# Сравниваются не только выходы, но и ГРАДИЕНТЫ: ошибка в
# индексах корзины может дать правильный вектор и при этом
# отправить градиент не в тот вес.
#
# Допуск узкий. Измеренное расхождение между двумя способами —
# около 1e-07 на значениях порядка единицы, то есть один-два ulp
# float32; настоящая ошибка индексации даёт на порядки больше.
# ============================================================


CPU = torch.device("cpu")

ATOL = 1e-6


def ones(length: int) -> torch.Tensor:
    return torch.ones(length, dtype=torch.bool)


def reference(model: Model, data) -> dict:
    """
    Тот же проход, но по одному сегменту за раз.

    Прямоугольников здесь нет вовсе: каждое событие, каждая анкета
    и каждая история считаются отдельным вызовом энкодера без
    заполнителя. Это независимый способ получить тот же ответ.
    """

    count = data.events.segments

    dated = []
    tokens: dict[int, torch.Tensor] = {}

    for event in range(count):

        start = int(data.events.cu_seqlens[event])
        length = int(data.events.lengths[event])
        index = torch.arange(start, start + length)

        piece = model.event(
            model.embedding.embed(
                data.key_ids[index], data.value_ids[index],
                data.positions[index], ones(length),
            ).unsqueeze(0),
            torch.zeros(1, length, dtype=torch.bool),
            data.calendar[event : event + 1],
        )

        dated.append(piece.dated[0])

        for number in range(int(data.target_token.numel())):
            if int(data.target_event[number]) == event:
                tokens[number] = piece.tokens[0, int(data.target_inside[number])]

    dated = torch.stack(dated)

    profile = []

    for client in range(data.clients):

        start = int(data.profiles.cu_seqlens[client])
        length = int(data.profiles.lengths[client])
        index = torch.arange(start, start + length)

        profile.append(
            model.profile(
                model.embedding.embed(
                    data.profile_key_ids[index], data.profile_value_ids[index],
                    data.profile_positions[index], ones(length),
                ).unsqueeze(0),
                data.profile_time_log[index].unsqueeze(0),
                ones(length).unsqueeze(0),
            )[0]
        )

    profile = torch.stack(profile)

    width = data.clients + count

    flat = (
        profile.new_zeros(width, profile.shape[-1])
        .index_copy(0, data.history_profile_slot, profile)
        .index_copy(0, data.history_event_slot, dated)
    )

    parts = []
    places = []

    for client in range(data.clients):

        start = int(data.history.cu_seqlens[client])
        length = int(data.history.lengths[client])
        index = torch.arange(start, start + length)

        parts.append(
            model.history(
                flat[index].unsqueeze(0),
                data.history_positions[index].unsqueeze(0),
                None,
            )[0]
        )
        places.append(index)

    out = flat.new_zeros(width, flat.shape[-1]).index_copy(
        0, torch.cat(places), torch.cat(parts)
    )

    client_vectors = out[data.history_profile_slot]
    event_vectors = out[data.history_event_slot]

    if tokens:
        token_vectors = torch.stack([tokens[number] for number in sorted(tokens)])
    else:
        token_vectors = model.embedding.weight[:0]

    logits = model.head(
        token_vectors,
        event_vectors[data.target_event],
        client_vectors[data.target_client],
        model.embedding.weight,
    )

    return {
        "dated": dated,
        "profile": profile,
        "client": client_vectors,
        "event": event_vectors,
        "logits": logits,
    }


def bucketed(model: Model, data) -> dict:

    dated, token_vectors = model._events(data)
    profile = model._profiles(data)
    client_vectors, event_vectors = model._history(data, profile, dated)

    if token_vectors is None:
        token_vectors = client_vectors[:0]

    return {
        "dated": dated,
        "profile": profile,
        "client": client_vectors,
        "event": event_vectors,
        "logits": model.head(
            token_vectors,
            event_vectors[data.target_event],
            client_vectors[data.target_client],
            model.embedding.weight,
        ),
    }


# ============================================================
# КОРЗИНЫ ПРОТИВ ЧЕСТНОГО ПРОХОДА
# ============================================================


@pytest.mark.parametrize("chunk", [512, 2])
def test_buckets_give_the_same_vectors_as_one_segment_at_a_time(chunk: int):
    """
    Корзины и одиночные проходы обязаны сойтись. chunk = 2
    заставляет корзину дробиться на куски: именно там смещение
    target_row на начало куска и может потеряться.
    """

    built = world.model(events_per_chunk=chunk)
    built.eval()

    data = pack([short().client, long().client] + world.clients(), CPU)

    with torch.no_grad():
        left = bucketed(built, data)
        right = reference(built, data)

    for name in left:
        assert torch.allclose(left[name], right[name], atol=ATOL, rtol=ATOL), name


def test_buckets_send_gradients_to_the_same_weights():
    """
    Градиенты обоих способов совпадают повесово.

    Сравнение идёт на двух одинаково разыгранных моделях: иначе
    второй backward лёг бы поверх первого.
    """

    clients = [short().client, long().client] + world.clients()

    left_model = world.model()
    right_model = world.model()

    left_model.eval()
    right_model.eval()

    bucketed(left_model, pack(clients, CPU))["logits"].sum().backward()
    reference(right_model, pack(clients, CPU))["logits"].sum().backward()

    left = dict(left_model.named_parameters())
    right = dict(right_model.named_parameters())

    assert set(left) == set(right)

    touched = 0

    for name, parameter in left.items():

        other = right[name].grad

        if parameter.grad is None and other is None:
            continue

        assert parameter.grad is not None and other is not None, name

        scale = float(parameter.grad.abs().max())

        assert torch.allclose(
            parameter.grad, other, atol=max(ATOL, scale * 1e-5), rtol=1e-5
        ), name

        touched += 1

    # Градиент обязан дойти до всех пяти частей, иначе совпадение
    # ничего не значит.
    assert touched == len(left)


# ============================================================
# ОДИН КЛИЕНТ ПРОТИВ MICRO-BATCH
# ============================================================


def test_client_alone_matches_the_same_client_in_a_micro_batch(model):
    """
    Главный regression: клиент в компании считается так же, как в
    одиночку — те же логиты, те же цели, тот же вклад в потери.
    """

    alone_data = pack([short().client], CPU)
    together_data = pack([short().client, long().client] + world.clients(), CPU)

    with torch.no_grad():
        alone = model(alone_data)
        together = model(together_data)

    mine = together.client == 0

    assert together.place[mine].tolist() == alone.place.tolist()
    assert together.event[mine].tolist() == alone.event.tolist()
    assert together.targets[mine].tolist() == alone.targets.tolist()

    assert torch.allclose(together.logits[mine], alone.logits, atol=ATOL, rtol=ATOL)


@pytest.mark.parametrize("chunk", [None, 3])
def test_loss_of_a_micro_batch_is_the_mean_over_its_targets(model, monkeypatch, chunk):
    """
    loss * count обязан быть суммой потерь по целям: на этом
    держится нормировка окна накопления. В том числе когда потери
    считаются кусками по три цели.
    """

    if chunk is not None:
        monkeypatch.setattr(mlm_model, "TARGETS_PER_CHUNK", chunk)

    data = pack([short().client, long().client] + world.clients(), CPU)

    with torch.no_grad():
        out = model(data)

    assert bool((out.targets != IGNORE).all())

    each = F.cross_entropy(
        out.logits, out.targets, reduction="none",
        label_smoothing=model.label_smoothing,
    )

    assert out.count == each.numel()
    assert float(out.loss) * out.count == pytest.approx(float(each.sum()), rel=1e-6)


# ============================================================
# ПОТЕРИ КУСКАМИ
# ============================================================


def loss_inputs(count: int):
    """
    Голова, её входы и цели в float64: сравнение кусков с целым
    упирается в ulp float64, а не float32.
    """

    generator = torch.Generator().manual_seed(11)

    head = Mlm(world.DIM, world.SEED).double()

    parts = [
        torch.randn(count, world.DIM, generator=generator, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]

    weight = torch.randn(
        world.VOCAB, world.DIM, generator=generator, dtype=torch.float64, requires_grad=True
    )

    targets = torch.randint(0, world.VOCAB, (count,), generator=generator)

    return head, parts, weight, targets


def test_loss_in_pieces_is_the_smoothed_mean_cross_entropy(monkeypatch):
    """
    Куски по три цели с неровным хвостом и кусок из одних -100
    дают ту же потерю и те же градиенты, что кросс-энтропия
    целиком: среднее по строкам с меткой, со сглаживанием.
    """

    monkeypatch.setattr(mlm_model, "TARGETS_PER_CHUNK", 3)

    head, parts, weight, targets = loss_inputs(11)

    targets[3:6] = IGNORE
    targets[9] = IGNORE

    leaves = [*parts, weight, *head.parameters()]

    mine = mlm_loss(head, *parts, weight, targets, 0.1)

    whole = F.cross_entropy(
        head(*parts, weight), targets, ignore_index=IGNORE, label_smoothing=0.1
    )

    assert float(mine.detach()) == pytest.approx(float(whole.detach()), rel=1e-12)

    for left, right in zip(torch.autograd.grad(mine, leaves), torch.autograd.grad(whole, leaves)):
        assert torch.allclose(left, right, atol=1e-12, rtol=1e-10)


def test_loss_without_targets_is_a_connected_zero():

    head, parts, weight, targets = loss_inputs(4)

    targets[:] = IGNORE

    loss = mlm_loss(head, *parts, weight, targets, 0.1)

    assert float(loss.detach()) == 0.0 and loss.requires_grad

    loss.backward()

    assert all(float(part.grad.abs().sum()) == 0.0 for part in parts)


def wide(shapes: list[tuple]) -> list[tuple]:
    """
    Сохранённые тензоры шириной в словарь и длиннее самой таблицы:
    строки по целям, а не вес. Голова законно сохраняет
    weight.t() [d, словарь] — он сюда не попадает.
    """

    return [
        shape for shape in shapes
        if shape[-1:] == (world.VOCAB,) and torch.Size(shape).numel() > world.DIM * world.VOCAB
    ]


def saved_shapes(run) -> list[tuple]:
    """
    Формы всего, что autograd сохранил для backward за run().
    """

    shapes: list[tuple] = []

    def save(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        run()

    return shapes


def test_loss_keeps_no_vocabulary_wide_tensor_for_backward(monkeypatch):
    """
    До backward потери держат только входы кусков: ни одной
    матрицы [цели, словарь]. Контроль — та же кросс-энтропия
    целиком: у неё такая матрица есть, то есть перехват
    сохранённых тензоров действительно её видит.
    """

    monkeypatch.setattr(mlm_model, "TARGETS_PER_CHUNK", 16)

    head, parts, weight, targets = loss_inputs(40)

    whole = saved_shapes(
        lambda: F.cross_entropy(head(*parts, weight), targets, label_smoothing=0.1)
    )

    assert wide(whole)

    mine = saved_shapes(lambda: mlm_loss(head, *parts, weight, targets, 0.1))

    assert mine, "перехват ничего не увидел"
    assert not wide(mine), mine


def test_the_model_takes_its_loss_in_pieces(monkeypatch, clients):
    """
    Тот же запрет на уровне всего прохода: Model.forward берёт
    потери у mlm_loss, а не у полных логитов. Логиты целиком он
    строит для точности, но матрица [цели, словарь] для backward
    не сохраняется. Верни forward кросс-энтропию по logits — здесь
    появится её log_softmax.
    """

    monkeypatch.setattr(mlm_model, "TARGETS_PER_CHUNK", 3)

    built = world.model()
    built.eval()

    data = pack(clients, CPU)

    found: dict = {}

    shapes = saved_shapes(lambda: found.update(out=built(data)))

    assert found["out"].count > max(3, world.DIM)
    assert not wide(shapes), wide(shapes)


# ============================================================
# НАБОР ВЕСОВ
# ============================================================


def test_attention_choice_does_not_change_the_trainable_weights():
    """
    Выбор бэкенда — это способ счёта, а не другая модель:
    varlen-помощники своих весов не заводят.
    """

    base = world.model(attention="sdpa").state_dict()
    other = world.model(attention="flash").state_dict()

    assert list(base) == list(other)

    for name, value in base.items():
        assert value.shape == other[name].shape
        assert torch.equal(value, other[name])


def test_head_is_a_single_projection_onto_the_shared_table():
    """
    У головы ровно два веса, отдельной выходной таблицы нет:
    логиты считаются той же таблицей эмбеддингов.
    """

    head = Mlm(world.DIM, world.SEED)

    assert sorted(head.state_dict()) == ["proj.bias", "proj.weight"]
    assert head.state_dict()["proj.weight"].shape == (world.DIM, 3 * world.DIM)

    built = world.model()

    table = built.embedding.weight

    token = torch.randn(4, world.DIM)
    event = torch.randn(4, world.DIM)
    client = torch.randn(4, world.DIM)

    logits = built.head(token, event, client, table)

    expected = built.head.proj(torch.cat([token, event, client], dim=-1)) @ table.t()

    assert logits.shape == (4, table.shape[0])
    assert torch.equal(logits, expected)
