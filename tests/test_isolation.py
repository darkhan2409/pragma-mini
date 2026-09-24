from __future__ import annotations

import pytest
import torch

from src.mlm.model import pack

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# Плоский micro-batch лежит одним массивом, и границы держатся
# только на масках и раскладке. Если маска потеряется, соседи
# начнут видеть друг друга — и результат останется правдоподобным,
# просто неправильным.
#
# Проверка прямая: меняем данные ЧУЖОГО клиента и требуем, чтобы
# у нашего не сдвинулся ни один бит. Отдельно — то же для событий
# внутри одного клиента.
#
# Клиенты подобраны так, чтобы попасть в ОДНУ корзину разной
# длины: тогда у короткого есть заполнитель, и проверяется в том
# числе он. Заполнитель здесь опасен вдвойне: его индекс смотрит
# на начало собственного сегмента, то есть на настоящее значение.
# ============================================================


CPU = torch.device("cpu")


def short() -> world.Made:
    """
    Два события по три токена, анкета из трёх.
    """

    return world.make(
        "short",
        [
            [(world.KEY_A, [10], False), (world.KEY_B, [11], True)],
            [(world.KEY_A, [12], False), (world.KEY_B, [13], False)],
        ],
        [(world.KEY_A, [20]), (world.KEY_B, [21])],
    )


def long(values: tuple[int, int, int] = (14, 15, 16)) -> world.Made:
    """
    Три события по четыре токена, анкета из четырёх.

    Длины подобраны нарочно: 3 и 4 дают один ключ ceil(log2) = 2,
    поэтому короткий и длинный лежат в одной корзине, и у
    короткого появляется заполнитель.
    """

    return world.make(
        "long",
        [
            [(world.KEY_A, [values[0]], False), (world.KEY_B, [30, 31], False)],
            [(world.KEY_A, [values[1]], True), (world.KEY_C, [32, 33], False)],
            [(world.KEY_A, [values[2]], False), (world.KEY_C, [34, 35], False)],
        ],
        [(world.KEY_A, [22]), (world.KEY_B, [23]), (world.KEY_C, [24])],
    )


def stages(model, clients: list) -> dict:
    """
    Все промежуточные векторы прохода, а не только логиты.

    Сравнивать логиты мало: ошибка изоляции может спрятаться в
    векторе события или клиента и проявиться лишь на некоторых
    целях.
    """

    data = pack(clients, CPU)

    with torch.no_grad():
        dated, tokens = model._events(data)
        profile = model._profiles(data)
        client, event = model._history(data, profile, dated)
        out = model(data)

    return {
        "data": data,
        "dated": dated,
        "tokens": tokens,
        "profile": profile,
        "client": client,
        "event": event,
        "logits": out.logits,
        "target_client": out.client,
    }


def mine(result: dict, number: int) -> dict:
    """
    Только то, что принадлежит клиенту number.
    """

    data = result["data"]

    events = data.user_of_event == number
    targets = result["target_client"] == number

    return {
        "dated": result["dated"][events],
        "profile": result["profile"][number],
        "client": result["client"][number],
        "event": result["event"][events],
        "logits": result["logits"][targets],
    }


# Выходы энкодеров и отдельно логиты: у головы своё матричное
# умножение, и число его строк равно числу целей micro-batch.
VECTORS = ("dated", "profile", "client", "event")

EVERYTHING = VECTORS + ("logits",)


def same(left: dict, right: dict, names: tuple = EVERYTHING) -> None:
    """
    Побитовое совпадение. Требуется там, где форма прямоугольников
    одна и та же: тогда и порядок сложений один и тот же, и
    расхождению взяться неоткуда.
    """

    for name in names:
        assert torch.equal(left[name], right[name]), name


def close(left: dict, right: dict, names: tuple = EVERYTHING) -> None:
    """
    Совпадение в пределах float32. Требуется там, где меняется
    ШИРИНА корзины: с другой шириной меняется порядок сложений
    внутри внимания, и последний бит уплывает.

    Допуск узкий намеренно: измеренное расхождение — 2.4e-07 при
    значениях порядка единицы, то есть один ulp. Настоящая утечка
    дала бы на порядки больше.
    """

    for name in names:
        assert torch.allclose(left[name], right[name], atol=1e-6, rtol=1e-6), name


# ============================================================
# ЧУЖИЕ ДАННЫЕ
# ============================================================


def test_other_client_cannot_change_my_vectors(model):
    """
    Меняем значения длинного клиента и требуем побитового
    совпадения всего, что посчитано для короткого.
    """

    first = stages(model, [short().client, long().client])
    second = stages(model, [short().client, long((17, 18, 19)).client])

    same(mine(first, 0), mine(second, 0))

    # Сам длинный при этом обязан измениться, иначе тест
    # проверял бы, что модель игнорирует вход.
    assert not torch.equal(mine(first, 1)["client"], mine(second, 1)["client"])


def test_other_client_cannot_change_my_vectors_through_the_profile(model):

    first = stages(model, [short().client, long().client])

    other = world.make(
        "long",
        [
            [(world.KEY_A, [14], False), (world.KEY_B, [30, 31], False)],
            [(world.KEY_A, [15], True), (world.KEY_C, [32, 33], False)],
            [(world.KEY_A, [16], False), (world.KEY_C, [34, 35], False)],
        ],
        [(world.KEY_C, [25]), (world.KEY_C, [26]), (world.KEY_A, [27])],
    )

    second = stages(model, [short().client, other.client])

    same(mine(first, 0), mine(second, 0))
    assert not torch.equal(mine(first, 1)["profile"], mine(second, 1)["profile"])


def test_order_of_clients_in_the_batch_changes_nothing(model):
    """
    Соседи бывают с любой стороны. Формы корзин от порядка не
    зависят, поэтому совпадение обязано быть побитовым.
    """

    before = stages(model, [long().client, short().client])
    after = stages(model, [short().client, long().client])

    same(mine(before, 1), mine(after, 0))
    same(mine(before, 0), mine(after, 1))


def test_company_does_not_change_me_beyond_the_last_bit(model):
    """
    В одиночку корзина уже, и порядок сложений другой. Значения
    обязаны совпасть в пределах float32 — но не больше, и это
    честно проверяется узким допуском.
    """

    alone = stages(model, [short().client])
    company = stages(model, [short().client, long().client])

    close(mine(alone, 0), mine(company, 0))


# ============================================================
# СОБЫТИЯ ОДНОГО КЛИЕНТА
# ============================================================


def test_tokens_of_one_event_do_not_see_another(model):
    """
    Внимание события ограничено самим событием. Меняем значения
    третьего события и требуем, чтобы первые два не дрогнули.
    """

    first = stages(model, [long().client])
    second = stages(model, [long((14, 15, 39)).client])

    assert torch.equal(first["dated"][:2], second["dated"][:2])
    assert not torch.equal(first["dated"][2], second["dated"][2])


def test_event_vector_comes_from_the_marker_column(model):
    """
    Вектор события — это колонка [EVT] плюс календарь. Проверяется
    сравнением с прямым вызовом энкодера на одном событии.
    """

    made = long()

    data = pack([made.client], CPU)

    with torch.no_grad():

        dated, _ = model._events(data)

        for number in range(data.events.segments):

            start = int(data.events.cu_seqlens[number])
            length = int(data.events.lengths[number])
            index = torch.arange(start, start + length)

            piece = model.event(
                model.embedding.embed(
                    data.key_ids[index],
                    data.value_ids[index],
                    data.positions[index],
                    torch.ones(length, dtype=torch.bool),
                ).unsqueeze(0),
                torch.zeros(1, length, dtype=torch.bool),
                data.calendar[number : number + 1],
            )

            assert torch.allclose(dated[number], piece.dated[0], atol=1e-6)


# ============================================================
# ИСТОРИЯ
# ============================================================


def test_history_of_one_client_never_reaches_another(model):
    """
    Вектор клиента считается только по его собственной истории.

    Проверка идёт мимо пересчёта векторов событий: в историю
    подаются одни и те же векторы, меняется лишь то, кому они
    принадлежат.
    """

    clients = [short().client, long().client]

    data = pack(clients, CPU)

    width = data.clients + data.events.segments

    torch.manual_seed(3)

    profile = torch.randn(data.clients, world.DIM)
    dated = torch.randn(data.events.segments, world.DIM)

    with torch.no_grad():
        first = model._history(data, profile, dated)

    # Меняются только события ЧУЖОГО клиента.
    other = dated.clone()
    other[data.user_of_event == 1] = torch.randn(int((data.user_of_event == 1).sum()), world.DIM)

    with torch.no_grad():
        second = model._history(data, profile, other)

    assert torch.equal(first[0][0], second[0][0])
    assert torch.equal(first[1][data.user_of_event == 0], second[1][data.user_of_event == 0])
    assert not torch.equal(first[0][1], second[0][1])


def test_content_of_the_padded_rectangle_does_not_leak(model):
    """
    Короткая история лежит в прямоугольнике рядом с длинной, и её
    хвост закрыт маской.

    Ширину прямоугольника задаёт длинный сосед, поэтому при смене
    ЕГО значений ширина не меняется — и результат короткого обязан
    совпасть побитово. Именно так отличается утечка от
    неассоциативности сложения.
    """

    first = stages(model, [short().client, long().client])
    second = stages(model, [short().client, long((17, 18, 19)).client])

    same(mine(first, 0), mine(second, 0))


@pytest.mark.parametrize("count", [2, 3])
def test_number_of_neighbours_changes_nothing(model, count: int):
    """
    Число соседей меняет высоту прямоугольника, но не его ширину,
    поэтому все векторы энкодеров обязаны совпасть побитово.
    """

    one = stages(model, [short().client, long().client])
    many = stages(model, [short().client] + [long().client] * count)

    same(mine(one, 0), mine(many, 0), VECTORS)

    # У логитов число строк равно числу целей micro-batch, и с
    # другим числом строк матричное умножение головы складывает в
    # другом порядке: измеренное расхождение — 1.2e-07, один ulp.
    close(mine(one, 0), mine(many, 0), ("logits",))
