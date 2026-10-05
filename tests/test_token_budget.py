from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.mlm.inputs import Size, cost, micro_batches


# ============================================================
# ИДЕЯ
# ============================================================
#
# Группы строк набора — это единица чтения. Настоящий батч модели
# собирается здесь, по цене клиента в токенах, и от разбивки
# файла зависеть не должен.
#
# Цена — n_tokens + profile_n_tokens + 1: токены событий, токены
# анкеты и слот [USR], как в dynamiq-ai/pragmatiq. Число событий
# в цену не входит, но история клиента (n_events + 1) всё равно
# не длиннее цены. Проверяется и то, что по цене никто не
# теряется, не дублируется и не переставляется.
# ============================================================


def sizes(*numbers: int) -> list[Size]:
    """
    Клиенты, заданные прямо ценой: cost(Size) = сумма + 1.
    """

    return [Size(number - 1, 0) for number in numbers]


def prices(batches: list[list]) -> list[list[int]]:
    return [[cost(item) for item in batch] for batch in batches]


# ============================================================
# ЦЕНА
# ============================================================


def test_price_counts_event_tokens_profile_tokens_and_the_user_slot(clients):
    """
    Слот [USR] — то самое +1. Число событий отдельно не
    считается.
    """

    for client in clients:

        assert cost(client) == client.n_tokens + client.profile_n_tokens + 1


def test_price_does_not_depend_on_the_number_of_events(clients):
    """
    Те же токены, собранные в одно событие вместо многих, стоят
    столько же: в бюджет идут токены, а не события.
    """

    many = max(clients, key=lambda client: client.n_events)

    one = replace(
        many,
        event_starts=many.event_starts[:1],
        event_lengths=np.array([many.n_tokens], dtype=many.event_lengths.dtype),
        event_time_log=many.event_time_log[:1],
        calendar=many.calendar[:1],
        event_time=many.event_time[:1],
    )

    assert many.n_events > 1 and one.n_events == 1
    assert cost(one) == cost(many)


def test_history_of_a_client_fits_in_its_price(clients):
    """
    Почему события можно не считать: у каждого события на нулевой
    позиции [EVT], поэтому событий не больше токенов, и история
    клиента — n_events + 1 позиций — не длиннее его цены. Бюджет
    micro-batch ограничивает и историю.
    """

    for client in clients:

        assert all(length >= 1 for length in client.event_lengths)
        assert client.n_events + 1 <= cost(client)


def test_price_of_a_size_matches_the_price_of_its_client(clients):
    """
    Size несёт ровно те два числа, которые читает cost: подсчёт
    micro-batch'ей до обучения обязан совпасть с тем, что потом
    случится на настоящих клиентах.
    """

    for client in clients:

        short = Size(client.n_tokens, client.profile_n_tokens)

        assert cost(short) == cost(client)


# ============================================================
# РАЗБИВКА
# ============================================================


def test_nobody_is_lost_reordered_or_repeated(clients):

    batches = list(micro_batches(clients, 20))

    seen = [client.client_id for batch in batches for client in batch]

    assert seen == [client.client_id for client in clients]
    assert all(batch for batch in batches)


def test_batch_never_exceeds_the_budget():

    budget = 20

    batches = list(micro_batches(sizes(7, 6, 6, 5, 9), budget))

    assert prices(batches) == [[7, 6, 6], [5, 9]]
    assert all(sum(batch) <= budget for batch in prices(batches))


def test_exact_budget_is_allowed():
    """
    Сравнение строгое: батч ровно по бюджету проходит целиком, и
    off-by-one здесь был бы виден только на границе.
    """

    assert prices(list(micro_batches(sizes(10, 10), 20))) == [[10, 10]]
    assert prices(list(micro_batches(sizes(10, 11), 20))) == [[10], [11]]


def test_a_client_dearer_than_the_whole_budget_goes_alone():
    """
    Такого клиента не разрезать: он идёт один, а соседи к нему не
    прилипают ни спереди, ни сзади.
    """

    batches = prices(list(micro_batches(sizes(5, 50, 6), 20)))

    assert batches == [[5], [50], [6]]


def test_two_oversized_clients_do_not_share_a_batch():

    assert prices(list(micro_batches(sizes(50, 60), 20))) == [[50], [60]]


def test_empty_stream_yields_nothing():

    assert list(micro_batches([], 20)) == []


def test_single_client_is_one_batch(clients):

    batches = list(micro_batches(clients[:1], 10_000))

    assert len(batches) == 1
    assert batches[0] == clients[:1]


# ============================================================
# ХРАНЕНИЕ НЕ ВЛИЯЕТ НА ИСПОЛНЕНИЕ
# ============================================================


@pytest.mark.parametrize("split", [1, 2, 3, 5])
def test_grouping_does_not_depend_on_how_the_stream_was_chunked(clients, split: int):
    """
    Поток клиентов один и тот же, как бы он ни приходил: разбивка
    по micro-batch зависит только от цен и бюджета.
    """

    def chunked():
        for first in range(0, len(clients), split):
            yield from clients[first : first + split]

    budget = 24

    assert prices(list(micro_batches(chunked(), budget))) == prices(
        list(micro_batches(clients, budget))
    )


def test_budget_of_one_puts_every_client_alone(clients):
    """
    Крайний случай: бюджет меньше любой цены — каждый сам по себе.
    """

    batches = list(micro_batches(clients, 1))

    assert [len(batch) for batch in batches] == [1] * len(clients)
