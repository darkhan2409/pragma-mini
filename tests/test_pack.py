from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.mlm.inputs import IGNORE
from src.mlm.model import pack

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# pack превращает список клиентов в плоский micro-batch без
# заполнителя. Всё, что связывает токен с его событием и клиентом,
# считается здесь один раз, и дальше модель верит этим числам.
#
# Поэтому каждое отображение сверяется с НЕЗАВИСИМЫМ перебором по
# клиентам, а не с той же формулой: совпадение формулы с собой
# ничего не доказывает.
# ============================================================


CPU = torch.device("cpu")


@pytest.fixture
def packed(clients):
    return pack(clients, CPU), clients


# ============================================================
# НЕЗАВИСИМЫЙ ПЕРЕБОР
# ============================================================


def walk(clients: list) -> dict:
    """
    Те же величины, посчитанные перебором клиент за клиентом.
    """

    event_of_token: list[int] = []
    user_of_event: list[int] = []
    profile_slot: list[int] = []
    event_slot: list[int] = []
    positions: list[float] = []

    targets: list[dict] = []

    token_base = 0
    event_base = 0
    slot = 0

    for number, client in enumerate(clients):

        profile_slot.append(slot)
        positions.append(0.0)
        slot += 1

        for local in range(client.n_events):

            event = event_base + local

            user_of_event.append(number)
            event_slot.append(slot)
            positions.append(float(client.event_time_log[local]))
            slot += 1

            start = int(client.event_starts[local])
            length = int(client.event_lengths[local])

            for inside in range(length):

                place = start + inside

                event_of_token.append(event)

                if int(client.labels[place]) != IGNORE:
                    targets.append(
                        {
                            "token": token_base + place,
                            "event": event,
                            "client": number,
                            "place": place,
                            "local": local,
                            "inside": inside,
                            "label": int(client.labels[place]),
                        }
                    )

        token_base += client.n_tokens
        event_base += client.n_events

    return {
        "event_of_token": event_of_token,
        "user_of_event": user_of_event,
        "profile_slot": profile_slot,
        "event_slot": event_slot,
        "positions": positions,
        "targets": targets,
    }


# ============================================================
# ТОКЕН -> СОБЫТИЕ -> КЛИЕНТ
# ============================================================


def test_token_belongs_to_the_event_that_covers_it(packed):

    data, clients = packed

    assert data.event_of_token.tolist() == walk(clients)["event_of_token"]


def test_event_belongs_to_its_client(packed):

    data, clients = packed

    assert data.user_of_event.tolist() == walk(clients)["user_of_event"]


def test_the_chain_never_leaves_its_client(packed):
    """
    Токен, его событие и его клиент обязаны быть согласованы: у
    токена клиента c событие тоже принадлежит c.
    """

    data, clients = packed

    client_of_token = data.user_of_event[data.event_of_token]

    expected = np.repeat(
        np.arange(len(clients)), [client.n_tokens for client in clients]
    )

    assert client_of_token.tolist() == expected.tolist()


def test_lengths_add_up(packed):

    data, clients = packed

    assert data.clients == len(clients)
    assert data.key_ids.numel() == sum(client.n_tokens for client in clients)
    assert data.events.segments == sum(client.n_events for client in clients)
    assert data.profile_key_ids.numel() == sum(
        client.profile_n_tokens for client in clients
    )
    assert data.calendar.shape == (data.events.segments, 6)


# ============================================================
# СЛОТЫ ИСТОРИИ
# ============================================================


def test_history_slots_cover_every_row_exactly_once(packed):
    """
    Анкеты и события вместе занимают 0..B+E-1 ровно по разу: ни
    одна строка истории не пустует и ни одна не занята дважды.
    """

    data, clients = packed

    width = data.clients + data.events.segments

    seen = sorted(
        data.history_profile_slot.tolist() + data.history_event_slot.tolist()
    )

    assert seen == list(range(width))


def test_client_history_is_its_profile_then_its_events(packed):
    """
    Сегмент истории клиента начинается его анкетой, дальше идут
    только его события — это и проверяет cu_seqlens раскладки.
    """

    data, clients = packed

    bounds = data.history.cu_seqlens.tolist()

    expected = walk(clients)

    assert data.history_profile_slot.tolist() == expected["profile_slot"]
    assert data.history_event_slot.tolist() == expected["event_slot"]

    for number, client in enumerate(clients):

        first, last = bounds[number], bounds[number + 1]

        assert last - first == client.n_events + 1
        assert int(data.history_profile_slot[number]) == first

        mine = data.history_event_slot[data.user_of_event == number].tolist()

        assert mine == list(range(first + 1, last))


def test_profile_slot_sits_at_zero_time_and_events_keep_theirs(packed):
    """
    Анкета якорится на нуле — в тот же момент, что самое свежее
    событие; у событий остаются их собственные лог-секунды.
    """

    data, clients = packed

    expected = walk(clients)["positions"]

    assert data.history_positions.dtype == torch.float32
    assert data.history_positions.tolist() == pytest.approx(expected)
    assert data.history_positions[data.history_profile_slot].tolist() == [0.0] * len(
        clients
    )
    assert data.history_positions[data.history_event_slot].tolist() == pytest.approx(
        data.event_time_log.tolist()
    )


# ============================================================
# ЦЕЛИ
# ============================================================


def test_every_target_is_named_correctly(packed):
    """
    Шесть чисел на цель: где токен в micro-batch, какое у него
    событие, чей он клиент, какое место занимает у клиента, каким
    по счёту идёт его событие и каким токеном он стоит внутри
    события.
    """

    data, clients = packed

    expected = walk(clients)["targets"]

    assert data.target_token.tolist() == [item["token"] for item in expected]
    assert data.target_event.tolist() == [item["event"] for item in expected]
    assert data.target_client.tolist() == [item["client"] for item in expected]
    assert data.target_place.tolist() == [item["place"] for item in expected]
    assert data.target_local.tolist() == [item["local"] for item in expected]
    assert data.target_inside.tolist() == [item["inside"] for item in expected]


def test_target_bucket_and_row_locate_the_event_in_its_rectangle(packed):
    """
    target_bucket и target_row говорят, в какой корзине и в какой
    её строке лежит событие цели. Отсюда энкодер события достаёт
    вектор токена, и ошибка здесь дала бы чужой вектор.
    """

    data, _ = packed

    for number in range(data.target_token.numel()):

        event = int(data.target_event[number])
        bucket = data.events.buckets[int(data.target_bucket[number])]
        row = int(data.target_row[number])

        assert int(bucket.segments[row]) == event

        inside = int(data.target_inside[number])

        assert int(bucket.index[row, inside]) == int(data.target_token[number])
        assert bool(bucket.mask[row, inside])


def test_targets_hold_the_hidden_value_and_never_the_ignore(packed):
    """
    В targets попадают только настоящие цели. Если бы туда
    просочился -100, loss (среднее по неигнорируемым) перестал бы
    быть суммой, делённой на count, и нормировка окна разъехалась
    бы.
    """

    data, clients = packed

    labels = data.labels[data.target_token]

    assert bool((labels != IGNORE).all())
    assert labels.tolist() == [item["label"] for item in walk(clients)["targets"]]
    assert data.target_token.tolist() == sorted(data.target_token.tolist())


def test_client_without_targets_adds_nothing(clients):
    """
    Клиент без целей участвует в проходе, но целей не приносит.
    """

    quiet = [client for client in clients if client.n_targets == 0]

    assert quiet, "в наборе должен быть клиент без целей"

    data = pack(quiet, CPU)

    assert data.target_token.numel() == 0
    assert data.events.segments == sum(client.n_events for client in quiet)


# ============================================================
# ИСПОРЧЕННЫЙ КЛИЕНТ
# ============================================================


def test_events_that_do_not_lie_in_a_row_are_refused(clients):

    broken = replace(clients[0], event_starts=clients[0].event_starts + 1)

    with pytest.raises(ValueError, match="события не лежат подряд"):
        pack([broken], CPU)


def test_events_that_do_not_cover_every_token_are_refused(clients):

    client = clients[0]

    lengths = client.event_lengths.copy()
    lengths[-1] -= 1

    broken = replace(client, event_lengths=lengths)

    with pytest.raises(ValueError, match="не покрывают"):
        pack([broken], CPU)


def test_event_without_a_marker_is_refused(clients):
    """
    Событие нулевой длины невозможно: маркер [EVT] есть всегда.
    """

    made = world.make("broken", [[(world.KEY_A, [10], False)]], [(world.KEY_A, [20])])

    client = made.client

    broken = replace(
        client,
        event_starts=np.array([0, 2], dtype=np.int64),
        event_lengths=np.array([2, 0], dtype=np.int64),
        event_time_log=np.array([1.0, 0.0], dtype=np.float32),
        calendar=world.calendar_of(2),
    )

    with pytest.raises(ValueError, match="события: пустой сегмент"):
        pack([broken], CPU)


def test_profile_without_a_marker_is_refused(clients):
    """
    Анкета нулевой длины невозможна: маркер [USR] есть всегда.
    """

    empty = np.array([], dtype=np.int64)

    broken = replace(
        clients[0],
        profile_key_ids=empty,
        profile_value_ids=empty,
        profile_positions=empty,
        profile_time_log=np.array([], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="анкеты: пустой сегмент"):
        pack([broken], CPU)
