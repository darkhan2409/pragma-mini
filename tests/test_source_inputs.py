from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.dataset.settings import SAMPLES_FILE, dataset_dir
from src.mlm.inputs import IGNORE, InputError, Source, _check, micro_batches

from tests import world
from tests.test_training_math import every_value


# ============================================================
# ИДЕЯ
# ============================================================
#
# Source читает один файл — набор 05 — и считает при чтении время
# и маску. Клиент доходит до модели ровно таким, каким лежит в
# наборе, без заполнителя; группа строк — только единица чтения.
#
# Здесь же проверяются инварианты цели — то, из-за чего обучение
# было бы бессмысленным, а не просто неточным: цель обязана быть
# закрыта [MASK] на входе и лежать внутри события, которому
# разрешено быть целью.
# ============================================================


def settle(root, people: list[list[world.Made]], val: list[world.Made] | None = None) -> None:

    world.install(
        root,
        {
            "train": people,
            "val": [val if val is not None else world.population("v")],
        },
    )


def three() -> list[world.Made]:

    return [
        world.make(
            "a",
            [[(world.KEY_A, [10], True), (world.KEY_B, [11, 12], False)]],
            [(world.KEY_A, [20])],
        ),
        world.make(
            "b",
            [[(world.KEY_C, [13], False)], [(world.KEY_A, [14, 15, 16], True)]],
            [(world.KEY_B, [21]), (world.KEY_C, [22])],
        ),
        world.make("c", [[]], [(world.KEY_A, [23])]),
    ]


# ============================================================
# ЧТО ДОХОДИТ ДО МОДЕЛИ
# ============================================================


def test_each_client_reaches_the_model_as_stored(stage):
    """
    Клиенты разной длины в одной группе строк: до модели доходит
    ровно то, что лежит в наборе, без заполнителя.
    """

    people = three()

    settle(stage, [people])

    assert len({made.client.n_tokens for made in people}) > 1

    for made, client in zip(people, Source("train").clients()):

        assert client.client_id == made.client.client_id
        assert client.n_tokens == made.client.n_tokens
        assert client.n_events == made.client.n_events
        assert client.profile_n_tokens == made.client.profile_n_tokens

        assert client.key_ids.tolist() == made.client.key_ids.tolist()
        assert client.positions.tolist() == made.client.positions.tolist()
        assert client.event_starts.tolist() == made.client.event_starts.tolist()
        assert client.profile_value_ids.tolist() == made.client.profile_value_ids.tolist()
        assert client.calendar.shape == (client.n_events, 6)
        assert world.PAD not in client.key_ids.tolist()

        # Маска меняет только значения: вне целей и [UNK] видно
        # исходное значение набора.
        kept = (client.labels == IGNORE) & (client.value_ids != world.MASK) & (client.value_ids != world.UNK)
        assert client.value_ids[kept].tolist() == made.value_ids_source[kept].tolist()


def test_sizes_agree_with_the_clients_they_describe(stage):
    """
    sizes читает три колонки без времени и масок — по ним считается число
    micro-batch'ей до обучения. Разойтись с настоящими клиентами
    они не имеют права.
    """

    settle(stage, [three()[:2], three()[2:]])

    short = list(Source("train").sizes())
    full = list(Source("train").clients())

    assert len(short) == len(full)

    for size, client in zip(short, full):
        assert size.n_tokens == client.n_tokens
        assert size.profile_n_tokens == client.profile_n_tokens


def test_row_groups_are_batches(stage):

    settle(stage, [three()[:2], three()[2:]])

    source = Source("train")

    assert source.count == 2
    assert [client.client_id for client in source.batch(0)] == ["a", "b"]
    assert [client.client_id for client in source.batch(1)] == ["c"]

    for index in range(source.count):
        for client in source.batch(index):
            assert client.batch_index == index


def test_storage_layout_does_not_change_the_micro_batches(stage):
    """
    Группы строк — единица чтения. Одни и те же клиенты, разложенные
    по файлу иначе, обязаны собраться в те же micro-batch'и.
    """

    people = three()

    settle(stage, [people])

    one = [
        [client.client_id for client in batch]
        for batch in micro_batches(Source("train").clients(), 14)
    ]

    settle(stage, [people[:1], people[1:2], people[2:]])

    many = [
        [client.client_id for client in batch]
        for batch in micro_batches(Source("train").clients(), 14)
    ]

    assert one == many
    assert len(one) > 1


def test_asking_for_a_batch_that_is_not_there(stage):

    settle(stage, [three()])

    with pytest.raises(InputError, match="номера от 0 до 0"):
        Source("train").batch(1)


# ============================================================
# НАБОР, КОТОРЫЙ ПРОЧИТАТЬ НЕЛЬЗЯ
# ============================================================


def test_missing_input_names_the_command_that_makes_it(stage):

    settle(stage, [three()])

    (dataset_dir("train") / SAMPLES_FILE).unlink()

    with pytest.raises(InputError, match="python -m src.dataset.run train"):
        Source("train")


def test_a_file_of_another_schema_is_refused(stage):

    settle(stage, [three()])

    pq.write_table(pa.table({"client_id": ["a"]}), dataset_dir("train") / SAMPLES_FILE)

    with pytest.raises(InputError, match="другой схемой"):
        Source("train")


# ============================================================
# ИНВАРИАНТЫ ЦЕЛИ
# ============================================================


def client_of(made: world.Made):
    return made.client


def test_a_target_must_be_closed_by_mask_on_the_input():
    """
    Если бы на входе стояло настоящее значение, модель видела бы
    то, что должна предсказать.
    """

    made = three()[0]

    client = made.client

    values = client.value_ids.copy()
    values[client.labels != IGNORE] = 42

    with pytest.raises(InputError, match="у цели на входе стоит не"):
        _check(replace(client, value_ids=values), np.ones(client.n_events, dtype=bool), world.MASK)


def test_a_target_outside_the_target_window_is_refused():

    client = three()[0].client

    with pytest.raises(InputError, match="вне периода целей"):
        _check(client, np.zeros(client.n_events, dtype=bool), world.MASK)


def test_a_client_without_targets_passes_every_check():

    client = three()[2].client

    assert client.n_targets == 0

    _check(client, np.zeros(client.n_events, dtype=bool), world.MASK)


def test_real_files_satisfy_the_invariants(stage):
    """
    Тот же разбор, но на наборе с маской при чтении: каждая цель
    закрыта [MASK], у неё есть событие, и это событие разрешено.
    """

    settle(stage, [three()])

    for client in Source("train", masking=every_value()).clients():

        where = np.nonzero(client.labels != IGNORE)[0]

        assert bool((client.value_ids[where] == world.MASK).all())

        owner = np.searchsorted(client.event_starts, where, side="right") - 1

        inside = (where >= client.event_starts[owner]) & (
            where < client.event_starts[owner] + client.event_lengths[owner]
        )

        assert bool(inside.all())


def test_every_group_is_masked_while_reading(stage):
    """
    Файла масок нет ни у одной группы: и train, и val получают цели
    при чтении набора.
    """

    settle(stage, [three()], val=three())

    for group in ("train", "val"):

        clients = list(Source(group, masking=every_value()).clients())

        assert clients
        assert sum(client.n_targets for client in clients) > 0, group
