from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.batching.settings import BATCHES_FILE, batches_dir
from src.masking.settings import MASKED_FILE, masked_dir
from src.mlm.inputs import IGNORE, InputError, Source, _check, micro_batches

from tests import world
from tests.test_training_math import every_value


# ============================================================
# ИДЕЯ
# ============================================================
#
# Source сводит два файла, собранных разными этапами, и доверять
# их совпадению нельзя: строки сверяются по клиенту и по номеру
# батча, а не по позиции.
#
# Здесь же проверяются инварианты цели — то, из-за чего обучение
# было бы бессмысленным, а не просто неточным: цель обязана быть
# закрыта [MASK] на входе и лежать внутри события, которому
# разрешено быть целью.
#
# Заполнитель до модели не доходит: клиент отрезается по своим
# настоящим длинам.
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


def test_padding_is_cut_off_before_the_model(stage):
    """
    В файле строки выровнены по самому длинному клиенту батча; до
    модели доходят только настоящие длины.
    """

    people = three()

    settle(stage, [people])

    width = max(made.client.n_tokens for made in people)

    assert width > min(made.client.n_tokens for made in people)

    for made, client in zip(people, Source("train").clients()):

        assert client.client_id == made.client.client_id
        assert client.n_tokens == made.client.n_tokens
        assert client.n_events == made.client.n_events
        assert client.profile_n_tokens == made.client.profile_n_tokens

        assert client.key_ids.tolist() == made.client.key_ids.tolist()
        assert client.labels.tolist() == made.client.labels.tolist()
        assert client.calendar.shape == (client.n_events, 6)
        assert world.PAD not in client.key_ids.tolist()


def test_sizes_agree_with_the_clients_they_describe(stage):
    """
    sizes читает три колонки без масок — по ним считается число
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
        assert size.n_events == client.n_events


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
    Группы строк — это хранение. Одни и те же клиенты, разложенные
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
# ДВА ФАЙЛА ОБЯЗАНЫ СОВПАДАТЬ
# ============================================================


def test_missing_input_names_the_command_that_makes_it(stage):

    settle(stage, [three()])

    (masked_dir("train") / MASKED_FILE).unlink()

    with pytest.raises(InputError, match="python -m src.masking.run train"):
        Source("train")


def test_a_file_of_another_schema_is_refused(stage):

    settle(stage, [three()])

    path = masked_dir("train") / MASKED_FILE

    pq.write_table(pa.table({"client_id": ["a"]}), path)

    with pytest.raises(InputError, match="собран другой схемой"):
        Source("train")


def test_files_built_at_different_times_are_refused(stage):
    """
    Разное число групп строк значит, что батчи и маски собраны из
    разных данных.
    """

    people = three()

    settle(stage, [people])

    world.write_masked(masked_dir("train") / MASKED_FILE, [people[:1], people[1:]])

    with pytest.raises(InputError, match="собраны в разное время"):
        Source("train")


def test_rows_are_matched_by_client_not_by_position(stage):

    people = three()

    settle(stage, [people])

    world.write_masked(
        masked_dir("train") / MASKED_FILE, [[people[1], people[0], people[2]]]
    )

    with pytest.raises(InputError, match="в батчах клиент a, а в масках b"):
        list(Source("train").clients())


def test_a_wrong_batch_number_inside_the_file_is_caught(stage):

    settle(stage, [three()])

    path = batches_dir("train") / BATCHES_FILE

    table = pq.read_table(path)

    broken = table.set_column(
        table.schema.get_field_index("batch_index"),
        "batch_index",
        pa.array([7] * table.num_rows, type=pa.int32()),
    )

    pq.write_table(broken, path)

    with pytest.raises(InputError, match="batch_index батчей равен 7"):
        list(Source("train").clients())


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
        _check(
            replace(client, value_ids=values),
            np.ones(client.n_events, dtype=bool),
            np.ones(client.n_events, dtype=bool),
            world.MASK,
        )


def test_a_target_outside_the_target_window_is_refused():

    client = three()[0].client

    with pytest.raises(InputError, match="вне периода целей"):
        _check(
            client,
            np.ones(client.n_events, dtype=bool),
            np.zeros(client.n_events, dtype=bool),
            world.MASK,
        )


def test_padding_among_the_events_is_refused():

    client = three()[1].client

    mask = np.ones(client.n_events, dtype=bool)
    mask[-1] = False

    with pytest.raises(InputError, match="есть заполнитель"):
        _check(client, mask, np.ones(client.n_events, dtype=bool), world.MASK)


def test_a_client_without_targets_passes_every_check():

    client = three()[2].client

    assert client.n_targets == 0

    _check(
        client,
        np.ones(client.n_events, dtype=bool),
        np.zeros(client.n_events, dtype=bool),
        world.MASK,
    )


def test_real_files_satisfy_the_invariants(stage):
    """
    Тот же разбор, но на файлах: каждая цель закрыта [MASK], у неё
    есть событие, и это событие разрешено.
    """

    settle(stage, [three()])

    for client in Source("train").clients():

        where = np.nonzero(client.labels != IGNORE)[0]

        assert bool((client.value_ids[where] == world.MASK).all())

        owner = np.searchsorted(client.event_starts, where, side="right") - 1

        inside = (where >= client.event_starts[owner]) & (
            where < client.event_starts[owner] + client.event_lengths[owner]
        )

        assert bool(inside.all())


def test_dynamic_masking_reads_only_the_batches(stage):
    """
    Обучение маску train из файла не читает: 08_masked ему не
    нужен вовсе.
    """

    settle(stage, [three()])

    (masked_dir("train") / MASKED_FILE).unlink()

    source = Source("train", masking=every_value())

    assert source.masked_path is None

    clients = list(source.clients())

    assert clients
    assert sum(client.n_targets for client in clients) > 0
