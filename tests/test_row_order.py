from __future__ import annotations

import pytest
import torch

from src.mlm.inputs import Prefetch, Source
from src.mlm.settings import checkpoint_path
from src.mlm.train import horizon, train, train_source

from tests import world
from tests.test_checkpoint_resume import compare, read
from tests.test_scheduler import many
from tests.test_training_math import every_value, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Перестановка групп строк train на эпоху (shuffle_row_groups):
#
#   - без неё проход идёт в порядке файла, как раньше;
#   - одна и та же эпоха даёт тот же порядок, соседняя — другой, и
#     клиенты не теряются и не повторяются;
#   - подготовка впрок в другом процессе идёт тем же порядком;
#   - горизонт cosine — ровно число шагов всех эпох плана;
#   - продолжение посреди переставленной эпохи совпадает с
#     непрерывным обучением.
# ============================================================


def spread(root, groups: int = 6, dropout: float = 0.0) -> None:
    """
    Мир из groups групп строк по два клиента.
    """

    people = many(2 * groups)

    world.install(
        root,
        {
            "train": [people[2 * number: 2 * number + 2] for number in range(groups)],
            "val": [world.population("v")],
        },
        dropout=dropout,
    )


def ids(source) -> list[str]:
    return [client.client_id for client in source.clients()]


def test_without_shuffle_the_pass_follows_the_file(stage):

    spread(stage)

    source = train_source(tiny(), every_value(), epoch=3)

    assert source.order == list(range(6))
    assert ids(source) == [f"c{number}" for number in range(12)]


def test_the_same_epoch_gives_the_same_order_and_the_next_one_another(stage):

    spread(stage)

    config = tiny(shuffle_row_groups=True)

    first = train_source(config, every_value(), epoch=1)
    again = train_source(config, every_value(), epoch=1)
    second = train_source(config, every_value(), epoch=2)

    assert first.order == again.order
    assert first.order != second.order
    assert sorted(first.order) == sorted(second.order) == list(range(6))

    assert ids(first) == ids(again)
    assert sorted(ids(first)) == sorted(ids(second)) == sorted(f"c{number}" for number in range(12))


def test_the_mask_of_a_client_does_not_depend_on_its_place(stage):

    spread(stage)

    masking = every_value()

    plain = {client.client_id: client.labels for client in train_source(tiny(), masking, 1).clients()}
    moved = train_source(tiny(shuffle_row_groups=True), masking, 1)

    for client in moved.clients():
        assert torch.equal(torch.as_tensor(client.labels), torch.as_tensor(plain[client.client_id]))


def test_prefetch_in_another_process_keeps_the_shuffled_order(stage):

    spread(stage)

    source = train_source(tiny(shuffle_row_groups=True), every_value(), epoch=2)

    assert source.order != list(range(6))
    assert ids(Prefetch(source, workers=2)) == ids(source)


def test_the_horizon_is_the_steps_of_the_whole_shuffled_plan(stage):

    spread(stage)

    config = tiny(token_budget=6, shuffle_row_groups=True)
    masking = every_value()

    planned = horizon(config, masking, 2)

    train(config, epochs=2, max_steps=None, masking=masking)

    assert read(checkpoint_path())["step"] == planned


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_resume_inside_a_shuffled_epoch_matches_straight_training(stage, dropout: float):

    config = tiny(token_budget=6, warmup_steps=2, shuffle_row_groups=True)
    masking = every_value()

    spread(stage, dropout=dropout)
    train(config, epochs=2, max_steps=17, masking=masking)
    straight = read(checkpoint_path())

    # 12 клиентов по шагу: 17 и 15 — внутри второй, переставленной эпохи.
    assert straight["epoch_complete"] is False and straight["epoch"] == 2

    spread(stage, dropout=dropout)
    train(config, epochs=2, max_steps=15, masking=masking)
    train(config, epochs=2, max_steps=17, masking=masking, resume=True)

    compare(straight, read(checkpoint_path()))
