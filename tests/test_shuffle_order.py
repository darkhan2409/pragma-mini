from __future__ import annotations

import json
import math

import pytest

from src.masking.settings import MaskingConfig
from src.mlm.inputs import Prefetch, Size, micro_batches
from src.mlm.settings import TELEMETRY_FILE, MlmConfig, checkpoint_path, train_dir
from src.mlm.train import horizon, train, train_source

from tests import world
from tests.test_checkpoint_resume import compare, read
from tests.test_training_math import every_value, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# shuffle_order переставляет клиентов train каждую эпоху: группы
# строк — перестановкой из stable_hash("order", seed, эпоха),
# клиентов внутри группы — перестановкой из seed эпохи и номера
# группы. Эпоха повторяется побитно (на этом стоят resume, K эпохи и
# горизонт cosine), соседние эпохи и реплики с другим seed идут в
# другом порядке, а маска клиента от порядка не зависит: её розыгрыш
# ключуется id клиента.
#
# С перестановкой упаковка в micro-batch'и у эпох разная, поэтому
# горизонт складывает шаги каждой эпохи, а не умножает первую.
# ============================================================


GROUPS, PER_GROUP = 4, 4


def people() -> list[list[world.Made]]:
    """
    Четыре группы строк по четыре клиента разной длины: от порядка
    зависит, как бюджет соберёт их в micro-batch'и.
    """

    batches = []

    for group in range(GROUPS):

        batch = []

        for place in range(PER_GROUP):

            number = group * PER_GROUP + place
            events = [[(world.KEY_A, [10 + number % 7], True)] for _ in range(number % 5 + 1)]

            batch.append(world.make(f"c{number:02d}", events, [(world.KEY_A, [20])], batch_index=group))

        batches.append(batch)

    return batches


def lay(stage) -> None:

    world.install(stage, {"train": people(), "val": [world.population("v")]})


def ids(source) -> list[str]:
    return [client.client_id for client in source.clients()]


def file_order() -> list[str]:
    return [made.client.client_id for batch in people() for made in batch]


def random_mask() -> MaskingConfig:
    """
    Маска с настоящим розыгрышем: цели у клиента зависят от seed.
    """

    return MaskingConfig(seed=11, informativeness_weighted_masking=False)


def steps_per_epoch(config: MlmConfig, masking: MaskingConfig, epoch: int) -> int:

    sizes = train_source(config, masking, epoch).sizes()

    return math.ceil(sum(1 for _ in micro_batches(sizes, config.token_budget)) / config.grad_accum_steps)


def steps_telemetry() -> list[dict]:
    """
    Строки шагов без часов: время и ожидание у прогонов разные.
    """

    rows = [json.loads(line) for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()]

    return [
        {key: value for key, value in row.items() if key not in ("time", "seconds", "wait_seconds")}
        for row in rows if row["kind"] == "step"
    ]


# ============================================================
# ПОРЯДОК
# ============================================================


def test_epochs_get_different_orders_and_an_epoch_repeats(stage):

    lay(stage)

    config = tiny(shuffle_order=True)
    masking = every_value()

    first, second = ids(train_source(config, masking, 1)), ids(train_source(config, masking, 2))

    assert first != second
    assert sorted(first) == sorted(second) == sorted(file_order())
    assert first != file_order()

    # Та же эпоха — тот же поток.
    assert ids(train_source(config, masking, 1)) == first

    # Реплика шумового пола (другой MlmConfig.seed) идёт своим порядком.
    replica = tiny(shuffle_order=True, seed=config.seed + 40)

    assert ids(train_source(replica, masking, 1)) != first


def test_client_mask_does_not_depend_on_order(stage):

    lay(stage)

    masking = random_mask()

    for epoch in (1, 2):

        plain = {client.client_id: client for client in train_source(tiny(), masking, epoch).clients()}
        mixed = {
            client.client_id: client
            for client in train_source(tiny(shuffle_order=True), masking, epoch).clients()
        }

        assert set(plain) == set(mixed)

        # Маска разыграна по-настоящему: цели есть не у каждого значения.
        assert any((client.labels == -100).sum() > 0 for client in plain.values())

        for client_id, client in plain.items():
            assert mixed[client_id].value_ids.tolist() == client.value_ids.tolist(), client_id
            assert mixed[client_id].labels.tolist() == client.labels.tolist(), client_id
            assert mixed[client_id].reason == client.reason, client_id


def test_sizes_follow_the_shuffled_order(stage, capsys):
    """
    sizes() идёт тем же порядком, что clients(): по ним считаются число
    micro-batch'ей, K эпохи и горизонт. Горизонт обязан равняться
    числу шагов, которые обучение на самом деле сделало.
    """

    lay(stage)

    config = tiny(shuffle_order=True, token_budget=18, grad_accum_steps=2, warmup_steps=1)
    masking = every_value()
    epochs = 3

    for epoch in range(1, epochs + 1):

        source = train_source(config, masking, epoch)

        assert list(source.sizes()) == [
            Size(client.n_tokens, client.profile_n_tokens) for client in source.clients()
        ]

    # Упаковка у эпох разная — иначе тест не отличил бы сумму от
    # умножения первой эпохи.
    per_epoch = [steps_per_epoch(config, masking, epoch) for epoch in range(1, epochs + 1)]

    assert len(set(per_epoch)) > 1

    planned = horizon(config, masking, epochs)

    assert planned == sum(per_epoch)

    capsys.readouterr()

    train(config, epochs=epochs, max_steps=None, masking=masking)

    rows = steps_telemetry()

    assert len(rows) == planned
    assert [sum(1 for row in rows if row["epoch"] == epoch) for epoch in range(1, epochs + 1)] == per_epoch

    run = next(
        json.loads(line)
        for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()
        if json.loads(line)["kind"] == "run"
    )

    assert run["total_steps"] == planned


def test_resume_mid_epoch_with_shuffle(stage):
    """
    Остановка посреди второй эпохи и --resume: продолжение пропускает
    пройденные micro-batch'и того же перемешанного потока, и чекпойнт
    с телеметрией шагов те же, что без остановки.
    """

    config = tiny(shuffle_order=True, token_budget=24, warmup_steps=2)
    masking = every_value()

    lay(stage)

    train(config, epochs=2, max_steps=None, masking=masking)

    straight, straight_steps = read(checkpoint_path()), steps_telemetry()

    first_epoch = sum(1 for row in straight_steps if row["epoch"] == 1)
    second_epoch = len(straight_steps) - first_epoch

    assert second_epoch >= 3

    lay(stage)

    train(config, epochs=2, max_steps=first_epoch + 2, masking=masking)

    paused = read(checkpoint_path())

    assert paused["epoch_complete"] is False
    assert paused["epoch"] == 2
    assert paused["micro_batches_done"] == 2

    train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    compare(straight, read(checkpoint_path()))

    assert steps_telemetry() == straight_steps


def test_prefetch_workers_keep_the_shuffled_stream(stage):
    """
    Review Focus 2. Процессы подготовки получают источник через
    __getstate__ и делят order между собой: поток клиентов — и порядок
    групп, и порядок внутри группы — тот же, что без процессов.
    """

    lay(stage)

    source = train_source(tiny(shuffle_order=True), random_mask(), 2)

    alone = [(client.client_id, client.labels.tolist()) for client in Prefetch(source, 0).clients()]
    together = [(client.client_id, client.labels.tolist()) for client in Prefetch(source, 2).clients()]

    assert together == alone

    # Перемешаны и группы, и клиенты внутри групп: иначе процессы,
    # потерявшие перестановку группы, совпали бы с одним процессом.
    order = [client_id for client_id, _ in alone]
    by_group = [sorted(order[start:start + PER_GROUP]) for start in range(0, len(order), PER_GROUP)]

    assert source.order != list(range(GROUPS))
    assert any(
        order[start:start + PER_GROUP] != sorted(order[start:start + PER_GROUP])
        for start in range(0, len(order), PER_GROUP)
    )
    assert sorted(map(tuple, by_group)) == sorted(
        tuple(made.client.client_id for made in batch) for batch in people()
    )


def test_shuffle_off_is_the_previous_order(stage):

    lay(stage)

    assert MlmConfig().shuffle_order is False
    assert MlmConfig.from_dict({}).shuffle_order is False
    assert MlmConfig.from_dict(tiny(shuffle_order=True).as_dict()).shuffle_order is True

    config = tiny()

    for epoch in (1, 2):

        source = train_source(config, every_value(), epoch)

        assert source.order == list(range(GROUPS))
        assert source.shuffle is None
        assert ids(source) == file_order()
