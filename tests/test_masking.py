from __future__ import annotations

import pytest

from src.masking.apply import IGNORE, MaskError, apply
from src.masking.choose import EVENT, KEY, NONE, VALUE, Value, choose, values_of
from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source
from src.mlm.train import for_epoch

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import settle


# ============================================================
# ИДЕЯ
# ============================================================
#
# Маска обязана быть воспроизводимой и чистой: те же данные и тот
# же конфиг дают ту же маску, а трогает она только значения
# настоящих событий, которым разрешено быть целью.
#
# У обучения маска своя на каждую эпоху. Seed эпохи выводится из
# seed конфига и номера эпохи через blake2b, поэтому он не
# зависит ни от процесса, ни от порядка чтения: одна и та же
# эпоха обязана давать одну и ту же маску даже после перезапуска,
# иначе продолжение внутри эпохи училось бы на других целях.
#
# val при этом остаётся с фиксированной маской из файла: сравнивать
# эпохи между собой можно только по одной и той же мерке.
# ============================================================


def row_of(made: world.Made, *, targetable: bool | None = None) -> dict:
    """
    Строка батча в том виде, в каком её читает маскер.
    """

    client = made.client

    allowed = made.targetable if targetable is None else targetable

    return {
        "client_id": client.client_id,
        "key_ids": client.key_ids.tolist(),
        "value_ids": made.value_ids_source.tolist(),
        "positions": client.positions.tolist(),
        "event_starts": client.event_starts.tolist(),
        "event_lengths": client.event_lengths.tolist(),
        "event_mask": [True] * client.n_events,
        "target_event_mask": [allowed] * client.n_events,
    }


def sample() -> world.Made:
    """
    Три события: многокусковое значение, повтор того же ключа и
    событие из одного маркера.
    """

    return world.make(
        "m-1",
        [
            [(world.KEY_A, [10], False), (world.KEY_B, [11, 12, 13], False)],
            [(world.KEY_A, [14], False), (world.KEY_C, [15, 16], False)],
            [],
        ],
        [(world.KEY_A, [20])],
    )


def rates(**overrides) -> MaskingConfig:

    base = dict(
        seed=11,
        value_probability=0.0,
        event_probability=0.0,
        key_probability=0.0,
        unknown_probability=0.0,
    )

    base.update(overrides)

    return MaskingConfig(**base)


# ============================================================
# ГРАНИЦЫ ЗНАЧЕНИЙ
# ============================================================


def test_value_starts_at_every_zero_position_after_the_marker():
    """
    Значение начинается там, где positions возвращается к нулю, а
    маркер [EVT] в значения не входит вовсе.
    """

    found = values_of(row_of(sample()))

    assert found == [
        Value(event=0, key_id=world.KEY_A, start=1, length=1),
        Value(event=0, key_id=world.KEY_B, start=2, length=3),
        Value(event=1, key_id=world.KEY_A, start=6, length=1),
        Value(event=1, key_id=world.KEY_C, start=7, length=2),
    ]


def test_event_without_permission_has_no_values():
    """
    Событие вне периода целей не разбирается: значений в нём для
    маскера не существует.
    """

    assert values_of(row_of(sample(), targetable=False)) == []


def test_event_of_one_marker_contributes_nothing():

    only = world.make("bare", [[]], [(world.KEY_A, [20])])

    assert values_of(row_of(only)) == []


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================


def test_the_same_client_and_config_give_the_same_choice():

    made = sample()
    config = rates(value_probability=0.5)

    first = choose("train", row_of(made), config)
    second = choose("train", row_of(made), config)

    assert first.choices == second.choices


def test_the_group_is_part_of_the_stream():
    """
    Поток розыгрыша задан группой и client_id: один и тот же
    клиент в train и в val маскируется по-разному.
    """

    made = sample()
    config = rates(value_probability=0.5)

    assert choose("train", row_of(made), config).choices != choose(
        "val", row_of(made), config
    ).choices


def test_a_different_seed_gives_a_different_mask():

    made = sample()

    assert choose("train", row_of(made), rates(value_probability=0.5, seed=1)).choices != (
        choose("train", row_of(made), rates(value_probability=0.5, seed=2)).choices
    )


# ============================================================
# ЧТО МАСКА ТРОГАЕТ
# ============================================================


def test_all_pieces_of_one_value_share_one_decision():
    """
    Название из трёх кусков скрывается целиком: открыть один
    кусок значило бы показать часть ответа.
    """

    made = sample()

    result = apply(
        "m-1", row_of(made), choose("train", row_of(made), rates(value_probability=1.0)).choices,
        world.MASK, world.UNK,
    )

    for start, length in ((1, 1), (2, 3), (6, 1), (7, 2)):

        piece = result["value_ids"][start : start + length]
        labels = result["labels"][start : start + length]

        assert piece == [world.MASK] * length
        assert labels == made.value_ids_source.tolist()[start : start + length]


def test_markers_and_untouched_events_keep_their_values():

    made = sample()
    row = row_of(made)

    result = apply(
        "m-1", row, choose("train", row, rates(value_probability=1.0)).choices,
        world.MASK, world.UNK,
    )

    source = made.value_ids_source.tolist()

    assert len(result["value_ids"]) == len(source)

    for place in (0, 5, 9):
        assert result["value_ids"][place] == source[place] == world.EVT
        assert result["labels"][place] == IGNORE
        assert result["reason"][place] == NONE


def test_zero_rates_change_nothing():

    made = sample()
    row = row_of(made)

    result = apply("m-1", row, choose("train", row, rates()).choices, world.MASK, world.UNK)

    assert result["value_ids"] == made.value_ids_source.tolist()
    assert result["labels"] == [IGNORE] * made.client.n_tokens
    assert set(result["reason"]) == {NONE}


def test_unknown_replacement_spoils_the_input_without_a_label():
    """
    Часть выбранных значений уходит в [UNK]: значение портится, но
    ни метки, ни причины не пишется — ни награды, ни штрафа.
    """

    made = sample()
    row = row_of(made)

    result = apply(
        "m-1", row,
        choose("train", row, rates(value_probability=1.0, unknown_probability=1.0)).choices,
        world.MASK, world.UNK,
    )

    for place in (1, 2, 3, 4, 6, 7, 8):
        assert result["value_ids"][place] == world.UNK
        assert result["labels"][place] == IGNORE
        assert result["reason"][place] == NONE


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"event_probability": 1.0}, EVENT),
        ({"key_probability": 1.0}, KEY),
        ({"value_probability": 1.0}, VALUE),
    ],
)
def test_each_mechanism_writes_its_own_reason(overrides: dict, reason: str):

    made = sample()
    row = row_of(made)

    result = apply(
        "m-1", row, choose("train", row, rates(**overrides)).choices,
        world.MASK, world.UNK,
    )

    assert set(result["reason"]) - {NONE} == {reason}


def test_the_strongest_reason_wins():
    """
    Приоритет event > key > value: когда сработали все три,
    причина у значения одна и это event.
    """

    made = sample()
    row = row_of(made)

    result = apply(
        "m-1", row,
        choose(
            "train", row,
            rates(event_probability=1.0, key_probability=1.0, value_probability=1.0),
        ).choices,
        world.MASK, world.UNK,
    )

    assert set(result["reason"]) - {NONE} == {EVENT}


# ============================================================
# ОШИБКИ
# ============================================================


def test_overlapping_values_are_refused():

    made = sample()
    row = row_of(made)

    choices = choose("train", row, rates(value_probability=1.0)).choices

    with pytest.raises(MaskError, match="выбрана дважды"):
        apply("m-1", row, choices + choices[:1], world.MASK, world.UNK)


def test_a_value_outside_the_row_is_refused():

    from src.masking.choose import Choice

    made = sample()
    row = row_of(made)

    beyond = Choice(
        value=Value(event=0, key_id=world.KEY_A, start=made.client.n_tokens, length=2),
        reason=VALUE,
        unknown=False,
    )

    with pytest.raises(MaskError, match="не помещается"):
        apply("m-1", row, [beyond], world.MASK, world.UNK)


# ============================================================
# МАСКА ЭПОХИ
# ============================================================


def test_epoch_seed_is_stable_and_process_independent():
    """
    Seed эпохи выведен blake2b от seed конфига и номера эпохи: он
    обязан быть одним и тем же при каждом запуске, иначе
    продолжение внутри эпохи училось бы на других целях.
    """

    base = MaskingConfig(seed=42)

    assert for_epoch(base, 1) == for_epoch(base, 1)
    # Числа закреплены нарочно: смена рецепта seed'а эпохи
    # сделала бы прошлые чекпойнты невоспроизводимыми.
    assert for_epoch(base, 1).seed == 1_984_046_095
    assert for_epoch(base, 2).seed == 940_584_071


def test_neighbouring_epochs_differ():

    base = MaskingConfig(seed=42)

    seeds = {for_epoch(base, epoch).seed for epoch in range(1, 6)}

    assert len(seeds) == 5
    assert base.seed not in seeds


def test_only_probabilities_stay_the_same_across_epochs():

    base = MaskingConfig(seed=42, value_probability=0.3, unknown_probability=0.2)

    later = for_epoch(base, 7)

    assert later.value_probability == base.value_probability
    assert later.event_probability == base.event_probability
    assert later.key_probability == base.key_probability
    assert later.unknown_probability == base.unknown_probability


def test_train_mask_of_one_epoch_repeats_exactly(stage):
    """
    Дважды открытый Source одной и той же эпохи даёт те же
    видимые значения и те же метки.
    """

    settle(stage, train_people=many())

    masking = for_epoch(MaskingConfig(seed=9, value_probability=0.5), 1)

    first = list(Source("train", masking=masking).clients())
    second = list(Source("train", masking=masking).clients())

    assert [client.client_id for client in first] == [
        client.client_id for client in second
    ]

    for left, right in zip(first, second):
        assert left.value_ids.tolist() == right.value_ids.tolist()
        assert left.labels.tolist() == right.labels.tolist()


def test_two_epochs_see_different_targets(stage):

    settle(stage, train_people=many())

    base = MaskingConfig(seed=9, value_probability=0.5)

    def labels(epoch: int) -> list[list[int]]:
        return [
            client.labels.tolist()
            for client in Source("train", masking=for_epoch(base, epoch)).clients()
        ]

    assert labels(1) != labels(2)


def test_validation_mask_comes_from_the_file_and_never_changes(stage):
    """
    val читает фиксированную маску 08_masked: номер эпохи на неё
    не влияет, иначе эпохи было бы не с чем сравнивать.
    """

    settle(stage, train_people=many())

    first = [client.labels.tolist() for client in Source("val").clients()]
    second = [client.labels.tolist() for client in Source("val").clients()]

    assert first == second
    assert Source("val").masking is None
