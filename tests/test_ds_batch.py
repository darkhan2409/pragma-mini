from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.dataset.build import build_dataset
from src.dataset.collate import CollateError, collate
from src.dataset.dependencies import DEP_CAUSE_EVENT, DEP_IN_CONTEXT, DEP_PROFILE
from src.dataset.reader import Dataset

from tests.ds_fixtures import config_all, inputs_of, ready


# ============================================================
# ИДЕЯ
# ============================================================
#
# Смесь короткой, длинной и пустой истории: одинаковые примеры
# выравниваются одинаково и границ не проверяют вовсе.
# ============================================================


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Dataset, object]:

    base = tmp_path_factory.mktemp("ds_batch")

    root, out, target = ready(base)

    config = config_all()

    inputs = inputs_of(root, out, target, config)

    result = build_dataset(inputs, config, base / "home")

    return Dataset.open(result.directory, verify_files=False), inputs


@pytest.fixture(scope="module")
def mixed(built) -> list:
    """
    Пустая, короткая и длинная история одной группы.
    """

    dataset, _inputs = built

    samples = list(dataset.iter_samples("train"))

    return sorted(samples, key=lambda item: item.n_events)


# ============================================================
# ГРАНИЦЫ И МАСКИ
# ============================================================


def test_mixed_batch_keeps_every_boundary(mixed):
    """
    Каждое событие batch совпадает с тем, что лежало в примере,
    токен в токен.
    """

    batch = collate(mixed)

    assert batch.n_samples == len(mixed)
    assert batch.n_rows == sum(item.n_events for item in mixed)

    for number, sample in enumerate(mixed):
        for slot in range(sample.n_events):

            row = int(batch.history_rows[number, slot])
            length = int(sample.event_lengths[slot])
            start = int(sample.event_starts[slot])

            assert np.array_equal(
                batch.event_key_ids[row, :length],
                sample.key_ids[start : start + length],
            )
            assert np.array_equal(
                batch.event_value_ids[row, :length],
                sample.value_ids[start : start + length],
            )
            assert int(batch.event_sample[row]) == number
            assert int(batch.event_slot[row]) == slot


def test_masks_mean_real_not_padding(mixed):
    """
    True это настоящее значение. Прежний слой модели держал
    обратное соглашение, и унаследовать его молча было бы
    ошибкой на ровном месте.
    """

    batch = collate(mixed)

    assert int(batch.event_token_mask.sum()) == sum(item.n_tokens for item in mixed)

    for row in range(batch.n_rows):

        length = int(batch.event_lengths[row])

        assert bool(batch.event_token_mask[row, :length].all())
        assert not bool(batch.event_token_mask[row, length:].any())

    assert int(batch.profile_token_mask.sum()) == sum(item.profile_tokens for item in mixed)


def test_empty_history_takes_its_row_and_no_events(mixed):
    """
    Пустая история занимает строку batch и не даёт ни одного
    события: ничего не выдумывается.
    """

    batch = collate(mixed)

    empty = [number for number, item in enumerate(mixed) if item.n_events == 0]

    assert empty, "в фикстуре есть молчащий клиент"

    for number in empty:
        assert int(batch.n_events[number]) == 0
        assert not bool(batch.history_mask[number].any())
        assert bool((batch.history_rows[number] == -1).all())
        assert int(batch.profile_token_mask[number].sum()) >= 1, "маркер профиля на месте"


def test_batch_of_only_empty_histories_is_allowed(mixed):

    empty = [item for item in mixed if item.n_events == 0]

    batch = collate(empty)

    assert batch.n_rows == 0
    assert batch.event_key_ids.shape == (0, 0)
    assert batch.history_mask.shape == (len(empty), 0)


def test_target_candidates_exclude_markers_and_padding(mixed):
    """
    Область допустимых целей это не выбранные цели: маркер и
    выравнивание в неё не входят никогда.
    """

    batch = collate(mixed)

    assert not bool(batch.target_candidate_mask[:, 0].any()), "маркер целью не бывает"

    assert bool((batch.target_candidate_mask & ~batch.event_token_mask).sum() == 0)

    for row in range(batch.n_rows):
        if not bool(batch.event_eligible[row]):
            assert not bool(batch.target_candidate_mask[row].any())


# ============================================================
# ЗНАЧЕНИЯ И ЗАВИСИМОСТИ
# ============================================================


def test_value_addresses_point_at_the_same_tokens(mixed):
    """
    Адрес значения это пара «событие, столбец», и после сборки
    batch он указывает на те же токены.
    """

    batch = collate(mixed)

    offset = 0

    for number, sample in enumerate(mixed):

        for index in range(sample.n_values):

            row = int(batch.value_event[offset + index])
            column = int(batch.value_start[offset + index])
            length = int(batch.value_length[offset + index])

            slot = int(sample.value_event[index])
            start = int(sample.event_starts[slot]) + int(sample.value_start[index])

            assert np.array_equal(
                batch.event_value_ids[row, column : column + length],
                sample.value_ids[start : start + length],
            )

        offset += sample.n_values

    assert offset == int(batch.value_offsets[-1])


def test_dependencies_move_into_batch_coordinates(mixed):
    """
    Индекс значения сдвигается на начало примера, а «нет адреса»
    остаётся «нет адреса».
    """

    batch = collate(mixed)

    total = sum(len(item.dependencies) for item in mixed)

    assert batch.dep_value.size == total

    for index in range(batch.dep_value.size):

        assert 0 <= int(batch.dep_value[index]) < int(batch.value_offsets[-1])

        status = int(batch.dep_status[index])

        if status == DEP_IN_CONTEXT:
            assert int(batch.dep_source_event[index]) >= 0
            assert int(batch.dep_source_value[index]) >= 0
        elif status == DEP_CAUSE_EVENT:
            assert int(batch.dep_source_event[index]) >= 0
            assert int(batch.dep_source_value[index]) == -1
        elif status == DEP_PROFILE:
            assert 0 <= int(batch.dep_source_value[index]) < int(batch.profile_value_offsets[-1])


def test_groups_are_never_mixed(built):

    dataset, _inputs = built

    train = next(iter(dataset.iter_samples("train")))
    val = next(iter(dataset.iter_samples("val")))

    with pytest.raises(CollateError, match="не смешиваются"):
        collate([train, val])


def test_empty_batch_is_refused():

    with pytest.raises(CollateError, match="из нуля примеров"):
        collate([])


# ============================================================
# КАНАЛЫ
# ============================================================


def test_channels_travel_per_event(mixed):

    batch = collate(mixed)

    assert batch.calendar.shape == (batch.n_rows, 6)
    assert batch.hour_known.size == batch.n_rows
    assert batch.hours_to_cutoff.size == batch.n_rows

    for number, sample in enumerate(mixed):
        for slot in range(sample.n_events):

            row = int(batch.history_rows[number, slot])

            assert np.allclose(batch.calendar[row], sample.calendar[slot * 6 : slot * 6 + 6])
            assert bool(batch.hour_known[row]) == bool(sample.hour_known[slot])


def test_unknown_history_age_is_not_zero(mixed):
    """
    Неизвестный возраст истории приходит как NaN рядом с
    признаком известности, а не нулём.
    """

    batch = collate(mixed)

    for number, sample in enumerate(mixed):

        if sample.history_age_days is None:
            assert np.isnan(batch.history_age_days[number])
            assert not bool(batch.history_age_known[number])
        else:
            assert bool(batch.history_age_known[number])
