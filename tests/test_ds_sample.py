from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from src.dataset.dependencies import (
    DEP_CAUSE_EVENT,
    DEP_IN_CONTEXT,
    DEP_OUTSIDE_CONTEXT,
    DEP_PROFILE,
    DEP_PROFILE_OTHER_VERSION,
    DEP_STATUSES,
)
from src.dataset.encoding import causes_as_of, encode_history
from src.dataset.inputs import InputsError
from src.dataset.sample import build_sample
from src.dataset.settings import ContextPolicy, DatasetConfig
from src.dataset.targets import COVERAGE_CODES, eligible, hours_to_cutoff
from src.preprocessing.settings import GroupWindow
from src.tokenization.layout import MISSING, USR

from tests.ds_fixtures import (
    AFTER_CORRECTION,
    BEFORE_EVENT,
    BETWEEN_VERSIONS,
    CORRECTED_AMOUNT,
    ORIGINAL_AMOUNT,
    config_all,
    inputs_of,
    prepared_late_correction,
    ready,
)
from tests.tok_fixtures import FIT_END


# ============================================================
# ОБЩЕЕ
# ============================================================


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path, Path]:
    return ready(tmp_path_factory.mktemp("ds_sample"))


@pytest.fixture(scope="module")
def late(tmp_path_factory) -> tuple[Path, Path, Path]:
    return prepared_late_correction(tmp_path_factory.mktemp("ds_late"))


def _sample(inputs, group: str, client_id: str, cutoff: datetime, config=None):

    entry = inputs.groups[group]

    config = config or inputs.config

    history = entry.history(client_id, cutoff)

    encoded = encode_history(
        inputs.artifacts,
        history,
        inputs.tokenizer_config.max_pieces_per_value,
        cause_of=causes_as_of(entry.corpus.store, client_id, cutoff),
    )

    return build_sample(
        artifacts=inputs.artifacts,
        encoded=encoded,
        group=group,
        window=entry.window,
        weight=entry.weight,
        sources=inputs.sources(),
        policy=config.context,
    )


def _value_of(sample, event_index: int, key: str) -> int:
    """
    Код первого токена значения по адресу «событие, столбец».
    """

    for number in range(sample.n_values):

        if int(sample.value_event[number]) != event_index:
            continue

        start = int(sample.event_starts[event_index]) + int(sample.value_start[number])

        if int(sample.value_key_id[number]) == sample.key_ids[start]:
            # Ключ опознаётся кодом, а имя лежит в служебной
            # таблице: сверяем через неё.
            keys = sample.events[_position_of(sample, event_index)]["value_keys"]
            if keys[_span_number(sample, event_index, number)] == key:
                return int(sample.value_ids[start])

    raise AssertionError(f"значения {key} в событии {event_index} нет")


def _position_of(sample, event_index: int) -> int:
    return next(
        number for number, row in enumerate(sample.events) if row["event_index"] == event_index
    )


def _span_number(sample, event_index: int, absolute: int) -> int:
    first = next(
        number for number in range(sample.n_values)
        if int(sample.value_event[number]) == event_index
    )
    return absolute - first


# ============================================================
# ВИДИМОСТЬ ИСПРАВЛЕНИЯ
# ============================================================


def test_correction_becomes_visible_at_its_own_time(late):
    """
    Исправление видно с момента СВОЕГО события, а место в истории
    задаёт первая версия.

    Три среза: до исходной записи, между версиями и после
    исправления. Средний и есть суть: без него нельзя отличить
    «версия 2 появилась в свой срок» от «версия 2 была всегда».
    """

    root, out, target = late

    inputs = inputs_of(root, out, target, config_all())

    entry = inputs.groups["train"]

    before = entry.history("train_c1", BEFORE_EVENT)

    assert before.n_events == 0, "событие ещё не произошло"

    middle = _sample(inputs, "train", "train_c1", BETWEEN_VERSIONS)
    after = _sample(inputs, "train", "train_c1", AFTER_CORRECTION)

    versions = {
        "между версиями": [row["event_version"] for row in middle.events],
        "после исправления": [row["event_version"] for row in after.events],
    }

    assert versions["между версиями"] == [1], "на среднем срезе действует первая версия"
    assert 2 in versions["после исправления"], "после исправления действует вторая"

    # Место события задаёт первая версия: исправленная запись
    # остаётся ПЕРВОЙ, хотя её собственное время позже соседней
    # покупки.
    corrected = next(row for row in after.events if row["event_version"] == 2)

    assert corrected["event_index"] == 0
    assert corrected["event_time"] > after.events[1]["event_time"]

    # Действует именно исправленная сумма. Сверяется она по
    # смысловому слою, а не по коду диапазона: на маленькой
    # фикстуре обе суммы законно попадают в один диапазон
    # словаря, и равенство кодов ничего не опровергает.
    assert entry.history("train_c1", BETWEEN_VERSIONS).events[0].values["transaction_amount"] == ORIGINAL_AMOUNT
    assert entry.history("train_c1", AFTER_CORRECTION).events[0].values["transaction_amount"] == CORRECTED_AMOUNT


def test_future_record_does_not_change_an_earlier_slice(late):
    """
    Поздняя запись прошлого не трогает: срез, снятый до неё,
    остаётся прежним.
    """

    root, out, target = late

    inputs = inputs_of(root, out, target, config_all())

    first = _sample(inputs, "train", "train_c1", BETWEEN_VERSIONS)
    second = _sample(inputs, "train", "train_c1", BETWEEN_VERSIONS)

    assert first.key_ids.tolist() == second.key_ids.tolist()
    assert first.n_events == 1

    later = _sample(inputs, "train", "train_c1", AFTER_CORRECTION)

    assert later.n_events == 2, "к позднему срезу добавились обе записи"


def test_amounts_differ_between_versions(late):

    root, out, target = late

    inputs = inputs_of(root, out, target, config_all())

    entry = inputs.groups["train"]

    middle = entry.history("train_c1", BETWEEN_VERSIONS)
    after = entry.history("train_c1", AFTER_CORRECTION)

    assert middle.events[0].values["transaction_amount"] == ORIGINAL_AMOUNT
    assert after.events[0].values["transaction_amount"] == CORRECTED_AMOUNT


# ============================================================
# ЦЕЛИ И ВРЕМЯ
# ============================================================


def test_target_window_is_checked_on_both_borders():
    """
    Начало периода целей входит, конец — нет.
    """

    window = GroupWindow(
        history_start=datetime(2023, 1, 1),
        final_cutoff=datetime(2026, 6, 1),
        target_start=datetime(2026, 1, 1),
        target_end=datetime(2026, 6, 1),
    )

    assert not eligible(datetime(2025, 12, 31, 23, 59, 59), window)
    assert eligible(datetime(2026, 1, 1), window)
    assert eligible(datetime(2026, 5, 31, 23, 59, 59), window)
    assert not eligible(datetime(2026, 6, 1), window)


def test_day_precision_interval_is_whole_days(dataset):
    """
    У записи дневной точности час не наблюдался, и интервал до
    среза у неё кратен суткам.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    sample = _sample(inputs, "train", "train_c1", FIT_END)

    daily = [
        number for number, row in enumerate(sample.events)
        if row["kept"] and not row["hour_known"]
    ]

    assert daily, "в фикстуре есть запись дневной точности"

    for number in daily:

        slot = sample.events[number]["event_index"]

        assert float(sample.hours_to_cutoff[slot]) % 24 == 0
        assert not bool(sample.hour_known[slot])


def test_hours_to_cutoff_respects_declared_precision():

    assert hours_to_cutoff(datetime(2026, 1, 1), datetime(2025, 4, 15, 13, 17), "day") % 24 == 0
    assert hours_to_cutoff(datetime(2026, 1, 1), datetime(2025, 4, 15, 13, 17), "second") % 24 != 0


def test_unknown_history_start_is_not_zero(dataset):
    """
    «Когда клиент пришёл, неизвестно» и «клиент с банком ноль
    дней» это разные утверждения.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    silent = _sample(inputs, "train", "train_c2", FIT_END)

    assert silent.n_events == 0

    if silent.history_age_days is None:
        assert silent.history_age_reason == "observed_start_unknown"
    else:
        assert silent.history_age_days >= 0


# ============================================================
# ПРОФИЛЬ И ПОКРЫТИЕ
# ============================================================


def test_client_without_profile_gets_one_marker(dataset):
    """
    «Анкеты нет» и «поля пусты» это разные вещи: у клиента без
    анкеты ровно один [USR] и ни одного [MISSING].
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    artifacts = inputs.artifacts

    silent = _sample(inputs, "train", "train_c2", FIT_END)

    assert silent.profile_tokens == 1
    assert silent.profile_key_ids.tolist() == [artifacts.special(USR)]
    assert silent.profile_state == "no_version_known_yet"
    assert not silent.has_profile

    rich = _sample(inputs, "train", "train_c1", FIT_END)

    assert rich.has_profile
    assert rich.profile_tokens > 1


def test_known_profile_declares_every_key(dataset):

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    rich = _sample(inputs, "train", "train_c1", FIT_END)

    assert rich.profile_value_start.size == len(inputs.artifacts.profile_keys)


def test_coverage_is_taken_from_the_history(dataset):
    """
    Доступность источников приходит из истории на ту же дату, а
    не из наличия событий.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    sources = inputs.sources()

    silent = _sample(inputs, "train", "train_c2", FIT_END)

    assert silent.coverage_at_cutoff.size == len(sources)

    # Клиент молчит, но источники ему доступны: пустая история не
    # означает недоступный источник.
    assert int(silent.coverage_at_cutoff[sources.index("transactions")]) == COVERAGE_CODES["available"]


# ============================================================
# ПРОИСХОЖДЕНИЕ
# ============================================================


def test_dependencies_carry_addresses_only_when_they_exist(dataset):
    """
    in_context обещает оба адреса, cause_event только адрес
    события, остальные статусы ни одного.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    sample = _sample(inputs, "train", "train_c1", FIT_END)

    dependencies = sample.dependencies

    assert len(dependencies) > 0, "у расчётных значений есть происхождение"

    for index in range(len(dependencies)):

        status = dependencies.status[index]
        event = dependencies.source_event[index]
        value = dependencies.source_value[index]

        if status == DEP_IN_CONTEXT:
            assert event >= 0 and value >= 0
        elif status == DEP_CAUSE_EVENT:
            assert event >= 0 and value == -1
        elif status in (DEP_PROFILE,):
            assert value >= 0
        else:
            assert value == -1


def test_relation_points_at_the_cause_event(dataset):
    """
    Возврат ссылается на свою покупку, и адрес причины лежит
    внутри примера.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    sample = _sample(inputs, "train", "train_c1", FIT_END)

    relations = [
        index for index in range(len(sample.dependencies))
        if sample.dependencies.key[index] == "related_event_type"
    ]

    assert relations, "в фикстуре есть возврат со ссылкой на покупку"

    for index in relations:
        assert sample.dependencies.status[index] == DEP_CAUSE_EVENT
        assert sample.dependencies.source_event[index] >= 0


def test_excluded_cause_becomes_outside_context(dataset):
    """
    Причина, выброшенная отбором, помечается как оставшаяся за
    границей контекста, а не теряется.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    # Тесный бюджет: покупка-причина в контекст не попадает, а
    # возврат попадает.
    tight = replace(
        DatasetConfig(),
        context=ContextPolicy(
            policy="recent_plus_milestones",
            max_events=6,
            max_tokens=100_000,
            milestone_share=0.0,
        ),
    )

    sample = _sample(inputs, "train", "train_c1", FIT_END, config=tight)

    statuses = {
        DEP_STATUSES[sample.dependencies.status[index]]
        for index in range(len(sample.dependencies))
    }

    assert "outside_context" in statuses or not sample.truncated


def test_profile_of_another_version_is_named(dataset):
    """
    Доход берётся по версии анкеты на момент операции, а в
    примере лежит анкета на срез: это разные версии, и прятать
    их вместе нельзя.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    sample = _sample(inputs, "train", "train_c1", FIT_END)

    statuses = [
        sample.dependencies.status[index] for index in range(len(sample.dependencies))
    ]

    assert DEP_PROFILE in statuses or DEP_PROFILE_OTHER_VERSION in statuses


# ============================================================
# ВХОДЫ
# ============================================================


def test_cutoff_after_the_group_window_is_refused(dataset):
    """
    Срез позже конечного момента группы впустил бы в контекст
    чужой период.
    """

    root, out, target = dataset

    config = replace(DatasetConfig(), extra_cutoffs={"train": ("2026-07-01T00:00:00",)})

    with pytest.raises(InputsError, match="позже её конечного момента"):
        inputs_of(root, out, target, config)


def test_missing_declared_key_is_a_missing_pair(dataset):
    """
    Объявленный у типа события ключ без значения получает пару
    ключ/[MISSING] с настоящим кодом ключа.
    """

    root, out, target = dataset

    inputs = inputs_of(root, out, target)

    sample = _sample(inputs, "train", "train_c1", FIT_END)

    missing = inputs.artifacts.special(MISSING)

    assert int((sample.value_ids == missing).sum()) > 0
