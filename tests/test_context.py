from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pytest

from src.dataset.context import ContextError, EventStub, select
from src.dataset.settings import (
    MAX_EVENTS,
    POLICY_ALL,
    POLICY_RECENT,
    ConfigError,
    ContextPolicy,
    DatasetConfig,
)

from tests.test_profile_state import (
    EARLY,
    QUIET_SNAPSHOT,
    RAW_CLIENT,
    prepare,
    write_profile_vocab,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# История длиннее MAX_EVENTS событий режется до последних
# MAX_EVENTS: непрерывный хвост в исходном порядке, без вех и без
# отдельно сохранённых старых событий. Проверяется:
#
#   границы       11999 и 12000 не режутся, 12001 и 20300 — режутся;
#   хвост         первым остаётся бывший n-11999-й (по счёту с 1),
#                 последний не меняется, порядок тот же;
#   пример        все массивы событий собраны из оставшихся
#                 заново, маска целей — их, анкета не тронута;
#   оценка        в val и test потеря события периода целей
#                 останавливает сборку, train резать можно.
# ============================================================


def stubs(count: int) -> list[EventStub]:
    return [EventStub(index=number, n_tokens=2, eligible=False) for number in range(count)]


def test_default_is_the_last_12000_events():

    assert MAX_EVENTS == 12000
    assert DatasetConfig().context == ContextPolicy(policy=POLICY_RECENT, max_events=12000)


@pytest.mark.parametrize("count", [11999, 12000, 12001, 20300])
def test_long_history_keeps_its_last_12000_events(count: int):

    selection = select(stubs(count), ContextPolicy())

    kept = min(count, 12000)

    assert selection.n_kept == kept
    assert selection.truncated == (count > 12000)
    assert selection.n_excluded == count - kept

    # Первым остаётся бывший (n - 11999)-й по счёту с единицы, то
    # есть индекс n - 12000; последний — прежний последний; между
    # ними ни пропуска, ни перестановки.
    assert selection.kept[0] == count - kept
    assert selection.kept[-1] == count - 1
    assert selection.kept == list(range(count - kept, count))
    assert selection.excluded == list(range(count - kept))


def test_excluded_events_of_the_target_period_are_counted():

    events = [EventStub(index=number, n_tokens=3, eligible=number % 2 == 0) for number in range(10)]

    selection = select(events, ContextPolicy(max_events=6))

    assert selection.excluded == [0, 1, 2, 3]
    assert selection.excluded_eligible == 2
    assert selection.excluded_tokens == 12
    assert selection.kept_tokens == 18


def test_policy_all_refuses_a_history_beyond_its_declared_limit():

    select(stubs(5), ContextPolicy(policy=POLICY_ALL, max_events=None))

    with pytest.raises(ContextError, match="политике all"):
        select(stubs(6), ContextPolicy(policy=POLICY_ALL, max_events=5))


def test_recent_without_a_limit_is_a_configuration_error():

    with pytest.raises(ConfigError, match="без max_events"):
        ContextPolicy(policy=POLICY_RECENT, max_events=None).validate()

    with pytest.raises(ConfigError, match="неизвестные ключи"):
        ContextPolicy.from_dict({"milestone_share": 0.25})


# ============================================================
# ПРИМЕР ИЗ УСЕЧЁННОЙ ИСТОРИИ
# ============================================================


def long_client(count: int, window):
    """
    count событий по минуте друг от друга, последние — в периоде
    целей группы. Длина события чередуется (2 и 3 токена), чтобы
    смещения событий были разными; каждое седьмое — изменение
    анкеты, которое целью не бывает.
    """

    from src.dataset.tokenized import TokenizedClient, TokenizedEvent
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.specials import EVT, USR

    artifacts = FrozenArtifacts.load()

    evt = artifacts.special(EVT)
    key = artifacts.key_id("profile_gender")
    value = artifacts.categorical_id("profile_gender", "F")

    start = window.target_start - timedelta(minutes=count // 2)

    events = []

    for number in range(count):

        width = 2 + number % 2

        events.append(
            TokenizedEvent(
                event_time=start + timedelta(minutes=number),
                event_type="profile_change" if number % 7 == 0 else "purchase",
                key_ids=[evt] + [key] * (width - 1),
                value_ids=[evt] + [value] * (width - 1),
                positions=[0] * width,
                calendar=[float(number)] + [0.0] * 5,
            )
        )

    usr = artifacts.special(USR)

    profile = dict(
        profile_key_ids=[usr, artifacts.key_id("profile_city"), artifacts.key_id("profile_lifelong")],
        profile_value_ids=[
            usr,
            artifacts.categorical_id("profile_city", "Astana"),
            artifacts.categorical_id("profile_lifelong", "bank_registered"),
        ],
        profile_positions=[0, 0, 0],
        profile_time=[None, None, window.target_start - timedelta(days=900)],
    )

    return artifacts, TokenizedClient(client_id=RAW_CLIENT, events=events, **profile), profile


def test_truncated_sample_is_rebuilt_from_the_kept_events(stage):

    from src.dataset.sample import build_sample
    from src.dataset.targets import can_be_target, eligible
    from src.preprocessing.settings import PreprocessingConfig

    write_profile_vocab(stage)

    window = PreprocessingConfig.load(None).windows["val"]

    count = 12001

    artifacts, client, profile = long_client(count, window)

    sample = build_sample(artifacts, client, window, ContextPolicy())

    kept = client.events[1:]

    assert sample.truncated and sample.excluded_events == 1
    assert sample.n_events == 12000

    # Смещения — с нуля по длинам оставшихся событий.
    lengths = [event.n_tokens for event in kept]

    assert sample.event_lengths.tolist() == lengths
    assert sample.event_starts.tolist() == np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    assert sample.key_ids.tolist() == [token for event in kept for token in event.key_ids]

    # Время и календарь — тех же событий, в том же порядке.
    assert sample.event_time.tolist() == [event.event_time.replace(tzinfo=None) for event in kept]
    assert sample.calendar.reshape(-1, 6)[:, 0].tolist() == [float(number) for number in range(1, count)]

    # Маска целей — оставшихся событий, а не сдвинутая чужая.
    assert sample.target_event_mask.tolist() == [
        eligible(event.event_time, window) and can_be_target(event.event_type) for event in kept
    ]
    assert sample.target_event_mask.any() and not sample.target_event_mask.all()

    # Анкета и её вехи отбором не затрагиваются.
    assert sample.profile_key_ids.tolist() == profile["profile_key_ids"]
    assert sample.profile_value_ids.tolist() == profile["profile_value_ids"]
    assert sample.profile_positions.tolist() == profile["profile_positions"]
    assert sample.profile_time.tolist() == [
        None, None, profile["profile_time"][2].replace(tzinfo=None)
    ]


def test_history_at_the_limit_is_kept_whole(stage):

    from src.dataset.sample import build_sample
    from src.preprocessing.settings import PreprocessingConfig

    write_profile_vocab(stage)

    window = PreprocessingConfig.load(None).windows["val"]

    artifacts, client, _ = long_client(12000, window)

    sample = build_sample(artifacts, client, window, ContextPolicy())

    assert not sample.truncated
    assert sample.n_events == 12000
    assert sample.event_time.tolist()[0] == client.events[0].event_time.replace(tzinfo=None)


# ============================================================
# ОЦЕНОЧНЫЕ ГРУППЫ
# ============================================================
#
# У клиента четыре события: два старых и два в периоде целей val
# (EARLY). Предел 2 режет только старые, предел 1 — уже цель.
# ============================================================


def build(stage, group: str, max_events: int) -> dict:

    from src.dataset.build import build_group
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    artifacts = FrozenArtifacts.load()

    encode_group(artifacts, group, TokenizerConfig.load(None))

    return build_group(
        artifacts, group, DatasetConfig(context=ContextPolicy(max_events=max_events))
    )


def test_evaluation_group_keeps_all_its_targets_or_stops(stage):

    from src.dataset.build import BuildError
    from src.dataset.settings import dataset_dir

    write_profile_vocab(stage)

    prepare(stage, EARLY, QUIET_SNAPSHOT)

    report = build(stage, "val", max_events=2)

    assert report["counts"]["truncated"] == 1
    assert report["counts"]["excluded_events"] == 2
    assert (report["counts"]["max_events_before"], report["counts"]["max_events"]) == (4, 2)

    meta = json.loads((dataset_dir("val") / "meta.json").read_text(encoding="utf-8"))

    assert meta["context"]["policy"] == POLICY_RECENT
    assert meta["context"]["max_events"] == 2
    assert (meta["truncated_samples"], meta["excluded_events"]) == (1, 2)

    with pytest.raises(BuildError, match="периода целей"):
        build(stage, "val", max_events=1)


def test_train_may_lose_old_targets_to_the_limit(stage):
    """
    train учится на всей своей истории как на целях, и отбор
    контекста законно отнимает у неё старые события.
    """

    write_profile_vocab(stage)

    prepare(stage, EARLY, QUIET_SNAPSHOT, group="train")

    report = build(stage, "train", max_events=1)

    assert report["counts"]["truncated"] == 1
    assert report["counts"]["max_events"] == 1
