from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.temporal.position import SECOND_US, TemporalError, check, log_age, time_log

from tests.test_downstream import chain
from tests.test_profile_state import EARLY, QUIET_SNAPSHOT, RAW_CLIENT


# ============================================================
# ИДЕЯ
# ============================================================
#
# Точка отсчёта времени событий — выбор набора 05 (time_anchor):
# cutoff T примера (по умолчанию) или последнее событие (прежний
# отсчёт). Позиции считает читатель набора. Проверяется, что:
#
#   - без выбора набор записывает отсчёт от cutoff;
#   - от cutoff позиция — ровно сжатая давность до T, и ноль у
#     последнего события только тогда, когда оно в самом T;
#   - одна и та же лента на более позднем T отличается ровно
#     давностью — её модель при таком отсчёте видит;
#   - усечение старой истории не двигает позиции оставшихся;
#   - событие позже T и сломанный порядок останавливают чтение;
#   - выбор записан в meta.json набора, и вход на T и читатель
#     обучения берут его оттуда; без записи оба отказывают.
# ============================================================


T = datetime(2026, 1, 30, 19, tzinfo=timezone.utc)

HOUR_US = 3600 * SECOND_US


def before(*hours: int) -> list[datetime]:
    return [T - timedelta(hours=value) for value in hours]


def exact(ages_us: np.ndarray) -> list[float]:
    return log_age(ages_us).astype(np.float32).tolist()


def test_from_the_last_event_nothing_changes():

    got = time_log("c", before(48, 5, 1))

    assert got == exact(np.array([47, 4, 0], dtype=np.int64) * HOUR_US)

    check("c", got, 3, "last_event")


def test_from_the_cutoff_the_position_is_the_compressed_age_at_t():

    got = time_log("c", before(48, 5, 1), T)

    assert got == exact(np.array([48, 5, 1], dtype=np.int64) * HOUR_US)
    assert got[-1] > 0.0

    check("c", got, 3, "cutoff")

    # Ноль у последнего события — правило отсчёта от него, а не от T.
    with pytest.raises(TemporalError, match="а не ноль"):
        check("c", got, 3, "last_event")

    assert time_log("c", [T], T) == [0.0]


def test_the_same_feed_at_a_later_cutoff_differs_by_the_gap():

    later = T + timedelta(days=30)
    gap = 30 * 24 * HOUR_US

    near = time_log("c", before(48, 5, 1), T)
    far = time_log("c", before(48, 5, 1), later)

    assert far == exact(np.array([48, 5, 1], dtype=np.int64) * HOUR_US + gap)
    assert all(old > new for old, new in zip(far, near))


def test_dropping_old_events_keeps_the_positions_of_the_rest():

    times = before(900, 400, 48, 5, 1)

    assert time_log("c", times[2:], T) == time_log("c", times, T)[2:]


def test_an_event_after_the_cutoff_stops_the_stage():

    with pytest.raises(TemporalError, match="позже cutoff"):
        time_log("c", [T - timedelta(hours=1), T + timedelta(seconds=1)], T)


@pytest.mark.parametrize("cutoff", [None, T])
def test_events_out_of_order_stop_the_stage(cutoff):

    with pytest.raises(TemporalError, match="по возрастанию"):
        time_log("c", before(1, 5), cutoff)


def test_positions_growing_towards_the_end_are_refused():

    with pytest.raises(TemporalError, match="растут"):
        check("c", [3.0, 5.0], 2, "cutoff")


def test_by_default_time_is_counted_from_the_cutoff(stage):

    from src.dataset.settings import META_FILE, DatasetConfig, dataset_dir
    from src.preprocessing.artifacts import read_json

    assert DatasetConfig.load(None).time_anchor == "cutoff"

    chain(stage, EARLY, QUIET_SNAPSHOT)

    assert read_json(dataset_dir("train") / META_FILE)["time_anchor"] == "cutoff"


@pytest.mark.parametrize("anchor", ["last_event", "cutoff"])
def test_the_stage_records_its_anchor_and_the_input_at_t_follows_it(stage, anchor: str):

    from src.dataset.settings import META_FILE, dataset_dir
    from src.downstream.at_cutoff import ClientsAtCutoff
    from src.downstream.settings import cutoff
    from src.preprocessing.artifacts import read_json

    chain(stage, EARLY, QUIET_SNAPSHOT, anchor)

    for group in ("train", "val"):
        assert read_json(dataset_dir(group) / META_FILE)["time_anchor"] == anchor

    moment = cutoff("val")
    client = ClientsAtCutoff("val", moment).client(RAW_CLIENT)

    ages = np.array(
        [int((moment - event_time) / timedelta(microseconds=1)) for event_time in client.event_time]
    )
    if anchor == "last_event":
        ages = ages - ages[-1]

    assert client.event_time_log.tolist() == exact(ages)


def test_the_input_refuses_data_without_a_recorded_anchor(stage):

    from src.dataset.settings import META_FILE, dataset_dir
    from src.downstream.at_cutoff import ClientsAtCutoff, CutoffError
    from src.downstream.settings import cutoff
    from src.mlm.inputs import InputError, Source
    from src.preprocessing.artifacts import read_json, write_json

    chain(stage, EARLY, QUIET_SNAPSHOT)

    # Набор, собранный кодом до записи точки отсчёта.
    for group in ("train", "val"):
        path = dataset_dir(group) / META_FILE
        meta = read_json(path)
        del meta["time_anchor"]
        write_json(path, meta)

    with pytest.raises(CutoffError, match="точка отсчёта"):
        ClientsAtCutoff("val", cutoff("val"))

    with pytest.raises(InputError, match="точка отсчёта времени"):
        Source("val")
