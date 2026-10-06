from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import pyarrow.parquet as pq
import pytest

from src.generator import config, emit
from src.generator import engine  # noqa: F401  (engine_month регистрируется из engine)
from src.generator.engine_month import month_activity
from src.generator.observe.envelope import Event


# ============================================================
# ИДЕЯ
# ============================================================
#
# Прошлое клиента не зависит от того, где кончается выгрузка:
# generate(end=T1) и generate(end=T2 > T1) совпадают до T1 во всём,
# что генератор отдаёт наружу, — в ленте, в датированных фактах
# анкеты и в правде симуляции truth/ (переходы скрытого состояния и
# недельные снимки).
#
# Конец окна берётся и посреди месяца, и ровно на его границе:
# у границы месяца стоят снимки остатков и месячные начисления,
# и именно там прошлое раньше читало строки, датированные
# будущим.
#
# Мир группы — seed, клиенты и registration_end — один и тот же;
# меняется только конец выгрузки.
# ============================================================


START = datetime(2024, 1, 1)
LONG = datetime(2024, 7, 1)
ENDS = (datetime(2024, 4, 1), datetime(2024, 4, 17))
SEEDS = (101, 202, 303)
CLIENTS = 8
COMMUNITY = 4


def generate(out, seed: int, end: datetime, clients: int = CLIENTS) -> None:

    emit.generate_dataset(
        total_clients=clients,
        out_dir=out,
        seed=seed,
        world_seed=42,
        history_start=START,
        history_end=end,
        registration_end=LONG,
        workers=1,
        community_size=COMMUNITY,
        quiet=True,
    )


def local(text: str) -> datetime:
    return datetime.fromisoformat(text)


def events_before(directory, moment: datetime) -> dict[str, list[tuple]]:

    table = pq.read_table(directory / "events.parquet")
    boundary = moment.replace(tzinfo=config.TIMEZONE)
    rows: dict[str, list[tuple]] = defaultdict(list)

    for client, when, source, payload in zip(
        *[table.column(name).to_pylist() for name in ("client_id", "event_time", "source", "payload")]
    ):
        if local(when) < boundary:
            rows[client].append((when, source, payload))

    return dict(rows)


def truth_before(directory, name: str, moment: datetime) -> list[dict]:

    boundary = moment.replace(tzinfo=config.TIMEZONE)

    return [
        row for row in pq.read_table(directory / "truth" / f"{name}.parquet").to_pylist()
        if row["time"] < boundary
    ]


def profile_facts_before(directory, moment: datetime) -> dict[str, tuple]:

    boundary = moment.replace(tzinfo=config.TIMEZONE)

    return {
        row["client_id"]: (
            [(item["type"], item["event_time"], item["source_id"])
             for item in row["lifelong"] if item["event_time"] < boundary],
            [(item["start_date"], item["record_time"])
             for item in row["employment"] if item["record_time"] < boundary],
        )
        for row in pq.read_table(directory / "profile.parquet").to_pylist()
    }


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:

    base = tmp_path_factory.mktemp("prefix")
    result = {}

    for seed in SEEDS:
        for end in ENDS + (LONG,):
            out = base / f"{seed}-{end:%Y%m%d}"
            generate(out, seed, end)
            result[seed, end] = out

    return result


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("end", ENDS, ids=lambda item: f"{item:%Y-%m-%d}")
def test_the_tape_before_the_end_does_not_depend_on_the_end(runs, seed, end):

    short = events_before(runs[seed, end], end)
    long = events_before(runs[seed, LONG], end)

    assert short, "проверка вырождена: в коротком окне нет событий"
    assert short.keys() == long.keys()

    for client, rows in short.items():
        assert rows == long[client], client


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("end", ENDS, ids=lambda item: f"{item:%Y-%m-%d}")
def test_the_dated_profile_facts_do_not_depend_on_the_end(runs, seed, end):

    short = profile_facts_before(runs[seed, end], end)
    long = profile_facts_before(runs[seed, LONG], end)

    for client, facts in short.items():
        assert facts == long[client], client


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("end", ENDS, ids=lambda item: f"{item:%Y-%m-%d}")
@pytest.mark.parametrize("name", ["transitions", "states"])
def test_the_hidden_state_before_the_end_does_not_depend_on_the_end(runs, seed, end, name):

    short = truth_before(runs[seed, end], name, end)
    long = truth_before(runs[seed, LONG], name, end)

    assert short == long


def test_a_client_depends_only_on_its_own_community(tmp_path):
    """
    Полное сообщество живёт одинаково, сколько бы ни было клиентов
    после него: добавленное сообщество не трогает прежних.
    """

    alone = tmp_path / "alone"
    more = tmp_path / "more"

    generate(alone, SEEDS[0], ENDS[0], clients=CLIENTS)
    generate(more, SEEDS[0], ENDS[0], clients=CLIENTS + COMMUNITY)

    first = events_before(alone, ENDS[0])
    second = events_before(more, ENDS[0])

    assert first
    for client, rows in first.items():
        assert rows == second[client], client

    for name in ("transitions", "states"):
        before = truth_before(alone, name, ENDS[0])
        known = {row["client_id"] for row in before}
        after = [row for row in truth_before(more, name, ENDS[0]) if row["client_id"] in known]
        assert before, name
        assert before == after, name


def test_a_later_planning_horizon_only_adds_the_future(tmp_path, monkeypatch):
    """
    Планы клиента идут вперёд по времени, и горизонт планирования
    только обрезает будущее: поднятый на год, он не меняет ни одной
    строки выгрузки внутри прежнего окна. Раньше число событий и их
    даты раскладывались по длине горизонта, и его продление
    переписывало прошлое.
    """

    base = tmp_path / "base"
    later = tmp_path / "later"

    generate(base, SEEDS[1], ENDS[1])

    monkeypatch.setattr(config, "PLANNING_END", config.PLANNING_END + timedelta(days=365))

    generate(later, SEEDS[1], ENDS[1])

    for name in ("events.parquet", "profile.parquet", "truth/transitions.parquet", "truth/states.parquet"):
        assert pq.read_table(base / name).equals(pq.read_table(later / name)), name


# ============================================================
# СТРОКИ С БУДУЩИМ ВРЕМЕНЕМ
# ============================================================


def event(kind: str, moment: datetime, account: str) -> Event:
    return Event(
        client_id="c1", event_time=moment, source="transactions",
        event_type=kind, payload={"account_id": account}, ordinal=0,
    )


def test_a_month_snapshot_ignores_rows_dated_after_it():
    """
    Активация карты или chargeback, датированные следующим месяцем,
    в ленте есть только тогда, когда выгрузка до них дотягивается.
    Снимок конца месяца на них не смотрит.
    """

    month = datetime(2024, 3, 1)
    moment = datetime(2024, 3, 31, 23, 55)

    inside = [event("purchase", datetime(2024, 3, 10, 12), "acc_1")]
    future = inside + [event("card_activated", moment + timedelta(days=3), "acc_2")]

    assert month_activity(inside, month, moment) == month_activity(future, month, moment)
    assert month_activity([event("card_activated", moment + timedelta(days=3), "acc_2")], month, moment) == (set(), True)
