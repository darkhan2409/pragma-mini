from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pyarrow.parquet as pq
import pytest

from src.generator import emit
from src.generator.config import EVENT_TYPES
from src.generator.profile import LIFELONG_TYPES, lifelong, utc


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вехи анкеты — это даты, которые генератор уже знал и терял при
# выгрузке: начало отношений персоны и установка приложения.
# Проверяется, что в выгрузку попадают ИМЕННО они, а не похожие
# даты, придуманные заново:
#
#   relationship_started и kyc_passed — Persona.relationship_start;
#   app_adopted — app_adoption клиента, и только если она была;
#   все вехи строго раньше as_of, а as_of — конец выгрузки.
#
# Даты персоны снимаются сразу после выгрузки, в том же
# состоянии розыгрыша: так сравниваются с тем, по чему жила
# симуляция, а не с пересчётом под другим окном.
# ============================================================


CLIENTS = 8
COMMUNITY = 4

START = datetime(2024, 1, 1)
END = datetime(2024, 4, 1)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:

    from src.generator.life.persona import app_adoption, draw_persona
    from src.generator.world import communities

    out = tmp_path_factory.mktemp("lifelong")

    manifest = emit.generate_dataset(
        total_clients=CLIENTS,
        out_dir=out,
        seed=100,
        world_seed=42,
        history_start=START,
        history_end=END,
        workers=1,
        community_size=COMMUNITY,
        quiet=True,
    )

    personas = {}

    for ordinal in communities.client_ordinals(CLIENTS):
        persona = draw_persona(ordinal)
        personas[persona.client_id] = (persona.relationship_start, app_adoption(ordinal))

    return {
        "manifest": manifest,
        "profile": pq.read_table(out / "profile.parquet").to_pylist(),
        "events": pq.read_table(out / "events.parquet").to_pylist(),
        "personas": personas,
    }


def items(row: dict) -> dict[str, datetime]:
    return {item["type"]: item["event_time"] for item in row["lifelong"]}


def test_as_of_is_the_end_of_the_window(run):

    boundary = utc(END)

    for row in run["profile"]:
        assert row["as_of"] == boundary, row["client_id"]


def test_relationship_milestones_are_the_persona_start(run):
    """
    Начало отношений — та же дата, что у персоны, и KYC в тот же
    момент: клиент принят сразу после идентификации.
    """

    for row in run["profile"]:

        start, _ = run["personas"][row["client_id"]]
        found = items(row)

        assert found["relationship_started"] == utc(start), row["client_id"]
        assert found["kyc_passed"] == utc(start), row["client_id"]


def test_app_adopted_only_where_the_app_was_installed(run):
    """
    Веха есть ровно у тех, кто поставил приложение раньше as_of, и
    дата у неё та самая.
    """

    seen = {True: 0, False: 0}

    for row in run["profile"]:

        _, adopted = run["personas"][row["client_id"]]
        found = items(row)

        expected = adopted is not None and utc(adopted) < row["as_of"]

        seen[expected] += 1

        if expected:
            assert found["app_adopted"] == utc(adopted), row["client_id"]
        else:
            assert "app_adopted" not in found, row["client_id"]

    # Проверка не вырождена: есть клиенты с вехой и без неё.
    assert seen[True] and seen[False], seen


def test_milestones_are_before_as_of_and_in_order(run):

    for row in run["profile"]:

        order = [
            (item["event_time"], LIFELONG_TYPES.index(item["type"])) for item in row["lifelong"]
        ]

        assert order == sorted(order), row["client_id"]
        assert all(moment < row["as_of"] for moment, _ in order), row["client_id"]


def test_relationship_before_the_window_survives(run):
    """
    Клиент, пришедший до начала выгрузки: событий того времени в
    ленте нет, а веха о приходе есть.
    """

    start = utc(START)

    old = [
        row for row in run["profile"]
        if items(row)["relationship_started"] < start
    ]

    assert old, "проверка вырождена: все клиенты пришли в окне"

    for row in old:

        moments = [
            datetime.fromisoformat(event["event_time"])
            for event in run["events"]
            if event["client_id"] == row["client_id"]
        ]

        assert all(moment >= start for moment in moments), row["client_id"]


def test_milestones_are_not_events_of_the_tape():
    """
    Вехи живут только в анкете. Будь они ещё и событиями ленты,
    закрытое событие подсказывалось бы вехой с тем же временем.
    """

    assert not set(LIFELONG_TYPES) & set(EVENT_TYPES)


def test_milestone_exactly_at_as_of_is_left_out():
    """
    Граница полуоткрытая, как у событий: веха в самый момент as_of
    снимку ещё не известна.
    """

    boundary = datetime(2024, 4, 1)

    assert lifelong(boundary, None, boundary) == []

    before = boundary - timedelta(microseconds=1)

    found = lifelong(before, boundary, boundary)

    assert found == [
        {"type": "relationship_started", "event_time": utc(before)},
        {"type": "kyc_passed", "event_time": utc(before)},
    ]


def test_local_time_becomes_utc():

    moment = datetime(2024, 1, 1)

    assert utc(moment) == datetime(2023, 12, 31, 19, tzinfo=timezone.utc)
