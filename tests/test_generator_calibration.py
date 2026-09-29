from __future__ import annotations

import json
from datetime import datetime

import pyarrow.parquet as pq
import pytest

from src.generator import config, emit
from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.behaviour import habits
from src.generator.behaviour.sessions import GOAL_FLOW
from src.generator.life import events as life_events
from src.generator.life import lifecycle
from src.generator.life.persona import draw_persona
from src.generator.params.population import ACTIVITY_MODES
from src.generator.world import communities


# ============================================================
# ИДЕЯ
# ============================================================
#
# Правила, которые появились при калибровке по проверке на 1000
# клиентах:
#
#   - глубина сессии задана для каждой цели по её имени: ключи
#     параметра звались иначе, чем цели, и шесть целей из десяти
#     молча брали запасное значение;
#   - счета и подписки клиент проводит через банк с долей по режиму
#     активности: тихий держит их в другом банке;
#   - деньги извне идут туда, где клиент живёт: в паузе, где молчат
#     его переводы, внешних приходов в ленте нет.
# ============================================================


START = datetime(2024, 1, 1)
END = datetime(2025, 1, 1)
CLIENTS = 40
SEED = 77


def activate(seed: int, overrides: dict | None = None):

    config.activate_horizon(START, END, END)

    settings = emit._build_params(None, None, 4)

    if overrides:
        settings = settings.with_overrides(overrides)

    params_module.activate(settings)
    rng_module.configure(seed, settings.fingerprint(), 42)

    return settings


@pytest.fixture
def restore():

    saved = (config.HISTORY_START, config.HISTORY_END, config.REGISTRATION_END)

    yield

    config.activate_horizon(*saved)
    rng_module.clear_caches()


def test_every_session_goal_has_its_own_depth(restore):

    settings = activate(1)

    assert set(settings.activity.session_extra_screens) == set(GOAL_FLOW)


def recurring(share: float) -> tuple[int, int]:

    activate(1, {"activity": {"recurring_in_bank_share": {mode: share for mode in ACTIVITY_MODES}}})

    bills = subscriptions = 0

    for ordinal in range(1, 61):
        persona = draw_persona(ordinal)
        bills += len(habits._bills(persona, life_events.plan_events(persona)))
        subscriptions += len(habits._subscriptions(persona, persona.settlement))

    return bills, subscriptions


def test_recurring_payments_follow_the_share_through_the_bank(restore):

    bills, subscriptions = recurring(1.0)

    assert bills > 0 and subscriptions > 0

    assert recurring(0.0) == (0, 0)


@pytest.fixture(scope="module")
def tape(tmp_path_factory):

    out = tmp_path_factory.mktemp("calibration") / "tape"

    emit.generate_dataset(
        total_clients=CLIENTS, out_dir=out, seed=SEED, world_seed=42, history_start=START,
        history_end=END, registration_end=END, workers=1, community_size=4, quiet=True,
    )

    rows = pq.read_table(out / "events.parquet", columns=["client_id", "event_time", "payload"]).to_pylist()

    return [(row["client_id"], row["event_time"], json.loads(row["payload"])) for row in rows]


def test_money_from_outside_does_not_arrive_in_a_pause(tape, restore):

    activate(SEED)

    pauses = {}

    for ordinal in range(1, CLIENTS + 1):
        persona = draw_persona(ordinal)
        pauses[communities.client_id(ordinal)] = lifecycle.plan_pauses(
            persona, life_events.plan_events(persona)
        )

    quiet = [
        (client_id, pause) for client_id, items in pauses.items() for pause in items
        if "transfers" in lifecycle.silenced_streams((pause,), pause.start)
    ]

    assert quiet, "в выборке есть паузы, где молчат переводы"

    inbound = [
        (client_id, datetime.fromisoformat(event_time).replace(tzinfo=None))
        for client_id, event_time, payload in tape
        if payload["type"] == "transfer_in" and payload.get("reason") == "inbound"
    ]

    assert inbound, "в ленте есть внешние приходы"

    for client_id, moment in inbound:
        assert "transfers" not in lifecycle.silenced_streams(pauses[client_id], moment), (client_id, moment)
