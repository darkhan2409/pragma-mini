from __future__ import annotations

import copy
import math
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from src.generator import config, emit
from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.behaviour import engagement
from src.generator.life.persona import draw_persona


# ============================================================
# ИДЕЯ
# ============================================================
#
# Отношения клиента с банком — состояние с памятью, а не план:
#
#   - недовольство затухает по прошедшим дням, имеет потолок, а
#     хороший исход обращения снимает его часть;
#   - улика действует со своего дня, не раньше;
#   - одна неприятность не делает уход вероятным, а несколько
#     подряд, стресс и перенос жизни в другой банк — делают;
#   - вернуться можно при любом недовольстве;
#   - внешняя отлучка от недовольства, доли банка и стресса не
#     зависит;
#   - регулярные платежи уходят из банка по одному и возвращаются,
#     у зарплаты гистерезис;
#   - при глубокой миграции покупки здесь редки, но не нулевые
#     (прежде пауза other_bank обнуляла их, G2);
#   - петля «недовольство → жалоба → отказ → недовольство»
#     не разгоняется.
# ============================================================


START = datetime(2024, 1, 1)
END = datetime(2025, 1, 1)


@pytest.fixture
def world():

    saved = (config.HISTORY_START, config.HISTORY_END, config.REGISTRATION_END)

    def activate(overrides: dict | None = None):
        config.activate_horizon(START, END, END)
        settings = emit._build_params(None, None, 4)
        if overrides:
            settings = settings.with_overrides(overrides)
        params_module.activate(settings)
        rng_module.configure(77, settings.fingerprint(), 42)
        return settings

    yield activate

    config.activate_horizon(*saved)
    rng_module.clear_caches()


class Client:
    """
    Ровно то, что ядру нужно от состояния клиента.
    """

    def __init__(self, ordinal: int, mode: str = "regular"):
        persona = draw_persona(ordinal)
        self.persona = replace(
            persona, activity_mode=mode, relationship_start=datetime(2020, 1, 1),
            vanished_after_registration=False,
        )
        self.ordinal = ordinal
        self.client_id = persona.client_id
        self.life_events = ()
        self.income_streams = ()
        self.engagement = engagement.start(self.persona)
        self.engagement.observed = True


def days(start: datetime, count: int):
    return [start + timedelta(days=offset) for offset in range(count)]


# ============================================================
# НЕДОВОЛЬСТВО
# ============================================================


def test_friction_halves_in_its_half_life_and_has_a_cap(world):

    world()

    state = Client(1)
    item = state.engagement
    half = params_module.active().engagement.friction_half_life_days

    item.friction, item.friction_day = 1.0, START.toordinal()
    engagement._fold(item, START.toordinal() + int(half))

    assert item.friction == pytest.approx(0.5)

    for _ in range(50):
        engagement._pend(item, START + timedelta(days=40), 1.0, 0.0)

    engagement._fold(item, START.toordinal() + 41)

    assert item.friction <= params_module.active().engagement.friction_cap


def test_a_good_resolution_takes_part_of_the_friction_away(world):

    world()

    item = Client(1).engagement
    item.friction, item.friction_day = 2.0, START.toordinal()

    engagement._pend(item, START, 0.0, params_module.active().engagement.case_relief)
    engagement._fold(item, START.toordinal() + 1)

    assert item.friction < 2.0 * 0.6


def test_evidence_acts_from_its_own_day_not_before(world):

    world()

    item = Client(1).engagement
    item.friction_day = START.toordinal()

    engagement._pend(item, START + timedelta(days=5, hours=14), 1.0, 0.0)

    engagement._fold(item, START.toordinal() + 5)
    assert item.friction == 0.0

    engagement._fold(item, START.toordinal() + 6)
    assert item.friction > 0.9


# ============================================================
# ОПАСНОСТИ
# ============================================================


def test_one_bad_experience_does_not_make_leaving_likely(world):

    world()

    state = Client(2)
    item = state.engagement
    worst = max(params_module.active().engagement.evidence.values())

    item.friction = worst
    item.frailty = 1.0

    rate, _ = engagement.lapse_hazard(item, state.persona, START, 0.0, 0.0)

    # Меньше процента в день и меньше пятой части за месяц.
    assert engagement._chance(rate) < 0.01
    assert 1.0 - (1.0 - engagement._chance(rate)) ** 30 < 0.2


def test_hazards_grow_with_friction_stress_and_migration_and_fall_with_ties(world):

    world()

    state = Client(3)
    item = state.engagement

    def lapse(friction=0.0, stress=0.0, x=0.0, ties=0.0):
        probe = copy.copy(item)
        probe.friction, probe.x = friction, x
        return engagement.lapse_hazard(probe, state.persona, START, stress, ties)[0]

    assert lapse() < lapse(friction=1.0) < lapse(friction=3.0)
    assert lapse() < lapse(stress=0.5) < lapse(stress=1.0)
    assert lapse() < lapse(x=-0.5) < lapse(x=-2.0)
    assert lapse(ties=2.0) < lapse(ties=1.0) < lapse()

    def migration(friction=0.0, stress=0.0, ties=0.0, change=False):
        probe = copy.copy(item)
        probe.friction = friction
        return engagement.migration_hazard(probe, state.persona, stress, ties, change)[0]

    assert migration() < migration(friction=2.0)
    assert migration() < migration(stress=0.8)
    assert migration() < migration(change=True)
    assert migration(ties=2.0) < migration()


def test_returning_is_possible_at_any_friction_and_harder_when_unhappy(world):

    world()

    item = Client(4).engagement
    item.regime = engagement.LAPSED

    cap = params_module.active().engagement.friction_cap

    rates = []
    for friction in (0.0, 1.0, cap):
        probe = copy.copy(item)
        probe.friction = friction
        rates.append(engagement.return_hazard(probe, START, 0.0))

    assert all(rate > 0.0 for rate in rates)
    assert rates[0] > rates[1] > rates[2]

    clicked = copy.copy(item)
    clicked.winback_at = START - timedelta(days=3)

    assert engagement.return_hazard(clicked, START, 0.0) > rates[0]


def test_an_exogenous_absence_does_not_read_the_relationship(world):
    """
    Две копии клиента с разным недовольством, долей банка и
    стрессом уходят в отлучку в одни и те же дни: отлучка внешняя.
    Уход и миграция выключены, чтобы режим у копий был один, а
    отлучки участить, чтобы проверка не выродилась.
    """

    world({"engagement": {"lapse_rate": 0.0, "migration_rate": 0.0, "away_rate": 0.05}})

    calm = Client(5)
    upset = Client(5)
    upset.engagement.friction, upset.engagement.x = 3.0, -1.5

    for day in days(START, 400):
        for state, stress in ((calm, 0.0), (upset, 0.9)):
            state.engagement.friction_day = day.toordinal()
            engagement._step(state, day, stress, 0.0, False, False)
        upset.engagement.friction = 3.0

    away = [(row["time"], row["value"]) for row in calm.engagement.journal if row["component"] == "away"]

    assert away, "за 400 дней хоть одна отлучка"
    assert away == [(row["time"], row["value"]) for row in upset.engagement.journal if row["component"] == "away"]

    assert engagement.away_hazard(calm.persona, datetime(2024, 7, 15)) > engagement.away_hazard(
        calm.persona, datetime(2024, 1, 15)
    )


def test_the_day_draw_depends_only_on_client_and_day(world):

    world()

    first, second = Client(6), Client(6)
    other = Client(7)

    for day in days(START, 120):
        engagement._step(first, day, 0.1, 1.0, False, False)
        engagement._step(other, day, 0.1, 1.0, False, False)
        engagement._step(second, day, 0.1, 1.0, False, False)

    assert first.engagement.x == second.engagement.x
    assert first.engagement.journal == second.engagement.journal


# ============================================================
# ПОТОКИ
# ============================================================


def test_regular_payments_leave_one_by_one_and_come_back(world):

    world()

    state = Client(8)
    keys = [("bill", (f"kind{index}", index)) for index in range(12)]

    held = []
    for x in [0.0, -0.2, -0.5, -1.0, -2.0, -4.0]:
        state.engagement.x = x
        held.append({key for key in keys if engagement.in_bank(state, *key)})

    assert held[0] == set(keys)
    assert all(later <= earlier for earlier, later in zip(held, held[1:]))
    assert len(held[-1]) < len(held[0])

    state.engagement.x = 0.0
    assert {key for key in keys if engagement.in_bank(state, *key)} == set(keys)


def test_the_salary_switch_has_hysteresis(world):

    world()

    state = Client(9)
    item = state.engagement
    gap = params_module.active().engagement.tie_hysteresis

    def settle(level: float):
        item.x = math.log(level) / item.sensitivity["ties"]
        item.target = item.x
        item.half_life = 1e9
        engagement._step(state, START, 0.0, 0.0, False, False)

    settle(item.salary_threshold * 0.9)
    assert item.salary_here is False

    settle(item.salary_threshold + gap / 2)
    assert item.salary_here is False

    settle(item.salary_threshold + gap * 2)
    assert item.salary_here is True


def test_purchases_go_on_rarely_during_a_deep_migration(world):
    """
    Клиент, перенёсший жизнь в другой банк, иногда ещё платит
    здешней картой. Прежняя пауза other_bank обнуляла покупки целиком
    (аудит 2026-10-05, G2).
    """

    world()

    state = Client(10)
    state.engagement.x = math.log(0.05)

    assert "purchases" not in engagement.silenced(state, START)
    assert 0.0 < engagement.factor(state, "purchases") < 0.5


def test_a_lapsed_client_with_debt_keeps_only_service_visits(world):

    world()

    state = Client(11)
    state.engagement.regime = engagement.LAPSED

    state.engagement.servicing = False
    assert "sessions" in engagement.silenced(state, START)

    state.engagement.servicing = True
    assert "sessions" not in engagement.silenced(state, START)
    assert {"purchases", "transfers", "cash", "bills"} <= engagement.silenced(state, START)
    assert engagement.factor(state, "sessions") <= params_module.active().engagement.servicing


def test_the_complaint_loop_does_not_run_away(world):
    """
    Худший случай: каждая жалоба кончается отказом и долгим сроком.
    Прибавка за кулдаун меньше, чем недовольство успевает забыться
    у потолка, — неподвижная точка ниже потолка.
    """

    settings = world().engagement

    rate, cooldown = settings.complaint
    worst = settings.evidence["case_declined"] + settings.evidence["case_slow"]

    kept = 2.0 ** (-cooldown / settings.friction_half_life_days)

    fixed_point = worst / (1.0 - kept)

    assert fixed_point < settings.friction_cap
    assert min(0.05, rate * settings.friction_cap) <= 0.05
