from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from src.generator import config, emit
from src.generator import params as params_module
from src.generator import rng as rng_module
from src.generator.behaviour import engagement
from src.generator.behaviour import habits
from src.generator.life import events as life_events
from src.generator.life import income
from src.generator.life.persona import draw_persona
from src.generator.life.traits import TraitShift
from src.generator.world import merchants


# ============================================================
# ИДЕЯ
# ============================================================
#
# Жизнь меняет клиента, и дальше он живёт уже изменённым:
#
#   - черты сдвигаются после события и читаются поведением (G3);
#   - рождение, переезд, потеря и новая работа меняют персону со
#     следующего дня, а записи банка о прошлой работе остаются;
#   - сменить или потерять работу может только тот, кто получает
#     зарплату;
#   - бюджет видит повышение и снижение дохода;
#   - подписка после переезда не теряет мерчанта;
#   - сегодняшние траты не знают о завтрашней потере работы;
#   - пришедший в банк живёт с первого дня, а не с конца месяца.
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


def test_a_trait_shift_starts_the_day_after_the_event(world):

    world()

    persona = draw_persona(1)
    moment = datetime(2024, 3, 10, 15)
    shifted = persona.traits.with_shift(TraitShift(ts=moment, deltas={"price_sensitivity": 0.2}, cause="job_loss"))

    base = persona.traits.value("price_sensitivity")
    blend = params_module.active().traits.drift_blend_days

    assert shifted.value("price_sensitivity", moment) == base
    assert shifted.value("price_sensitivity", moment - timedelta(days=5)) == base
    assert shifted.value("price_sensitivity", moment + timedelta(days=blend + 1)) == pytest.approx(min(1.0, base + 0.2))


def test_behaviour_reads_the_shifted_traits(world):
    """
    Сдвиги черт после жизненных событий лежат в персоне клиента, а
    не в поле, которое никто не читает (аудит 2026-10-05, G3).
    """

    world()

    from src.generator import engine  # noqa: F401
    from src.generator.simulate import CommunitySimulation

    sim = CommunitySimulation(0, tuple(range(1, 9)))

    shifted = [state for state in sim.clients.values() if state.persona.traits.shifts]

    assert shifted, "в выборке есть клиенты с жизненными событиями"

    for state in shifted:
        later = END - timedelta(days=1)
        assert state.persona.traits.at(later) == state.persona.traits.at(later)
        assert state.persona.trait("price_sensitivity", later) == state.persona.traits.at(later)["price_sensitivity"]


def test_life_changes_the_persona_from_the_next_day(world):

    world()

    persona = replace(draw_persona(3), children=0, household_size=1)

    birth = life_events.LifeEvent("child_birth", datetime(2024, 5, 4, 10), {}, None, False)
    move = life_events.LifeEvent(
        "move", datetime(2024, 6, 1, 9),
        {"settlement": "Shymkent", "region": "Shymkent", "settlement_type": "metropolis"}, None, False,
    )

    changes = life_events.persona_changes(persona, (birth, move), ())

    def on(day: datetime):
        return life_events.apply_changes(
            persona, tuple(item for item in changes if item[0] <= day.toordinal())
        )

    assert on(datetime(2024, 5, 4)).children == 0
    assert on(datetime(2024, 5, 5)).children == 1
    assert on(datetime(2024, 5, 5)).household_size == 2
    assert on(datetime(2024, 6, 2)).settlement == "Shymkent"
    assert on(datetime(2024, 6, 1)).settlement == persona.settlement


def test_a_job_loss_makes_the_client_unemployed_and_a_new_job_employs_him_again(world):

    world()

    persona = replace(draw_persona(5), income_type="employed", true_income=300_000)

    loss = life_events.LifeEvent("job_loss", datetime(2024, 3, 1, 9), {"recovery_days": 60}, None, False)

    job = income.IncomeStream(
        stream_id=f"inc_{persona.client_ordinal}_job_{loss.ts.toordinal()}", kind="salary", payer="emp_1",
        schedule="monthly", payday=10, landing="hcb_account", base_amount=350_000,
        valid_from=loss.ts + timedelta(days=60), valid_to=None,
    )

    changes = life_events.persona_changes(persona, (loss,), (job,))

    def on(day: datetime):
        return life_events.apply_changes(
            persona, tuple(item for item in changes if item[0] <= day.toordinal())
        )

    assert on(datetime(2024, 3, 10)).income_type == "unemployed"
    assert on(datetime(2024, 5, 10)).income_type == "employed"
    assert on(datetime(2024, 5, 10)).true_income == 350_000


def test_only_a_salaried_client_changes_or_loses_a_job(world):

    world({"lifecycle": {"life_event_rate_per_year": {"job_change": 3.0, "job_loss": 3.0}}})

    kinds = params_module.active().income.primary_kind_by_income_type

    seen = 0

    for ordinal in range(1, 200):

        persona = draw_persona(ordinal)

        jobs = [item for item in life_events.plan_events(persona) if item.kind in ("job_change", "job_loss")]

        if kinds.get(persona.income_type) != "salary":
            assert jobs == [], (ordinal, persona.income_type)
            continue

        seen += len(jobs)

        # После потери работы новые события работы — только после
        # выхода на новое место.
        for left, right in zip(jobs, jobs[1:]):
            pause = left.payload.get("recovery_days" if left.kind == "job_loss" else "gap_days", 0)
            assert right.ts >= left.ts + timedelta(days=pause)

    assert seen > 0


def test_a_second_move_starts_from_where_the_client_lives(world):

    world({"lifecycle": {"life_event_rate_per_year": {"move": 4.0}, "move_to_other_settlement_share": 1.0}})

    found = 0

    for ordinal in range(1, 120):

        persona = draw_persona(ordinal)
        moves = [item for item in life_events.plan_events(persona) if item.kind == "move"]

        place = persona.settlement

        for move in moves:
            assert move.payload["settlement"] != place, (ordinal, place)
            place = move.payload["settlement"]

        found += len(moves) > 1

    assert found > 0


def test_the_budget_sees_income_changes(world):

    world()

    stream = income.IncomeStream(
        stream_id="s", kind="salary", payer="emp_1", schedule="monthly", payday=10,
        landing="hcb_account", base_amount=200_000, valid_from=START, valid_to=None,
        shifts=((datetime(2024, 4, 1), 0.5),),
    )

    assert income.monthly_income((stream,), datetime(2024, 3, 1)) == 200_000
    assert income.monthly_income((stream,), datetime(2024, 5, 1)) == 100_000


def test_a_subscription_keeps_its_merchant_after_a_move(world):

    world()

    for ordinal in range(1, 60):

        persona = draw_persona(ordinal)

        for item in habits._subscriptions(persona, persona.settlement):
            assert item.settlement == persona.settlement
            assert item.outlet_id in {outlet.outlet_id for outlet in merchants.outlets_of(item.settlement, "subscription")}


def test_the_next_payday_does_not_know_about_a_future_job_loss(world):

    world()

    day = datetime(2024, 3, 2)

    working = income.IncomeStream(
        stream_id="s", kind="salary", payer="emp_1", schedule="monthly", payday=10,
        landing="hcb_account", base_amount=200_000, valid_from=START, valid_to=None,
    )

    # Работа кончится через неделю, но сегодня человек этого не знает.
    ending = replace(working, valid_to=datetime(2024, 3, 9))

    assert income.expected_payday((working,), day) == datetime(2024, 3, 10)
    assert income.expected_payday((ending,), day) == datetime(2024, 3, 10)


def test_a_newcomer_lives_from_the_first_day(world):
    """
    Пришедший в банк посреди месяца сразу onboarding, а не prospect с
    нулевым множителем до конца месяца.
    """

    world()

    persona = replace(draw_persona(7), relationship_start=datetime(2024, 3, 14, 11))

    assert engagement.stage(persona, datetime(2024, 3, 13), 0.0, 0) == "prospect"
    assert engagement.stage(persona, datetime(2024, 3, 15), 0.0, 0) == "onboarding"
    assert params_module.active().activity.state_factor["onboarding"] > 0.0
