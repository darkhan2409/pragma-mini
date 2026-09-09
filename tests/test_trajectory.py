"""
Скрытая динамика поведения: смесь сценариев и формы кривых.

Траектория ни к какой downstream-задаче не привязана: она лишь
меняет интенсивность и состав наблюдаемых событий.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pytest

from src.generator.config import FEATURE_END, HISTORY_START, LABEL_END
from src.generator.persona import draw_persona
from src.generator.trajectory import (
    ACTIVITY_SCENARIOS,
    STRESS_SCENARIOS,
    behavior_state,
    build_trajectory,
)


POPULATION = 3000

E = FEATURE_END


def day(offset: int) -> datetime:
    return E + timedelta(days=offset)


def activity(client_id: int, offset: int) -> float:
    return behavior_state(client_id, day(offset)).activity_multiplier


def stress(client_id: int, offset: int) -> float:
    return behavior_state(client_id, day(offset)).credit_stress


@pytest.fixture(scope="module")
def trajectories():
    return [build_trajectory(client_id) for client_id in range(POPULATION)]


def clients_with(trajectories, *, activity_scenario=None, stress_scenario=None, limit=10):
    found = []

    for trajectory in trajectories:

        if activity_scenario and trajectory.activity_scenario != activity_scenario:
            continue

        if stress_scenario and trajectory.stress_scenario != stress_scenario:
            continue

        found.append(trajectory.client_id)

        if len(found) == limit:
            break

    assert found, (activity_scenario, stress_scenario)
    return found


# ============================================================
# БАЗА
# ============================================================


def test_trajectory_is_deterministic():
    assert build_trajectory(17) == build_trajectory(17)

    ts = datetime(2025, 9, 14, 13, 0)
    assert behavior_state(17, ts) == behavior_state(17, ts)


def test_scenarios_are_neutral_names(trajectories):
    assert set(ACTIVITY_SCENARIOS) == {
        "stable",
        "gradual_decline",
        "sudden_drop",
        "temporary_dip",
    }
    assert set(STRESS_SCENARIOS) == {
        "none",
        "gradual",
        "sudden",
        "late_onset",
        "recovery",
    }

    for trajectory in trajectories:
        assert trajectory.activity_scenario in ACTIVITY_SCENARIOS
        assert trajectory.stress_scenario in STRESS_SCENARIOS


def test_state_values_are_bounded(trajectories):
    sample = [
        HISTORY_START + timedelta(days=offset)
        for offset in range(0, (LABEL_END - HISTORY_START).days + 1, 20)
    ]

    for trajectory in trajectories[:200]:
        for ts in sample:

            state = behavior_state(trajectory.client_id, ts)

            assert 0.0 < state.activity_multiplier <= 2.0
            assert 0.0 <= state.credit_stress <= 1.0
            assert 0.0 <= state.decline_pressure <= 1.0
            assert 0.0 <= state.utilization_pressure <= 1.0
            assert state.spending_multiplier > 0.0
            assert state.discretionary_multiplier > 0.0


# ============================================================
# СМЕСЬ
# ============================================================


def test_activity_mixture_shares(trajectories):
    counts = Counter(t.activity_scenario for t in trajectories)

    declining = counts["gradual_decline"] + counts["sudden_drop"]
    stable = POPULATION - declining

    assert 0.04 <= declining / POPULATION <= 0.14
    assert 0.50 <= counts["gradual_decline"] / declining <= 0.80
    assert 0.06 <= counts["temporary_dip"] / stable <= 0.18


def test_stress_mixture_shares(trajectories):
    counts = Counter(t.stress_scenario for t in trajectories)

    stressed = counts["gradual"] + counts["sudden"] + counts["late_onset"]
    calm = POPULATION - stressed

    assert 0.10 <= stressed / POPULATION <= 0.22
    assert 0.05 <= counts["recovery"] / calm <= 0.16

    for scenario in ("gradual", "sudden", "late_onset"):
        assert counts[scenario] > 0, scenario


def test_scenario_follows_persona(trajectories):
    declining, stable = [], []

    for trajectory in trajectories:
        value = draw_persona(trajectory.client_id).volatility

        if trajectory.activity_scenario in ("gradual_decline", "sudden_drop"):
            declining.append(value)
        else:
            stable.append(value)

    mean = lambda values: sum(values) / len(values)

    assert mean(declining) > mean(stable) + 0.05


# ============================================================
# ФОРМЫ КРИВЫХ
# ============================================================


def test_stable_stays_near_normal(trajectories):
    for client_id in clients_with(trajectories, activity_scenario="stable", limit=40):
        for offset in range(-700, 91, 20):
            assert 0.65 <= activity(client_id, offset) <= 1.45


def test_gradual_decline_shape(trajectories):
    for client_id in clients_with(trajectories, activity_scenario="gradual_decline"):
        assert activity(client_id, -180) > 0.70
        assert activity(client_id, 0) < 0.50
        assert activity(client_id, 60) < 0.10


def test_sudden_drop_has_no_precursor(trajectories):
    for client_id in clients_with(trajectories, activity_scenario="sudden_drop"):
        assert activity(client_id, -20) > 0.70
        assert activity(client_id, 45) < 0.10


def test_temporary_dip_recovers(trajectories):
    for client_id in clients_with(trajectories, activity_scenario="temporary_dip"):
        lowest = min(activity(client_id, offset) for offset in range(-120, 1, 5))
        assert lowest < 0.70
        assert activity(client_id, 89) > 0.75


def test_stress_shapes(trajectories):
    for client_id in clients_with(trajectories, stress_scenario="none", limit=40):
        assert max(stress(client_id, offset) for offset in range(-700, 91, 20)) < 0.20

    for client_id in clients_with(trajectories, stress_scenario="gradual"):
        assert stress(client_id, -300) < 0.20
        assert stress(client_id, 0) > 0.60

    for client_id in clients_with(trajectories, stress_scenario="sudden"):
        assert stress(client_id, -60) < 0.25
        assert stress(client_id, 45) > 0.70

    for client_id in clients_with(trajectories, stress_scenario="late_onset"):
        # Предвестников в окне признаков нет.
        assert stress(client_id, -1) < 0.25
        assert stress(client_id, 60) > 0.70

    for client_id in clients_with(trajectories, stress_scenario="recovery"):
        assert max(stress(client_id, offset) for offset in range(-300, -59, 5)) > 0.40
        assert stress(client_id, -20) < 0.25


def test_utilization_follows_stress(trajectories):
    calm = clients_with(trajectories, stress_scenario="none", limit=20)
    stressed = clients_with(trajectories, stress_scenario="gradual", limit=20)

    calm_pressure = sum(
        behavior_state(c, E).utilization_pressure for c in calm
    ) / len(calm)

    stressed_pressure = sum(
        behavior_state(c, E).utilization_pressure for c in stressed
    ) / len(stressed)

    assert stressed_pressure > 2.0 * calm_pressure
