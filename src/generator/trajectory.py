from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import numpy as np

from .config import FEATURE_END, HISTORY_START, LABEL_END
from .persona import Persona, draw_persona
from .rng import NS_TRAJECTORY, client_rng


# ============================================================
# ИДЕЯ
# ============================================================
#
# Persona статична. Траектория описывает, как поведение
# клиента МЕНЯЕТСЯ во времени:
#
#     Persona
#        ↓
#     mixture assignment   (сценарий активности + сценарий стресса)
#        ↓
#     piecewise-linear curves
#        ↓
#     BehaviorState(t)
#
# Все узлы кривых разыгрываются ОДИН раз детерминированно
# из client_rng(client_id, NS_TRAJECTORY).
#
# behavior_state(client_id, ts) не использует RNG:
# это чистая интерполяция.
#
# Траектория это ТОЛЬКО скрытая динамика поведения.
# Она не привязана ни к какой downstream-задаче: сценарии
# сдвигают интенсивность и состав наблюдаемых событий,
# а что из этого будет меткой, решает derive.py.
# ============================================================


# ============================================================
# SCENARIOS
# ============================================================

ACTIVITY_SCENARIOS = (
    "stable",
    "gradual_decline",
    "sudden_drop",
    "temporary_dip",
)

STRESS_SCENARIOS = (
    "none",
    "gradual",
    "sudden",
    "late_onset",
    "recovery",
)


# ============================================================
# MIXTURE PARAMETERS
# ============================================================
#
# Доли ниже задают структуру смеси.
#
# Смесь обязательна: если бы спад всегда был плавным,
# а стресс всегда с предвестниками, задача сводилась бы
# к паре счётчиков за последний месяц.
# ============================================================

# P(сценарий спада) = DECLINE_BASE + DECLINE_SLOPE * volatility
DECLINE_BASE = 0.02
DECLINE_SLOPE = 0.30
DECLINE_MAX = 0.60

# Среди спадающих: доля плавного спада, остальное резкий обрыв.
GRADUAL_DECLINE_SHARE = 0.65

# Среди остающихся: доля временного провала с восстановлением.
TEMPORARY_DIP_SHARE = 0.12

# P(кредитный стресс) = STRESS_BASE + STRESS_SLOPE * risk
STRESS_BASE = 0.01
STRESS_SLOPE = 0.60
STRESS_MAX = 0.50

# Среди стрессовых: плавный / внезапный / без предвестников.
GRADUAL_SHARE = 0.55
SUDDEN_SHARE = 0.25
# остаток -> late_onset

# Среди нестрессовых: доля «стресс был, но восстановился».
RECOVERY_SHARE = 0.10

# Естественные колебания активности между узлами.
BASE_DRIFT_SIGMA = 0.08
BASE_ANCHOR_STEP_DAYS = 90

# Остаточная активность после обрыва (не строгий ноль).
ACTIVITY_FLOOR = 0.04

# Постоянный фон кредитного стресса от persona.risk.
BASELINE_STRESS_SCALE = 0.10


# ============================================================
# STATE
# ============================================================


@dataclass(frozen=True)
class BehaviorState:
    """
    Состояние поведения клиента в конкретный день.

    activity_multiplier:
        множитель интенсивности событий
        (транзакции, app-сессии, переводы). 1.0 = норма.

    spending_multiplier:
        множитель среднего чека.

    discretionary_multiplier:
        множитель веса дискреционных категорий
        (рестораны, путешествия, одежда...).

    utilization_pressure:
        0..1, насколько выбран кредитный лимит сверх базового
        уровня (в стрессе клиент живёт на кредитке).

    credit_stress:
        0..1, драйвер просрочек, loan-intent, напоминаний.

    decline_pressure:
        0..1, насколько активность ниже нормы.
    """

    activity_multiplier: float
    spending_multiplier: float
    discretionary_multiplier: float
    utilization_pressure: float
    credit_stress: float
    decline_pressure: float


# ============================================================
# CURVE
# ============================================================


@dataclass(frozen=True)
class Curve:
    """
    Кусочно-линейная кривая по дням (ordinal).

    До первого узла значение первого узла,
    после последнего значение последнего.
    """

    days: tuple[int, ...]
    values: tuple[float, ...]

    def at(self, day: int) -> float:

        days = self.days

        if day <= days[0]:
            return self.values[0]

        if day >= days[-1]:
            return self.values[-1]

        # индекс первого узла строго правее day
        right = bisect_right(days, day)
        left = right - 1

        d0 = days[left]
        d1 = days[right]

        v0 = self.values[left]
        v1 = self.values[right]

        weight = (day - d0) / (d1 - d0)

        return v0 + weight * (v1 - v0)


def make_curve(points: list[tuple[int, float]]) -> Curve:

    if not points:
        raise ValueError("curve needs at least one point")

    points = sorted(points, key=lambda p: p[0])

    days = tuple(int(d) for d, _ in points)
    values = tuple(float(v) for _, v in points)

    if len(set(days)) != len(days):
        raise ValueError(f"duplicate curve days: {days}")

    return Curve(days=days, values=values)


# ============================================================
# TRAJECTORY
# ============================================================


@dataclass(frozen=True)
class Trajectory:

    client_id: int

    activity_scenario: str
    stress_scenario: str

    # естественные колебания активности вокруг 1.0
    activity_base: Curve

    # сценарный множитель активности (1.0 = без эффекта)
    activity_curve: Curve

    # добавка к фону кредитного стресса, 0..1
    stress_curve: Curve

    baseline_stress: float

    # --------------------------------------------------------

    def activity_at(self, day: int) -> float:
        return self.activity_base.at(day) * self.activity_curve.at(day)

    def stress_at(self, day: int) -> float:
        return min(1.0, max(0.0, self.baseline_stress + self.stress_curve.at(day)))

    def state_for_day(self, day: int) -> BehaviorState:

        activity = self.activity_at(day)
        stress = self.stress_at(day)

        return BehaviorState(
            activity_multiplier=activity,
            spending_multiplier=1.0 - 0.25 * stress,
            discretionary_multiplier=1.0 - 0.55 * stress,
            utilization_pressure=min(1.0, 0.85 * stress + 0.10),
            credit_stress=stress,
            decline_pressure=min(1.0, max(0.0, 1.0 - activity)),
        )

    def state_at(self, ts: datetime) -> BehaviorState:
        return self.state_for_day(ts.toordinal())


# ============================================================
# MIXTURE ASSIGNMENT
# ============================================================


def choose_activity_scenario(
    persona: Persona,
    rng: np.random.Generator,
) -> str:

    p_decline = float(
        np.clip(
            DECLINE_BASE + DECLINE_SLOPE * persona.volatility,
            0.0,
            DECLINE_MAX,
        )
    )

    # Порядок обращений к rng фиксирован: всегда два вызова.
    u_decline = rng.random()
    u_kind = rng.random()

    if u_decline < p_decline:
        if u_kind < GRADUAL_DECLINE_SHARE:
            return "gradual_decline"
        return "sudden_drop"

    if u_kind < TEMPORARY_DIP_SHARE:
        return "temporary_dip"

    return "stable"


def choose_stress_scenario(
    persona: Persona,
    rng: np.random.Generator,
) -> str:

    p_stress = float(
        np.clip(
            STRESS_BASE + STRESS_SLOPE * persona.risk,
            0.0,
            STRESS_MAX,
        )
    )

    u_stress = rng.random()
    u_kind = rng.random()

    if u_stress < p_stress:
        if u_kind < GRADUAL_SHARE:
            return "gradual"
        if u_kind < GRADUAL_SHARE + SUDDEN_SHARE:
            return "sudden"
        return "late_onset"

    if u_kind < RECOVERY_SHARE:
        return "recovery"

    return "none"


# ============================================================
# CURVE BUILDERS
# ============================================================


def build_activity_base(rng: np.random.Generator) -> Curve:
    """
    Медленный дрейф активности: узлы каждые ~90 дней,
    значения exp(N(0, sigma)) вокруг 1.0.
    """

    first = HISTORY_START.toordinal() - BASE_ANCHOR_STEP_DAYS
    last = LABEL_END.toordinal() + BASE_ANCHOR_STEP_DAYS

    points: list[tuple[int, float]] = []

    day = first
    while day <= last:
        value = float(np.exp(rng.normal(0.0, BASE_DRIFT_SIGMA)))
        points.append((day, value))
        day += BASE_ANCHOR_STEP_DAYS

    return make_curve(points)


def build_activity_curve(
    scenario: str,
    rng: np.random.Generator,
) -> Curve:

    end = FEATURE_END.toordinal()

    # --------------------------------------------------------
    # GRADUAL DECLINE
    # --------------------------------------------------------
    #
    # Спад начинается за 60–120 дней до границы признаков,
    # к границе активность 15–40% от нормы,
    # через месяц после границы почти ноль.
    # --------------------------------------------------------

    if scenario == "gradual_decline":

        start = end - int(rng.integers(60, 121))
        floor_at_end = float(rng.uniform(0.15, 0.40))

        return make_curve(
            [
                (start, 1.0),
                (end, floor_at_end),
                (end + 30, ACTIVITY_FLOOR),
            ]
        )

    # --------------------------------------------------------
    # SUDDEN DROP
    # --------------------------------------------------------
    #
    # До точки обрыва поведение нормальное.
    # Обрыв от 10 дней до границы до 20 дней после.
    # --------------------------------------------------------

    if scenario == "sudden_drop":

        cut = end + int(rng.integers(-10, 21))

        return make_curve(
            [
                (cut, 1.0),
                (cut + 3, ACTIVITY_FLOOR),
            ]
        )

    # --------------------------------------------------------
    # TEMPORARY DIP (hard negative)
    # --------------------------------------------------------
    #
    # Провал перед границей, восстановление в окне меток.
    # --------------------------------------------------------

    if scenario == "temporary_dip":

        start = end - int(rng.integers(45, 121))
        bottom = end - int(rng.integers(0, 31))
        bottom = max(bottom, start + 10)

        depth = float(rng.uniform(0.30, 0.60))
        recovered = end + int(rng.integers(20, 61))

        return make_curve(
            [
                (start, 1.0),
                (bottom, depth),
                (recovered, 1.0),
            ]
        )

    # --------------------------------------------------------
    # STABLE
    # --------------------------------------------------------

    if scenario == "stable":
        return make_curve([(end, 1.0)])

    raise ValueError(f"unknown churn scenario: {scenario}")


def build_stress_curve(
    scenario: str,
    rng: np.random.Generator,
) -> Curve:

    end = FEATURE_END.toordinal()

    # --------------------------------------------------------
    # GRADUAL STRESS
    # --------------------------------------------------------
    #
    # Стресс нарастает 5–8 месяцев до границы
    # и остаётся высоким в окне меток.
    # --------------------------------------------------------

    if scenario == "gradual":

        start = end - int(rng.integers(150, 241))
        peak = float(rng.uniform(0.70, 0.95))

        return make_curve(
            [
                (start, 0.0),
                (end, peak),
            ]
        )

    # --------------------------------------------------------
    # SUDDEN SHOCK
    # --------------------------------------------------------
    #
    # Резкий финансовый шок за месяц до или после границы.
    # --------------------------------------------------------

    if scenario == "sudden":

        # Шок не позже +10 дней от границы: иначе два пропуска
        # подряд не успевают уместиться в 90-дневное окно меток.
        shock = end + int(rng.integers(-45, 11))
        peak = float(rng.uniform(0.80, 1.00))

        return make_curve(
            [
                (shock, 0.0),
                (shock + 5, peak),
            ]
        )

    # --------------------------------------------------------
    # LATE ONSET
    # --------------------------------------------------------
    #
    # Предвестники только внутри окна меток:
    # в признаках их нет.
    # --------------------------------------------------------

    if scenario == "late_onset":

        # Строго после границы признаков, но в первые недели окна:
        # предвестников в признаках нет, дефолт в окне возможен.
        onset = end + int(rng.integers(0, 16))
        peak = float(rng.uniform(0.80, 1.00))

        return make_curve(
            [
                (onset, 0.0),
                (onset + 5, peak),
            ]
        )

    # --------------------------------------------------------
    # RECOVERY (hard negative)
    # --------------------------------------------------------
    #
    # Стресс был в середине истории и сошёл на нет
    # задолго до границы.
    # --------------------------------------------------------

    if scenario == "recovery":

        start = end - int(rng.integers(240, 301))
        peak_day = end - int(rng.integers(90, 151))
        back = end - int(rng.integers(30, 61))

        peak = float(rng.uniform(0.50, 0.80))

        return make_curve(
            [
                (start, 0.0),
                (peak_day, peak),
                (back, 0.0),
            ]
        )

    # --------------------------------------------------------
    # NONE
    # --------------------------------------------------------

    if scenario == "none":
        return make_curve([(end, 0.0)])

    raise ValueError(f"unknown credit scenario: {scenario}")


# ============================================================
# BUILD
# ============================================================


@lru_cache(maxsize=131_072)
def build_trajectory(client_id: int) -> Trajectory:
    """
    Детерминированная траектория клиента.

    Кэшируется: состояние спрашивает каждое событие клиента.
    """

    persona = draw_persona(client_id)
    rng = client_rng(client_id, NS_TRAJECTORY)

    # Порядок розыгрыша фиксирован. Не переставлять.
    activity_scenario = choose_activity_scenario(persona, rng)
    stress_scenario = choose_stress_scenario(persona, rng)

    activity_base = build_activity_base(rng)
    activity_curve = build_activity_curve(activity_scenario, rng)
    stress_curve = build_stress_curve(stress_scenario, rng)

    return Trajectory(
        client_id=client_id,
        activity_scenario=activity_scenario,
        stress_scenario=stress_scenario,
        activity_base=activity_base,
        activity_curve=activity_curve,
        stress_curve=stress_curve,
        baseline_stress=BASELINE_STRESS_SCALE * persona.risk,
    )


# ============================================================
# PUBLIC API
# ============================================================


@lru_cache(maxsize=262_144)
def behavior_state_for_day(client_id: int, day: int) -> BehaviorState:
    """
    Состояние на день (ordinal). Кэшируется:
    его спрашивает каждое событие и каждое ребро в этот день.
    """

    return build_trajectory(client_id).state_for_day(day)


def behavior_state(client_id: int, ts: datetime) -> BehaviorState:
    """
    Состояние поведения клиента в момент ts.

    Чистая функция: без RNG, одинаковый вход даёт одинаковый выход.
    Разрешение дневное.
    """

    return behavior_state_for_day(client_id, ts.toordinal())
