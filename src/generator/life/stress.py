from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_STRESS, keyed_rng
from .events import EVENT_KINDS
from .persona import Persona

# Номер вида жизненного события в ключах розыгрыша.
EVENT_KIND_CODES = {kind: index for index, kind in enumerate(EVENT_KINDS)}


# ============================================================
# ФИНАНСОВЫЙ СТРЕСС
# ============================================================
#
# Стресс это ПЕРИОД с причиной, нарастанием, пиком, спадом и
# исходом, а не постоянный тип клиента.
#
# Он влияет на всё сразу и постепенно: траты, остатки,
# утилизацию лимита, платёжную дисциплину, обращения в
# поддержку, интерес к рефинансированию и использование
# приложения.
# ============================================================


@dataclass(frozen=True)
class StressEpisode:
    trigger: str
    start: datetime
    peak_start: datetime
    peak_end: datetime
    end: datetime
    intensity: float
    resolution: str
    resolved_at: datetime | None

    def level(self, ts: datetime) -> float:
        """
        Интенсивность на дату: нарастание, плато, спад, хвост.

        У неразрешённого эпизода конца по существу нет: причина
        осталась, и после окончания окна давление держится на
        доле интенсивности до конца истории. Раньше исход
        unresolved ничем не отличался от восстановления дохода.
        """

        if ts < self.start:
            return 0.0

        if ts >= self.end:
            if self.resolution != "unresolved":
                return 0.0
            tail = params_module.active().stress.unresolved_tail_share
            return self.intensity * float(tail)

        if ts < self.peak_start:
            span = max(1.0, (self.peak_start - self.start).total_seconds())
            return self.intensity * ((ts - self.start).total_seconds() / span)

        if ts <= self.peak_end:
            return self.intensity

        span = max(1.0, (self.end - self.peak_end).total_seconds())

        return self.intensity * (1.0 - (ts - self.peak_end).total_seconds() / span)


def _trigger_events(persona: Persona, events: tuple) -> list[tuple[str, datetime, int]]:
    """
    Жизненные события, породившие стресс: причина, момент и ключ
    розыгрыша эпизода. Ключ — вид события и его номер среди своих
    (LifeEvent.serial), а не место в общем списке: событие другого
    вида или вне окна не меняет розыгрыш этого.
    """

    settings = params_module.active().stress

    triggers: list[tuple[str, datetime, int]] = []

    mapping = {
        "job_loss": "job_loss",
        "job_change": "job_change",
        "illness": "illness",
        "big_purchase": "big_purchase",
        "divorce": "divorce",
        "move": "move",
        "income_down": "obligation_growth",
    }

    for event in events:

        trigger = mapping.get(event.kind)

        if trigger is None:
            continue

        key = 1000 * (1 + EVENT_KIND_CODES[event.kind]) + event.serial

        rng = keyed_rng(NS_STRESS, persona.client_ordinal, 1, key)

        if rng.random() < settings.trigger_probability.get(trigger, 0.0):
            triggers.append((trigger, event.ts, key))

    return triggers


def plan_episodes(persona: Persona, events: tuple) -> tuple:
    """
    Эпизоды стресса клиента на горизонте.
    """

    settings = params_module.active().stress

    discipline = persona.trait("financial_discipline")
    savings = persona.trait("savings_propensity")

    candidates = _trigger_events(persona, events)

    # Спонтанные шоки идут вперёд по времени: следующий — через
    # экспоненциальный интервал после предыдущего. Горизонт
    # планирования их только обрезает.
    clock = 0.0
    index = 0

    while settings.random_shock_per_year > 0.0:
        item_rng = keyed_rng(NS_STRESS, persona.client_ordinal, 3, index)
        clock += -math.log(1.0 - item_rng.random()) / settings.random_shock_per_year * 365.25
        moment = config.HISTORY_START + timedelta(days=int(clock))
        if moment >= config.PLANNING_END:
            break
        candidates.append(("random_shock", moment, index))
        index += 1

    candidates.sort(key=lambda item: (item[1], item[2]))

    episodes: list[StressEpisode] = []

    last_end: datetime | None = None

    for trigger, moment, key in candidates:

        if len(episodes) >= settings.max_episodes:
            break

        if last_end is not None and moment < last_end + timedelta(days=settings.cooldown_days):
            continue

        rng = keyed_rng(NS_STRESS, persona.client_ordinal, 4, key)

        low, high = settings.intensity_range[trigger]
        intensity = rng.uniform(low, high)

        intensity *= 1.0 - settings.discipline_intensity_factor * discipline
        intensity *= 1.0 - settings.savings_buffer_factor * savings

        if episodes:
            intensity *= settings.repeat_boost

        intensity = float(min(1.0, max(0.05, intensity)))

        low_days, high_days = settings.length_days[trigger]
        length = rng.integers(low_days, high_days + 1)
        length = int(max(10, length * (1.0 - settings.discipline_length_factor * discipline)))

        onset = int(length * rng.uniform(*settings.onset_share))
        decay = int(length * rng.uniform(*settings.decay_share))

        start = moment
        peak_start = start + timedelta(days=max(1, onset))
        end = start + timedelta(days=length)
        peak_end = max(peak_start, end - timedelta(days=max(1, decay)))

        resolution = rng.weighted(settings.resolution_weights)

        resolved_at = end if resolution != "unresolved" else None

        episodes.append(
            StressEpisode(
                trigger=trigger,
                start=start,
                peak_start=peak_start,
                peak_end=peak_end,
                end=end,
                intensity=intensity,
                resolution=resolution,
                resolved_at=resolved_at,
            )
        )

        last_end = end

    return tuple(episodes)


def level_at(episodes: tuple, ts: datetime) -> float:
    """
    Суммарный уровень стресса на дату.
    """

    total = 0.0

    for episode in episodes:
        total += episode.level(ts)

    return float(min(1.0, total))


def active_episode(episodes: tuple, ts: datetime) -> StressEpisode | None:

    for episode in episodes:
        if episode.start <= ts < episode.end:
            return episode

    return None


__all__ = ["StressEpisode", "active_episode", "level_at", "plan_episodes"]
