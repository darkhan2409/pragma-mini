from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .. import params as params_module
from ..rng import NS_TRAITS, numpy_rng, state_cache


# ============================================================
# НЕПРЕРЫВНЫЕ СКРЫТЫЕ ХАРАКТЕРИСТИКИ
# ============================================================
#
# Архетип задаёт РАСПРЕДЕЛЕНИЕ, а не поведение: он сдвигает
# средние, но разброс внутри архетипа остаётся большим, и два
# клиента одного архетипа заметно различаются.
#
# Характеристики коррелированы, поэтому разыгрываются гауссовой
# копулой по общей матрице, а не независимо.
#
# Значения живут ТОЛЬКО в скрытой истине и медленно смещаются
# после жизненных событий.
# ============================================================


def _logit(value: float) -> float:
    value = min(max(value, 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@state_cache
def _cholesky() -> np.ndarray:
    """
    Нижнетреугольный множитель корреляционной матрицы.
    Матрица чинится до положительно определённой обрезкой
    собственных значений: иначе разложение не существует.
    """

    settings = params_module.active().traits

    names = list(settings.names)
    size = len(names)

    matrix = np.eye(size)

    index = {name: position for position, name in enumerate(names)}

    for (left, right), value in settings.correlations.items():
        if left not in index or right not in index:
            continue
        matrix[index[left], index[right]] = value
        matrix[index[right], index[left]] = value

    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-6, None)

    repaired = vectors @ np.diag(values) @ vectors.T

    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)

    return np.linalg.cholesky(repaired)


@dataclass(frozen=True)
class TraitShift:
    ts: datetime
    deltas: dict
    cause: str


@dataclass(frozen=True)
class Traits:
    """
    Скрытый портрет клиента. Базовые значения и накопленные
    сдвиги после жизненных событий.
    """

    base: dict
    channel_preferences: dict
    shifts: tuple = ()

    def at(self, ts: datetime) -> dict:
        """
        Значения характеристик на дату с учётом дрейфа.
        """

        if not self.shifts:
            return self.base

        settings = params_module.active().traits

        values = dict(self.base)

        for shift in self.shifts:

            if shift.ts > ts:
                break

            elapsed = (ts - shift.ts).days

            weight = min(1.0, max(0.0, elapsed / max(1, settings.drift_blend_days)))

            for name, delta in shift.deltas.items():
                if name in values:
                    values[name] = float(min(1.0, max(0.0, values[name] + delta * weight)))

        return values

    def value(self, name: str, ts: datetime | None = None) -> float:
        if ts is None:
            return float(self.base.get(name, 0.5))
        return float(self.at(ts).get(name, 0.5))

    def with_shift(self, shift: TraitShift) -> Traits:
        return Traits(
            base=self.base,
            channel_preferences=self.channel_preferences,
            shifts=tuple(sorted(self.shifts + (shift,), key=lambda item: item.ts)),
        )


def draw_traits(
    client_ordinal: int,
    life_stage: str,
    hcb_role: str,
    activity_mode: str,
    settlement_type: str,
) -> Traits:
    """
    Характеристики клиента по его архетипу.
    """

    settings = params_module.active().traits

    names = list(settings.names)

    means = {name: float(settings.base_mean[name]) for name in names}

    for table, key in (
        (settings.stage_shift, life_stage),
        (settings.role_shift, hcb_role),
        (settings.mode_shift, activity_mode),
        (settings.settlement_shift, settlement_type),
    ):
        for name, delta in (table.get(key) or {}).items():
            if name in means:
                means[name] = min(0.97, max(0.03, means[name] + delta))

    rng = numpy_rng(NS_TRAITS, client_ordinal)

    noise = rng.standard_normal(len(names))

    correlated = _cholesky() @ noise

    base = {}

    for position, name in enumerate(names):
        centre = _logit(means[name])
        value = _sigmoid(centre + settings.spread * settings.logistic_scale * float(correlated[position]))
        base[name] = float(min(1.0, max(0.0, value)))

    digital = base["digital_affinity"]

    weights = {}

    for channel in settings.channels:
        factor = settings.channel_digital_factor.get(channel, 1.0)
        exponent = 2.0 * digital - 1.0
        weights[channel] = settings.channel_base.get(channel, 0.05) * (factor ** exponent)

    total = sum(weights.values()) or 1.0

    return Traits(
        base=base,
        channel_preferences={name: value / total for name, value in weights.items()},
    )


def event_shift(cause: str, ts: datetime) -> TraitShift | None:
    """
    Сдвиг характеристик после жизненного события.
    """

    settings = params_module.active().traits

    deltas = settings.drift_per_event.get(cause)

    if not deltas:
        return None

    return TraitShift(ts=ts, deltas=dict(deltas), cause=cause)


__all__ = ["TraitShift", "Traits", "draw_traits", "event_shift"]
