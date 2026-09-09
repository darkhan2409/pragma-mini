from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from src.tokenizer.dataset import TokenBatch

from .batching import BatchError
from .config import ModelConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Время у события описывается двумя числами в часах:
#
#   gap = ts события − ts предыдущего события
#   age = cutoff − ts события
#
# Gap один порядок не задаёт: у равных timestamps он ноль, и
# без второго признака недавняя активность не отличалась бы от
# давно законченной. Age эту разницу и даёт, а сам порядок
# несёт отдельное синусоидальное кодирование позиции.
#
# Обе величины сжимаются f(t) = 8·log1p(t / 8): часы и месяцы
# должны попадать в один разумный диапазон.
#
# Единицы измеряются честно: сначала вычитаются datetime64[us],
# и только потом длительность делится на микросекунды в часе.
# Обратный порядок терял бы разницу в микросекунды.
# ============================================================


HOUR_US = 3_600_000_000

TIME_SCALE = 8.0

SINUSOIDAL_BASE = 10_000.0


# ============================================================
# ЧАСЫ
# ============================================================


def hours_between(later: np.ndarray, earlier: np.ndarray) -> np.ndarray:
    """
    Разница двух datetime64 в часах.
    """

    delta = np.asarray(later) - np.asarray(earlier)

    return delta.astype("timedelta64[us]").astype(np.int64) / HOUR_US


def _example_starts(example_of_event: np.ndarray) -> np.ndarray:
    """
    Маска первых событий каждого примера.
    """

    starts = np.zeros(example_of_event.size, dtype=bool)

    if example_of_event.size == 0:
        return starts

    starts[0] = True
    starts[1:] = example_of_event[1:] != example_of_event[:-1]

    return starts


def gap_hours(batch: TokenBatch) -> np.ndarray:
    """
    Часы до предыдущего события ПОЛНОЙ истории.

    У первого события истории gap равен нулю: раньше него
    ничего не было. Равные timestamps тоже дают ноль, это
    законно и означает «в тот же момент».
    """

    ts = np.asarray(batch.ts)

    if ts.size == 0:
        return np.zeros(0, dtype=np.float64)

    example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)

    gaps = np.zeros(ts.size, dtype=np.float64)

    gaps[1:] = hours_between(ts[1:], ts[:-1])

    gaps[_example_starts(example_of_event)] = 0.0

    negative = np.flatnonzero(gaps < 0)

    if negative.size:
        position = int(negative[0])
        raise BatchError(
            f"событие {position} примера {int(example_of_event[position])} раньше предыдущего: "
            f"gap {gaps[position]:.6f} ч; история обязана быть упорядочена по (ts, seq)"
        )

    return gaps


def age_hours(batch: TokenBatch, cutoffs: np.ndarray) -> np.ndarray:
    """
    Часы от события до cutoff его примера.
    """

    ts = np.asarray(batch.ts)

    if ts.size == 0:
        return np.zeros(0, dtype=np.float64)

    example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)

    ages = hours_between(np.asarray(cutoffs)[example_of_event], ts)

    late = np.flatnonzero(ages <= 0)

    if late.size:
        position = int(late[0])
        raise BatchError(
            f"событие {position} примера {int(example_of_event[position])} не раньше cutoff: "
            f"age {ages[position]:.6f} ч; в историю входят только события строго до cutoff"
        )

    return ages


# ============================================================
# СЖАТИЕ И ПРОЕКЦИЯ
# ============================================================


def squash(hours: torch.Tensor, scale: float = TIME_SCALE) -> torch.Tensor:
    """
    f(t) = 8·log1p(t / 8): часы и месяцы в одном масштабе.
    """

    return scale * torch.log1p(hours / scale)


class TimeEncoding(nn.Module):
    """
    Два временных признака в вектор d_model.

    Без bias: нулевые gap и age обязаны давать нулевой вклад,
    иначе профиль и padding получили бы временное слагаемое.
    """

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.config = config

        self.linear = nn.Linear(2, config.d_model, bias=False)

    def forward(self, hours: torch.Tensor) -> torch.Tensor:

        if hours.ndim != 2 or hours.shape[1] != 2:
            raise BatchError(f"ожидалось [n, 2] с gap и age, получено {tuple(hours.shape)}")

        if hours.numel() and bool((hours < 0).any()):
            raise BatchError("отрицательный временной интервал: gap и age не могут быть меньше нуля")

        # log1p всегда в float32: под autocast bf16 у часов
        # осталось бы восемь бит мантиссы, и сутки не отличались
        # бы от суток с четвертью.
        return self.linear(squash(hours.float()))


# ============================================================
# ПОЗИЦИИ ИСТОРИИ
# ============================================================


def sinusoidal_positions(
    length: int,
    d_model: int,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Классическое синусоидальное кодирование позиции в истории.

    Позиция 0 это профиль, дальше события по порядку. Таблица
    позиций полей внутри события здесь не участвует: это разные
    оси, и путать их нельзя.
    """

    if length < 1:
        raise ValueError("длина последовательности должна быть положительной")

    if d_model % 2 != 0:
        raise ValueError("d_model должен быть чётным для синусоидального кодирования")

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)

    steps = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)

    frequency = torch.exp(-math.log(SINUSOIDAL_BASE) * steps / d_model)

    angles = positions * frequency

    encoding = torch.zeros(length, d_model, device=device, dtype=torch.float32)

    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)

    return encoding.to(dtype)
