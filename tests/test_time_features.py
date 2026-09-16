"""
Временные величины и признаки: возраст события до cutoff,
сжатие, календарь и простой клиента.

Единицы считаются честно: сначала вычитаются timestamps, потом
длительность переводится в часы.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from src.model.batching import BatchError
from src.model.config import ModelConfig
from src.model.time_features import (
    TIME_SCALE,
    age_hours,
    hours_between,
    squash_np,
)

from tests.helpers_data import synthetic_batch



@pytest.fixture(scope="module")
def config():
    return ModelConfig(vocab_size=64, d_model=8, n_heads=2, max_position_embeddings=16)


# ============================================================
# ЧАСЫ
# ============================================================


def test_one_hour_is_one():
    later = np.array([np.datetime64(datetime(2025, 1, 1, 12), "us")])
    earlier = np.array([np.datetime64(datetime(2025, 1, 1, 11), "us")])

    assert hours_between(later, earlier)[0] == 1.0


def test_equal_timestamps_give_zero():
    moment = np.array([np.datetime64(datetime(2025, 1, 1, 12), "us")])

    assert hours_between(moment, moment)[0] == 0.0


def test_subtraction_happens_before_the_division():
    """
    Микросекундная разница обязана остаться ненулевой: перевод
    во float до вычитания её бы потерял.
    """

    earlier = np.array([np.datetime64(datetime(2025, 1, 1, 12), "us")])
    later = earlier + np.timedelta64(1, "us")

    assert hours_between(later, earlier)[0] == pytest.approx(1 / 3_600_000_000)


# ============================================================
# GAP И AGE
# ============================================================


def test_age_counts_down_to_cutoff():
    batch, meta = synthetic_batch([[0, 2, 3, 8]], cutoff_hours=10)

    assert age_hours(batch, meta.cutoffs).tolist() == [10.0, 8.0, 7.0, 2.0]


def test_event_at_cutoff_is_rejected():
    batch, meta = synthetic_batch([[0, 5]], cutoff_hours=5)

    with pytest.raises(BatchError, match="не раньше cutoff"):
        age_hours(batch, meta.cutoffs)


def test_backwards_history_is_rejected():
    """
    Порядок ленты проверяется до всякой арифметики времени:
    возраст считается уже по проверенному batch.
    """

    from src.model.history_batching import validate_history_batch

    batch, meta = synthetic_batch([[0, 5, 2]], cutoff_hours=24, sort=False)

    with pytest.raises(BatchError, match="не упорядочена"):
        validate_history_batch(batch, meta)


# ============================================================
# СЖАТИЕ
# ============================================================


def test_squash_keeps_zero_at_zero():
    assert float(squash_np(0.0)) == 0.0


def test_squash_matches_the_formula():
    value = float(squash_np(TIME_SCALE))

    assert value == pytest.approx(TIME_SCALE * math.log(2.0), rel=1e-6)


def test_squash_is_monotonic():
    values = squash_np([0.0, 1.0, 10.0, 100.0, 1000.0])

    assert (values[1:] > values[:-1]).all()


def test_squash_compresses_the_long_tail():
    """
    Час и месяц обязаны попадать в один разумный диапазон.
    """

    hour = float(squash_np(1.0))
    month = float(squash_np(720.0))

    assert month / hour < 40
