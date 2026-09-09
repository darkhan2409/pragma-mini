"""
Время события: gap до предыдущего и age до cutoff.

Единицы считаются честно: сначала вычитаются timestamps, потом
длительность переводится в часы.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from src.model.batching import BatchError
from src.model.config import ModelConfig
from src.model.time_encoding import (
    TIME_SCALE,
    TimeEncoding,
    age_hours,
    gap_hours,
    hours_between,
    sinusoidal_positions,
    squash,
)

from tests.test_history_batching import synthetic_batch


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


def test_gap_follows_the_full_history():
    batch, meta = synthetic_batch([[0, 2, 3, 8]], cutoff_hours=24)

    assert gap_hours(batch).tolist() == [0.0, 2.0, 1.0, 5.0]


def test_first_event_of_history_has_zero_gap():
    batch, meta = synthetic_batch([[0, 2], [0, 4]], cutoff_hours=24)

    gaps = gap_hours(batch)

    assert gaps.tolist() == [0.0, 2.0, 0.0, 4.0]


def test_equal_timestamps_give_zero_gap():
    batch, meta = synthetic_batch([[0, 1, 1, 3]], cutoff_hours=24)

    assert gap_hours(batch).tolist() == [0.0, 1.0, 0.0, 2.0]


def test_age_counts_down_to_cutoff():
    batch, meta = synthetic_batch([[0, 2, 3, 8]], cutoff_hours=10)

    assert age_hours(batch, meta.cutoffs).tolist() == [10.0, 8.0, 7.0, 2.0]


def test_event_at_cutoff_is_rejected():
    batch, meta = synthetic_batch([[0, 5]], cutoff_hours=5)

    with pytest.raises(BatchError, match="не раньше cutoff"):
        age_hours(batch, meta.cutoffs)


def test_backwards_history_is_rejected():
    batch, meta = synthetic_batch([[0, 5, 2]], cutoff_hours=24, sort=False)

    with pytest.raises(BatchError, match="раньше предыдущего"):
        gap_hours(batch)


# ============================================================
# СЖАТИЕ
# ============================================================


def test_squash_keeps_zero_at_zero():
    assert squash(torch.zeros(1)).item() == 0.0


def test_squash_matches_the_formula():
    value = squash(torch.tensor([TIME_SCALE])).item()

    assert value == pytest.approx(TIME_SCALE * math.log(2.0), rel=1e-6)


def test_squash_is_monotonic():
    hours = torch.tensor([0.0, 1.0, 10.0, 100.0, 1000.0])

    values = squash(hours)

    assert (values[1:] > values[:-1]).all()


def test_squash_compresses_the_long_tail():
    """
    Час и месяц обязаны попадать в один разумный диапазон.
    """

    hour = squash(torch.tensor([1.0])).item()
    month = squash(torch.tensor([720.0])).item()

    assert month / hour < 40


# ============================================================
# ПРОЕКЦИЯ
# ============================================================


def test_zero_time_gives_zero_vector(config):
    time = TimeEncoding(config)

    out = time(torch.zeros(3, 2))

    assert out.shape == (3, config.d_model)
    assert torch.count_nonzero(out) == 0


def test_projection_has_no_bias(config):
    assert TimeEncoding(config).linear.bias is None


def test_wrong_shape_is_rejected(config):
    with pytest.raises(BatchError, match=r"\[n, 2\]"):
        TimeEncoding(config)(torch.zeros(3, 3))


def test_negative_interval_is_rejected(config):
    with pytest.raises(BatchError, match="отрицательный"):
        TimeEncoding(config)(torch.tensor([[-1.0, 2.0]]))


def test_different_times_give_different_vectors(config):
    time = TimeEncoding(config).eval()

    with torch.no_grad():
        a = time(torch.tensor([[1.0, 10.0]]))
        b = time(torch.tensor([[1.0, 500.0]]))

    assert not torch.allclose(a, b)


# ============================================================
# ПОЗИЦИИ ИСТОРИИ
# ============================================================


def test_positions_have_the_expected_shape():
    encoding = sinusoidal_positions(5, 8)

    assert encoding.shape == (5, 8)


def test_position_zero_is_sin_zero_cos_zero():
    encoding = sinusoidal_positions(3, 8)

    assert encoding[0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_positions_are_distinguishable():
    encoding = sinusoidal_positions(64, 16)

    similarity = encoding @ encoding.T

    assert not torch.allclose(encoding[1], encoding[2])
    assert similarity.diagonal().min() > 0


def test_positions_do_not_depend_on_the_field_table(config):
    """
    Позиция в истории и позиция поля внутри события это разные
    оси: таблица полей ограничена, история нет.
    """

    long = sinusoidal_positions(config.max_position_embeddings * 4, config.d_model)

    assert long.shape[0] > config.max_position_embeddings


def test_odd_width_is_rejected():
    with pytest.raises(ValueError, match="чётным"):
        sinusoidal_positions(4, 7)


def test_dtype_and_device_are_respected():
    encoding = sinusoidal_positions(4, 8, dtype=torch.float64)

    assert encoding.dtype == torch.float64
