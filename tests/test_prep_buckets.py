"""
Bucketization: границы обучаются на train и больше не меняются.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from src.preprocessing.buckets import (
    BUCKET_DTYPE,
    STATUS_CONSTANT,
    STATUS_FITTED,
    STATUS_NO_FIT_DATA,
    BucketSpec,
    apply_buckets,
    bucketize_numpy,
    edges_artifact,
    fit_edges,
    load_specs,
)


def fit(values, buckets: int = 16) -> BucketSpec:
    return fit_edges("test", "field", np.asarray(values), buckets)


# ============================================================
# ГРАНИЦЫ
# ============================================================


def test_uniform_values_give_requested_buckets():
    spec = fit(np.arange(1000))

    assert spec.status == STATUS_FITTED
    assert spec.actual_bucket_count == 16
    assert len(spec.edges) == 15
    assert spec.train_min == 0
    assert spec.train_max == 999


def test_edges_are_observed_values():
    """
    inverted_cdf: граница это реально встреченное значение,
    а не интерполяция между соседями.
    """

    values = np.array([1, 2, 3, 100, 200, 300])

    spec = fit(values, buckets=3)

    assert set(spec.edges) <= set(values.tolist())


def test_ties_collapse_buckets():
    """
    Дискретное поле со срока: 16 запрошенных схлопываются.
    """

    values = np.repeat([6, 12, 24, 36, 60], 40)

    spec = fit(values)

    assert spec.requested_buckets == 16
    assert spec.actual_bucket_count < 16
    assert len(spec.edges) == len(set(spec.edges))


def test_constant_field_gets_single_bucket():
    spec = fit(np.full(500, 7))

    assert spec.status == STATUS_CONSTANT
    assert spec.actual_bucket_count == 1
    assert spec.edges == ()
    assert bucketize_numpy(np.array([7, 1, 99]), spec).tolist() == [0, 0, 0]


def test_empty_field_has_no_fit_data():
    spec = fit(np.array([], dtype=np.float64))

    assert spec.status == STATUS_NO_FIT_DATA
    assert spec.edges == ()
    assert spec.train_min is None
    assert spec.n_fit_values == 0


def test_nan_among_fit_values_is_rejected():
    with pytest.raises(ValueError):
        fit(np.array([1.0, np.nan, 3.0]))


# ============================================================
# РАСКЛАДКА ЗНАЧЕНИЙ
# ============================================================


def test_every_train_bucket_is_non_empty():
    """
    Ключевой инвариант: границы не создают пустых корзин.
    """

    rng = np.random.default_rng(0)

    for values in (
        rng.integers(0, 5, 500),
        rng.lognormal(8, 1.5, 5000).astype(np.int64),
        np.repeat([1, 1, 1, 2, 3], 100),
        np.arange(37),
    ):
        spec = fit(values)

        if spec.status != STATUS_FITTED:
            continue

        counts = np.bincount(bucketize_numpy(values, spec), minlength=spec.actual_bucket_count)

        assert (counts > 0).all(), (spec.edges, counts)


def test_edge_value_falls_into_lower_bucket():
    """
    Правило включения (e[i-1], e[i]]: значение на границе
    остаётся в нижней корзине.
    """

    spec = fit(np.arange(100), buckets=4)

    edge = spec.edges[0]

    assert bucketize_numpy(np.array([edge]), spec)[0] == 0
    assert bucketize_numpy(np.array([edge + 1]), spec)[0] == 1


def test_values_outside_train_range_go_to_extreme_buckets():
    spec = fit(np.arange(100, 200))

    buckets = bucketize_numpy(np.array([-10 ** 9, 10 ** 9]), spec)

    assert buckets[0] == 0
    assert buckets[1] == spec.actual_bucket_count - 1


def test_missing_stays_missing():
    spec = fit(np.arange(100))

    column = pa.array([5, None, 90], pa.int64())

    buckets = apply_buckets(column, spec)

    assert buckets.type == BUCKET_DTYPE
    assert buckets.to_pylist()[1] is None
    assert buckets.to_pylist()[0] is not None


def test_no_fit_data_gives_all_null_buckets():
    spec = fit(np.array([], dtype=np.float64))

    buckets = apply_buckets(pa.array([1.0, 2.0], pa.float64()), spec)

    assert buckets.null_count == 2


def test_apply_buckets_keeps_length():
    spec = fit(np.arange(50))

    for length in (0, 1, 7):
        column = pa.array(list(range(length)), pa.int64())
        assert len(apply_buckets(column, spec)) == length


# ============================================================
# ARTIFACT
# ============================================================


def test_json_round_trip():
    spec = fit(np.arange(1000))

    artifact = edges_artifact({spec.key: spec}, 16, 1)

    restored = load_specs(artifact)[spec.key]

    assert restored == spec


def test_artifact_records_rule_and_status():
    specs = {
        ("a", "x"): fit(np.arange(100)),
        ("a", "y"): fit(np.full(10, 3)),
        ("a", "z"): fit(np.array([], dtype=np.float64)),
    }

    for key, spec in list(specs.items()):
        specs[key] = BucketSpec(key[0], key[1], spec.status, spec.requested_buckets,
                                spec.actual_bucket_count, spec.edges, spec.train_min,
                                spec.train_max, spec.n_fit_values)

    artifact = edges_artifact(specs, 16, 1)

    assert artifact["rule"]["inclusion"] == "right_closed"
    assert artifact["fields"]["a"]["z"]["edges"] is None
    assert artifact["fields"]["a"]["y"]["actual_bucket_count"] == 1
