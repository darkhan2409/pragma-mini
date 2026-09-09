"""
Статистики: пропуски отдельно, метрики по непустым значениям.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pyarrow as pa
import pytest

from src.preprocessing.artifacts import dumps_json, json_ready
from src.preprocessing.buckets import fit_edges
from src.preprocessing.config import REGISTRY, Settings
from src.preprocessing.stats import (
    FLAG_HIGH_MISSING,
    FLAG_LOW_ENTROPY,
    FLAG_NEAR_CONSTANT,
    FieldAccumulator,
    entropy_bits,
    field_stats_entry,
    rare_share,
    sorted_distribution,
    unigram_entry,
    value_counts_table,
    value_key,
)


SETTINGS = Settings()

MCC = REGISTRY[("transaction", "mcc")]
AMOUNT = REGISTRY[("transaction", "amount")]
ONLINE = REGISTRY[("transaction", "is_online")]


def accumulate(spec, values, arrow_type=None) -> FieldAccumulator:
    acc = FieldAccumulator(spec)
    acc.update(pa.array(values, arrow_type or spec.arrow_type))
    return acc


# ============================================================
# МЕТРИКИ
# ============================================================


def test_entropy_of_uniform_four_values_is_two_bits():
    assert entropy_bits({"a": 10, "b": 10, "c": 10, "d": 10}) == pytest.approx(2.0)


def test_entropy_of_constant_is_zero():
    assert entropy_bits({"a": 42}) == 0.0


def test_entropy_of_empty_is_zero():
    assert entropy_bits({}) == 0.0


def test_rare_share_uses_threshold():
    counts = {"common": 100, "rare_a": 19, "rare_b": 1}

    share, values = rare_share(counts, 20)

    assert values == 2
    assert share == pytest.approx(20 / 120)


def test_rare_share_of_empty():
    assert rare_share({}, 20) == (0.0, 0)


def test_distribution_sorted_by_count_then_value():
    rows = sorted_distribution({"b": 5, "a": 5, "c": 9})

    assert [row[0] for row in rows] == ["c", "a", "b"]
    assert sum(row[2] for row in rows) == pytest.approx(1.0)


def test_value_key_normalises_types():
    assert value_key(True) == "true"
    assert value_key(np.bool_(False)) == "false"
    assert value_key(5.0) == "5"
    assert value_key(5.5) == "5.5"


# ============================================================
# АККУМУЛЯТОР
# ============================================================


def test_missing_counted_among_records_of_its_type():
    acc = accumulate(MCC, ["5411", None, "5812", None])

    assert acc.n_total == 4
    assert acc.n_missing == 2
    assert acc.n_valid == 2


def test_accumulator_merges_batches():
    acc = FieldAccumulator(MCC)

    acc.update(pa.array(["a", "b"], pa.string()))
    acc.update(pa.array(["a", None], pa.string()))

    assert acc.n_total == 4
    assert acc.value_counts() == {"a": 2, "b": 1}


def test_numeric_accumulator_keeps_values():
    acc = accumulate(AMOUNT, [10, 20, None, 30])

    assert acc.numeric_values().tolist() == [10, 20, 30]
    assert acc.n_missing == 1


def test_empty_batch_changes_nothing():
    acc = FieldAccumulator(MCC)
    acc.update(pa.array([], pa.string()))

    assert acc.n_total == 0


# ============================================================
# ЗАПИСЬ СТАТИСТИК
# ============================================================


def test_categorical_entry_shape():
    acc = accumulate(MCC, ["a"] * 30 + ["b"] * 10 + [None] * 10)

    entry = field_stats_entry(MCC, acc, None, SETTINGS)

    assert entry["n_total"] == 50
    assert entry["n_missing"] == 10
    assert entry["n_valid"] == 40
    assert entry["missing_rate"] == pytest.approx(0.2)
    assert entry["cardinality"] == 2
    assert entry["top1_value"] == "a"
    assert entry["top1_frequency"] == pytest.approx(0.75)
    assert entry["distribution"][0][0] == "a"


def test_numeric_entry_has_buckets_and_sidecar():
    values = list(range(1000))

    acc = accumulate(AMOUNT, values)

    spec = fit_edges("transaction", "amount", np.asarray(values), 16)

    entry = field_stats_entry(AMOUNT, acc, spec, SETTINGS, "value_counts/transaction__amount.parquet")

    assert entry["buckets"]["actual_bucket_count"] == 16
    assert entry["buckets"]["empty_buckets"] == 0
    assert sum(row[1] for row in entry["buckets"]["distribution"]) == 1000
    assert entry["numeric_summary"]["min"] == 0
    assert entry["distribution_file"].endswith(".parquet")
    assert "distribution" not in entry


def test_flags_fire_on_thresholds():
    high_missing = accumulate(MCC, ["a"] + [None] * 99)
    assert FLAG_HIGH_MISSING in field_stats_entry(MCC, high_missing, None, SETTINGS)["flags"]

    near_constant = accumulate(MCC, ["a"] * 999 + ["b"])
    flags = field_stats_entry(MCC, near_constant, None, SETTINGS)["flags"]
    assert FLAG_NEAR_CONSTANT in flags
    assert FLAG_LOW_ENTROPY in flags


def test_no_flags_on_healthy_field():
    acc = accumulate(MCC, [f"v{i % 20}" for i in range(400)])

    assert field_stats_entry(MCC, acc, None, SETTINGS)["flags"] == []


def test_boolean_distribution_uses_words():
    acc = accumulate(ONLINE, [True, True, False, None])

    entry = field_stats_entry(ONLINE, acc, None, SETTINGS)

    assert {row[0] for row in entry["distribution"]} == {"true", "false"}


# ============================================================
# BASELINE
# ============================================================


def test_unigram_mode_and_distribution():
    acc = accumulate(MCC, ["a"] * 30 + ["b"] * 10 + [None] * 10)

    entry = unigram_entry(MCC, acc, None)

    assert entry["mode"] == "a"
    assert entry["mode_probability"] == pytest.approx(0.75)
    assert entry["missing_rate"] == pytest.approx(0.2)
    assert sum(row[1] for row in entry["distribution"]) == pytest.approx(1.0)


def test_unigram_numeric_uses_buckets():
    values = list(range(100))

    acc = accumulate(AMOUNT, values)
    spec = fit_edges("transaction", "amount", np.asarray(values), 4)

    entry = unigram_entry(AMOUNT, acc, spec)

    assert entry["encoding"] == "bucket"
    assert isinstance(entry["mode"], int)
    assert len(entry["distribution"]) == spec.actual_bucket_count


def test_unigram_of_empty_field():
    acc = FieldAccumulator(MCC)

    entry = unigram_entry(MCC, acc, None)

    assert entry["mode"] is None
    assert entry["distribution"] == []


def test_value_counts_table_sorted():
    acc = accumulate(AMOUNT, [5, 5, 5, 9, 9, 1])

    table = value_counts_table(acc)

    assert table.column("value").to_pylist() == [5, 9, 1]
    assert table.column("count").to_pylist() == [3, 2, 1]
    assert table.schema.field("value").type == AMOUNT.arrow_type


# ============================================================
# СЕРИАЛИЗАЦИЯ
# ============================================================


def test_json_ready_handles_numpy_and_datetime():
    value = json_ready(
        {
            "int": np.int64(5),
            "float": np.float64(0.5),
            "bool": np.bool_(True),
            "ts": datetime(2026, 6, 1),
            "array": np.array([1, 2]),
            "set": {"b", "a"},
        }
    )

    assert value["int"] == 5 and isinstance(value["int"], int)
    assert value["bool"] is True
    assert value["ts"] == "2026-06-01T00:00:00"
    assert value["array"] == [1, 2]
    assert value["set"] == ["a", "b"]


def test_json_ready_rejects_nan():
    with pytest.raises(ValueError):
        json_ready({"x": float("nan")})


def test_json_is_sorted_and_newline_terminated():
    text = dumps_json({"b": 1, "a": 2})

    assert text.index('"a"') < text.index('"b"')
    assert text.endswith("\n")
