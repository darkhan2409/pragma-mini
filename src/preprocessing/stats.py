from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .buckets import STATUS_NO_FIT_DATA, BucketSpec, bucket_shares, bucketize_numpy
from .config import KIND_BOOLEAN, KIND_CATEGORICAL, KIND_METADATA, KIND_NUMERIC, FieldSpec, Settings


# ============================================================
# ИДЕЯ
# ============================================================
#
# Статистики считаются потоково: аккумулятор поля получает
# батчи значений и хранит либо счётчики (категории, булевы),
# либо сами непустые значения (numeric, для quantile-границ).
#
# Пропуски считаются среди записей своего namespace. Остальные
# метрики считаются по непустым значениям. Распределения
# хранятся массивами [значение, count, доля] с сортировкой
# (count desc, значение asc): объект с sort_keys упорядочил бы
# "10" раньше "9".
# ============================================================


FLAG_HIGH_MISSING = "high_missing"
FLAG_NEAR_CONSTANT = "near_constant"
FLAG_LOW_ENTROPY = "low_entropy"
FLAG_UNEVEN_BUCKETS = "uneven_buckets"

NUMERIC_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


# ============================================================
# АККУМУЛЯТОР
# ============================================================


def value_key(value: Any) -> str:
    """
    Ключ значения в распределении: строка, устойчивая к типу.
    """

    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)

    return str(value)


class FieldAccumulator:

    def __init__(self, spec: FieldSpec):
        self.spec = spec
        self.n_total = 0
        self.n_missing = 0
        self.counts: Counter[str] = Counter()
        self.chunks: list[np.ndarray] = []

    def update(self, values: pa.Array | pa.ChunkedArray) -> None:

        length = len(values)

        if length == 0:
            return

        self.n_total += length
        self.n_missing += values.null_count

        present = pc.drop_null(values)

        if len(present) == 0:
            return

        if self.spec.kind == KIND_NUMERIC:
            self.chunks.append(np.asarray(present.to_numpy(zero_copy_only=False)))
            return

        counted = pc.value_counts(present)

        for item in counted.to_pylist():
            self.counts[value_key(item["values"])] += int(item["counts"])

    # --------------------------------------------------------

    @property
    def n_valid(self) -> int:
        return self.n_total - self.n_missing

    def numeric_values(self) -> np.ndarray:

        if not self.chunks:
            return np.array([], dtype=np.float64)

        return np.concatenate(self.chunks)

    def value_counts(self) -> Counter[str]:
        """
        Счётчики по значениям; для numeric строятся по факту.
        """

        if self.spec.kind != KIND_NUMERIC:
            return self.counts

        values = self.numeric_values()

        if values.size == 0:
            return Counter()

        uniques, counts = np.unique(values, return_counts=True)

        return Counter({value_key(value): int(count) for value, count in zip(uniques, counts)})


# ============================================================
# МЕТРИКИ
# ============================================================


def entropy_bits(counts: dict[str, int]) -> float:

    total = sum(counts.values())

    if total == 0:
        return 0.0

    return float(-sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0))


def rare_share(counts: dict[str, int], threshold: int) -> tuple[float, int]:
    """
    Доля непустых наблюдений со значениями, встречающимися реже
    порога, и число таких значений.
    """

    total = sum(counts.values())

    if total == 0:
        return 0.0, 0

    rare = {value: c for value, c in counts.items() if c < threshold}

    return sum(rare.values()) / total, len(rare)


def sorted_distribution(counts: dict[str, int]) -> list[list[Any]]:

    total = sum(counts.values())

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    return [[value, int(c), c / total] for value, c in ordered]


def numeric_summary(values: np.ndarray) -> dict[str, Any] | None:

    if values.size == 0:
        return None

    quantiles = np.quantile(values, NUMERIC_QUANTILES, method="inverted_cdf")

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "quantiles": [[q, float(v)] for q, v in zip(NUMERIC_QUANTILES, quantiles)],
    }


def bucket_section(values: np.ndarray, spec: BucketSpec, settings: Settings) -> dict[str, Any]:

    if spec.status == STATUS_NO_FIT_DATA:
        return {"status": spec.status, "actual_bucket_count": 0, "distribution": [], "flags": []}

    buckets = bucketize_numpy(values, spec) if values.size else np.array([], dtype=np.int16)

    shares = bucket_shares(buckets, spec.actual_bucket_count)

    counts = np.bincount(buckets, minlength=spec.actual_bucket_count) if values.size else np.zeros(spec.actual_bucket_count, dtype=np.int64)

    expected = 1.0 / spec.actual_bucket_count

    max_share = float(shares.max()) if shares.size else 0.0

    unevenness = max_share / expected if expected > 0 else 0.0

    flags = [FLAG_UNEVEN_BUCKETS] if unevenness > settings.uneven_buckets else []

    return {
        "status": spec.status,
        "requested_buckets": spec.requested_buckets,
        "actual_bucket_count": spec.actual_bucket_count,
        "collapsed_buckets": spec.requested_buckets - spec.actual_bucket_count,
        "distribution": [[int(i), int(c), float(s)] for i, (c, s) in enumerate(zip(counts, shares))],
        "expected_share": expected,
        "max_bucket_share": max_share,
        "unevenness": unevenness,
        "empty_buckets": int((counts == 0).sum()),
        "flags": flags,
    }


# ============================================================
# ЗАПИСИ ARTIFACTS
# ============================================================


def metadata_entry(spec: FieldSpec) -> dict[str, Any]:
    return {
        "kind": KIND_METADATA,
        "role": spec.role,
        "arrow_type": str(spec.arrow_type),
        "predictable": False,
        "excluded": True,
        "note": spec.note,
    }


def field_stats_entry(
    spec: FieldSpec,
    acc: FieldAccumulator,
    bucket: BucketSpec | None,
    settings: Settings,
    distribution_file: str | None = None,
) -> dict[str, Any]:

    counts = acc.value_counts()

    n_valid = acc.n_valid

    missing_rate = acc.n_missing / acc.n_total if acc.n_total else 0.0

    top = max(counts.items(), key=lambda item: (item[1], item[0]), default=None)

    top1_frequency = top[1] / n_valid if top and n_valid else 0.0

    entropy = entropy_bits(counts)

    share_rare, n_rare = rare_share(counts, settings.rare_count_threshold)

    flags: list[str] = []

    if missing_rate > settings.high_missing:
        flags.append(FLAG_HIGH_MISSING)

    if n_valid and top1_frequency > settings.near_constant:
        flags.append(FLAG_NEAR_CONSTANT)

    if n_valid and entropy < settings.low_entropy_bits:
        flags.append(FLAG_LOW_ENTROPY)

    entry: dict[str, Any] = {
        "kind": spec.kind,
        "role": spec.role,
        "arrow_type": str(spec.arrow_type),
        "predictable": spec.predictable,
        "n_total": acc.n_total,
        "n_missing": acc.n_missing,
        "n_valid": n_valid,
        "missing_rate": missing_rate,
        "cardinality": len(counts),
        "top1_value": top[0] if top else None,
        "top1_frequency": top1_frequency,
        "entropy_bits": entropy,
        "rare_share": share_rare,
        "rare_values": n_rare,
        "rare_count_threshold": settings.rare_count_threshold,
        "flags": flags,
    }

    if spec.kind in (KIND_CATEGORICAL, KIND_BOOLEAN):
        entry["distribution"] = sorted_distribution(counts)

    if spec.kind == KIND_NUMERIC:
        values = acc.numeric_values()
        entry["numeric_summary"] = numeric_summary(values)
        entry["buckets"] = bucket_section(values, bucket, settings) if bucket else None
        entry["distribution_file"] = distribution_file

    return entry


def unigram_entry(spec: FieldSpec, acc: FieldAccumulator, bucket: BucketSpec | None) -> dict[str, Any]:
    """
    Baseline «всегда предсказывай самое частое»: мода, её
    вероятность среди непустых и полное unigram-распределение.
    Numeric кодируется bucket'ами.
    """

    n_valid = acc.n_valid

    missing_rate = acc.n_missing / acc.n_total if acc.n_total else 0.0

    entry: dict[str, Any] = {
        "kind": spec.kind,
        "n_total": acc.n_total,
        "n_valid": n_valid,
        "missing_rate": missing_rate,
    }

    if spec.kind == KIND_NUMERIC:

        entry["encoding"] = "bucket"

        if bucket is None or bucket.status == STATUS_NO_FIT_DATA or n_valid == 0:
            entry.update({"actual_bucket_count": 0, "mode": None, "mode_probability": 0.0, "distribution": []})
            return entry

        values = acc.numeric_values()
        buckets = bucketize_numpy(values, bucket)
        counts = np.bincount(buckets, minlength=bucket.actual_bucket_count)
        shares = counts / counts.sum()

        mode = int(np.argmax(counts))

        entry.update(
            {
                "actual_bucket_count": bucket.actual_bucket_count,
                "mode": mode,
                "mode_probability": float(shares[mode]),
                "distribution": [[int(i), float(s)] for i, s in enumerate(shares)],
            }
        )
        return entry

    counts = acc.value_counts()

    entry["encoding"] = "value"

    if not counts:
        entry.update({"mode": None, "mode_probability": 0.0, "distribution": []})
        return entry

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    entry.update(
        {
            "mode": ordered[0][0],
            "mode_probability": ordered[0][1] / n_valid,
            "distribution": [[value, c / n_valid] for value, c in ordered],
        }
    )

    return entry


def value_counts_table(acc: FieldAccumulator) -> pa.Table:
    """
    Полное распределение numeric-поля для sidecar-parquet:
    (value, count), сортировка count desc, value asc.
    """

    values = acc.numeric_values()

    arrow_type = acc.spec.arrow_type

    if values.size == 0:
        return pa.table({"value": pa.array([], arrow_type), "count": pa.array([], pa.int64())})

    uniques, counts = np.unique(values, return_counts=True)

    order = np.lexsort((uniques, -counts))

    return pa.table(
        {
            "value": pa.array(uniques[order]).cast(arrow_type),
            "count": pa.array(counts[order].astype(np.int64)),
        }
    )
