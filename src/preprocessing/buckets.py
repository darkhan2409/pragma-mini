from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


# ============================================================
# ИДЕЯ
# ============================================================
#
# Quantile-bucket'ы для numeric-полей, обученные только на
# непустых train-значениях.
#
# Границы берутся методом inverted_cdf: каждая граница это
# наблюдённое значение, а не интерполяция между соседями.
# Одинаковые границы схлопываются, граница, равная максимуму,
# отбрасывается. Тогда при правиле "bucket = число границ,
# строго меньших значения" каждый bucket на train непуст.
#
# Правило включения: bucket i покрывает (e[i-1], e[i]],
# значение на границе уходит в нижний bucket. Значения вне
# train-диапазона естественно попадают в крайние bucket'ы.
# Пропуск остаётся пропуском: bucket = null.
# ============================================================


STATUS_FITTED = "fitted"
STATUS_CONSTANT = "constant"
STATUS_NO_FIT_DATA = "no_fit_data"

BUCKET_DTYPE = pa.int16()

RULE = {
    "method": "quantile_inverted_cdf_unique_drop_max",
    "inclusion": "right_closed",
    "description": (
        "bucket 0 = (-inf, edges[0]]; bucket i = (edges[i-1], edges[i]]; "
        "last bucket = (edges[-1], +inf); bucket = count(edges < value); "
        "values outside the train range fall into the extreme buckets; "
        "missing value -> null bucket; constant field -> single bucket 0; "
        "no_fit_data -> all buckets null"
    ),
}


@dataclass(frozen=True)
class BucketSpec:
    namespace: str
    field: str
    status: str
    requested_buckets: int
    actual_bucket_count: int
    edges: tuple[float, ...]
    train_min: float | None
    train_max: float | None
    n_fit_values: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.namespace, self.field)

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "requested_buckets": self.requested_buckets,
            "actual_bucket_count": self.actual_bucket_count,
            "edges": list(self.edges) if self.status != STATUS_NO_FIT_DATA else None,
            "train_min": self.train_min,
            "train_max": self.train_max,
            "n_fit_values": self.n_fit_values,
        }

    @staticmethod
    def from_json(namespace: str, field: str, data: dict) -> "BucketSpec":
        return BucketSpec(
            namespace=namespace,
            field=field,
            status=data["status"],
            requested_buckets=int(data["requested_buckets"]),
            actual_bucket_count=int(data["actual_bucket_count"]),
            edges=tuple(data["edges"] or ()),
            train_min=data["train_min"],
            train_max=data["train_max"],
            n_fit_values=int(data["n_fit_values"]),
        )


def _plain(value: np.generic | float, integer: bool) -> float | int:
    return int(value) if integer else float(value)


def fit_edges(namespace: str, field: str, values: np.ndarray, requested: int) -> BucketSpec:
    """
    values: непустые train-значения (numpy, без NaN).
    """

    if requested < 1:
        raise ValueError("число bucket'ов должно быть положительным")

    values = np.asarray(values)

    if values.size == 0:
        return BucketSpec(namespace, field, STATUS_NO_FIT_DATA, requested, 0, (), None, None, 0)

    if np.issubdtype(values.dtype, np.floating) and np.isnan(values).any():
        raise ValueError("NaN среди fit-значений: пропуски отфильтровываются раньше")

    integer = np.issubdtype(values.dtype, np.integer)

    low = _plain(values.min(), integer)
    high = _plain(values.max(), integer)

    if low == high:
        return BucketSpec(namespace, field, STATUS_CONSTANT, requested, 1, (), low, high, int(values.size))

    probabilities = np.linspace(0.0, 1.0, requested + 1)[1:-1]

    raw_edges = np.quantile(values, probabilities, method="inverted_cdf")

    edges = np.unique(raw_edges)

    # Граница, равная максимуму, дала бы пустой верхний bucket.
    edges = edges[edges < values.max()]

    plain_edges = tuple(_plain(edge, integer) for edge in edges)

    return BucketSpec(
        namespace=namespace,
        field=field,
        status=STATUS_FITTED,
        requested_buckets=requested,
        actual_bucket_count=len(plain_edges) + 1,
        edges=plain_edges,
        train_min=low,
        train_max=high,
        n_fit_values=int(values.size),
    )


def bucketize_numpy(values: np.ndarray, spec: BucketSpec) -> np.ndarray:
    """
    Индексы bucket'ов для плотного массива без пропусков.
    """

    if spec.status == STATUS_NO_FIT_DATA:
        raise ValueError("у поля нет обученных границ")

    if not spec.edges:
        return np.zeros(len(values), dtype=np.int16)

    edges = np.asarray(spec.edges, dtype=np.float64)

    return np.searchsorted(edges, np.asarray(values, dtype=np.float64), side="left").astype(np.int16)


def apply_buckets(values: pa.Array | pa.ChunkedArray, spec: BucketSpec) -> pa.Array:
    """
    Колонка bucket'ов той же длины: null там, где значение null
    или у поля нет fit-данных.
    """

    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()

    length = len(values)

    if spec.status == STATUS_NO_FIT_DATA or length == 0:
        return pa.nulls(length, BUCKET_DTYPE)

    null_mask = pc.is_null(values).to_numpy(zero_copy_only=False)

    filled = pc.fill_null(values, 0).to_numpy(zero_copy_only=False)

    buckets = bucketize_numpy(filled, spec)

    return pa.array(buckets, type=BUCKET_DTYPE, mask=null_mask)


def bucket_shares(buckets: np.ndarray, actual_bucket_count: int) -> np.ndarray:

    if actual_bucket_count == 0 or len(buckets) == 0:
        return np.zeros(actual_bucket_count, dtype=np.float64)

    counts = np.bincount(buckets, minlength=actual_bucket_count).astype(np.float64)

    return counts / counts.sum()


def edges_artifact(specs: dict[tuple[str, str], BucketSpec], default_buckets: int, schema_version: int) -> dict:

    fields: dict[str, dict[str, dict]] = {}

    for (namespace, field), spec in sorted(specs.items()):
        fields.setdefault(namespace, {})[field] = spec.to_json()

    return {
        "schema_version": schema_version,
        "rule": RULE,
        "default_buckets": default_buckets,
        "bucket_dtype": str(BUCKET_DTYPE),
        "fields": fields,
    }


def load_specs(artifact: dict) -> dict[tuple[str, str], BucketSpec]:
    return {
        (namespace, field): BucketSpec.from_json(namespace, field, data)
        for namespace, fields in artifact["fields"].items()
        for field, data in fields.items()
    }
