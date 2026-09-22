from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocessing.artifacts import read_json

from .fit import TrainCorpus
from .scan import FitStatistics
from .schema import SemanticSchema
from .settings import (
    BUCKETS_FILE,
    METHOD_FIXED,
    METHOD_QUANTILE,
    METHOD_UNFITTED,
    NEGATIVE_INVALID,
    ZERO_SEPARATE,
    TokenizerConfig,
    tokenizer_path,
)
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Бакетизация заменяет точные числа диапазонами и объясняет каждый
# из них.
#
# Границы считает train и только train. Там, где наблюдений
# слишком мало, берётся заранее объявленная шкала, а не «лучшие»
# границы по маленькой выборке и тем более не границы по
# validation или test.
#
# Интервал это [lower, upper): значение на границе принадлежит
# следующему диапазону. Последний диапазон открыт сверху,
# первый покрывает всё, что ниже наблюдавшегося на train.
#
# Ноль, пропуск, невозможное значение и число вне шкалы это
# четыре разные вещи, и одинаково они не кодируются.
#
# Метка диапазона несёт ключ, границы и единицу, поэтому чужие
# B7 склеить по номеру нельзя: transaction_amount[200000,inf)KZT
# ни с чем не спутаешь.
# ============================================================


SOURCE_TRAIN = "train_quantiles"
SOURCE_CONFIG = "config_fixed"
SOURCE_FALLBACK = "config_fallback"
SOURCE_NONE = "none"

# Исход поиска диапазона.
FOUND_BUCKET = "bucket"
FOUND_INVALID = "invalid"
FOUND_UNKNOWN = "unknown"


class BucketsError(ValueError):
    """
    Числовые границы построить нельзя.
    """


def format_number(value: float) -> str:
    """
    Запись числа в метке: целое остаётся целым, дробное
    записывается так, чтобы читаться обратно.
    """

    if value == int(value) and abs(value) < 1e15:
        return str(int(value))

    return repr(value)


@dataclass(frozen=True)
class Bucket:
    index: int
    label: str
    lower: float | None
    upper: float | None
    zero: bool = False

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "lower": self.lower,
            "upper": self.upper,
            "zero": self.zero,
        }


def _label(key: str, unit: str | None, lower: float | None, upper: float | None, zero: bool) -> str:

    suffix = unit or ""

    if zero:
        return f"{key}=0{suffix}"

    left = "(-inf" if lower is None else f"[{format_number(lower)}"
    right = "inf)" if upper is None else f"{format_number(upper)})"

    return f"{key}{left},{right}{suffix}"


def build_buckets_list(key: str, unit: str | None, boundaries: tuple[float, ...],
                       zero_policy: str) -> tuple[Bucket, ...]:
    """
    Диапазоны ключа по его границам.
    """

    buckets: list[Bucket] = []

    if zero_policy == ZERO_SEPARATE:
        buckets.append(Bucket(0, _label(key, unit, 0.0, 0.0, True), 0.0, 0.0, True))

    edges: list[float | None] = [None, *boundaries, None]

    for position in range(len(edges) - 1):
        lower, upper = edges[position], edges[position + 1]
        buckets.append(
            Bucket(len(buckets), _label(key, unit, lower, upper, False), lower, upper, False)
        )

    return tuple(buckets)


@dataclass(frozen=True)
class FittedEncoder:
    """
    Готовое правило кодирования одного числового ключа.

    Тот же объект работает и при сборке границ, и при
    кодировании: второй реализации правила «куда попало
    значение» не существует.
    """

    key: str
    unit: str | None
    method: str
    boundaries: tuple[float, ...]
    zero_policy: str
    negative_policy: str
    buckets: tuple[Bucket, ...]

    @property
    def has_zero_bucket(self) -> bool:
        return bool(self.buckets) and self.buckets[0].zero

    def locate(self, value: float) -> tuple[str, int | None]:
        """
        Куда попадает значение: в диапазон, в невозможное или в
        неизвестное.
        """

        number = float(value)

        if math.isnan(number) or math.isinf(number):
            return FOUND_INVALID, None

        if self.method == METHOD_UNFITTED or not self.buckets:
            return FOUND_UNKNOWN, None

        if self.negative_policy == NEGATIVE_INVALID and number < 0.0:
            return FOUND_INVALID, None

        if self.has_zero_bucket and number == 0.0:
            return FOUND_BUCKET, 0

        offset = 1 if self.has_zero_bucket else 0

        return FOUND_BUCKET, offset + bisect.bisect_right(self.boundaries, number)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "unit": self.unit,
            "method": self.method,
            "boundaries": list(self.boundaries),
            "zero_policy": self.zero_policy,
            "negative_policy": self.negative_policy,
            "buckets": [bucket.as_dict() for bucket in self.buckets],
        }

    @staticmethod
    def from_dict(data: dict) -> "FittedEncoder":

        boundaries = tuple(float(value) for value in data["boundaries"])

        if list(boundaries) != sorted(set(boundaries)):
            raise BucketsError(f"границы ключа {data['key']} не возрастают строго: {boundaries}")

        buckets = tuple(
            Bucket(
                index=int(item["index"]),
                label=item["label"],
                lower=None if item["lower"] is None else float(item["lower"]),
                upper=None if item["upper"] is None else float(item["upper"]),
                zero=bool(item["zero"]),
            )
            for item in data["buckets"]
        )

        return FittedEncoder(
            key=data["key"],
            unit=data["unit"],
            method=data["method"],
            boundaries=boundaries,
            zero_policy=data["zero_policy"],
            negative_policy=data["negative_policy"],
            buckets=buckets,
        )


def quantile_boundaries(values: list[float], bins: int, algorithm: str) -> tuple[float, ...]:
    """
    Границы по train: примерно равное число наблюдений в каждом
    диапазоне, а не равная ширина в тенге.

    Совпавшие границы схлопываются, границы не выше минимума
    отбрасываются. После этого пустого диапазона быть не может:
    каждая граница это наблюдавшееся значение, и оно лежит в
    диапазоне, который с неё начинается.
    """

    if not values:
        return ()

    array = np.asarray(values, dtype=float)

    probabilities = [index / bins for index in range(1, bins)]

    edges = np.quantile(array, probabilities, method=algorithm)

    minimum = float(array.min())

    return tuple(sorted({float(edge) for edge in edges if float(edge) > minimum}))


def _samples(stats: FitStatistics) -> dict[str, list[float]]:
    """
    Отобранные значения каждого числового ключа.
    """

    return {key: stats.numeric[key].values() for key in sorted(stats.numeric)}


def _distribution(encoder: FittedEncoder, values: list[float]) -> tuple[list[int], dict[str, int]]:

    counts = [0] * len(encoder.buckets)
    other = {"invalid": 0, "unknown": 0}

    for value in values:

        found, index = encoder.locate(value)

        if found == FOUND_BUCKET:
            counts[index] += 1
        else:
            other[found] += 1

    return counts, other


def build_buckets(
    train: TrainCorpus,
    value_vocab: dict,
    config: TokenizerConfig,
    schema: SemanticSchema,
) -> dict:
    """
    Границы, политики и токены каждого числового ключа.
    """

    stats = train.statistics

    summary = {key: stats.numeric[key].summary() for key in sorted(stats.numeric)}
    samples = _samples(stats)

    entries: dict[str, dict] = {}
    encoders: dict[str, FittedEncoder] = {}
    warnings: list[str] = []

    for key in sorted(schema.numeric_keys):

        spec = config.numeric_encoders[key]
        info = schema.info(key)

        source_key = spec.fit_source or key
        measured = summary.get(source_key, {})

        observed = measured.get("n", 0)
        clients = measured.get("clients", 0)

        # Значения, по которым учатся границы: ноль исключается,
        # если он отдельный диапазон, а невозможный минус не
        # участвует в шкале никогда.
        usable = [
            value
            for value in samples.get(source_key, [])
            if not (spec.zero_policy == ZERO_SEPARATE and value == 0.0)
            and not (spec.negative_policy == NEGATIVE_INVALID and value < 0.0)
        ]

        boundaries: tuple[float, ...] = ()
        method = spec.method
        source = SOURCE_NONE
        note = ""

        if spec.method == METHOD_FIXED:
            boundaries = tuple(float(value) for value in spec.boundaries)
            source = SOURCE_CONFIG
            note = "шкала задана бизнесом и от выборки не зависит"

        elif spec.method == METHOD_QUANTILE:

            enough = (
                len(usable) >= config.numeric_min_values
                and clients >= config.numeric_min_clients
            )

            if enough:
                boundaries = quantile_boundaries(usable, spec.bins, config.quantile_algorithm)
                source = SOURCE_TRAIN
                note = f"границы по {len(usable)} значениям train у {clients} клиентов"

            if not boundaries:

                if spec.fallback:
                    boundaries = tuple(float(value) for value in spec.fallback)
                    method = METHOD_FIXED
                    source = SOURCE_FALLBACK
                    note = (
                        f"наблюдений {len(usable)} у {clients} клиентов, порог "
                        f"{config.numeric_min_values}/{config.numeric_min_clients}: "
                        "взята заранее объявленная шкала, по test границы не считаются"
                    )
                    warnings.append(f"{key}: объявленная шкала вместо квантилей ({note})")
                else:
                    method = METHOD_UNFITTED
                    source = SOURCE_NONE
                    note = "шкалы нет: значение получит числовое [UNK]"
                    warnings.append(f"{key}: кодировщика нет, значения станут [UNK]")

        else:
            note = spec.reason or "кодировщик объявлен отсутствующим"

        if method == METHOD_UNFITTED:
            buckets: tuple[Bucket, ...] = ()
        else:
            buckets = build_buckets_list(key, info.unit, boundaries, spec.zero_policy)

        encoder = FittedEncoder(
            key=key,
            unit=info.unit,
            method=method,
            boundaries=boundaries,
            zero_policy=spec.zero_policy,
            negative_policy=spec.negative_policy,
            buckets=buckets,
        )

        encoders[key] = encoder

        own = samples.get(key, [])
        counts, other = _distribution(encoder, own)

        # Пустых диапазонов после квантилей быть не может, и это
        # проверяется на тех значениях, по которым границы и
        # считались. Объявленный нулевой диапазон в проверку не
        # входит: он существует по решению о смысле нуля, а не
        # по наблюдениям, и пустым быть вправе.
        if source == SOURCE_TRAIN:

            fitted_counts, _ = _distribution(encoder, usable)

            empty = [
                bucket.label
                for bucket, count in zip(encoder.buckets, fitted_counts)
                if count == 0 and not bucket.zero
            ]

            if empty:
                raise BucketsError(
                    f"ключ {key}: квантильные границы оставили пустой диапазон {empty}: "
                    "такого быть не может, проверьте алгоритм квантилей"
                )

        entries[key] = {
            **encoder.as_dict(),
            "requested_method": spec.method,
            "requested_bins": spec.bins,
            "actual_bins": len(encoder.buckets),
            "boundary_source": source,
            "note": note,
            "fit_source": spec.fit_source,
            "weight_rule": info.weight_rule,
            "missing_policy": "пары нет; у расчётного ключа отсутствие объясняет причина",
            "invalid_policy": "[INVALID]",
            "below_range_policy": "первый диапазон",
            "above_range_policy": "последний диапазон, открытый сверху",
            "fallback": list(spec.fallback),
            "reason": spec.reason,
            "fit": {
                "values": len(usable),
                "clients": clients,
                "observed": observed,
                "sampled": bool(measured.get("sampled")),
                "sample_k": measured.get("sample_k"),
                "minimum": measured.get("minimum"),
                "maximum": measured.get("maximum"),
                "zeros": measured.get("n_zero", 0),
                "negatives": measured.get("n_negative", 0),
                "invalid": measured.get("n_invalid", 0),
                "algorithm": config.quantile_algorithm if source == SOURCE_TRAIN else None,
            },
            "distribution": {
                "buckets": counts,
                "invalid": other["invalid"],
                "unknown": other["unknown"],
                "rule": "распределение считается по выборке train этого же ключа",
            },
        }

    # --- токены диапазонов ---
    #
    # Диапазоны продолжают пространство ID сразу за
    # категориальными значениями: порядок — ключ по имени,
    # внутри ключа номер диапазона.

    first_bucket_id = int(value_vocab["next_id"])

    next_id = first_bucket_id

    by_key: dict[str, list[int]] = {}

    for key in sorted(entries):

        ids: list[int] = []

        for bucket in entries[key]["buckets"]:
            bucket["id"] = next_id
            ids.append(next_id)
            next_id += 1

        by_key[key] = ids

    report = {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "fit": train.as_dict(),
        "config_sha256": config.sha256(),
        "first_bucket_id": first_bucket_id,
        "size": next_id - first_bucket_id,
        "next_id": next_id,
        "fit_period": {"until": train.fit_end.isoformat(), "rule": "граница исключительная"},
        "rules": {
            "interval": "[lower, upper): значение на границе принадлежит следующему диапазону",
            "last": "последний диапазон открыт сверху",
            "first": "первый диапазон покрывает допустимые значения ниже наблюдавшихся на train",
            "empty": "совпавшие границы схлопываются, пустых train-диапазонов не остаётся",
            "label": "метка несёт ключ, границы и единицу: чужие диапазоны по номеру не склеиваются",
            "quantile": f"{config.quantile_algorithm} по выборке bottom-k, без seed и без влияния порядка",
            "threshold": (
                f"квантили считаются от {config.numeric_min_values} значений у "
                f"{config.numeric_min_clients} клиентов, иначе объявленная шкала или [UNK]"
            ),
        },
        "counts": {
            "keys": len(entries),
            "by_source": _by(entries, "boundary_source"),
            "by_method": _by(entries, "method"),
            "buckets": sum(item["actual_bins"] for item in entries.values()),
            "buckets_without_observations": sum(
                sum(1 for count in item["distribution"]["buckets"] if count == 0)
                for item in entries.values()
            ),
        },
        "encoders": {key: entries[key] for key in sorted(entries)},
        "by_key": by_key,
        "warnings": warnings,
    }

    return report


def load_buckets(directory: Path | None = None) -> dict:
    """
    Числовые диапазоны предыдущего этапа.
    """

    path = (Path(directory) / BUCKETS_FILE) if directory else tokenizer_path(BUCKETS_FILE)

    if not path.exists():
        raise BucketsError(f"нет {path}: выполните python -m src.tokenization.run buckets")

    return read_json(path)


def _by(entries: dict[str, dict], field_name: str) -> dict[str, int]:

    counts: dict[str, int] = {}

    for item in entries.values():
        counts[item[field_name]] = counts.get(item[field_name], 0) + 1

    return dict(sorted(counts.items()))


def load_encoders(registry: dict) -> dict[str, FittedEncoder]:
    """
    Замороженные кодировщики из реестра.
    """

    return {key: FittedEncoder.from_dict(item) for key, item in registry["encoders"].items()}


__all__ = [
    "FOUND_BUCKET",
    "FOUND_INVALID",
    "FOUND_UNKNOWN",
    "SOURCE_CONFIG",
    "SOURCE_FALLBACK",
    "SOURCE_NONE",
    "SOURCE_TRAIN",
    "Bucket",
    "BucketsError",
    "FittedEncoder",
    "build_buckets",
    "build_buckets_list",
    "load_buckets",
    "format_number",
    "load_encoders",
    "quantile_boundaries",
]
