from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.preprocessing.artifacts import read_json

from .fit import TrainCorpus
from .keyvocab import next_id
from .scan import FitStatistics
from .schema import SemanticSchema
from .settings import (
    BUCKETS_FILE,
    METHOD_FIXED,
    METHOD_QUANTILE,
    METHOD_UNFITTED,
    NEGATIVE_INVALID,
    ZERO_SEPARATE,
    NumericEncoder,
    TokenizerConfig,
    vocab_path,
)


# ============================================================
# ЭТАП 4: ЧИСЛОВЫЕ ДИАПАЗОНЫ
# ============================================================
#
# Бакетизация заменяет точные числа диапазонами.
#
# Границы считает train и только train. Там, где наблюдений
# слишком мало, берётся заранее объявленная шкала, а не «лучшие»
# границы по маленькой выборке и тем более не границы по
# validation или test.
#
# Файл отвечает на один вопрос:
#
#   ключ -> <ключ>_bucket_<номер> -> ID, min, max
#
# Правило диапазона: min включительно, max не включительно.
# null в min означает, что диапазон открыт снизу, null в max —
# что открыт сверху. Отдельный нулевой диапазон записан как
# min = max = 0 и проверяется ДО обычных полуоткрытых: ноль там,
# где он объявлен событием, принадлежит ему, а не диапазону
# вокруг нуля.
#
# Значение ниже самой нижней границы попадает в первый диапазон,
# выше самой верхней — в последний: крайние диапазоны открыты, и
# ничего не теряется.
#
# Объявленная невозможность минуса влияет только на ОБУЧЕНИЕ
# границ: отрицательное значение у суммы это дефект данных, и в
# шкалу оно не входит. Отбрасывать такие числа обязан
# препроцессинг, а не словарь.
#
# Имя диапазона это его читаемое название в словаре:
# amount_due_bucket_3 ни с чем не спутаешь, и чужие номера по
# порядковому номеру не склеиваются.
#
# Две настройки кодировщика (по умолчанию: negative_bins у
# balance_after, split_by по direction у transaction_amount):
#
#   negative_bins  минус учится отдельной шкалой с границей в
#                  нуле: долг не делит диапазон с малым остатком;
#   split_by       своя шкала на каждое значение категориального
#                  ключа того же события (transaction_amount по
#                  direction): transaction_amount_debit_bucket_3.
#                  У такого диапазона в файле есть split_by и when
#                  (значение условия; null — события без него), и
#                  при кодировании число ищется только среди
#                  диапазонов своего условия. Условие, которого
#                  train не видел, даёт [UNK].
# ============================================================


SOURCE_TRAIN = "train_quantiles"
SOURCE_CONFIG = "config_fixed"
SOURCE_FALLBACK = "config_fallback"
SOURCE_NONE = "none"

# Исход поиска диапазона.
FOUND_BUCKET = "bucket"
FOUND_UNKNOWN = "unknown"


class BucketsError(ValueError):
    """
    Числовые границы построить или прочитать нельзя.
    """


@dataclass(frozen=True)
class Bucket:
    """
    Один диапазон: имя, границы и номер токена.

    minimum включительно, maximum не включительно; None означает
    открытую сторону. Нулевой диапазон это minimum == maximum == 0.
    """

    name: str
    minimum: float | None
    maximum: float | None
    token_id: int = -1
    # Шкала ключа делится по значениям split_by; when — значение
    # условия этого диапазона (None — события без условия).
    split_by: str | None = None
    when: str | None = None

    @property
    def zero(self) -> bool:
        return self.minimum == 0.0 and self.maximum == 0.0

    def contains(self, value: float) -> bool:

        if self.zero:
            return value == 0.0

        if self.minimum is not None and value < self.minimum:
            return False

        if self.maximum is not None and value >= self.maximum:
            return False

        return True

    def as_dict(self) -> dict:

        out = {"id": self.token_id, "min": self.minimum, "max": self.maximum}

        if self.split_by is not None:
            out.update(split_by=self.split_by, when=self.when)

        return out


def bucket_name(key: str, number: int) -> str:
    """
    Устойчивое имя диапазона: <ключ>_bucket_<номер с единицы>.
    """

    return f"{key}_bucket_{number}"


def build_bucket_list(key: str, boundaries: tuple[float, ...], zero_policy: str) -> list[Bucket]:
    """
    Диапазоны ключа по его границам, без номеров токенов.
    """

    buckets: list[Bucket] = []

    if zero_policy == ZERO_SEPARATE:
        buckets.append(Bucket(bucket_name(key, 1), 0.0, 0.0))

    edges: list[float | None] = [None, *boundaries, None]

    for position in range(len(edges) - 1):
        buckets.append(
            Bucket(bucket_name(key, len(buckets) + 1), edges[position], edges[position + 1])
        )

    return buckets


def locate(buckets: tuple[Bucket, ...], value: float) -> tuple[str, Bucket | None]:
    """
    Куда попадает значение: в диапазон или в неизвестное.

    Нулевой диапазон проверяется первым: он существует по решению
    о смысле нуля, а не по наблюдениям.
    """

    number = _number(value)

    if not buckets:
        return FOUND_UNKNOWN, None

    for bucket in buckets:
        if bucket.zero and bucket.contains(number):
            return FOUND_BUCKET, bucket

    for bucket in buckets:
        if not bucket.zero and bucket.contains(number):
            return FOUND_BUCKET, bucket

    return FOUND_UNKNOWN, None


def _number(value: object) -> float:

    number = float(value)

    if math.isnan(number) or math.isinf(number):
        raise BucketsError(
            f"значение {value!r} не число: такие значения обязан отбрасывать препроцессинг, "
            "до словаря они доходить не должны"
        )

    return number


class Scale:
    """
    Диапазоны одной шкалы, подготовленные к поиску: ответ тот же, что
    у locate(buckets, value), но обычный диапазон ищется двоичным
    поиском по нижним границам.

    Это верно только для сплошной шкалы: обычные диапазоны подряд,
    первый открыт снизу, последний сверху, каждый начинается там,
    где кончился предыдущий, и границы не убывают. Тогда значение лежит в последнем
    диапазоне, чей minimum <= значения, — его и находит bisect_right,
    и это тот же первый диапазон, который принял бы перебор. Шкала
    не сплошная — поиск перебором, как в locate.
    """

    def __init__(self, buckets: tuple[Bucket, ...]):

        self.buckets = tuple(buckets)
        self.zero = next((bucket for bucket in self.buckets if bucket.zero), None)
        self.ordinary = tuple(bucket for bucket in self.buckets if not bucket.zero)

        ordinary = self.ordinary

        edges = [bucket.minimum for bucket in ordinary[1:]]

        solid = (
            bool(ordinary)
            and ordinary[0].minimum is None
            and ordinary[-1].maximum is None
            and all(previous.maximum is not None and previous.maximum == following.minimum
                    for previous, following in zip(ordinary, ordinary[1:]))
            and all(lower <= upper for lower, upper in zip(edges, edges[1:]))
        )

        self.edges: list[float] | None = edges if solid else None

    def locate(self, value: object) -> Bucket | None:
        """
        Диапазон значения или None (неизвестное).
        """

        if self.edges is None:
            return locate(self.buckets, value)[1]

        number = _number(value)

        if self.zero is not None and number == 0.0:
            return self.zero

        return self.ordinary[bisect_right(self.edges, number)]


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


def _counts(buckets: list[Bucket], values: list[float]) -> dict[str, int]:
    """
    Сколько значений попало в каждый диапазон.
    """

    counts = {bucket.name: 0 for bucket in buckets}

    for value in values:

        found, bucket = locate(tuple(buckets), value)

        if found == FOUND_BUCKET and bucket is not None:
            counts[bucket.name] += 1

    return counts


def _quantile_scale(usable: list[float], spec: NumericEncoder, config: TokenizerConfig) -> tuple[float, ...]:
    """
    Квантильные границы шкалы; с negative_bins — отдельно у минуса
    и у остального, с границей в нуле.
    """

    if spec.negative_bins is None:
        return quantile_boundaries(usable, spec.bins, config.quantile_algorithm)

    negatives = [value for value in usable if value < 0.0]
    rest = [value for value in usable if value >= 0.0]

    edges = set(quantile_boundaries(rest, spec.bins, config.quantile_algorithm))

    # Граница в нуле — только когда train видел обе стороны: иначе
    # одна из них осталась бы пустым диапазоном. Квантили минуса —
    # только когда его хватает на шкалу, иначе минус один диапазон.
    if negatives and rest:

        edges.add(0.0)

        if len(negatives) >= config.numeric_min_values:
            edges.update(quantile_boundaries(negatives, spec.negative_bins, config.quantile_algorithm))

    return tuple(sorted(edges))


def _scale(
    label: str,
    spec: NumericEncoder,
    values: list[float],
    clients: int,
    config: TokenizerConfig,
    warnings: list[str],
) -> list[Bucket]:
    """
    Диапазоны одной шкалы без номеров: ключа целиком или его доли
    при split_by. label — имя шкалы в именах диапазонов.
    """

    # Значения, по которым учатся границы: ноль исключается, если
    # он отдельный диапазон, а невозможный минус не участвует в
    # шкале никогда.
    usable = [
        value
        for value in values
        if not (spec.zero_policy == ZERO_SEPARATE and value == 0.0)
        and not (spec.negative_policy == NEGATIVE_INVALID and value < 0.0)
    ]

    boundaries: tuple[float, ...] = ()
    method = spec.method
    source = SOURCE_NONE

    if spec.method == METHOD_FIXED:
        boundaries = tuple(float(value) for value in spec.boundaries)
        source = SOURCE_CONFIG

    elif spec.method == METHOD_QUANTILE:

        enough = len(usable) >= config.numeric_min_values and clients >= config.numeric_min_clients

        if enough:
            boundaries = _quantile_scale(usable, spec, config)
            source = SOURCE_TRAIN

        if not boundaries:

            if spec.fallback:
                boundaries = tuple(float(value) for value in spec.fallback)
                method = METHOD_FIXED
                source = SOURCE_FALLBACK
                warnings.append(
                    f"{label}: объявленная шкала вместо квантилей (наблюдений {len(usable)} "
                    f"у {clients} клиентов, порог {config.numeric_min_values}/"
                    f"{config.numeric_min_clients})"
                )
            else:
                method = METHOD_UNFITTED
                warnings.append(f"{label}: шкалы нет, значения станут [UNK]")

    if method == METHOD_UNFITTED:
        return []

    buckets = build_bucket_list(label, boundaries, spec.zero_policy)

    # Пустых диапазонов после квантилей быть не может, и это
    # проверяется на тех значениях, по которым границы и
    # считались. Объявленный нулевой диапазон в проверку не
    # входит: он существует по решению о смысле нуля.
    if source == SOURCE_TRAIN:

        counted = _counts(buckets, usable)

        empty = [bucket.name for bucket in buckets if not bucket.zero and counted[bucket.name] == 0]

        if empty:
            raise BucketsError(
                f"шкала {label}: квантильные границы оставили пустой диапазон {empty}: "
                "такого быть не может, проверьте алгоритм квантилей"
            )

    return buckets


def build_buckets(
    train: TrainCorpus,
    value_vocab: dict,
    config: TokenizerConfig,
    schema: SemanticSchema,
) -> tuple[dict[str, dict[str, dict]], list[str]]:
    """
    Диапазоны и их токены для каждого числового ключа.

    Возвращает сам словарь и предупреждения для терминала: в файл
    предупреждения не едут.
    """

    stats = train.statistics

    summary = {key: stats.numeric[key].summary() for key in sorted(stats.numeric)}
    samples = _samples(stats)

    warnings: list[str] = []

    prepared: dict[str, list[Bucket]] = {}

    for key in sorted(schema.numeric_keys):

        spec = config.numeric_encoders[key]

        if spec.split_by is None:
            source_key = spec.fit_source or key
            prepared[key] = _scale(
                key, spec, samples.get(source_key, []),
                summary.get(source_key, {}).get("clients", 0), config, warnings,
            )
            continue

        # Условия — в порядке имени, события без условия последними.
        conditions = sorted(
            (condition for name, condition in stats.split_numeric if name == key),
            key=lambda condition: (condition is None, condition or ""),
        )

        prepared[key] = []

        for condition in conditions:

            sketch = stats.split_numeric[(key, condition)]

            label = f"{key}_{condition}" if condition is not None else f"{key}_no_{spec.split_by}"

            prepared[key].extend(
                replace(bucket, split_by=spec.split_by, when=condition)
                for bucket in _scale(
                    label, spec, sketch.values(), sketch.summary()["clients"], config, warnings
                )
            )

    # --- номера ---
    #
    # Диапазоны продолжают пространство ID сразу за категориями:
    # порядок — ключ по имени, внутри ключа номер диапазона.

    number = next_id(value_vocab)

    out: dict[str, dict[str, dict]] = {}

    for key in sorted(prepared):

        entries: dict[str, dict] = {}

        for bucket in prepared[key]:
            entries[bucket.name] = replace(bucket, token_id=number).as_dict()
            number += 1

        out[key] = entries

    return out, warnings


def load_buckets(directory: Path | None = None) -> dict[str, dict[str, dict]]:
    """
    Числовые диапазоны предыдущего этапа.
    """

    path = (Path(directory) / BUCKETS_FILE) if directory else vocab_path(BUCKETS_FILE)

    if not path.exists():
        raise BucketsError(f"нет {path}: выполните python -m src.tokenization.run buckets")

    return read_json(path)


def read_buckets(data: dict[str, dict[str, dict]]) -> dict[str, tuple[Bucket, ...]]:
    """
    Диапазоны словаря в виде, которым кодируют.
    """

    out: dict[str, tuple[Bucket, ...]] = {}

    for key, entries in data.items():

        buckets = [
            Bucket(
                name=name,
                minimum=None if item["min"] is None else float(item["min"]),
                maximum=None if item["max"] is None else float(item["max"]),
                token_id=int(item["id"]),
                split_by=item.get("split_by"),
                when=item.get("when"),
            )
            for name, item in entries.items()
        ]

        if len({bucket.split_by for bucket in buckets}) > 1:
            raise BucketsError(f"ключ {key}: у диапазонов разные split_by")

        # Шкала каждого условия обязана быть сплошной сама по себе.
        scales: dict[str | None, list[Bucket]] = {}

        for bucket in buckets:
            scales.setdefault(bucket.when, []).append(bucket)

        for when, scale in scales.items():
            _check_order(key if when is None else f"{key} ({when})", scale)

        out[key] = tuple(buckets)

    return out


def _check_order(key: str, buckets: list[Bucket]) -> None:
    """
    Диапазоны ключа идут подряд, не перекрываются и покрывают всю
    шкалу.

    Иначе два диапазона приняли бы одно значение, и результат
    зависел бы от порядка чтения файла.
    """

    ordinary = [bucket for bucket in buckets if not bucket.zero]

    if not ordinary:
        return

    if ordinary[0].minimum is not None:
        raise BucketsError(
            f"ключ {key}: первый диапазон {ordinary[0].name} закрыт снизу, "
            "и значения ниже него потерялись бы"
        )

    if ordinary[-1].maximum is not None:
        raise BucketsError(
            f"ключ {key}: последний диапазон {ordinary[-1].name} закрыт сверху, "
            "и значения выше него потерялись бы"
        )

    for previous, following in zip(ordinary, ordinary[1:]):

        if previous.maximum != following.minimum:
            raise BucketsError(
                f"ключ {key}: диапазон {following.name} начинается с {following.minimum}, "
                f"а {previous.name} закончился на {previous.maximum}"
            )


__all__ = [
    "FOUND_BUCKET",
    "FOUND_UNKNOWN",
    "SOURCE_CONFIG",
    "SOURCE_FALLBACK",
    "SOURCE_NONE",
    "SOURCE_TRAIN",
    "Bucket",
    "BucketsError",
    "Scale",
    "bucket_name",
    "build_bucket_list",
    "build_buckets",
    "load_buckets",
    "locate",
    "quantile_boundaries",
    "read_buckets",
]
