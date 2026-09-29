from __future__ import annotations

from dataclasses import dataclass, field

from src.preprocessing.keys import CATEGORICAL, NUMERIC, TEXT

from .finalvocab import FrozenArtifacts
from .fit import TrainCorpus
from .schema import ORIGIN_PROFILE_CHANGE, WEIGHT_PER_EVENT, SemanticSchema
from .specials import UNK


# ============================================================
# ЗНАЧЕНИЯ TRAIN В НОМЕРАХ СЛОВАРЯ
# ============================================================
#
# Статистика для весов value-маскирования (src.masking.weights)
# берётся из того же прохода по train, на котором учится словарь
# (fit.read_train), а не отдельным чтением группы.
#
# Значение записывается теми же номерами токенов, что даёт
# кодирование 04, — ровно так его видит маскер:
#
#   категория  categorical_id или [UNK];
#   текст      куски BPE нормализованной строки;
#   число      номер диапазона (со split_by — среди диапазонов
#              условия), у ключа без шкалы [UNK].
#
# Берутся только ключи событий: анкета и изменения анкеты целями
# маски не бывают. Категории и тексты сосчитаны точно. Числа — по
# выборке fit (k значений с наименьшим хэшем): пока значений не
# больше k, выборка и есть все значения, иначе счёт диапазона —
# оценка n · доля в выборке, и ключ помечен estimated.
# ============================================================


@dataclass
class ValueCounts:
    """
    Вхождения значений train по ключу: токены значения → счёт.
    """

    counts: dict[str, dict[tuple[int, ...], float]] = field(default_factory=dict)
    labels: dict[str, dict[tuple[int, ...], str]] = field(default_factory=dict)
    key_ids: dict[str, int] = field(default_factory=dict)
    estimated: set[str] = field(default_factory=set)

    def add(self, key: str, tokens: tuple[int, ...], count: float, label: str) -> None:
        table = self.counts.setdefault(key, {})
        table[tokens] = table.get(tokens, 0.0) + count
        self.labels.setdefault(key, {}).setdefault(tokens, label)


def event_keys(schema: SemanticSchema, artifacts: FrozenArtifacts) -> dict[str, int]:
    """
    Ключи событий, у которых есть номер в словаре.
    """

    found: dict[str, int] = {}

    for key, info in schema.keys.items():

        if info.weight_rule != WEIGHT_PER_EVENT or info.origin == ORIGIN_PROFILE_CHANGE:
            continue

        key_id = artifacts.key_id(key)

        if key_id is not None:
            found[key] = key_id

    return found


def value_counts(train: TrainCorpus, artifacts: FrozenArtifacts, schema: SemanticSchema) -> ValueCounts:
    """
    Вхождения значений ключей событий train в номерах словаря.
    """

    stats = train.statistics
    unknown = artifacts.special(UNK)

    keys = event_keys(schema, artifacts)

    result = ValueCounts(key_ids=dict(keys))

    for (key, _kind, text), entry in stats.categorical.items():

        if key not in keys or schema.keys[key].value_kind != CATEGORICAL:
            continue

        found = artifacts.categorical_id(key, text)

        result.add(key, (unknown if found is None else found,), entry.count, text)

    for key, entries in stats.text.items():

        if key not in keys or schema.keys[key].value_kind != TEXT:
            continue

        for normalized, entry in entries.items():

            if artifacts.bpe.enabled:
                tokens = tuple(artifacts.piece_id(piece) for piece in artifacts.bpe.pieces(normalized))
            else:
                tokens = (unknown,)

            result.add(key, tokens, entry.count, normalized)

    for key in keys:

        if schema.keys[key].value_kind != NUMERIC:
            continue

        buckets = artifacts.buckets.get(key, ())
        split = buckets[0].split_by if buckets else None

        if split is None:
            sketches = [(None, stats.numeric.get(key))]
        else:
            sketches = [
                ({split: condition}, sketch)
                for (name, condition), sketch in sorted(
                    stats.split_numeric.items(), key=lambda item: (item[0][0], item[0][1] or "")
                )
                if name == key
            ]

        for record, sketch in sketches:

            if sketch is None or sketch.n == 0:
                continue

            sample = sketch.values()

            # Всё значения, пока их не больше k; иначе каждое
            # отобранное представляет n / k вхождений.
            share = sketch.n / len(sample)

            if sketch.sampled:
                result.estimated.add(key)

            for number in sample:

                found = artifacts.bucket_id(key, number, record)
                token = unknown if found is None else found

                result.add(key, (token,), share, artifacts.describe(token))

    return result


__all__ = [
    "ValueCounts",
    "event_keys",
    "value_counts",
]
