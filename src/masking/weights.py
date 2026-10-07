from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping

import numpy as np

from src.preprocessing.artifacts import read_json
from src.tokenization.finalvocab import vocabulary_digest
from src.tokenization.settings import VALUE_WEIGHTS_FILE, vocab_path

from .settings import MaskingConfig


# ============================================================
# ВЕСА ЗНАЧЕНИЙ ДЛЯ МЕХАНИЗМА VALUE
# ============================================================
#
# Механизм value выбирает цель с вероятностью, которая зависит от
# статистики TRAIN, а не одна на все значения. Иначе много целей
# уходит на почти константные поля (currency почти всегда KZT), и
# точность MLM растёт за их счёт. Розыгрыш остаётся случайным:
# статистика задаёт вероятность, поток KeyedRandom решает.
#
# Для ключа k по вхождениям его значений в train: N_k — число
# вхождений, p(v|k) — доля значения, K_k — число различных
# значений, H_k = -Σ p log p (в натах).
#
#   key_weight    kw_k = H_k / log K_k — нормированная энтропия:
#                 0 у константного ключа, 1 у равномерного;
#   value_weight  vw = clip((-log p(v|k) + λ) / (H_k + λ), 0.25, 3):
#                 среднее по вхождениям ключа — единица до обрезки,
#                 частое значение ниже, редкое выше, но не выше 3:
#                 сюрприз растёт логарифмом, а не как 1/p;
#   scale         g = Σ c / Σ c·kw·vw по всем вхождениям train, чтобы
#                 средняя вероятность до обрезки оставалась равной
#                 value_probability: веса перераспределяют цели, а не
#                 убавляют их.
#
#   P(k, v) = clip(value_probability · g · kw · vw, min, max)
#
# Значение известного ключа, которого в train не было (новый текст,
# [UNK]), считается редким: vw = 3. У ключа без статистики P —
# просто value_probability в тех же пределах. value_probability = 0
# выключает механизм: P = 0 без нижнего предела.
#
# Статистику считает fit по train (src.tokenization.valuestats) и
# кладёт рядом со словарём; val и test берут её как есть.
# ============================================================


# Сглаживание сюрприза, в натах: не даёт ключу с почти нулевой
# энтропией раздуть вес редкого значения до бесконечности.
SMOOTHING = 1.0

VALUE_WEIGHT_MIN = 0.25
VALUE_WEIGHT_MAX = 3.0


class WeightsError(ValueError):
    """
    Весов значений нет или они не от этого словаря.
    """


def entropy(counts: list[float]) -> float:
    """
    Энтропия распределения счётчиков, в натах.
    """

    total = float(sum(counts))

    return -sum((count / total) * math.log(count / total) for count in counts if count > 0)


def key_weight(counts: list[float]) -> float:
    """
    Нормированная энтропия ключа: 0 при одном значении.
    """

    distinct = sum(1 for count in counts if count > 0)

    if distinct < 2:
        return 0.0

    return entropy(counts) / math.log(distinct)


def value_weight(count: float, total: float, key_entropy: float) -> float:
    """
    Сглаженный и ограниченный сюрприз значения относительно
    энтропии его ключа.
    """

    surprise = -math.log(count / total)

    weight = (surprise + SMOOTHING) / (key_entropy + SMOOTHING)

    return min(VALUE_WEIGHT_MAX, max(VALUE_WEIGHT_MIN, weight))


def build(
    counts: Mapping[str, Mapping[tuple[int, ...], float]],
    key_ids: Mapping[str, int],
    labels: Mapping[str, Mapping[tuple[int, ...], str]],
    vocabulary: str,
    estimated: frozenset[str] = frozenset(),
) -> dict:
    """
    Файл весов: статистика train по ключам и значениям и всё, из
    чего вероятность восстанавливается однозначно.

    counts — вхождения значений в train по ключу, значение задано
    номерами своих токенов, как его видит маскер. estimated — ключи,
    у которых счёт оценён по выборке (числа).
    """

    keys: dict[str, dict] = {}

    total = 0.0
    weighted = 0.0

    for key in sorted(counts):

        table = {tokens: float(count) for tokens, count in counts[key].items() if count > 0}

        if not table:
            continue

        n = sum(table.values())
        numbers = list(table.values())

        spread = entropy(numbers)
        kw = key_weight(numbers)

        rows = []

        # Самые частые значения сверху: так файл читается глазами.
        for tokens, count in sorted(table.items(), key=lambda item: (-item[1], item[0])):

            vw = value_weight(count, n, spread)

            total += count
            weighted += count * kw * vw

            rows.append(
                {
                    "tokens": list(tokens),
                    "label": labels.get(key, {}).get(tokens, ""),
                    "count": _number(count),
                    "frequency": count / n,
                    "value_weight": vw,
                }
            )

        keys[key] = {
            "key_id": int(key_ids[key]),
            "count": _number(n),
            "distinct": len(table),
            "entropy": spread,
            "normalized_entropy": kw,
            "key_weight": kw,
            "estimated": key in estimated,
            "values": rows,
        }

    return {
        "vocabulary": vocabulary,
        "method": {
            "key_weight": "normalized_entropy",
            "value_weight": "clip((-log p + smoothing) / (entropy + smoothing), min, max)",
            "smoothing": SMOOTHING,
            "value_weight_min": VALUE_WEIGHT_MIN,
            "value_weight_max": VALUE_WEIGHT_MAX,
            "unseen_value_weight": VALUE_WEIGHT_MAX,
            "probability": "clip(value_probability * scale * key_weight * value_weight, "
                           "min_value_probability, max_value_probability)",
        },
        "occurrences": _number(total),
        # Нет ни одного информативного ключа — масштабировать нечего.
        "scale": total / weighted if weighted > 0.0 else 1.0,
        "keys": keys,
    }


def _number(value: float) -> int | float:
    """
    Счёт для файла: целым, где он точный, иначе оценкой.
    """

    return int(value) if float(value).is_integer() else round(float(value), 3)


@dataclass(frozen=True)
class ValueWeights:
    """
    Замороженные веса: номер ключа → (key_weight, {токены: value_weight}).
    """

    scale: float
    keys: dict[int, tuple[float, dict[tuple[int, ...], float]]]

    @staticmethod
    def from_payload(payload: dict) -> "ValueWeights":
        return ValueWeights(
            scale=float(payload["scale"]),
            keys={
                int(item["key_id"]): (
                    float(item["key_weight"]),
                    {tuple(row["tokens"]): float(row["value_weight"]) for row in item["values"]},
                )
                for item in payload["keys"].values()
            },
        )

    @cached_property
    def _singles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Для поиска массивом: известные ключи по возрастанию и их
        key_weight; значения из одного куска — код (ключ << 32 | токен)
        по возрастанию и value_weight.
        """

        known = sorted(self.keys)

        codes = sorted(
            ((key << 32) | tokens[0], value)
            for key, (_, table) in self.keys.items()
            for tokens, value in table.items()
            if len(tokens) == 1
        )

        return (
            np.array(known, dtype=np.int64),
            np.array([self.keys[key][0] for key in known], dtype=np.float64),
            np.array([code for code, _ in codes], dtype=np.int64),
            np.array([value for _, value in codes], dtype=np.float64),
        )

    def probabilities(self, key_ids: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                      value_ids: np.ndarray, config: MaskingConfig) -> np.ndarray:
        """
        probability каждого значения разом: значение — токены
        value_ids[start:start + length] ключа key_id.

        Формула та же и в том же порядке действий:
        ((value_probability · scale) · key_weight) · value_weight, затем
        границы. Значение из одного куска ищется в таблице массивом, из
        нескольких — probability по одному. Нечисловой вес (у min и max
        Python и numpy он ведёт себя по-разному) — тоже по одному.
        """

        out = np.empty(key_ids.size, dtype=np.float64)

        base = config.value_probability

        if base <= 0.0:
            out[:] = 0.0
            return out

        known, key_weights, codes, values = self._singles

        single = np.flatnonzero(lengths == 1)

        keys = key_ids[single]

        present = np.zeros(keys.size, dtype=bool)
        weight = np.zeros(keys.size, dtype=np.float64)

        if known.size:
            place = np.minimum(np.searchsorted(known, keys), known.size - 1)
            present = known[place] == keys
            weight = key_weights[place]

        value_weight = np.full(keys.size, VALUE_WEIGHT_MAX, dtype=np.float64)

        if codes.size:
            wanted = (keys << 32) | value_ids[starts[single]]
            spot = np.minimum(np.searchsorted(codes, wanted), codes.size - 1)
            hit = codes[spot] == wanted
            value_weight[hit] = values[spot[hit]]

        raw = np.where(present, (base * self.scale) * weight * value_weight, base)

        if bool(np.isfinite(raw).all()):
            out[single] = np.minimum(config.max_value_probability, np.maximum(config.min_value_probability, raw))
        else:
            single = np.zeros(0, dtype=np.int64)

        rest = np.setdiff1d(np.arange(key_ids.size), single, assume_unique=True)

        for index in rest.tolist():
            start = int(starts[index])
            tokens = tuple(value_ids[start:start + int(lengths[index])].tolist())
            out[index] = self.probability(int(key_ids[index]), tokens, config)

        return out

    def probability(self, key_id: int, tokens: tuple[int, ...], config: MaskingConfig) -> float:
        """
        Вероятность цели механизма value для значения ключа.
        """

        base = config.value_probability

        if base <= 0.0:
            return 0.0

        found = self.keys.get(int(key_id))

        if found is None:
            raw = base
        else:
            weight, table = found
            raw = base * self.scale * weight * table.get(tuple(tokens), VALUE_WEIGHT_MAX)

        return min(config.max_value_probability, max(config.min_value_probability, raw))


def load_value_weights(directory: Path | None = None) -> ValueWeights:
    """
    Веса значений словаря. Файл обязан быть собран под текущий
    словарь: номера токенов значений — номера этого словаря.
    """

    path = (Path(directory) / VALUE_WEIGHTS_FILE) if directory else vocab_path(VALUE_WEIGHTS_FILE)

    if not path.exists():
        raise WeightsError(f"нет {path}: выполните python -m src.tokenization.run fit")

    payload = read_json(path)

    if payload.get("vocabulary") != vocabulary_digest(directory):
        raise WeightsError(
            f"{path} собран под другой словарь: выполните "
            "python -m src.tokenization.run value-weights"
        )

    return ValueWeights.from_payload(payload)


__all__ = [
    "SMOOTHING",
    "VALUE_WEIGHT_MAX",
    "VALUE_WEIGHT_MIN",
    "ValueWeights",
    "WeightsError",
    "build",
    "entropy",
    "key_weight",
    "load_value_weights",
    "value_weight",
]
