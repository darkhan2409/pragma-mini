from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
