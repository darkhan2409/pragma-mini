from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что решает человек, а не данные, живёт здесь: seed и
# четыре вероятности.
#
# Здесь нет ни одного значения, посчитанного по данным.
#
# Вероятности взяты из pragmatiq/training/masking.py, но одна из
# них поменяла смысл. Там механизм token разыгрывал КАЖДЫЙ токен,
# и текстовое значение могло оказаться скрытым наполовину. У нас
# текст занимает несколько позиций с одним ключом, и половина
# названия это подсказка, а не задача. Поэтому token заменён на
# value: 0.15 относится к ЗНАЧЕНИЮ целиком.
#
# Маска на диске не хранится: её разыгрывает читатель входа модели
# (src.mlm.inputs.Source, src.embedding.inputs.Source). Розыгрыш
# ключуется seed, группой и клиентом, поэтому одинаков при каждом
# чтении, в любом порядке и в любом процессе.
# ============================================================


class ConfigError(ValueError):
    """
    Конфигурация маскирования невозможна.
    """


@dataclass(frozen=True)
class MaskingConfig:
    """
    Решения человека о том, что прятать.
    """

    # Seed розыгрыша. У val и test маска с этим seed одна и та же
    # при каждом чтении; train каждую эпоху выводит из него свой
    # (src.mlm.train.for_epoch).
    seed: int = 42

    # Отдельное значение целиком, со всеми его кусками BPE. При
    # informativeness_weighted_masking — средняя вероятность по train:
    # у каждого значения своя (src.masking.weights).
    value_probability: float = 0.15

    # Событие целиком, со всеми его значениями.
    event_probability: float = 0.10

    # Ключ у клиента, со всеми его значениями в допустимых
    # событиях. Разыгрывается пара (клиент, ключ), а не ключ
    # вообще: иначе выбор одного клиента решал бы за всех.
    key_probability: float = 0.10

    # Доля выбранных значений, которая уходит в [UNK] вместо
    # [MASK]. Такое значение испорчено, но в loss не входит:
    # модель видит незнакомое значение и не получает за него
    # ни награды, ни штрафа.
    unknown_probability: float = 0.10

    # Значение ключа, выбранного механизмом key, в событии вне целей
    # (контекст) независимо портится в [UNK] с этой вероятностью —
    # целиком, со всеми кусками, без метки. Без порчи на val (цели —
    # последние 4 месяца) скрытое значение почти всегда видно в более
    # раннем событии клиента, и key проверяет копирование, а не
    # восстановление; закрыть контекст целиком значило бы отнять и
    # законные подсказки. Это отдельный механизм: unknown_probability
    # от него не растёт. На train период целей — весь контекст, и
    # портить там нечего.
    key_context_corruption_probability: float = 0.5

    # Взвешивать ли механизм value по информативности: вероятность
    # цели значения зависит от энтропии его ключа и частоты самого
    # значения в train (value_weights.json словаря) и лежит в
    # [min_value_probability, max_value_probability]. Механизмы event
    # и key, [UNK] и порча контекста от этого не меняются. Выключено —
    # у всех значений ровно value_probability, как прежде.
    informativeness_weighted_masking: bool = True
    min_value_probability: float = 0.05
    max_value_probability: float = 0.35

    def validate(self) -> None:

        for name in (
            "value_probability",
            "event_probability",
            "key_probability",
            "unknown_probability",
            "key_context_corruption_probability",
            "min_value_probability",
            "max_value_probability",
        ):
            probability = getattr(self, name)

            if not 0.0 <= probability <= 1.0:
                raise ConfigError(f"{name} обязан лежать в [0, 1], получено {probability}")

        if self.min_value_probability > self.max_value_probability:
            raise ConfigError(
                f"min_value_probability {self.min_value_probability} больше "
                f"max_value_probability {self.max_value_probability}"
            )

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "value_probability": self.value_probability,
            "event_probability": self.event_probability,
            "key_probability": self.key_probability,
            "unknown_probability": self.unknown_probability,
            "key_context_corruption_probability": self.key_context_corruption_probability,
            "informativeness_weighted_masking": self.informativeness_weighted_masking,
            "min_value_probability": self.min_value_probability,
            "max_value_probability": self.max_value_probability,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "MaskingConfig":

        base = MaskingConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига маскирования: {sorted(unknown)}")

        config = replace(
            base,
            seed=int(data.get("seed", base.seed)),
            value_probability=float(data.get("value_probability", base.value_probability)),
            event_probability=float(data.get("event_probability", base.event_probability)),
            key_probability=float(data.get("key_probability", base.key_probability)),
            unknown_probability=float(
                data.get("unknown_probability", base.unknown_probability)
            ),
            key_context_corruption_probability=float(
                data.get("key_context_corruption_probability", base.key_context_corruption_probability)
            ),
            informativeness_weighted_masking=bool(
                data.get("informativeness_weighted_masking", base.informativeness_weighted_masking)
            ),
            min_value_probability=float(data.get("min_value_probability", base.min_value_probability)),
            max_value_probability=float(data.get("max_value_probability", base.max_value_probability)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "MaskingConfig":

        if path is None:
            config = MaskingConfig()
            config.validate()
            return config

        return MaskingConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "ConfigError",
    "MaskingConfig",
]
