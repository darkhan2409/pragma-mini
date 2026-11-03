from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что решает человек, а не данные, живёт здесь: seed,
# вероятности механизмов и группы зависимых полей.
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


# Имя ключа типа события в словаре ключей — то же, что
# src.dataset.tokenized.EVENT_TYPE_KEY; здесь своё, чтобы конфиг маски
# не тянул за собой словарь и набор.
EVENT_TYPE = "event_type"


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

    # Группы зависимых полей события (имена ключей словаря). Поле
    # группы, ставшее целью механизмом value или key, закрывает все
    # присутствующие поля своей группы в том же событии: они тоже
    # становятся целями. Иначе цель восстанавливалась бы из видимого
    # партнёра (merchant_name → mcc, direction → знак суммы), и loss
    # мерил бы копирование. Пусто — правила нет, маска прежняя.
    field_groups: tuple[tuple[str, ...], ...] = ()

    # event_type — цель только механизма event: механизмы value и key
    # его не выбирают. Фильтр стоит после всех розыгрышей, поэтому
    # остальные цели и их [UNK] побитно те же, что без флага.
    event_type_event_only: bool = False

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

        seen: set[str] = set()

        for group in self.field_groups:

            if len(group) < 2:
                raise ConfigError(f"группа полей {list(group)} из одного ключа: ей нечего закрывать")

            for key in group:

                if key in seen:
                    raise ConfigError(f"ключ {key} стоит в двух группах полей (или дважды в одной)")

                seen.add(key)

        # Замыкание сделало бы event_type целью value или key — ровно
        # то, что флаг запрещает.
        if self.event_type_event_only and EVENT_TYPE in seen:
            raise ConfigError(
                f"{EVENT_TYPE} в группе полей при event_type_event_only: замыкание делало бы его целью value/key"
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
            "field_groups": [list(group) for group in self.field_groups],
            "event_type_event_only": self.event_type_event_only,
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
            field_groups=_groups(data.get("field_groups", base.field_groups)),
            event_type_event_only=bool(data.get("event_type_event_only", base.event_type_event_only)),
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


def _groups(data: Any) -> tuple[tuple[str, ...], ...]:
    """
    Группы полей из JSON: список списков имён ключей.
    """

    if not isinstance(data, (list, tuple)) or not all(isinstance(group, (list, tuple)) for group in data):
        raise ConfigError(f"field_groups обязан быть списком списков имён ключей, получено {data!r}")

    if not all(isinstance(key, str) for group in data for key in group):
        raise ConfigError(f"в field_groups имена ключей обязаны быть строками, получено {data!r}")

    return tuple(tuple(group) for group in data)


__all__ = [
    "EVENT_TYPE",
    "ConfigError",
    "MaskingConfig",
]
