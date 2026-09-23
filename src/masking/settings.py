from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR



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
# ============================================================


# Один каталог на группу и один файл в нём.
#
#   data/08_masked/<group>/masked.parquet
MASKED_DIR = DATA_DIR / "08_masked"

MASKED_FILE = "masked.parquet"


class ConfigError(ValueError):
    """
    Конфигурация маскирования невозможна.
    """


@dataclass(frozen=True)
class MaskingConfig:
    """
    Решения человека о том, что прятать.
    """

    # Seed розыгрыша. Результат сохраняется на диск, поэтому
    # маска это свойство файла, а не эпохи: у val и test она
    # фиксирована ровно так же, как у train.
    seed: int = 42

    # Отдельное значение целиком, со всеми его кусками BPE.
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

    def validate(self) -> None:

        for name in (
            "value_probability",
            "event_probability",
            "key_probability",
            "unknown_probability",
        ):
            probability = getattr(self, name)

            if not 0.0 <= probability <= 1.0:
                raise ConfigError(f"{name} обязан лежать в [0, 1], получено {probability}")

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "value_probability": self.value_probability,
            "event_probability": self.event_probability,
            "key_probability": self.key_probability,
            "unknown_probability": self.unknown_probability,
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


def masked_dir(group: str) -> Path:
    """
    Каталог масок группы.
    """

    return MASKED_DIR / group


__all__ = [
    "MASKED_DIR",
    "MASKED_FILE",
    "ConfigError",
    "MaskingConfig",
    "masked_dir",
]
