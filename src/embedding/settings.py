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
# Всё, что решает человек, а не данные, живёт здесь: длина
# вектора и seed начального розыгрыша весов.
#
# Здесь нет ни одного значения, посчитанного по данным. Размер
# словаря в конфигурацию НЕ входит: он свойство final_vocab.json
# и читается файлом, иначе два числа про одно и то же могли бы
# разойтись.
# ============================================================


# Один каталог на группу: веса слоя и отметка происхождения.
#
#   data/09_embeddings/<group>/weights.pt
#   data/09_embeddings/<group>/lineage.json
EMBEDDINGS_DIR = DATA_DIR / "09_embeddings"

WEIGHTS_FILE = "weights.pt"


class ConfigError(ValueError):
    """
    Конфигурация входного слоя невозможна.
    """


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Решения человека о входном слое.
    """

    # Длина вектора токена. Ни к числу токенов словаря, ни к
    # ширине батча не привязана: это отдельное решение.
    dim: int = 128

    # Seed розыгрыша начальных весов. Веса сохраняются рядом с
    # отчётом, поэтому один и тот же seed обязан давать тот же
    # файл.
    seed: int = 42

    def validate(self) -> None:

        if self.dim < 2:
            raise ConfigError(f"dim обязан быть не меньше двух, получено {self.dim}")

        # Синусоида пишется парами sin/cos, и на нечётной длине
        # последняя пара осталась бы незакрытой.
        if self.dim % 2:
            raise ConfigError(f"dim обязан быть чётным, получено {self.dim}")

    def as_dict(self) -> dict:
        return {
            "dim": self.dim,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "EmbeddingConfig":

        base = EmbeddingConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига эмбеддингов: {sorted(unknown)}")

        config = replace(
            base,
            dim=int(data.get("dim", base.dim)),
            seed=int(data.get("seed", base.seed)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "EmbeddingConfig":

        if path is None:
            config = EmbeddingConfig()
            config.validate()
            return config

        return EmbeddingConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def embeddings_dir(group: str) -> Path:
    """
    Каталог отчёта группы.
    """

    return EMBEDDINGS_DIR / group


__all__ = [
    "EMBEDDINGS_DIR",
    "WEIGHTS_FILE",
    "ConfigError",
    "EmbeddingConfig",
    "embeddings_dir",
]
