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
# Всё, что решает человек: сколько блоков трансформера, сколько
# голов, какой FFN, какой dropout и каким seed разыграны веса.
#
# Длины вектора d здесь нет: она приходит из весов входного слоя
# (data/09_embeddings/<group>/weights.pt).
#
# Размера порции тоже нет, и он не нужен: вся анкета батча это
# [B, P, d], где P около двадцати. Считается одним куском.
# ============================================================


# Один каталог на группу и два файла в нём.
#
#   data/11_profiles/<group>/profiles.parquet
#   data/11_profiles/<group>/weights.pt
PROFILES_DIR = DATA_DIR / "11_profiles"

PROFILES_FILE = "profiles.parquet"

WEIGHTS_FILE = "weights.pt"


class ConfigError(ValueError):
    """
    Конфигурация энкодера анкеты невозможна.
    """


@dataclass(frozen=True)
class ProfileConfig:
    """
    Решения человека об энкодере анкеты.
    """

    # Seed розыгрыша весов энкодера. Свой, отдельно от входного
    # слоя: тот разыгран раньше и лежит на диске.
    seed: int = 42

    # Блоков трансформера. В эталоне анкета — самый мелкий
    # энкодер модели: depth_profile 1/1/3/9 против depth_event
    # 2/5/16/45. У нас событие берёт два блока, анкета один.
    layers: int = 1

    heads: int = 4

    # Ширина FFN. Эталон держит 4 * d.
    feedforward: int = 512

    # При сборке артефакта слой стоит в eval, и dropout выключен.
    dropout: float = 0.1

    # Основание лестницы частот TimeRoPE по времени вех — то же,
    # что у энкодера истории: шкала времени у них общая.
    rope_base: float = 10000.0

    def validate(self) -> None:

        for name in ("layers", "heads", "feedforward"):

            value = getattr(self, name)

            if value < 1:
                raise ConfigError(f"{name} обязан быть положительным, получено {value}")

        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout обязан лежать в [0, 1), получено {self.dropout}")

        if self.rope_base <= 1.0:
            raise ConfigError(f"rope_base обязан быть больше единицы, получено {self.rope_base}")

    def check_dim(self, dim: int) -> None:
        """
        Сверка с длиной вектора, пришедшей из весов этапа 09.
        """

        if dim % self.heads:
            raise ConfigError(
                f"d = {dim} не делится на {self.heads} голов: "
                "каждая голова берёт свою часть вектора"
            )

        if (dim // self.heads) % 2:
            raise ConfigError(
                f"на голову приходится {dim // self.heads} чисел, а TimeRoPE "
                "поворачивает их парами и требует чётности"
            )

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward": self.feedforward,
            "dropout": self.dropout,
            "rope_base": self.rope_base,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ProfileConfig":

        base = ProfileConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига энкодера анкеты: {sorted(unknown)}")

        config = replace(
            base,
            seed=int(data.get("seed", base.seed)),
            layers=int(data.get("layers", base.layers)),
            heads=int(data.get("heads", base.heads)),
            feedforward=int(data.get("feedforward", base.feedforward)),
            dropout=float(data.get("dropout", base.dropout)),
            rope_base=float(data.get("rope_base", base.rope_base)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "ProfileConfig":

        if path is None:
            config = ProfileConfig()
            config.validate()
            return config

        return ProfileConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def profiles_dir(group: str) -> Path:
    """
    Каталог векторов клиентов группы.
    """

    return PROFILES_DIR / group


__all__ = [
    "PROFILES_DIR",
    "PROFILES_FILE",
    "WEIGHTS_FILE",
    "ConfigError",
    "ProfileConfig",
    "profiles_dir",
]
