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
# Всё, что решает человек: глубина и ширина энкодера, основание
# лестницы частот RoPE, seed и устройство счёта.
#
# Длины вектора d здесь нет: она приходит из векторов этапов 10
# и 11 и сверяется между ними.
#
# Капа на длину истории здесь тоже НЕТ, и это решение. Замерено:
# самая длинная история группы — 23 319 событий, и полное точное
# внимание по ней занимает 694 мс и 194 МиБ. Обрезать нечего.
# ============================================================


# Один каталог на группу и два файла в нём.
#
#   data/12_history/<group>/history.parquet
#   data/12_history/<group>/weights.pt
HISTORY_DIR = DATA_DIR / "12_history"

HISTORY_FILE = "history.parquet"

WEIGHTS_FILE = "weights.pt"

DEVICES = ("auto", "cpu", "cuda")


class ConfigError(ValueError):
    """
    Конфигурация энкодера истории невозможна.
    """


@dataclass(frozen=True)
class HistoryConfig:
    """
    Решения человека об энкодере истории.
    """

    # Seed розыгрыша весов энкодера.
    seed: int = 42

    # Блоков трансформера. В эталоне depth_history 1/2/6/18
    # против depth_event 2/5/16/45: история глубже анкеты, но
    # мельче события.
    layers: int = 2

    heads: int = 4

    # Ширина FFN. Эталон держит 4 * d.
    feedforward: int = 512

    # При сборке артефакта слой стоит в eval, и dropout выключен.
    dropout: float = 0.1

    # Основание лестницы частот TimeRoPE. В эталоне помечено как
    # догадка; проверено, что при d/heads = 32 временная ось
    # живая — косинусная близость на шести месяцах 0.149 при
    # пороге 0.85.
    rope_base: float = 10000.0

    # Где считать. auto берёт CUDA, если она есть: на самой
    # длинной истории это 694 мс против 3.6 с.
    device: str = "auto"

    def validate(self) -> None:

        for name in ("layers", "heads", "feedforward"):

            value = getattr(self, name)

            if value < 1:
                raise ConfigError(f"{name} обязан быть положительным, получено {value}")

        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"dropout обязан лежать в [0, 1), получено {self.dropout}")

        if self.rope_base <= 1.0:
            raise ConfigError(f"rope_base обязан быть больше единицы, получено {self.rope_base}")

        if self.device not in DEVICES:
            raise ConfigError(f"device обязан быть одним из {list(DEVICES)}, получено {self.device!r}")

    def check_dim(self, dim: int) -> None:
        """
        Сверка с длиной вектора, пришедшей из векторов этапов 10 и 11.
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
            "device": self.device,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "HistoryConfig":

        base = HistoryConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига энкодера истории: {sorted(unknown)}")

        config = replace(
            base,
            seed=int(data.get("seed", base.seed)),
            layers=int(data.get("layers", base.layers)),
            heads=int(data.get("heads", base.heads)),
            feedforward=int(data.get("feedforward", base.feedforward)),
            dropout=float(data.get("dropout", base.dropout)),
            rope_base=float(data.get("rope_base", base.rope_base)),
            device=str(data.get("device", base.device)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "HistoryConfig":

        if path is None:
            config = HistoryConfig()
            config.validate()
            return config

        return HistoryConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def history_dir(group: str) -> Path:
    """
    Каталог векторов истории группы.
    """

    return HISTORY_DIR / group


__all__ = [
    "DEVICES",
    "HISTORY_DIR",
    "HISTORY_FILE",
    "WEIGHTS_FILE",
    "ConfigError",
    "HistoryConfig",
    "history_dir",
]
