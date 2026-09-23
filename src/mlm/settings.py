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
# Всё, что решает человек про голову: seed, сглаживание меток,
# сколько предсказаний показывать, размер порции, устройство и
# параметры оптимизатора обучения.
#
# Размерности, глубины и seed'ы четырёх энкодеров сюда НЕ
# входят: они приходят из весов этапов 09-12 вместе с их
# конфигурациями. Двум числам про одно и то же негде разойтись.
# ============================================================


# Один каталог на группу и три файла в нём.
#
#   data/13_mlm/<group>/targets.parquet
#   data/13_mlm/<group>/preview.html
#   data/13_mlm/<group>/weights.pt
MLM_DIR = DATA_DIR / "13_mlm"

TARGETS_FILE = "targets.parquet"

PREVIEW_FILE = "preview.html"

WEIGHTS_FILE = "weights.pt"

DEVICES = ("auto", "cpu", "cuda")

# Чекпойнт обучения лежит отдельно от 13_mlm: отчёт очищает свой
# каталог целиком и стёр бы его.
#
#   data/14_train/checkpoint.pt
TRAIN_DIR = DATA_DIR / "14_train"

CHECKPOINT_FILE = "checkpoint.pt"


class ConfigError(ValueError):
    """
    Конфигурация головы невозможна.
    """


@dataclass(frozen=True)
class MlmConfig:
    """
    Решения человека о голове и об отчёте.
    """

    # Seed розыгрыша весов головы. Остальное загружается.
    seed: int = 42

    # Сглаживание меток. В эталоне это константа модуля со
    # значением 0.1; здесь настройка, но с тем же значением.
    label_smoothing: float = 0.1

    # Сколько предсказаний показывать и сохранять на цель.
    top_k: int = 5

    # Событий за один проход энкодера события.
    events_per_chunk: int = 512

    device: str = "auto"

    # AdamW обучения. Отчёт их не использует.
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    def validate(self) -> None:

        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError(
                f"label_smoothing обязан лежать в [0, 1), получено {self.label_smoothing}"
            )

        if self.learning_rate <= 0.0:
            raise ConfigError(
                f"learning_rate обязан быть положительным, получено {self.learning_rate}"
            )

        if self.weight_decay < 0.0:
            raise ConfigError(
                f"weight_decay не может быть отрицательным, получено {self.weight_decay}"
            )

        for name in ("top_k", "events_per_chunk"):

            value = getattr(self, name)

            if value < 1:
                raise ConfigError(f"{name} обязан быть положительным, получено {value}")

        if self.device not in DEVICES:
            raise ConfigError(
                f"device обязан быть одним из {list(DEVICES)}, получено {self.device!r}"
            )

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "label_smoothing": self.label_smoothing,
            "top_k": self.top_k,
            "events_per_chunk": self.events_per_chunk,
            "device": self.device,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "MlmConfig":

        base = MlmConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига головы: {sorted(unknown)}")

        config = replace(
            base,
            seed=int(data.get("seed", base.seed)),
            label_smoothing=float(data.get("label_smoothing", base.label_smoothing)),
            top_k=int(data.get("top_k", base.top_k)),
            events_per_chunk=int(data.get("events_per_chunk", base.events_per_chunk)),
            device=str(data.get("device", base.device)),
            learning_rate=float(data.get("learning_rate", base.learning_rate)),
            weight_decay=float(data.get("weight_decay", base.weight_decay)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "MlmConfig":

        if path is None:
            config = MlmConfig()
            config.validate()
            return config

        return MlmConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def mlm_dir(group: str) -> Path:
    """
    Каталог результатов головы для группы.
    """

    return MLM_DIR / group


def checkpoint_path() -> Path:
    """
    Чекпойнт обучения. Учится только train, поэтому группы в пути нет.
    """

    return TRAIN_DIR / CHECKPOINT_FILE


__all__ = [
    "CHECKPOINT_FILE",
    "DEVICES",
    "MLM_DIR",
    "PREVIEW_FILE",
    "TARGETS_FILE",
    "TRAIN_DIR",
    "WEIGHTS_FILE",
    "ConfigError",
    "MlmConfig",
    "checkpoint_path",
    "mlm_dir",
]
