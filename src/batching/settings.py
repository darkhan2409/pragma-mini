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
# Всё, что решает человек, а не данные, живёт здесь: сколько
# клиентов лежит в одном батче и какое окно упорядочивается по
# длине перед нарезкой.
#
# Здесь нет ни одного значения, посчитанного по данным.
# ============================================================


# Один каталог на группу и один файл в нём.
#
#   data/07_batches/<group>/batches.parquet
BATCHES_DIR = DATA_DIR / "07_batches"

BATCHES_FILE = "batches.parquet"


class ConfigError(ValueError):
    """
    Конфигурация батчей невозможна.
    """


@dataclass(frozen=True)
class BatchingConfig:
    """
    Решения человека о нарезке на батчи.
    """

    # Сколько клиентов в батче. Последний батч группы короче:
    # добивать его нечем, а выдумывать клиента нельзя.
    batch_size: int = 32

    # Окно сортировки, в батчах. Клиенты читаются потоково, и по
    # длине упорядочивается окно, а не весь файл: иначе память
    # росла бы вместе с группой, а клиентов одного батча
    # пришлось бы собирать по всему файлу.
    #
    # 1 означает «окно равно батчу»: сортировать внутри него
    # нечего, и батчи совпадут с порядком файла. Отдельного
    # переключателя для этого нет.
    sort_window_batches: int = 64

    @property
    def window_clients(self) -> int:
        return self.batch_size * self.sort_window_batches

    def validate(self) -> None:

        if self.batch_size < 1:
            raise ConfigError("batch_size обязан быть положительным")

        if self.sort_window_batches < 1:
            raise ConfigError("sort_window_batches обязан быть положительным")

    def as_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "sort_window_batches": self.sort_window_batches,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "BatchingConfig":

        base = BatchingConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига батчей: {sorted(unknown)}")

        config = replace(
            base,
            batch_size=int(data.get("batch_size", base.batch_size)),
            sort_window_batches=int(data.get("sort_window_batches", base.sort_window_batches)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "BatchingConfig":

        if path is None:
            config = BatchingConfig()
            config.validate()
            return config

        return BatchingConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def batches_dir(group: str) -> Path:
    """
    Каталог батчей группы.
    """

    return BATCHES_DIR / group


__all__ = [
    "BATCHES_DIR",
    "BATCHES_FILE",
    "BatchingConfig",
    "ConfigError",
    "batches_dir",
]
