from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.build import SAMPLES_SCHEMA
from src.dataset.settings import SAMPLES_FILE, dataset_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственный вход этапа: собранная группа.
#
#   data/05_dataset/<group>/samples.parquet
#
# Читается потоково, по группам строк: держать в памяти нужно
# ровно одну. Строка это клиент целиком, и события в ней уже
# отобраны и упорядочены датасетом.
# ============================================================


class SamplesError(ValueError):
    """
    Собранную группу прочитать нельзя.
    """


class SamplesGroup:
    """
    Файл примеров по стандартному пути.
    """

    def __init__(self, group: str, directory: Path | None = None):

        self.group = group
        self.directory = Path(directory) if directory is not None else dataset_dir(group)
        self.path = self.directory / SAMPLES_FILE

        if not self.path.exists():
            raise SamplesError(
                f"нет {self.path}: выполните python -m src.dataset.run {group}"
            )

        self._file = pq.ParquetFile(self.path)

        # Схема сверяется с той, которой датасет пишет сейчас:
        # иначе несовпадение всплыло бы посреди расчёта, уже без
        # имени виноватого этапа.
        if not self._file.schema_arrow.equals(SAMPLES_SCHEMA, check_metadata=False):
            raise SamplesError(
                f"{self.path} собран другой схемой примеров: выполните "
                f"python -m src.dataset.run {group} заново"
            )

    @property
    def rows(self) -> int:
        return self._file.metadata.num_rows

    def groups(self) -> Iterator[pa.Table]:
        """
        Группы строк по одной, в порядке файла.
        """

        for number in range(self._file.num_row_groups):
            yield self._file.read_row_group(number)


__all__ = [
    "SamplesError",
    "SamplesGroup",
]
