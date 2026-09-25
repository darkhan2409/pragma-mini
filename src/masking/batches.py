from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.lineage import lineage_problem
from src.batching.build import BATCHES_SCHEMA
from src.batching.settings import BATCHES_FILE, batches_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственный вход маскирования: выровненная группа.
#
#   data/07_batches/<group>/batches.parquet
#
# Один батч это одна группа строк parquet, и читается он целиком:
# маскирование идёт по клиентам, но файл на выходе обязан лечь
# теми же группами строк, что и вход. Строка i выхода это строка
# i входа.
#
# Память ограничена одним батчем, а не файлом.
# ============================================================


class BatchesError(ValueError):
    """
    Выровненную группу прочитать нельзя.
    """


class BatchesGroup:
    """
    Файл батчей по стандартному пути.
    """

    def __init__(self, group: str, directory: Path | None = None):

        self.group = group
        self.directory = Path(directory) if directory is not None else batches_dir(group)
        self.path = self.directory / BATCHES_FILE

        if not self.path.exists():
            raise BatchesError(
                f"нет {self.path}: выполните python -m src.batching.run {group}"
            )

        self._file = pq.ParquetFile(self.path)

        # Схема сверяется с той, которой батчер пишет сейчас.
        # Это ловит файл, собранный до смены формата: иначе
        # несовпадение всплыло бы посреди разбора значений, уже
        # без имени виноватого этапа.
        if not self._file.schema_arrow.equals(BATCHES_SCHEMA, check_metadata=False):
            raise BatchesError(
                f"{self.path} собран другой схемой батчей: выполните "
                f"python -m src.batching.run {group} заново"
            )

        # Схема от смысла анкеты не зависит: происхождение
        # сверяется отдельно.
        problem = lineage_problem(self.directory, f"python -m src.batching.run {group}")

        if problem is not None:
            raise BatchesError(problem)

    @property
    def rows(self) -> int:
        return self._file.metadata.num_rows

    def batches(self) -> Iterator[pa.Table]:
        """
        Батчи по одному, в порядке файла.
        """

        for number in range(self._file.num_row_groups):
            yield self._file.read_row_group(number)


__all__ = [
    "BatchesError",
    "BatchesGroup",
]
