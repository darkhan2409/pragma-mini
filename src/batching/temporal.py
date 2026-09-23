from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.temporal.build import TEMPORAL_SCHEMA
from src.temporal.settings import TEMPORAL_FILE, temporal_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственный вход батчера: группа с временными позициями.
#
#   data/06_temporal/<group>/temporal.parquet
#
# Файл читается потоково, по группам строк, и отдаётся окнами
# ровно по W клиентов. Окно упорядочивается по длине, и уже из
# него режутся батчи: рядом оказываются истории близкой длины, и
# заполнителя нужно меньше.
#
# Сортируется ОКНО, а не весь файл. Глобальный порядок заставил
# бы собирать клиентов одного батча по всему файлу, а это либо
# многократное перечитывание, либо память размером с группу.
# Здесь каждый байт читается ровно один раз.
#
# Окно живёт буферами Arrow, а не объектами Python: длины
# считает list_value_length, порядок — sort_indices и take.
# Поэтому потолок памяти это окно, а не группа.
#
# Границы окна задаются числом клиентов, а не тем, какими
# группами строк собран вход: иначе результат зависел бы от
# настроек чужого этапа.
# ============================================================


# Длина примера в токенах. Колонки с таким именем во входе нет:
# она считается на месте и служит только ключом сортировки.
LENGTH = "n_tokens"


class TemporalError(ValueError):
    """
    Группу с временными позициями прочитать нельзя.
    """


class TemporalGroup:
    """
    Файл временных позиций по стандартному пути.
    """

    def __init__(self, group: str, directory: Path | None = None):

        self.group = group
        self.directory = Path(directory) if directory is not None else temporal_dir(group)
        self.path = self.directory / TEMPORAL_FILE

        if not self.path.exists():
            raise TemporalError(
                f"нет {self.path}: выполните python -m src.temporal.run {group}"
            )

        self._file = pq.ParquetFile(self.path)

        # Схема сверяется с той, которой этап временных позиций
        # пишет сейчас. Это ловит файл, собранный до смены
        # формата: иначе несовпадение всплыло бы где-то в
        # выравнивании, уже без имени виноватого этапа.
        if not self._file.schema_arrow.equals(TEMPORAL_SCHEMA, check_metadata=False):
            raise TemporalError(
                f"{self.path} собран другой схемой: выполните "
                f"python -m src.temporal.run {group} заново"
            )

    @property
    def rows(self) -> int:
        return self._file.metadata.num_rows

    def windows(self, size: int) -> Iterator[pa.Table]:
        """
        Окна ровно по size клиентов, упорядоченные по длине.

        Последнее окно короче: добирать его нечем.
        """

        carry: pa.Table | None = None

        for number in range(self._file.num_row_groups):

            chunk = self._file.read_row_group(number)

            carry = chunk if carry is None else pa.concat_tables([carry, chunk])

            while carry.num_rows >= size:
                yield _ordered(carry.slice(0, size))
                carry = carry.slice(size)

        if carry is not None and carry.num_rows:
            yield _ordered(carry)


def _ordered(window: pa.Table) -> pa.Table:
    """
    Окно по возрастанию длины примера.

    Ключ полный: client_id уникален, по одной строке на клиента,
    поэтому порядок не зависит от устойчивости сортировки и двух
    разных ответов дать не может.
    """

    window = window.append_column(
        LENGTH, pc.list_value_length(window.column("key_ids")).cast(pa.int32())
    )

    order = pc.sort_indices(
        window, sort_keys=[(LENGTH, "ascending"), ("client_id", "ascending")]
    )

    return window.take(order)


__all__ = [
    "LENGTH",
    "TemporalError",
    "TemporalGroup",
]
