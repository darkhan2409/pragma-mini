from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.build import SAMPLES_SCHEMA
from src.dataset.lineage import lineage
from src.preprocessing.artifacts import read_json
from src.dataset.settings import META_FILE, SAMPLES_FILE, dataset_dir


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

        # Схема не меняется от того, на какой момент снята
        # анкета, поэтому одной её мало: набор прежней сборки
        # несёт в примерах анкету другого смысла.
        meta_path = self.directory / META_FILE

        if not meta_path.exists():
            raise SamplesError(
                f"нет {meta_path}: набор собран прежним кодом — выполните "
                f"python -m src.dataset.run {group} заново"
            )

        self.meta = read_json(meta_path)

        # Этап ставит на свой результат клеймо lineage() текущего
        # кода, поэтому вход обязан ему соответствовать: иначе
        # набор прежнего смысла вышел бы из этапа с новым клеймом.
        # Окно у набора своей группы, а в клейме — окна всех групп.
        current = lineage()

        needed = dict(current, windows=current["windows"].get(group))

        found = {
            "dataset_format": self.meta.get("format"),
            "profile_semantics": self.meta.get("profile_semantics"),
            "profile_lifelong_types": self.meta.get("profile_lifelong_types"),
            "windows": self.meta.get("window"),
        }

        if found != needed:
            raise SamplesError(
                f"{meta_path}: формат, смысл анкеты, вехи и окна {found}, а нужны {needed} — "
                f"набор собран прежним кодом: выполните python -m src.dataset.run {group} заново"
            )

        # Момент, на который собрана анкета: от него считается
        # давность вех.
        self.cutoff = datetime.fromisoformat(self.meta["events_cutoff"])

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
