from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.build import SAMPLES_SCHEMA
from src.dataset.lineage import lineage
from src.preprocessing.artifacts import read_json
from src.dataset.settings import META_FILE, SAMPLES_FILE, TIME_ANCHORS, dataset_dir

from .position import check, check_profile, profile_time_log, time_log


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вход модели на диске один: собранная группа.
#
#   data/05_dataset/<group>/samples.parquet
#
# Читается потоково, по группам строк: держать в памяти нужно
# ровно одну. Строка это клиент целиком, и события в ней уже
# отобраны и упорядочены датасетом.
#
# Временные позиции на диске не хранятся: их считает читатель
# (TemporalGroup) из event_time и profile_time той же строки, по
# точке отсчёта из meta.json набора. Отдельного этапа с копией
# примера ради двух вычислимых колонок нет.
#
#   event_time_log    давность события до cutoff примера
#                     (time_anchor last_event: до последнего события);
#   profile_time_log  давность вехи анкеты до cutoff примера, ноль
#                     у [USR] и Attributes.
# ============================================================


# Пример плюс две колонки времени, каждая следом за временем, из
# которого посчитана. Схема выводится из SAMPLES_SCHEMA, а не
# переписывается рядом: иначе новое поле примера молча
# потерялось бы при чтении.
TIME_LOG = pa.field("event_time_log", pa.list_(pa.float32()))

PROFILE_TIME_LOG = pa.field("profile_time_log", pa.list_(pa.float32()))


def _schema() -> pa.Schema:

    fields = list(SAMPLES_SCHEMA)

    for column, after in ((TIME_LOG, "event_time"), (PROFILE_TIME_LOG, "profile_time")):

        index = [field.name for field in fields].index(after)

        fields.insert(index + 1, column)

    return pa.schema(fields)


TEMPORAL_SCHEMA = _schema()


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

    @property
    def count(self) -> int:
        """
        Сколько групп строк в наборе.
        """

        return self._file.num_row_groups

    def groups(self) -> Iterator[pa.Table]:
        """
        Группы строк по одной, в порядке файла.
        """

        for number in range(self._file.num_row_groups):
            yield self._file.read_row_group(number)


class TemporalGroup(SamplesGroup):
    """
    Набор группы с временными позициями, посчитанными при чтении.
    """

    def __init__(self, group: str, directory: Path | None = None):

        super().__init__(group, directory)

        anchor = self.meta.get("time_anchor")

        if anchor not in TIME_ANCHORS:
            raise SamplesError(
                f"{self.directory / META_FILE}: точка отсчёта времени {anchor!r} не из "
                f"{list(TIME_ANCHORS)} — набор собран прежним кодом: выполните "
                f"python -m src.dataset.run {group} заново"
            )

        self.anchor = anchor

    def row_group(self, number: int, columns: list[str] | None = None) -> pa.Table:
        """
        Группа строк number с временными позициями.

        columns — какие колонки TEMPORAL_SCHEMA нужны, по умолчанию
        все. Для позиций читается и время, из которого они считаются,
        но наружу выходят только запрошенные колонки.
        """

        wanted = list(TEMPORAL_SCHEMA.names) if columns is None else list(columns)

        unknown = [name for name in wanted if name not in TEMPORAL_SCHEMA.names]

        if unknown:
            raise SamplesError(f"колонок {unknown} во входе модели нет")

        needed = set(wanted)

        if TIME_LOG.name in needed:
            needed |= {"client_id", "event_time"}

        if PROFILE_TIME_LOG.name in needed:
            needed |= {"client_id", "profile_time"}

        table = self._file.read_row_group(
            number, columns=[name for name in SAMPLES_SCHEMA.names if name in needed]
        )

        if TIME_LOG.name in needed or PROFILE_TIME_LOG.name in needed:
            table = self._with_time(table, TIME_LOG.name in needed, PROFILE_TIME_LOG.name in needed)

        return table.select(wanted)

    def _with_time(self, table: pa.Table, events: bool, profile: bool) -> pa.Table:

        clients = table.column("client_id").to_pylist()

        # Ноль у последнего события — правило отсчёта от него; от
        # cutoff последнее событие лежит на своей давности до T.
        cutoff = self.cutoff if self.anchor == "cutoff" else None

        if events:

            positions = []

            for client_id, moments in zip(clients, table.column("event_time").to_pylist()):
                value = time_log(client_id, moments, cutoff)
                check(client_id, value, len(moments), self.anchor)
                positions.append(value)

            table = table.append_column(TIME_LOG, pa.array(positions, type=TIME_LOG.type))

        if profile:

            ages = []

            for client_id, times in zip(clients, table.column("profile_time").to_pylist()):
                value = profile_time_log(client_id, times, self.cutoff)
                check_profile(client_id, value, times)
                ages.append(value)

            table = table.append_column(PROFILE_TIME_LOG, pa.array(ages, type=PROFILE_TIME_LOG.type))

        return table


__all__ = [
    "PROFILE_TIME_LOG",
    "TEMPORAL_SCHEMA",
    "TIME_LOG",
    "SamplesError",
    "SamplesGroup",
    "TemporalGroup",
]
