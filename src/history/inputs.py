from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.lineage import lineage_problem
from src.dataset.settings import dataset_dir
from src.event.build import EVENTS_SCHEMA
from src.event.settings import EVENTS_FILE, events_dir
from src.profile.build import PROFILES_SCHEMA
from src.profile.settings import PROFILES_FILE, profiles_dir
from src.temporal.position import TemporalError
from src.temporal.samples import SamplesError, TemporalGroup


# ============================================================
# ИСТОРИЯ КЛИЕНТА ИЗ ТРЁХ ФАЙЛОВ
# ============================================================
#
# История собирается из трёх мест, и каждое отвечает за своё:
#
#   data/05_dataset  — сколько у клиента событий и какая у каждого
#       временная позиция (считается при чтении, TemporalGroup);
#   data/08_events   — вектор каждого события ПОСЛЕ календаря,
#       строкой на событие;
#   data/09_profiles — вектор анкеты, он же z_a.
#
# У набора и анкет строка это клиент, у событий — событие,
# поэтому сверка разная: клиенты сверяются построчно, а события
# режутся по n_events и сверяются по client_id и по номеру внутри
# клиента. Файлы собираются разными командами, и верить их
# согласию на слово нельзя.
#
# Колонка event (до календаря) из этапа 08 не читается: в историю
# идёт только event_dated.
#
# Заполнителя нет вовсе. Читатель отдаёт ровно n_events векторов
# и ровно n_events позиций.
# ============================================================


SAMPLE_COLUMNS = ["client_id", "event_time_log"]

EVENT_COLUMNS = ["batch_index", "client_id", "event", "vector"]

PROFILE_COLUMNS = ["batch_index", "client_id", "dim", "profile"]


class InputError(ValueError):
    """
    Историю собрать нельзя.
    """


@dataclass(frozen=True)
class Client:
    """
    Одна история целиком, без заполнителя.
    """

    batch_index: int
    client_id: str

    profile: np.ndarray    # [d] вектор анкеты
    events: np.ndarray     # [n, d] векторы событий после календаря
    positions: np.ndarray  # [n] float32, log-секунды до точки отсчёта (time_anchor набора)

    @property
    def n_events(self) -> int:
        return int(self.events.shape[0])

    @property
    def length(self) -> int:
        """
        Длина последовательности вместе с [USR].
        """

        return self.n_events + 1


class Source:
    """
    Три файла группы, открытые один раз.
    """

    def __init__(self, group: str):

        self.group = group

        self.samples_path = dataset_dir(group)
        self.events_path = events_dir(group) / EVENTS_FILE
        self.profiles_path = profiles_dir(group) / PROFILES_FILE

        try:
            self._samples = TemporalGroup(group, self.samples_path)
        except SamplesError as error:
            raise InputError(str(error)) from error

        self._events = _open(
            self.events_path, EVENTS_SCHEMA, f"python -m src.event.run {group}"
        )
        self._profiles = _open(
            self.profiles_path, PROFILES_SCHEMA, f"python -m src.profile.run {group}"
        )

        # Схема векторов анкеты от энкодера не зависит: векторы
        # прежнего энкодера выглядели бы исправными.
        problem = lineage_problem(self.profiles_path.parent, f"python -m src.profile.run {group}")

        if problem:
            raise InputError(problem)

        counts = {
            "набора": self._samples.count,
            "событий": self._events.num_row_groups,
            "анкет": self._profiles.num_row_groups,
        }

        if len(set(counts.values())) != 1:
            raise InputError(
                f"групп строк разное число: {counts} — файлы собраны в разное время"
            )

    @property
    def count(self) -> int:
        return self._samples.count

    @property
    def dim(self) -> int:
        """
        Длина вектора, известная до чтения самих векторов.
        """

        return _dim(self._profiles.read_row_group(0, columns=["dim"]).to_pylist())

    def batch(self, index: int) -> list[Client]:
        """
        Истории одного батча, по клиенту на элемент.
        """

        if index < 0 or index >= self.count:
            raise InputError(
                f"батча {index} нет: в группе {self.count} батчей, "
                f"номера от 0 до {self.count - 1}"
            )

        try:
            batch = self._samples.row_group(index, columns=SAMPLE_COLUMNS).to_pylist()
        except TemporalError as error:
            raise InputError(str(error)) from error

        events = self._events.read_row_group(index, columns=EVENT_COLUMNS)
        profiles = self._profiles.read_row_group(index, columns=PROFILE_COLUMNS).to_pylist()

        if len(batch) != len(profiles):
            raise InputError(
                f"батч {index}: строк {len(batch)} в наборе, {len(profiles)} в анкетах"
            )

        dim = _dim(profiles)

        vectors = _matrix(events, "vector", dim, index)

        expected = sum(len(row["event_time_log"]) for row in batch)

        if vectors.shape[0] != expected:
            raise InputError(
                f"батч {index}: векторов событий {vectors.shape[0]}, а событий у "
                f"клиентов {expected} — файлы собраны в разное время"
            )

        owners = events.column("client_id").to_pylist()
        numbers = events.column("event").to_pylist()

        clients: list[Client] = []
        first = 0

        for row, (here, there) in enumerate(zip(batch, profiles)):

            _agree(index, row, here, there)

            positions = np.asarray(here["event_time_log"], dtype=np.float32)

            n_events = int(positions.size)

            _slice_agrees(index, here["client_id"], owners, numbers, first, n_events)

            if n_events and bool((positions[1:] > positions[:-1]).any()):
                raise InputError(
                    f"батч {index}, клиент {here['client_id']}: временные позиции растут "
                    "к концу ленты и не соответствуют событиям"
                )

            clients.append(
                Client(
                    batch_index=index,
                    client_id=here["client_id"],
                    profile=np.asarray(there["profile"], dtype=np.float32),
                    events=vectors[first:first + n_events],
                    positions=positions,
                )
            )

            first += n_events

        return clients


def _open(path: Path, schema: pa.Schema, command: str) -> pq.ParquetFile:
    """
    Файл этапа по стандартному пути, со сверкой схемы.
    """

    if not path.exists():
        raise InputError(f"нет {path}: выполните {command}")

    handle = pq.ParquetFile(path)

    if not handle.schema_arrow.equals(schema, check_metadata=False):
        raise InputError(f"{path} собран другой схемой: выполните {command} заново")

    return handle


def _dim(profiles: list[dict]) -> int:
    """
    Длина вектора. Её объявляет этап 09 отдельной колонкой; у
    событий она выводится из длины самого вектора.
    """

    dims = {row["dim"] for row in profiles}

    if len(dims) != 1:
        raise InputError(f"длина вектора разная у клиентов анкеты: {sorted(dims)}")

    return int(dims.pop())


def _agree(index: int, row: int, sample: dict, profile: dict) -> None:
    """
    Набор и анкеты говорят про одного и того же клиента.
    """

    if sample["client_id"] != profile["client_id"]:
        raise InputError(
            f"батч {index}, строка {row}: в наборе клиент {sample['client_id']}, "
            f"а в анкетах {profile['client_id']}"
        )

    if profile["batch_index"] != index:
        raise InputError(
            f"батч {index}, строка {row}: в анкетах batch_index {profile['batch_index']}"
        )


def _slice_agrees(index: int, client_id: str, owners: list, numbers: list,
                  first: int, n_events: int) -> None:
    """
    Кусок файла событий принадлежит этому клиенту и идёт по порядку.

    Строки событий лежат подряд по клиентам, но это проверяется, а
    не предполагается: сдвиг на одного клиента дал бы чужую
    историю без единой ошибки.
    """

    for step in range(n_events):

        place = first + step

        if owners[place] != client_id:
            raise InputError(
                f"батч {index}: на месте {place} файла событий клиент "
                f"{owners[place]}, а ожидался {client_id}"
            )

        if numbers[place] != step:
            raise InputError(
                f"батч {index}, клиент {client_id}: событие на месте {place} "
                f"пронумеровано {numbers[place]}, а ожидалось {step}"
            )


def _matrix(table: pa.Table, name: str, dim: int, index: int):
    """
    Списковая колонка как матрица [строк, dim].

    to_pylist здесь не годится: строк десятки тысяч, а в каждой
    сотня чисел. Берётся плоский буфер arrow и переразмечается.
    """

    column = table.column(name).combine_chunks()

    array = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column

    flat = array.values.to_numpy()

    if flat.size != table.num_rows * dim:
        raise InputError(
            f"батч {index}: в колонке {name} {flat.size} чисел вместо "
            f"{table.num_rows} * {dim}"
        )

    return flat.reshape(table.num_rows, dim)


__all__ = [
    "BATCH_COLUMNS",
    "Client",
    "InputError",
    "Source",
]
