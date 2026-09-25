from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.dataset.lineage import lineage_problem
from src.batching.build import BATCHES_SCHEMA
from src.batching.settings import BATCHES_FILE, batches_dir
from src.masking.build import MASKED_SCHEMA
from src.masking.settings import MASKED_FILE, masked_dir
from src.tokenization.specials import EVT, PAD, USR, load_special_tokens


# ============================================================
# ВХОД МОДЕЛИ ИЗ ДВУХ ФАЙЛОВ
# ============================================================
#
# Батч собирается из двух файлов, лежащих рядом:
#
#   data/07_batches/<group>/batches.parquet  — ключи, номера
#       кусков, анкета, маски и границы событий;
#   data/08_masked/<group>/masked.parquet    — значения, которые
#       модели РАЗРЕШЕНО видеть.
#
# Строка i одного файла это строка i другого, и батч это одна и
# та же группа строк в обоих. Но верить этому на слово нельзя:
# файлы собираются разными командами и могут разъехаться, если
# один пересобрали, а другой нет. Поэтому до объединения
# сверяются batch_index, client_id и равенство value_ids_source
# исходным value_ids батча.
#
# labels сюда не приходят вовсе: из масок читаются ровно четыре
# колонки, и обещание «в эмбеддинг подаются только видимые
# значения» видно по списку колонок, а не по доверию к коду ниже.
#
# Календарь и event_time_log тоже не читаются: они лежат в
# batches.parquet, но в списке колонок их нет. Это вход
# эмбеддингов, а не энкодера события.
#
# Энкодер события читает тот же файл и тем же кодом, но календарь
# ему нужен. Поэтому у читателя есть один флаг: просить колонку
# или нет. Обещание «эмбеддинги календаря не видят» от этого не
# слабеет — его держит значение флага, а не отсутствие кода.
# ============================================================


# Что берётся из батчей. Календаря и временных позиций здесь
# намеренно нет: их просят отдельно.
BATCH_COLUMNS = [
    "batch_index",
    "client_id",
    "n_tokens",
    "n_events",
    "profile_n_tokens",
    "key_ids",
    "value_ids",
    "positions",
    "token_mask",
    "event_starts",
    "event_lengths",
    "event_time",
    "event_mask",
    "target_event_mask",
    "profile_key_ids",
    "profile_value_ids",
    "profile_positions",
    "profile_token_mask",
]

# Что берётся из масок. labels в списке нет.
MASKED_COLUMNS = [
    "batch_index",
    "client_id",
    "value_ids_source",
    "value_ids",
]

# Шесть чисел на событие. Нужны энкодеру события, не входу.
CALENDAR_COLUMN = "calendar"

CALENDAR_PER_EVENT = 6

# Устройство последовательности: границы событий и их время.
# Слой их не видит — по ним собирают события те, кто идёт дальше.
STRUCTURE_COLUMNS = [
    "client_id",
    "n_tokens",
    "n_events",
    "profile_n_tokens",
    "event_starts",
    "event_lengths",
    "event_time",
    "event_mask",
    "target_event_mask",
]


class InputError(ValueError):
    """
    Вход модели собрать нельзя.
    """


@dataclass(frozen=True)
class BatchInput:
    """
    Ровно то, что видит слой эмбеддингов.

    Значения уже видимые: часть из них заменена маскером на
    [MASK] или [UNK]. Исходных значений и labels здесь нет.
    """

    batch_index: int
    client_ids: tuple[str, ...]

    # события, [B, T]
    key_ids: torch.Tensor
    value_ids: torch.Tensor
    positions: torch.Tensor
    token_mask: torch.Tensor

    # анкета, [B, P]
    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_positions: torch.Tensor
    profile_token_mask: torch.Tensor

    @property
    def clients(self) -> int:
        return int(self.key_ids.shape[0])

    @property
    def width(self) -> int:
        return int(self.key_ids.shape[1])

    @property
    def profile_width(self) -> int:
        return int(self.profile_key_ids.shape[1])


@dataclass(frozen=True)
class Loaded:
    """
    Батч целиком: вход слоя, материал отчёта и итоги сверки.
    """

    model: BatchInput
    rows: list[dict]

    # [B, E * 6], и только если календарь просили. Иначе None:
    # пустая матрица выглядела бы как «календарь есть, просто
    # нулевой».
    calendar: np.ndarray | None = None


class Source:
    """
    Пара файлов группы, открытая один раз.

    Батчей в группе бывает много, а заголовки parquet и сверка
    схем одни на файл: открывать их заново на каждый батч значило
    бы перечитывать одно и то же.
    """

    def __init__(self, group: str, with_calendar: bool = False):

        self.group = group
        self.with_calendar = with_calendar

        self.columns = list(BATCH_COLUMNS)

        if with_calendar:
            self.columns.append(CALENDAR_COLUMN)

        self.batches_path = batches_dir(group) / BATCHES_FILE
        self.masked_path = masked_dir(group) / MASKED_FILE

        self._batches = _open(
            self.batches_path, BATCHES_SCHEMA, f"python -m src.batching.run {group}"
        )
        self._masked = _open(
            self.masked_path, MASKED_SCHEMA, f"python -m src.masking.run {group}"
        )

        if self._batches.num_row_groups != self._masked.num_row_groups:
            raise InputError(
                f"батчей {self._batches.num_row_groups}, а масок "
                f"{self._masked.num_row_groups}: файлы собраны в разное время, "
                f"выполните python -m src.masking.run {group}"
            )

    @property
    def count(self) -> int:
        return self._batches.num_row_groups

    def batch(self, index: int) -> Loaded:
        """
        Один батч из двух файлов, со сверкой.
        """

        if index < 0 or index >= self.count:
            raise InputError(
                f"батча {index} нет: в группе {self.count} батчей, "
                f"номера от 0 до {self.count - 1}"
            )

        batch = self._batches.read_row_group(index, columns=self.columns)
        mask = self._masked.read_row_group(index, columns=MASKED_COLUMNS)

        if batch.num_rows == 0:
            raise InputError(f"батч {index} пуст: клиентов в нём нет")

        if batch.num_rows != mask.num_rows:
            raise InputError(
                f"батч {index}: клиентов {batch.num_rows}, а строк масок {mask.num_rows}"
            )

        arrays = _arrays(batch, mask)

        _check(index, arrays, load_special_tokens())

        model = BatchInput(
            batch_index=index,
            client_ids=tuple(arrays["client_id"]),
            key_ids=_tensor(arrays["key_ids"], torch.int64),
            value_ids=_tensor(arrays["visible_value_ids"], torch.int64),
            positions=_tensor(arrays["positions"], torch.int64),
            token_mask=_tensor(arrays["token_mask"], torch.bool),
            profile_key_ids=_tensor(arrays["profile_key_ids"], torch.int64),
            profile_value_ids=_tensor(arrays["profile_value_ids"], torch.int64),
            profile_positions=_tensor(arrays["profile_positions"], torch.int64),
            profile_token_mask=_tensor(arrays["profile_token_mask"], torch.bool),
        )

        calendar = None

        if self.with_calendar:
            calendar = _matrix(
                batch, CALENDAR_COLUMN, _width(batch, CALENDAR_COLUMN), np.float32
            )

            events = _width(batch, "event_mask")

            if calendar.shape[1] != events * CALENDAR_PER_EVENT:
                raise InputError(
                    f"батч {index}: календарь из {calendar.shape[1]} чисел вместо "
                    f"{events * CALENDAR_PER_EVENT} — не по шесть на событие"
                )

        return Loaded(
            model=model,
            rows=batch.select(STRUCTURE_COLUMNS).to_pylist(),
            calendar=calendar,
        )


def load_batch(group: str, index: int) -> Loaded:
    """
    Один батч группы, когда остальные не нужны.
    """

    return Source(group).batch(index)


def _open(path: Path, schema: pa.Schema, command: str) -> pq.ParquetFile:
    """
    Файл этапа по стандартному пути, со сверкой схемы.
    """

    if not path.exists():
        raise InputError(f"нет {path}: выполните {command}")

    handle = pq.ParquetFile(path)

    # Схема сверяется с той, которой этап пишет сейчас. Это ловит
    # файл, собранный до смены формата: иначе несовпадение
    # всплыло бы посреди разбора, уже без имени виноватого этапа.
    if not handle.schema_arrow.equals(schema, check_metadata=False):
        raise InputError(f"{path} собран другой схемой: выполните {command} заново")

    # Схема от смысла анкеты не зависит: происхождение каталога
    # сверяется отдельно.
    problem = lineage_problem(path.parent, command)

    if problem is not None:
        raise InputError(problem)

    return handle


def _arrays(batch: pa.Table, mask: pa.Table) -> dict:
    """
    Колонки обоих файлов как numpy-матрицы [B, ширина].
    """

    width = _width(batch, "key_ids")
    profile_width = _width(batch, "profile_key_ids")

    return {
        "batch_index": batch.column("batch_index").to_pylist(),
        "client_id": batch.column("client_id").to_pylist(),
        "n_tokens": np.asarray(batch.column("n_tokens").to_pylist(), dtype=np.int64),
        "profile_n_tokens": np.asarray(
            batch.column("profile_n_tokens").to_pylist(), dtype=np.int64
        ),

        "key_ids": _matrix(batch, "key_ids", width, np.int64),
        "batch_value_ids": _matrix(batch, "value_ids", width, np.int64),
        "positions": _matrix(batch, "positions", width, np.int64),
        "token_mask": _matrix(batch, "token_mask", width, np.bool_),

        "profile_key_ids": _matrix(batch, "profile_key_ids", profile_width, np.int64),
        "profile_value_ids": _matrix(batch, "profile_value_ids", profile_width, np.int64),
        "profile_positions": _matrix(batch, "profile_positions", profile_width, np.int64),
        "profile_token_mask": _matrix(
            batch, "profile_token_mask", profile_width, np.bool_
        ),

        "masked_batch_index": mask.column("batch_index").to_pylist(),
        "masked_client_id": mask.column("client_id").to_pylist(),
        "source_value_ids": _matrix(mask, "value_ids_source", width, np.int64),
        "visible_value_ids": _matrix(mask, "value_ids", width, np.int64),
    }


def _width(table: pa.Table, name: str) -> int:
    """
    Ширина колонки списков по первой строке.

    Равенство длин между строками батчер уже проверил при записи,
    а здесь его подтверждает общее число элементов.
    """

    return len(table.column(name)[0])


def _matrix(table: pa.Table, name: str, width: int, dtype) -> np.ndarray:

    column = table.column(name).combine_chunks()

    if isinstance(column, pa.ChunkedArray):
        column = column.chunk(0)

    flat = column.flatten().to_numpy(zero_copy_only=False)

    if flat.size != table.num_rows * width:
        raise InputError(
            f"колонка {name}: {flat.size} значений вместо "
            f"{table.num_rows} * {width} — строки разной длины"
        )

    return flat.astype(dtype, copy=False).reshape(table.num_rows, width)


def _tensor(values: np.ndarray, dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values)).to(dtype)


def _check(index: int, arrays: dict, specials: dict) -> None:
    """
    Пять сверок до объединения. Любая несовпавшая — отказ;
    считать тут нечего, наружу идёт только согласие файлов.
    """

    pad = specials[PAD]
    markers = (specials[EVT], specials[USR])

    clients, width = arrays["key_ids"].shape
    profile_width = arrays["profile_key_ids"].shape[1]

    # 1. Один и тот же батч.
    for number in arrays["batch_index"] + arrays["masked_batch_index"]:
        if number != index:
            raise InputError(
                f"батч {index}: в строке записан batch_index {number} — "
                "файлы разъехались, пересоберите маски"
            )

    # 2. Один и тот же клиент, в том же порядке.
    pairs = zip(arrays["client_id"], arrays["masked_client_id"])

    for row, (left, right) in enumerate(pairs):
        if left != right:
            raise InputError(
                f"батч {index}, строка {row}: в батчах клиент {left}, а в масках {right}"
            )

    # 3. Маскер начинал с ТЕХ ЖЕ значений, что лежат в батче.
    difference = arrays["source_value_ids"] != arrays["batch_value_ids"]

    if difference.any():
        row, position = (int(number) for number in np.argwhere(difference)[0])
        client = arrays["client_id"][row]
        raise InputError(
            f"батч {index}, клиент {client}, позиция {position}: "
            f"value_ids_source {arrays['source_value_ids'][row, position]} не совпадает "
            f"с value_ids батча {arrays['batch_value_ids'][row, position]} — "
            "маски сняты с другого батча"
        )

    # 4. У маркера один ID в обоих слотах, и маскер его не трогал.
    marker = np.isin(arrays["key_ids"], markers)

    wrong = marker & (arrays["visible_value_ids"] != arrays["key_ids"])

    if wrong.any():
        row, position = (int(number) for number in np.argwhere(wrong)[0])
        client = arrays["client_id"][row]
        raise InputError(
            f"батч {index}, клиент {client}, позиция {position}: "
            f"маркер {arrays['key_ids'][row, position]} в слоте ключа, но "
            f"{arrays['visible_value_ids'][row, position]} в слоте значения"
        )

    # 5. Заполнитель ровно там, где его обещает маска.
    _check_padding(
        index,
        arrays["client_id"],
        arrays["n_tokens"],
        width,
        pad,
        "",
        arrays["key_ids"],
        arrays["visible_value_ids"],
        arrays["token_mask"],
    )

    _check_padding(
        index,
        arrays["client_id"],
        arrays["profile_n_tokens"],
        profile_width,
        pad,
        "анкеты ",
        arrays["profile_key_ids"],
        arrays["profile_value_ids"],
        arrays["profile_token_mask"],
    )



def _check_padding(
    index: int,
    client_ids: list,
    lengths: np.ndarray,
    width: int,
    pad: int,
    what: str,
    key_ids: np.ndarray,
    value_ids: np.ndarray,
    token_mask: np.ndarray,
) -> None:
    """
    Маска обязана совпадать с тем, где лежит [PAD].

    Слой зануляет выход по маске, поэтому расхождение маски и
    заполнителя означало бы ненулевой вектор у пустого места.
    """

    numbers = np.arange(width, dtype=np.int64)[None, :]

    expected = numbers < lengths[:, None]

    if not np.array_equal(token_mask, expected):
        row = int(np.argwhere(token_mask != expected)[0][0])
        raise InputError(
            f"батч {index}, клиент {client_ids[row]}: маска {what}не совпадает "
            f"с длиной {int(lengths[row])}"
        )

    tail = ~expected

    for name, values in (("ключей", key_ids), ("значений", value_ids)):

        if not bool((values[tail] == pad).all()):
            row = int(np.argwhere(tail & (values != pad))[0][0])
            raise InputError(
                f"батч {index}, клиент {client_ids[row]}: в хвосте {what}{name} "
                "лежит не только [PAD]"
            )

        if bool((values[expected] == pad).any()):
            row = int(np.argwhere(expected & (values == pad))[0][0])
            raise InputError(
                f"батч {index}, клиент {client_ids[row]}: [PAD] встретился среди "
                f"настоящих {what}{name}"
            )


__all__ = [
    "BATCH_COLUMNS",
    "CALENDAR_COLUMN",
    "CALENDAR_PER_EVENT",
    "MASKED_COLUMNS",
    "STRUCTURE_COLUMNS",
    "BatchInput",
    "InputError",
    "Loaded",
    "Source",
    "load_batch",
]
