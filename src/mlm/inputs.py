from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.batching.build import BATCHES_SCHEMA
from src.batching.settings import BATCHES_FILE, batches_dir
from src.embedding.inputs import CALENDAR_PER_EVENT
from src.masking.apply import apply
from src.masking.build import MASKED_SCHEMA
from src.masking.choose import choose
from src.masking.settings import MASKED_FILE, MaskingConfig, masked_dir
from src.tokenization.specials import MASK, UNK, load_special_tokens


# ============================================================
# ВХОД МОДЕЛИ
# ============================================================
#
# Модель читает два файла и только их:
#
#   data/07_batches  — ключи, номера кусков, границы событий,
#       календарь, временные позиции и анкета;
#   data/08_masked   — значения, которые модели РАЗРЕШЕНО видеть,
#       и метки целей.
#
# Файлы этапов 09-12 сюда не приходят вовсе: там лежат снимки при
# начальных весах, а модель обязана считать всё сама.
#
# value_ids_source не читается: исходное значение цели уже лежит
# в labels, и второй его копии не нужно.
#
# Заполнитель наружу не выходит: каждый массив обрезан по
# настоящей длине своего клиента. Поэтому [PAD] не может попасть
# ни во вход модели, ни в потери — его там просто нет.
#
# labels и reason в модель НЕ подаются. Первое уходит в потери,
# второе в отчёт.
#
# Обучение train читает маску не из файла, а разыгрывает её на
# лету тем же маскером этапа 08 (choose + apply) по
# немаскированным value_ids из 07_batches: так каждая эпоха
# получает свою маску. val, test и отчёт читают 08_masked.
# ============================================================


BATCH_COLUMNS = [
    "batch_index",
    "client_id",
    "n_tokens",
    "n_events",
    "profile_n_tokens",
    "key_ids",
    "positions",
    "event_starts",
    "event_lengths",
    "event_mask",
    "target_event_mask",
    "calendar",
    "event_time_log",
    "event_time",
    "profile_key_ids",
    "profile_value_ids",
    "profile_positions",
]

MASKED_COLUMNS = ["batch_index", "client_id", "value_ids", "labels", "reason"]

# Соглашение PyTorch: позиция вне loss.
IGNORE = -100


class InputError(ValueError):
    """
    Вход модели собрать нельзя.
    """


@dataclass(frozen=True)
class Client:
    """
    Один клиент целиком, без единого заполнителя.
    """

    batch_index: int
    client_id: str

    # --- события, длиной n_tokens ---
    key_ids: np.ndarray
    value_ids: np.ndarray      # видимые модели
    positions: np.ndarray
    labels: np.ndarray         # -100 там, где цели нет
    reason: list[str]

    # --- события, длиной n_events ---
    event_starts: np.ndarray
    event_lengths: np.ndarray
    event_time_log: np.ndarray
    calendar: np.ndarray       # [n_events, 6]

    # Только для отчёта: в модель время события не подаётся, его
    # место занимает event_time_log внутри TimeRoPE.
    event_time: list

    # --- анкета, длиной profile_n_tokens ---
    profile_key_ids: np.ndarray
    profile_value_ids: np.ndarray
    profile_positions: np.ndarray

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.size)

    @property
    def n_events(self) -> int:
        return int(self.event_starts.size)

    @property
    def n_targets(self) -> int:
        return int((self.labels != IGNORE).sum())


class Source:
    """
    Пара файлов группы, открытая один раз.

    masking задан — маска не читается из 08_masked, а
    разыгрывается при чтении батча по этому конфигу. Одинаковый
    конфиг даёт одинаковую маску.
    """

    def __init__(self, group: str, masking: MaskingConfig | None = None):

        self.group = group
        self.masking = masking

        self.batches_path = batches_dir(group) / BATCHES_FILE
        self.masked_path = None if masking is not None else masked_dir(group) / MASKED_FILE

        self._batches = _open(
            self.batches_path, BATCHES_SCHEMA, f"python -m src.batching.run {group}"
        )

        specials = load_special_tokens()

        self.mask_id = specials[MASK]
        self.unknown_id = specials[UNK]

        if masking is not None:
            return

        self._masked = _open(
            self.masked_path, MASKED_SCHEMA, f"python -m src.masking.run {group}"
        )

        if self._batches.num_row_groups != self._masked.num_row_groups:
            raise InputError(
                f"батчей {self._batches.num_row_groups}, а масок "
                f"{self._masked.num_row_groups}: файлы собраны в разное время"
            )

    @property
    def count(self) -> int:
        return self._batches.num_row_groups

    def batch(self, index: int) -> list[Client]:
        """
        Клиенты одного батча, каждый обрезанный по своей длине.
        """

        if index < 0 or index >= self.count:
            raise InputError(
                f"батча {index} нет: в группе {self.count} батчей, "
                f"номера от 0 до {self.count - 1}"
            )

        if self.masking is not None:
            batch = self._batches.read_row_group(
                index, columns=BATCH_COLUMNS + ["value_ids"]
            ).to_pylist()
            masked = [self._mask(index, row) for row in batch]
        else:
            batch = self._batches.read_row_group(index, columns=BATCH_COLUMNS).to_pylist()
            masked = self._masked.read_row_group(index, columns=MASKED_COLUMNS).to_pylist()

        if len(batch) != len(masked):
            raise InputError(
                f"батч {index}: клиентов {len(batch)}, а строк масок {len(masked)}"
            )

        return [
            self._client(index, row, here, there)
            for row, (here, there) in enumerate(zip(batch, masked))
        ]

    def _mask(self, index: int, row: dict) -> dict:
        """
        Строка масок, разыгранная тем же маскером, что и этап 08.
        """

        selection = choose(self.group, row, self.masking)

        masked = apply(
            row["client_id"], row, selection.choices, self.mask_id, self.unknown_id
        )
        masked["batch_index"] = index

        return masked

    def _client(self, index: int, row: int, batch: dict, masked: dict) -> Client:

        if batch["client_id"] != masked["client_id"]:
            raise InputError(
                f"батч {index}, строка {row}: в батчах клиент {batch['client_id']}, "
                f"а в масках {masked['client_id']}"
            )

        for name, value in (("batch_index батчей", batch["batch_index"]),
                            ("batch_index масок", masked["batch_index"])):
            if value != index:
                raise InputError(f"батч {index}, строка {row}: {name} равен {value}")

        n_tokens = int(batch["n_tokens"])
        n_events = int(batch["n_events"])
        profile_tokens = int(batch["profile_n_tokens"])

        client = Client(
            batch_index=index,
            client_id=batch["client_id"],
            key_ids=_ints(batch["key_ids"][:n_tokens]),
            value_ids=_ints(masked["value_ids"][:n_tokens]),
            positions=_ints(batch["positions"][:n_tokens]),
            labels=_ints(masked["labels"][:n_tokens]),
            reason=list(masked["reason"][:n_tokens]),
            event_starts=_ints(batch["event_starts"][:n_events]),
            event_lengths=_ints(batch["event_lengths"][:n_events]),
            event_time_log=np.asarray(
                batch["event_time_log"][:n_events], dtype=np.float32
            ),
            calendar=np.asarray(batch["calendar"], dtype=np.float32).reshape(
                -1, CALENDAR_PER_EVENT
            )[:n_events],
            event_time=list(batch["event_time"][:n_events]),
            profile_key_ids=_ints(batch["profile_key_ids"][:profile_tokens]),
            profile_value_ids=_ints(batch["profile_value_ids"][:profile_tokens]),
            profile_positions=_ints(batch["profile_positions"][:profile_tokens]),
        )

        _check(client, _bools(batch["event_mask"][:n_events]),
               _bools(batch["target_event_mask"][:n_events]), self.mask_id)

        return client


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


def _check(client: Client, event_mask: np.ndarray, target_mask: np.ndarray,
           mask_id: int) -> None:
    """
    Инварианты целей.

    Проверяется то, на чём стоит весь этап: цель обязана быть
    настоящим токеном настоящего допустимого события, и на входе
    вместо неё обязан стоять [MASK]. Иначе модель училась бы
    предсказывать то, что и так видит.
    """

    if not bool(event_mask.all()):
        raise InputError(
            f"{client.client_id}: среди первых {client.n_events} событий есть "
            "заполнитель — маска событий не совпадает с n_events"
        )

    where = np.nonzero(client.labels != IGNORE)[0]

    if where.size == 0:
        return

    if not bool((client.value_ids[where] == mask_id).all()):
        raise InputError(
            f"{client.client_id}: у цели на входе стоит не [MASK] — модель видела бы "
            "то, что должна предсказать"
        )

    owner = np.searchsorted(client.event_starts, where, side="right") - 1

    if not bool(target_mask[owner].all()):
        raise InputError(
            f"{client.client_id}: цель нашлась в событии вне периода целей"
        )

    inside = (where >= client.event_starts[owner]) & (
        where < client.event_starts[owner] + client.event_lengths[owner]
    )

    if not bool(inside.all()):
        raise InputError(f"{client.client_id}: цель не попала ни в одно событие")


def _ints(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _bools(values) -> np.ndarray:
    return np.asarray(values, dtype=bool)


__all__ = [
    "BATCH_COLUMNS",
    "IGNORE",
    "MASKED_COLUMNS",
    "Client",
    "InputError",
    "Source",
]
