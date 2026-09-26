from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset.lineage import lineage_problem
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
# Векторы этапов 10–12 сюда не приходят вовсе: там лежат снимки при
# начальных весах, а модель обязана считать всё сама.
#
# value_ids_source не читается: исходное значение цели уже лежит
# в labels, и второй его копии не нужно.
#
# Заполнитель наружу не выходит: каждый массив клиента обрезан по
# его настоящей длине. Проход из нескольких клиентов (model.pack)
# тоже плоский, без [PAD]: клиенты лежат подряд, границы — в
# cu_seqlens.
#
# labels и reason в модель НЕ подаются. Первое уходит в потери,
# второе в отчёт.
#
# Обучение train читает маску не из файла, а разыгрывает её на
# лету тем же маскером этапа 08 (choose + apply) по
# немаскированным value_ids из 07_batches: так каждая эпоха
# получает свою маску. val, test и отчёт читают 08_masked.
#
# Группа строк 07_batches здесь — только единица хранения и
# чтения. Сколько клиентов модель считает за один проход, решает
# не она, а micro_batches: клиенты идут потоком и собираются по
# бюджету позиций.
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
    "profile_time_log",
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
    profile_time_log: np.ndarray  # давность вехи до cutoff, ноль у [USR] и Attributes

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.size)

    @property
    def n_events(self) -> int:
        return int(self.event_starts.size)

    @property
    def profile_n_tokens(self) -> int:
        return int(self.profile_key_ids.size)

    @property
    def n_targets(self) -> int:
        return int((self.labels != IGNORE).sum())


class Size(NamedTuple):
    """
    Длины клиента без самих массивов.

    Ровно те три числа, которые читает cost: micro_batches по
    Size делит поток так же, как по настоящим клиентам.
    """

    n_tokens: int
    profile_n_tokens: int
    n_events: int


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

    def clients(self) -> Iterator[Client]:
        """
        Клиенты группы по одному, в порядке файла.

        В памяти держится одна группа строк: файл читается по мере
        прохода, а не целиком.
        """

        for index in range(self.count):
            yield from self.batch(index)

    def sizes(self) -> Iterator[Size]:
        """
        Длины клиентов группы в порядке файла.

        Читаются только три целых колонки 07_batches, без масок:
        маскирование значения заменяет, а длины не меняет. Так
        число micro-batch'ей эпохи известно до обучения.
        """

        columns = ["n_tokens", "profile_n_tokens", "n_events"]

        for index in range(self.count):

            table = self._batches.read_row_group(index, columns=columns).to_pydict()

            for tokens, profile, events in zip(*(table[name] for name in columns)):
                yield Size(int(tokens), int(profile), int(events))

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
            profile_time_log=np.asarray(
                batch["profile_time_log"][:profile_tokens], dtype=np.float32
            ),
        )

        _check(client, _bools(batch["event_mask"][:n_events]),
               _bools(batch["target_event_mask"][:n_events]), self.mask_id)

        return client


def cost(client: Client) -> int:
    """
    Во сколько позиций обходится клиент одному проходу модели.

    Каждая позиция проходит хотя бы один трансформер:

      n_tokens          — токены событий, энкодер события;
      profile_n_tokens  — токены анкеты, энкодер анкеты;
      n_events + 1      — позиции истории: по одной на событие и
                          одна на вектор анкеты в слоте [USR].
    """

    return client.n_tokens + client.profile_n_tokens + client.n_events + 1


def micro_batches(clients: Iterable[Client], token_budget: int) -> Iterator[list[Client]]:
    """
    Клиенты подряд, собранные в проходы модели по бюджету.

    Клиент добавляется, пока сумма стоимостей не превысит бюджет;
    следующий, который превысил бы его, открывает новый проход.
    Клиент дороже всего бюджета не теряется: он идёт отдельным
    проходом. Пустых проходов не бывает.
    """

    batch: list[Client] = []
    spent = 0

    for client in clients:

        price = cost(client)

        if batch and spent + price > token_budget:
            yield batch
            batch, spent = [], 0

        batch.append(client)
        spent += price

    if batch:
        yield batch


def _open(path: Path, schema: pa.Schema, command: str) -> pq.ParquetFile:
    """
    Файл этапа по стандартному пути, со сверкой схемы.
    """

    if not path.exists():
        raise InputError(f"нет {path}: выполните {command}")

    handle = pq.ParquetFile(path)

    if not handle.schema_arrow.equals(schema, check_metadata=False):
        raise InputError(f"{path} собран другой схемой: выполните {command} заново")

    # Схема от смысла анкеты не зависит: происхождение каталога
    # сверяется отдельно.
    problem = lineage_problem(path.parent, command)

    if problem is not None:
        raise InputError(problem)

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
    "Size",
    "Source",
    "cost",
    "micro_batches",
]
