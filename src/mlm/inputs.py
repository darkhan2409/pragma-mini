from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, NamedTuple

import numpy as np
import pyarrow.compute as pc
from torch.utils.data import IterableDataset

from src.dataset.settings import dataset_dir
from src.embedding.inputs import CALENDAR_PER_EVENT
from src.masking.apply import apply
from src.masking.choose import choose
from src.masking.settings import MaskingConfig
from src.masking.weights import ValueWeights, WeightsError, load_value_weights
from src.temporal.position import TemporalError
from src.temporal.samples import SamplesError, TemporalGroup
from src.tokenization.specials import MASK, UNK, load_special_tokens


# ============================================================
# ВХОД МОДЕЛИ
# ============================================================
#
# На диске у модели один вход — собранная группа:
#
#   data/05_dataset/<group>/samples.parquet
#
# Остальное считается при чтении, по строке клиента:
#
#   временные позиции  — src.temporal.samples.TemporalGroup, по точке
#       отсчёта из meta.json набора;
#   маска              — маскер (choose + apply) по конфигу masking:
#       какие значения модели РАЗРЕШЕНО видеть и какие метки у целей.
#
# Розыгрыш маски ключуется seed, группой и клиентом, поэтому val и
# test с одним конфигом получают одну и ту же маску при каждом
# чтении, а train каждую эпоху свою (seed эпохи, src.mlm.train).
#
# Векторы этапов 08–10 сюда не приходят вовсе: там лежат снимки при
# начальных весах, а модель обязана считать всё сама.
#
# Заполнителя нет нигде: пример клиента читается как есть. Проход из
# нескольких клиентов (model.pack) тоже плоский: клиенты лежат
# подряд, границы — в cu_seqlens.
#
# labels и reason в модель НЕ подаются. Первое уходит в потери,
# второе в отчёт.
#
# Группа строк набора здесь — только единица чтения. Сколько
# клиентов модель считает за один проход, решает не она, а
# micro_batches: клиенты идут потоком и собираются по бюджету
# позиций.
# ============================================================


INPUT_COLUMNS = [
    "client_id",
    "key_ids",
    "value_ids",
    "positions",
    "event_starts",
    "event_lengths",
    "target_event_mask",
    "calendar",
    "event_time_log",
    "event_time",
    "profile_key_ids",
    "profile_value_ids",
    "profile_positions",
    "profile_time_log",
]

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

    # Номер группы строк набора, из которой прочитан клиент.
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
    Вход модели одной группы, открытый один раз.

    masking — по какому конфигу разыгрывается маска; без него
    MaskingConfig(). Одинаковый конфиг даёт одинаковую маску.
    """

    def __init__(self, group: str, masking: MaskingConfig | None = None):

        self.group = group
        self.masking = masking if masking is not None else MaskingConfig()

        self.directory = dataset_dir(group)

        specials = load_special_tokens()

        self.mask_id = specials[MASK]
        self.unknown_id = specials[UNK]

        self.weights = _weights(self.masking)

        self._open()

        # Порядок групп строк прохода: по умолчанию — порядок файла.
        self.order = list(range(self.count))

    def shuffle(self, seed: int) -> None:
        """
        Проход по группам строк в перестановке, заданной seed.

        Клиенты внутри группы и их маска не меняются: маска
        разыгрывается по строке, а не по месту в проходе.
        """

        self.order = np.random.default_rng(seed).permutation(self.count).tolist()

    def _open(self) -> None:

        try:
            self._samples = TemporalGroup(self.group, self.directory)
        except SamplesError as error:
            raise InputError(str(error)) from error

    # Источник переезжает в процесс подготовки данных (Prefetch)
    # путём, номерами токенов и готовыми весами маски, а не
    # глобалами каталогов: новый процесс импортирует settings заново
    # и о подменённых каталогах (тесты, --out) не знает. Файл набора
    # открывается там же заново.
    def __getstate__(self) -> dict:
        return {
            name: getattr(self, name)
            for name in ("group", "masking", "weights", "directory", "mask_id", "unknown_id", "order")
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open()

    @property
    def count(self) -> int:
        return self._samples.count

    def batch(self, index: int) -> list[Client]:
        """
        Клиенты одной группы строк набора.
        """

        if index < 0 or index >= self.count:
            raise InputError(
                f"группы строк {index} нет: в группе {self.group} их {self.count}, "
                f"номера от 0 до {self.count - 1}"
            )

        try:
            rows = self._samples.row_group(index, columns=INPUT_COLUMNS).to_pylist()
        except TemporalError as error:
            raise InputError(str(error)) from error

        return [self._client(index, row, self._mask(row)) for row in rows]

    def clients(self) -> Iterator[Client]:
        """
        Клиенты группы по одному, в порядке прохода (order).

        В памяти держится одна группа строк: файл читается по мере
        прохода, а не целиком.
        """

        for index in self.order:
            yield from self.batch(index)

    def sizes(self) -> Iterator[Size]:
        """
        Длины клиентов группы в порядке прохода (order).

        Читаются только три колонки-списка, без времени и масок:
        маскирование значения заменяет, а длины не меняет. Так
        число micro-batch'ей эпохи известно до обучения.
        """

        columns = ["key_ids", "profile_key_ids", "event_starts"]

        for index in self.order:

            table = self._samples.row_group(index, columns=columns)

            lengths = [pc.list_value_length(table.column(name)).to_pylist() for name in columns]

            for tokens, profile, events in zip(*lengths):
                yield Size(int(tokens), int(profile), int(events))

    def _mask(self, row: dict) -> dict:
        """
        Маска клиента, разыгранная при чтении.
        """

        selection = choose(self.group, row, self.masking, self.weights)

        return apply(
            row["client_id"], row, selection.choices, self.mask_id, self.unknown_id,
            selection.corrupted,
        )

    def _client(self, index: int, row: dict, masked: dict) -> Client:

        client = Client(
            batch_index=index,
            client_id=row["client_id"],
            key_ids=_ints(row["key_ids"]),
            value_ids=_ints(masked["value_ids"]),
            positions=_ints(row["positions"]),
            labels=_ints(masked["labels"]),
            reason=list(masked["reason"]),
            event_starts=_ints(row["event_starts"]),
            event_lengths=_ints(row["event_lengths"]),
            event_time_log=np.asarray(row["event_time_log"], dtype=np.float32),
            calendar=np.asarray(row["calendar"], dtype=np.float32).reshape(
                -1, CALENDAR_PER_EVENT
            ),
            event_time=list(row["event_time"]),
            profile_key_ids=_ints(row["profile_key_ids"]),
            profile_value_ids=_ints(row["profile_value_ids"]),
            profile_positions=_ints(row["profile_positions"]),
            profile_time_log=np.asarray(row["profile_time_log"], dtype=np.float32),
        )

        _check(client, _bools(row["target_event_mask"]), self.mask_id)

        return client


def _weights(masking: MaskingConfig) -> ValueWeights | None:
    """
    Веса value-маскирования словаря, если маска их просит.

    Грузятся один раз на источник и едут в процессы подготовки
    вместе с ним: процесс не знает подменённых каталогов словаря.
    """

    if not masking.informativeness_weighted_masking:
        return None

    try:
        return load_value_weights()
    except WeightsError as error:
        raise InputError(str(error)) from error


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


class Prefetch:
    """
    Клиенты источника, подготовленные заранее в workers отдельных
    процессах; workers=0 — в этом же процессе.

    Без неё группа строк (32 клиента) читалась, считала время и
    маскировалась на
    Python в главном потоке, пока GPU стоял: около трети эпохи. В
    отдельном процессе подготовка следующих групп идёт, пока модель
    считает текущие, и GIL обучения она не занимает.

    Порядок клиентов тот же, что у source.clients(): процесс i берёт
    группы строк order[i], order[i + n], …, а DataLoader отдаёт их по
    кругу.
    Маска разыгрывается тем же кодом из той же строки, поэтому она
    побитно та же.
    """

    # Сколько групп строк на процесс готовится впрок.
    AHEAD = 2

    def __init__(self, source: Source, workers: int):
        self.source = source
        self.workers = workers

    def clients(self) -> Iterator[Client]:

        if self.workers == 0:
            yield from self.source.clients()
            return

        import torch
        from torch.utils.data import DataLoader

        # Свой генератор: иначе DataLoader взял бы seed процессов из
        # глобального и сдвинул поток dropout обучения — прогон с
        # подготовкой впрок разошёлся бы с прогоном без неё.
        loader = DataLoader(
            _RowGroups(self.source),
            batch_size=None,
            num_workers=self.workers,
            prefetch_factor=self.AHEAD,
            collate_fn=_as_is,
            generator=torch.Generator(),
        )

        for batch in loader:
            yield from batch


class _RowGroups(IterableDataset):
    """
    Группы строк источника: одна группа — список её клиентов.
    """

    def __init__(self, source: Source):
        self.source = source

    def __iter__(self) -> Iterator[list[Client]]:

        from torch.utils.data import get_worker_info

        info = get_worker_info()

        first, step = (info.id, info.num_workers) if info is not None else (0, 1)

        for index in self.source.order[first::step]:
            yield self.source.batch(index)


def _as_is(batch):
    """
    Группа строк без переделки: клиенты едут как есть, без
    превращения массивов в тензоры.
    """

    return batch


def _check(client: Client, target_mask: np.ndarray, mask_id: int) -> None:
    """
    Инварианты целей.

    Проверяется то, на чём стоит весь этап: цель обязана быть
    токеном допустимого события, и на входе
    вместо неё обязан стоять [MASK]. Иначе модель училась бы
    предсказывать то, что и так видит.
    """

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
    "IGNORE",
    "INPUT_COLUMNS",
    "Prefetch",
    "Client",
    "InputError",
    "Size",
    "Source",
    "cost",
    "micro_batches",
]
