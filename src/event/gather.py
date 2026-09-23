from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.embedding.inputs import CALENDAR_PER_EVENT, Loaded


# ============================================================
# ОТБОР НАСТОЯЩИХ СОБЫТИЙ
# ============================================================
#
# Батч выровнен заполнителем, и в val места под события заняты им
# на 83 %. Считать их бессмысленно: у пустого слота длина ноль,
# то есть строка без единого настоящего токена — внимание по
# пустому множеству. Фильтр по event_mask это условие
# корректности, а не оптимизация.
#
# Настоящие события собираются в плоский список и обходятся
# порциями. Ширину порции задаёт её собственный максимум длины, а
# не максимум по всему батчу.
#
# Порядок обхода — ПО ДЛИНЕ. Это безопасно ровно потому, что
# события изолированы: вектор события не зависит ни от соседей,
# ни от того, в какой порции он оказался. Ради этого весь этап и
# существует, и проверкой служит равенство результата с обходом
# без сортировки.
#
# Выигрыш измерен на val при порции 1024: заполнено токенами
# 90.2 % против 25.3 %, работа внимания 4.8 млн против 33.3 млн
# при идеальных 3.0 млн. Внимание квадратично по длине, поэтому
# лишний заполнитель стоит дороже, чем кажется.
#
# Границы значений здесь не разбираются: событие берётся целиком,
# как есть. Что у него внутри — дело энкодера.
# ============================================================


class GatherError(ValueError):
    """
    Настоящие события собрать нельзя.
    """


@dataclass(frozen=True)
class Events:
    """
    Плоский список настоящих событий батча.

    Нумерация своя, сквозная по батчу; клиент и номер события
    внутри клиента сохранены, чтобы вернуть результат на место.
    """

    client: np.ndarray   # [n] строка клиента в батче
    slot: np.ndarray     # [n] номер события внутри клиента
    start: np.ndarray    # [n] первый токен события, он же маркер
    length: np.ndarray   # [n] сколько токенов, считая маркер

    @property
    def count(self) -> int:
        return int(self.client.size)

    @property
    def tokens(self) -> int:
        return int(self.length.sum())


@dataclass(frozen=True)
class Chunk:
    """
    Одна порция событий, готовая к выборке токенов.
    """

    where: np.ndarray    # [n] места этих событий в плоском списке
    client: np.ndarray   # [n]
    slot: np.ndarray     # [n]
    column: np.ndarray   # [n, L] номера токенов в строке клиента
    pad: np.ndarray      # [n, L] True там, где токена нет

    @property
    def count(self) -> int:
        return int(self.where.size)

    @property
    def width(self) -> int:
        return int(self.column.shape[1])


def gather(loaded: Loaded) -> Events:
    """
    Настоящие события батча одним списком.
    """

    rows = loaded.rows

    event_mask = np.asarray([row["event_mask"] for row in rows], dtype=bool)
    starts = np.asarray([row["event_starts"] for row in rows], dtype=np.int64)
    lengths = np.asarray([row["event_lengths"] for row in rows], dtype=np.int64)
    n_tokens = np.asarray([row["n_tokens"] for row in rows], dtype=np.int64)

    client, slot = np.nonzero(event_mask)

    events = Events(
        client=client.astype(np.int64),
        slot=slot.astype(np.int64),
        start=starts[client, slot],
        length=lengths[client, slot],
    )

    _check(events, n_tokens)

    return events


def chunks(events: Events, size: int, sort: bool = True) -> Iterator[Chunk]:
    """
    Порции событий, от коротких к длинным.

    sort=False оставлен не для настройки, а для проверки: обход в
    порядке батча обязан дать те же векторы. Если не даст —
    внимание где-то протекло между событиями.
    """

    if size < 1:
        raise GatherError(f"размер порции обязан быть положительным, получено {size}")

    # Устойчивая сортировка: при равной длине порядок остаётся
    # порядком батча, и результат не зависит от того, как numpy
    # разложил равные ключи.
    order = (
        np.argsort(events.length, kind="stable")
        if sort
        else np.arange(events.count, dtype=np.int64)
    )

    for first in range(0, order.size, size):

        where = order[first:first + size]

        length = events.length[where]

        numbers = np.arange(int(length.max()), dtype=np.int64)[None, :]

        pad = numbers >= length[:, None]

        start = events.start[where][:, None]

        # Заполнитель адресуется на собственный маркер события:
        # такой номер заведомо существует, а значение всё равно
        # закрыто маской. Так не нужен ни обрез по ширине строки,
        # ни отдельная ветка.
        yield Chunk(
            where=where,
            client=events.client[where],
            slot=events.slot[where],
            column=np.where(pad, start, start + numbers),
            pad=pad,
        )


def calendar_of(calendar: np.ndarray, chunk: Chunk) -> np.ndarray:
    """
    Шесть чисел календаря на каждое событие порции: [n, 6].

    Календарь лежит плоско, по шесть чисел на событие, в том же
    порядке, что и сами события.
    """

    offset = chunk.slot[:, None] * CALENDAR_PER_EVENT

    return calendar[
        chunk.client[:, None], offset + np.arange(CALENDAR_PER_EVENT, dtype=np.int64)
    ]


def _check(events: Events, n_tokens: np.ndarray) -> None:
    """
    Инварианты отобранного.

    Их держат датасет и батчер, но проверяются они здесь: дальше
    начинается арифметика с номерами токенов, и сломанная граница
    дала бы не ошибку, а тихо неверный вектор.

    Про маску токенов отдельной проверки нет и не нужно: читатель
    уже убедился, что token_mask истинна ровно на [0, n_tokens).
    Значит «событие внутри настоящих токенов» и означает «все его
    токены настоящие».
    """

    if events.count == 0:
        return

    if int(events.length.min()) < 1:
        raise GatherError(
            "среди настоящих событий нашлось пустое: у события обязан быть хотя бы "
            "маркер [EVT]"
        )

    over = events.start + events.length > n_tokens[events.client]

    if bool(over.any()):
        first = int(np.argmax(over))
        raise GatherError(
            f"событие {events.slot[first]} клиента в строке {events.client[first]} "
            f"занимает токены [{events.start[first]}, "
            f"{events.start[first] + events.length[first]}), а настоящих у него "
            f"{n_tokens[events.client[first]]}"
        )


__all__ = [
    "Chunk",
    "Events",
    "GatherError",
    "calendar_of",
    "chunks",
    "gather",
]
