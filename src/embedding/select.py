from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


# ============================================================
# ЧТО ПОКАЗАТЬ
# ============================================================
#
# В батче миллионы позиций, и вываливать их в отчёт бессмысленно:
# читать его будет человек, который проверяет вход. Ему хватит
# одного настоящего события, нескольких строк анкеты и по одному
# маркеру и заполнителю.
#
# Отсюда же берётся и список позиций, которые вообще считаются
# слоем: считается ровно показанное, а не весь батч.
#
# Правило отбора записано здесь и ни от чего, кроме самих данных,
# не зависит: никакой client_id руками не вписан. Если в батче
# нет клиента со всеми нужными чертами, отбор так и говорит и
# берёт первого — придумывать пример нельзя.
# ============================================================


# Сколько строк анкеты показать. Первая из них — маркер [USR].
PROFILE_ROWS = 5

# Предел на число значений одного показанного события.
EVENT_VALUES = 12


@dataclass(frozen=True)
class ShownValue:
    """
    Одно значение внутри события: где лежит и что с ним стало.
    """

    start: int
    length: int
    key_id: int
    masked: bool

    @property
    def places(self) -> range:
        return range(self.start, self.start + self.length)


@dataclass(frozen=True)
class Shown:
    """
    Что попадёт в отчёт.
    """

    client: int
    client_id: str
    n_tokens: int
    n_events: int
    profile_n_tokens: int

    event: int
    time: datetime | None
    target: bool

    marker: int
    example: ShownValue
    values: tuple[ShownValue, ...]
    profile: tuple[int, ...]
    pad: int | None
    note: str


def select(
    rows: list[dict],
    key_ids: np.ndarray,
    visible: np.ndarray,
    positions: np.ndarray,
    mask_id: int,
    unknown_id: int,
) -> Shown:
    """
    Клиент, событие, анкета и заполнитель для отчёта.
    """

    width = key_ids.shape[1]

    scored = []

    for client, row in enumerate(rows):

        events = _events(row, key_ids[client], visible[client], positions[client],
                         mask_id)

        rank, event = _best_event(events)

        scored.append((rank, row["n_tokens"] == width, row["n_tokens"],
                       row["client_id"], client, row, events, event))

    rank, crowded, _, _, client, row, events, event = min(scored)

    note = ""

    if rank or crowded:
        note = (
            "в этом батче не нашлось клиента, у которого рядом оказались бы и "
            "заполнитель, и значение из нескольких кусков, и маска; показано "
            "ближайшее к этому"
        )

    values = events.get(event, ())[:EVENT_VALUES]

    n_tokens = int(row["n_tokens"])

    return Shown(
        client=client,
        client_id=row["client_id"],
        n_tokens=n_tokens,
        n_events=int(row["n_events"]),
        profile_n_tokens=int(row["profile_n_tokens"]),
        event=event,
        time=row["event_time"][event],
        target=bool(row["target_event_mask"][event]),
        marker=int(row["event_starts"][event]),
        example=_example(values, visible[client], mask_id, unknown_id),
        values=values,
        profile=tuple(range(min(PROFILE_ROWS, int(row["profile_n_tokens"])))),
        # Заполнитель берётся сразу за последним настоящим
        # токеном. У самого длинного клиента батча его нет вовсе.
        pad=n_tokens if n_tokens < width else None,
        note=note,
    )


def _best_event(events: dict[int, tuple[ShownValue, ...]]) -> tuple[int, int]:
    """
    Какое событие показать и насколько оно удачное.

    Удачное это то, где человек за один экран увидит всё сразу:
    значение из нескольких кусков, читаемое целиком, и рядом
    хотя бы одну маску. Чем меньше ранг, тем ближе к этому.
    """

    def has(values, many: bool, masked: bool | None) -> bool:
        return any(
            (value.length > 1) == many and (masked is None or value.masked == masked)
            for value in values
        )

    ranked = [
        [index for index, values in events.items()
         if has(values, True, False) and has(values, False, True)],
        [index for index, values in events.items() if has(values, True, False)],
        [index for index, values in events.items() if has(values, True, True)],
        list(events),
    ]

    for rank, found in enumerate(ranked):
        if found:
            return rank, min(found)

    return len(ranked), 0


def _example(values: tuple[ShownValue, ...], visible: np.ndarray, mask_id: int,
             unknown_id: int) -> ShownValue:
    """
    Главный пример: значение из одного куска, которое модель
    видит как настоящее значение.

    Ни спрятанное маской, ни незнакомое словарю для первого
    знакомства не годится: на главном экране должно быть видно,
    как складываются ключ и обычное значение.
    """

    plain = [
        value
        for value in values
        if value.length == 1 and int(visible[value.start]) not in (mask_id, unknown_id)
    ]

    return (plain or list(values))[0]


def _events(
    row: dict,
    key_ids: np.ndarray,
    visible: np.ndarray,
    positions: np.ndarray,
    mask_id: int,
) -> dict[int, tuple[ShownValue, ...]]:
    """
    Значения каждого настоящего события клиента.

    Границы значений читаются так же, как их читает маскер: ноль
    в positions открывает значение, а нулевая позиция окна это
    маркер события, и разбор начинается сразу за ней.
    """

    found: dict[int, tuple[ShownValue, ...]] = {}

    for event, start in enumerate(row["event_starts"]):

        if not row["event_mask"][event]:
            continue

        end = start + row["event_lengths"][event]

        spans: list[ShownValue] = []
        opened = -1

        for index in range(start + 1, end):

            if positions[index] != 0:
                continue

            if opened >= 0:
                spans.append(
                    _value(opened, index - opened, key_ids, visible, mask_id)
                )

            opened = index

        if opened >= 0:
            spans.append(_value(opened, end - opened, key_ids, visible, mask_id))

        found[event] = tuple(spans)

    return found


def _value(start: int, length: int, key_ids: np.ndarray, visible: np.ndarray,
           mask_id: int) -> ShownValue:

    return ShownValue(
        start=start,
        length=length,
        key_id=int(key_ids[start]),
        masked=bool(visible[start] == mask_id),
    )


__all__ = [
    "EVENT_VALUES",
    "PROFILE_ROWS",
    "Shown",
    "ShownValue",
    "select",
]
