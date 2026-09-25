from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# ОДИН БАТЧ
# ============================================================
#
# Батч это несколько клиентов, выровненных до общей длины. Осей
# выравнивания три, и ширина каждой это максимум по клиентам
# ЭТОГО батча:
#
#   T  токены событий
#   E  события; календарь при этом плоский, 6 * E чисел
#   P  токены анкеты
#
# Настоящее лежит СЛЕВА, заполнитель справа. Поэтому маска это
# ровно «номер меньше длины», а все смещения примера остаются
# теми же числами и внутри выровненной строки: event_starts не
# пересчитывается, и границы значений по positions читаются как
# раньше.
#
# ВНИМАНИЕ. Токен [PAD] неотличим от маркера события по
# содержимому: у него тоже совпадают key_ids и value_ids, а
# positions равен нулю. Отличить их можно ТОЛЬКО по маске.
# Читать последовательность мимо маски нельзя.
#
# Этап ничего не маскирует и не выбирает: target_event_mask
# переносится как есть, а сам батч ничего не знает о значениях
# внутри событий.
# ============================================================


# Календарь хранится плоско, по шесть чисел на событие: так же,
# как в примере. Совпадение формы это то, что делает обещание
# «настоящая часть равна источнику» буквальным.
CALENDAR_PER_EVENT = 6


class BatchError(ValueError):
    """
    Батч собрать нельзя.
    """


@dataclass(frozen=True)
class Widths:
    """
    Три оси выравнивания одного батча.
    """

    tokens: int
    events: int
    profile_tokens: int


def widths(rows: list[dict]) -> Widths:
    """
    Ширины батча: максимум по каждой оси.

    Батч из одних молчащих клиентов даёт нулевые ширины, и это
    рабочий случай: массивы событий просто пусты.
    """

    return Widths(
        tokens=max((len(row["key_ids"]) for row in rows), default=0),
        events=max((len(row["event_starts"]) for row in rows), default=0),
        profile_tokens=max((len(row["profile_key_ids"]) for row in rows), default=0),
    )


def pack(index: int, rows: list[dict], pad: int) -> list[dict]:
    """
    Строки батча: клиенты, выровненные заполнителем.
    """

    if not rows:
        raise BatchError(f"батч {index}: пустой батч не пишется")

    size = widths(rows)

    packed = [_client(index, row, size, pad) for row in rows]

    for row in packed:
        check(row, size, pad)

    return packed


def _client(index: int, row: dict, size: Widths, pad: int) -> dict:

    n_tokens = len(row["key_ids"])
    n_events = len(row["event_starts"])
    profile_n_tokens = len(row["profile_key_ids"])

    return {
        "batch_index": index,
        "client_id": row["client_id"],

        "n_tokens": n_tokens,
        "n_events": n_events,
        "profile_n_tokens": profile_n_tokens,

        "key_ids": _pad(row["key_ids"], size.tokens, pad),
        "value_ids": _pad(row["value_ids"], size.tokens, pad),
        "positions": _pad(row["positions"], size.tokens, 0),
        "token_mask": _mask(n_tokens, size.tokens),

        # Заполнитель смещения это КОНЕЦ последовательности, а не
        # ноль. Пустое событие в конце безвредно: код, забывший
        # маску, ничего не тронет. Ноль же указывал бы на первый
        # настоящий токен клиента, и тот же код затёр бы его.
        "event_starts": _pad(row["event_starts"], size.events, n_tokens),
        "event_lengths": _pad(row["event_lengths"], size.events, 0),
        # Времени у заполнителя нет: нулевого момента не
        # существует, а настоящая дата рядом с event_mask = False
        # читалась бы как правда.
        "event_time": _pad(row["event_time"], size.events, None),
        # А вот у временной позиции заполнитель именно ноль, и он
        # совпадает со значением настоящего последнего события:
        # позиция это расстояние, у которого ноль осмыслен, и
        # выдумкой он не становится. Отличить заполнитель от
        # настоящего нуля можно ТОЛЬКО по event_mask.
        "event_time_log": _pad(row["event_time_log"], size.events, 0.0),
        "calendar": _pad(
            row["calendar"], size.events * CALENDAR_PER_EVENT, 0.0
        ),
        "event_mask": _mask(n_events, size.events),

        "target_event_mask": _pad(row["target_event_mask"], size.events, False),

        "profile_key_ids": _pad(row["profile_key_ids"], size.profile_tokens, pad),
        "profile_value_ids": _pad(row["profile_value_ids"], size.profile_tokens, pad),
        "profile_positions": _pad(row["profile_positions"], size.profile_tokens, 0),
        # Как у событий: ноль заполнителя совпадает с нулём [USR]
        # и Attributes, и отличить его можно ТОЛЬКО по маске.
        "profile_time_log": _pad(row["profile_time_log"], size.profile_tokens, 0.0),
        "profile_token_mask": _mask(profile_n_tokens, size.profile_tokens),
    }


def _pad(values, width: int, fill) -> list:
    """
    Значения слева, заполнитель справа.
    """

    values = list(values)

    if len(values) > width:
        raise BatchError(f"массив длиной {len(values)} не помещается в ширину {width}")

    return values + [fill] * (width - len(values))


def _mask(length: int, width: int) -> list[bool]:
    return [True] * length + [False] * (width - length)


def check(row: dict, size: Widths, pad: int) -> None:
    """
    Инварианты одной строки батча.

    Проверяется своё: ширины, согласие масок с длинами и то, что
    заполнитель лёг с нужной стороны. Устройство самого примера
    здесь не перепроверяется — его проверил датасет перед тем,
    как записать.
    """

    client = row["client_id"]

    for name, width in (
        ("key_ids", size.tokens),
        ("value_ids", size.tokens),
        ("positions", size.tokens),
        ("token_mask", size.tokens),
        ("event_starts", size.events),
        ("event_lengths", size.events),
        ("event_time", size.events),
        ("event_time_log", size.events),
        ("event_mask", size.events),
        ("target_event_mask", size.events),
        ("profile_key_ids", size.profile_tokens),
        ("profile_value_ids", size.profile_tokens),
        ("profile_positions", size.profile_tokens),
        ("profile_time_log", size.profile_tokens),
        ("profile_token_mask", size.profile_tokens),
    ):
        if len(row[name]) != width:
            raise BatchError(
                f"{client}: {name} длиной {len(row[name])} при ширине батча {width}"
            )

    if len(row["calendar"]) != size.events * CALENDAR_PER_EVENT:
        raise BatchError(
            f"{client}: календарь из {len(row['calendar'])} чисел вместо "
            f"{size.events * CALENDAR_PER_EVENT}"
        )

    for name, length, width in (
        ("token_mask", row["n_tokens"], size.tokens),
        ("event_mask", row["n_events"], size.events),
        ("profile_token_mask", row["profile_n_tokens"], size.profile_tokens),
    ):
        if row[name] != [True] * length + [False] * (width - length):
            raise BatchError(f"{client}: {name} не совпадает с длиной {length}")

    # У последнего настоящего события временная позиция обязана
    # быть нулём: она и есть точка отсчёта. Если ноль уехал,
    # значит события выше по конвейеру переставили или обрезали,
    # а позиции перенесли как есть, вместо того чтобы пересчитать.
    if row["n_events"] and row["event_time_log"][row["n_events"] - 1] != 0.0:
        raise BatchError(
            f"{client}: у последнего настоящего события позиция "
            f"{row['event_time_log'][row['n_events'] - 1]!r}, а не ноль: "
            "временные позиции не соответствуют событиям"
        )

    # [USR] это якорь анкеты: его время — сам cutoff, то есть ноль.
    if row["profile_n_tokens"] and row["profile_time_log"][0] != 0.0:
        raise BatchError(
            f"{client}: у маркера анкеты временная позиция "
            f"{row['profile_time_log'][0]!r}, а не ноль"
        )

    # Заполнитель лёг справа, а не поверх настоящих токенов.
    for name, length in (
        ("key_ids", row["n_tokens"]),
        ("value_ids", row["n_tokens"]),
        ("profile_key_ids", row["profile_n_tokens"]),
        ("profile_value_ids", row["profile_n_tokens"]),
    ):
        values = row[name]

        if pad in values[:length]:
            raise BatchError(
                f"{client}: [PAD] встретился среди настоящих токенов {name}"
            )

        if any(value != pad for value in values[length:]):
            raise BatchError(f"{client}: хвост {name} заполнен не только [PAD]")


__all__ = [
    "CALENDAR_PER_EVENT",
    "BatchError",
    "Widths",
    "check",
    "pack",
    "widths",
]
