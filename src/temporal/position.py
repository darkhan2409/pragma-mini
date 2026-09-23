from __future__ import annotations

import math
from datetime import datetime

import numpy as np


# ============================================================
# ВРЕМЕННАЯ ПОЗИЦИЯ СОБЫТИЯ
# ============================================================
#
# Расстояние в секундах до ПОСЛЕДНЕГО выбранного события
# клиента, сжатое логарифмом:
#
#   event_time_log[i] = 8 * log1p(delta_i / 8)
#
# У последнего события позиция ровно 0, у более старых она тем
# больше, чем они древнее. Пустая история даёт пустой массив.
#
# Масштаб 8 секунд взят из статьи вслед за референсом
# pragmatiq/data/tokenizer.py. При малых промежутках
# 8*log1p(x/8) почти равно x, поэтому секунды сохраняются
# линейно, а месяцы сжимаются в логарифм: в банковской ленте
# события идут то секундами, то месяцами, и без сжатия дальний
# хвост забил бы шкалу целиком.
#
# Это НЕ календарь: календарь говорит, когда по часам произошло
# событие, а здесь лежит только расстояние между событиями. И не
# positions: там номер куска внутри значения. Три разных канала,
# и смешивать их нельзя.
#
# Точка отсчёта это последнее ВЫБРАННОЕ событие. Отбор сделал
# датасет, и если он отбросил хвост истории, отсчёт идёт от
# нового хвоста, а не от события, которого в примере нет.
# ============================================================


# Масштаб сжатия времени, в секундах.
TIME_SCALE = 8.0

SECOND_US = 1_000_000


class TemporalError(ValueError):
    """
    Временные позиции посчитать нельзя.
    """


def time_log(client_id: str, event_time: list[datetime]) -> list[float]:
    """
    Временные позиции всех событий примера.
    """

    if not event_time:
        return []

    # Пояс снимается явно: время уже в UTC, и numpy хранит его
    # без пояса. Второго пересчёта здесь быть не должно.
    moments = np.asarray(
        [moment.replace(tzinfo=None) for moment in event_time], dtype="datetime64[us]"
    )

    delta = (moments[-1] - moments).astype("timedelta64[us]").astype(np.int64)

    # Отрицательное расстояние это сломанный порядок событий, а
    # не повод его сгладить: референс здесь клампит, но молчаливо
    # исправленная ошибка данных страшнее остановленного этапа.
    if int(delta.min()) < 0:
        raise TemporalError(
            f"{client_id}: события идут не по возрастанию времени, и расстояние "
            "до последнего получилось отрицательным"
        )

    positions = TIME_SCALE * np.log1p(delta / (SECOND_US * TIME_SCALE))

    return positions.astype(np.float32).tolist()


def check(client_id: str, event_time_log: list[float], n_events: int) -> None:
    """
    Инварианты временных позиций.
    """

    if len(event_time_log) != n_events:
        raise TemporalError(
            f"{client_id}: временных позиций {len(event_time_log)} при {n_events} событиях"
        )

    for index, position in enumerate(event_time_log):

        if not math.isfinite(position):
            raise TemporalError(f"{client_id}: позиция {index} не конечна: {position!r}")

        if position < 0.0:
            raise TemporalError(f"{client_id}: позиция {index} отрицательна: {position!r}")

    # Последнее событие это точка отсчёта, и ноль у него не
    # случайность, а определение. Если он уехал, значит отсчёт
    # вёлся не от того события.
    if n_events and event_time_log[-1] != 0.0:
        raise TemporalError(
            f"{client_id}: у последнего события позиция {event_time_log[-1]!r}, а не ноль"
        )


__all__ = [
    "SECOND_US",
    "TIME_SCALE",
    "TemporalError",
    "check",
    "time_log",
]
