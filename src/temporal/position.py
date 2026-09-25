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
#
# ВРЕМЯ ТОКЕНОВ АНКЕТЫ. Шкала та же, 8 * log1p(delta / 8), но
# точка отсчёта другая — cutoff T примера:
#
#   profile_time_log[j] = 8 * log1p((T - t_j) / 8)   у вехи Lifelong,
#   profile_time_log[j] = 0                          у [USR] и Attributes.
#
# Анкета описывает клиента на T, и её время привязано к T, а не к
# последнему событию: веха 2022 года дальше вехи 2025-го, когда бы
# ни было последнее событие. Ноль у Attributes значит «состояние
# на T», у [USR] — якорь; различает их ключ, а не время.
# ============================================================


# Масштаб сжатия времени, в секундах.
TIME_SCALE = 8.0

SECOND_US = 1_000_000

# Форма шкалы словами — для метаданных этапов.
TIME_TRANSFORM = f"{TIME_SCALE:g}*log1p(seconds/{TIME_SCALE:g})"


class TemporalError(ValueError):
    """
    Временные позиции посчитать нельзя.
    """


def log_age(delta_us: np.ndarray) -> np.ndarray:
    """
    Сжатое расстояние по целым микросекундам: 8 * log1p(сек / 8).
    """

    return TIME_SCALE * np.log1p(delta_us / (SECOND_US * TIME_SCALE))


def _utc_naive(moments: list[datetime]) -> np.ndarray:
    """
    Моменты UTC как datetime64[us]. Пояс снимается явно: время уже
    в UTC, и numpy хранит его без пояса. Второго пересчёта здесь
    быть не должно.
    """

    return np.asarray([moment.replace(tzinfo=None) for moment in moments], dtype="datetime64[us]")


def time_log(client_id: str, event_time: list[datetime]) -> list[float]:
    """
    Временные позиции всех событий примера.
    """

    if not event_time:
        return []

    moments = _utc_naive(event_time)

    delta = (moments[-1] - moments).astype("timedelta64[us]").astype(np.int64)

    # Отрицательное расстояние это сломанный порядок событий, а
    # не повод его сгладить: референс здесь клампит, но молчаливо
    # исправленная ошибка данных страшнее остановленного этапа.
    if int(delta.min()) < 0:
        raise TemporalError(
            f"{client_id}: события идут не по возрастанию времени, и расстояние "
            "до последнего получилось отрицательным"
        )

    return log_age(delta).astype(np.float32).tolist()


def profile_time_log(
    client_id: str, times: list[datetime | None], cutoff: datetime
) -> list[float]:
    """
    Временные позиции токенов анкеты: давность вехи до cutoff,
    ноль у недатированных.
    """

    out = np.zeros(len(times), dtype=np.float32)

    dated = [index for index, moment in enumerate(times) if moment is not None]

    if not dated:
        return out.tolist()

    anchor = _utc_naive([cutoff])[0]

    delta = (anchor - _utc_naive([times[index] for index in dated])).astype(
        "timedelta64[us]"
    ).astype(np.int64)

    # Веха не позже cutoff — иначе анкета знала бы будущее. Это
    # ошибка данных, а не повод обрезать расстояние до нуля.
    if int(delta.min()) <= 0:
        raise TemporalError(
            f"{client_id}: веха анкеты не раньше cutoff {cutoff.isoformat()}"
        )

    out[dated] = log_age(delta).astype(np.float32)

    return out.tolist()


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


def check_profile(
    client_id: str, profile_time_log: list[float], times: list[datetime | None]
) -> None:
    """
    Инварианты времени анкеты: у недатированного токена ровно
    ноль, у вехи — конечное положительное расстояние.
    """

    if len(profile_time_log) != len(times):
        raise TemporalError(
            f"{client_id}: позиций анкеты {len(profile_time_log)} при {len(times)} токенах"
        )

    for index, (position, moment) in enumerate(zip(profile_time_log, times)):

        if moment is None:
            if position != 0.0:
                raise TemporalError(
                    f"{client_id}: у недатированного токена анкеты {index} позиция {position!r}"
                )
        elif not math.isfinite(position) or position <= 0.0:
            raise TemporalError(
                f"{client_id}: у вехи анкеты {index} позиция {position!r}, а нужна положительная"
            )


__all__ = [
    "SECOND_US",
    "TIME_SCALE",
    "TIME_TRANSFORM",
    "TemporalError",
    "check",
    "check_profile",
    "log_age",
    "profile_time_log",
    "time_log",
]
