from __future__ import annotations

from datetime import datetime

from src.preprocessing.history import (
    COVERAGE_AVAILABLE,
    COVERAGE_ENDED,
    COVERAGE_NOT_LAUNCHED,
    COVERAGE_NOT_SEEN,
)
from src.preprocessing.semantic.time import interval_hours
from src.preprocessing.settings import GroupWindow
from src.preprocessing.split import mlm_target_eligible


# ============================================================
# ИДЕЯ
# ============================================================
#
# Где вообще разрешено выбирать цели, что известно о времени и
# что известно о доступности источников.
#
# Датасет НИЧЕГО не маскирует. Он только говорит, какие события
# лежат в периоде будущих целей своей группы: старая история
# validation и test остаётся видимым контекстом и оценочной
# целью не становится. Окончательный выбор целей это работа
# Masker.
#
# Правило периода не переписывается: оно живёт в этапе
# разделения, и здесь именно оно импортируется. Вторая
# реализация «того же самого» рано или поздно разошлась бы с
# первой на границе.
#
# Время: календарь и признак наблюдавшегося часа копируются как
# есть, а интервал до среза считается в ОБЪЯВЛЕННОЙ точности
# события. Неизвестное начало наблюдения не становится нулём:
# «клиент с банком ноль дней» и «когда он пришёл, неизвестно»
# это разные утверждения.
# ============================================================


# Состояние источника кодируется числом, потому что едет
# колонкой рядом с числами. Имена лежат рядом в манифесте:
# колонка чисел без списка имён ничего не значит.
COVERAGE_STATES: tuple[str, ...] = (
    COVERAGE_AVAILABLE,
    COVERAGE_NOT_LAUNCHED,
    COVERAGE_NOT_SEEN,
    COVERAGE_ENDED,
    # Строки покрытия у этого клиента по этому источнику нет
    # вовсе: это не «источник молчал», это «о нём ничего не
    # сказано».
    "no_row",
)

COVERAGE_CODES: dict[str, int] = {name: number for number, name in enumerate(COVERAGE_STATES)}

NO_ROW = COVERAGE_CODES["no_row"]

# Почему возраст истории неизвестен.
AGE_UNKNOWN_REASON = "observed_start_unknown"


class TargetsError(ValueError):
    """
    Признак цели или канал времени посчитать нельзя.
    """


def eligible(event_time: datetime, window: GroupWindow) -> bool:
    """
    Событие лежит в периоде будущих целей своей группы.

    Это ровно то же правило, что записал этап разделения: оно
    импортировано, а не повторено.
    """

    return mlm_target_eligible(event_time, window)


def hours_to_cutoff(cutoff: datetime, event_time: datetime, precision: str) -> float:
    """
    Сколько часов от события до среза в объявленной точности
    события.

    У записи дневной точности час не наблюдался, поэтому и
    интервал до среза у неё кратен суткам: считать иначе значило
    бы выдумать время суток, которого источник не знает.
    """

    # Сам срез назначает человек, и он точен до секунды.
    return interval_hours(cutoff, "second", event_time, precision)


def history_age_days(relationship, cutoff: datetime) -> tuple[float | None, str | None]:
    """
    Сколько дней видно отношения клиента с банком на этот срез.

    Пусто, когда начало наблюдения неизвестно. Ноль сюда не
    подставляется: это сказало бы модели, что клиент пришёл
    сегодня.
    """

    observed = getattr(relationship, "observed_days", None)

    if observed is None:
        return None, AGE_UNKNOWN_REASON

    return float(observed), None


def coverage_codes(coverage, sources: tuple[str, ...]) -> list[int]:
    """
    Состояние каждого объявленного источника на срез.

    Порядок источников задан снаружи и одинаков у всех примеров
    набора: иначе колонки соседних клиентов означали бы разное.
    """

    state_of = {item.source: item.state for item in coverage}

    out: list[int] = []

    for name in sources:

        state = state_of.get(name)

        if state is None:
            out.append(NO_ROW)
            continue

        code = COVERAGE_CODES.get(state)

        if code is None:
            raise TargetsError(
                f"источник {name}: состояние {state!r} этому формату неизвестно. "
                "Состояния покрытия объявляет этап истории на дату"
            )

        out.append(code)

    return out


def coverage_details(coverage) -> list[dict]:
    """
    Даты и недатированные причины покрытия: служебные сведения
    рядом с колонкой состояний.
    """

    return [
        {
            "source": item.source,
            "state": item.state,
            "first_available_at": item.first_available_at,
            "first_seen": item.first_seen,
            "last_available_at": item.last_available_at,
            "reason": item.reason,
        }
        for item in sorted(coverage, key=lambda item: item.source)
    ]


__all__ = [
    "AGE_UNKNOWN_REASON",
    "COVERAGE_CODES",
    "COVERAGE_STATES",
    "NO_ROW",
    "TargetsError",
    "coverage_codes",
    "coverage_details",
    "eligible",
    "history_age_days",
    "hours_to_cutoff",
]
