from __future__ import annotations

from datetime import datetime

from src.preprocessing.semantic.time import interval_hours
from src.preprocessing.settings import GroupWindow
from src.preprocessing.corpus import mlm_target_eligible


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


def hours_to_cutoff(cutoff: datetime, event_time: datetime) -> float:
    """
    Сколько часов от события до среза в объявленной точности
    события.

    У записи дневной точности час не наблюдался, поэтому и
    интервал до среза у неё кратен суткам: считать иначе значило
    бы выдумать время суток, которого источник не знает.
    """

    # Сам срез назначает человек, и он точен до секунды.
    return interval_hours(cutoff, event_time)


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


__all__ = [
    "AGE_UNKNOWN_REASON",
    "TargetsError",
    "eligible",
    "history_age_days",
    "hours_to_cutoff",
]
