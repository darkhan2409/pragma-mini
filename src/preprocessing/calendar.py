from __future__ import annotations

import numpy as np

from .settings import CALENDAR_ENCODING


# ============================================================
# ИДЕЯ
# ============================================================
#
# Календарь это ОТДЕЛЬНЫЙ временной канал модели, а не поле
# события. Относительное время (возраст события, расстояние до
# соседа) его не выражает: зарплата приходит в свой день месяца,
# платежи по счетам в свой, а сессии приложения в свои часы.
# Периоды заданы календарём, а не выучены из данных.
#
# Считается из event_time и только из него. Ни час, ни день
# недели, ни день месяца не хранятся ни в RAW, ни в canonical и
# не попадают в fields модельного события: дублировать время
# отдельными полями значит завести два источника правды.
#
# Признаки числовые и в словари не входят: они не токены, не
# проходят BPE, не бакетизируются, не маскируются и не
# предсказываются. В модель они приходят отдельным входом.
#
# Время в выгрузке наивное и записано в поясе контракта
# (Asia/Almaty), поэтому час суток и день недели берутся из
# значения как есть. Пояс объявлен в конфигурации, входит в
# отпечаток этапа и в манифест: смена пояса или длины цикла
# требует пересборки.
# ============================================================


HOUR_US = 3_600_000_000

CALENDAR_FEATURES = len(CALENDAR_ENCODING["features"])

CYCLES = CALENDAR_ENCODING["cycles"]

# 1970-01-01 был четвергом, поэтому сдвиг на три дня делает
# понедельник нулём недельного цикла.
WEEK_SHIFT = 3


def calendar_features(ts) -> np.ndarray:
    """
    Час, день недели и день месяца события на единичной окружности.

    Колонки: sin/cos часа, sin/cos дня недели, sin/cos дня месяца.

    Час дробный: 09:30 и 09:00 это разные моменты суток. День
    недели считается от 1970-01-01, который был четвергом, поэтому
    сдвиг на три дня делает понедельник нулём. День месяца берётся
    нумерацией с нуля: первое число это 0.
    """

    moments = np.asarray(ts).astype("datetime64[us]")

    if moments.size == 0:
        return np.zeros((0, CALENDAR_FEATURES), dtype=np.float32)

    if np.isnat(moments).any():
        position = int(np.flatnonzero(np.isnat(moments))[0])
        raise ValueError(
            f"событие {position}: нет event_time, календарь считать не из чего. "
            "Пустое время останавливает preprocess проверкой RAW"
        )

    day = moments.astype("datetime64[D]")

    hour = (moments - day).astype("timedelta64[us]").astype(np.int64) / HOUR_US

    day_of_week = (day.astype(np.int64) + WEEK_SHIFT) % CYCLES["day_of_week"]

    day_of_month = (day - moments.astype("datetime64[M]")).astype(np.int64)

    angles = np.stack(
        [
            2.0 * np.pi * hour / CYCLES["hour_of_day"],
            2.0 * np.pi * day_of_week / CYCLES["day_of_week"],
            2.0 * np.pi * day_of_month / CYCLES["day_of_month"],
        ],
        axis=1,
    )

    out = np.empty((moments.size, CALENDAR_FEATURES), dtype=np.float32)

    out[:, 0::2] = np.sin(angles)
    out[:, 1::2] = np.cos(angles)

    return out


__all__ = [
    "CALENDAR_ENCODING",
    "CALENDAR_FEATURES",
    "CYCLES",
    "HOUR_US",
    "calendar_features",
]
