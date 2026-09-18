from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn

from src.preprocessing.calendar import CALENDAR_FEATURES, calendar_features  # noqa: F401
from src.tokenizer.dataset import TokenBatch

from .batching import BatchError
from .config import ModelConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Время во внимании, а не в векторе события: q и k поворачиваются
# на угол, пропорциональный возрасту элемента (см. rotary.py).
# Отсюда берётся всё, что для этого нужно.
#
#   age = cutoff − ts события          возраст в часах
#   squash(t) = 8·log1p(t / 8)         сжатие: часы и месяцы
#                                      обязаны попадать в один
#                                      разумный диапазон
#
# Координата элемента это squash(age − age самого свежего), а
# разность координат пары и есть время между ними. Разрыв до
# предыдущего события отдельным числом не нужен: поворот его
# и выражает.
#
# Единицы измеряются честно: сначала вычитаются datetime64[us],
# и только потом длительность делится на микросекунды в часе.
# Обратный порядок терял бы разницу в микросекунды.
#
# Рядом два признака, которые относительное время выразить НЕ
# может и которые поэтому приходят отдельно: календарь события
# и простой клиента до cutoff.
# ============================================================


HOUR_US = 3_600_000_000

TIME_SCALE = 8.0

# ============================================================
# ЧАСЫ
# ============================================================


def hours_between(later: np.ndarray, earlier: np.ndarray) -> np.ndarray:
    """
    Разница двух datetime64 в часах.
    """

    delta = np.asarray(later) - np.asarray(earlier)

    return delta.astype("timedelta64[us]").astype(np.int64) / HOUR_US


def age_hours(batch: TokenBatch, cutoffs: np.ndarray) -> np.ndarray:
    """
    Часы от события до cutoff его примера.
    """

    ts = np.asarray(batch.ts)

    if ts.size == 0:
        return np.zeros(0, dtype=np.float64)

    example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)

    ages = hours_between(np.asarray(cutoffs)[example_of_event], ts)

    late = np.flatnonzero(ages <= 0)

    if late.size:
        position = int(late[0])
        raise BatchError(
            f"событие {position} примера {int(example_of_event[position])} не раньше cutoff: "
            f"age {ages[position]:.6f} ч; в историю входят только события строго до cutoff"
        )

    return ages


# ============================================================
# СЖАТИЕ И ПРОЕКЦИЯ
# ============================================================


def squash_np(hours, scale: float = TIME_SCALE) -> np.ndarray:
    """
    Та же функция на numpy: подготовка входа идёт до torch.

    Считается во float64 и только потом приводится вызывающим к
    float32: разница в микросекунды обязана пережить сжатие.
    """

    return scale * np.log1p(np.asarray(hours, dtype=np.float64) / scale)


# ============================================================
# КАЛЕНДАРЬ СОБЫТИЯ И ПРОСТОЙ КЛИЕНТА
# ============================================================
#
# Календарь это то, чего относительное время не выражает: зарплата
# приходит в свой день месяца, платежи по счетам в свой, а сессии
# приложения в свои часы. Периоды заданы календарём, а не выучены.
#
# Простой клиента это часы от последнего события до cutoff.
# Координата последнего элемента истории равна нулю, то есть
# сама история не знает, когда она кончилась. Признак
# возвращает это знание, и приписывается он профилю, потому что
# относится к клиенту целиком, а не к отдельному событию.
# ============================================================


# Календарь приходит из препроцессинга: правила (пояс, длины
# циклов, порядок колонок) объявлены в его конфигурации, входят
# в отпечаток этапа и в манифест. Здесь только использование,
# второй реализации нет.

# Масштаб простоя: год. Делением на squash(год) признак попадает
# в разумный диапазон, не теряя различий между днём и месяцем.
INACTIVITY_NORM_HOURS = 24.0 * 365.0


def inactivity_feature(hours) -> np.ndarray:
    """
    Часы простоя в сжатый признак порядка единицы.
    """

    return (squash_np(hours) / squash_np(INACTIVITY_NORM_HOURS)).astype(np.float32)


class FeatureMLP(nn.Module):
    """
    Небольшой числовой признак в вектор d_model.

    Последний слой инициализируется НУЛЁМ. На шаге 0 признак не
    прибавляет ничего, и вход History Encoder равен чистому
    вектору события: масштаб слагаемого модель выбирает сама, а
    не получает навязанным.

    Следствие, которое надо знать при чтении градиентов: на
    первом шаге градиент скрытого слоя равен нулю, потому что
    проходит через нулевые веса выхода. Это ожидаемо и не
    означает оборванного графа.
    """

    def __init__(self, n_features: int, config: ModelConfig):

        super().__init__()

        self.n_features = int(n_features)

        self.hidden = nn.Linear(self.n_features, config.d_model)

        self.activation = nn.GELU() if config.activation == "gelu" else nn.ReLU()

        self.output = nn.Linear(config.d_model, config.d_model)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        if features.ndim != 2 or features.shape[1] != self.n_features:
            raise BatchError(
                f"ожидалось [n, {self.n_features}], получено {tuple(features.shape)}"
            )

        # Признаки считаются во float32: под autocast bf16 у
        # синусов и сжатых часов осталось бы восемь бит мантиссы.
        return self.output(self.activation(self.hidden(features.float())))

