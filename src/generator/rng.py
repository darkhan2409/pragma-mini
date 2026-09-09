from __future__ import annotations

import hashlib
import math
import struct
from bisect import bisect_right
from itertools import accumulate
from statistics import NormalDist
from typing import Sequence

import numpy as np

from .config import SEED


# ============================================================
# RNG NAMESPACES
# ============================================================

# Номера освобождены и не переиспользуются без нужды:
# 6 (graph), 7 (missing), 11 (season), 12 (transfers), 999 (label).

NS_PERSONA = 1
NS_TX = 2
NS_APP = 3
NS_COMM = 4
NS_PRODUCT = 5
NS_MERCHANT = 8
NS_CHAIN = 9
NS_NOISE = 10
NS_TRAJECTORY = 13
NS_CREDIT = 14
NS_PROFILE = 15
NS_SCREEN = 16
NS_OPERATION = 17
NS_BANNER = 18
NS_COVERAGE = 19
NS_FUNNEL = 20

# Потоки генерации v2. Номера новые: ключи v1 обязаны
# остаться за своими розыгрышами, иначе v1 изменится.
NS_V2_HABITS = 21
NS_V2_SCENARIO = 22
NS_V2_NAV = 23
NS_V2_OUTCOME = 24
NS_V2_OUTAGE = 25
NS_V2_TX = 26
NS_V2_INTEREST = 27


# ============================================================
# CLIENT RNG (numpy)
# ============================================================
#
# Для редких розыгрышей уровня клиента (персона, граф,
# узлы траектории, параметры обязательства) используем numpy:
# там нужны beta и прочие распределения, а вызовов мало.
# ============================================================


def client_rng(
    client_id: int,
    namespace: int,
    seed: int = SEED,
) -> np.random.Generator:
    """
    Детерминированный RNG-поток для конкретного клиента
    и конкретного источника данных.
    """

    return np.random.default_rng(
        [seed, namespace, client_id]
    )


# ============================================================
# KEYED RANDOM (hash stream)
# ============================================================
#
# Для событий нужны десятки тысяч независимых потоков на клиента.
# Конструировать numpy Generator на каждый (30 мкс) дорого.
#
# KeyedRandom даёт i-й розыгрыш потока как
#
#     blake2b(key || i) / 2**64
#
# Каждый розыгрыш независим, стоит около 2 мкс, а API повторяет
# используемое подмножество numpy Generator.
# ============================================================

_NORMAL = NormalDist()

_UNIT = float(2 ** 64)


class KeyedRandom:

    __slots__ = ("_prefix", "_index")

    def __init__(self, key: Sequence[int]) -> None:
        self._prefix = struct.pack(f"<{len(key)}q", *key)
        self._index = 0

    # --------------------------------------------------------

    def random(self) -> float:
        """
        Равномерное число в [0, 1).
        """

        digest = hashlib.blake2b(
            self._prefix + struct.pack("<q", self._index),
            digest_size=8,
        ).digest()

        self._index += 1

        return int.from_bytes(digest, "little") / _UNIT

    def integers(self, low: int, high: int | None = None) -> int:
        """
        Целое в [low, high), как numpy.
        """

        if high is None:
            low, high = 0, low

        return low + int(self.random() * (high - low))

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + self.random() * (high - low)

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        u = min(max(self.random(), 1e-12), 1.0 - 1e-12)
        return loc + scale * _NORMAL.inv_cdf(u)

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return math.exp(self.normal(mean, sigma))

    def poisson(self, lam: float) -> int:
        """
        Обратная функция распределения по одному равномерному числу.
        Для больших lam нормальное приближение.
        """

        if lam <= 0.0:
            return 0

        if lam > 60.0:
            return max(0, int(round(self.normal(lam, math.sqrt(lam)))))

        u = self.random()

        probability = math.exp(-lam)
        cdf = probability
        k = 0

        while u > cdf and k < 1000:
            k += 1
            probability *= lam / k
            cdf += probability

        return k

    def choice(self, a, p=None):
        """
        Элемент из a (последовательность) или индекс из range(a) (int),
        с весами p (не обязаны суммироваться к 1).
        """

        if isinstance(a, (int, np.integer)):
            size = int(a)
            items = None
        else:
            size = len(a)
            items = a

        u = self.random()

        if p is None:
            index = int(u * size)
        else:
            cumulative = list(accumulate(float(x) for x in p))
            total = cumulative[-1]
            index = bisect_right(cumulative, u * total)
            index = min(index, size - 1)

        return index if items is None else items[index]


def keyed_rng(*key: int) -> KeyedRandom:
    return KeyedRandom(key)


# ============================================================
# EVENT RNG (LOCALITY)
# ============================================================
#
# Идентичность события:
#
#     namespace, client_id, day, index_in_day, component
#
# Изменение числа событий в один день не пересеивает
# другие дни, а изменение содержимого не трогает
# количество и время.
# ============================================================

COMPONENT_COUNT = 1
COMPONENT_TIME = 2
COMPONENT_CONTENT = 3
COMPONENT_NOISE = 4

# Компоненты v2. Отдельные номера нужны, чтобы связанная
# покупка не делила поток с обычной покупкой того же дня.
COMPONENT_CHANNEL = 5
COMPONENT_LINKED = 6


def event_rng(
    namespace: int,
    client_id: int,
    day: int,
    index_in_day: int,
    component: int,
    seed: int = SEED,
) -> KeyedRandom:
    """
    RNG одного события клиента в конкретный день.
    day: ordinal даты (datetime.toordinal()).
    """

    return KeyedRandom((seed, namespace, client_id, day, index_in_day, component))


def day_rng(
    namespace: int,
    client_id: int,
    day: int,
    component: int = COMPONENT_COUNT,
    seed: int = SEED,
) -> KeyedRandom:
    """
    RNG дня клиента (например, число событий за день).
    """

    return event_rng(namespace, client_id, day, 0, component, seed)


def second_of_day(ts) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def ts_rng(
    namespace: int,
    client_id: int,
    ts,
    component: int,
    seed: int = SEED,
) -> KeyedRandom:
    """
    RNG по идентичности события с уже известным временем:
    день и секунда дня вместо индекса.
    """

    return event_rng(
        namespace,
        client_id,
        ts.toordinal(),
        second_of_day(ts),
        component,
        seed,
    )
