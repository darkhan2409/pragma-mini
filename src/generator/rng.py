from __future__ import annotations

import hashlib
import math
import struct
from bisect import bisect_right
from itertools import accumulate
from statistics import NormalDist
from typing import Any, Callable, Sequence

import numpy as np

from .config import SEED


# ============================================================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ РОЗЫГРЫША
# ============================================================
#
# Один seed и один отпечаток конфигурации на процесс. Всё, что
# кэшируется по client_id, обязано кэшироваться вместе с ними:
# иначе смена seed в одном процессе вернула бы старое значение.
# ============================================================

_STATE: dict[str, Any] = {"seed": SEED, "fingerprint": "default"}


def configure(seed: int = SEED, fingerprint: str = "default") -> None:
    """
    Устанавливает seed и отпечаток параметров процесса.
    Сбрасывает все кэши, привязанные к состоянию.
    """

    if _STATE["seed"] != seed or _STATE["fingerprint"] != fingerprint:
        clear_caches()

    _STATE["seed"] = int(seed)
    _STATE["fingerprint"] = str(fingerprint)


def current_seed() -> int:
    return int(_STATE["seed"])


def state_key() -> tuple[int, str]:
    return (int(_STATE["seed"]), str(_STATE["fingerprint"]))


_CACHES: list[dict] = []


def clear_caches() -> None:
    for cache in _CACHES:
        cache.clear()


def state_cache(func: Callable) -> Callable:
    """
    Кэш уровня процесса, ключ которого включает seed и отпечаток
    параметров. Заменяет lru_cache там, где результат зависит
    от глобального состояния розыгрыша.
    """

    cache: dict = {}
    _CACHES.append(cache)

    def wrapper(*args):
        key = (state_key(), args)
        hit = cache.get(key)
        if hit is None:
            hit = func(*args)
            cache[key] = hit
        return hit

    wrapper.cache = cache  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(func, "__name__", "cached")
    wrapper.__doc__ = func.__doc__

    return wrapper


# ============================================================
# НЕЙМСПЕЙСЫ
# ============================================================
#
# Номер закреплён за своим розыгрышем навсегда: переиспользование
# освободившегося номера незаметно сдвинуло бы уже порождённые
# данные.
# ============================================================

NS_PERSONA = 1
NS_TRAITS = 2
NS_GRAPH = 3
NS_INCOME = 4
NS_STRESS = 5
NS_LIFE = 6
NS_LIFECYCLE = 7
NS_HOUSEHOLD = 8
NS_CATALOG = 9
NS_MERCHANT = 10
NS_HABITS = 11
NS_NEEDS = 12
NS_ROUTINE = 13
NS_SESSION = 14
NS_SCREEN = 15
NS_OPERATION = 16
NS_OUTCOME = 17
NS_BANNER = 18
NS_COMM = 19
NS_ADOPTION = 20
NS_FUNNEL = 21
NS_LEDGER = 22
NS_LOAN = 23
NS_DEPOSIT = 24
NS_CARD = 25
NS_FRAUD = 26
NS_SUPPORT = 27
NS_OBSERVE = 28
NS_COVERAGE = 29
NS_PROFILE = 30
NS_OUTAGE = 31
NS_TRANSFER = 32
NS_DEVICE = 33
NS_TEST_ACCOUNT = 34
NS_PRODUCT_TIME = 35


# ============================================================
# КОМПОНЕНТЫ СОБЫТИЯ
# ============================================================

COMPONENT_COUNT = 1
COMPONENT_TIME = 2
COMPONENT_CONTENT = 3
COMPONENT_CHANNEL = 4
COMPONENT_LINKED = 5
COMPONENT_OUTCOME = 6


# ============================================================
# NUMPY-ПОТОК КЛИЕНТА
# ============================================================


def numpy_rng(*key: int) -> np.random.Generator:
    """
    Поток numpy для редких розыгрышей уровня клиента, где нужны
    beta, gamma и многомерная нормаль.
    """

    return np.random.default_rng([current_seed(), *key])


# ============================================================
# KEYED RANDOM
# ============================================================
#
# i-й розыгрыш потока это blake2b(key || i) / 2**64. Розыгрыши
# независимы, стоят около двух микросекунд, и поток полностью
# определяется своим ключом.
# ============================================================

_NORMAL = NormalDist()

_UNIT = float(2 ** 64)


class KeyedRandom:

    __slots__ = ("_prefix", "_index")

    def __init__(self, key: Sequence[int]) -> None:
        self._prefix = struct.pack(f"<{len(key)}q", *key)
        self._index = 0

    def random(self) -> float:
        digest = hashlib.blake2b(
            self._prefix + struct.pack("<q", self._index),
            digest_size=8,
        ).digest()

        self._index += 1

        return int.from_bytes(digest, "little") / _UNIT

    def integers(self, low: int, high: int | None = None) -> int:
        if high is None:
            low, high = 0, low

        if high <= low:
            return int(low)

        return int(low) + int(self.random() * (int(high) - int(low)))

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + self.random() * (high - low)

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        u = min(max(self.random(), 1e-12), 1.0 - 1e-12)
        return loc + scale * _NORMAL.inv_cdf(u)

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return math.exp(self.normal(mean, sigma))

    def poisson(self, lam: float) -> int:
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

    def chance(self, probability: float) -> bool:
        return self.random() < probability

    def choice(self, a, p=None):
        """
        Элемент из a (последовательность) или индекс из range(a),
        с весами p, которые не обязаны суммироваться к единице.
        """

        if isinstance(a, (int, np.integer)):
            size = int(a)
            items = None
        else:
            items = list(a)
            size = len(items)

        if size <= 0:
            raise ValueError("пустой набор для choice")

        u = self.random()

        if p is None:
            index = int(u * size)
        else:
            cumulative = list(accumulate(max(0.0, float(x)) for x in p))
            total = cumulative[-1]
            if total <= 0.0:
                index = int(u * size)
            else:
                index = bisect_right(cumulative, u * total)
                index = min(index, size - 1)

        return index if items is None else items[index]

    def weighted(self, mapping: dict) -> Any:
        """
        Ключ словаря с весами-значениями.
        """

        keys = list(mapping)
        return self.choice(keys, p=[mapping[key] for key in keys])

    def sample(self, items: Sequence, count: int) -> list:
        """
        Без возвращения, порядок исходной последовательности.
        """

        pool = list(items)

        count = max(0, min(count, len(pool)))

        chosen: list = []

        for _ in range(count):
            index = self.integers(0, len(pool))
            chosen.append(pool.pop(index))

        return chosen


def keyed_rng(*key: int) -> KeyedRandom:
    """
    Поток, включающий seed процесса.
    """

    return KeyedRandom((current_seed(), *key))


# ============================================================
# RNG СОБЫТИЯ
# ============================================================
#
# Идентичность события: namespace, сущность, день, индекс в дне,
# компонент. Изменение числа событий одного дня не пересеивает
# другие дни, а изменение содержимого не трогает время.
# ============================================================


def event_rng(
    namespace: int,
    entity: int,
    day: int,
    index_in_day: int,
    component: int,
) -> KeyedRandom:
    return keyed_rng(namespace, entity, day, index_in_day, component)


def day_rng(
    namespace: int,
    entity: int,
    day: int,
    component: int = COMPONENT_COUNT,
) -> KeyedRandom:
    return event_rng(namespace, entity, day, 0, component)


def stable_hash(*parts: object) -> int:
    """
    Устойчивый между процессами хэш строкового ключа.
    PYTHONHASHSEED на него не влияет.
    """

    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")

    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def stable_unit(*parts: object) -> float:
    """
    Устойчивое число в [0, 1) по строковому ключу и seed.
    """

    return stable_hash(current_seed(), *parts) / _UNIT


def second_of_day(ts) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second
