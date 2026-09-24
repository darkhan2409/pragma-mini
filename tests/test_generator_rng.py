from __future__ import annotations

import types

import pytest

from src.generator import rng
from src.generator.rng import KeyedRandom


# ============================================================
# ИДЕЯ
# ============================================================
#
# Граница «единица» недостижима случайным прогоном: вероятность
# около 5.6e-17 на розыгрыш. Поэтому она подаётся искусственно —
# подменяется хеш, а не сам random(). Так проверяется настоящий
# путь вычисления, а не переопределённый метод.
#
# Раньше uint64, близкий к максимуму, давал ровно 1.0 после
# деления на 2**64: шаг сетки float64 у единицы равен 2**-53.
# Следствия были три — chance(1.0) возвращал False, integers
# включал верхнюю границу, невзвешенный choice выходил за список.
# ============================================================


def digest_of(value: bytes):
    """
    Подмена blake2b: любой ключ даёт заданные восемь байт.
    """

    class Fixed:
        @staticmethod
        def digest() -> bytes:
            return value

    return types.SimpleNamespace(blake2b=lambda *args, **kwargs: Fixed)


@pytest.fixture
def at_maximum(monkeypatch):
    """
    Поток, у которого каждый розыгрыш идёт от максимального uint64.
    """

    monkeypatch.setattr(rng, "hashlib", digest_of(b"\xff" * 8))

    return KeyedRandom((1, 2, 3))


@pytest.fixture
def at_minimum(monkeypatch):

    monkeypatch.setattr(rng, "hashlib", digest_of(b"\x00" * 8))

    return KeyedRandom((1, 2, 3))


# ============================================================
# САМА ГРАНИЦА
# ============================================================


def test_maximum_uint64_stays_below_one(at_maximum):
    """
    Наибольший возможный хеш обязан дать число строго меньше
    единицы, иначе полуинтервал [0, 1) перестаёт быть таковым.
    """

    value = at_maximum.random()

    assert 0.0 <= value < 1.0

    # И это именно ближайшее снизу число, а не грубое усечение:
    # все остальные значения обязаны остаться прежними.
    assert value == pytest.approx(1.0, rel=1e-15)


def test_minimum_uint64_is_zero(at_minimum):

    assert at_minimum.random() == 0.0


def test_ordinary_values_are_not_touched(monkeypatch):
    """
    Прижатие границы не имеет права сдвинуть обычные значения:
    иначе прежние выгрузки перестали бы воспроизводиться.
    """

    half = (1 << 63).to_bytes(8, "little")

    monkeypatch.setattr(rng, "hashlib", digest_of(half))

    assert KeyedRandom((1,)).random() == 0.5


# ============================================================
# ТРИ ПРЕЖНИХ СЛЕДСТВИЯ
# ============================================================


def test_certain_event_always_happens(at_maximum):
    """
    chance(1.0) обязан быть истинным при любом розыгрыше.
    """

    assert at_maximum.chance(1.0) is True


def test_impossible_event_never_happens(at_minimum):

    assert at_minimum.chance(0.0) is False


@pytest.mark.parametrize("size", [1, 2, 3, 10])
def test_choice_without_weights_stays_in_range(at_maximum, size: int):
    """
    Раньше int(1.0 * size) == size выходило за список.
    """

    items = list(range(size))

    assert at_maximum.choice(items) == items[-1]


@pytest.mark.parametrize("size", [1, 2, 3, 10])
def test_choice_with_weights_stays_in_range(at_maximum, size: int):

    items = list(range(size))

    assert at_maximum.choice(items, p=[1.0] * size) == items[-1]


def test_choice_by_count_stays_in_range(at_maximum):
    """
    choice(int) возвращает индекс: он обязан быть допустимым.
    """

    assert at_maximum.choice(5) == 4


@pytest.mark.parametrize("low, high", [(0, 10), (5, 6), (-3, 4), (0, 1)])
def test_integers_keep_the_half_open_interval(at_maximum, low: int, high: int):
    """
    Верхняя граница исключена: integers(low, high) это [low, high).
    """

    value = at_maximum.integers(low, high)

    assert low <= value < high


def test_integers_with_empty_range_returns_low(at_maximum):

    assert at_maximum.integers(7, 7) == 7


def test_uniform_stays_below_the_top(at_maximum):

    assert 0.0 <= at_maximum.uniform(0.0, 1.0) < 1.0


def test_weighted_picks_the_last_key_at_the_top(at_maximum):

    assert at_maximum.weighted({"a": 1.0, "b": 1.0, "c": 1.0}) == "c"


# ============================================================
# ПОТОК НЕ СЛОМАН
# ============================================================


def test_real_stream_is_unchanged_and_reproducible():
    """
    Без подмены хеша поток обязан остаться прежним: одинаковый
    ключ даёт одинаковую последовательность, разные ключи —
    разные.
    """

    first = [KeyedRandom((5, 7)).random() for _ in range(3)]
    again = [KeyedRandom((5, 7)).random() for _ in range(3)]
    other = [KeyedRandom((5, 8)).random() for _ in range(3)]

    assert first == again
    assert first != other
    assert all(0.0 <= value < 1.0 for value in first + other)
