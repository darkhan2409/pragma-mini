"""
Границы и ключи генератора случайности — контролируемыми входами.

    python audit/2026-09-24-gen/checks/rng_bounds.py

Редкие численные границы не ловятся большим прогоном: вероятность
получить 1.0 из uint64/2**64 порядка 5.6e-17. Поэтому граница
подаётся руками, а проверяется настоящий код, который её
обрабатывает.

Каждая проверка печатает вердикт и основание. Гипотезы из
CONTRACTS.md здесь подтверждаются или опровергаются — и
опровержение такой же результат, как подтверждение.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

sys.path.insert(0, str(ROOT))

from src.generator import config, rng  # noqa: E402
from src.generator.rng import KeyedRandom, keyed_rng, stable_hash  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str) -> None:

    RESULTS.append({"check": name, "verdict": verdict, "detail": detail})

    print(f"[{verdict}] {name}: {detail}")


class Fixed(KeyedRandom):
    """
    Тот же KeyedRandom, но random() отдаёт заданное число.

    Нужен, чтобы подать методам границу, которую иначе не
    дождаться: вся остальная арифметика остаётся настоящей.
    """

    def __init__(self, value: float) -> None:
        super().__init__((0,))
        self._value = value

    def random(self) -> float:
        return self._value


# ============================================================
# ГРАНИЦА ПРЕОБРАЗОВАНИЯ ХЭША В FLOAT
# ============================================================


def check_unit_bound() -> None:
    """
    RNG-H2: может ли random() дать ровно 1.0.
    """

    top = int.from_bytes(b"\xff" * 8, "little") / float(2 ** 64)

    if top == 1.0:
        record(
            "H2 граница float",
            "ПОДТВЕРЖДЕНА",
            f"(2**64-1)/2**64 в float64 равно {top!r}: значение 1.0 достижимо, "
            "хотя вероятность около 5.6e-17 на розыгрыш",
        )
    else:
        record(
            "H2 граница float",
            "ОПРОВЕРГНУТА",
            f"верхнее значение {top!r} строго меньше 1.0",
        )

    # Сколько верхних значений uint64 округляются до 1.0.
    limit = 2 ** 64
    count = sum(1 for delta in range(1, 4096) if (limit - delta) / float(limit) == 1.0)

    record(
        "H2 ширина границы",
        "СПРАВКА",
        f"до 1.0 округляются {count} верхних значений uint64 из первых 4095",
    )


def check_chance_at_one() -> None:
    """
    Следствие H2: chance(1.0) при u == 1.0.
    """

    result = Fixed(1.0).chance(1.0)

    if result is False:
        record(
            "H2 следствие: chance(1.0)",
            "ПОДТВЕРЖДЕНА",
            "при u == 1.0 сравнение u < p даёт False: событие с вероятностью 1 "
            "не происходит. Частота события около 5.6e-17",
        )
    else:
        record("H2 следствие: chance(1.0)", "ОПРОВЕРГНУТА", f"вернулось {result}")


def check_choice_at_one() -> None:
    """
    Следствие H2: невзвешенный choice при u == 1.0.
    """

    items = ["a", "b", "c"]

    try:
        value = Fixed(1.0).choice(items)
        record(
            "H2 следствие: choice без весов",
            "ОПРОВЕРГНУТА",
            f"вернулось {value!r}, выхода за диапазон нет",
        )
    except IndexError as error:
        record(
            "H2 следствие: choice без весов",
            "ПОДТВЕРЖДЕНА",
            f"int(1.0 * 3) == 3 выходит за список: {type(error).__name__}: {error}",
        )

    # Взвешенная ветка ограничена min(index, size - 1).
    try:
        value = Fixed(1.0).choice(items, [1.0, 1.0, 1.0])
        record(
            "H2 следствие: choice с весами",
            "ОПРОВЕРГНУТА",
            f"ограничение сработало, вернулось {value!r}",
        )
    except IndexError as error:
        record(
            "H2 следствие: choice с весами",
            "ПОДТВЕРЖДЕНА",
            f"ограничение не спасло: {type(error).__name__}: {error}",
        )


def check_integers_at_one() -> None:

    value = Fixed(1.0).integers(0, 10)

    if value == 10:
        record(
            "H2 следствие: integers",
            "ПОДТВЕРЖДЕНА",
            "integers(0, 10) при u == 1.0 вернул 10 — верхняя граница включена, "
            "хотя по смыслу полуинтервала не должна",
        )
    else:
        record("H2 следствие: integers", "ОПРОВЕРГНУТА", f"вернулось {value}")


# ============================================================
# КЛЮЧ ПОТОКА
# ============================================================


def check_key_aliasing() -> None:
    """
    RNG-H1: может ли дописанный элемент ключа наложиться на
    более поздний розыгрыш прежнего потока.
    """

    # Поток keyed_rng(a, b), третий розыгрыш: хешируется
    # pack(root, a, b) + pack(2).
    short = KeyedRandom((7, 11, 13))
    short.random()
    short.random()
    third = short.random()

    # Поток keyed_rng(a, b, 2), первый розыгрыш: хешируется
    # pack(root, a, b, 2) + pack(0).
    longer = KeyedRandom((7, 11, 13, 2))
    first = longer.random()

    if third == first:
        record(
            "H1 наложение потоков",
            "ПОДТВЕРЖДЕНА",
            "третий розыгрыш keyed_rng(a, b) совпал с первым розыгрышем "
            "keyed_rng(a, b, 2)",
        )
    else:
        record(
            "H1 наложение потоков",
            "ОПРОВЕРГНУТА",
            "совпадения нет: счётчик розыгрыша дописывается ВСЕГДА, поэтому "
            f"длины ключей различаются ({len(struct.pack('<3q', 7, 11, 13)) + 8} "
            f"против {len(struct.pack('<4q', 7, 11, 13, 2)) + 8} байт) и хеши разные",
        )

    # Единственное настоящее условие совпадения: одинаковая
    # склейка ключа и счётчика.
    same = KeyedRandom((7, 11, 13)).random() == KeyedRandom((7, 11, 13)).random()

    record(
        "H1 условие совпадения",
        "СПРАВКА",
        f"одинаковый ключ даёт одинаковый поток: {same}; наложение возможно только "
        "при равной длине ключа и равных элементах",
    )


def check_stable_hash_types() -> None:
    """
    RNG-H4: stable_hash приводит части к строке.
    """

    if stable_hash(1) == stable_hash("1"):
        record(
            "H4 типы в stable_hash",
            "ПОДТВЕРЖДЕНА",
            "stable_hash(1) == stable_hash('1'): число и строка дают один ключ, "
            "разделителя типов нет",
        )
    else:
        record("H4 типы в stable_hash", "ОПРОВЕРГНУТА", "ключи различаются")

    # Разделитель \x1f: может ли он быть подделан содержимым.
    collide = stable_hash("a\x1fb") == stable_hash("a", "b")

    record(
        "H4 разделитель",
        "ПОДТВЕРЖДЕНА" if collide else "ОПРОВЕРГНУТА",
        f"stable_hash('a\\x1fb') == stable_hash('a', 'b'): {collide}",
    )


def check_fingerprint_in_key() -> None:
    """
    RNG-H5: входит ли отпечаток параметров в ключ потока.
    """

    rng.configure(seed=100, fingerprint="AAA", world_seed=42)
    first = keyed_rng(1, 2, 3).random()

    rng.configure(seed=100, fingerprint="BBB", world_seed=42)
    second = keyed_rng(1, 2, 3).random()

    if first == second:
        record(
            "H5 отпечаток параметров",
            "ПОДТВЕРЖДЕНА",
            "при одном seed и разных отпечатках поток чисел тот же: отпечаток "
            "влияет только на ключ кэша, не на случайность",
        )
    else:
        record("H5 отпечаток параметров", "ОПРОВЕРГНУТА", "потоки различаются")


def check_world_ids_seedless() -> None:
    """
    RNG-H6: зависят ли идентификаторы мира от seed.
    """

    left = stable_hash("brand", "grocery", "Magnum")

    rng.configure(seed=999, fingerprint="x", world_seed=7)

    right = stable_hash("brand", "grocery", "Magnum")

    if left == right:
        record(
            "H6 идентификаторы мира",
            "ПОДТВЕРЖДЕНА",
            "stable_hash не читает ни seed, ни world_seed: идентификаторы брендов, "
            "точек и поселений одинаковы во всех прогонах",
        )
    else:
        record("H6 идентификаторы мира", "ОПРОВЕРГНУТА", "значения различаются")


# ============================================================
# КЭШ
# ============================================================


def check_cache_key() -> None:
    """
    RNG-H3: горизонт в ключе кэша и промах на None.
    """

    rng.configure(seed=100, fingerprint="x", world_seed=42)

    calls = {"count": 0}

    @rng.state_cache
    def horizon_dependent() -> str:
        calls["count"] += 1
        return config.HISTORY_START.isoformat()

    start, end = config.HISTORY_START, config.HISTORY_END

    try:
        first = horizon_dependent()

        config.activate_horizon(
            start.replace(year=start.year + 1), end.replace(year=end.year + 1)
        )

        second = horizon_dependent()

    finally:
        config.activate_horizon(start, end)

    if first == second and calls["count"] == 1:
        record(
            "H3 горизонт в ключе кэша",
            "ПОДТВЕРЖДЕНА",
            f"после activate_horizon кэш отдал прежнее значение {first} и функцию "
            "не пересчитал: ключ кэша это (seed, world_seed, fingerprint), "
            "горизонта в нём нет",
        )
    else:
        record(
            "H3 горизонт в ключе кэша",
            "ОПРОВЕРГНУТА",
            f"значения {first} и {second}, вызовов {calls['count']}",
        )

    # Промах на None.
    misses = {"count": 0}

    @rng.state_cache
    def returns_none():
        misses["count"] += 1
        return None

    returns_none()
    returns_none()
    returns_none()

    if misses["count"] == 3:
        record(
            "H3 промах на None",
            "ПОДТВЕРЖДЕНА",
            "функция, законно вернувшая None, пересчитывается каждый раз: "
            "в кэше промах определяется как hit is None",
        )
    else:
        record("H3 промах на None", "ОПРОВЕРГНУТА", f"вызовов {misses['count']}")


# ============================================================
# ВЫРОЖДЕННЫЕ ВХОДЫ
# ============================================================


def check_degenerate() -> None:

    stream = KeyedRandom((1, 2))

    cases = []

    # Нулевая и единичная вероятность.
    cases.append(("chance(0.0)", lambda: Fixed(0.0).chance(0.0), False))
    cases.append(("chance(1.0) при u=0", lambda: Fixed(0.0).chance(1.0), True))

    for name, call, expected in cases:
        value = call()
        record(
            f"вырожденный вход: {name}",
            "PASS" if value == expected else "FAIL",
            f"вернулось {value}, ожидалось {expected}",
        )

    # Пустой набор.
    for name, call in (
        ("choice([])", lambda: stream.choice([])),
        ("sample([], 1)", lambda: stream.sample([], 1)),
        ("weighted({})", lambda: stream.weighted({})),
    ):
        try:
            value = call()
            record(f"пустой набор: {name}", "СПРАВКА", f"вернулось {value!r} без ошибки")
        except Exception as error:  # noqa: BLE001 - тип исключения и есть результат
            record(
                f"пустой набор: {name}",
                "СПРАВКА",
                f"{type(error).__name__}: {error}",
            )

    # Нулевые веса.
    try:
        value = stream.weighted({"a": 0.0, "b": 0.0})
        record("нулевые веса", "СПРАВКА", f"weighted вернул {value!r}")
    except Exception as error:  # noqa: BLE001
        record("нулевые веса", "СПРАВКА", f"{type(error).__name__}: {error}")

    # Отрицательный вес.
    try:
        value = stream.weighted({"a": -1.0, "b": 1.0})
        record("отрицательный вес", "СПРАВКА", f"weighted вернул {value!r}")
    except Exception as error:  # noqa: BLE001
        record("отрицательный вес", "СПРАВКА", f"{type(error).__name__}: {error}")

    # Ключ вне int64.
    try:
        KeyedRandom((2 ** 63,))
        record("ключ вне int64", "СПРАВКА", "принят без ошибки")
    except struct.error as error:
        record("ключ вне int64", "СПРАВКА", f"struct.error: {error}")


def main() -> int:

    check_unit_bound()
    check_chance_at_one()
    check_choice_at_one()
    check_integers_at_one()
    check_key_aliasing()
    check_stable_hash_types()
    check_fingerprint_in_key()
    check_world_ids_seedless()
    check_cache_key()
    check_degenerate()

    destination = AUDIT / "evidence" / "rng-bounds.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(f"результатов {len(RESULTS)} -> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
