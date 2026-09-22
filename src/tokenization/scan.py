from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable

from src.preprocessing.canonical.events import normalize_text
from src.preprocessing.rawdata import ContentDigest
from src.preprocessing.read import ClientHistory
from src.preprocessing.keys import CATEGORICAL, NUMERIC, REFERENCE, TEXT

from .schema import SemanticSchema, WEIGHT_PER_CLIENT


# ============================================================
# ИДЕЯ
# ============================================================
#
# Один поток по разрешённому корпусу, из которого выходит всё,
# что понадобится словарям: какие значения встречались, как
# часто, у скольких клиентов, какие числа наблюдались, какие
# тексты и где значения не было вовсе.
#
# Три правила, без которых статистика врала бы.
#
# 1. Единица подсчёта у ключа своя. Значение события считается
#    по событиям, значение профиля — по клиентам: версия профиля
#    на fit_end это один факт, а не сто повторов по числу
#    покупок клиента.
#
# 2. Значение это ПАРА «тип, значение». True, 1 и "1" в Python
#    сравниваются между собой, и словарь бы их склеил; у
#    токенизатора это разные значения разных доменов.
#
# 3. Числовая выборка не зависит ни от порядка чтения, ни от
#    seed: берутся k наименьших по хэшу единицы дедупликации.
#    Тот же корпус даёт ту же выборку в любом порядке и при
#    любом числе процессов.
# ============================================================


class ScanError(ValueError):
    """
    Смысловое значение не укладывается в объявленный контракт.
    """


TYPE_BOOL = "bool"
TYPE_INT = "int"
TYPE_FLOAT = "float"
TYPE_STR = "str"

# Порядок типов при сортировке значений домена: сначала булево,
# потом целые, потом строки. Внутри типа сравниваются сами
# значения, а не их запись.
TYPE_ORDER: dict[str, int] = {TYPE_BOOL: 0, TYPE_INT: 1, TYPE_FLOAT: 2, TYPE_STR: 3}


def value_type(value: object) -> str:
    """
    Тип значения так, как его понимает словарь.
    """

    # Булево проверяется первым: в Python bool это подкласс int.
    if isinstance(value, bool):
        return TYPE_BOOL

    if isinstance(value, int):
        return TYPE_INT

    if isinstance(value, float):
        return TYPE_FLOAT

    if isinstance(value, str):
        return TYPE_STR

    raise ScanError(f"значение типа {type(value).__name__} смысловым слоем не объявлено: {value!r}")


def value_text(value: object) -> str:
    """
    Запись значения, по которой оно хранится и сравнивается.

    Обратима внутри своего типа: float через repr, булево
    словом, всё остальное как есть.
    """

    kind = value_type(value)

    if kind == TYPE_BOOL:
        return "true" if value else "false"

    if kind == TYPE_FLOAT:
        return repr(value)

    return str(value)


def _unit_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


# ------------------------------------------------------------
# ЧИСЛОВАЯ ВЫБОРКА
# ------------------------------------------------------------


class NumericSketch:
    """
    k наименьших по хэшу значений ключа и итоги по всем.

    Выборка нужна там, где значений больше, чем разумно держать
    в памяти. Она не случайная в смысле seed: хэш единицы
    дедупликации задан данными, поэтому результат одинаков при
    любом порядке чтения.
    """

    def __init__(self, k: int, distinct_cap: int):
        self.k = k
        self.distinct_cap = distinct_cap
        self._items: list[tuple[int, float]] = []
        self.n = 0
        self.n_zero = 0
        self.n_negative = 0
        self.n_invalid = 0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.clients = 0
        self._last_client: str | None = None
        self._distinct: set[float] = set()
        self.distinct_exact = True

    def add(self, value: float, unit: str, client_id: str) -> None:

        if self._last_client != client_id:
            self._last_client = client_id
            self.clients += 1

        number = float(value)

        if math.isnan(number) or math.isinf(number):
            self.n_invalid += 1
            return

        self.n += 1

        if number == 0.0:
            self.n_zero += 1
        elif number < 0.0:
            self.n_negative += 1

        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

        if self.distinct_exact:
            self._distinct.add(number)
            if len(self._distinct) > self.distinct_cap:
                self.distinct_exact = False
                self._distinct = set()

        self._items.append((_unit_hash(unit), number))

        if len(self._items) > 2 * self.k:
            self._trim()

    def _trim(self) -> None:
        self._items.sort()
        del self._items[self.k:]

    @property
    def sampled(self) -> bool:
        return self.n > self.k

    def values(self) -> list[float]:
        """
        Отобранные значения в порядке возрастания самого числа.
        """

        self._trim()

        return sorted(value for _hash, value in self._items)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "clients": self.clients,
            "n_zero": self.n_zero,
            "n_negative": self.n_negative,
            "n_invalid": self.n_invalid,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "distinct": len(self._distinct) if self.distinct_exact else None,
            "distinct_exact": self.distinct_exact,
            "sampled": self.sampled,
            "sample_k": self.k,
            "sample_size": len(self._items),
            "sample_rule": "k наименьших blake2b по (клиент, событие, версия, ключ); порядок чтения не влияет",
        }


# ------------------------------------------------------------
# СТАТИСТИКА
# ------------------------------------------------------------


@dataclass
class TextEntry:
    count: int = 0
    clients: int = 0
    last_client: str | None = None
    example_raw: str = ""
    max_bytes: int = 0


@dataclass
class CategoryEntry:
    count: int = 0
    clients: int = 0
    last_client: str | None = None


@dataclass
class KeyCounter:
    events: int = 0
    clients: int = 0
    last_client: str | None = None


@dataclass
class FitStatistics:
    """
    Всё, что увидел один проход по корпусу.
    """

    categorical: dict[tuple[str, str, str], CategoryEntry] = field(default_factory=dict)
    numeric: dict[str, NumericSketch] = field(default_factory=dict)
    text: dict[str, dict[str, TextEntry]] = field(default_factory=dict)
    text_empty: dict[str, int] = field(default_factory=dict)

    key_counts: dict[str, KeyCounter] = field(default_factory=dict)
    key_types: dict[str, set[str]] = field(default_factory=dict)

    event_types: dict[str, int] = field(default_factory=dict)
    unknown_event_types: dict[str, int] = field(default_factory=dict)
    missing: dict[tuple[str, str], int] = field(default_factory=dict)
    unknown_keys: dict[str, int] = field(default_factory=dict)

    reference_values: dict[str, int] = field(default_factory=dict)

    clients: int = 0
    events: int = 0
    values: int = 0
    clients_without_profile: int = 0
    profiles: int = 0
    limitations: dict[str, int] = field(default_factory=dict)

    content = None

    def key_counter(self, key: str) -> KeyCounter:
        counter = self.key_counts.get(key)
        if counter is None:
            counter = KeyCounter()
            self.key_counts[key] = counter
        return counter

    def observed_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.key_counts))


def _note_key(stats: FitStatistics, key: str, kind: str, client_id: str) -> None:

    counter = stats.key_counter(key)
    counter.events += 1

    if counter.last_client != client_id:
        counter.last_client = client_id
        counter.clients += 1

    stats.key_types.setdefault(key, set()).add(kind)


def _add_categorical(stats: FitStatistics, key: str, value: object, client_id: str) -> None:

    kind = value_type(value)
    entry_key = (key, kind, value_text(value))

    entry = stats.categorical.get(entry_key)

    if entry is None:
        entry = CategoryEntry()
        stats.categorical[entry_key] = entry

    entry.count += 1

    if entry.last_client != client_id:
        entry.last_client = client_id
        entry.clients += 1


def _add_text(stats: FitStatistics, key: str, value: object, client_id: str) -> None:

    if not isinstance(value, str):
        raise ScanError(f"ключ {key} объявлен текстом, а значение пришло как {type(value).__name__}: {value!r}")

    normalized = normalize_text(value)

    if normalized is None:
        # Пустая строка и строка из пробелов это не отсутствие
        # значения: поле пришло, текста в нём нет.
        stats.text_empty[key] = stats.text_empty.get(key, 0) + 1
        return

    bucket = stats.text.setdefault(key, {})

    entry = bucket.get(normalized)

    if entry is None:
        entry = TextEntry(example_raw=value)
        bucket[normalized] = entry

    entry.count += 1
    entry.max_bytes = max(entry.max_bytes, len(normalized.encode("utf-8")))

    if entry.last_client != client_id:
        entry.last_client = client_id
        entry.clients += 1


def _add_numeric(stats: FitStatistics, key: str, value: object, unit: str, client_id: str,
                 sample_k: int, distinct_cap: int) -> None:

    kind = value_type(value)

    if kind not in (TYPE_INT, TYPE_FLOAT):
        raise ScanError(f"ключ {key} объявлен числом, а значение пришло как {kind}: {value!r}")

    sketch = stats.numeric.get(key)

    if sketch is None:
        sketch = NumericSketch(sample_k, distinct_cap)
        stats.numeric[key] = sketch

    sketch.add(float(value), unit, client_id)


def scan(
    histories: Iterable[ClientHistory],
    schema: SemanticSchema,
    sample_k: int,
    distinct_cap: int,
) -> FitStatistics:
    """
    Один проход по историям корпуса.

    Историй в памяти не накапливается: на выходе только счётчики
    и выборки.
    """

    stats = FitStatistics()
    digest = ContentDigest()

    for history in histories:

        client_id = history.client_id
        stats.clients += 1

        # --- события ---

        for event in history.events:

            stats.events += 1

            event_type = event.values.get("event_type")

            if event_type is None:
                raise ScanError(
                    f"событие {event.stable_event_index} клиента {client_id} пришло без типа: "
                    "смысловой слой обязан его выдавать"
                )

            stats.event_types[event_type] = stats.event_types.get(event_type, 0) + 1

            values = event.model_values()

            unit_prefix = f"{client_id}\x1f{event.stable_event_index}\x1f"

            for key, value in sorted(values.items()):

                info = schema.keys.get(key)

                if info is None:
                    stats.unknown_keys[key] = stats.unknown_keys.get(key, 0) + 1
                    continue

                digest.add(
                    {
                        "unit": "event",
                        "client_id": client_id,
                        "stable_event_index": event.stable_event_index,
                        "key": key,
                        "type": value_type(value),
                        "value": value_text(value),
                    }
                )

                if info.value_kind == REFERENCE:
                    # Ссылка кодом не становится: считается только
                    # её наличие, чтобы отчёт показал, сколько
                    # связей уходит в метаданные.
                    stats.reference_values[key] = stats.reference_values.get(key, 0) + 1
                    continue

                stats.values += 1

                _note_key(stats, key, value_type(value), client_id)

                if info.value_kind == NUMERIC:
                    _add_numeric(stats, key, value, unit_prefix + key, client_id, sample_k, distinct_cap)
                elif info.value_kind == CATEGORICAL:
                    _add_categorical(stats, key, value, client_id)
                elif info.value_kind == TEXT:
                    _add_text(stats, key, value, client_id)
                else:
                    raise ScanError(f"ключ {key}: неизвестный вид значения {info.value_kind!r}")

            # --- объявленное, но отсутствующее ---

            declared = schema.declared(event_type)

            if not declared:
                stats.unknown_event_types[event_type] = stats.unknown_event_types.get(event_type, 0) + 1

            for key in declared:
                if key not in values:
                    slot = (event_type, key)
                    stats.missing[slot] = stats.missing.get(slot, 0) + 1

        # --- профиль ---

        if not history.profile:
            stats.clients_without_profile += 1
        else:
            stats.profiles += 1

            prefix = f"{client_id}\x1f"

            for key, value in sorted(history.profile.items()):

                info = schema.keys.get(key)

                if info is None:
                    stats.unknown_keys[key] = stats.unknown_keys.get(key, 0) + 1
                    continue

                if info.weight_rule != WEIGHT_PER_CLIENT:
                    raise ScanError(
                        f"ключ {key} пришёл из профиля, но объявлен событийным: "
                        "правило веса разошлось с происхождением"
                    )

                stats.values += 1

                digest.add(
                    {
                        "unit": "profile",
                        "client_id": client_id,
                        "key": key,
                        "type": value_type(value),
                        "value": value_text(value),
                    }
                )

                _note_key(stats, key, value_type(value), client_id)

                if info.value_kind == NUMERIC:
                    _add_numeric(stats, key, value, prefix + key, client_id, sample_k, distinct_cap)
                elif info.value_kind == CATEGORICAL:
                    _add_categorical(stats, key, value, client_id)
                elif info.value_kind == TEXT:
                    _add_text(stats, key, value, client_id)
                else:
                    raise ScanError(f"ключ {key}: неизвестный вид значения {info.value_kind!r}")

        for item in history.limitations:
            stats.limitations[item] = stats.limitations.get(item, 0) + 1

    stats.content = digest

    return stats


__all__ = [
    "TYPE_BOOL",
    "TYPE_FLOAT",
    "TYPE_INT",
    "TYPE_ORDER",
    "TYPE_STR",
    "CategoryEntry",
    "FitStatistics",
    "KeyCounter",
    "NumericSketch",
    "ScanError",
    "TextEntry",
    "scan",
    "value_text",
    "value_type",
]
