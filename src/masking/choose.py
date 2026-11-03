from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from src.generator.rng import KeyedRandom, stable_hash

from src.tokenization.schema import WEIGHT_PER_EVENT, SemanticSchema

from .settings import EVENT_TYPE, ConfigError, MaskingConfig
from .weights import ValueWeights, WeightsError


# ============================================================
# ОТБОР
# ============================================================
#
# Единственное место со случайностью. Словаря здесь нет вовсе:
# отбор решает ЧТО прятать, а чем именно — решает подстановка.
#
# Допустимо только значение события, у которого стоит
# target_event_mask. [EVT], [USR], анкета, время и календарь
# недопустимы по построению: разбор идёт лишь по окнам таких
# событий и начинается со следующего после маркера токена.
# Заполнителя в строке нет: маскер читает пример как есть.
#
# Три механизма разыгрываются совместно, и сильнейшая причина
# побеждает: event > key > value. Так устроен и референс,
# pragmatiq/training/masking.py, где приоритет задан порядком
# перезаписи mask_type.
#
# Отличие от референса одно, и оно важное. Там механизм token
# разыгрывал каждый токен, и текст из нескольких кусков BPE мог
# оказаться скрытым наполовину. Здесь разыгрывается ЗНАЧЕНИЕ, и
# все его куски получают одно решение: половина названия была бы
# подсказкой, а не задачей.
# ============================================================


EVENT = "event"
KEY = "key"
VALUE = "value"

# Пустая причина значит «не выбрано». Её же получает значение,
# ушедшее в [UNK]: оно испорчено, но целью не является.
NONE = ""


# Номера потоков. Потоки независимы, поэтому добавление значения
# не сдвигает розыгрыш событий, а смена числа ключей не трогает
# розыгрыш [UNK]. Номер закреплён за своим розыгрышем навсегда.
_STREAM_EVENT = 1
_STREAM_KEY = 2
_STREAM_VALUE = 3
_STREAM_UNKNOWN = 4
_STREAM_CONTEXT = 5


@dataclass(frozen=True)
class Value:
    """
    Одно значение внутри окна своего события.
    """

    event: int
    key_id: int
    start: int
    length: int


@dataclass(frozen=True)
class Values:
    """
    Значения подряд массивами: событие, ключ, начало и длина. Объекты
    Value из них строятся только для тех, кому нужен список.
    """

    event: np.ndarray
    key_id: np.ndarray
    start: np.ndarray
    length: np.ndarray

    def __len__(self) -> int:
        return int(self.event.size)

    def take(self, index: np.ndarray) -> "Values":
        return Values(self.event[index], self.key_id[index], self.start[index], self.length[index])

    def as_list(self) -> list[Value]:
        return [
            Value(*item)
            for item in zip(self.event.tolist(), self.key_id.tolist(), self.start.tolist(), self.length.tolist())
        ]

    @staticmethod
    def of(values: list[Value]) -> "Values":
        return Values(*(np.array([getattr(value, name) for value in values], dtype=np.int64)
                        for name in ("event", "key_id", "start", "length")))


# Код причины выбора: индекс в REASONS.
REASONS = (NONE, EVENT, KEY, VALUE)


@dataclass(frozen=True)
class Choice:
    """
    Выбранное значение, механизм выбора и его судьба.

    unknown означает [UNK]: значение портится, но в loss не
    входит, и причина в файл не пишется.
    """

    value: Value
    reason: str
    unknown: bool


@dataclass(frozen=True)
class Selection:
    """
    Что нашлось у клиента и что из этого выбрано.

    Допустимые события и значения возвращаются вместе с выбором,
    чтобы отчёт не разбирал ту же строку второй раз. corrupted —
    значения выбранного механизмом key ключа вне целей, испорченные
    в [UNK] без метки (key_context_corruption_probability).

    Внутри — массивы: found (допустимые значения), picked (номера
    выбранных среди них), reasons (код причины, REASONS), unknown
    (ушло ли в [UNK]) и spoiled (испорченный контекст). values,
    choices и corrupted — те же значения объектами, по требованию.
    """

    events: int
    found: Values
    picked: np.ndarray
    reasons: np.ndarray
    unknown: np.ndarray
    spoiled: Values

    @cached_property
    def values(self) -> list[Value]:
        return self.found.as_list()

    @cached_property
    def choices(self) -> list[Choice]:
        values = self.found.take(self.picked).as_list()
        return [
            Choice(value, REASONS[code], unknown)
            for value, code, unknown in zip(values, self.reasons.tolist(), self.unknown.tolist())
        ]

    @cached_property
    def corrupted(self) -> tuple[Value, ...]:
        return tuple(self.spoiled.as_list())


@dataclass(frozen=True)
class FieldGroups:
    """
    Группы полей конфига маски в номерах ключей текущего словаря.

    of_key — номер группы по key_id, -1 у ключа вне групп; event_type —
    key_id ключа event_type при event_type_event_only, иначе -1.
    """

    of_key: np.ndarray
    event_type: int


def field_groups(config: MaskingConfig, key_vocab: dict[str, int], schema: SemanticSchema) -> FieldGroups | None:
    """
    FieldGroups конфига по словарю ключей (имя → key_id). None — групп
    нет и флаг выключен: маска прежняя, и choose их не ждёт.

    Ключ группы обязан быть ключом события по реестру смыслов (schema,
    правило веса per_event): ключ анкеты есть и в словаре, но в окне
    события не встречается, и в группе он молча ничего бы не закрывал.
    """

    if not config.field_groups and not config.event_type_event_only:
        return None

    config.validate()

    names = [key for group in config.field_groups for key in group]

    if config.event_type_event_only:
        names.append(EVENT_TYPE)

    strangers = [key for key in names if key not in schema.keys or schema.keys[key].weight_rule != WEIGHT_PER_EVENT]

    if strangers:
        raise ConfigError(f"ключи {strangers} не ключи событий по реестру смыслов: группа их не закроет")

    missing = [key for key in names if key not in key_vocab]

    if missing:
        raise ConfigError(f"ключей {missing} нет в словаре ключей (key_vocab.json)")

    of_key = np.full(max(key_vocab.values()) + 1, -1, dtype=np.int64)

    for number, group in enumerate(config.field_groups):
        of_key[[key_vocab[key] for key in group]] = number

    return FieldGroups(of_key, key_vocab[EVENT_TYPE] if config.event_type_event_only else -1)


_EMPTY = np.zeros(0, dtype=np.int64)

_NOTHING = Values(_EMPTY, _EMPTY, _EMPTY, _EMPTY)


def values_of(row: dict, targets_only: bool = True) -> list[Value]:
    """
    Допустимые значения клиента, в порядке последовательности.

    Событие допустимо, когда оно лежит в периоде целей своей
    группы (targets_only=False — любое событие).
    Внутри окна значение открывает positions == 0; нулевая позиция
    окна это маркер события, и разбор начинается сразу за ней.
    """

    found = _parse(row)

    if found is None:
        return _values_slowly(row, targets_only)

    if targets_only:
        found = found.take(_targets(row, found))

    return found.as_list()


def _targets(row: dict, found: Values) -> np.ndarray:
    return np.asarray(row["target_event_mask"], dtype=bool)[found.event]


def _parse(row: dict) -> Values | None:
    """
    Все значения всех событий разом, в порядке _values_slowly:
    событие за событием, внутри — по началам. None — строка не
    разбирается массивами (окно вне строки, маска короче событий):
    тогда её разбирает и называет ошибку прежний обход.
    """

    starts = np.asarray(row["event_starts"], dtype=np.int64)
    lengths = np.asarray(row["event_lengths"], dtype=np.int64)
    positions = np.asarray(row["positions"], dtype=np.int64)
    key_ids = np.asarray(row["key_ids"], dtype=np.int64)

    if len(row["target_event_mask"]) < starts.size or lengths.size != starts.size:
        return None

    # Окно события — range(start + 1, start + length).
    inner = np.maximum(lengths - 1, 0)
    total = int(inner.sum())

    if total == 0:
        return _NOTHING

    event_of = np.repeat(np.arange(starts.size, dtype=np.int64), inner)
    index = np.repeat(starts + 1, inner) + (np.arange(total, dtype=np.int64)
                                             - np.repeat(np.cumsum(inner) - inner, inner))

    if int(index.min()) < 0 or int(index.max()) >= min(positions.size, key_ids.size):
        return None

    opened = positions[index] == 0

    start = index[opened]
    event = event_of[opened]

    # Значение кончается там, где в том же событии открылось
    # следующее, иначе — в конце события.
    end = (starts + lengths)[event]
    same = np.zeros(start.size, dtype=bool)
    same[:-1] = event[1:] == event[:-1]
    end[:-1] = np.where(same[:-1], start[1:], end[:-1])

    return Values(event, key_ids[start], start, end - start)


def _values_slowly(row: dict, targets_only: bool) -> list[Value]:
    """
    Тот же разбор обходом: путь, на котором неправильная строка
    называет свою ошибку.
    """

    key_ids = row["key_ids"]
    positions = row["positions"]

    found: list[Value] = []

    for event, start in enumerate(row["event_starts"]):

        if targets_only and not row["target_event_mask"][event]:
            continue

        end = start + row["event_lengths"][event]

        opened = -1

        for index in range(start + 1, end):

            if positions[index] != 0:
                continue

            if opened >= 0:
                found.append(
                    Value(event, key_ids[opened], opened, index - opened)
                )

            opened = index

        if opened >= 0:
            found.append(Value(event, key_ids[opened], opened, end - opened))

    return found


def value_chance(value: Value, row: dict, config: MaskingConfig,
                 weights: ValueWeights | None) -> float:
    """
    Вероятность цели механизма value для одного значения.

    Без взвешивания — ровно value_probability, как прежде; со
    взвешиванием — по весам train для его ключа и токенов.
    """

    if not config.informativeness_weighted_masking:
        return config.value_probability

    if weights is None:
        raise WeightsError(
            "взвешенное value-маскирование требует весов train (value_weights.json словаря)"
        )

    tokens = tuple(row["value_ids"][value.start:value.start + value.length])

    return weights.probability(value.key_id, tokens, config)


def value_chances(values: Values, row: dict, config: MaskingConfig,
                  weights: ValueWeights | None) -> np.ndarray:
    """
    value_chance каждого значения разом.
    """

    if not config.informativeness_weighted_masking:
        return np.full(len(values), config.value_probability, dtype=np.float64)

    if weights is None:
        raise WeightsError(
            "взвешенное value-маскирование требует весов train (value_weights.json словаря)"
        )

    return weights.probabilities(values.key_id, values.start, values.length,
                                 np.asarray(row["value_ids"], dtype=np.int64), config)


def choose(group: str, row: dict, config: MaskingConfig,
           weights: ValueWeights | None = None, fields: FieldGroups | None = None) -> Selection:
    """
    Что спрятать у одного клиента.

    Клиент без допустимых целей даёт пустой выбор: это рабочий
    случай, а не ошибка. weights нужны при
    informativeness_weighted_masking; fields — при field_groups или
    event_type_event_only (field_groups()), без них выбор прежний.

    Разбор строки один на все розыгрыши: значения целей и значения
    вне целей (для порчи контекста) — части одного разбора, в том же
    порядке. Розыгрыши — те же потоки, в том же числе и порядке:
    каждый поток тянет столько чисел и в той очерёдности, что и
    розыгрыш по одному.
    """

    found = _parse(row)

    if found is None:
        # Строку массивами не разобрать: прежний порядок обходов —
        # значения целей, события периода, все значения — назовёт ошибку
        # там же, где раньше; разобранная строка идёт дальше как есть.
        _values_slowly(row, targets_only=True)
        [event for event in range(len(row["event_starts"])) if row["target_event_mask"][event]]
        found = Values.of(_values_slowly(row, targets_only=False))

    mask = np.asarray(row["target_event_mask"], dtype=bool)[:len(row["event_starts"])]

    eligible = np.flatnonzero(mask)

    inside = mask[found.event]

    values = found.take(inside)

    if not len(values):
        return Selection(int(eligible.size), values, _EMPTY, _EMPTY.astype(np.uint8),
                         np.zeros(0, dtype=bool), _NOTHING)

    # Ключ потока включает группу и клиента, поэтому розыгрыш не
    # зависит ни от порядка чтения, ни от того, в каком батче
    # клиент оказался.
    client = stable_hash(group, row["client_id"]) % (2 ** 31)

    def stream(number: int) -> KeyedRandom:
        return KeyedRandom((config.seed, number, client))

    # Событие разыгрывается, даже если значений у него нет: иначе
    # состав значений сдвигал бы розыгрыш соседних событий.
    chosen_events = np.zeros(mask.size, dtype=bool)
    chosen_events[eligible] = stream(_STREAM_EVENT).randoms(eligible.size) < config.event_probability

    # Ключи обходятся по возрастанию, а не в порядке встречи:
    # порядок розыгрыша не должен зависеть от того, какое событие
    # попалось первым.
    keys = np.unique(values.key_id)
    chosen_keys = stream(_STREAM_KEY).randoms(keys.size) < config.key_probability

    # Один розыгрыш на значение, как и без взвешивания: веса меняют
    # только вероятность, а не число и порядок розыгрышей.
    chances = value_chances(values, row, config, weights)
    chosen_values = stream(_STREAM_VALUE).randoms(len(values)) < chances

    # Сильнейшая причина побеждает: event > key > value.
    reasons = np.select(
        [chosen_events[values.event], chosen_keys[np.searchsorted(keys, values.key_id)], chosen_values],
        [REASONS.index(EVENT), REASONS.index(KEY), REASONS.index(VALUE)],
        REASONS.index(NONE),
    ).astype(np.uint8)

    picked = np.flatnonzero(reasons)
    codes = reasons[picked]

    unknown = stream(_STREAM_UNKNOWN).randoms(picked.size) < config.unknown_probability

    spoils = keys[chosen_keys]

    # Группы и event_type — только после всех розыгрышей: ни один поток
    # не тянет ни числом больше или меньше, поэтому решения и [UNK]
    # остальных значений те же, что без них.
    if fields is not None:

        # event_type — цель только механизма event: его выбор механизмами
        # key и value выбрасывается вместе с решением [UNK], а его
        # контекст не портится.
        kept = ~((values.key_id[picked] == fields.event_type) & (codes != REASONS.index(EVENT)))
        picked, codes, unknown = picked[kept], codes[kept], unknown[kept]
        spoils = spoils[spoils != fields.event_type]

        picked, codes, unknown = _close_groups(values, picked, codes, unknown, fields.of_key)

    # Выбранный ключ портится и в контексте: каждое его значение вне
    # целей уходит в [UNK] с вероятностью
    # key_context_corruption_probability, без метки. Поток свой у
    # каждой пары (клиент, ключ), и розыгрыш идёт по вхождениям ключа
    # в порядке ленты: решение не зависит ни от порядка чтения, ни от
    # других ключей, а цели и их розыгрыш не меняются ни на бит.
    candidates = np.flatnonzero(~inside & np.isin(found.key_id, spoils))

    spoiled = np.zeros(candidates.size, dtype=bool)

    for key_id in np.unique(found.key_id[candidates]).tolist():
        own = found.key_id[candidates] == key_id
        spoiled[own] = KeyedRandom((config.seed, _STREAM_CONTEXT, client, key_id)).randoms(
            int(own.sum())) < config.key_context_corruption_probability

    return Selection(int(eligible.size), values, picked, codes, unknown, found.take(candidates[spoiled]))


def _close_groups(values: Values, picked: np.ndarray, codes: np.ndarray,
                  unknown: np.ndarray, of_key: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Замыкание групп полей: цель value или key, чей ключ в группе,
    закрывает все ещё не выбранные значения ключей той же группы в том
    же событии. Партнёр получает сильнейшую причину своих источников
    (меньший ненулевой код REASONS) и не уходит в [UNK]: он закрыт ради
    цели, а не разыгран. Источник в [UNK] не цель и ничего не закрывает;
    event закрывает событие и так. Значения у события только
    присутствующие, поэтому отсутствующее поле группы не закрывается, а
    текст закрывается целиком — подстановка идёт по диапазону значения.
    """

    group = np.full(len(values), -1, dtype=np.int64)
    known = values.key_id < of_key.size
    group[known] = of_key[values.key_id[known]]

    sources = np.isin(codes, (REASONS.index(KEY), REASONS.index(VALUE))) & ~unknown & (group[picked] >= 0)

    if not sources.any():
        return picked, codes, unknown

    # Пара (событие, группа) одним числом.
    pair = values.event * (int(of_key.max()) + 1) + group

    source_pair = pair[picked[sources]]
    source_code = codes[sources]

    # Сильнейшая причина пары — первая после сортировки по паре и коду.
    order = np.lexsort((source_code, source_pair))
    pairs, first = np.unique(source_pair[order], return_index=True)
    strongest = source_code[order][first]

    free = np.ones(len(values), dtype=bool)
    free[picked] = False

    candidates = np.flatnonzero(free & (group >= 0))
    place = np.minimum(np.searchsorted(pairs, pair[candidates]), pairs.size - 1)
    hit = pairs[place] == pair[candidates]

    added = candidates[hit]

    merged = np.concatenate([picked, added])
    order = np.argsort(merged, kind="stable")

    return (
        merged[order],
        np.concatenate([codes, strongest[place[hit]]])[order],
        np.concatenate([unknown, np.zeros(added.size, dtype=bool)])[order],
    )


__all__ = [
    "EVENT",
    "KEY",
    "NONE",
    "VALUE",
    "Choice",
    "FieldGroups",
    "Selection",
    "Value",
    "choose",
    "field_groups",
    "value_chance",
    "values_of",
]
