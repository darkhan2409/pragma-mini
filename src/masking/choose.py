from __future__ import annotations

from dataclasses import dataclass

from src.generator.rng import KeyedRandom, stable_hash

from .settings import MaskingConfig
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
class Selection:
    """
    Что нашлось у клиента и что из этого выбрано.

    Допустимые события и значения возвращаются вместе с выбором,
    чтобы отчёт не разбирал ту же строку второй раз. corrupted —
    значения выбранного механизмом key ключа вне целей, испорченные
    в [UNK] без метки (key_context_corruption_probability).
    """

    events: int
    values: list["Value"]
    choices: list["Choice"]
    corrupted: tuple["Value", ...] = ()


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


def values_of(row: dict, targets_only: bool = True) -> list[Value]:
    """
    Допустимые значения клиента, в порядке последовательности.

    Событие допустимо, когда оно лежит в периоде целей своей
    группы (targets_only=False — любое событие).
    Внутри окна значение открывает positions == 0; нулевая позиция
    окна это маркер события, и разбор начинается сразу за ней.
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


def choose(group: str, row: dict, config: MaskingConfig,
           weights: ValueWeights | None = None) -> Selection:
    """
    Что спрятать у одного клиента.

    Клиент без допустимых целей даёт пустой выбор: это рабочий
    случай, а не ошибка. weights нужны при
    informativeness_weighted_masking.
    """

    values = values_of(row)

    eligible = [
        event
        for event in range(len(row["event_starts"]))
        if row["target_event_mask"][event]
    ]

    if not values:
        return Selection(len(eligible), values, [])

    # Ключ потока включает группу и клиента, поэтому розыгрыш не
    # зависит ни от порядка чтения, ни от того, в каком батче
    # клиент оказался.
    client = stable_hash(group, row["client_id"]) % (2 ** 31)

    def stream(number: int) -> KeyedRandom:
        return KeyedRandom((config.seed, number, client))

    # Событие разыгрывается, даже если значений у него нет: иначе
    # состав значений сдвигал бы розыгрыш соседних событий.
    events = stream(_STREAM_EVENT)

    chosen_events = {
        event: events.chance(config.event_probability) for event in eligible
    }

    # Ключи обходятся по возрастанию, а не в порядке встречи:
    # порядок розыгрыша не должен зависеть от того, какое событие
    # попалось первым.
    keys = stream(_STREAM_KEY)

    chosen_keys = {
        key_id: keys.chance(config.key_probability)
        for key_id in sorted({value.key_id for value in values})
    }

    # Один розыгрыш на значение, как и без взвешивания: веса меняют
    # только вероятность, а не число и порядок розыгрышей.
    singles = stream(_STREAM_VALUE)

    chosen_values = [
        singles.chance(value_chance(value, row, config, weights)) for value in values
    ]

    unknown = stream(_STREAM_UNKNOWN)

    picked: list[Choice] = []

    for index, value in enumerate(values):

        if chosen_events[value.event]:
            reason = EVENT
        elif chosen_keys[value.key_id]:
            reason = KEY
        elif chosen_values[index]:
            reason = VALUE
        else:
            continue

        picked.append(
            Choice(value, reason, unknown.chance(config.unknown_probability))
        )

    # Выбранный ключ портится и в контексте: каждое его значение вне
    # целей уходит в [UNK] с вероятностью
    # key_context_corruption_probability, без метки. Поток свой у
    # каждой пары (клиент, ключ), и розыгрыш идёт по вхождениям ключа
    # в порядке ленты: решение не зависит ни от порядка чтения, ни от
    # других ключей, а цели и их розыгрыш не меняются ни на бит.
    targets = set(eligible)
    streams: dict[int, KeyedRandom] = {}
    corrupted: list[Value] = []

    for value in values_of(row, targets_only=False):

        if value.event in targets or not chosen_keys.get(value.key_id, False):
            continue

        if value.key_id not in streams:
            streams[value.key_id] = KeyedRandom((config.seed, _STREAM_CONTEXT, client, value.key_id))

        if streams[value.key_id].chance(config.key_context_corruption_probability):
            corrupted.append(value)

    return Selection(len(eligible), values, picked, tuple(corrupted))


__all__ = [
    "EVENT",
    "KEY",
    "NONE",
    "VALUE",
    "Choice",
    "Selection",
    "Value",
    "choose",
    "value_chance",
    "values_of",
]
