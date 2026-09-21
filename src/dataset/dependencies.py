from __future__ import annotations

from dataclasses import dataclass, field

from src.preprocessing.semantic.keys import RELATION_KEYS, TIMING_KEYS

from .encoding import EncodedEvent, EncodedRecord


# ============================================================
# ИДЕЯ
# ============================================================
#
# Какое значение примера посчитано из какого другого значения.
#
# Это нужно будущему Masker, и нужно по существенной причине:
# спрятать сумму операции, оставив на виду её отношение к
# доходу, значит спрятать цель и тут же показать ответ. Список
# слагаемых обязан ехать вместе с примером.
#
# Зависимость выражается ИНДЕКСАМИ ЗНАЧЕНИЙ, а не именами
# ключей: одно и то же имя встречается в каждом событии, и
# адрес «значение номер N этого примера» однозначен, а имя нет.
#
# Статус называет, где находится источник:
#
#   in_context             в этом же примере, адрес известен
#   outside_context        событие видно банку, но в пример не
#                          попало по отбору; тождество осталось
#                          в служебной таблице
#   profile                значение анкеты, действующей на срез
#   profile_other_version  анкета другой версии: доход брался на
#                          момент операции, а в примере лежит
#                          анкета на срез
#   external               справочник, сущность или прошлое
#                          клиента целиком: позиционного адреса
#                          у такого источника нет
#   source_value_missing   событие найдено, а объявленного
#                          ключа в его записи нет
#   cause_event            источник это событие-причина ЦЕЛИКОМ:
#                          адрес события есть, одного значения
#                          у такого источника нет
#
# Отсюда правило для потребителя: in_context обещает ОБА адреса,
# cause_event только адрес события, остальные статусы ни одного.
# Говорить «источник найден» и «адреса нет» одновременно нельзя.
#
# Чего здесь сознательно НЕТ: источников интервалов. Признаки
# since_previous_hours и родственные зависят от ВРЕМЕНИ соседних
# событий, а не от их значений, и прятать вместе с ними нечего.
# Это записано в манифест набора, а не умолчано.
# ============================================================


DEP_IN_CONTEXT = 0
DEP_OUTSIDE_CONTEXT = 1
DEP_PROFILE = 2
DEP_PROFILE_OTHER_VERSION = 3
DEP_EXTERNAL = 4
DEP_SOURCE_VALUE_MISSING = 5
DEP_CAUSE_EVENT = 6

DEP_STATUSES: tuple[str, ...] = (
    "in_context",
    "outside_context",
    "profile",
    "profile_other_version",
    "external",
    "source_value_missing",
    "cause_event",
)

# Ключи, происхождение которых не отслеживается: они считаются
# из времени соседних записей, а не из их значений.
NOT_TRACKED: tuple[str, ...] = tuple(sorted(TIMING_KEYS))

# Признаки связи: их источник это событие-причина целиком.
RELATION_FEATURES: tuple[str, ...] = tuple(sorted(RELATION_KEYS))


class DependencyError(ValueError):
    """
    Происхождение значения разобрать нельзя.
    """


@dataclass
class Dependencies:
    """
    Пары «производное значение — его источник».
    """

    value: list[int] = field(default_factory=list)
    source_event: list[int] = field(default_factory=list)
    source_value: list[int] = field(default_factory=list)
    status: list[int] = field(default_factory=list)
    key: list[str] = field(default_factory=list)

    def add(self, value: int, source_event: int, source_value: int, status: int, key: str) -> None:
        self.value.append(value)
        self.source_event.append(source_event)
        self.source_value.append(source_value)
        self.status.append(status)
        self.key.append(key)

    def __len__(self) -> int:
        return len(self.value)

    def counts(self) -> dict[str, int]:

        out = {name: 0 for name in DEP_STATUSES}

        for code in self.status:
            out[DEP_STATUSES[code]] += 1

        return out


def _value_index(record: EncodedRecord, key: str) -> int | None:
    """
    Место значения этого ключа внутри записи.

    У ключа значение одно: пары строятся по ключу, и второй пары
    того же ключа в записи не бывает.
    """

    try:
        return record.value_keys.index(key)
    except ValueError:
        return None


def resolve(
    events: tuple[EncodedEvent, ...],
    kept: list[int],
    value_offsets: list[int],
    profile: EncodedRecord,
    profile_version,
    cause_of: dict[tuple[str, int], str | None],
) -> Dependencies:
    """
    Происхождение значений отобранных событий.

    events это ВСЯ видимая история: источник, не попавший в
    пример, обязан быть узнан как исключённый, а не потерян.

    cause_of приходит из истории НА ЭТОТ СРЕЗ и опознаётся парой
    «запись, версия»: лента клиента целиком содержала бы строки,
    которых на срезе ещё нет, и версии, которые ещё не
    действовали.
    """

    slot_of_position = {position: number for number, position in enumerate(kept)}

    position_of_identity = {
        (item.event_id, item.event_version): position for position, item in enumerate(events)
    }

    position_of_event_id: dict[str, int] = {}

    for position, item in enumerate(events):
        position_of_event_id[item.event_id] = position

    out = Dependencies()

    for slot, position in enumerate(kept):

        event = events[position]
        record = event.record

        base = value_offsets[slot]

        # --- расчётные значения ---

        for key, sources in sorted(event.provenance.items()):

            local = _value_index(record, key)

            if local is None:
                # Значение посчиталось, но в запись не попало:
                # такого быть не должно, и молчать об этом нельзя.
                raise DependencyError(
                    f"событие {event.event_id}: у значения {key} есть происхождение, "
                    "а самого значения в записи нет"
                )

            target = base + local

            for source in sources:
                _add_source(out, target, key, source, events, slot_of_position,
                            position_of_identity, value_offsets, profile, profile_version)

        # --- признаки связи ---

        cause_id = cause_of.get((event.event_id, event.event_version))

        if cause_id is None:
            continue

        cause_position = position_of_event_id.get(cause_id)

        if cause_position is None:
            # Причина не видна на этот срез, и связи у события
            # тогда нет вовсе: семантика её не построила.
            continue

        cause_slot = slot_of_position.get(cause_position)

        for key in RELATION_FEATURES:

            local = _value_index(record, key)

            if local is None:
                continue

            if cause_slot is None:
                out.add(base + local, -1, -1, DEP_OUTSIDE_CONTEXT, key)
                continue

            # Источник это всё событие-причина целиком: вид связи
            # и интервал считаются по нему, а не по одному его
            # значению. Поэтому и статус свой: адрес события есть,
            # адреса значения не существует.
            out.add(base + local, cause_slot, -1, DEP_CAUSE_EVENT, key)

    return out


def _add_source(
    out: Dependencies,
    target: int,
    key: str,
    source: dict,
    events: tuple[EncodedEvent, ...],
    slot_of_position: dict[int, int],
    position_of_identity: dict[tuple, int],
    value_offsets: list[int],
    profile: EncodedRecord,
    profile_version,
) -> None:

    kind = source.get("kind")

    if kind == "event":

        identity = (source.get("event_id"), source.get("event_version"))

        position = position_of_identity.get(identity)

        if position is None:
            raise DependencyError(
                f"значение {key} посчитано из записи {identity}, которой нет в видимой истории: "
                "происхождение указывает в пустоту"
            )

        slot = slot_of_position.get(position)

        if slot is None:
            out.add(target, -1, -1, DEP_OUTSIDE_CONTEXT, key)
            return

        local = _value_index(events[position].record, source.get("key"))

        if local is None:
            # «Источник найден» и «адреса источника нет» вместе
            # не говорят: потребитель, получив in_context, вправе
            # рассчитывать на индекс. Это отдельное состояние.
            out.add(target, slot, -1, DEP_SOURCE_VALUE_MISSING, key)
            return

        out.add(target, slot, value_offsets[slot] + local, DEP_IN_CONTEXT, key)
        return

    if kind == "profile":

        # Доход берётся по версии анкеты, действовавшей в момент
        # операции, а в примере лежит анкета на срез. Это разные
        # версии, и прятать их вместе нельзя.
        same = profile_version is not None and source.get("profile_version") == profile_version

        if not same:
            out.add(target, -1, -1, DEP_PROFILE_OTHER_VERSION, key)
            return

        local = _value_index(profile, source.get("key"))

        if local is None:
            out.add(target, -1, -1, DEP_SOURCE_VALUE_MISSING, key)
            return

        out.add(target, -1, local, DEP_PROFILE, key)
        return

    # Справочник, сущность и прошлое клиента целиком: у такого
    # источника нет одного значения, которое можно было бы
    # спрятать вместе с производным.
    out.add(target, -1, -1, DEP_EXTERNAL, key)


__all__ = [
    "DEP_CAUSE_EVENT",
    "DEP_EXTERNAL",
    "DEP_IN_CONTEXT",
    "DEP_OUTSIDE_CONTEXT",
    "DEP_PROFILE",
    "DEP_PROFILE_OTHER_VERSION",
    "DEP_SOURCE_VALUE_MISSING",
    "DEP_STATUSES",
    "NOT_TRACKED",
    "RELATION_FEATURES",
    "Dependencies",
    "DependencyError",
    "resolve",
]
