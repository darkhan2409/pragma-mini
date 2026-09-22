from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.preprocessing.history import history_as_of
from src.preprocessing.semantic.as_of import SemanticHistory
from src.tokenization.encode import (
    EncodedRecord,
    absent_reasons,
    encode_event,
    encode_profile,
    event_identity,
    profile_known,
    provenance,
    references,
)
from src.tokenization.layout import FrozenArtifacts


# ============================================================
# ИДЕЯ
# ============================================================
#
# Тонкий слой между смысловой историей и примером: история на
# дату превращается в закодированные записи вместе с тем, что
# едет рядом с каждой из них.
#
# Здесь КОДИРУЕТСЯ ВСЯ видимая история, до всякого отбора.
# Порядок именно такой не случайно:
#
#   - интервалы и расчёты семантика посчитала по полной истории,
#     и отбор контекста их не меняет. Кодировать после отбора
#     значило бы делать вид, что исключённых событий не было;
#   - отбор по бюджету токенов требует знать длину каждого
#     события, а длина известна только после кодирования;
#   - зависимость на исключённое событие должна знать, куда она
#     указывала, а не просто потеряться.
#
# Ничего не обучается и не решается: правила приходят готовыми
# из замороженного комплекта.
# ============================================================


class EncodingError(ValueError):
    """
    История на дату закодирована быть не может.
    """


@dataclass(frozen=True)
class EncodedEvent:
    """
    Одно событие: его токены и всё, что едет рядом.
    """

    record: EncodedRecord

    event_id: str
    stable_event_index: int
    event_time: datetime
    source: str
    event_type: str | None
    calendar: tuple[float, ...]

    refs: dict = field(default_factory=dict)
    absent_reasons: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return self.record.n_tokens

    @property
    def n_values(self) -> int:
        return self.record.n_values


@dataclass(frozen=True)
class EncodedHistory:
    """
    Вся видимая история клиента на дату в закодированном виде.
    """

    client_id: str
    cutoff: datetime
    events: tuple[EncodedEvent, ...]
    profile: EncodedRecord
    profile_meta: dict
    has_profile: bool
    relationship: object
    limitations: tuple[str, ...]

    # Причина каждой видимой записи по ключу «запись, версия».
    # Пусто, когда причины нет или она на этот срез не видна.
    cause_of: dict[tuple[str, int], str | None] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.events)


def causes_as_of(store, client_id: str, cutoff: datetime) -> dict[tuple[str, int], str | None]:
    """
    Причина каждой видимой записи: «запись, версия» → причина.

    Читается через историю на дату, а не по всей ленте клиента.
    Разница принципиальная: лента содержит и строки, которых на
    этот срез ещё не существовало, и версии, которые ещё не
    действовали. Карта по одному event_id смешала бы версии, а
    карта по всей ленте протащила бы в пример знание из будущего.

    Стоит это второго прохода истории на дату, примерно шестую
    часть времени смыслового слоя. Дешевле нет: смысловой слой
    идентификатор причины наружу не отдаёт, а трогать его ради
    этого нельзя.
    """

    table = history_as_of(store, client_id, cutoff).events

    if table.num_rows == 0:
        return {}

    columns = table.select(["event_id", "cause_event_id"]).to_pylist()

    return {
        row["event_id"]: row["cause_event_id"]
        for row in columns
    }


def encode_history(
    artifacts: FrozenArtifacts,
    history: SemanticHistory,
    limit: int,
    cause_of: dict[tuple[str, int], str | None] | None = None,
) -> EncodedHistory:
    """
    Смысловая история на дату в закодированном виде.

    Второй реализации кодирования здесь нет: событие и профиль
    кодирует токенизатор, а этот слой только собирает результат
    вместе с трассировкой.
    """

    identity = event_identity(history)

    # Причина, по которой у связи нет длительности, лежит рядом с
    # самой связью, а не в событии: источник объявил время грубее
    # разницы, и порядок двух записей неизвестен.
    reason_of_event = {
        item.stable_event_index: item.reason
        for item in history.relations
        if getattr(item, "reason", None)
    }

    events: list[EncodedEvent] = []

    previous: int | None = None

    for event in history.events:

        # Порядок истории это ДЕЛОВОЙ порядок, и задаёт его
        # stable_event_index, а не время записи.
        #
        # Разница не теоретическая. Исправление, датированное
        # позже исходного события, меняет время видимой записи, но
        # остаётся на месте первой версии: у такого клиента
        # event_time вдоль истории шагает назад, и это законно.
        # Проверять надо то, что действительно обязано расти.
        if previous is not None and event.stable_event_index <= previous:
            raise EncodingError(
                f"клиент {history.client_id}: событие {event.event_id} нарушает деловой порядок "
                f"истории (stable_event_index {event.stable_event_index} после {previous})"
            )

        previous = event.stable_event_index

        events.append(
            EncodedEvent(
                record=encode_event(artifacts, event, limit),
                event_id=event.event_id,
                stable_event_index=event.stable_event_index,
                event_time=event.event_time,
                source=event.source,
                event_type=event.values.get("event_type"),
                calendar=tuple(event.calendar),
                refs=references(artifacts, event),
                absent_reasons=absent_reasons(event, reason_of_event.get(event.stable_event_index)),
                provenance=provenance(event, identity),
            )
        )

    if len(events) != history.n_events:
        raise EncodingError(
            f"клиент {history.client_id}: закодировано {len(events)} событий из {history.n_events}"
        )

    return EncodedHistory(
        client_id=history.client_id,
        cutoff=history.cutoff,
        events=tuple(events),
        profile=encode_profile(artifacts, history, limit),
        profile_meta=dict(history.profile_meta or {}),
        has_profile=profile_known(history),
        relationship=history.relationship,
        limitations=tuple(history.limitations),
        cause_of=dict(cause_of or {}),
    )


__all__ = [
    "EncodedEvent",
    "EncodedHistory",
    "EncodingError",
    "causes_as_of",
    "encode_history",
]
