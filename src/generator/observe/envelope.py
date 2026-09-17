from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from ..config import (
    EVENT_TYPE_SOURCE,
    INITIATOR_SYSTEM,
    PAYLOAD_FIELDS,
    SOURCE_PRECISION,
)
from ..rng import stable_hash


# ============================================================
# КОНВЕРТ СОБЫТИЯ
# ============================================================
#
#   event_time      когда действие или факт произошли
#   record_time     когда запись стала доступна банку
#   effective_at    с какого момента действует изменение
#   time_precision  точность времени источника
#   event_version   версия исправляемой записи
#   correlation_id  связь частей одной бизнес-цепочки
#   link_type       тип связи
#
# Исправление сохраняет event_id, увеличивает event_version и
# получает новый record_time. Повторная доставка той же записи
# отличается от нового бизнес-события именно этим.
#
# Неприменимое поле отсутствует или равно null. Заполнять его
# случайным значением ради плотности таблицы запрещено.
# ============================================================


@dataclass
class Event:
    event_id: str
    client_id: str
    event_type: str
    source: str
    event_time: datetime
    payload: dict
    record_time: datetime | None = None
    effective_at: datetime | None = None
    time_precision: str = "second"
    sequence_number: int = 0
    event_version: int = 1
    change_initiator: str = INITIATOR_SYSTEM
    correlation_id: str | None = None
    link_type: str | None = None
    is_test_account: bool = False

    def copy_as_duplicate(self, record_time: datetime) -> Event:
        """
        Повторная доставка той же записи: версия прежняя,
        record_time другой.
        """

        return replace(self, record_time=record_time, link_type="duplicate")

    def copy_as_correction(self, record_time: datetime, payload: dict) -> Event:
        """
        Исправление: тот же event_id, версия выше, новый
        record_time.
        """

        return replace(
            self,
            payload=payload,
            record_time=record_time,
            event_version=self.event_version + 1,
            link_type="correction",
        )


class EventFactory:
    """
    Выдаёт события одного клиента со стабильными
    идентификаторами.
    """

    def __init__(self, client_id: str, is_test_account: bool = False) -> None:
        self.client_id = client_id
        self.is_test_account = is_test_account
        self._counters: dict[str, int] = {}

    def next_id(self, source: str) -> str:
        index = self._counters.get(source, 0) + 1
        self._counters[source] = index
        return f"ev{stable_hash(self.client_id, source, index) % 10 ** 15:015d}"

    def make(
        self,
        event_type: str,
        ts: datetime,
        payload: dict,
        initiator: str = INITIATOR_SYSTEM,
        correlation_id: str | None = None,
        link_type: str | None = None,
        effective_at: datetime | None = None,
        precision: str | None = None,
    ) -> Event:

        source = EVENT_TYPE_SOURCE[event_type]

        allowed = PAYLOAD_FIELDS[event_type]

        clean = {name: payload.get(name) for name in allowed}

        return Event(
            event_id=self.next_id(source),
            client_id=self.client_id,
            event_type=event_type,
            source=source,
            event_time=ts,
            payload=clean,
            record_time=None,
            effective_at=effective_at if effective_at is not None else ts,
            time_precision=precision or SOURCE_PRECISION[source],
            change_initiator=initiator,
            correlation_id=correlation_id,
            link_type=link_type,
            is_test_account=self.is_test_account,
        )


def round_to_precision(ts: datetime, precision: str) -> datetime:
    """
    Точность источника: витрина кредитного обслуживания вовсе
    теряет время, коммуникации округлены до минуты.
    """

    if precision == "day":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)

    if precision == "month":
        return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if precision == "minute":
        return ts.replace(second=0, microsecond=0)

    return ts.replace(microsecond=0)


__all__ = ["Event", "EventFactory", "round_to_precision"]
