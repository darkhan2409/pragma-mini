from __future__ import annotations

import json
from datetime import datetime

from .config import EVENT_TYPE_PRIORITY


# ============================================================
# ЛЕНТА СОБЫТИЙ
# ============================================================
#
# Единая лента это основной выход генератора. Типизированных
# таблиц по источникам нет: источник это колонка конверта, а
# состав payload описан каталогом ключей в манифесте.
#
# Порядок при одинаковом event_time задаётся приоритетом типа
# события и отражает причинность.
# ============================================================


ENVELOPE_COLUMNS = (
    "event_id",
    "client_id",
    "event_type",
    "source",
    "event_time",
    "record_time",
    "effective_at",
    "time_precision",
    "sequence_number",
    "event_version",
    "change_initiator",
    "correlation_id",
    "link_type",
    "is_test_account",
    "payload",
)


def payload_json(payload: dict) -> str:
    """
    Компактный JSON с устойчивым порядком ключей.
    """

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def sort_key(event) -> tuple:
    """
    Порядок ленты клиента: время, затем причинный приоритет
    типа события, затем версия записи.
    """

    return (
        event.event_time,
        EVENT_TYPE_PRIORITY.get(event.event_type, 99),
        event.event_version,
        event.event_id,
    )


def event_row(event) -> dict:
    """
    Строка ленты: конверт плюс payload в JSON.
    """

    return {
        "event_id": event.event_id,
        "client_id": event.client_id,
        "event_type": event.event_type,
        "source": event.source,
        "event_time": event.event_time,
        "record_time": event.record_time,
        "effective_at": event.effective_at,
        "time_precision": event.time_precision,
        "sequence_number": event.sequence_number,
        "event_version": event.event_version,
        "change_initiator": event.change_initiator,
        "correlation_id": event.correlation_id,
        "link_type": event.link_type,
        "is_test_account": event.is_test_account,
        "payload": payload_json(event.payload),
    }


def parse_payload(row: dict) -> dict:
    return json.loads(row["payload"])


__all__ = [
    "ENVELOPE_COLUMNS",
    "event_row",
    "parse_payload",
    "payload_json",
    "sort_key",
]
