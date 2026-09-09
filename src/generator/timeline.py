from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import (
    EVENT_TYPE_BY_SOURCE,
    EVENT_TYPE_PRIORITY,
    MAX_TOKENS_PER_EVENT,
    PROFILE_DYNAMIC_FIELDS,
)
from .history import STREAM_FIELDS, ClientHistory


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единая лента событий клиента:
#
#     client_id | ts | event_type | payload
#
# Плюс seq: порядковый номер в ленте. Он нужен только для
# детерминированного tie-break и информации не несёт.
#
# Совпадающие ts неизбежны: баннер показывается вместе с экраном,
# договоры без времени падают в полночь, зарплата и покупка
# могут прийти в одну секунду. Порядок при равном ts задаётся
# приоритетом типа события из config.EVENT_TYPES, а внутри
# одного типа порядком события в своём потоке.
#
# Профиль попадает в ленту компактно: только меняющиеся поля.
# Полный срез читается Profile Encoder-ом из profile.parquet.
# ============================================================


@dataclass(frozen=True)
class TimelineEvent:
    client_id: int
    ts: datetime
    seq: int
    event_type: str
    payload: dict[str, Any]


# ============================================================
# PAYLOAD
# ============================================================


def transaction_payload(event) -> dict[str, Any]:
    return {
        "amount": event.amount,
        "direction": event.direction,
        "mcc": event.mcc,
        "merchant_city": event.merchant_city,
        "merchant_country": event.merchant_country,
        "is_online": event.is_online,
        "is_subscription": event.is_subscription,
    }


def product_payload(event) -> dict[str, Any]:
    return {
        "product_type": event.product_type,
        "amount_or_limit": event.amount_or_limit,
        "term": event.term,
        "product_subtype": event.product_subtype,
        "timestamp_quality": event.timestamp_quality,
    }


def communication_payload(event) -> dict[str, Any]:
    return {
        "channel": event.channel,
        "template": event.template,
        "day_of_week": event.day_of_week,
        "hour": event.hour,
        "delivered": event.delivered,
    }


def screen_payload(event) -> dict[str, Any]:
    return {
        "session_id": event.session_id,
        "firebase_screen": event.firebase_screen,
        "product": event.product,
        "funnel_stage": event.funnel_stage,
        "reject_reason": event.reject_reason,
    }


def operation_payload(event) -> dict[str, Any]:
    return {
        "domain": event.domain,
        "operation": event.operation,
        "status": event.status,
    }


def banner_payload(event) -> dict[str, Any]:
    return {
        "slot": event.slot,
        "offer": event.offer,
        "action": event.action,
    }


def profile_payload(snapshot) -> dict[str, Any]:
    """
    В ленту уходят только меняющиеся поля профиля.
    """

    return {field: snapshot.values.get(field) for field in PROFILE_DYNAMIC_FIELDS}


PAYLOAD_BUILDERS = {
    "profile": profile_payload,
    "transactions": transaction_payload,
    "product_events": product_payload,
    "communications": communication_payload,
    "app_screens": screen_payload,
    "app_operations": operation_payload,
    "banners": banner_payload,
}


# ------------------------------------------------------------
# РЕВИЗИЯ 2
# ------------------------------------------------------------
#
# session_id идёт первым: порядок ключей payload обязан
# совпадать с порядком колонок таблицы, а там он стоит сразу
# после ts. Это сверяет и preprocessing (_sample_key_order),
# и тест соответствия ленты таблицам.
# ------------------------------------------------------------


def operation_payload_r2(event) -> dict[str, Any]:
    return {
        "session_id": event.session_id,
        "domain": event.domain,
        "operation": event.operation,
        "status": event.status,
    }


def banner_payload_r2(event) -> dict[str, Any]:
    return {
        "session_id": event.session_id,
        "slot": event.slot,
        "offer": event.offer,
        "action": event.action,
    }


PAYLOAD_BUILDERS_R2 = {
    **PAYLOAD_BUILDERS,
    "app_operations": operation_payload_r2,
    "banners": banner_payload_r2,
}

BUILDERS_BY_REVISION = {1: PAYLOAD_BUILDERS, 2: PAYLOAD_BUILDERS_R2}


def payload_builders(revision: int) -> dict:

    if revision not in BUILDERS_BY_REVISION:
        raise ValueError(f"неизвестная ревизия схемы RAW: {revision!r}")

    return BUILDERS_BY_REVISION[revision]


# ============================================================
# СБОРКА
# ============================================================


def build_timeline(history: ClientHistory) -> list[TimelineEvent]:
    """
    Хронологическая лента всех потоков клиента.
    """

    rows: list[tuple[datetime, int, int, str, dict[str, Any]]] = []

    # Набор полей payload задаёт ревизия схемы, а её знает сама
    # история: передавать ревизию отдельным аргументом значило
    # бы позволить ленте разойтись с таблицами.
    builders = payload_builders(history.revision)

    for source in STREAM_FIELDS:

        event_type = EVENT_TYPE_BY_SOURCE[source]
        priority = EVENT_TYPE_PRIORITY[event_type]
        build = builders[source]

        for index, event in enumerate(history.events(source)):
            rows.append((event.ts, priority, index, event_type, build(event)))

    rows.sort(key=lambda row: (row[0], row[1], row[2]))

    return [
        TimelineEvent(
            client_id=history.client_id,
            ts=ts,
            seq=seq,
            event_type=event_type,
            payload=payload,
        )
        for seq, (ts, _, _, event_type, payload) in enumerate(rows)
    ]


def payload_json(payload: dict[str, Any]) -> str:
    """
    Компактный JSON с устойчивым порядком ключей.
    """

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def timeline_rows(history: ClientHistory) -> list[dict[str, Any]]:

    return [
        {
            "client_id": event.client_id,
            "ts": event.ts,
            "seq": event.seq,
            "event_type": event.event_type,
            "payload": payload_json(event.payload),
        }
        for event in build_timeline(history)
    ]


def max_payload_fields() -> int:
    """
    Сколько полей payload у самого широкого типа события.
    Проверяется тестом против MAX_TOKENS_PER_EVENT.
    """

    return max(
        len(PROFILE_DYNAMIC_FIELDS),
        7,  # transactions
        5,  # product_events / app_screens
        5,  # communications
        4,  # app_operations / banners в ревизии 2
    )


assert max_payload_fields() + 2 <= MAX_TOKENS_PER_EVENT, (
    "MAX_TOKENS_PER_EVENT меньше, чем требуется самому широкому событию"
)
