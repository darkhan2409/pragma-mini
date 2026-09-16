from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from ..life.persona import Persona
from ..rng import NS_SUPPORT, event_rng, stable_hash
from ..world.dictionaries import SUPPORT_CHANNELS, SUPPORT_TOPICS


# ============================================================
# ОБСЛУЖИВАНИЕ
# ============================================================
#
# Обращение связано с ПРИЧИНОЙ: неуспешной операцией, спорной
# транзакцией, блокировкой карты, ошибкой приложения, платежом
# по кредиту, изменением данных или жалобой.
#
# Решение обращения порождает последствие: разблокировку,
# возврат, исправление записи или повторный контакт.
# ============================================================


TOPIC_BY_CAUSE = {
    "failed_operation": "operation_question",
    "declined_payment": "operation_question",
    "card_blocked": "card_block",
    "fraud_alert": "fraud_report",
    "disputed_transaction": "dispute",
    "app_error": "app_error",
    "missed_installment": "loan_payment",
    "delinquency": "loan_restructure",
    "profile_change": "data_change",
    "statement": "statement_request",
    "product_question": "product_question",
    "deposit_question": "deposit_question",
    "complaint": "complaint",
}

RESOLUTION_BY_TOPIC = {
    "operation_question": ("explained", "record_corrected", "escalated"),
    "card_block": ("card_unblocked", "card_reissued", "explained"),
    "dispute": ("chargeback_started", "refund_issued", "declined"),
    "fraud_report": ("card_reissued", "chargeback_started", "escalated"),
    "app_error": ("explained", "escalated"),
    "loan_payment": ("explained", "record_corrected"),
    "loan_restructure": ("escalated", "declined", "explained"),
    "data_change": ("record_corrected", "document_sent"),
    "complaint": ("explained", "escalated", "declined"),
    "statement_request": ("document_sent",),
    "product_question": ("explained",),
    "deposit_question": ("explained", "document_sent"),
}

RESOLUTION_WEIGHTS = {
    "card_block": (0.58, 0.30, 0.12),
    "dispute": (0.52, 0.30, 0.18),
    "fraud_report": (0.55, 0.30, 0.15),
}


@dataclass(frozen=True)
class CasePlan:
    case_id: str
    topic: str
    channel: str
    opened_at: datetime
    updated_at: datetime | None
    resolved_at: datetime
    resolution: str
    cause_event_id: str | None


def contact_probability(persona: Persona, cause: str, ts: datetime, stress: float) -> float:
    """
    Вероятность, что клиент обратится в поддержку.
    """

    settings = params_module.active().stress

    base = {
        "failed_operation": 0.06,
        "declined_payment": 0.09,
        "card_blocked": 0.42,
        "fraud_alert": 0.55,
        "disputed_transaction": 0.85,
        "app_error": 0.05,
        "missed_installment": 0.08,
        "delinquency": 0.16,
        "profile_change": 0.25,
        "statement": 0.10,
    }.get(cause, 0.05)

    base *= 0.5 + 1.2 * persona.trait("sociality", ts)

    if cause in ("missed_installment", "delinquency"):
        base *= 1.0 + (settings.support_contact_boost - 1.0) * stress

    return float(min(0.95, base))


def open_case(
    persona: Persona,
    cause: str,
    ts: datetime,
    cause_event_id: str | None,
    index: int,
) -> CasePlan:
    """
    Обращение с исходом и сроком решения.
    """

    rng = event_rng(NS_SUPPORT, persona.client_ordinal, ts.toordinal(), index, 3)

    topic = TOPIC_BY_CAUSE.get(cause, "product_question")

    preferences = persona.traits.channel_preferences

    channel_weights = {
        "chat": 0.45 + 0.9 * preferences.get("app", 0.3),
        "call_center": 0.35 + 1.2 * preferences.get("call_center", 0.05),
        "branch": 0.12 + 1.5 * preferences.get("branch", 0.05),
        "email": 0.08,
    }

    channel = rng.weighted(channel_weights)

    options = RESOLUTION_BY_TOPIC.get(topic, ("explained",))

    weights = RESOLUTION_WEIGHTS.get(topic)

    resolution = str(rng.choice(list(options), p=list(weights) if weights else None))

    hours = rng.integers(1, 96)

    updated = ts + timedelta(hours=int(rng.integers(1, max(2, hours))))

    return CasePlan(
        case_id=f"case_{stable_hash('case', persona.client_id, ts.toordinal(), index) % 10 ** 11:011d}",
        topic=topic,
        channel=channel,
        opened_at=ts,
        updated_at=updated if rng.random() < 0.55 else None,
        resolved_at=ts + timedelta(hours=int(hours)),
        resolution=resolution,
        cause_event_id=cause_event_id,
    )


__all__ = [
    "RESOLUTION_BY_TOPIC",
    "TOPIC_BY_CAUSE",
    "CasePlan",
    "contact_probability",
    "open_case",
]
