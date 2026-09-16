from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from ..config import HISTORY_END, HISTORY_START, SOURCES, SOURCE_AVAILABILITY
from ..life.persona import Persona
from ..rng import NS_COVERAGE, keyed_rng, stable_hash, state_cache


# ============================================================
# ПОКРЫТИЕ ИСТОЧНИКОВ
# ============================================================
#
# Покрытие отличает «события не было» от «источник ещё не
# подключён», «клиент в источнике не появился», «согласия нет»
# и «источник отвалился».
#
# opening_state описывает то, что было ДО первого наблюдения,
# и не выдаётся за наблюдавшееся прошлое.
# ============================================================


STATUS_FULL = "full"
STATUS_PARTIAL = "partial"
STATUS_LATE_START = "late_start"
STATUS_ENDED = "ended"
STATUS_NONE = "none"

REASON_NOT_CONNECTED = "source_not_connected"
REASON_NOT_ONBOARDED = "client_not_onboarded"
REASON_NO_CONSENT = "no_consent"
REASON_OUTAGE = "source_outage"
REASON_RELATIONSHIP_CLOSED = "relationship_closed"
REASON_TEST_ACCOUNT = "test_account"

APP_SOURCES = ("app_screens", "app_operations", "banners")

CORE_SOURCES = ("profile", "applications", "product_events", "loans", "transactions", "support", "antifraud")


@dataclass(frozen=True)
class Coverage:
    client_id: str
    source: str
    first_available_at: datetime
    last_available_at: datetime | None
    first_seen: datetime | None
    coverage_status: str
    coverage_reason: str | None
    opening_state: str | None


@state_cache
def app_adoption(client_ordinal: int) -> datetime | None:
    """
    Когда клиент установил приложение. Часть когорты не
    устанавливает его никогда.
    """

    from ..life.persona import draw_persona

    settings = params_module.active().defects

    persona = draw_persona(client_ordinal)

    rng = keyed_rng(NS_COVERAGE, client_ordinal, 1)

    digital = persona.trait("digital_affinity")

    probability = settings.app_adoption_share * (
        settings.app_adoption_digital_factor + (1.0 - settings.app_adoption_digital_factor) * 0.5 + digital * 0.7
    )

    if rng.random() >= min(0.99, probability):
        return None

    start = max(HISTORY_START, persona.relationship_start)

    offset = int(rng.integers(0, 420)) - 180

    adopted = start + timedelta(days=max(0, offset))

    if adopted >= HISTORY_END:
        return None

    return adopted.replace(hour=0, minute=0, second=0, microsecond=0)


@state_cache
def consent_date(client_ordinal: int) -> datetime | None:
    """
    Когда клиент дал согласие на коммуникации.
    """

    from ..life.persona import draw_persona

    settings = params_module.active().defects

    persona = draw_persona(client_ordinal)

    if not persona.consent_marketing:
        return None

    rng = keyed_rng(NS_COVERAGE, client_ordinal, 2)

    if rng.random() >= settings.consent_share:
        return None

    start = persona.relationship_start

    given = start + timedelta(days=int(rng.integers(0, 200)))

    if given >= HISTORY_END:
        return None

    return given.replace(hour=0, minute=0, second=0, microsecond=0)


def client_start(persona: Persona, source: str) -> datetime | None:

    if source in APP_SOURCES:
        return app_adoption(persona.client_ordinal)

    if source == "communications":
        return consent_date(persona.client_ordinal)

    return persona.relationship_start


def first_seen(persona: Persona, source: str) -> datetime | None:

    start = client_start(persona, source)

    if start is None:
        return None

    moment = max(SOURCE_AVAILABILITY[source], start)

    if moment >= HISTORY_END:
        return None

    return moment


@state_cache
def source_outages(client_ordinal: int, source: str) -> tuple:
    """
    Дни, когда источник не доносил записи вовсе.
    """

    settings = params_module.active().defects

    per_year = settings.outage_days_per_year.get(source, 0.0)

    if per_year <= 0.0:
        return ()

    rng = keyed_rng(NS_COVERAGE, client_ordinal, 3, stable_hash(source) % 97)

    span_days = (HISTORY_END - SOURCE_AVAILABILITY[source]).days

    if span_days <= 0:
        return ()

    count = rng.poisson(per_year * span_days / 365.25)

    days = []

    for index in range(min(count, 12)):
        item = keyed_rng(NS_COVERAGE, client_ordinal, 4, index)
        offset = int(item.integers(0, span_days))
        days.append((SOURCE_AVAILABILITY[source] + timedelta(days=offset)).date())

    return tuple(sorted(set(days)))


def in_outage(client_ordinal: int, source: str, ts: datetime) -> bool:
    return ts.date() in source_outages(client_ordinal, source)


def coverage_rows(persona: Persona, opening: dict | None = None, closed_at: datetime | None = None) -> list:
    """
    Строка покрытия на каждую пару клиент и источник.
    """

    rows: list[Coverage] = []

    for source in SOURCES:

        available = SOURCE_AVAILABILITY[source]

        seen = first_seen(persona, source)

        outages = source_outages(persona.client_ordinal, source)

        last_available: datetime | None = None
        status = STATUS_FULL
        reason: str | None = None

        if persona.is_test_account:
            status = STATUS_PARTIAL
            reason = REASON_TEST_ACCOUNT

        if seen is None:

            status = STATUS_NONE

            if source in APP_SOURCES:
                reason = REASON_NOT_ONBOARDED
            elif source == "communications":
                reason = REASON_NO_CONSENT
            else:
                reason = REASON_NOT_CONNECTED

        else:

            if seen > available:
                status = STATUS_LATE_START
                reason = (
                    REASON_NOT_ONBOARDED
                    if source in APP_SOURCES or persona.relationship_start > available
                    else REASON_NOT_CONNECTED
                )

            if outages and status in (STATUS_FULL, STATUS_LATE_START):
                status = STATUS_PARTIAL
                reason = REASON_OUTAGE

            if closed_at is not None and closed_at < HISTORY_END:
                status = STATUS_ENDED
                reason = REASON_RELATIONSHIP_CLOSED
                last_available = closed_at

        rows.append(
            Coverage(
                client_id=persona.client_id,
                source=source,
                first_available_at=available,
                last_available_at=last_available,
                first_seen=seen,
                coverage_status=status,
                coverage_reason=reason,
                opening_state=(
                    json.dumps(opening.get(source), ensure_ascii=False, default=str)
                    if opening and opening.get(source)
                    else None
                ),
            )
        )

    return rows


__all__ = [
    "APP_SOURCES",
    "Coverage",
    "REASON_NOT_CONNECTED",
    "REASON_NOT_ONBOARDED",
    "REASON_NO_CONSENT",
    "REASON_OUTAGE",
    "REASON_RELATIONSHIP_CLOSED",
    "REASON_TEST_ACCOUNT",
    "STATUS_ENDED",
    "STATUS_FULL",
    "STATUS_LATE_START",
    "STATUS_NONE",
    "STATUS_PARTIAL",
    "app_adoption",
    "consent_date",
    "coverage_rows",
    "first_seen",
    "in_outage",
    "source_outages",
]
