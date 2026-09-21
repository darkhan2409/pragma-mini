from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..config import SOURCES, SOURCE_AVAILABILITY
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
    outage_days: str | None


@state_cache
def app_adoption(client_ordinal: int) -> datetime | None:
    """
    Когда клиент установил приложение.

    Приложение — норма, а не исключение: его ставит подавляющее
    большинство. Небольшая доля не ставит никогда, и это не
    дефект данных, а часть жизни: остаются люди, которые ходят
    в отделение.

    Дата установки всегда попадает ВНУТРЬ окна наблюдения.
    Раньше она могла вылететь за его границу, и клиент молча
    оставался без приложения — доля пользователей оказывалась
    заметно ниже объявленной.
    """

    from ..life.persona import draw_persona

    settings = params_module.active().defects

    persona = draw_persona(client_ordinal)

    rng = keyed_rng(NS_COVERAGE, client_ordinal, 1)

    digital = persona.trait("digital_affinity")

    # Цифровая склонность двигает вероятность мягко: разница
    # между самым и наименее цифровым клиентом — проценты, а не
    # разы. Приложением пользуются почти все.
    probability = settings.app_adoption_share * (0.94 + 0.12 * digital)

    if rng.random() >= min(0.995, probability):
        return None

    # Раньше начала наблюдения приложения быть не может, как и
    # раньше того дня, когда человек стал клиентом.
    start = max(config.HISTORY_START, persona.relationship_start)

    # Половина окна на то, чтобы установить: у большинства это
    # случается вскоре после начала отношений с банком.
    span_days = max(1, (config.HISTORY_END - start).days)

    offset = int(rng.integers(0, max(1, span_days // 2)))

    adopted = start + timedelta(days=offset)

    if adopted >= config.HISTORY_END:
        adopted = start

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

    if given >= config.HISTORY_END:
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

    if moment >= config.HISTORY_END:
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

    span_days = (config.HISTORY_END - SOURCE_AVAILABILITY[source]).days

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

        # Сбой засчитывается только внутри наблюдаемого отрезка
        # этого клиента. День, когда витрина молчала, а клиент ещё
        # не пришёл или уже ушёл, ничего о его покрытии не говорит:
        # такой день делал источник «частичным» на пустом месте.
        #
        # Отрезок закрыт с обеих сторон: началом наблюдения и
        # концом отношений, если они закончились раньше выгрузки.
        stop = config.HISTORY_END if closed_at is None else min(closed_at, config.HISTORY_END)

        outages = tuple(
            day
            for day in source_outages(persona.client_ordinal, source)
            if seen is not None and seen.date() <= day < stop.date()
        )

        last_available: datetime | None = None
        status = STATUS_FULL
        reason: str | None = None

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

            if closed_at is not None and closed_at < config.HISTORY_END:
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
                # Сбой без даты не проверить и не учесть: раньше
                # строка сообщала «был сбой», а когда именно —
                # знал только генератор.
                outage_days=(
                    json.dumps([day.isoformat() for day in outages])
                    if outages
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
