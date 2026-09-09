from __future__ import annotations

from datetime import datetime, timedelta

from ..app import BannerEvent
from ..persona import Persona
from ..rng import KeyedRandom
from ..world import (
    ACTION_CLICKED,
    ACTION_SHOWN,
    BANNER_OFFER_PRODUCT,
    BANNER_OFFER_WEIGHTS,
    BANNER_OFFERS,
    BANNER_SLOT_WEIGHTS,
    BANNER_SLOTS,
)
from .config import OWNED_BANNER_FACTOR


# ============================================================
# ВИТРИНА ПРЕДЛОЖЕНИЙ
# ============================================================
#
# Отличие от v1 одно: оффер уже имеющегося продукта показывается
# редко, как и в коммуникациях. Тайминги, слоты и форма события
# те же самые.
# ============================================================


BANNER_CTR = 0.016


def banner_events_v2(
    client_id: int,
    persona: Persona,
    screen_ts: datetime,
    owned: frozenset[str],
    rng: KeyedRandom,
    session_id: str | None = None,
) -> tuple[list[BannerEvent], list[tuple[str, datetime]]]:

    weights = [
        weight * (OWNED_BANNER_FACTOR if BANNER_OFFER_PRODUCT[offer] in owned else 1.0)
        for offer, weight in zip(BANNER_OFFERS, BANNER_OFFER_WEIGHTS)
    ]

    events: list[BannerEvent] = []
    clicks: list[tuple[str, datetime]] = []

    used: set[str] = set()

    for _ in range(rng.integers(1, 4)):

        slot = str(rng.choice(BANNER_SLOTS, p=BANNER_SLOT_WEIGHTS))

        if slot in used:
            continue

        used.add(slot)

        offer = str(rng.choice(BANNER_OFFERS, p=weights))

        shown_ts = screen_ts + timedelta(seconds=rng.integers(0, 4))

        events.append(
            BannerEvent(
                client_id=client_id,
                ts=shown_ts,
                slot=slot,
                offer=offer,
                action=ACTION_SHOWN,
                session_id=session_id,
            )
        )

        product = BANNER_OFFER_PRODUCT[offer]

        ctr = BANNER_CTR * (0.6 + 1.2 * persona.digital_affinity)

        if product in ("cash_loan", "credit_card"):
            ctr *= 1.0 + 1.5 * persona.credit_need

        if product in owned:
            ctr *= 0.4

        if rng.random() < min(0.30, ctr):

            click_ts = shown_ts + timedelta(seconds=rng.integers(2, 25))

            events.append(
                BannerEvent(
                    client_id=client_id,
                    ts=click_ts,
                    slot=slot,
                    offer=offer,
                    action=ACTION_CLICKED,
                    session_id=session_id,
                )
            )

            if product is not None:
                clicks.append((product, click_ts))

    return events, clicks


__all__ = ["BANNER_CTR", "banner_events_v2"]
