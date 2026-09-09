from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .persona import Persona, draw_persona
from .rng import (
    COMPONENT_CONTENT,
    COMPONENT_COUNT,
    COMPONENT_TIME,
    NS_COMM,
    KeyedRandom,
    day_rng,
    event_rng,
)
from .trajectory import behavior_state
from .world import (
    CAMPAIGN_CHANNELS,
    CAMPAIGN_PRODUCT,
    CAMPAIGN_TEMPLATES,
    DELIVERY_RATE,
)


# ============================================================
# КОНТРАКТ
# ============================================================
#
#     ts, channel, template, day_of_week, hour, delivered
#
# КАМПАНИИ В КОНТРАКТЕ НЕТ. Бизнес-категория отправки лежит
# в витрине, к которой нет доступа (пункт П2 отчёта), поэтому
# единственный способ отличить кредитное предложение от
# сервисного сообщения это префикс шаблона.
#
# day_of_week и hour дублируют ts: витрина денормализована,
# и preprocessing должен уметь с этим жить.
#
# Факт КЛИКА в RAW тоже не приходит: он остаётся latent
# и служит только причиной последующей заявки.
# ============================================================


@dataclass(frozen=True)
class CommunicationEvent:
    client_id: int
    ts: datetime

    channel: str
    template: str
    day_of_week: int
    hour: int
    delivered: bool


@dataclass(frozen=True)
class CommunicationOutcome:
    """
    Событие плюс скрытая часть: кампания, продукт и клик.
    В RAW уходит только event.
    """

    event: CommunicationEvent
    campaign: str
    product: str | None
    clicked: bool


CAMPAIGNS: tuple[str, ...] = tuple(CAMPAIGN_PRODUCT)


# ============================================================
# ИНТЕНСИВНОСТЬ
# ============================================================


def daily_communication_rate(persona: Persona, ts: datetime) -> float:
    """
    Банк пишет клиенту заметно реже, чем клиент заходит в приложение.
    """

    rate = 0.08 + 0.10 * persona.activity + 0.08 * persona.digital_affinity

    if ts.weekday() >= 5:
        rate *= 0.80

    if ts.month == 12:
        rate *= 1.20

    return float(rate)


# ============================================================
# ВЫБОР КАМПАНИИ
# ============================================================
#
# Право на кампанию зависит от владения продуктами:
# предлагать кредитку владельцу кредитки почти бессмысленно,
# а напоминание о платеже без кредита невозможно.
# ============================================================

OWNED_OFFER_FACTOR = 0.10

BASE_CAMPAIGN_WEIGHTS: dict[str, float] = {
    "cash_loan_offer": 2.0,
    "credit_card_offer": 1.3,
    "deposit_offer": 1.1,
    "insurance_offer": 0.8,
    "debit_card_offer": 0.6,
    "cashback": 1.5,
    "payment_reminder": 1.0,
    "security": 0.5,
    "service": 0.9,
    "nps_survey": 0.3,
}


def campaign_weights(
    persona: Persona,
    stress: float,
    owned: frozenset[str],
) -> np.ndarray:

    credit = persona.credit_need
    digital = persona.digital_affinity

    weights = dict(BASE_CAMPAIGN_WEIGHTS)

    weights["cash_loan_offer"] *= 0.4 + 2.0 * credit
    weights["credit_card_offer"] *= 0.6 + 1.2 * credit
    weights["deposit_offer"] *= 0.6 + 1.2 * (1.0 - credit)
    weights["insurance_offer"] *= 0.6 + 1.0 * persona.mobility
    weights["cashback"] *= 0.7 + 1.0 * digital
    weights["nps_survey"] *= 0.4 + 1.4 * digital

    # Напоминание о платеже возможно только при кредитном продукте
    # и усиливается при стрессе.
    if not (owned & {"credit_card", "cash_loan"}):
        weights["payment_reminder"] = 0.0
    else:
        weights["payment_reminder"] *= 1.0 + 3.0 * stress

    # Предложение продукта, который уже есть, почти не отправляется.
    for campaign, product in CAMPAIGN_PRODUCT.items():
        if product is not None and product in owned:
            weights[campaign] *= OWNED_OFFER_FACTOR

    # Кэшбэк адресован держателям карт.
    if not (owned & {"debit_card", "credit_card"}):
        weights["cashback"] = 0.0

    return np.array([weights[campaign] for campaign in CAMPAIGNS], dtype=float)


def choose_campaign(
    persona: Persona,
    stress: float,
    owned: frozenset[str],
    rng: KeyedRandom,
) -> str:

    weights = campaign_weights(persona, stress, owned)

    total = weights.sum()

    if total <= 0:
        return "service"

    cumulative = np.cumsum(weights)

    index = int(np.searchsorted(cumulative, rng.random() * total, side="right"))

    return CAMPAIGNS[min(index, len(CAMPAIGNS) - 1)]


# ============================================================
# КАНАЛ И ШАБЛОН
# ============================================================


def choose_channel(
    campaign: str,
    persona: Persona,
    app_adopted: bool,
    rng: KeyedRandom,
) -> str:

    channels = CAMPAIGN_CHANNELS[campaign]

    weights = []

    for channel in channels:
        if channel == "push":
            # Банк видит, установлено ли приложение, и почти
            # не шлёт push тем, у кого его нет.
            weight = 0.5 + 1.6 * persona.digital_affinity
            weights.append(weight if app_adopted else weight * 0.15)
        elif channel == "sms":
            weights.append(1.2 - 0.4 * persona.digital_affinity)
        else:
            weights.append(0.7)

    return str(rng.choice(channels, p=weights))


def choose_template(campaign: str, rng: KeyedRandom) -> str:
    """
    Шаблоны кампании с длинным хвостом: первый берут чаще всего.
    """

    templates = CAMPAIGN_TEMPLATES[campaign]

    weights = [1.0 / (index + 1) ** 1.4 for index in range(len(templates))]

    return str(rng.choice(templates, p=weights))


def is_delivered(
    channel: str,
    persona: Persona,
    app_adopted: bool,
    rng: KeyedRandom,
) -> bool:
    """
    Доставка по каналу.

    push доходит только если приложение установлено и уведомления
    включены; признака достижимости в RAW нет.
    """

    if channel == "push" and not (app_adopted and persona.push_reachable):
        return False

    return rng.random() < DELIVERY_RATE[channel]


def is_clicked(
    campaign: str,
    channel: str,
    persona: Persona,
    stress: float,
    rng: KeyedRandom,
) -> bool:
    """
    Скрытая реакция на доставленное сообщение.
    """

    probability = 0.04 + 0.16 * persona.digital_affinity

    if channel == "call":
        probability += 0.10

    product = CAMPAIGN_PRODUCT[campaign]

    if product in ("cash_loan", "credit_card"):
        probability += 0.20 * persona.credit_need + 0.15 * stress
    elif product == "deposit":
        probability += 0.10 * (1.0 - persona.credit_need)

    return rng.random() < min(0.80, probability)


# ============================================================
# ВРЕМЯ ОТПРАВКИ
# ============================================================

SEND_HOURS = tuple(range(9, 21))

SEND_WEIGHTS = (
    0.06, 0.09, 0.11, 0.11, 0.10, 0.09,
    0.09, 0.08, 0.08, 0.07, 0.07, 0.05,
)


def draw_send_time(day: datetime, rng: KeyedRandom) -> datetime:

    hour = int(rng.choice(SEND_HOURS, p=SEND_WEIGHTS))

    return day.replace(
        hour=hour,
        minute=rng.integers(0, 60),
        second=rng.integers(0, 60),
        microsecond=0,
    )


# ============================================================
# ГЕНЕРАЦИЯ
# ============================================================


def generate_communications_for_day(
    client_id: int,
    day: datetime,
    start: datetime,
    end: datetime,
    owned: frozenset[str],
    app_adopted: bool,
) -> list[CommunicationOutcome]:
    """
    Отправки одного дня с учётом владения продуктами.
    """

    persona = draw_persona(client_id)

    stress = behavior_state(client_id, day).credit_stress

    count = day_rng(NS_COMM, client_id, day.toordinal()).poisson(
        daily_communication_rate(persona, day)
    )

    outcomes: list[CommunicationOutcome] = []

    for index in range(count):

        time_rng = event_rng(NS_COMM, client_id, day.toordinal(), index, COMPONENT_TIME)

        ts = draw_send_time(day, time_rng)

        if not (start <= ts < end):
            continue

        rng = event_rng(NS_COMM, client_id, day.toordinal(), index, COMPONENT_CONTENT)

        campaign = choose_campaign(persona, stress, owned, rng)
        channel = choose_channel(campaign, persona, app_adopted, rng)
        template = choose_template(campaign, rng)

        delivered = is_delivered(channel, persona, app_adopted, rng)

        clicked = delivered and is_clicked(campaign, channel, persona, stress, rng)

        outcomes.append(
            CommunicationOutcome(
                event=CommunicationEvent(
                    client_id=client_id,
                    ts=ts,
                    channel=channel,
                    template=template,
                    day_of_week=ts.weekday(),
                    hour=ts.hour,
                    delivered=delivered,
                ),
                campaign=campaign,
                product=CAMPAIGN_PRODUCT[campaign],
                clicked=clicked,
            )
        )

    return outcomes
