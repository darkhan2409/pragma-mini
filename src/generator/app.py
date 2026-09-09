from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache

from .persona import Persona, draw_persona
from .rng import (
    COMPONENT_CONTENT,
    COMPONENT_COUNT,
    COMPONENT_TIME,
    NS_APP,
    NS_BANNER,
    NS_FUNNEL,
    NS_OPERATION,
    NS_SCREEN,
    KeyedRandom,
    day_rng,
    event_rng,
    keyed_rng,
)
from .trajectory import behavior_state
from .world import (
    ACTION_CLICKED,
    ACTION_SHOWN,
    BANNER_OFFER_PRODUCT,
    BANNER_OFFER_WEIGHTS,
    BANNER_OFFERS,
    BANNER_SLOT_WEIGHTS,
    BANNER_SLOTS,
    BROWSE_SCREENS,
    DOMAIN_ADOPTION,
    DOMAIN_OPERATIONS,
    FUNNEL_SCREENS,
    OPERATION_STATUS_WEIGHTS,
    OPERATION_STATUSES,
    REJECT_REASON_WEIGHTS,
    REJECT_REASONS,
    SCREEN_HOME,
    SCREEN_OFFERS,
)


# ============================================================
# КОНТРАКТЫ
# ============================================================
#
# APP_SCREENS      ts, session_id, firebase_screen, product,
#                  funnel_stage, reject_reason
#
# APP_OPERATIONS   ts, domain, operation, status
#
# BANNERS          ts, slot, offer, action (shown | clicked)
#
# Все три потока рождаются внутри сессии приложения, поэтому
# генерируются вместе: баннер показывается НА экране и делит
# с ним ровно один timestamp, что даёт естественные совпадения
# ts и требует детерминированного tie-break.
# ============================================================


@dataclass(frozen=True)
class AppScreenEvent:
    client_id: int
    ts: datetime

    session_id: str
    firebase_screen: str
    product: str | None
    funnel_stage: str | None
    reject_reason: str | None


# Сессия у операции и баннера появилась в ревизии схемы 2.
# repr=False здесь обязателен: золотые дайджесты V1 это
# sha256(repr(history)), и добавление поля в repr сделало бы
# замороженную версию невоспроизводимой. V1 поле не заполняет,
# и её схема отбрасывает его при записи.


@dataclass(frozen=True)
class AppOperationEvent:
    client_id: int
    ts: datetime

    domain: str
    operation: str
    status: str

    session_id: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class BannerEvent:
    client_id: int
    ts: datetime

    slot: str
    offer: str
    action: str

    session_id: str | None = field(default=None, repr=False, compare=False)


@dataclass
class AppSession:
    session_id: str
    started_at: datetime

    screens: list[AppScreenEvent] = field(default_factory=list)
    operations: list[AppOperationEvent] = field(default_factory=list)
    banners: list[BannerEvent] = field(default_factory=list)

    # Скрытая часть: во что клиент вникал и по чему кликнул.
    browsed_products: tuple[str, ...] = ()
    clicked_offers: tuple[tuple[str, datetime], ...] = ()


# ============================================================
# ДОМЕНЫ КЛИЕНТА
# ============================================================

DOMAIN_PRODUCT: dict[str, str] = {
    "loans": "cash_loan",
    "deposits": "deposit",
    "insurance": "insurance",
    "cards": "credit_card",
}


@lru_cache(maxsize=131_072)
def adopted_domains(client_id: int) -> tuple[str, ...]:
    """
    Домены приложения, которыми клиент вообще пользуется.

    Покрытие домена это свойство клиента, а не дня: у половины
    когорты в bdp_capp_transfers нет ни одной строки никогда.
    """

    persona = draw_persona(client_id)

    rng = keyed_rng(NS_APP, client_id, 0, 0)

    domains = ["home"]

    for domain, share in DOMAIN_ADOPTION.items():

        if domain == "auth":
            continue

        probability = share * (0.6 + 0.8 * persona.digital_affinity)

        if rng.random() < min(0.97, probability):
            domains.append(domain)

    # Профиль и поддержка доступны всем, но заходят туда редко.
    domains.append("profile")

    if "insurance" not in domains and rng.random() < 0.10:
        domains.append("insurance")

    return tuple(dict.fromkeys(domains))


def domain_weights(client_id: int) -> tuple[tuple[str, ...], list[float]]:

    persona = draw_persona(client_id)

    domains = [d for d in adopted_domains(client_id) if d in BROWSE_SCREENS]

    weights = []

    for domain in domains:
        if domain == "home":
            weights.append(3.0)
        elif domain == "transfers":
            weights.append(2.0 + 1.5 * persona.digital_affinity)
        elif domain == "payments":
            weights.append(1.8)
        elif domain == "cards":
            weights.append(1.5)
        elif domain == "loans":
            weights.append(0.5 + 2.0 * persona.credit_need)
        elif domain == "deposits":
            weights.append(0.4 + 1.0 * (1.0 - persona.credit_need))
        elif domain == "market":
            weights.append(0.5 + 1.2 * persona.digital_affinity)
        else:
            weights.append(0.5)

    return tuple(domains), weights


# ============================================================
# ИНТЕНСИВНОСТЬ СЕССИЙ
# ============================================================


def daily_session_rate(persona: Persona, ts: datetime) -> float:

    rate = 0.10 + 0.25 * persona.activity + 0.35 * persona.digital_affinity

    if ts.weekday() >= 5:
        rate *= 0.92

    rate *= behavior_state(persona.client_id, ts).activity_multiplier

    return float(rate)


SESSION_HOURS = tuple(range(24))

SESSION_HOUR_WEIGHTS = (
    0.010, 0.005, 0.003, 0.003, 0.003, 0.008,
    0.025, 0.055, 0.080, 0.085, 0.075, 0.070,
    0.075, 0.075, 0.065, 0.065, 0.070, 0.085,
    0.100, 0.100, 0.085, 0.060, 0.035, 0.015,
)


def draw_session_start(day: datetime, rng: KeyedRandom) -> datetime:

    hour = int(rng.choice(SESSION_HOURS, p=SESSION_HOUR_WEIGHTS))

    return day.replace(
        hour=hour,
        minute=rng.integers(0, 60),
        second=rng.integers(0, 60),
        microsecond=0,
    )


def make_session_id(client_id: int, day: datetime, index: int) -> str:
    """
    Непрозрачный идентификатор сессии, как session_id в GA4.
    """

    rng = keyed_rng(NS_SCREEN, client_id, day.toordinal(), index)

    return str(rng.integers(1_000_000_000, 9_999_999_999))


# ============================================================
# БАННЕРЫ
# ============================================================

BANNER_CTR = 0.016


def banner_events(
    client_id: int,
    screen_ts: datetime,
    slot_pool: tuple[str, ...],
    slot_weights: tuple[float, ...],
    count: int,
    rng: KeyedRandom,
) -> tuple[list[BannerEvent], list[tuple[str, datetime]]]:
    """
    Показы и клики в одном экране.

    Показ делит timestamp с экраном: баннер отрисовывается вместе
    с ним. Клик приходит через несколько секунд.
    """

    persona = draw_persona(client_id)

    events: list[BannerEvent] = []
    clicks: list[tuple[str, datetime]] = []

    used: set[str] = set()

    for _ in range(count):

        slot = str(rng.choice(slot_pool, p=slot_weights))

        if slot in used:
            continue

        used.add(slot)

        offer = str(rng.choice(BANNER_OFFERS, p=BANNER_OFFER_WEIGHTS))

        # Показ логируется при отрисовке экрана: иногда той же
        # секундой, чаще на секунду-две позже.
        shown_ts = screen_ts + timedelta(seconds=rng.integers(0, 4))

        events.append(
            BannerEvent(
                client_id=client_id,
                ts=shown_ts,
                slot=slot,
                offer=offer,
                action=ACTION_SHOWN,
            )
        )

        product = BANNER_OFFER_PRODUCT[offer]

        ctr = BANNER_CTR * (0.6 + 1.2 * persona.digital_affinity)

        if product in ("cash_loan", "credit_card"):
            ctr *= 1.0 + 1.5 * persona.credit_need

        if rng.random() < min(0.30, ctr):

            click_ts = shown_ts + timedelta(seconds=rng.integers(2, 25))

            events.append(
                BannerEvent(
                    client_id=client_id,
                    ts=click_ts,
                    slot=slot,
                    offer=offer,
                    action=ACTION_CLICKED,
                )
            )

            if product is not None:
                clicks.append((product, click_ts))

    return events, clicks


# ============================================================
# ОБЫЧНАЯ СЕССИЯ
# ============================================================


def browse_session(
    client_id: int,
    day: datetime,
    index: int,
) -> AppSession:
    """
    Сессия просмотра: домашний экран, несколько экранов доменов,
    иногда витрина предложений с баннерами, иногда операции.
    """

    persona = draw_persona(client_id)

    time_rng = event_rng(NS_SCREEN, client_id, day.toordinal(), index, COMPONENT_TIME)
    rng = event_rng(NS_SCREEN, client_id, day.toordinal(), index, COMPONENT_CONTENT)

    started_at = draw_session_start(day, time_rng)
    session_id = make_session_id(client_id, day, index)

    session = AppSession(session_id=session_id, started_at=started_at)

    domains, weights = domain_weights(client_id)

    # Сессия начинается со входа, домашний экран отрисовывается
    # через секунду-другую: это разные системы и разные события.
    ts = started_at + timedelta(seconds=rng.integers(1, 5))

    screens: list[AppScreenEvent] = []
    browsed: list[str] = []

    # Первый экран всегда домашний.
    screens.append(
        AppScreenEvent(
            client_id=client_id,
            ts=ts,
            session_id=session_id,
            firebase_screen=SCREEN_HOME,
            product=None,
            funnel_stage=None,
            reject_reason=None,
        )
    )

    depth = 1 + rng.poisson(1.8)
    depth = min(depth, 8)

    show_offers = rng.random() < 0.22 + 0.20 * persona.digital_affinity

    banner_screen_ts: datetime | None = None

    for step in range(depth):

        ts = ts + timedelta(seconds=rng.integers(4, 95))

        if show_offers and step == 0:
            firebase_screen = SCREEN_OFFERS
            domain = "home"
            banner_screen_ts = ts
        else:
            domain = str(rng.choice(domains, p=weights))
            firebase_screen = str(rng.choice(BROWSE_SCREENS[domain]))

        product = DOMAIN_PRODUCT.get(domain)

        if product is not None:
            browsed.append(product)

        screens.append(
            AppScreenEvent(
                client_id=client_id,
                ts=ts,
                session_id=session_id,
                firebase_screen=firebase_screen,
                product=product,
                funnel_stage=None,
                reject_reason=None,
            )
        )

        # Операция в домене: не каждый экран её порождает.
        if domain in DOMAIN_OPERATIONS and domain in adopted_domains(client_id):

            if rng.random() < 0.30:

                operation_rng = event_rng(
                    NS_OPERATION,
                    client_id,
                    day.toordinal(),
                    index * 100 + step,
                    COMPONENT_CONTENT,
                )

                session.operations.append(
                    AppOperationEvent(
                        client_id=client_id,
                        ts=ts + timedelta(seconds=operation_rng.integers(1, 40)),
                        domain=domain,
                        operation=str(operation_rng.choice(DOMAIN_OPERATIONS[domain])),
                        status=str(
                            operation_rng.choice(
                                OPERATION_STATUSES, p=OPERATION_STATUS_WEIGHTS
                            )
                        ),
                    )
                )

    # Вход в приложение это тоже операция домена auth.
    auth_rng = event_rng(
        NS_OPERATION, client_id, day.toordinal(), index, COMPONENT_COUNT
    )

    # Не каждая сессия начинается с явного входа: сессию
    # продолжают в уже открытом приложении.
    if auth_rng.random() < 0.65:
        session.operations.append(
            AppOperationEvent(
                client_id=client_id,
                ts=started_at,
                domain="auth",
                operation="biometry_login" if auth_rng.random() < 0.55 else "login",
                status=str(
                    auth_rng.choice(OPERATION_STATUSES, p=OPERATION_STATUS_WEIGHTS)
                ),
            )
        )

    # Баннеры на витрине предложений.
    if banner_screen_ts is not None:

        banner_rng = event_rng(
            NS_BANNER, client_id, day.toordinal(), index, COMPONENT_CONTENT
        )

        events, clicks = banner_events(
            client_id=client_id,
            screen_ts=banner_screen_ts,
            slot_pool=BANNER_SLOTS,
            slot_weights=BANNER_SLOT_WEIGHTS,
            count=banner_rng.integers(1, 4),
            rng=banner_rng,
        )

        session.banners.extend(events)
        session.clicked_offers = tuple(clicks)

    session.screens = screens
    session.browsed_products = tuple(dict.fromkeys(browsed))

    return session


# ============================================================
# СЕССИЯ ЗАЯВКИ
# ============================================================


def funnel_reject_reason(rng: KeyedRandom) -> str:
    return str(rng.choice(REJECT_REASONS, p=REJECT_REASON_WEIGHTS))


def application_session(
    client_id: int,
    product: str,
    started_at: datetime,
    approved: bool,
) -> AppSession:
    """
    Сессия подачи заявки: воронка view -> application -> kyc ->
    approved | rejected.

    reject_reason заполняется только на экране отказа.
    """

    rng = keyed_rng(NS_FUNNEL, client_id, started_at.toordinal(), started_at.hour)

    session_id = str(rng.integers(1_000_000_000, 9_999_999_999))

    session = AppSession(session_id=session_id, started_at=started_at)

    stages = ["view", "application", "kyc", "approved" if approved else "rejected"]

    gaps = [0, rng.integers(30, 200), rng.integers(60, 600), rng.integers(20, 900)]

    ts = started_at

    screens: list[AppScreenEvent] = []

    for stage, gap in zip(stages, gaps):

        ts = ts + timedelta(seconds=gap)

        screens.append(
            AppScreenEvent(
                client_id=client_id,
                ts=ts,
                session_id=session_id,
                firebase_screen=FUNNEL_SCREENS[stage],
                product=product,
                funnel_stage=stage,
                reject_reason=(
                    funnel_reject_reason(rng) if stage == "rejected" else None
                ),
            )
        )

    session.screens = screens
    session.browsed_products = (product,)

    return session


# ============================================================
# СЕССИИ ДНЯ
# ============================================================


def sessions_for_day(client_id: int, day: datetime) -> list[AppSession]:
    """
    Обычные сессии одного дня. Сессии заявок добавляет chains.py.
    """

    persona = draw_persona(client_id)

    count = day_rng(NS_SCREEN, client_id, day.toordinal()).poisson(
        daily_session_rate(persona, day)
    )

    return [browse_session(client_id, day, index) for index in range(count)]
