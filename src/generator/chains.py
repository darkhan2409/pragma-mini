from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .app import (
    AppOperationEvent,
    AppScreenEvent,
    BannerEvent,
    application_session,
    sessions_for_day,
)
from .communications import CommunicationEvent, generate_communications_for_day
from .coverage import app_adoption
from .persona import Persona, draw_persona
from .products import PRODUCT_TYPES, ProductEvent, ProductState
from .rng import NS_CHAIN, KeyedRandom, keyed_rng, second_of_day
from .trajectory import behavior_state


# ============================================================
# ИДЕЯ
# ============================================================
#
# Потоки нельзя генерировать по отдельности: они связаны.
#
#     владение продуктами на начало дня
#         -> право на кампанию
#     доставленная кампания + скрытый клик
#         -> заявка
#     баннер в приложении + клик
#         -> заявка
#     просмотр раздела приложения
#         -> слабый интерес, изредка заявка
#     заявка
#         -> воронка в приложении (view/application/kyc)
#         -> одобрение -> договор
#         -> отказ -> reject_reason и никакого договора
#
# Клиент без приложения проходит заявку офлайн: экранов нет,
# договор появляется сам по себе. Это не пропуск данных,
# а другой канал продажи.
# ============================================================


# Задержка между интересом и подачей заявки.
COMMUNICATION_DELAY = (60 * 60, 5 * 24 * 60 * 60)
BANNER_DELAY = (5 * 60, 2 * 24 * 60 * 60)
BROWSE_DELAY = (30 * 60, 3 * 24 * 60 * 60)

# Не больше одной заявки в этот период.
APPLICATION_COOLDOWN_DAYS = 20

# Вероятность подать заявку после соответствующего сигнала.
APPLY_AFTER_COMMUNICATION = 0.14
APPLY_AFTER_BANNER = 0.17
APPLY_AFTER_BROWSE = 0.012

SOURCE_COMMUNICATION = 1
SOURCE_BANNER = 2
SOURCE_BROWSE = 3

# Базовая вероятность одобрения заявки.
APPROVAL_BASE = {
    "cash_loan": 0.55,
    "credit_card": 0.62,
    "deposit": 0.98,
    "insurance": 0.95,
    "debit_card": 0.97,
}


@dataclass
class LifecycleResult:
    communications: list[CommunicationEvent] = field(default_factory=list)
    app_screens: list[AppScreenEvent] = field(default_factory=list)
    app_operations: list[AppOperationEvent] = field(default_factory=list)
    banners: list[BannerEvent] = field(default_factory=list)
    product_events: list[ProductEvent] = field(default_factory=list)

    # Состояние продуктов нужно профилю: он строится из него.
    state: ProductState | None = None


@dataclass(frozen=True)
class Application:
    """
    Запланированная заявка: интерес уже возник, форма ещё не подана.
    """

    ts: datetime
    product: str
    source: int


# ============================================================
# ВЕРОЯТНОСТИ
# ============================================================


def chain_rng(client_id: int, source: int, ts: datetime) -> KeyedRandom:
    """
    RNG цепочки по идентичности события-источника.
    """

    return keyed_rng(NS_CHAIN, client_id, ts.toordinal(), second_of_day(ts), source)


def approval_probability(
    product: str,
    persona: Persona,
    stress: float,
    owned: frozenset[str],
) -> float:
    """
    Решение банка по заявке.
    """

    probability = APPROVAL_BASE[product]

    if product in ("cash_loan", "credit_card"):

        income_factor = min(1.0, persona.declared_income / 700_000)

        probability += 0.20 * income_factor
        probability -= 0.35 * stress
        probability -= 0.15 * persona.risk

        # Действующий кредит снижает шанс на новый.
        if "cash_loan" in owned and product == "cash_loan":
            probability -= 0.25

    return float(min(0.99, max(0.02, probability)))


# ============================================================
# ЖИЗНЕННЫЙ ЦИКЛ
# ============================================================


def derive_lifecycle(
    client_id: int,
    start: datetime,
    end: datetime,
) -> LifecycleResult:
    """
    Согласованные между собой коммуникации, экраны, операции,
    баннеры и договоры одного клиента на [start, end).
    """

    if end <= start:
        raise ValueError("end must be after start")

    persona = draw_persona(client_id)

    state = ProductState(client_id)

    adoption = app_adoption(client_id)

    result = LifecycleResult()

    pending: list[Application] = []
    last_application: datetime | None = None

    day = start.replace(hour=0, minute=0, second=0, microsecond=0)

    while day < end:

        owned = state.owned_at(day)
        stress = behavior_state(client_id, day).credit_stress

        app_active = adoption is not None and day >= adoption

        # ----------------------------------------------------
        # КОММУНИКАЦИИ
        # ----------------------------------------------------

        outcomes = generate_communications_for_day(
            client_id=client_id,
            day=day,
            start=start,
            end=end,
            owned=owned,
            app_adopted=app_active,
        )

        for outcome in outcomes:

            result.communications.append(outcome.event)

            if not outcome.clicked or outcome.product is None:
                continue

            rng = chain_rng(client_id, SOURCE_COMMUNICATION, outcome.event.ts)

            if rng.random() >= APPLY_AFTER_COMMUNICATION:
                continue

            delay = rng.integers(*COMMUNICATION_DELAY)

            pending.append(
                Application(
                    ts=outcome.event.ts + timedelta(seconds=delay),
                    product=outcome.product,
                    source=SOURCE_COMMUNICATION,
                )
            )

        # ----------------------------------------------------
        # СЕССИИ ПРИЛОЖЕНИЯ
        # ----------------------------------------------------

        if app_active:

            for session in sessions_for_day(client_id, day):

                result.app_screens.extend(session.screens)
                result.app_operations.extend(session.operations)
                result.banners.extend(session.banners)

                for product, click_ts in session.clicked_offers:

                    rng = chain_rng(client_id, SOURCE_BANNER, click_ts)

                    if rng.random() >= APPLY_AFTER_BANNER:
                        continue

                    pending.append(
                        Application(
                            ts=click_ts + timedelta(seconds=rng.integers(*BANNER_DELAY)),
                            product=product,
                            source=SOURCE_BANNER,
                        )
                    )

                for product in session.browsed_products:

                    rng = chain_rng(client_id, SOURCE_BROWSE, session.started_at)

                    interest = APPLY_AFTER_BROWSE * (1.0 + 2.0 * persona.credit_need)

                    if rng.random() >= interest:
                        continue

                    pending.append(
                        Application(
                            ts=session.started_at
                            + timedelta(seconds=rng.integers(*BROWSE_DELAY)),
                            product=product,
                            source=SOURCE_BROWSE,
                        )
                    )

        # ----------------------------------------------------
        # ЗАЯВКИ, СОЗРЕВШИЕ К ЭТОМУ ДНЮ
        # ----------------------------------------------------

        next_day = day + timedelta(days=1)

        matured = [item for item in pending if item.ts < next_day]

        if matured:
            pending = [item for item in pending if item.ts >= next_day]

        for application in sorted(matured, key=lambda item: item.ts):

            if application.ts < start or application.ts >= end:
                continue

            if application.product not in PRODUCT_TYPES:
                continue

            if state.blocked(application.product, application.ts):
                continue

            if (
                last_application is not None
                and application.ts - last_application
                < timedelta(days=APPLICATION_COOLDOWN_DAYS)
            ):
                continue

            last_application = application.ts

            rng = chain_rng(client_id, 9, application.ts)

            probability = approval_probability(
                product=application.product,
                persona=persona,
                stress=behavior_state(client_id, application.ts).credit_stress,
                owned=state.owned_at(application.ts),
            )

            approved = rng.random() < probability

            # Клиент с приложением подаёт заявку в нём.
            if adoption is not None and application.ts >= adoption:

                session = application_session(
                    client_id=client_id,
                    product=application.product,
                    started_at=application.ts,
                    approved=approved,
                )

                result.app_screens.extend(session.screens)

                decision_ts = session.screens[-1].ts

            else:
                decision_ts = application.ts + timedelta(
                    seconds=rng.integers(600, 3 * 24 * 3600)
                )

            if not approved:
                continue

            open_ts = decision_ts + timedelta(seconds=rng.integers(60, 6 * 3600))

            if open_ts >= end:
                continue

            state.open(application.product, open_ts)

        day = next_day

    # --------------------------------------------------------
    # ИТОГ
    # --------------------------------------------------------

    result.product_events = state.visible_events()
    result.state = state

    result.communications.sort(key=lambda event: event.ts)
    result.app_screens.sort(key=lambda event: event.ts)
    result.app_operations.sort(key=lambda event: event.ts)
    result.banners.sort(key=lambda event: event.ts)

    return result
