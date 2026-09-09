from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..app import DOMAIN_PRODUCT, AppOperationEvent, AppScreenEvent, AppSession
from ..chains import (
    APPLICATION_COOLDOWN_DAYS,
    APPLY_AFTER_BANNER,
    APPLY_AFTER_COMMUNICATION,
    BANNER_DELAY,
    COMMUNICATION_DELAY,
    SOURCE_BANNER,
    SOURCE_BROWSE,
    SOURCE_COMMUNICATION,
    Application,
    LifecycleResult,
    approval_probability,
    chain_rng,
)
from ..communications import generate_communications_for_day
from ..coverage import app_adoption
from ..persona import Persona, draw_persona
from ..products import PRODUCT_TYPES
from ..rng import (
    COMPONENT_CHANNEL,
    COMPONENT_CONTENT,
    COMPONENT_LINKED,
    NS_BANNER,
    NS_V2_INTEREST,
    NS_V2_OUTCOME,
    NS_V2_TX,
    event_rng,
)
from ..trajectory import behavior_state
from .banners import banner_events_v2
from .config import (
    APPLY_AFTER_EXPLORE,
    AUTH_TTL_HOURS,
    BILL_APP_WINDOW_DAYS,
    DEPOSIT_OPEN_DELAY,
    EXPLORE_DEPTH_FACTOR,
    EXPLORE_REPEAT_CAP,
    EXPLORE_REPEAT_FACTOR,
    FREE_PAYMENT_AMOUNT,
    OFFER_FACTOR,
    REJECTION_FACTOR,
)
from .context import (
    KIND_FAILURE,
    KIND_OFFER,
    KIND_REJECTION,
    KIND_REMINDER,
    KIND_UNFINISHED,
    KIND_VIEW,
    ClientContext,
)
from .funnel import application_session_v2
from .habits import BillDue, ClientHabits, bill_amount, bills_due, client_habits
from .outcomes import FAILED, SUCCESS, OperationContext, draw_status, in_outage
from .products import ProductStateV2
from .sessions import (
    KIND_SCREEN,
    TAG_AUTH,
    PaymentIntent,
    Proposal,
    Resolution,
    ScenarioRun,
    new_session,
    run_session,
    session_starts_for_day,
)
from .state import BillRef, StateView, accounts_of, operation_feasible


# ============================================================
# ИДЕЯ
# ============================================================
#
# Дневной цикл v1 сохранён: владение -> кампания -> заявка ->
# воронка -> договор. Добавлен ПЛАНИРОВЩИК.
#
# Сессии дня и созревшие заявки исполняются в общем порядке
# СОБЫТИЙ, а не пачками: сессия, начатая в 09:00, и сессия,
# начатая в 09:15, чередуются по времени своих шагов. Каждое
# решение видит только то, что произошло раньше его секунды.
#
# Здесь же, в момент действия:
#   проверяется жёсткое ограничение (успех может быть невозможен)
#   разыгрывается исход
#   применяется последствие
#
# Успех без последствия невозможен по построению: каждая
# успешная pay_* порождает ровно одну запись об оплате, а
# каждая запись об оплате порождает ровно одну транзакцию.
# ============================================================


@dataclass(frozen=True)
class PaymentRecord:
    """
    Одна оплата. Ровно одна транзакция на запись.

    bill_key = None это свободный платёж: он ничей счёт не гасит
    и имеет собственные назначение и сумму.
    """

    ts: datetime
    mcc: str
    amount: int
    kind: str
    paid_via: str
    bill_key: tuple[int, int] | None = None
    op_key: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class LinkedPurchase:
    op_key: tuple[int, int, int, int]
    operation: str
    ts: datetime


@dataclass
class LifecycleResultV2(LifecycleResult):
    payments: tuple[PaymentRecord, ...] = ()
    linked_purchases: tuple[LinkedPurchase, ...] = ()
    card_blocks: tuple[tuple[datetime, datetime | None], ...] = ()
    runs: tuple[ScenarioRun, ...] = ()


# ============================================================
# ИНТЕРЕС К ЗАЯВКЕ
# ============================================================


def application_interest(
    product: str,
    depth: str,
    views: int,
    rejected: bool,
    offered: bool,
    credit_need: float,
) -> float:
    """
    Вероятность подать заявку после осознанного изучения раздела.
    """

    value = APPLY_AFTER_EXPLORE * EXPLORE_DEPTH_FACTOR[depth]

    value *= min(EXPLORE_REPEAT_CAP, 1.0 + EXPLORE_REPEAT_FACTOR * views)

    if rejected:
        value *= REJECTION_FACTOR

    if offered:
        value *= OFFER_FACTOR

    if product in ("cash_loan", "credit_card"):
        value *= 1.0 + 1.5 * credit_need

    return float(min(0.9, value))


# ============================================================
# СЧЕТА
# ============================================================


@dataclass
class OpenBill:
    due: BillDue
    index: int
    amount: int
    deadline: datetime

    @property
    def key(self) -> tuple[int, int]:
        return (self.index, self.due.month)


def bill_channel_rng(client_id: int, due: BillDue, index: int):
    return event_rng(
        NS_V2_TX, client_id, due.due.toordinal(), 600 + index, COMPONENT_CHANNEL
    )


def bill_amount_rng(client_id: int, due: BillDue, index: int):
    return event_rng(
        NS_V2_TX, client_id, due.due.toordinal(), 600 + index, COMPONENT_CONTENT
    )


# ============================================================
# СЕССИЯ В РАБОТЕ
# ============================================================


@dataclass
class Runner:
    index: int
    session: AppSession
    generator: object
    run: ScenarioRun
    proposal: Proposal | None = None


def _advance(runner: Runner, resolution: Resolution | None) -> None:

    try:
        runner.proposal = runner.generator.send(resolution)
    except StopIteration:
        runner.proposal = None


# ============================================================
# ЖИЗНЕННЫЙ ЦИКЛ
# ============================================================


def derive_lifecycle_v2(
    client_id: int,
    start: datetime,
    end: datetime,
) -> LifecycleResultV2:

    if end <= start:
        raise ValueError("end must be after start")

    persona = draw_persona(client_id)
    habits = client_habits(client_id)

    state = ProductStateV2(client_id)

    adoption = app_adoption(client_id)

    context = ClientContext()

    result = LifecycleResultV2()

    payments: list[PaymentRecord] = []
    linked: list[LinkedPurchase] = []
    runs: list[ScenarioRun] = []

    pending: list[Application] = []
    last_application: datetime | None = None

    # Действующая авторизация: приложение помнит вход.
    session_state = {"authorized_until": None}

    schedule = bills_due(habits)

    by_day: dict[int, list[tuple[int, BillDue]]] = {}

    for index, due in enumerate(schedule):
        by_day.setdefault(due.due.toordinal(), []).append((index, due))

    open_bills: list[OpenBill] = []

    day = start.replace(hour=0, minute=0, second=0, microsecond=0)

    def probe(ts: datetime) -> StateView:

        owned_now = state.owned_at(ts)
        valid = session_state["authorized_until"]

        return StateView(
            ts=ts,
            owned=owned_now,
            blocking=state.blocking_products(ts),
            closable=state.closable_products(ts),
            accounts=accounts_of(owned_now),
            card_blocked=state.card_blocked_at(ts),
            authorized=valid is not None and ts < valid,
            due_bills=tuple(
                BillRef(
                    key=item.key,
                    kind=item.due.bill.kind,
                    mcc=item.due.bill.mcc,
                    amount=item.amount,
                    due=item.due.due,
                )
                for item in open_bills
                if item.due.due <= ts
            ),
        )

    while day < end:

        next_day = day + timedelta(days=1)

        context.prune(day)

        owned = state.owned_at(day)

        app_active = adoption is not None and day >= adoption

        # ----------------------------------------------------
        # КОММУНИКАЦИИ
        # ----------------------------------------------------

        for outcome in generate_communications_for_day(
            client_id=client_id,
            day=day,
            start=start,
            end=end,
            owned=owned,
            app_adopted=app_active,
        ):

            result.communications.append(outcome.event)

            if outcome.product is not None:
                context.record(outcome.event.ts, KIND_OFFER, outcome.product)

            if outcome.campaign == "payment_reminder":
                context.record(outcome.event.ts, KIND_REMINDER)

            if not outcome.clicked or outcome.product is None:
                continue

            rng = chain_rng(client_id, SOURCE_COMMUNICATION, outcome.event.ts)

            if rng.random() >= APPLY_AFTER_COMMUNICATION:
                continue

            pending.append(
                Application(
                    ts=outcome.event.ts
                    + timedelta(seconds=rng.integers(*COMMUNICATION_DELAY)),
                    product=outcome.product,
                    source=SOURCE_COMMUNICATION,
                )
            )

        # ----------------------------------------------------
        # СРОКИ ОПЛАТЫ
        # ----------------------------------------------------

        for index, due in by_day.get(day.toordinal(), ()):

            amount = bill_amount(due, bill_amount_rng(client_id, due, index))

            channel = bill_channel_rng(client_id, due, index)

            in_app = (
                app_active
                and "payments" in _adopted(client_id)
                and channel.random() < habits.app.in_app_bills_share
            )

            if in_app:
                open_bills.append(
                    OpenBill(
                        due=due,
                        index=index,
                        amount=amount,
                        deadline=due.due + timedelta(days=BILL_APP_WINDOW_DAYS),
                    )
                )
            else:
                payments.append(
                    PaymentRecord(
                        ts=_direct_pay_ts(due.due, channel),
                        mcc=due.bill.mcc,
                        amount=amount,
                        kind=due.bill.kind,
                        paid_via="direct",
                        bill_key=(index, due.month),
                    )
                )

        # ----------------------------------------------------
        # СЕССИИ ДНЯ
        # ----------------------------------------------------

        runners: list[Runner] = []

        if app_active:

            for index, started_at in session_starts_for_day(client_id, day):

                session = new_session(client_id, day, index, started_at)

                run = ScenarioRun(
                    session_id=session.session_id, started_at=started_at
                )
                runs.append(run)

                runner = Runner(
                    index=index,
                    session=session,
                    run=run,
                    generator=run_session(
                        client_id=client_id,
                        day=day,
                        index=index,
                        started_at=started_at,
                        view=context.view(
                            started_at,
                            due_bills=probe(started_at).due_bills,
                            card_blocked=state.card_blocked_at(started_at),
                            credit_need=persona.credit_need,
                        ),
                        habits=habits,
                        persona=persona,
                        probe=probe,
                        run=run,
                    ),
                )

                _advance(runner, None)

                runners.append(runner)

        # ----------------------------------------------------
        # ОБЩАЯ ОЧЕРЕДЬ СОБЫТИЙ ДНЯ
        # ----------------------------------------------------
        #
        # Заявки и шаги всех сессий идут строго по времени.
        # Сортировки готового результата было бы недостаточно:
        # решение должно приниматься на своё состояние.
        # ----------------------------------------------------

        while True:

            ready = [item for item in pending if item.ts < next_day]

            application_ts = min((item.ts for item in ready), default=None)

            live = [runner for runner in runners if runner.proposal is not None]

            proposal_ts = min((r.proposal.ts for r in live), default=None)

            if application_ts is None and proposal_ts is None:
                break

            if proposal_ts is None or (
                application_ts is not None and application_ts <= proposal_ts
            ):

                application = min(
                    (item for item in ready if item.ts == application_ts),
                    key=lambda item: (item.ts, item.product),
                )

                pending.remove(application)

                last_application = _process_application(
                    application=application,
                    client_id=client_id,
                    persona=persona,
                    state=state,
                    adoption=adoption,
                    start=start,
                    end=end,
                    last_application=last_application,
                    result=result,
                    context=context,
                )

                continue

            runner = min(live, key=lambda item: (item.proposal.ts, item.index))

            resolution, expiry = _commit(
                proposal=runner.proposal,
                runner=runner,
                client_id=client_id,
                day=day,
                persona=persona,
                habits=habits,
                state=state,
                open_bills=open_bills,
                payments=payments,
                linked=linked,
                context=context,
                authorized_until=session_state["authorized_until"],
            )

            if expiry is not None:
                session_state["authorized_until"] = expiry

            _advance(runner, resolution)

        # ----------------------------------------------------
        # ИТОГИ СЕССИЙ
        # ----------------------------------------------------

        for runner in runners:

            session = runner.session
            run = runner.run

            result.app_screens.extend(session.screens)
            result.app_operations.extend(session.operations)
            result.banners.extend(session.banners)

            if not session.screens:
                continue

            last_ts = session.screens[-1].ts

            for product in run.viewed_products:
                context.record(last_ts, KIND_VIEW, product)

            if run.abandoned and not run.completed:
                context.record(last_ts, KIND_UNFINISHED, run.scenario)

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

            view = context.view(last_ts, credit_need=persona.credit_need)

            for product, depth in sorted(run.explore_depth.items()):

                if product not in PRODUCT_TYPES:
                    continue

                interest = application_interest(
                    product=product,
                    depth=depth,
                    views=view.views.get(product, 0),
                    rejected=product in view.recent_rejections,
                    offered=product in view.recent_offers,
                    credit_need=persona.credit_need,
                )

                rng = event_rng(
                    NS_V2_INTEREST,
                    client_id,
                    day.toordinal(),
                    runner.index,
                    1 + PRODUCT_TYPES.index(product),
                )

                if rng.random() >= interest:
                    continue

                pending.append(
                    Application(
                        ts=session.started_at
                        + timedelta(seconds=rng.integers(1_800, 259_200)),
                        product=product,
                        source=SOURCE_BROWSE,
                    )
                )

        # ----------------------------------------------------
        # ПРОСРОЧЕННЫЕ СЧЕТА
        # ----------------------------------------------------

        remaining: list[OpenBill] = []

        for item in open_bills:

            if item.deadline >= next_day:
                remaining.append(item)
                continue

            payments.append(
                PaymentRecord(
                    ts=_direct_pay_ts(
                        item.deadline,
                        bill_channel_rng(client_id, item.due, item.index),
                        offset_days=1,
                    ),
                    mcc=item.due.bill.mcc,
                    amount=item.amount,
                    kind=item.due.bill.kind,
                    paid_via="direct",
                    bill_key=item.key,
                )
            )

        open_bills = remaining

        day = next_day

    # --------------------------------------------------------
    # ИТОГ
    # --------------------------------------------------------

    result.product_events = state.visible_events()
    result.state = state

    result.payments = tuple(sorted(payments, key=lambda item: item.ts))
    result.linked_purchases = tuple(sorted(linked, key=lambda item: item.ts))
    result.card_blocks = state.blocked_intervals()
    result.runs = tuple(runs)

    result.communications.sort(key=lambda event: event.ts)
    result.app_screens.sort(key=lambda event: event.ts)
    result.app_operations.sort(key=lambda event: event.ts)
    result.banners.sort(key=lambda event: event.ts)

    return result


# ============================================================
# ИСПОЛНЕНИЕ ДЕЙСТВИЯ
# ============================================================


def _commit(
    proposal: Proposal,
    runner: Runner,
    client_id: int,
    day: datetime,
    persona: Persona,
    habits: ClientHabits,
    state: ProductStateV2,
    open_bills: list[OpenBill],
    payments: list[PaymentRecord],
    linked: list[LinkedPurchase],
    context: ClientContext,
    authorized_until: datetime | None,
) -> tuple[Resolution, datetime | None]:
    """
    Жёсткое ограничение, затем исход, затем последствие.
    Всё на момент самого действия.
    """

    ts = proposal.ts
    session = runner.session

    if proposal.kind == KIND_SCREEN:

        session.screens.append(
            AppScreenEvent(
                client_id=client_id,
                ts=ts,
                session_id=session.session_id,
                firebase_screen=proposal.screen,
                product=DOMAIN_PRODUCT.get(proposal.domain),
                funnel_stage=None,
                reject_reason=None,
            )
        )

        if runner.run.first_screen_ts is None:
            runner.run.first_screen_ts = ts

        runner.run.screens += 1

        if proposal.offers:

            events, clicks = banner_events_v2(
                client_id=client_id,
                persona=persona,
                screen_ts=ts,
                owned=state.owned_at(ts),
                rng=event_rng(
                    NS_BANNER,
                    client_id,
                    day.toordinal(),
                    runner.index,
                    COMPONENT_CONTENT,
                ),
                session_id=session.session_id,
            )

            session.banners.extend(events)
            session.clicked_offers = session.clicked_offers + tuple(clicks)

        return Resolution(), None

    operation = proposal.operation

    owned_now = state.owned_at(ts)

    view = StateView(
        ts=ts,
        owned=owned_now,
        blocking=state.blocking_products(ts),
        closable=state.closable_products(ts),
        accounts=accounts_of(owned_now),
        card_blocked=state.card_blocked_at(ts),
        authorized=authorized_until is not None and ts < authorized_until,
    )

    behavior = behavior_state(client_id, ts)

    status = draw_status(
        OperationContext(
            operation=operation,
            domain=proposal.domain,
            attempt=proposal.attempt,
            outage=in_outage(proposal.domain, ts),
            stress=behavior.credit_stress,
            utilization=behavior.utilization_pressure,
            digital=persona.digital_affinity,
            card_blocked=view.card_blocked,
            feasible=operation_feasible(operation, view),
        ),
        event_rng(
            NS_V2_OUTCOME,
            client_id,
            day.toordinal(),
            runner.index * 100 + proposal.step,
            proposal.attempt,
        ),
    )

    # Операция родилась внутри этой сессии, и её принадлежность
    # известна точно. Восстанавливать её потом по близости во
    # времени было бы догадкой, поэтому она едет с событием.
    session.operations.append(
        AppOperationEvent(
            client_id=client_id,
            ts=ts,
            domain=proposal.domain,
            operation=operation,
            status=status,
            session_id=session.session_id,
        )
    )

    if proposal.domain != "auth":

        if runner.run.first_action_ts is None:
            runner.run.first_action_ts = ts

        runner.run.actions += 1

    if status == FAILED:
        context.record(ts, KIND_FAILURE, operation)

    expiry: datetime | None = None

    if status == SUCCESS:

        # Момент, когда завершилась авторизация ИМЕННО этой
        # сессии. Смена пин-кода и перепривязка устройства
        # в настройках тоже относятся к домену auth, но входом
        # не являются: их отличает метка фазы.
        if proposal.tag == TAG_AUTH and runner.run.auth_success_ts is None:
            runner.run.auth_success_ts = ts

        expiry = _apply_success(
            operation=operation,
            intent=proposal.intent,
            ts=ts,
            op_key=(day.toordinal(), runner.index, proposal.step, proposal.attempt),
            client_id=client_id,
            habits=habits,
            state=state,
            open_bills=open_bills,
            payments=payments,
            linked=linked,
        )

    return Resolution(status=status), expiry


FREE_PAYMENT_MCC = {
    "pay_utility": "4900",
    "pay_mobile": "4814",
    "pay_internet": "4899",
    "pay_fine": "9222",
    "pay_tax": "9311",
}

FREE_PAYMENT_KIND = {
    "pay_utility": "utility",
    "pay_mobile": "mobile",
    "pay_internet": "internet",
    "pay_fine": "fine",
    "pay_tax": "tax",
}


def _apply_success(
    operation: str,
    intent: PaymentIntent | None,
    ts: datetime,
    op_key: tuple[int, int, int, int],
    client_id: int,
    habits: ClientHabits,
    state: ProductStateV2,
    open_bills: list[OpenBill],
    payments: list[PaymentRecord],
    linked: list[LinkedPurchase],
) -> datetime | None:
    """
    Последствие успеха. Возвращает новый срок авторизации,
    если операция его меняет.
    """

    if operation in ("login", "biometry_login", "device_bind"):
        return ts + timedelta(hours=AUTH_TTL_HOURS)

    if operation == "logout":
        return ts

    if operation == "deposit_open":
        state.open_after(
            "deposit",
            ts + timedelta(seconds=DEPOSIT_OPEN_DELAY[0]),
            not_before=ts,
        )
        return None

    if operation == "deposit_close":
        state.close_early("deposit", ts)
        return None

    if operation == "loan_early_repay":
        state.close_early("cash_loan", ts)
        return None

    if operation == "card_block":
        state.block_card(ts)
        return None

    if operation == "card_unblock":
        state.unblock_card(ts)
        return None

    if operation in ("pay_qr", "market_order"):
        linked.append(LinkedPurchase(op_key=op_key, operation=operation, ts=ts))
        return None

    if operation not in FREE_PAYMENT_MCC:
        return None

    # ----------------------------------------------------
    # ОПЛАТА
    # ----------------------------------------------------
    #
    # Успех обязан иметь ровно одно последствие. Счёт гасится
    # только один раз: он удаляется из открытых, и повтор
    # той же операции его уже не найдёт. Если счёта нет,
    # платёж не исчезает, а становится свободным.
    # ----------------------------------------------------

    bill = None

    if intent is not None and intent.bill_key is not None:
        bill = next((item for item in open_bills if item.key == intent.bill_key), None)

    if bill is not None:

        open_bills.remove(bill)

        payments.append(
            PaymentRecord(
                ts=ts,
                mcc=bill.due.bill.mcc,
                amount=bill.amount,
                kind=bill.due.bill.kind,
                paid_via="app",
                bill_key=bill.key,
                op_key=op_key,
            )
        )

        return None

    base, sigma = FREE_PAYMENT_AMOUNT[FREE_PAYMENT_KIND[operation]]

    rng = event_rng(
        NS_V2_TX, client_id, op_key[0], op_key[1] * 100 + op_key[2], COMPONENT_LINKED
    )

    value = rng.lognormal(math.log(base), sigma)

    payments.append(
        PaymentRecord(
            ts=ts,
            mcc=FREE_PAYMENT_MCC[operation],
            amount=int(round(min(2_000_000.0, max(200.0, value)) / 10) * 10),
            kind="free",
            paid_via="app",
            bill_key=None,
            op_key=op_key,
        )
    )

    return None


# ============================================================
# ЗАЯВКИ
# ============================================================


def _process_application(
    application: Application,
    client_id: int,
    persona: Persona,
    state: ProductStateV2,
    adoption: datetime | None,
    start: datetime,
    end: datetime,
    last_application: datetime | None,
    result: LifecycleResultV2,
    context: ClientContext,
) -> datetime | None:

    if application.ts < start or application.ts >= end:
        return last_application

    if application.product not in PRODUCT_TYPES:
        return last_application

    if state.blocked(application.product, application.ts):
        return last_application

    if (
        last_application is not None
        and application.ts - last_application
        < timedelta(days=APPLICATION_COOLDOWN_DAYS)
    ):
        return last_application

    rng = chain_rng(client_id, 9, application.ts)

    stress = behavior_state(client_id, application.ts).credit_stress
    owned = state.owned_at(application.ts)

    approved = rng.random() < approval_probability(
        product=application.product,
        persona=persona,
        stress=stress,
        owned=owned,
    )

    if adoption is not None and application.ts >= adoption:

        funnel = application_session_v2(
            client_id=client_id,
            product=application.product,
            started_at=application.ts,
            approved=approved,
            persona=persona,
            stress=stress,
            owned=owned,
        )

        result.app_screens.extend(funnel.screens)

        decision_ts = funnel.screens[-1].ts

        if not approved:
            context.record(decision_ts, KIND_REJECTION, application.product)

    else:
        decision_ts = application.ts + timedelta(
            seconds=rng.integers(600, 3 * 24 * 3600)
        )

    if approved:

        open_ts = decision_ts + timedelta(seconds=rng.integers(60, 6 * 3600))

        if open_ts < end:
            # Договор не может попасть в реестр раньше решения,
            # даже если время открытия в реестре теряется.
            state.open_after(application.product, open_ts, not_before=decision_ts)

    return application.ts


def _direct_pay_ts(anchor: datetime, rng, offset_days: int = 0) -> datetime:

    day = anchor + timedelta(days=offset_days + int(rng.integers(0, 4)))

    return day.replace(
        hour=int(rng.integers(9, 21)),
        minute=int(rng.integers(0, 60)),
        second=int(rng.integers(0, 60)),
        microsecond=0,
    )


def _adopted(client_id: int) -> tuple[str, ...]:
    from ..app import adopted_domains

    return adopted_domains(client_id)


__all__ = [
    "LifecycleResultV2",
    "LinkedPurchase",
    "PaymentRecord",
    "application_interest",
    "derive_lifecycle_v2",
]
