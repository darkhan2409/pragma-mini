from __future__ import annotations

from datetime import datetime, timedelta

from .behaviour import communications as comm_module
from .behaviour import outcomes as outcome_module
from .config import (
    HISTORY_END,
    INITIATOR_BANK,
    INITIATOR_CLIENT,
    INITIATOR_SYSTEM,
    SOURCE_AVAILABILITY,
)
from .engine import _HANDLERS, _emit_money, _touch_client
from .finance import deposits as deposit_rules
from .finance import loans as loan_rules
from .finance.entities import CARD_BLOCKED, Offer
from .finance.ledger import COUNTERPART_GOVERNMENT
from .life import stress as stress_module
from .rng import COMPONENT_OUTCOME, NS_SESSION, event_rng, stable_hash
from .simulate import ClientState
from .world.dictionaries import (
    BANNER_OFFERS,
    BANNER_OFFER_FAMILY,
    BANNER_SLOTS,
    BANNER_SLOT_WEIGHTS,
    ERROR_CODES,
)


# ============================================================
# ПРИЛОЖЕНИЕ И КОММУНИКАЦИИ
# ============================================================
#
# Число экранов зависит от ЦЕЛИ сессии. Защищённое действие
# возможно только после успешного входа.
#
# Денежная операция исполняется только тогда, когда ей есть что
# менять: открытый счёт к оплате, просроченный платёж,
# заблокированная карта или действующий депозит. Успеха без
# последствия не бывает.
# ============================================================


BANNERS_AVAILABLE_FROM = SOURCE_AVAILABILITY["banners"]


def _on_session(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    session = payload["session"]

    stress = stress_module.level_at(state.stress_episodes, ts)

    authorized = False

    for position, step in enumerate(session.steps):

        if step.ts >= HISTORY_END:
            break

        rng = event_rng(
            NS_SESSION,
            state.ordinal,
            ts.toordinal(),
            payload["index"] * 100 + position,
            COMPONENT_OUTCOME,
        )

        if step.kind == "screen":

            if not authorized:
                continue

            state.emit(
                state.factory.make(
                    "app_screen",
                    step.ts,
                    {
                        "firebase_screen": step.screen,
                        "domain": step.domain,
                        "product_id": None,
                        "funnel_stage": None,
                        "reject_reason": None,
                    },
                    initiator=INITIATOR_CLIENT,
                    correlation_id=session.session_id,
                    link_type="session",
                )
            )

            if step.offers:
                _show_banners(state, step.ts, session, rng)

            continue

        operation = step.operation

        feasible = True
        insufficient = False

        target_bill = None
        target_loan = None
        target_card = None
        target_deposit = None

        if operation in ("login", "biometry_login"):
            feasible = True
        elif not authorized:
            feasible = False

        if operation and operation.startswith("pay_"):
            target_bill = state.open_bills[0] if state.open_bills else None
            if target_bill is None:
                feasible = False
            elif not state.ledger.payment_sources(step.ts, target_bill["amount"]):
                feasible = False
                insufficient = True

        if operation == "card_unblock":
            target_card = next(
                (item for item in state.cards.values() if item.releasable()), None
            )
            if target_card is None:
                feasible = False

        if operation == "loan_repay":
            target_loan = next(
                (
                    item
                    for item in state.loans.values()
                    if not item.closed and loan_rules.arrears_amount(item) > 0
                ),
                None,
            )
            if target_loan is None:
                feasible = False

        if operation == "deposit_topup":
            target_deposit = next(
                (item for item in state.deposits.values() if not item.closed and item.topup_allowed),
                None,
            )
            if target_deposit is None:
                feasible = False

        card_blocked = any(item.is_blocked_at(step.ts) for item in state.cards.values())

        context = outcome_module.OperationContext(
            operation=operation,
            domain=step.domain,
            attempt=1,
            outage=outcome_module.in_outage(step.domain, step.ts),
            stress=stress,
            digital=state.persona.trait("digital_affinity", step.ts),
            card_blocked=card_blocked,
            insufficient=insufficient,
            feasible=feasible,
        )

        status = outcome_module.draw_status(context, rng)

        state.emit(
            state.factory.make(
                "app_operation",
                step.ts,
                {
                    "domain": step.domain,
                    "operation": operation,
                    "status": status,
                    "amount": target_bill["amount"] if target_bill else None,
                    "error_code": str(rng.choice(list(ERROR_CODES))) if status == "failed" else None,
                    "device_new": session.device_new,
                },
                initiator=INITIATOR_CLIENT,
                correlation_id=session.session_id,
                link_type="session",
            )
        )

        _touch_client(state, step.ts)

        if status == "failed":
            state.recent_failure_at = step.ts

        if status != "success":
            continue

        if operation in ("login", "biometry_login"):
            authorized = True
            continue

        if target_bill is not None:
            _pay_bill_in_app(state, step.ts, target_bill, session, rng)

        if target_card is not None:
            unblock_card(state, step.ts, target_card, INITIATOR_CLIENT, "client_request")

        if target_loan is not None:
            from .engine_credit import repay_loan

            repay_loan(state, step.ts, target_loan, loan_rules.arrears_amount(target_loan), "app")

        if target_deposit is not None:
            _topup_deposit(state, step.ts, target_deposit, rng)


def _show_banners(state: ClientState, ts: datetime, session, rng) -> None:

    if ts < BANNERS_AVAILABLE_FROM:
        return

    from . import params as params_module

    if rng.random() >= params_module.active().activity.banner_screen_share:
        return

    count = 1 + int(rng.random() < 0.30)

    for index in range(count):

        slot = str(rng.choice(list(BANNER_SLOTS), p=list(BANNER_SLOT_WEIGHTS)))
        offer = str(rng.choice(list(BANNER_OFFERS)))

        family = BANNER_OFFER_FAMILY.get(offer)

        offer_id = None
        campaign = None

        if family:
            offer_id = f"off_{stable_hash('banner', state.client_id, ts.toordinal(), index) % 10 ** 12:012d}"
            campaign = comm_module.campaign_for_family(family)

        shown_at = ts + timedelta(seconds=int(rng.integers(0, 4)))

        body = {
            "slot": slot,
            "offer": offer,
            "offer_id": offer_id,
            "product_id": None,
            "campaign_code": campaign,
        }

        shown = state.emit(
            state.factory.make(
                "banner_shown",
                shown_at,
                body,
                initiator=INITIATOR_BANK,
                correlation_id=session.session_id,
                link_type="session",
            )
        )

        # База подобрана так, чтобы ПОСЛЕ множителей по чертам и
        # семейству продукта наблюдаемый CTR держался около
        # эталонных 2.35 % из отчёта банка.
        ctr = 0.0168 * (0.6 + 1.2 * state.persona.trait("digital_affinity", ts))

        if family in ("cash_loan", "credit_card", "installment"):
            ctr *= 1.0 + 1.5 * state.persona.trait("credit_appetite", ts)

        if family and family in state.owned_families(ts):
            ctr *= 0.4

        if rng.random() >= min(0.30, ctr):
            continue

        state.emit(
            state.factory.make(
                "banner_clicked",
                shown_at + timedelta(seconds=int(rng.integers(2, 25))),
                body,
                initiator=INITIATOR_CLIENT,
                correlation_id=shown.event_id,
                link_type="offer",
            )
        )

        if family:
            state.offers.append(
                Offer(
                    offer_id=offer_id,
                    client_id=state.client_id,
                    product_id=None,
                    product_code=None,
                    product_family=family,
                    created_at=shown_at,
                    campaign_code=campaign or "",
                    channel="banner",
                )
            )


def _pay_bill_in_app(state: ClientState, ts: datetime, bill: dict, session, rng) -> None:

    sources = state.ledger.payment_sources(ts, bill["amount"])

    if not sources:
        return

    account = sources[0]

    body = dict(bill["body"])
    body["channel"] = "app"

    card = state.usable_card(account.account_id, ts)
    body["card_id"] = card.card_id if card else None

    counterpart = (
        f"merchant:{body.get('outlet_id')}" if body.get("outlet_id") else COUNTERPART_GOVERNMENT
    )

    _emit_money(
        state,
        ts + timedelta(seconds=int(rng.integers(5, 60))),
        "bill_payment",
        account.account_id,
        bill["amount"],
        "debit",
        counterpart,
        body,
        INITIATOR_CLIENT,
        correlation_id=session.session_id,
        link_type="session",
    )

    if bill in state.open_bills:
        state.open_bills.remove(bill)


def _topup_deposit(state: ClientState, ts: datetime, deposit, rng) -> None:

    # Условия продукта проверяются здесь, а не только при выборе
    # цели: пополнять можно между открытием и окончанием срока.
    if not deposit_rules.can_topup(deposit, ts):
        return

    amount = int(round(max(5_000, state.persona.true_income * rng.uniform(0.05, 0.35)) / 1_000) * 1_000)

    sources = [
        item
        for item in state.ledger.payment_sources(ts, amount)
        if item.account_id != deposit.account_id
    ]

    if not sources:
        return

    _own_transfer(
        state,
        ts,
        "deposit_topup",
        sources[0].account_id,
        deposit.account_id,
        amount,
        deposit.contract_id,
        "deposit_topup",
    )

    deposit.principal += amount


def _own_transfer(
    state: ClientState,
    ts: datetime,
    event_type: str,
    source_id: str,
    target_id: str,
    amount: int,
    contract_id: str,
    reason: str,
) -> None:
    """
    Перевод между своими счетами: две стороны с одинаковой
    суммой и отметкой own_account, чтобы деньги не выглядели
    ни внешним расходом, ни внешним доходом.

    Проводка одна: её делает первая нога, вторая только записывает
    событие с остатком. Иначе деньги двигались бы дважды.
    """

    debit = _emit_money(
        state, ts, event_type, source_id, amount, "debit", target_id,
        {
            "channel": "app",
            "contract_id": contract_id,
            "counterparty": "own_account",
            "reason": reason,
            "merchant_country": "KZ",
        },
        INITIATOR_CLIENT,
        correlation_id=contract_id,
        link_type="contract",
    )

    _emit_money(
        state, ts + timedelta(seconds=1), event_type, target_id, amount, "credit", source_id,
        {
            "channel": "system",
            "contract_id": contract_id,
            "counterparty": "own_account",
            "cause_event_id": debit.event_id,
            "reason": reason,
            "merchant_country": "KZ",
        },
        INITIATOR_SYSTEM,
        correlation_id=contract_id,
        link_type="contract",
        post=False,
    )


def unblock_card(state: ClientState, ts: datetime, card, initiator: str, reason: str) -> None:

    from .finance import cards as card_rules

    # Постоянную блокировку не снимает никто. Проверка стоит
    # здесь, а не только у вызывающих: путей разблокировки
    # четыре, и пропустить один слишком легко.
    if not card.releasable():
        return

    card_rules.unblock(card, ts)

    contract = state.contracts.get(card.contract_id)

    state.emit(
        state.factory.make(
            "card_unblocked",
            ts,
            {
                "product_id": contract.product_id if contract else None,
                "product_code": card.product_code,
                "product_version": contract.product_version if contract else 1,
                "tariff_version": contract.tariff_version if contract else 1,
                "product_family": contract.product_family if contract else "debit_card",
                "contract_id": card.contract_id,
                "account_id": card.account_id,
                "card_id": card.card_id,
                "reason": reason,
                "timestamp_quality": "exact",
            },
            initiator=initiator,
            correlation_id=card.contract_id,
            link_type="contract",
        )
    )


def _on_communication(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    contact = payload["contact"]

    state.comm_fatigue += 1

    product_id = None

    if contact.product_family:
        for item in payload["candidates"]:
            if item.view.family == contact.product_family:
                product_id = item.view.record.product_id
                break

    state.emit(
        state.factory.make(
            "communication_sent",
            ts,
            {
                "channel": contact.channel,
                "template": contact.template,
                "campaign_code": contact.campaign_code,
                "offer_id": contact.offer_id,
                "product_id": product_id,
                "purpose": contact.purpose,
                "delivered": contact.delivered,
            },
            initiator=INITIATOR_BANK,
            correlation_id=contact.offer_id,
            link_type="offer" if contact.offer_id else None,
        )
    )

    if not contact.clicked:
        return

    state.note(ts, "click", contact.campaign_code, {"channel": contact.channel})

    if contact.product_family and contact.offer_id:
        state.offers.append(
            Offer(
                offer_id=contact.offer_id,
                client_id=state.client_id,
                product_id=product_id or "",
                product_code=None,
                product_family=contact.product_family,
                created_at=ts,
                campaign_code=contact.campaign_code,
                channel=contact.channel,
            )
        )


def _on_block_expired(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Долгая блокировка снимается сама: банк перевыпускает карту
    или возвращает её в строй, и это видно в ленте.
    """

    card = state.cards.get(payload["card_id"])

    if card is None or card.status != CARD_BLOCKED:
        return

    unblock_card(state, ts, card, "bank_employee", "block_expired")


_HANDLERS["card_block_expired"] = _on_block_expired
_HANDLERS["session"] = _on_session
_HANDLERS["communication"] = _on_communication


__all__ = ["unblock_card"]
