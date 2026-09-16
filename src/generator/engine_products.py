from __future__ import annotations

import json
from datetime import datetime, timedelta

from . import params as params_module
from .behaviour import adoption as adoption_module
from .behaviour import support as support_module
from .behaviour import fraud as fraud_behaviour
from .config import (
    HISTORY_END,
    HISTORY_START,
    INITIATOR_BANK,
    INITIATOR_CLIENT,
    INITIATOR_EXTERNAL,
    INITIATOR_SYSTEM,
    PROFILE_FIELDS,
)
from .engine import _HANDLERS, _emit_money, _touch_client
from .engine_app import unblock_card
from .engine_credit import emit_product_closed
from .finance import cards as card_rules
from .finance import deposits as deposit_rules
from .finance import loans as loan_rules
from .finance.entities import (
    ACCOUNT_DEPOSIT,
    CARD_ACTIVE,
    CARD_BLOCKED,
    CONTRACT_CLOSED,
    Application,
    Card,
)
from .finance.ledger import COUNTERPART_BANK
from .life import calendar as cal
from .life import lifecycle as lifecycle_module
from .life import stress as stress_module
from .rng import (
    COMPONENT_CONTENT,
    NS_ADOPTION,
    NS_CARD,
    NS_DEPOSIT,
    NS_FRAUD,
    NS_PROFILE,
    NS_SUPPORT,
    event_rng,
    keyed_rng,
    stable_hash,
)
from .simulate import ClientState, _application_id, _money
from .world import products as product_catalog
from .world.dictionaries import FUNNEL_SCREENS, MCC_CASH


# ============================================================
# ПРОДУКТЫ, АНТИФРОД И ПРОФИЛЬ
# ============================================================
#
# Распространение продукта:
#
#   eligibility -> предложение -> заявка -> решение
#   -> активация -> первое использование
#
# Версия продукта фиксируется на дате подписания договора и
# живёт с ним. Новая версия касается действующего договора
# только по applies_to, а переход на другой продукт это
# отдельное событие миграции.
# ============================================================


REJECT_CASCADE = (
    ("age_limit", lambda persona, ts, income, dpd, loans: persona.age_at(ts) < 21 or persona.age_at(ts) > 72),
    ("income_not_confirmed", lambda persona, ts, income, dpd, loans: persona.income_type in ("unemployed", "student") or income < 95_000),
    ("existing_debt", lambda persona, ts, income, dpd, loans: loans >= 2),
    ("debt_service_ratio", lambda persona, ts, income, dpd, loans: False),
)


def _approval(state: ClientState, candidate, ts: datetime, stress: float, rng) -> tuple:

    settings = params_module.active().products

    persona = state.persona

    family = candidate.view.family

    base = settings.approval_base.get(family, 0.9)

    if family in ("cash_loan", "credit_card", "refinance", "installment"):

        base += settings.approval_income_factor * min(1.0, persona.declared_income / 700_000)
        base += settings.approval_discipline_factor * (persona.trait("financial_discipline", ts) - 0.5)
        base -= settings.approval_stress_penalty * stress

        worst = state.worst_dpd()

        if worst > 0:
            base -= settings.approval_dpd_penalty * min(1.0, worst / 60.0)

        if len(state.loans) >= 1:
            base -= settings.approval_existing_loan_penalty

    low, high = settings.approval_bounds

    probability = max(low, min(high, base))

    approved = rng.random() < probability

    if approved:
        return True, None

    open_loans = sum(1 for item in state.loans.values() if not item.closed)

    for reason, rule in REJECT_CASCADE:
        if rule(persona, ts, persona.declared_income, state.worst_dpd(), open_loans):
            return False, reason

    if stress > 0.55:
        return False, "scoring_declined"

    if rng.random() < settings.blacklist_share:
        return False, "blacklist"

    return False, rng.weighted(settings.reject_reason_weights)


def _funnel(state: ClientState, application: Application, ts: datetime, approved: bool,
            reject_reason: str | None, rng) -> datetime:
    """
    Воронка заявки в приложении: просмотр, форма, проверка,
    решение.
    """

    moment = ts

    stages = ["view", "application", "kyc", "approved" if approved else "rejected"]

    for index, stage in enumerate(stages):

        if index:
            moment = moment + timedelta(seconds=int(rng.integers(30, 900)))

        state.emit(
            state.factory.make(
                "app_screen",
                moment,
                {
                    "firebase_screen": FUNNEL_SCREENS[stage],
                    "product_id": application.product_id,
                    "funnel_stage": stage,
                    "reject_reason": reject_reason if stage == "rejected" else None,
                },
                initiator=INITIATOR_CLIENT,
                correlation_id=application.application_id,
                link_type="application",
            )
        )

    return moment


def _application_payload(application: Application, **extra) -> dict:

    body = {
        "application_id": application.application_id,
        "product_id": application.product_id,
        "product_code": application.product_code,
        "product_version": application.product_version,
        "offer_id": application.offer_id,
        "channel": application.channel,
        "requested_amount": application.requested_amount,
        "requested_term": application.requested_term,
        "decision": None,
        "reject_reason": None,
        "approved_amount": None,
        "approved_term": None,
    }

    body.update(extra)

    return body


def _on_adoption(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    settings = params_module.active().products

    persona = state.persona

    stress = payload["stress"]
    app_adopted = payload["app"]

    rng = event_rng(NS_ADOPTION, state.ordinal, ts.toordinal(), 0, COMPONENT_CONTENT)

    _apply_new_versions(state, ts)

    _migrate_products(sim, state, ts, rng)

    pool = adoption_module.candidates(
        persona,
        ts,
        state.held_codes(ts),
        state.held_counts(ts),
        state.assets(),
        app_adopted,
        bool(state.loans),
        len(state.open_contracts(ts)),
        stress,
    )

    if not pool:
        return

    recent_offer = None

    for offer in reversed(state.offers):
        if (ts - offer.created_at).days <= 14:
            recent_offer = offer
            break

    candidate = None

    if recent_offer is not None:
        candidate = next((item for item in pool if item.view.family == recent_offer.product_family), None)

    from_offer = candidate is not None

    if candidate is None:
        candidate = adoption_module.pick(pool, rng)

    if candidate is None:
        return

    probability = adoption_module.application_probability(
        persona, candidate, ts, from_offer, stress,
        total_weight=sum(item.weight for item in pool),
    )

    if rng.random() >= probability:
        return

    # --- заявка ---

    channels = tuple(candidate.version.channels) or ("branch",)

    channel = str(rng.choice(list(channels)))

    if channel == "app" and not app_adopted:
        channel = "branch"

    amount, term = sim._contract_terms(state, candidate.view, ts, rng)

    application = Application(
        application_id=_application_id(state.client_id, ts, len(state.applications)),
        client_id=state.client_id,
        product_id=candidate.view.record.product_id,
        product_code=candidate.view.code,
        product_family=candidate.view.family,
        product_version=candidate.version.product_version,
        channel=channel,
        submitted_at=ts,
        requested_amount=amount,
        requested_term=term,
        offer_id=recent_offer.offer_id if from_offer and recent_offer else None,
    )

    state.applications[application.application_id] = application

    state.emit(
        state.factory.make(
            "application_submitted",
            ts,
            _application_payload(application),
            initiator=INITIATOR_CLIENT,
            correlation_id=application.application_id,
            link_type="application",
        )
    )

    _touch_client(state, ts)

    approved, reason = _approval(state, candidate, ts, stress, rng)

    low, high = settings.decision_delay_seconds.get(channel, (60, 3600))

    decision_ts = ts + timedelta(seconds=int(rng.integers(low, high)))

    if channel == "app" and app_adopted:
        decision_ts = _funnel(state, application, ts, approved, reason, rng)

    if decision_ts >= HISTORY_END:
        return

    application.decision = "approved" if approved else "rejected"
    application.decided_at = decision_ts
    application.reject_reason = reason
    application.approved_amount = amount if approved else None
    application.approved_term = term if approved else None

    state.emit(
        state.factory.make(
            "application_decision",
            decision_ts,
            _application_payload(
                application,
                decision=application.decision,
                reject_reason=reason,
                approved_amount=application.approved_amount,
                approved_term=application.approved_term,
            ),
            initiator=INITIATOR_BANK,
            correlation_id=application.application_id,
            link_type="application",
        )
    )

    if not approved:
        return

    open_ts = decision_ts + timedelta(seconds=int(rng.integers(*settings.disbursement_delay_seconds)))

    if open_ts >= HISTORY_END:
        return

    contract = sim._open_contract(
        state,
        candidate.view,
        open_ts,
        amount,
        term,
        offer_id=application.offer_id,
        application_id=application.application_id,
        not_before=decision_ts,
    )

    _activate_product(sim, state, contract, candidate.view, open_ts, rng)


def _activate_product(sim, state: ClientState, contract, view, ts: datetime, rng) -> None:
    """
    Первое действие по новому договору: выдача кредита,
    зачисление на депозит, оплата страховки.
    """

    family = contract.product_family

    if family in ("cash_loan", "refinance", "installment"):

        account = state.primary_card_account(ts)

        if account is None or contract.amount_or_limit is None:
            return

        opened = next(
            (
                event
                for event in reversed(state.events)
                if event.event_type == "product_opened"
                and event.payload.get("contract_id") == contract.contract_id
            ),
            None,
        )

        if family != "installment":
            _emit_money(
                state,
                ts + timedelta(minutes=5),
                "loan_disbursement",
                account.account_id,
                int(contract.amount_or_limit),
                "credit",
                f"loan:{contract.contract_id}",
                {
                    "channel": "system",
                    "contract_id": contract.contract_id,
                    "cause_event_id": opened.event_id if opened else None,
                    "reason": "disbursement",
                    "merchant_country": "KZ",
                },
                INITIATOR_SYSTEM,
                correlation_id=contract.contract_id,
                link_type="contract",
            )

        loan = loan_rules.open_loan(
            contract.contract_id,
            int(contract.amount_or_limit),
            float(contract.rate or 0.28),
            int(contract.term or 12),
            ts,
            autopay=rng.random() < params_module.active().products.autopay_share,
        )

        state.loans[contract.contract_id] = loan

        state.emit(
            state.factory.make(
                "schedule_created",
                ts + timedelta(minutes=6),
                {
                    "contract_id": contract.contract_id,
                    "installment_no": len(loan.schedule),
                    "amount_due": loan.schedule[0].amount if loan.schedule else None,
                    "amount_paid": None,
                    "principal_outstanding": loan.principal_outstanding,
                    "days_past_due": 0,
                    "due_date": loan.schedule[0].due_date.date().isoformat() if loan.schedule else None,
                    "cause_event_id": opened.event_id if opened else None,
                    "reason": "annuity",
                },
                initiator=INITIATOR_SYSTEM,
                correlation_id=contract.contract_id,
                link_type="schedule",
            )
        )

        return

    if family in ("deposit", "deposit_certificate"):

        amount = int(contract.amount_or_limit or 0)

        sources = [
            item
            for item in state.ledger.payment_sources(ts, amount)
            if item.account_id != contract.account_id
        ]

        if not sources or amount <= 0:
            return

        from .engine_app import _own_transfer

        _own_transfer(
            state,
            ts + timedelta(minutes=2),
            "deposit_topup",
            sources[0].account_id,
            contract.account_id,
            amount,
            contract.contract_id,
            "initial_deposit",
        )

        terms = contract.terms

        state.deposits[contract.contract_id] = deposit_rules.open_deposit(
            contract_id=contract.contract_id,
            account_id=contract.account_id,
            amount=amount,
            rate=float(contract.rate or terms.get("rate") or 0.14),
            opened_at=ts,
            term_months=int(contract.term or 12),
            topup=bool(terms.get("topup", False)),
            withdrawal=bool(terms.get("withdrawal", False)),
            capitalisation=str(terms.get("capitalisation", "daily")),
        )

        return

    if family == "insurance":

        amount = int(contract.amount_or_limit or 0)

        sources = state.ledger.payment_sources(ts, amount)

        if not sources or amount <= 0:
            return

        _emit_money(
            state,
            ts + timedelta(minutes=3),
            "purchase",
            sources[0].account_id,
            amount,
            "debit",
            COUNTERPART_BANK,
            {
                "channel": "app",
                "contract_id": contract.contract_id,
                "reason": "insurance_premium",
                "mcc": "6300",
                "merchant_country": "KZ",
                "is_online": True,
                "is_subscription": False,
            },
            INITIATOR_CLIENT,
            correlation_id=contract.contract_id,
            link_type="contract",
        )


def _apply_new_versions(state: ClientState, ts: datetime) -> None:
    """
    Новая версия продукта касается действующего договора только
    по applies_to и только после уведомления.
    """

    catalog = product_catalog.catalog()

    for contract in list(state.contracts.values()):

        if not contract.is_open_at(ts) or not catalog.has(contract.product_code):
            continue

        view = catalog.view(contract.product_code)

        for version in view.later_versions(contract.opened_at):

            if version.applies_to != "existing_from_date":
                continue

            if (version.product_version, version.tariff_version) <= (
                contract.product_version, contract.tariff_version
            ):
                continue

            notice = version.notice_days or params_module.active().products.adoption.get(
                "notice_days_default", 30
            )

            effective = view.version_start(version) + timedelta(days=int(notice))

            if ts.date() != effective.date():
                continue

            terms_changed = version.product_version != contract.product_version

            event_type = "contract_terms_changed" if terms_changed else "product_repriced"

            contract.product_version = version.product_version
            contract.tariff_version = version.tariff_version
            contract.terms = dict(version.terms)

            state.pending_notice = True

            state.emit(
                state.factory.make(
                    event_type,
                    ts.replace(hour=0, minute=5),
                    {
                        "product_id": contract.product_id,
                        "product_code": contract.product_code,
                        "product_version": contract.product_version,
                        "tariff_version": contract.tariff_version,
                        "product_family": contract.product_family,
                        "contract_id": contract.contract_id,
                        "account_id": contract.account_id,
                        "card_id": contract.card_id,
                        "amount_or_limit": contract.amount_or_limit,
                        "term": contract.term,
                        "rate": contract.rate,
                        "reason": "tariff_update" if not terms_changed else "terms_update",
                        "timestamp_quality": "exact",
                    },
                    initiator=INITIATOR_BANK,
                    correlation_id=contract.contract_id,
                    link_type="contract",
                    effective_at=ts.replace(hour=0, minute=0, second=0),
                )
            )


# ============================================================
# МОШЕННИЧЕСТВО
# ============================================================


def _on_fraud_step(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    settings = params_module.active().fraud

    episode = payload["episode"]
    step = payload["step"]

    rng = event_rng(NS_FRAUD, state.ordinal, ts.toordinal(), payload["position"], COMPONENT_CONTENT)

    account = state.primary_card_account(ts)

    if account is None:
        return

    card = state.usable_card(account.account_id, ts)

    if step.kind == "probe":
        amount = int(step.amount_hint)
    else:
        limit = max(50_000, account.balance + account.credit_limit)
        amount = int(max(5_000, limit * step.amount_hint))

    amount = int(round(amount / 100) * 100)

    country = "TR" if step.foreign else "KZ"

    body = {
        "channel": "ecom" if step.online else "pos",
        "card_id": card.card_id if card else None,
        "merchant_name": "UNKNOWN MERCHANT" if episode.kind != "false_positive" else "TRAVEL SHOP",
        "mcc": "5999",
        "merchant_city": None if step.foreign else state.persona.settlement,
        "merchant_country": country,
        "is_online": step.online,
        "is_subscription": False,
        "reason": "purchase",
    }

    if card is None or not state.ledger.payment_sources(ts, amount):
        return

    event = _emit_money(
        state, ts, "purchase", account.account_id, amount, "debit",
        "external:unknown_merchant", body,
        INITIATOR_CLIENT if episode.kind == "false_positive" else INITIATOR_EXTERNAL,
    )

    state.purchases.append(event)

    state.note(ts, "fraud_episode_step", episode.kind, {"step": step.kind, "amount": amount})

    if step.kind != "strike" and episode.kind != "false_positive":
        return

    if not episode.detected:
        return

    alert_ts = ts + timedelta(minutes=episode.detect_delay_minutes)

    if alert_ts >= HISTORY_END:
        return

    band = fraud_behaviour.score_band(episode.kind, step.kind, step.foreign, step.amount_hint)

    alert = state.emit(
        state.factory.make(
            "fraud_alert",
            alert_ts,
            {
                "subject": "card",
                "card_id": card.card_id,
                "account_id": account.account_id,
                "score_band": band,
                "rule_code": fraud_behaviour.rule_code(episode.kind),
                "cause_event_id": event.event_id,
            },
            initiator=INITIATOR_SYSTEM,
            correlation_id=event.event_id,
            link_type="fraud_episode",
        )
    )

    state.fraud_alert_at = alert_ts

    decision = episode.decision

    decision_ts = alert_ts + timedelta(minutes=int(rng.integers(1, 60)))

    state.emit(
        state.factory.make(
            "fraud_decision",
            decision_ts,
            {
                "subject": "card",
                "card_id": card.card_id,
                "account_id": account.account_id,
                "decision": decision,
                "resolution": episode.client_response if episode.client_response != "no_response" else None,
                "cause_event_id": alert.event_id,
            },
            initiator=INITIATOR_BANK,
            correlation_id=alert.event_id,
            link_type="fraud_episode",
        )
    )

    if decision != "block":
        return

    card_rules.block(card, decision_ts, "fraud_suspicion", days=settings.card_block_days
                     if hasattr(settings, "card_block_days") else None)

    contract = state.contracts.get(card.contract_id)

    state.emit(
        state.factory.make(
            "card_blocked",
            decision_ts + timedelta(seconds=10),
            {
                "product_id": contract.product_id if contract else None,
                "product_code": card.product_code,
                "product_version": contract.product_version if contract else 1,
                "tariff_version": contract.tariff_version if contract else 1,
                "product_family": contract.product_family if contract else "debit_card",
                "contract_id": card.contract_id,
                "account_id": card.account_id,
                "card_id": card.card_id,
                "reason": "fraud_suspicion",
                "timestamp_quality": "exact",
            },
            initiator=INITIATOR_SYSTEM,
            correlation_id=alert.event_id,
            link_type="fraud_episode",
        )
    )

    # --- реакция клиента ---

    if episode.opens_case:

        case = support_module.open_case(
            state.persona, "fraud_alert" if episode.kind != "false_positive" else "card_blocked",
            decision_ts + timedelta(hours=int(rng.integers(1, 20))),
            alert.event_id, len(state.cases),
        )

        _emit_case(state, case)

        if episode.chargeback:

            back_ts = case.resolved_at + timedelta(days=int(rng.integers(*settings.chargeback_delay_days)))

            if back_ts < HISTORY_END:
                _emit_money(
                    state, back_ts, "chargeback", account.account_id, amount, "credit",
                    "external:unknown_merchant",
                    {
                        "channel": "system",
                        "card_id": card.card_id,
                        "cause_event_id": event.event_id,
                        "reason": "dispute_resolved",
                        "merchant_country": country,
                    },
                    INITIATOR_BANK,
                    correlation_id=case.case_id,
                    link_type="chargeback",
                )

    if episode.reissue:

        reissue_ts = decision_ts + timedelta(days=int(rng.integers(*settings.reissue_delay_days)))

        if reissue_ts < HISTORY_END:
            _reissue_card(state, card, reissue_ts)

    elif episode.client_response == "confirmed_by_client" or episode.kind == "false_positive":

        unblock_ts = decision_ts + timedelta(hours=int(rng.integers(*settings.unblock_delay_hours)))

        if unblock_ts < HISTORY_END:
            unblock_card(state, unblock_ts, card, INITIATOR_CLIENT, "confirmed_by_client")


def _reissue_card(state: ClientState, card, ts: datetime) -> None:

    contract = state.contracts.get(card.contract_id)

    card.status = "closed"
    card.closed_at = ts

    fresh = Card(
        card_id=f"crd_{stable_hash('reissue', card.card_id, ts.toordinal()) % 10 ** 12:012d}",
        account_id=card.account_id,
        client_id=card.client_id,
        contract_id=card.contract_id,
        product_code=card.product_code,
        issued_at=ts,
        activated_at=ts + timedelta(days=1),
        status=CARD_ACTIVE,
        reissued_from=card.card_id,
    )

    state.cards[fresh.card_id] = fresh

    if contract is not None:
        contract.card_id = fresh.card_id

    state.emit(
        state.factory.make(
            "card_reissued",
            ts,
            {
                "product_id": contract.product_id if contract else None,
                "product_code": card.product_code,
                "product_version": contract.product_version if contract else 1,
                "tariff_version": contract.tariff_version if contract else 1,
                "product_family": contract.product_family if contract else "debit_card",
                "contract_id": card.contract_id,
                "account_id": card.account_id,
                "card_id": fresh.card_id,
                "reason": "fraud_reissue",
                "timestamp_quality": "exact",
            },
            initiator=INITIATOR_BANK,
            correlation_id=card.card_id,
            link_type="contract",
        )
    )


def _emit_case(state: ClientState, case) -> None:

    state.cases.append(case)

    body = {
        "case_id": case.case_id,
        "channel": case.channel,
        "topic": case.topic,
        "status": "open",
        "resolution": None,
        "cause_event_id": case.cause_event_id,
    }

    state.emit(
        state.factory.make(
            "case_opened",
            case.opened_at,
            body,
            initiator=INITIATOR_CLIENT,
            correlation_id=case.case_id,
            link_type="case",
        )
    )

    if case.updated_at is not None and case.updated_at < HISTORY_END:
        state.emit(
            state.factory.make(
                "case_updated",
                case.updated_at,
                dict(body, status="in_progress"),
                initiator=INITIATOR_BANK,
                correlation_id=case.case_id,
                link_type="case",
            )
        )

    if case.resolved_at < HISTORY_END:
        state.emit(
            state.factory.make(
                "case_resolved",
                case.resolved_at,
                dict(body, status="resolved", resolution=case.resolution),
                initiator=INITIATOR_BANK,
                correlation_id=case.case_id,
                link_type="case",
            )
        )


# ============================================================
# ПРОФИЛЬ
# ============================================================


def _on_profile_change(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    event = payload["event"]

    fields = event.changes_profile

    if not fields:
        return

    for name in fields:

        old = state.profile_values.get(name)

        new = _new_profile_value(state, name, event)

        if new is None or str(new) == str(old):
            continue

        state.profile_values[name] = new

        state.emit(
            state.factory.make(
                "profile_change",
                ts,
                {
                    "field_name": name,
                    "old_value": None if old is None else str(old),
                    "new_value": str(new),
                    "change_source": "client" if event.kind in ("move", "wedding", "divorce") else "application",
                    "confirmed": event.confirmed,
                },
                initiator=INITIATOR_CLIENT,
                effective_at=event.ts,
            )
        )


def _new_profile_value(state: ClientState, name: str, event):

    payload = event.payload

    if name == "region":
        return payload.get("region")

    if name == "city":
        return payload.get("settlement")

    if name == "children":
        return payload.get("children_after")

    if name == "family_status":
        return payload.get("family_status_after")

    if name == "declared_income":
        current = state.profile_values.get("declared_income") or state.persona.declared_income
        factor = payload.get("factor") or payload.get("income_factor") or 1.0
        return int(round(current * float(factor) / 5000) * 5000)

    if name == "income_type":
        return "unemployed" if event.kind == "job_loss" else state.profile_values.get("income_type")

    if name == "industry":
        return state.profile_values.get("industry")

    if name == "salary_day":
        return state.profile_values.get("salary_day")

    return None


def _on_support_check(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Обращение связано с причиной, а решение имеет последствие.
    """

    cause = payload["cause"]

    rng = event_rng(NS_SUPPORT, state.ordinal, ts.toordinal(), len(state.cases), COMPONENT_CONTENT)

    probability = support_module.contact_probability(state.persona, cause, ts, payload["stress"])

    # Обращение по одному поводу не повторяется каждый день.
    probability /= 6.0

    if rng.random() >= probability:
        return

    cause_event_id = None

    for event in reversed(state.events):
        if cause == "failed_operation" and event.event_type == "app_operation":
            if event.payload.get("status") == "failed":
                cause_event_id = event.event_id
                break
        if cause == "delinquency" and event.event_type == "delinquency_registered":
            cause_event_id = event.event_id
            break
        if cause == "card_blocked" and event.event_type == "card_blocked":
            cause_event_id = event.event_id
            break

    case = support_module.open_case(state.persona, cause, ts, cause_event_id, len(state.cases))

    _emit_case(state, case)

    _touch_client(state, ts)

    if case.resolution == "card_unblocked":
        card = next(
            (item for item in state.cards.values() if item.is_blocked_at(case.resolved_at)), None
        )
        if card is not None and case.resolved_at < HISTORY_END:
            unblock_card(state, case.resolved_at, card, INITIATOR_BANK, "support_resolution")

    if case.resolution == "record_corrected":
        state.pending_notice = True


def _migrate_products(sim, state: ClientState, ts: datetime, rng) -> None:
    """
    Переход на продукт-преемник: добровольный по предложению,
    принудительный по уведомлению.
    """

    for view, target, policy in adoption_module.migration_targets(ts, state.held_codes(ts)):

        contract = next(
            (
                item
                for item in state.open_contracts(ts)
                if item.product_code == view.code
            ),
            None,
        )

        if contract is None:
            continue

        if policy == "voluntary":
            chance = 0.0015 * (0.5 + 1.5 * state.persona.trait("digital_affinity", ts))
        elif policy == "forced_with_notice":
            chance = 1.0 if ts >= view.version_start(view.version_at(ts)) else 0.0
        else:
            chance = 0.02 / 365.0

        if rng.random() >= chance:
            continue

        amount, term = sim._contract_terms(state, target, ts, rng)

        contract.status = CONTRACT_CLOSED
        contract.closed_at = ts

        emit_product_closed(state, ts, contract, "migrated_to_successor")

        fresh = sim._open_contract(
            state,
            target,
            ts + timedelta(minutes=10),
            amount,
            term,
            previous_product_id=contract.product_id,
            migration_reason="successor_offer" if policy == "voluntary" else "forced_migration",
        )

        _activate_product(sim, state, fresh, target, ts + timedelta(minutes=12), rng)

        state.note(ts, "adoption_decision", "migration", {"from": view.code, "to": target.code})

        return


_HANDLERS["support_check"] = _on_support_check
_HANDLERS["adoption"] = _on_adoption
_HANDLERS["fraud_step"] = _on_fraud_step
_HANDLERS["profile_change"] = _on_profile_change


__all__ = ["_emit_case"]
