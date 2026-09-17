from __future__ import annotations

import json
from datetime import datetime, timedelta

from . import params as params_module
from .config import (
    EVENT_TYPE_PRIORITY,
    HISTORY_END,
    INITIATOR_CLIENT,
    INITIATOR_SYSTEM,
    PROFILE_FIELDS,
)
from .engine import _HANDLERS, _emit_money
from .engine_credit import emit_product_closed
from .finance import cards as card_rules
from .finance import deposits as deposit_rules
from .finance.entities import (
    ACCOUNT_CREDIT_CARD,
    CARD_CLOSED,
    CONTRACT_CLOSED,
)
from .finance.ledger import COUNTERPART_BANK, COUNTERPART_GOVERNMENT
from .life import calendar as cal
from .life import lifecycle as lifecycle_module
from .life import stress as stress_module
from .observe import coverage as coverage_module
from .observe import defects as defect_module
from .rng import (
    NS_CARD_CREDIT,
    NS_CONSENT,
    NS_DEPOSIT,
    NS_DEPOSIT_CLOSE,
    NS_LEDGER,
    keyed_rng,
    stable_hash,
)
from .simulate import ClientState, CommunityResult
from .world.dictionaries import MCC_TRANSFER


# ============================================================
# КОНЕЦ МЕСЯЦА И СБОРКА
# ============================================================
#
# Раз в месяц банк начисляет проценты, списывает комиссии и
# снимает остатки. Тогда же пересчитывается состояние жизненного
# цикла и, если что-то изменилось, появляется новая версия
# профиля.
# ============================================================


def _expire_cards(sim, state: ClientState, day: datetime) -> None:
    """
    Карта заканчивается по сроку и перевыпускается заранее.
    Раньше перевыпуск бывал только после мошенничества.
    """

    from .engine_products import _reissue_card

    for card in list(state.cards.values()):

        if card.expires_at is None or card.closed_at is not None:
            continue

        if card.status == CARD_CLOSED:
            continue

        remaining = (card.expires_at - day).days

        if remaining > 31 or remaining < 0:
            continue

        moment = day.replace(hour=12, minute=int(stable_hash(card.card_id) % 60))

        if moment >= HISTORY_END:
            continue

        _reissue_card(state, card, moment, reason="expiry")


def _close_deposits_early(sim, state: ClientState, day: datetime) -> None:
    """
    Досрочное закрытие вклада: деньги понадобились раньше срока.
    В трудный период это случается чаще.
    """

    settings = params_module.active().products

    stress = stress_module.level_at(state.stress_episodes, day)

    boost = 1.0 + params_module.active().stress.deposit_close_boost * stress

    monthly = settings.deposit_early_close_share_per_year / 12.0 * boost

    for contract_id, deposit in list(state.deposits.items()):

        contract = state.contracts.get(contract_id)

        if contract is None or not contract.is_open_at(day):
            continue

        if deposit.matures_at <= day:
            continue

        rng = keyed_rng(NS_DEPOSIT_CLOSE, state.ordinal, day.toordinal(), stable_hash(contract_id) % 9973)

        if rng.random() >= monthly:
            continue

        _close_deposit(state, day.replace(hour=15), deposit, early=True)


def _withdraw_consent(state: ClientState, day: datetime) -> None:
    """
    Согласие на рассылку можно и отозвать. Раньше это была
    дорога в одну сторону.
    """

    if state.consent_at is None or day <= state.consent_at:
        return

    settings = params_module.active().activity

    rng = keyed_rng(NS_CONSENT, state.ordinal, day.toordinal())

    if rng.random() >= settings.consent_withdrawal_per_year / 12.0:
        return

    state.consent_at = None

    state.emit(
        state.factory.make(
            "profile_change",
            day.replace(hour=19, minute=40),
            {
                "field_name": "consent_marketing",
                "old_value": "true",
                "new_value": "false",
                "change_source": "client",
                "confirmed": True,
            },
            initiator=INITIATOR_CLIENT,
        )
    )


def _card_statement(sim, state: ClientState, day: datetime, month: datetime) -> None:
    """
    Выписка по карте рассрочки.

    Проценты на наличный долг, минимальный платёж месяца и его
    исполнение переводом со своего счёта. Неоплаченный платёж
    это пропуск и просрочка, как у обычного кредита.
    """

    settings = params_module.active().products

    month_index = cal.month_index(month)

    for contract_id, credit in list(state.card_credits.items()):

        if credit.closed:
            continue

        contract = state.contracts.get(contract_id)

        if contract is None or not contract.is_open_at(day):
            continue

        # --- проценты на наличный долг ---

        interest = card_rules.monthly_interest(credit)

        if interest > 0:

            credit.accrued_interest += interest

            _emit_money(
                state,
                day.replace(hour=23, minute=5),
                "fee_charge",
                credit.account_id,
                interest,
                "debit",
                COUNTERPART_BANK,
                {
                    "channel": "system",
                    "contract_id": contract_id,
                    "accrual_period": month.strftime("%Y-%m"),
                    "reason": "periodic_contract_rule",
                    "merchant_country": "KZ",
                },
                INITIATOR_SYSTEM,
                correlation_id=contract_id,
                link_type="contract",
            )

        payment = card_rules.minimum_payment(credit, month_index)

        if payment <= 0:
            continue

        due_ts = day.replace(hour=23, minute=10)

        due_event = state.emit(
            state.factory.make(
                "installment_due",
                due_ts,
                {
                    "contract_id": contract_id,
                    "installment_no": None,
                    "amount_due": payment,
                    "amount_paid": None,
                    "principal_outstanding": credit.outstanding,
                    "days_past_due": credit.dpd,
                    "due_date": day.date().isoformat(),
                    "cause_event_id": None,
                    "reason": "card_statement",
                },
                initiator=INITIATOR_SYSTEM,
                correlation_id=contract_id,
                link_type="schedule",
            )
        )

        paid = _pay_card(sim, state, day, credit, contract_id, payment, due_event, month_index)

        if paid:
            _card_arrears_cleared(state, day, credit, contract_id)
            continue

        _card_missed(state, day, credit, contract_id, payment, due_event)


def _pay_card(sim, state: ClientState, day, credit, contract_id, payment, due_event, month_index) -> bool:
    """
    Платёж по карте: перевод со своего счёта на счёт карты.
    Обе стороны помечены own_account, поэтому деньги клиента не
    исчезают и не появляются.
    """

    rng = keyed_rng(NS_CARD_CREDIT, state.ordinal, day.toordinal(), stable_hash(contract_id) % 9973)

    settings = params_module.active().products

    discipline = state.persona.trait("financial_discipline", day)

    low, high = settings.discipline_bands

    band = "high" if discipline > high else "mid" if discipline > low else "low"

    if rng.random() >= settings.on_time_payment_probability[band]:
        return False

    def own_sources(value: int) -> list:
        return [
            item
            for item in state.ledger.payment_sources(day, value)
            if item.kind != "credit_card"
        ]

    sources = own_sources(payment)

    # Как и по кредиту, деньги к сроку подтягивают из другого
    # банка или наличными.
    if not sources:

        from .engine_credit import _topup_before_payment

        if _topup_before_payment(state, day.replace(hour=22), payment, rng):
            sources = own_sources(payment)

    # Денег не хватило на весь минимальный платёж: платят
    # сколько могут, как и по обычному кредиту. Иначе карта
    # уходит в просрочку с первого же тесного месяца.
    if not sources:

        capacity = max(
            (
                item.available
                for item in state.ledger.accounts.values()
                if item.visible and item.is_open_at(day) and item.kind not in ("loan", "credit_card")
            ),
            default=0,
        )

        if capacity >= settings.partial_payment_min_share * payment:
            payment = int(capacity)
            sources = own_sources(payment)

    if not sources or payment <= 0:
        return False

    source = sources[0]

    moment = day.replace(hour=23, minute=15)

    _emit_money(
        state,
        moment,
        "loan_payment",
        source.account_id,
        payment,
        "debit",
        credit.account_id,
        {
            "channel": "app",
            "contract_id": contract_id,
            "counterparty": "own_account",
            "cause_event_id": due_event.event_id,
            "reason": "card_statement",
            "mcc": MCC_TRANSFER,
            "merchant_country": "KZ",
        },
        INITIATOR_CLIENT,
        correlation_id=contract_id,
        link_type="schedule",
    )

    _emit_money(
        state,
        moment + timedelta(seconds=1),
        "transfer_in",
        credit.account_id,
        payment,
        "credit",
        source.account_id,
        {
            "channel": "system",
            "contract_id": contract_id,
            "counterparty": "own_account",
            "reason": "card_statement",
            "mcc": MCC_TRANSFER,
            "merchant_country": "KZ",
        },
        INITIATOR_SYSTEM,
        correlation_id=contract_id,
        link_type="schedule",
    )

    applied = card_rules.apply_card_payment(credit, payment, month_index)

    state.emit(
        state.factory.make(
            "installment_paid",
            moment + timedelta(seconds=2),
            {
                "contract_id": contract_id,
                "installment_no": None,
                "amount_due": payment,
                "amount_paid": applied,
                "principal_outstanding": credit.outstanding,
                "days_past_due": 0,
                "due_date": day.date().isoformat(),
                "cause_event_id": due_event.event_id,
                "reason": "payment",
            },
            initiator=INITIATOR_CLIENT,
            correlation_id=contract_id,
            link_type="schedule",
        )
    )

    return True


def _card_missed(state: ClientState, day, credit, contract_id, payment, due_event) -> None:
    """
    Неоплаченная выписка: пропуск, рост просрочки и вехи.
    """

    settings = params_module.active().products

    credit.dpd += 30

    state.emit(
        state.factory.make(
            "installment_missed",
            day.replace(hour=23, minute=20),
            {
                "contract_id": contract_id,
                "installment_no": None,
                "amount_due": payment,
                "amount_paid": None,
                "principal_outstanding": credit.outstanding,
                "days_past_due": credit.dpd,
                "due_date": day.date().isoformat(),
                "cause_event_id": due_event.event_id,
                "reason": "missed",
            },
            initiator=INITIATOR_SYSTEM,
            correlation_id=contract_id,
            link_type="schedule",
        )
    )

    for milestone in settings.dpd_milestones:

        if credit.dpd < milestone or milestone in credit.delinquency_marks:
            continue

        credit.delinquency_marks = credit.delinquency_marks + (milestone,)

        state.emit(
            state.factory.make(
                "delinquency_registered",
                day.replace(hour=23, minute=25),
                {
                    "contract_id": contract_id,
                    "installment_no": None,
                    "amount_due": payment,
                    "amount_paid": None,
                    "principal_outstanding": credit.outstanding,
                    "days_past_due": milestone,
                    "due_date": None,
                    "cause_event_id": None,
                    "reason": "delinquency",
                },
                initiator=INITIATOR_SYSTEM,
                correlation_id=contract_id,
                link_type="schedule",
            )
        )

        break


def _card_arrears_cleared(state: ClientState, day, credit, contract_id) -> None:

    if credit.dpd <= 0:
        return

    state.emit(
        state.factory.make(
            "arrears_cleared",
            day.replace(hour=23, minute=28),
            {
                "contract_id": contract_id,
                "installment_no": None,
                "amount_due": None,
                "amount_paid": None,
                "principal_outstanding": credit.outstanding,
                "days_past_due": 0,
                "due_date": None,
                "cause_event_id": None,
                "reason": "arrears_cleared",
            },
            initiator=INITIATOR_SYSTEM,
            correlation_id=contract_id,
            link_type="schedule",
        )
    )

    credit.dpd = 0
    credit.delinquency_marks = ()


def _sweep_bills(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Счёт, не оплаченный в приложении к сроку, клиент платит
    мимо банка: в кассе или через другой банк.
    """

    remaining = []

    for bill in state.open_bills:

        if bill["deadline"] > ts:
            remaining.append(bill)
            continue

        rng = keyed_rng(NS_LEDGER, state.ordinal, ts.toordinal(), stable_hash(bill["kind"]) % 997)

        moment = ts.replace(hour=int(rng.integers(10, 20)), minute=int(rng.integers(0, 60)))

        sources = state.ledger.payment_sources(moment, bill["amount"])

        # В паузе клиент не платит и старые счета через этот
        # банк: они закрываются мимо него.
        silenced = lifecycle_module.silenced_streams(state.pauses, moment)

        if sources and "bills" not in silenced and rng.random() < 0.45:

            body = dict(bill["body"])
            body["channel"] = "branch"
            body["is_online"] = False

            _emit_money(
                state, moment, "bill_payment", sources[0].account_id, bill["amount"], "debit",
                COUNTERPART_GOVERNMENT, body, INITIATOR_CLIENT,
            )
        else:
            # Оплачено вне наблюдаемого контура.
            hidden = state.ledger.accounts[state.ledger.cash_id]
            if hidden.balance >= bill["amount"]:
                state.ledger.post(moment, "hidden", hidden.account_id,
                                  f"merchant:{bill['kind']}", bill["amount"])
            state.note(moment, "hidden_purchase", "bill_paid_outside", {"amount": bill["amount"]})

    state.open_bills = remaining


def month_end(sim, state: ClientState, day: datetime) -> None:

    settings = params_module.active().products

    month = cal.month_start(day)

    ts = day.replace(hour=23, minute=50, second=0, microsecond=0)

    # --- проценты по депозитам ---

    for contract_id, deposit in list(state.deposits.items()):

        if deposit.closed:
            continue

        account = state.ledger.get(deposit.account_id)

        if account is None or account.balance <= 0:
            continue

        interest = deposit_rules.monthly_interest(deposit, account.balance, month)

        if interest > 0:
            _emit_money(
                state, ts, "interest_credit", account.account_id, interest, "credit",
                COUNTERPART_BANK,
                {
                    "channel": "system",
                    "contract_id": contract_id,
                    "accrual_period": month.strftime("%Y-%m"),
                    "reason": "periodic_contract_rule",
                    "merchant_country": "KZ",
                },
                INITIATOR_SYSTEM,
                correlation_id=contract_id,
                link_type="contract",
            )

        if deposit_rules.matured(deposit, day):
            _close_deposit(state, ts, deposit)

    # --- ежемесячная комиссия за обслуживание ---

    for contract in state.open_contracts(day):

        fee = card_rules.monthly_fee(contract.terms)

        if fee <= 0 or contract.account_id is None:
            continue

        if not state.ledger.can_debit(contract.account_id, fee):
            continue

        _emit_money(
            state, ts, "fee_charge", contract.account_id, fee, "debit", COUNTERPART_BANK,
            {
                "channel": "system",
                "contract_id": contract.contract_id,
                "accrual_period": month.strftime("%Y-%m"),
                "reason": "periodic_contract_rule",
                "merchant_country": "KZ",
            },
            INITIATOR_SYSTEM,
            correlation_id=contract.contract_id,
            link_type="contract",
        )

    # --- кешбэк за месяц ---

    for (contract_id, account_id), value in sorted(state.pending_cashback.items()):

        if value <= 0:
            continue

        _emit_money(
            state, ts.replace(minute=45), "cashback_credit", account_id, value, "credit",
            COUNTERPART_BANK,
            {
                "channel": "system",
                "contract_id": contract_id,
                "accrual_period": month.strftime("%Y-%m"),
                "reason": "periodic_contract_rule",
                "merchant_country": "KZ",
            },
            INITIATOR_SYSTEM,
            correlation_id=contract_id,
            link_type="contract",
        )

    state.pending_cashback = {}

    # --- снимки остатков ---
    #
    # Витрина остатков присылает строку по счёту, по которому
    # был оборот. Замерший счёт попадает в неё лишь изредка,
    # и именно поэтому у спящего клиента бывают месяцы совсем
    # без событий.

    moved = {
        event.payload.get("account_id")
        for event in state.events
        if event.event_time >= month and event.payload.get("account_id")
    }

    snapshot_rng = keyed_rng(NS_LEDGER, state.ordinal, day.toordinal(), 31)

    silent_month = not any(event.event_time >= month for event in state.events)

    threshold = 0.08 if silent_month else 0.35

    for account in state.ledger.visible_accounts(day):

        if account.account_id not in moved and snapshot_rng.random() >= threshold:
            continue

        state.emit(
            state.factory.make(
                "balance_snapshot",
                ts.replace(minute=55),
                {
                    "amount": abs(account.balance),
                    "direction": "credit" if account.balance >= 0 else "debit",
                    "status": "approved",
                    "account_id": account.account_id,
                    "contract_id": account.contract_id,
                    "balance_after": account.balance,
                    "accrual_period": month.strftime("%Y-%m"),
                    "reason": "periodic_contract_rule",
                    "channel": "system",
                    "merchant_country": "KZ",
                },
                initiator=INITIATOR_SYSTEM,
                correlation_id=account.contract_id,
                link_type="contract",
            )
        )

    _card_statement(sim, state, day, month)
    _expire_cards(sim, state, day)
    _close_deposits_early(sim, state, day)
    _withdraw_consent(state, day)

    # --- счётчики месяца ---

    state.monthly_atm = 0
    state.monthly_atm_count = 0
    state.monthly_transfer = 0
    state.monthly_cashback = 0
    state.month_purchases = 0
    state.comm_fatigue = max(0, state.comm_fatigue - 4)

    # --- жизненный цикл ---

    _update_state(state, day)

    # --- версия профиля ---

    _update_profile(state, day)


def _close_deposit(state: ClientState, ts: datetime, deposit, early: bool = False) -> None:
    """
    Срок вышел: депозит либо пролонгируется на действующих
    условиях, либо закрывается с переводом остатка на карту.
    """

    settings = params_module.active().products

    account = state.ledger.get(deposit.account_id)

    contract = state.contracts.get(deposit.contract_id)

    if account is None or contract is None:
        return

    rng = keyed_rng(NS_DEPOSIT, state.ordinal, ts.toordinal(), stable_hash(deposit.contract_id) % 9973)

    from .world import products as product_catalog

    catalog = product_catalog.catalog()

    if not early and rng.random() < settings.deposit_rollover_share and catalog.has(contract.product_code):

        view = catalog.view(contract.product_code)

        version = view.version_at(ts)

        # Пролонгация применяет условия, действующие на дату
        # продления: это и есть applies_to on_renewal.
        contract.product_version = version.product_version
        contract.tariff_version = version.tariff_version
        contract.terms = dict(version.terms)
        contract.renewals += 1

        rate = version.terms.get("rate")

        if rate is None and "rate_by_term" in version.terms and contract.term:
            table = version.terms["rate_by_term"]
            rate = table.get(contract.term) or table.get(str(contract.term))

        contract.rate = float(rate) if rate is not None else contract.rate

        deposit.rate = float(contract.rate or deposit.rate)
        deposit.matures_at = cal.add_months(ts, int(contract.term or 12))

        state.emit(
            state.factory.make(
                "product_renewed",
                ts,
                {
                    "product_id": contract.product_id,
                    "product_code": contract.product_code,
                    "product_version": contract.product_version,
                    "tariff_version": contract.tariff_version,
                    "product_family": contract.product_family,
                    "contract_id": contract.contract_id,
                    "account_id": contract.account_id,
                    "card_id": None,
                    "amount_or_limit": account.balance,
                    "term": contract.term,
                    "rate": contract.rate,
                    "reason": "rollover",
                    "timestamp_quality": "exact",
                },
                initiator=INITIATOR_SYSTEM,
                correlation_id=contract.contract_id,
                link_type="contract",
            )
        )

        return

    target = state.primary_card_account(ts)

    if target is not None and account.balance > 0:

        from .engine_app import _own_transfer

        _own_transfer(
            state, ts, "deposit_withdrawal", account.account_id, target.account_id,
            account.balance, deposit.contract_id, "matured",
        )

    deposit.closed = True

    contract.status = CONTRACT_CLOSED
    contract.closed_at = ts
    account.closed_at = ts

    emit_product_closed(state, ts, contract, "matured")


def _update_state(state: ClientState, day: datetime) -> None:

    persona = state.persona

    stress = stress_module.level_at(state.stress_episodes, day)

    silence = (day - state.last_client_event).days if state.last_client_event else 9999

    month = cal.month_start(day)

    current = [
        event
        for event in state.events
        if event.change_initiator == INITIATOR_CLIENT and event.event_time >= month
    ]

    previous_month = cal.month_start(month - timedelta(days=1))

    previous = [
        event
        for event in state.events
        if event.change_initiator == INITIATOR_CLIENT
        and previous_month <= event.event_time < month
    ]

    ratio = len(current) / max(1, len(previous)) if previous else 1.0

    new_state, cause = lifecycle_module.month_state(
        persona=persona,
        ts=day,
        previous=state.state,
        days_since_client_event=silence,
        activity_ratio=ratio,
        stress_level=stress,
        worst_dpd=state.worst_dpd(),
        has_open_contract=bool(state.open_contracts(day)),
        returned_recently=state.returned_flag,
    )

    state.returned_flag = False

    if new_state != state.state:
        state.note(day, "state_transition", new_state, {"from": state.state, "cause": cause})
        state.state = new_state

    # Дата закрытия отношений живёт РОВНО пока клиент закрыт.
    # Клиент, который вернулся и снова покупает, отношений не
    # прекращал, и покрытие источников обязано это показывать:
    # иначе таблица покрытия говорит «источник кончился», а в
    # ленте после этой даты лежат сотни его операций.
    if new_state == lifecycle_module.STATE_CLOSED:
        if state.closed_at is None:
            state.closed_at = day
    elif state.closed_at is not None:
        state.closed_at = None


def _update_profile(state: ClientState, day: datetime) -> None:

    persona = state.persona

    values = dict(state.profile_values)

    open_contracts = state.open_contracts(day)

    credit_limit = sum(
        int(item.amount_or_limit or 0)
        for item in open_contracts
        if item.product_family == "credit_card"
    )

    used = -sum(
        account.balance
        for account in state.ledger.accounts.values()
        if account.kind == ACCOUNT_CREDIT_CARD and account.balance < 0
    )

    values.update(
        {
            "age": persona.age_at(day),
            "pensioner": persona.is_pensioner_at(day),
            "relationship_months": persona.relationship_months_at(day),
            "contracts_count": len(state.contracts),
            "active_contracts": len(open_contracts),
            "holds_credit_card": any(item.product_family == "credit_card" for item in open_contracts),
            "holds_debit_card": any(item.product_family == "debit_card" for item in open_contracts),
            "holds_deposit": any(
                item.product_family in ("deposit", "deposit_certificate") for item in open_contracts
            ),
            "credit_limit": float(credit_limit) if credit_limit else None,
            "credit_utilization": (
                round(min(1.2, used / credit_limit), 4) if credit_limit else None
            ),
        }
    )

    if values == state.profile_values and state.profile_versions:
        return

    rng = keyed_rng(NS_LEDGER, state.ordinal, day.toordinal(), 99)

    version = len(state.profile_versions) + 1

    if state.profile_versions:
        state.profile_versions[-1]["valid_to"] = day

    row = {
        "client_id": state.client_id,
        "profile_version": version,
        "valid_from": day,
        "valid_to": None,
        "record_time": day + timedelta(hours=int(rng.integers(1, 30))),
        "change_source": "system",
        "confirmed": True,
        "change_reason": "monthly_recalculation",
    }

    row.update({name: values.get(name) for name in PROFILE_FIELDS})

    state.profile_versions.append(row)

    state.profile_values = values


# ============================================================
# СБОРКА РЕЗУЛЬТАТА
# ============================================================


def finish(sim) -> CommunityResult:

    events: list = []
    profile_versions: list = []
    coverage_rows: list = []
    truth_clients: list = []
    truth_events: list = []
    truth_relationships: list = []

    for ordinal in sorted(sim.clients):

        state = sim.clients[ordinal]

        _emit_refunds(state)

        observed, corrections = defect_module.apply(state.events, ordinal)

        observed.sort(
            key=lambda item: (
                item.event_time,
                EVENT_TYPE_PRIORITY.get(item.event_type, 99),
                item.event_version,
                item.event_id,
            )
        )

        _replay_balances(state, observed)

        # Ошибка витрины вносится последней: остатки уже
        # посчитаны по настоящим суммам, и опечатка остаётся
        # только в той версии, которую банк потом исправил.
        defect_module.apply_first_version_errors(observed, corrections)

        for index, event in enumerate(observed):
            event.sequence_number = index
            events.append(_row(event))

        profile_versions.extend(state.profile_versions)

        for row in coverage_module.coverage_rows(
            state.persona, state.opening_state, state.closed_at
        ):
            coverage_rows.append(
                {
                    "client_id": row.client_id,
                    "source": row.source,
                    "first_available_at": row.first_available_at,
                    "last_available_at": row.last_available_at,
                    "first_seen": row.first_seen,
                    "coverage_status": row.coverage_status,
                    "coverage_reason": row.coverage_reason,
                    "opening_state": row.opening_state,
                }
            )

        truth_clients.append(_truth_client(state, sim.graph.households.get(ordinal)))
        truth_events.extend(state.truth)
        truth_events.extend(_truth_plan(state))

    for relation in sim.graph.relationships:

        persona = sim.personas[relation.client_ordinal]

        truth_relationships.append(
            {
                "client_id": persona.client_id,
                "counterpart_id": relation.counterpart.counterpart_id,
                "counterpart_kind": relation.counterpart.kind,
                "counterpart_client_id": (
                    sim.personas[relation.counterpart.client_ordinal].client_id
                    if relation.counterpart.client_ordinal in sim.personas
                    else None
                ),
                "relation_type": relation.relation_type,
                "strength": relation.strength,
                "typical_frequency": relation.typical_frequency,
                "typical_amount_low": relation.typical_amount_low,
                "typical_amount_high": relation.typical_amount_high,
                "valid_from": relation.valid_from,
                "valid_to": relation.valid_to,
                "household_id": relation.household_id,
            }
        )

    return CommunityResult(
        events=events,
        profile_versions=profile_versions,
        coverage=coverage_rows,
        truth_clients=truth_clients,
        truth_events=truth_events,
        truth_relationships=truth_relationships,
    )


def _emit_refunds(state: ClientState) -> None:
    """
    Возвраты и отмены: ссылаются на исходную операцию и не
    превышают её сумму.
    """

    for plan in defect_module.plan_refunds(state.purchases):

        if plan["ts"] >= HISTORY_END:
            continue

        cause = plan["cause"]

        account_id = cause.payload.get("account_id")

        if account_id is None:
            continue

        body = {
            "channel": "system",
            "card_id": cause.payload.get("card_id"),
            "merchant_id": cause.payload.get("merchant_id"),
            "outlet_id": cause.payload.get("outlet_id"),
            "merchant_name": cause.payload.get("merchant_name"),
            "mcc": cause.payload.get("mcc"),
            "merchant_city": cause.payload.get("merchant_city"),
            "merchant_country": cause.payload.get("merchant_country"),
            "cause_event_id": cause.event_id,
            "reason": plan["kind"],
            "is_online": cause.payload.get("is_online"),
            "is_subscription": False,
        }

        _emit_money(
            state,
            plan["ts"],
            plan["kind"],
            account_id,
            int(plan["amount"]),
            "credit",
            f"merchant:{cause.payload.get('outlet_id')}"
            if cause.payload.get("outlet_id")
            else "external:merchant",
            body,
            INITIATOR_SYSTEM,
            correlation_id=cause.event_id,
            link_type=plan["kind"],
        )


def _replay_balances(state: ClientState, observed: list) -> None:
    """
    balance_after пересчитывается по ИТОГОВОМУ порядку ленты,
    уже после дефектов наблюдаемости.

    Внутри симуляции проводки применяются в порядке принятия
    решений, а дефекты вдобавок огрубляют время части записей.
    Наблюдаемый остаток обязан продолжать предыдущий остаток
    того же счёта, поэтому цепочка строится по тому порядку,
    в котором строки лягут в датасет.

    Повторная доставка и исправление несут ТОТ ЖЕ остаток, что
    и первая версия записи: новых денег они не создают.
    """

    balances = {
        account.account_id: account.opening_balance
        for account in state.ledger.accounts.values()
        if account.visible
    }

    applied: dict = {}

    for event in observed:

        payload = event.payload

        account_id = payload.get("account_id")

        if account_id is None or account_id not in balances:
            continue

        if payload.get("status") != "approved":
            continue

        # Повторная доставка и исправление получают ВСЁ, что
        # пересчёт записал в первую версию. Иначе версии одной
        # записи разошлись бы по полям, которых исправление
        # вообще не касалось.
        if event.event_id in applied:
            payload.update(applied[event.event_id])
            continue

        if event.event_type == "balance_snapshot":
            value = balances[account_id]
            written = {
                "balance_after": value,
                "amount": abs(value),
                "direction": "credit" if value >= 0 else "debit",
            }
            payload.update(written)
            applied[event.event_id] = written
            continue

        amount = int(payload.get("amount") or 0)

        signed = amount if payload.get("direction") == "credit" else -amount

        balances[account_id] += signed

        payload["balance_after"] = balances[account_id]

        applied[event.event_id] = {"balance_after": balances[account_id]}


def _row(event) -> dict:

    return {
        "event_id": event.event_id,
        "client_id": event.client_id,
        "event_type": event.event_type,
        "source": event.source,
        "event_time": event.event_time,
        "record_time": event.record_time,
        "effective_at": event.effective_at,
        "time_precision": event.time_precision,
        "sequence_number": event.sequence_number,
        "event_version": event.event_version,
        "change_initiator": event.change_initiator,
        "correlation_id": event.correlation_id,
        "link_type": event.link_type,
        "is_test_account": event.is_test_account,
        "payload": json.dumps(event.payload, ensure_ascii=False, separators=(",", ":"), default=str),
    }


def _truth_client(state: ClientState, household_id: str | None = None) -> dict:

    persona = state.persona

    row = {
        "client_id": persona.client_id,
        "client_ordinal": persona.client_ordinal,
        "community_id": persona.community_id,
        "archetype": f"{persona.life_stage}|{persona.hcb_role}|{persona.activity_mode}",
        "life_stage": persona.life_stage,
        "hcb_role": persona.hcb_role,
        "activity_mode": persona.activity_mode,
        "settlement": persona.settlement,
        "settlement_type": persona.settlement_type,
        "true_income": persona.true_income,
        "visible_share": persona.visible_share,
        "is_test_account": persona.is_test_account,
        "registered_in_window": persona.registered_in_window,
        "vanished_after_registration": persona.vanished_after_registration,
        "night_segment": persona.night_segment,
        "household_id": household_id,
        "final_state": state.state,
        "hidden_cash": state.ledger.balance(state.ledger.cash_id),
        "hidden_other_bank": state.ledger.balance(state.ledger.other_bank_id),
    }

    for name, value in persona.traits.base.items():
        row[f"trait_{name}"] = round(float(value), 4)

    final = state.traits.at(HISTORY_END - timedelta(days=1)) if state.traits else persona.traits.base

    for name, value in final.items():
        row[f"trait_final_{name}"] = round(float(value), 4)

    return row


def _truth_plan(state: ClientState) -> list:

    rows = []

    client_id = state.client_id

    for event in state.life_events:
        rows.append(
            {
                "client_id": client_id,
                "ts": event.ts,
                "kind": "life_event",
                "key": event.kind,
                "value": json.dumps(event.payload, ensure_ascii=False, default=str),
            }
        )

    for episode in state.stress_episodes:
        rows.append(
            {
                "client_id": client_id,
                "ts": episode.start,
                "kind": "stress_start",
                "key": episode.trigger,
                "value": json.dumps(
                    {
                        "intensity": round(episode.intensity, 3),
                        "end": episode.end.isoformat(),
                        "resolution": episode.resolution,
                    },
                    ensure_ascii=False,
                ),
            }
        )
        rows.append(
            {
                "client_id": client_id,
                "ts": episode.end,
                "kind": "stress_end",
                "key": episode.resolution,
                "value": json.dumps({"trigger": episode.trigger}, ensure_ascii=False),
            }
        )

    for pause in state.pauses:
        rows.append(
            {
                "client_id": client_id,
                "ts": pause.start,
                "kind": "pause_start",
                "key": pause.kind,
                "value": json.dumps(
                    {
                        "reason": pause.reason,
                        "planned_end": pause.planned_end.isoformat(),
                        "actual_end": pause.actual_end.isoformat(),
                        "return_trigger": pause.return_trigger,
                    },
                    ensure_ascii=False,
                ),
            }
        )

    for episode in state.fraud_episodes:
        rows.append(
            {
                "client_id": client_id,
                "ts": episode.start,
                "kind": "fraud_episode",
                "key": episode.kind,
                "value": json.dumps(
                    {
                        "detected": episode.detected,
                        "decision": episode.decision,
                        "response": episode.client_response,
                        "chargeback": episode.chargeback,
                    },
                    ensure_ascii=False,
                ),
            }
        )

    for shift in (state.traits.shifts if state.traits else ()):
        rows.append(
            {
                "client_id": client_id,
                "ts": shift.ts,
                "kind": "trait_shift",
                "key": shift.cause,
                "value": json.dumps(shift.deltas, ensure_ascii=False),
            }
        )

    return rows


_HANDLERS["bill_sweep"] = _sweep_bills


__all__ = ["finish", "month_end"]
