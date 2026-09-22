from __future__ import annotations

from datetime import datetime, timedelta

from . import params as params_module

from .engine import _HANDLERS, _emit_money
from .finance import deposits as deposit_rules
from .finance import loans as loan_rules
from .finance.ledger import NON_PAYMENT_KINDS
from .finance.entities import CONTRACT_CLOSED
from .life import stress as stress_module
from .rng import NS_LOAN, NS_REPAY, keyed_rng, stable_hash
from .simulate import ClientState
from .world.dictionaries import MCC_CASH, MCC_SALARY, MCC_TRANSFER


# ============================================================
# ОБСЛУЖИВАНИЕ КРЕДИТА
# ============================================================
#
#   график -> срок платежа -> автосписание или пополнение
#   -> пропуск -> вехи DPD -> восстановление или
#   реструктуризация -> закрытие
#
# Остаток основного долга убывает ровно на основную часть
# платежа, а просрочка следует из нарушенного графика, а не
# из отдельного розыгрыша.
# ============================================================


def loan_payload(contract_id: str, loan, **extra) -> dict:

    body = {
        "contract_id": contract_id,
        "installment_no": None,
        "amount_due": None,
        "amount_paid": None,
        "principal_outstanding": loan.principal_outstanding,
        # Текущее состояние договора, а не то, что было при
        # последнем обходе месяца.
        "days_past_due": loan.dpd,
        "due_date": None,
        "cause_event_id": None,
        "reason": None,
    }

    body.update(extra)

    return body


def payment_plan(state: ClientState, loan, item) -> dict:
    """
    Намерение клиента по конкретному взносу.

    Решение принимается один раз на взнос и не зависит от
    порядка исполнения дня: заплатит ли клиент, за сколько
    дней до срока и в котором часу.
    """

    settings = params_module.active().products

    rng = keyed_rng(
        NS_REPAY,
        state.ordinal,
        stable_hash(loan.contract_id) % 9973,
        int(item.number),
    )

    discipline = state.persona.trait("financial_discipline", item.due_date)

    low, high = settings.discipline_bands

    band = "high" if discipline > high else "mid" if discipline > low else "low"

    stress = stress_module.level_at(state.stress_episodes, item.due_date)

    probability = settings.on_time_payment_probability[band]
    probability *= max(0.05, 1.0 - params_module.active().stress.missed_payment_boost * stress)

    will_pay = rng.random() < probability

    # Платят в день срока или в льготные дни. Раньше срока
    # взнос ещё не выставлен, а деньги к нему подтягиваются
    # отдельным пополнением внутри самого платежа.
    offset = int(rng.integers(0, settings.grace_days_before_missed + 1))

    return {
        "will_pay": will_pay,
        "offset": 0 if loan.autopay else offset,
        "hour": int(rng.integers(9, 21)),
        "minute": int(rng.integers(0, 60)),
        "band": band,
    }


def _on_loan_payment_intent(sim, state: ClientState, ts: datetime, payload: dict) -> None:
    """
    Клиент сам платит по графику: до срока, в срок или в
    льготные дни.
    """

    loan = state.loans.get(payload["contract_id"])

    if loan is None or loan.closed or loan.autopay:
        return

    item = loan.oldest_unpaid()

    if item is None or item.outstanding <= 0:
        return

    # Платят то, что должны на сегодня: текущий взнос вместе с
    # накопившимся долгом. Плановый платёж сессии приложения не
    # принадлежит, поэтому и канал у него не app.
    repay_loan(state, ts, loan, loan_rules.arrears_amount(loan), "ecom")


def _on_installment_due(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    contract_id = payload["contract_id"]

    loan = state.loans.get(contract_id)

    if loan is None or loan.closed:
        return

    item = payload["installment"]

    event = state.emit(
        state.factory.make(
            "installment_due",
            ts,
            loan_payload(
                contract_id,
                loan,
                installment_no=item.number,
                amount_due=item.amount,
                due_date=item.due_date.date().isoformat(),
                reason="schedule",
            ),
        )
    )

    loan_rules.register_due(loan, item, event.event_id)

    if loan.autopay:
        owed = max(int(item.amount), loan_rules.arrears_amount(loan))
        repay_loan(state, ts + timedelta(minutes=5), loan, owed, "system")


def _payment_sources(state: ClientState, ts: datetime, amount: int) -> list:
    """
    Счета, которыми платят кредит. Кредитной картой кредит не
    гасят, если это не разрешено параметром.
    """

    sources = state.ledger.payment_sources(ts, amount)

    if params_module.active().products.loan_payment_from_credit_card:
        return sources

    return [item for item in sources if item.kind != "credit_card"]


def _payment_capacity(state: ClientState, ts: datetime) -> int:

    allow_credit = params_module.active().products.loan_payment_from_credit_card

    values = [
        state.ledger.available_at(account.account_id, ts)
        for account in state.ledger.accounts.values()
        if account.visible
        and account.is_open_at(ts)
        and account.kind not in NON_PAYMENT_KINDS
        and (allow_credit or account.kind != "credit_card")
    ]

    return int(max(values)) if values else 0


def _withdraw_from_deposit(state: ClientState, ts: datetime, target, amount: int) -> bool:
    """
    Снятие со вклада на карту к сроку платежа.

    Вклад не платёжный счёт: списать с него взнос или покупку
    нельзя. Деньги выводятся отдельной операцией, и только если
    условия продукта разрешают снятие (deposits.can_withdraw).
    """

    candidates = [
        item
        for item in state.deposits.values()
        if deposit_rules.can_withdraw(item, ts, amount, state.ledger.balance(item.account_id))
    ]

    if not candidates:
        return False

    deposit = max(candidates, key=lambda item: state.ledger.balance(item.account_id))

    from .engine_app import _own_transfer

    moved = _own_transfer(
        state, ts, "deposit_withdrawal", deposit.account_id, target.account_id,
        amount, deposit.contract_id, "withdrawal_before_payment",
    )

    if not moved:
        return False

    deposit.principal = max(0, deposit.principal - amount)

    return True


def _topup_before_payment(state: ClientState, ts: datetime, amount: int, rng) -> bool:
    """
    Клиент переводит деньги из другого банка к сроку платежа.
    """

    settings = params_module.active().products

    if rng.random() >= settings.loan_topup_from_other_bank_share:
        return False

    account = state.primary_card_account(ts)

    if account is None:
        return False

    shortfall = max(0, amount - state.ledger.available_at(account.account_id, ts))

    if shortfall <= 0:
        return False

    # Деньги берут откуда есть: со счёта в другом банке или
    # наличными через банкомат.
    sources = state.ledger.hidden_sources(shortfall)

    if not sources:
        # Наличных и другого банка не хватило: остаётся вклад, если
        # его условия разрешают снятие. Это отдельная операция с
        # проверкой условий, а не платёж со вклада напрямую.
        return _withdraw_from_deposit(state, ts - timedelta(minutes=12), account, shortfall)

    hidden = sources[0]

    from_cash = hidden.account_id == state.ledger.cash_id

    _emit_money(
        state,
        ts - timedelta(minutes=12),
        "cash_deposit" if from_cash else "transfer_in",
        account.account_id,
        shortfall,
        "credit",
        hidden.account_id,
        {
            # Наличные вносит банкомат, перевод приходит из
            # другого банка. Сессии приложения тут нет ни в том,
            # ни в другом случае.
            "channel": "atm" if from_cash else "system",
            "counterparty": "Own account",
            "reason": "topup_before_installment",
            "mcc": MCC_CASH if from_cash else MCC_TRANSFER,
            "merchant_country": "KZ",
        },
    )

    return True


def _decline_payment(state: ClientState, ts: datetime, loan, item, amount: int) -> None:
    """
    Неудачное автосписание: банк попробовал и не смог.
    """

    _emit_money(
        state,
        ts,
        "loan_payment",
        None,
        max(1, int(amount)),
        "debit",
        "external:none",
        {
            "channel": "system",
            "contract_id": loan.contract_id,
            "cause_event_id": item.due_event_id,
            "reason": "installment",
            "decline_reason": "insufficient_funds",
            "mcc": MCC_SALARY,
            "merchant_country": "KZ",
        },
        status="declined",
    )


def repay_loan(
    state: ClientState,
    ts: datetime,
    loan,
    amount: int,
    channel: str,
    session_id: str | None = None,
) -> None:
    """
    Платёж по кредиту: сначала деньги, затем отметка в графике.

    session_id называет сессию приложения, если платёж сделан в
    ней. Без него канал app у денежной строки был бы ложью: в
    выгрузке появлялась операция «в приложении», которую нельзя
    связать ни с одной сессией.
    """

    if amount <= 0 or loan.closed:
        return

    item = loan.oldest_unpaid()

    if item is None:
        return

    settings = params_module.active().products

    sources = _payment_sources(state, ts, amount)

    if not sources:

        # Денег на счёте не хватает. Клиент сначала пробует
        # подтянуть их из другого банка, потом платит сколько
        # может, и только затем попытка проваливается.
        rng = keyed_rng(
            NS_REPAY,
            state.ordinal,
            stable_hash(loan.contract_id) % 9973,
            int(item.number),
            1,
        )

        if _topup_before_payment(state, ts, amount, rng):
            sources = _payment_sources(state, ts, amount)

    if not sources:

        capacity = _payment_capacity(state, ts)

        if capacity >= settings.partial_payment_min_share * amount and capacity > 0:
            amount = int(capacity)
            sources = _payment_sources(state, ts, amount)

    if not sources:

        if channel == "system":
            _decline_payment(state, ts, loan, item, amount)

        return

    account = sources[0]

    # Сколько на самом деле можно списать в этот момент. Ниже
    # apply_payment уже проставляет отметки в графике, и отменить
    # их за отказом было бы нечем — поэтому сумма ограничивается
    # ЗДЕСЬ, до первой отметки, а не проверяется после списания.
    amount = min(int(amount), state.ledger.available_at(account.account_id, ts))

    if amount <= 0:
        return

    # Один платёж закрывает столько взносов, на сколько хватает
    # денег. Клиент, отставший на месяц, догоняет график, а не
    # остаётся в вечной просрочке из-за того, что платёж всегда
    # уходит только в самый старый взнос.
    covered: list = []

    remaining = int(amount)

    while remaining > 0:

        target = loan.oldest_unpaid()

        if target is None:
            break

        paid = loan_rules.apply_payment(loan, target, remaining, ts)

        if paid <= 0:
            break

        covered.append((target, paid))
        remaining -= paid

    total = sum(value for _, value in covered)

    if total <= 0:
        return

    _emit_money(
        state,
        ts,
        "loan_payment",
        account.account_id,
        total,
        "debit",
        f"loan:{loan.contract_id}",
        {
            "channel": channel,
            "session_id": session_id,
            "contract_id": loan.contract_id,
            "cause_event_id": covered[0][0].due_event_id,
            "reason": "installment",
            "mcc": MCC_SALARY,
            "merchant_country": "KZ",
        },
    )

    for position, (target, paid) in enumerate(covered):
        state.emit(
            state.factory.make(
                "installment_paid",
                ts + timedelta(seconds=2 + position),
                loan_payload(
                    loan.contract_id,
                    loan,
                    installment_no=target.number,
                    amount_due=target.amount,
                    amount_paid=paid,
                    due_date=target.due_date.date().isoformat(),
                    cause_event_id=target.due_event_id,
                    reason="payment",
                ),
            )
        )

    if loan.dpd > 0 and loan_rules.arrears_amount(loan) == 0:
        _clear_arrears(state, ts, loan)


def _clear_arrears(state: ClientState, ts: datetime, loan) -> None:

    state.emit(
        state.factory.make(
            "arrears_cleared",
            ts + timedelta(seconds=5),
            loan_payload(loan.contract_id, loan, days_past_due=0, reason="arrears_cleared"),
        )
    )

    loan.dpd = 0
    loan.delinquency_marks = ()


def _on_loan_check(sim, state: ClientState, ts: datetime, payload: dict) -> None:

    settings = params_module.active().products

    contract_id = payload["contract_id"]

    loan = state.loans.get(contract_id)

    if loan is None or loan.closed:
        return

    stress = stress_module.level_at(state.stress_episodes, ts)

    discipline = state.persona.trait("financial_discipline", ts)

    rng = keyed_rng(NS_LOAN, state.ordinal, ts.toordinal(), stable_hash(contract_id) % 9973)

    # Автосписание повторяет попытку, пока идут льготные дни:
    # деньги могли прийти на счёт на день позже срока.
    if loan.autopay:

        pending = loan.oldest_unpaid()

        if pending is not None and pending.status == "due" and pending.outstanding > 0:

            elapsed = (ts - pending.due_date).days

            if 0 < elapsed <= settings.autopay_retry_days:
                repay_loan(state, ts + timedelta(seconds=15), loan, pending.outstanding, "system")

    arrears = loan_rules.arrears_amount(loan)

    if arrears > 0:

        low_band, high_band = settings.discipline_bands

        band = "high" if discipline > high_band else "mid" if discipline > low_band else "low"

        cure = settings.cure_probability_per_day[band] * (1.0 - 0.6 * stress)

        if rng.random() < cure:
            repay_loan(state, ts + timedelta(seconds=30), loan, arrears, "ecom")

    for item in loan.schedule:

        # Частично оплаченный взнос это тоже нарушение графика.
        # Пропустить его здесь значило бы зарегистрировать
        # просрочку без события о пропуске.
        if item.status not in ("due", "partially_paid"):
            continue

        if item.outstanding <= 0:
            continue

        if (ts - item.due_date).days <= settings.grace_days_before_missed:
            continue

        loan_rules.mark_missed(loan, item)

        state.emit(
            state.factory.make(
                "installment_missed",
                ts + timedelta(seconds=60),
                loan_payload(
                    contract_id,
                    loan,
                    installment_no=item.number,
                    amount_due=item.amount,
                    amount_paid=item.paid_amount or None,
                    days_past_due=(ts - item.due_date).days,
                    due_date=item.due_date.date().isoformat(),
                    cause_event_id=item.due_event_id,
                    reason="missed",
                ),
            )
        )

    dpd = loan_rules.days_past_due(loan, ts)

    loan.dpd = dpd

    milestone = loan_rules.milestone_reached(loan, dpd)

    if milestone is not None:

        loan.delinquency_marks = loan.delinquency_marks + (milestone,)

        state.emit(
            state.factory.make(
                "delinquency_registered",
                ts + timedelta(seconds=90),
                loan_payload(
                    contract_id,
                    loan,
                    amount_due=loan_rules.arrears_amount(loan),
                    days_past_due=milestone,
                    reason=f"dpd_{milestone}",
                ),
            )
        )

        if milestone >= 60 and rng.random() < settings.restructure_share_at_dpd60:

            loan_rules.restructure(loan, ts, extra_months=int(rng.integers(3, 12)))

            state.emit(
                state.factory.make(
                    "loan_restructured",
                    ts + timedelta(seconds=120),
                    loan_payload(contract_id, loan, days_past_due=0, reason="restructured"),
                )
            )

    arrears = loan_rules.arrears_amount(loan)

    if arrears == 0 and loan.principal_outstanding > 0:

        chance = settings.early_repayment_share_per_year / 365.0
        chance *= 0.4 + 1.6 * discipline
        chance *= max(0.1, 1.0 - stress)

        if rng.random() < chance and state.ledger.payment_sources(ts, loan_rules.payoff_amount(loan)):
            close_loan(state, ts, loan, early=True)
            return

    if loan.principal_outstanding <= 0 and arrears == 0:
        close_loan(state, ts, loan, early=False)


def close_loan(state: ClientState, ts: datetime, loan, early: bool, reason: str | None = None) -> None:

    contract = state.contracts.get(loan.contract_id)

    if early and loan.principal_outstanding > 0:

        payoff = loan_rules.payoff_amount(loan)

        sources = state.ledger.payment_sources(ts, payoff)

        if not sources:
            return

        paid = _emit_money(
            state,
            ts + timedelta(seconds=150),
            "loan_payment",
            sources[0].account_id,
            payoff,
            "debit",
            f"loan:{loan.contract_id}",
            {
                # Досрочное погашение вне сессии приложения.
                "channel": "ecom",
                "contract_id": loan.contract_id,
                "reason": "early_repayment",
                "merchant_country": "KZ",
            },
        )

        # Денег не списали — кредит не погашен.
        if paid.payload.get("status") != "approved":
            return

        loan.principal_outstanding = 0

        for item in loan.schedule:
            if item.status in ("scheduled", "due", "partially_paid", "missed"):
                item.status = "paid"
                item.paid_amount = item.amount
                item.principal_paid = item.principal

        state.emit(
            state.factory.make(
                "early_repayment",
                ts + timedelta(seconds=180),
                loan_payload(
                    loan.contract_id, loan, amount_due=payoff, amount_paid=payoff,
                    days_past_due=0, reason="early_repayment",
                ),
            )
        )

    loan.closed = True

    state.emit(
        state.factory.make(
            "loan_closed",
            ts + timedelta(seconds=210),
            loan_payload(loan.contract_id, loan, days_past_due=0,
                         reason=reason or ("early" if early else "scheduled")),
        )
    )

    if contract is not None:
        contract.status = CONTRACT_CLOSED
        contract.closed_at = ts
        emit_product_closed(state, ts, contract, reason or "loan_closed")


def emit_product_closed(state: ClientState, ts: datetime, contract, reason: str) -> None:

    state.emit(
        state.factory.make(
            "product_closed",
            ts + timedelta(seconds=240),
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
                "reason": reason,
            },
        )
    )


_HANDLERS["installment_due"] = _on_installment_due
_HANDLERS["loan_payment_intent"] = _on_loan_payment_intent
_HANDLERS["loan_check"] = _on_loan_check


__all__ = ["close_loan", "emit_product_closed", "loan_payload", "payment_plan", "repay_loan"]
