from __future__ import annotations

from datetime import datetime

from ..life import calendar as cal
from .entities import DepositState


# ============================================================
# ДЕПОЗИТЫ
# ============================================================
#
# Депозит существует ДО своих операций: пополнение и снятие
# возможны только между открытием и закрытием, и только если
# версия продукта их разрешает.
#
# Остаток равен сумме пополнений минус снятия плюс начисленные
# проценты по ставке версии ДОГОВОРА, а не текущей витрины.
# ============================================================


def open_deposit(
    contract_id: str,
    account_id: str,
    amount: int,
    rate: float,
    opened_at: datetime,
    term_months: int,
    topup: bool,
    withdrawal: bool,
    capitalisation: str = "daily",
) -> DepositState:

    return DepositState(
        contract_id=contract_id,
        account_id=account_id,
        principal=int(amount),
        rate=float(rate),
        opened_at=opened_at,
        matures_at=cal.add_months(opened_at.replace(hour=0, minute=0, second=0, microsecond=0), term_months),
        topup_allowed=bool(topup),
        withdrawal_allowed=bool(withdrawal),
        capitalisation=capitalisation,
    )


def monthly_interest(state: DepositState, ledger, month: datetime) -> int:
    """
    Проценты за месяц по ФАКТИЧЕСКОМУ остатку каждого дня.

    Считать по остатку на конец месяца нельзя: вклад, открытый
    тридцать первого числа, получал бы столько же, сколько
    пролежавший весь месяц, а пополнение под конец месяца давало
    бы доход, которого не было.

    Вклад, открытый в середине месяца, зарабатывает только за
    прожитые дни: до открытия остаток по счёту был нулевым.
    """

    if state.closed:
        return 0

    account = ledger.get(state.account_id)

    if account is None:
        return 0

    start = cal.month_start(month)
    stop = cal.next_month(month)

    moves = sorted(ledger.signed_moves(state.account_id, start, stop))

    # Остаток на начало месяца: текущий минус движения месяца.
    opening = account.balance - sum(delta for _, delta in moves)

    daily = state.rate / 365.0

    accrued = 0.0
    balance = opening
    moment = start

    for ts, delta in list(moves) + [(stop, 0)]:

        edge = min(ts, stop)

        # Проценты считаются по КАЛЕНДАРНЫМ дням: вклад, открытый
        # тридцать первого в десять утра, зарабатывает за этот
        # день, а не округляется до нуля.
        days = (edge.date() - moment.date()).days

        if days > 0 and balance > 0:
            if state.capitalisation == "daily":
                segment = balance * ((1.0 + daily) ** days - 1.0)
                # Капитализация: процент отрезка входит в базу
                # следующего отрезка того же месяца. Иначе вклад с
                # пополнением в середине месяца недополучал бы
                # процент на уже начисленный процент.
                balance += segment
            else:
                segment = balance * state.rate * days / 365.0

            accrued += segment

        moment = edge
        balance += delta

    return int(round(accrued))


def matured(state: DepositState, ts: datetime) -> bool:
    return not state.closed and ts >= state.matures_at


def early_penalty(state: DepositState, ts: datetime, accrued: int) -> int:
    """
    Досрочное закрытие: проценты пересчитываются по ставке до
    востребования, а она равна нулю — всё начисленное
    возвращается банку. Больше остатка счёта списать нельзя, но
    остаток видит только вызывающий: это ограничение ставит он.
    """

    if ts >= state.matures_at:
        return 0

    return int(accrued)


def can_topup(state: DepositState, ts: datetime) -> bool:
    return state.topup_allowed and not state.closed and state.opened_at <= ts < state.matures_at


def can_withdraw(state: DepositState, ts: datetime, amount: int, balance: int) -> bool:
    if state.closed or not state.withdrawal_allowed:
        return False
    if ts < state.opened_at:
        return False
    return balance >= amount


__all__ = [
    "can_topup",
    "can_withdraw",
    "early_penalty",
    "matured",
    "monthly_interest",
    "open_deposit",
]
