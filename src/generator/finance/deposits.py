from __future__ import annotations

from datetime import datetime, timedelta

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


def monthly_interest(state: DepositState, balance: int, month: datetime) -> int:
    """
    Проценты за месяц. Ежедневная капитализация приведена к
    месячному начислению: банк выплачивает его одной проводкой.
    """

    if balance <= 0 or state.closed:
        return 0

    days = (cal.next_month(month) - cal.month_start(month)).days

    daily = state.rate / 365.0

    if state.capitalisation == "daily":
        amount = balance * ((1.0 + daily) ** days - 1.0)
    else:
        amount = balance * state.rate / 12.0

    return int(round(amount))


def matured(state: DepositState, ts: datetime) -> bool:
    return not state.closed and ts >= state.matures_at


def early_penalty(state: DepositState, ts: datetime, accrued: int) -> int:
    """
    Досрочное закрытие: проценты пересчитываются, излишек
    возвращается банку.
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
