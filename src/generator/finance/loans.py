from __future__ import annotations

import math
from datetime import datetime

from .. import params as params_module
from ..life import calendar as cal
from .entities import Installment, LoanState


# ============================================================
# КРЕДИТНЫЙ ЦИКЛ
# ============================================================
#
# Выдача -> график -> ежемесячный платёж -> пропуск -> вехи
# просрочки -> восстановление или реструктуризация -> закрытие.
#
# График соответствует договору: аннуитет считается один раз по
# ставке и сроку ВЕРСИИ, зафиксированной на дате подписания.
# Остаток основного долга убывает ровно на основную часть
# платежа.
# ============================================================


def annuity_payment(principal: int, annual_rate: float, months: int) -> int:

    if months <= 0:
        return principal

    monthly = annual_rate / 12.0

    if monthly <= 0.0:
        return int(math.ceil(principal / months))

    factor = (1.0 + monthly) ** months

    return int(math.ceil(principal * monthly * factor / (factor - 1.0)))


def build_schedule(
    principal: int,
    annual_rate: float,
    months: int,
    first_due: datetime,
) -> list:
    """
    Аннуитетный график. Последний платёж добирает округления,
    чтобы сумма основных частей равнялась телу кредита.
    """

    payment = annuity_payment(principal, annual_rate, months)

    monthly = annual_rate / 12.0

    schedule: list[Installment] = []

    outstanding = principal

    for number in range(1, months + 1):

        interest = int(round(outstanding * monthly))

        principal_part = payment - interest

        if number == months or principal_part >= outstanding:
            principal_part = outstanding
            amount = principal_part + interest
        else:
            amount = payment

        outstanding -= principal_part

        schedule.append(
            Installment(
                number=number,
                due_date=cal.add_months(first_due, number - 1),
                amount=int(amount),
                principal=int(principal_part),
                interest=int(interest),
            )
        )

        if outstanding <= 0:
            break

    return schedule


def open_loan(
    contract_id: str,
    principal: int,
    annual_rate: float,
    months: int,
    disbursed_at: datetime,
    autopay: bool,
) -> LoanState:

    first_due = cal.add_months(disbursed_at.replace(hour=0, minute=0, second=0, microsecond=0), 1)

    schedule = build_schedule(principal, annual_rate, months, first_due)

    return LoanState(
        contract_id=contract_id,
        principal_outstanding=principal,
        schedule=schedule,
        rate=annual_rate,
        autopay=autopay,
    )


def due_today(state: LoanState, day: datetime) -> Installment | None:

    for item in state.schedule:
        if item.status == "scheduled" and item.due_date.date() == day.date():
            return item

    return None


def register_due(state: LoanState, item: Installment, event_id: str) -> None:
    item.status = "due"
    item.due_event_id = event_id


def apply_payment(state: LoanState, item: Installment, amount: int, ts: datetime) -> int:
    """
    Гасит платёж целиком или частично. Возвращает фактически
    зачисленную сумму.
    """

    payable = min(amount, item.outstanding)

    if payable <= 0:
        return 0

    item.paid_amount += payable
    item.paid_at = ts

    left = max(0, item.principal - item.principal_paid)

    if item.outstanding == 0:
        # Последний платёж по взносу добирает остаток тела
        # целиком: округление долей не оставляет хвоста.
        principal_part = left
    else:
        share = payable / max(1, item.amount)
        principal_part = min(left, int(round(item.principal * share)))

    item.principal_paid += principal_part

    state.principal_outstanding = max(0, state.principal_outstanding - principal_part)

    item.status = "paid" if item.outstanding == 0 else "partially_paid"

    # Просрочка пересчитывается ПРЯМО ЗДЕСЬ, а не при следующем
    # обходе месяца. Иначе событие платежа уносило бы вчерашнее
    # число дней, и соседние записи противоречили бы друг другу:
    # платёж «погасил» долг, а days_past_due в нём остался
    # прежним.
    state.dpd = days_past_due(state, ts)

    return payable


def mark_missed(state: LoanState, item: Installment) -> None:
    if item.status in ("due", "partially_paid"):
        item.status = "missed"


def arrears_amount(state: LoanState) -> int:
    return sum(
        item.outstanding
        for item in state.schedule
        if item.status in ("due", "partially_paid", "missed")
    )


def days_past_due(state: LoanState, day: datetime) -> int:
    """
    Дней ТЕКУЩЕЙ просрочки договора на указанный момент.

    Один смысл на весь генератор: сколько дней прошло с самого
    раннего платежа, который до сих пор не закрыт. Полностью
    погашенный график даёт ноль, частично погашенный — считает от
    того же самого взноса, потому что он всё ещё не закрыт.

    Это НЕ возраст конкретного взноса и НЕ веха просрочки, на
    которой банк зарегистрировал событие: и то, и другое —
    отдельные величины, и подменять ими текущее состояние
    договора нельзя.
    """

    settings = params_module.active().products

    oldest = None

    for item in state.schedule:

        if item.outstanding <= 0:
            continue

        if item.status == "missed":
            oldest = item
            break

        # Взнос со сроком и частично оплаченный взнос живут по
        # одному правилу: пока идут льготные дни, просрочки нет.
        # Иначе веха просрочки регистрировалась бы раньше, чем
        # событие о пропуске.
        if item.status in ("due", "partially_paid"):
            if (day - item.due_date).days > settings.grace_days_before_missed:
                oldest = item
                break

    if oldest is None:
        return 0

    return max(0, (day - oldest.due_date).days)


def milestone_reached(state: LoanState, dpd: int) -> int | None:
    """
    Веха DPD, которую надо зарегистрировать впервые.
    """

    settings = params_module.active().products

    for milestone in settings.dpd_milestones:
        if dpd >= milestone and milestone not in state.delinquency_marks:
            return milestone

    return None


def is_closed(state: LoanState) -> bool:
    return state.principal_outstanding <= 0 and all(
        item.status in ("paid", "scheduled") for item in state.schedule
    ) and all(item.status == "paid" for item in state.schedule)


def payoff_amount(state: LoanState) -> int:
    """
    Сколько нужно, чтобы закрыть кредит досрочно.
    """

    return int(state.principal_outstanding + arrears_amount(state))


def monthly_payment(state: LoanState) -> int:
    """
    Регулярный платёж по договору.

    Берётся из графика, а не из ближайшего неоплаченного:
    у кредита, где просрочены все оставшиеся платежи,
    обязательство никуда не делось.
    """

    for item in state.schedule:
        if item.status != "paid":
            return int(item.amount)

    return int(state.schedule[-1].amount) if state.schedule else 0


def debt_service(states: tuple, ts: datetime) -> int:
    """
    Месячная нагрузка по всем кредитам клиента.
    """

    total = 0

    for state in states:
        if state.closed or state.principal_outstanding <= 0:
            continue
        total += monthly_payment(state)

    return int(total)


def max_amount_for_dsr(
    income: int,
    existing_service: int,
    annual_rate: float,
    months: int,
    max_ratio: float,
) -> int:
    """
    Какую сумму банк готов выдать, чтобы платёж вместе с уже
    имеющимися обязательствами уложился в долговую нагрузку.

    Обратная функция к аннуитету: сколько тела соответствует
    платежу, который клиент ещё может себе позволить.
    """

    capacity = int(max_ratio * income) - int(existing_service)

    if capacity <= 0 or months <= 0:
        return 0

    monthly = annual_rate / 12.0

    if monthly <= 0.0:
        return int(capacity * months)

    factor = (1.0 + monthly) ** months

    return int(capacity * (factor - 1.0) / (monthly * factor))


def restructure(state: LoanState, ts: datetime, extra_months: int) -> None:
    """
    Реструктуризация: срок растёт, платёж падает, просрочка
    переносится в тело долга.
    """

    outstanding = state.principal_outstanding + arrears_amount(state)

    remaining = [item for item in state.schedule if item.status in ("scheduled",)]

    months = max(6, len(remaining) + extra_months)

    first_due = cal.add_months(ts.replace(hour=0, minute=0, second=0, microsecond=0), 1)

    kept = [item for item in state.schedule if item.status == "paid"]

    fresh = build_schedule(outstanding, state.rate, months, first_due)

    for offset, item in enumerate(fresh):
        item.number = len(kept) + offset + 1

    state.schedule = kept + fresh
    state.principal_outstanding = outstanding
    state.arrears = 0
    state.dpd = 0
    state.delinquency_marks = ()
    state.restructured = True


__all__ = [
    "max_amount_for_dsr",
    "monthly_payment",
    "annuity_payment",
    "apply_payment",
    "arrears_amount",
    "build_schedule",
    "days_past_due",
    "debt_service",
    "due_today",
    "is_closed",
    "mark_missed",
    "milestone_reached",
    "open_loan",
    "payoff_amount",
    "register_due",
    "restructure",
]
