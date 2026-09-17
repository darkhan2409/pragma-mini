from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .. import params as params_module
from ..rng import NS_HOUSEHOLD, keyed_rng
from . import calendar as cal
from .income import monthly_income
from .persona import Persona
from .stress import level_at


# ============================================================
# БЮДЖЕТ ДОМОХОЗЯЙСТВА
# ============================================================
#
# Месячный бюджет это то, из чего рождаются потребности:
# обязательные расходы идут первыми, дискреционные зависят от
# того, что осталось, а стресс срезает именно дискреционную
# часть.
# ============================================================


@dataclass(frozen=True)
class MonthlyBudget:
    month: datetime
    income: int
    mandatory: int
    rent: int
    debt_service: int
    discretionary: int
    stress: float

    @property
    def free_share(self) -> float:
        if self.income <= 0:
            return 0.0
        return max(0.0, self.discretionary / self.income)


def budget_for_month(
    persona: Persona,
    month: datetime,
    streams: tuple,
    stress_episodes: tuple,
    debt_service: int = 0,
) -> MonthlyBudget:
    """
    Бюджет месяца: доход, обязательные расходы, обслуживание
    долга и дискреционный остаток.
    """

    settings = params_module.active()

    income = monthly_income(streams, month)

    rng = keyed_rng(NS_HOUSEHOLD, persona.client_ordinal, cal.month_index(month))

    stress = level_at(stress_episodes, month)

    mandatory_share = persona.mandatory_share * rng.uniform(0.92, 1.08)

    household_factor = 1.0 + 0.18 * max(0, persona.household_size - 1)

    mandatory = int(income * min(0.88, mandatory_share) * min(1.6, household_factor))

    rent = int(income * persona.rent_share) if persona.rent_share else 0

    discretionary = max(0, income - mandatory - rent - debt_service)

    # Стресс режет именно дискреционную часть.
    discretionary = int(discretionary * (1.0 - settings.stress.discretionary_cut * stress))

    return MonthlyBudget(
        month=cal.month_start(month),
        income=income,
        mandatory=mandatory,
        rent=rent,
        debt_service=debt_service,
        discretionary=discretionary,
        stress=stress,
    )


def spending_factor(budget: MonthlyBudget, persona: Persona) -> float:
    """
    Во сколько раз клиент тратит больше или меньше обычного
    в этом месяце.
    """

    settings = params_module.active().stress

    factor = 1.0

    factor *= 1.0 - settings.grocery_cut * budget.stress * 0.5

    if budget.income > 0:
        share = budget.discretionary / budget.income
        factor *= 0.65 + 0.9 * min(1.0, share / 0.35)

    # Импульсивность влияет на ЧИСЛО покупок (behaviour/needs),
    # а не на сумму. Иначе она разгоняет и то и другое, бюджет
    # месяца гасит обе стороны, и связь черты с поведением
    # пропадает.


    return float(max(0.2, min(2.2, factor)))


def purchase_budget(budget: MonthlyBudget) -> int:
    """
    Сколько денег этого месяца может пройти покупками по картам
    этого банка. Аренда и обслуживание долга туда не входят.
    """

    settings = params_module.active().amounts

    available = max(0, budget.income - budget.rent - budget.debt_service)

    return int(available * settings.card_share_of_spend)


def budget_pressure(budget: MonthlyBudget, spent: int, ts: datetime) -> float:
    """
    Во сколько раз клиент сбавляет траты, если деньги месяца
    кончаются раньше времени.

    Это не бухгалтерский лимит, а поведение: человек, потративший
    к середине месяца всё, до зарплаты покупает заметно меньше и
    заметно дешевле.
    """

    settings = params_module.active().amounts

    target = purchase_budget(budget)

    if target <= 0:
        return 1.0

    days = cal.month_end(ts).day

    elapsed = max(1, ts.day) / days

    expected = target * elapsed

    if expected <= 0 or spent <= expected:
        return 1.0

    ratio = spent / expected

    pressure = 1.0 / (1.0 + settings.budget_pressure_strength * (ratio - 1.0))

    return float(max(settings.budget_pressure_floor, min(1.0, pressure)))


def funds_pressure(capacity: int, budget: MonthlyBudget) -> float:
    """
    Пустой счёт виден клиенту раньше, чем банку: человек, у
    которого не осталось денег, не ходит по магазинам и ждёт
    зарплату, а не собирает отказы.
    """

    settings = params_module.active().amounts

    if budget.income <= 0:
        return 1.0

    comfortable = budget.income * settings.comfortable_balance_share

    if comfortable <= 0:
        return 1.0

    return float(max(settings.empty_wallet_floor, min(1.0, capacity / comfortable)))


__all__ = [
    "MonthlyBudget",
    "funds_pressure",
    "budget_for_month",
    "budget_pressure",
    "purchase_budget",
    "spending_factor",
]
