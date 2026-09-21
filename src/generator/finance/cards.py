from __future__ import annotations

from datetime import datetime, timedelta

from .. import params as params_module
from .entities import CARD_ACTIVE, CARD_BLOCKED, CARD_CLOSED, Card, CardCreditState


# ============================================================
# КАРТЫ
# ============================================================
#
# Заблокированная карта НЕ проводит успешных покупок. Блокировка
# бывает клиентской, банковской и антифродовой, снимается
# клиентом, банком или сама по истечении срока, а перевыпуск
# создаёт НОВУЮ карту со ссылкой на прежнюю.
#
# Кешбэк и комиссии считаются по тарифу ВЕРСИИ договора.
# ============================================================


def block(card: Card, ts: datetime, reason: str, days: int | None = None,
          permanent: bool = False) -> None:
    """
    Временная заморозка имеет срок и снимается. Утраченная или
    скомпрометированная карта блокируется НАВСЕГДА: срока у неё
    нет, таймер её не разморозит, обслуживание возвращает только
    перевыпуск.
    """

    settings = params_module.active().products

    until = (
        None
        if permanent
        else ts + timedelta(days=days if days is not None else settings.card_block_max_days)
    )

    card.status = CARD_BLOCKED
    card.blocked_at = ts
    card.block_reason = reason
    card.blocked_until = until
    card.blocks.append((ts, until))


def unblock(card: Card, ts: datetime) -> None:

    for index, (started, ended) in enumerate(card.blocks):
        if started <= ts and (ended is None or ts < ended):
            card.blocks[index] = (started, ts)

    card.status = CARD_ACTIVE
    card.blocked_at = None
    card.blocked_until = None
    card.block_reason = None


def close(card: Card, ts: datetime) -> None:
    card.status = CARD_CLOSED
    card.closed_at = ts


def cashback_amount(terms: dict, category: str, amount: int, balance: int, deposits: int, months_since_open: int) -> int:
    """
    Кешбэк покупки по тарифу версии договора.
    """

    if not terms:
        return 0

    rate = float(terms.get("cashback_base", 0.0))

    intro = terms.get("cashback_intro")

    if intro is not None and months_since_open * 30 <= int(terms.get("cashback_intro_days", 0)):
        if category in tuple(terms.get("cashback_categories", ())):
            rate = float(intro)
    elif terms.get("cashback_after") is not None and category in tuple(terms.get("cashback_categories", ())):
        rate = float(terms["cashback_after"])

    if category in ("taxi", "utilities") and terms.get("cashback_utility_taxi"):
        rate = max(rate, float(terms["cashback_utility_taxi"]))

    if category in ("taxi", "travel", "hotel", "airline") and terms.get("cashback_travel"):
        rate = max(rate, float(terms["cashback_travel"]))

    if category == "pharmacy" and terms.get("cashback_pharmacy_first_year"):
        rate = max(
            rate,
            float(
                terms["cashback_pharmacy_first_year"]
                if months_since_open < 12
                else terms.get("cashback_pharmacy_after", rate)
            ),
        )

    bonus_balance = terms.get("cashback_balance_bonus")

    if bonus_balance and balance >= int(terms.get("balance_threshold", 0)):
        rate += float(bonus_balance)

    bonus_deposit = terms.get("cashback_deposit_bonus")

    if bonus_deposit and deposits >= int(terms.get("deposit_threshold", 0)):
        rate += float(bonus_deposit)

    value = int(round(amount * rate))

    cap_per_transaction = terms.get("cashback_cap_per_transaction")

    if cap_per_transaction:
        value = min(value, int(cap_per_transaction))

    return max(0, value)


def cashback_cap(terms: dict) -> int:
    return int(terms.get("cashback_cap", 0) or 0)


def withdrawal_fee(terms: dict, amount: int, used_this_month: int, count_this_month: int) -> int:
    """
    Комиссия за снятие по тарифу версии договора.
    """

    if not terms:
        return 0

    free_count = terms.get("atm_free_count_monthly")

    if free_count is not None:
        if count_this_month < int(free_count):
            return 0
    else:
        free_amount = int(terms.get("atm_free_monthly", 0) or 0)
        if used_this_month + amount <= free_amount:
            return 0

    rate = float(terms.get("atm_fee_rate", 0.0) or 0.0)

    if rate <= 0.0:
        return 0

    return max(int(terms.get("atm_fee_min", 0) or 0), int(round(amount * rate)))


def transfer_fee(terms: dict, amount: int, used_this_month: int) -> int:

    if not terms:
        return 0

    free_amount = int(terms.get("transfer_free_monthly", 0) or 0)

    if used_this_month + amount <= free_amount:
        return 0

    return int(terms.get("transfer_fee", 0) or 0)


def monthly_fee(terms: dict) -> int:
    return int(terms.get("fee_monthly", 0) or 0)


def open_credit(contract, terms: dict) -> "CardCreditState":
    """
    Долговое состояние карты рассрочки по условиям договора.
    """

    settings = params_module.active().products

    months = int(terms.get("installment_months") or settings.card_installment_months_default)

    return CardCreditState(
        contract_id=contract.contract_id,
        account_id=contract.account_id,
        installment_months=max(1, months),
        purchase_rate=float(terms.get("purchase_rate") or 0.0),
        cash_rate=float(terms.get("cash_rate_nominal") or 0.0),
    )


def add_purchase(state: "CardCreditState", amount: int, month_index: int, cause_event_id: str) -> None:
    """
    Покупка делится на равные части по числу месяцев рассрочки.
    Части встают в график, начиная со следующего месяца, и каждая
    помнит покупку, из которой родилась.
    """

    if amount <= 0:
        return

    months = max(1, state.installment_months)

    part = amount // months

    remainder = amount - part * (months - 1)

    for number in range(months):
        value = remainder if number == months - 1 else part
        if value > 0:
            state.parts.append((month_index + 1 + number, int(value), cause_event_id))


def add_cash(state: "CardCreditState", amount: int) -> None:
    """
    Снятие наличных и переводы с карты копят процентный долг.
    """

    if amount > 0:
        state.cash_principal += int(amount)


def reverse_purchase(state: "CardCreditState", amount: int, cause_event_id: str | None) -> int:
    """
    Возврат покупки снимает долг ЭТОЙ покупки.

    Гасятся только части, рождённые возвращённой покупкой, начиная
    с самой поздней: вернувшая деньги покупка не должна тянуть
    график до конца срока. Чужие части и наличный долг не
    трогаются: покупка, уже выплаченная выписками, долга не
    оставила, и её возврат это просто деньги на счёте. Гасить
    ими посторонний долг значило бы списать то, чего клиент не
    возвращал.

    Начисленные проценты не трогаются: они уже набежали на долг,
    который действительно существовал.

    Возвращает сколько долга снято.
    """

    remaining = int(amount)

    if remaining <= 0 or not cause_event_id:
        return 0

    released = 0
    kept: list = []

    for due, value, cause in sorted(state.parts, key=lambda part: part[0], reverse=True):

        if cause == cause_event_id and remaining > 0:
            take = min(remaining, value)
            remaining -= take
            released += take
            value -= take

        if value > 0:
            kept.append((due, value, cause))

    state.parts = sorted(kept, key=lambda part: part[0])

    return released


def monthly_interest(state: "CardCreditState") -> int:
    """
    Проценты месяца на наличный долг. Покупки в рассрочке
    процентов не несут: у карты рассрочки ставка покупок ноль.
    """

    if state.cash_principal <= 0 or state.cash_rate <= 0.0:
        return 0

    return int(round(state.cash_principal * state.cash_rate / 12.0))


def minimum_payment(state: "CardCreditState", month_index: int) -> int:
    """
    Минимальный платёж месяца: части рассрочки к сроку плюс
    доля наличного долга плюс начисленные проценты.
    """

    settings = params_module.active().products

    parts = state.due_for(month_index)

    cash = int(round(state.cash_principal * settings.card_cash_min_share))

    return int(parts + cash + state.accrued_interest)


def apply_card_payment(state: "CardCreditState", amount: int, month_index: int) -> int:
    """
    Платёж гасит сначала проценты, затем части рассрочки по
    сроку, затем наличный долг.
    """

    remaining = int(amount)

    if remaining <= 0:
        return 0

    paid = 0

    take = min(remaining, state.accrued_interest)
    state.accrued_interest -= take
    remaining -= take
    paid += take

    kept: list = []

    for due, value, cause in sorted(state.parts, key=lambda part: part[0]):

        if due <= month_index and remaining > 0:
            take = min(remaining, value)
            remaining -= take
            paid += take
            value -= take

        if value > 0:
            kept.append((due, value, cause))

    state.parts = kept

    if remaining > 0 and state.cash_principal > 0:
        take = min(remaining, state.cash_principal)
        state.cash_principal -= take
        remaining -= take
        paid += take

    # Просрочка это состояние договора, а не счётчик обходов:
    # закрытый долг обнуляет её сразу, здесь же. Иначе событие
    # платежа несло бы вчерашнее число дней и противоречило бы
    # соседним записям.
    if state.outstanding <= 0:
        state.dpd = 0
        state.delinquency_marks = ()

    return paid


__all__ = [
    "add_cash",
    "add_purchase",
    "apply_card_payment",
    "minimum_payment",
    "monthly_interest",
    "open_credit",
    "block",
    "cashback_amount",
    "cashback_cap",
    "close",
    "monthly_fee",
    "transfer_fee",
    "unblock",
    "withdrawal_fee",
]
