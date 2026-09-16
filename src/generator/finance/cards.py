from __future__ import annotations

from datetime import datetime, timedelta

from .. import params as params_module
from .entities import CARD_ACTIVE, CARD_BLOCKED, CARD_CLOSED, Card


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


def block(card: Card, ts: datetime, reason: str, days: int | None = None) -> None:

    settings = params_module.active().products

    until = ts + timedelta(days=days if days is not None else settings.card_block_max_days)

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


__all__ = [
    "block",
    "cashback_amount",
    "cashback_cap",
    "close",
    "monthly_fee",
    "transfer_fee",
    "unblock",
    "withdrawal_fee",
]
