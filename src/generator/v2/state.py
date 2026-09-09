from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import ACCOUNT_PRODUCTS, TRANSFER_OWN_MIN_ACCOUNTS


# ============================================================
# ИДЕЯ
# ============================================================
#
# Снимок состояния клиента НА МОМЕНТ действия и жёсткие правила
# поверх него.
#
# Порядок обязателен: сначала жёсткое ограничение, потом
# случайный исход. Пол вероятности исхода не должен разрешать
# невозможный успех, поэтому невозможность выражается флагом,
# а не малым весом.
#
# Различаются два уровня:
#
#   offered   действие вообще предлагается клиенту; заведомо
#             бессмысленные ветки не выбираются
#
#   feasible  действие МОЖЕТ завершиться успехом; клиент
#             может попробовать и получить отказ
#
# Заблокированная карта это как раз второй случай: попытка
# оплаты видна в данных, но пройти она не может.
# ============================================================


@dataclass(frozen=True)
class BillRef:
    """
    Счёт, открытый к оплате на этот момент.
    """

    key: tuple[int, int]
    kind: str
    mcc: str
    amount: int
    due: datetime


@dataclass(frozen=True)
class StateView:
    """
    Что верно о клиенте в конкретную секунду.
    """

    ts: datetime

    owned: frozenset[str] = frozenset()
    blocking: frozenset[str] = frozenset()
    closable: frozenset[str] = frozenset()
    accounts: int = 0
    card_blocked: bool = False
    authorized: bool = False
    due_bills: tuple[BillRef, ...] = ()


# Что операция требует от состояния.
REQUIRES_OWNED: dict[str, str] = {
    "limit_change": "credit_card",
    "loan_view": "cash_loan",
    "loan_schedule": "cash_loan",
    "loan_statement": "cash_loan",
    "deposit_view": "deposit",
    "deposit_topup": "deposit",
}

# Второй такой же договор невозможен, пока действует текущий.
FORBIDS_BLOCKING: dict[str, str] = {
    "deposit_open": "deposit",
}

# Закрыть можно только существующий открытый договор,
# которым уже можно распорядиться.
REQUIRES_CLOSABLE: dict[str, str] = {
    "deposit_close": "deposit",
    "loan_early_repay": "cash_loan",
}

NEEDS_CARD_BLOCKED: frozenset[str] = frozenset({"card_unblock"})
NEEDS_CARD_ACTIVE: frozenset[str] = frozenset({"card_block"})

# Списания, которые не проходят по заблокированной карте.
NEEDS_ACTIVE_CARD: frozenset[str] = frozenset(
    {
        "pay_utility",
        "pay_mobile",
        "pay_internet",
        "pay_fine",
        "pay_tax",
        "pay_qr",
        "transfer_card",
        "transfer_abroad",
        "market_order",
        "deposit_topup",
    }
)

# Операции, доступные без авторизации: сам вход и восстановление.
PUBLIC_OPERATIONS: frozenset[str] = frozenset(
    {"login", "biometry_login", "device_bind"}
)

# Успех обязан менять состояние.
STATEFUL_OPERATIONS: frozenset[str] = frozenset(
    {"deposit_open", "deposit_close", "loan_early_repay", "card_block", "card_unblock"}
)


def accounts_of(owned: frozenset[str]) -> int:
    return sum(1 for product in ACCOUNT_PRODUCTS if product in owned)


def operation_offered(operation: str | None, view: StateView) -> bool:
    """
    Предлагается ли действие. Ветка, ведущая к бессмысленному
    действию, просто не выбирается.
    """

    if operation is None:
        return True

    required = REQUIRES_OWNED.get(operation)

    if required is not None and required not in view.owned:
        return False

    forbidden = FORBIDS_BLOCKING.get(operation)

    if forbidden is not None and forbidden in (view.blocking or view.owned):
        return False

    closes = REQUIRES_CLOSABLE.get(operation)

    if closes is not None and closes not in view.closable:
        return False

    if operation in NEEDS_CARD_BLOCKED and not view.card_blocked:
        return False

    if operation in NEEDS_CARD_ACTIVE and view.card_blocked:
        return False

    if operation == "transfer_own" and view.accounts < TRANSFER_OWN_MIN_ACCOUNTS:
        return False

    return True


def operation_feasible(operation: str, view: StateView) -> bool:
    """
    Может ли действие завершиться успехом ПРЯМО СЕЙЧАС.

    Отличается от offered тем, что состояние могло измениться
    после выбора ветки: клиент уже нажал, но карта успела
    заблокироваться. Тогда попытка видна, а успех невозможен.
    """

    if operation not in PUBLIC_OPERATIONS and not view.authorized:
        return False

    if not operation_offered(operation, view):
        return False

    if view.card_blocked and operation in NEEDS_ACTIVE_CARD:
        return False

    return True


__all__ = [
    "BillRef",
    "NEEDS_ACTIVE_CARD",
    "PUBLIC_OPERATIONS",
    "REQUIRES_CLOSABLE",
    "REQUIRES_OWNED",
    "STATEFUL_OPERATIONS",
    "StateView",
    "accounts_of",
    "operation_feasible",
    "operation_offered",
]
