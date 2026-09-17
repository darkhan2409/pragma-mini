from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# СУЩНОСТИ
# ============================================================
#
# Счёт, карта, договор, заявка, график платежей и депозит.
# Идентификатор объекта не меняется за весь жизненный цикл.
#
# Договор запоминает ВЕРСИЮ продукта и тарифа на дату подписания
# и живёт на ней, пока обслуживается. Новая версия продукта
# касается его только по applies_to.
# ============================================================


ACCOUNT_CURRENT = "current"
ACCOUNT_CARD = "card"
ACCOUNT_CREDIT_CARD = "credit_card"
ACCOUNT_DEPOSIT = "deposit"
ACCOUNT_LOAN = "loan"
ACCOUNT_CASH = "cash"
ACCOUNT_OTHER_BANK = "other_bank"

VISIBLE_ACCOUNT_KINDS = frozenset(
    {ACCOUNT_CURRENT, ACCOUNT_CARD, ACCOUNT_CREDIT_CARD, ACCOUNT_DEPOSIT, ACCOUNT_LOAN}
)

CARD_ISSUED = "issued"
CARD_ACTIVE = "active"
CARD_BLOCKED = "blocked"
CARD_CLOSED = "closed"

CONTRACT_OPEN = "open"
CONTRACT_CLOSED = "closed"


@dataclass
class Account:
    account_id: str
    client_id: str
    kind: str
    balance: int = 0
    currency: str = "KZT"
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    contract_id: str | None = None
    product_code: str | None = None
    credit_limit: int = 0
    opening_balance: int = 0

    @property
    def visible(self) -> bool:
        return self.kind in VISIBLE_ACCOUNT_KINDS

    def is_open_at(self, ts: datetime) -> bool:
        if self.opened_at is not None and ts < self.opened_at:
            return False
        return self.closed_at is None or ts < self.closed_at

    @property
    def available(self) -> int:
        """
        Сколько можно списать: остаток плюс доступный кредитный
        лимит.
        """

        if self.kind == ACCOUNT_CREDIT_CARD:
            return self.credit_limit + self.balance

        return self.balance


@dataclass
class Card:
    card_id: str
    account_id: str
    client_id: str
    contract_id: str
    product_code: str
    issued_at: datetime
    activated_at: datetime | None = None
    status: str = CARD_ISSUED
    blocked_at: datetime | None = None
    blocked_until: datetime | None = None
    block_reason: str | None = None
    closed_at: datetime | None = None
    reissued_from: str | None = None
    expires_at: datetime | None = None

    # Интервалы блокировки. Состояние карты обязано зависеть от
    # ВРЕМЕНИ ОПЕРАЦИИ, а не от порядка исполнения действий:
    # блокировка датируется решением антифрода, а снятие своим
    # моментом, и покупка между ними пройти не может.
    blocks: list = field(default_factory=list)

    def is_blocked_at(self, ts: datetime) -> bool:
        for started, ended in self.blocks:
            if started <= ts and (ended is None or ts < ended):
                return True
        return False

    def usable_at(self, ts: datetime) -> bool:
        if self.activated_at is None or ts < self.activated_at:
            return False
        if self.closed_at is not None and ts >= self.closed_at:
            return False
        return not self.is_blocked_at(ts)


@dataclass
class Contract:
    contract_id: str
    client_id: str
    product_id: str
    product_code: str
    product_family: str
    product_version: int
    tariff_version: int
    opened_at: datetime
    status: str = CONTRACT_OPEN
    closed_at: datetime | None = None
    account_id: str | None = None
    card_id: str | None = None
    amount_or_limit: int | None = None
    term: int | None = None
    rate: float | None = None
    offer_id: str | None = None
    previous_product_id: str | None = None
    application_id: str | None = None
    terms: dict = field(default_factory=dict)
    renewals: int = 0

    def is_open_at(self, ts: datetime) -> bool:
        if ts < self.opened_at:
            return False
        return self.closed_at is None or ts < self.closed_at


@dataclass
class Application:
    application_id: str
    client_id: str
    product_id: str
    product_code: str
    product_family: str
    product_version: int
    channel: str
    submitted_at: datetime
    requested_amount: int | None = None
    requested_term: int | None = None
    offer_id: str | None = None
    decision: str | None = None
    decided_at: datetime | None = None
    reject_reason: str | None = None
    approved_amount: int | None = None
    approved_term: int | None = None


@dataclass
class Installment:
    number: int
    due_date: datetime
    amount: int
    principal: int
    interest: int
    paid_amount: int = 0
    paid_at: datetime | None = None
    status: str = "scheduled"
    due_event_id: str | None = None

    @property
    def outstanding(self) -> int:
        return max(0, self.amount - self.paid_amount)


@dataclass
class LoanState:
    contract_id: str
    principal_outstanding: int
    schedule: list
    rate: float
    autopay: bool
    dpd: int = 0
    arrears: int = 0
    delinquency_marks: tuple = ()
    closed: bool = False
    restructured: bool = False
    first_unpaid: int = 0

    def next_due(self, ts: datetime) -> Installment | None:
        for item in self.schedule:
            if item.status in ("scheduled", "due", "partially_paid") and item.due_date >= ts:
                return item
        return None

    def oldest_unpaid(self) -> Installment | None:
        for item in self.schedule:
            if item.status in ("due", "partially_paid", "missed"):
                return item
        return None


@dataclass
class CardCreditState:
    """
    Долг по карте рассрочки.

    Home Credit продаёт не револьверную карту, а карту
    рассрочки: покупка делится на равные части по числу
    месяцев из тарифа, а снятые наличные копят проценты по
    своей ставке. Минимальный платёж месяца это части к сроку
    плюс доля от наличного долга.
    """

    contract_id: str
    account_id: str
    installment_months: int
    purchase_rate: float
    cash_rate: float
    parts: list = field(default_factory=list)
    cash_principal: int = 0
    accrued_interest: int = 0
    dpd: int = 0
    delinquency_marks: tuple = ()
    closed: bool = False

    def due_for(self, month_index: int) -> int:
        """
        Сколько частей рассрочки приходится на этот месяц.
        """

        return int(sum(amount for due, amount in self.parts if due <= month_index))

    @property
    def outstanding(self) -> int:
        return int(sum(amount for _, amount in self.parts) + self.cash_principal)


@dataclass
class DepositState:
    contract_id: str
    account_id: str
    principal: int
    rate: float
    opened_at: datetime
    matures_at: datetime
    topup_allowed: bool
    withdrawal_allowed: bool
    capitalisation: str = "daily"
    accrued: int = 0
    closed: bool = False


@dataclass
class Offer:
    offer_id: str
    client_id: str
    product_id: str
    product_code: str
    product_family: str
    created_at: datetime
    campaign_code: str
    channel: str
    accepted: bool = False
    seen_at: datetime | None = None


@dataclass
class SupportCase:
    case_id: str
    client_id: str
    topic: str
    channel: str
    opened_at: datetime
    status: str = "open"
    resolution: str | None = None
    resolved_at: datetime | None = None
    cause_event_id: str | None = None


__all__ = [
    "CardCreditState",
    "ACCOUNT_CARD",
    "ACCOUNT_CASH",
    "ACCOUNT_CREDIT_CARD",
    "ACCOUNT_CURRENT",
    "ACCOUNT_DEPOSIT",
    "ACCOUNT_LOAN",
    "ACCOUNT_OTHER_BANK",
    "CARD_ACTIVE",
    "CARD_BLOCKED",
    "CARD_CLOSED",
    "CARD_ISSUED",
    "CONTRACT_CLOSED",
    "CONTRACT_OPEN",
    "VISIBLE_ACCOUNT_KINDS",
    "Account",
    "Application",
    "Card",
    "Contract",
    "DepositState",
    "Installment",
    "LoanState",
    "Offer",
    "SupportCase",
]
