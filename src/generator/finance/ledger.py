from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .entities import (
    ACCOUNT_CASH,
    ACCOUNT_CREDIT_CARD,
    ACCOUNT_OTHER_BANK,
    Account,
)


# ============================================================
# LEDGER
# ============================================================
#
# ЗАКОН СОХРАНЕНИЯ. Каждое движение денег это проводка с двумя
# сторонами, и сумма сторон равна нулю. Деньги не возникают и
# не исчезают: у любого движения есть явный источник и явный
# получатель.
#
# Стороны бывают внутренними (счета клиентов, включая скрытые
# наличные и счёт в другом банке) и внешними, названными явно:
#
#   merchant:<outlet>   торговая точка
#   employer:<id>       работодатель или плательщик
#   government          бюджет
#   external:<name>     контрагент за пределами банка
#   bank_pnl            доход и расход самого банка
#
# Отклонённая или отменённая клиентом операция проводки НЕ
# создаёт: баланс от неё не меняется.
# ============================================================


COUNTERPART_BANK = "bank_pnl"
COUNTERPART_GOVERNMENT = "government"


@dataclass(frozen=True)
class Posting:
    posting_id: str
    event_id: str
    ts: datetime
    debit: str
    credit: str
    amount: int

    @property
    def balanced(self) -> bool:
        return self.amount > 0


class InsufficientFunds(Exception):
    pass


class Ledger:
    """
    Счета клиента и все проводки по ним.
    """

    def __init__(self, client_id: str) -> None:

        self.client_id = client_id
        self.accounts: dict[str, Account] = {}
        self.postings: list[Posting] = []
        self._counter = 0

        self.cash_id = f"cash:{client_id}"
        self.other_bank_id = f"other:{client_id}"

        self.accounts[self.cash_id] = Account(
            account_id=self.cash_id, client_id=client_id, kind=ACCOUNT_CASH
        )
        self.accounts[self.other_bank_id] = Account(
            account_id=self.other_bank_id, client_id=client_id, kind=ACCOUNT_OTHER_BANK
        )

    # --------------------------------------------------------

    def add_account(self, account: Account) -> Account:
        self.accounts[account.account_id] = account
        return account

    def get(self, account_id: str) -> Account | None:
        return self.accounts.get(account_id)

    def balance(self, account_id: str) -> int:
        account = self.accounts.get(account_id)
        return account.balance if account else 0

    def available(self, account_id: str) -> int:
        account = self.accounts.get(account_id)
        return account.available if account else 0

    def visible_accounts(self, ts: datetime | None = None) -> list:
        return [
            account
            for account in self.accounts.values()
            if account.visible and (ts is None or account.is_open_at(ts))
        ]

    def total_visible_balance(self) -> int:
        return sum(
            account.balance
            for account in self.accounts.values()
            if account.visible and account.kind != ACCOUNT_CREDIT_CARD
        )

    def hidden_funds(self) -> int:
        return self.balance(self.cash_id) + self.balance(self.other_bank_id)

    # --------------------------------------------------------

    def post(
        self,
        ts: datetime,
        event_id: str,
        debit: str,
        credit: str,
        amount: int,
    ) -> Posting:
        """
        Одна проводка: со счёта debit на счёт credit.
        Внутренние стороны меняют остаток, внешние только
        называются.
        """

        amount = int(amount)

        if amount <= 0:
            raise ValueError("сумма проводки должна быть положительной")

        self._counter += 1

        posting = Posting(
            posting_id=f"pst_{self.client_id}_{self._counter}",
            event_id=event_id,
            ts=ts,
            debit=debit,
            credit=credit,
            amount=amount,
        )

        source = self.accounts.get(debit)

        if source is not None:
            source.balance -= amount

        target = self.accounts.get(credit)

        if target is not None:
            target.balance += amount

        self.postings.append(posting)

        return posting

    # --------------------------------------------------------

    def can_debit(self, account_id: str, amount: int) -> bool:
        account = self.accounts.get(account_id)
        if account is None:
            return False
        return account.available >= amount

    def payment_sources(self, ts: datetime, amount: int) -> list:
        """
        Счета, с которых можно заплатить указанную сумму,
        от самого подходящего к запасному.
        """

        candidates = [
            account
            for account in self.accounts.values()
            if account.visible
            and account.is_open_at(ts)
            and account.kind != "loan"
            and account.available >= amount
        ]

        order = {"card": 0, "current": 1, "credit_card": 2, "deposit": 3}

        candidates.sort(key=lambda item: (order.get(item.kind, 9), item.account_id))

        return candidates

    def payment_capacity(self, ts: datetime) -> int:
        """
        Сколько клиент способен заплатить прямо сейчас с самого
        подходящего счёта.
        """

        sources = [
            account.available
            for account in self.accounts.values()
            if account.visible
            and account.is_open_at(ts)
            and account.kind != "loan"
        ]

        return int(max(sources)) if sources else 0

    def hidden_sources(self, amount: int) -> list:
        return [
            account
            for account in (self.accounts[self.cash_id], self.accounts[self.other_bank_id])
            if account.balance >= amount
        ]


__all__ = [
    "COUNTERPART_BANK",
    "COUNTERPART_GOVERNMENT",
    "InsufficientFunds",
    "Ledger",
    "Posting",
]
