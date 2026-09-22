from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

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


# Счета, с которых не платят. Кредитный счёт — долг, а не деньги.
# Вклад — деньги, но не платёжные: покупки, счета, переводы и
# взносы идут с карт и текущих счетов, а со вклада деньги выводятся
# отдельной операцией с проверкой условий продукта
# (deposits.can_withdraw). Раньше вклад стоял в очереди источников,
# и клиенты платили со срочных вкладов, запрещённых к снятию.
NON_PAYMENT_KINDS: tuple[str, ...] = ("loan", "deposit")


# Насколько далеко порядок решений может разойтись с порядком
# ленты. Решение принимается не тогда, когда операция датирована:
# счёт к сроку подтягивают на двенадцать минут раньше платежа,
# просроченный счёт платят задним числом, кешбэк месяца начисляют
# до выписки, а датируют после неё. Дальше двух суток это
# расхождение не уходит ни в одном месте генератора.
RECENT_WINDOW = timedelta(days=2)


class Ledger:
    """
    Счета клиента и все проводки по ним.
    """

    def __init__(self, client_id: str) -> None:

        self.client_id = client_id
        self.accounts: dict[str, Account] = {}
        self.postings: list[Posting] = []
        # Проводки по счёту: начисление процентов смотрит только на
        # свой вклад, а не перебирает всю историю клиента.
        self.by_account: dict[str, list[Posting]] = {}
        # Недавние проводки наблюдаемых счетов: по ним считается
        # остаток НА МОМЕНТ операции, а не на момент решения.
        # Скрытые счета сюда не попадают — их в ленте нет вовсе.
        self.recent: dict[str, list[Posting]] = {}
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

        for side in (debit, credit):

            account = self.accounts.get(side)

            if account is None:
                continue

            self.by_account.setdefault(side, []).append(posting)

            if account.visible:
                self.recent.setdefault(side, []).append(posting)

        return posting

    def signed_moves(self, account_id: str, start: datetime, stop: datetime) -> list[tuple]:
        """
        Знаковые движения по счёту в полуинтервале [start, stop).
        """

        out: list[tuple] = []

        for posting in self.by_account.get(account_id, ()):

            if not (start <= posting.ts < stop):
                continue

            if posting.credit == account_id:
                out.append((posting.ts, posting.amount))
            else:
                out.append((posting.ts, -posting.amount))

        return out

    # --------------------------------------------------------

    def available_at(self, account_id: str, ts: datetime) -> int:
        """
        Сколько можно списать со счёта операцией, датированной ts,
        не уведя счёт в минус НИ В ОДНОЙ точке ленты.

        Остаток счёта знает только сумму всех проводок, а решения
        принимаются не в том порядке, в котором строки лягут в
        файл. Поэтому здесь считается иначе:

          проводка, датированная позже ts, из остатка вычитается —
          в ленте на этом месте её ещё нет;

          затем остаток прокручивается вперёд по уже известным
          проводкам, и берётся самая низкая его точка. Списание
          опускает всю дальнейшую цепочку на свою сумму, так что
          разрешить можно ровно эту глубину.

        Кредитному счёту минус разрешён: available знает про
        кредитный лимит. Обычному счёту лимита нет, и available
        равен остатку.
        """

        account = self.accounts.get(account_id)

        if account is None:
            return 0

        recent = self.recent.get(account_id)

        if not recent:
            return account.available

        # Чистится ЗДЕСЬ, по спрашиваемому моменту, а не по самой
        # поздней проводке: проводка бывает датирована и на неделю
        # вперёд (возврат, chargeback), и она обязана дожить до
        # своего дня. Всё, что старше окна расхождения, в ленте
        # заведомо стоит раньше любой следующей проверки.
        if recent[0].ts < ts - RECENT_WINDOW:
            recent[:] = [item for item in recent if item.ts >= ts - RECENT_WINDOW]

        ahead = sorted(
            (
                (posting.ts, posting.amount if posting.credit == account_id else -posting.amount)
                for posting in recent
                if posting.ts > ts
            ),
            key=lambda item: item[0],
        )

        if not ahead:
            return account.available

        running = account.available - sum(value for _, value in ahead)

        lowest = running

        for _, value in ahead:
            running += value
            lowest = min(lowest, running)

        return lowest

    def can_debit(self, account_id: str, amount: int, ts: datetime) -> bool:
        account = self.accounts.get(account_id)
        if account is None:
            return False
        return self.available_at(account_id, ts) >= amount

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
            and account.kind not in NON_PAYMENT_KINDS
            and self.available_at(account.account_id, ts) >= amount
        ]

        order = {"card": 0, "current": 1, "credit_card": 2}

        candidates.sort(key=lambda item: (order.get(item.kind, 9), item.account_id))

        return candidates

    def payment_capacity(self, ts: datetime) -> int:
        """
        Сколько клиент способен заплатить прямо сейчас с самого
        подходящего счёта.
        """

        sources = [
            self.available_at(account.account_id, ts)
            for account in self.accounts.values()
            if account.visible
            and account.is_open_at(ts)
            and account.kind not in NON_PAYMENT_KINDS
        ]

        return int(max(sources)) if sources else 0

    def hidden_sources(self, amount: int) -> list:
        return [
            account
            for account in (self.accounts[self.cash_id], self.accounts[self.other_bank_id])
            if account.balance >= amount
        ]


__all__ = [
    "NON_PAYMENT_KINDS",
    "COUNTERPART_BANK",
    "COUNTERPART_GOVERNMENT",
    "InsufficientFunds",
    "Ledger",
    "Posting",
]
