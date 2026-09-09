from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import (
    RECENT_FAILURE_DAYS,
    RECENT_OFFER_DAYS,
    RECENT_REJECTION_DAYS,
    RECENT_REMINDER_DAYS,
    RECENT_VIEW_DAYS,
    UNFINISHED_DAYS,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Причинность в v2 определяется ВРЕМЕНЕМ события, а не порядком
# обработки сессий. Сессии дня перекрываются: сессия с 09:00
# может закончиться в 09:55, а другая начаться в 09:15.
#
# Поэтому память клиента это журнал записей с timestamp, а
# любой запрос view(ts) возвращает только то, что произошло
# СТРОГО РАНЬШЕ ts. Сбой в 09:50 не виден решению в 09:15
# и не виден шагу в 09:40, но виден шагу в 09:56.
# ============================================================


KIND_FAILURE = "failure"
KIND_OFFER = "offer"
KIND_REMINDER = "reminder"
KIND_REJECTION = "rejection"
KIND_VIEW = "view"
KIND_UNFINISHED = "unfinished"
KIND_BILL_PAID = "bill_paid"

MAX_WINDOW_DAYS = max(
    RECENT_FAILURE_DAYS,
    RECENT_OFFER_DAYS,
    RECENT_REJECTION_DAYS,
    RECENT_REMINDER_DAYS,
    RECENT_VIEW_DAYS,
    UNFINISHED_DAYS,
)


@dataclass(frozen=True)
class Record:
    ts: datetime
    kind: str
    key: str


@dataclass(frozen=True)
class ContextView:
    """
    Всё, что клиент помнит на конкретный момент.
    """

    recent_failures: tuple[str, ...]
    recent_offers: frozenset[str]
    recent_reminder: bool
    recent_rejections: frozenset[str]
    views: dict[str, int]
    unfinished: str
    due_bills: tuple = ()
    card_blocked: bool = False
    credit_need: float = 0.0


@dataclass
class ClientContext:
    """
    Журнал наблюдаемых последствий. Хранит только недавнее:
    окно памяти ограничено, а прошлое уже сыграло свою роль.
    """

    records: list[Record] = field(default_factory=list)
    paid_bills: set[tuple[str, int]] = field(default_factory=set)

    def record(self, ts: datetime, kind: str, key: str = "") -> None:
        self.records.append(Record(ts=ts, kind=kind, key=key))

    def prune(self, day: datetime) -> None:
        """
        Выбрасывает записи, которые уже никого не интересуют.
        """

        edge = day - timedelta(days=MAX_WINDOW_DAYS + 1)

        if len(self.records) > 64:
            self.records = [item for item in self.records if item.ts >= edge]

    # --------------------------------------------------------

    def view(
        self,
        ts: datetime,
        due_bills: tuple = (),
        card_blocked: bool = False,
        credit_need: float = 0.0,
    ) -> ContextView:
        """
        Срез памяти на момент ts. Записи со временем >= ts
        не участвуют: будущее не влияет на прошлое.
        """

        failures: list[str] = []
        offers: set[str] = set()
        rejections: set[str] = set()
        views: dict[str, int] = {}
        reminder = False
        unfinished = ""
        unfinished_ts: datetime | None = None

        for item in self.records:

            if item.ts >= ts:
                continue

            age = (ts - item.ts).total_seconds() / 86400.0

            if item.kind == KIND_FAILURE:
                if age <= RECENT_FAILURE_DAYS:
                    failures.append(item.key)

            elif item.kind == KIND_OFFER:
                if age <= RECENT_OFFER_DAYS:
                    offers.add(item.key)

            elif item.kind == KIND_REMINDER:
                if age <= RECENT_REMINDER_DAYS:
                    reminder = True

            elif item.kind == KIND_REJECTION:
                if age <= RECENT_REJECTION_DAYS:
                    rejections.add(item.key)

            elif item.kind == KIND_VIEW:
                if age <= RECENT_VIEW_DAYS:
                    views[item.key] = views.get(item.key, 0) + 1

            elif item.kind == KIND_UNFINISHED:
                if age <= UNFINISHED_DAYS:
                    if unfinished_ts is None or item.ts > unfinished_ts:
                        unfinished = item.key
                        unfinished_ts = item.ts

        return ContextView(
            recent_failures=tuple(failures),
            recent_offers=frozenset(offers),
            recent_reminder=reminder,
            recent_rejections=frozenset(rejections),
            views=views,
            unfinished=unfinished,
            due_bills=due_bills,
            card_blocked=card_blocked,
            credit_need=credit_need,
        )


__all__ = [
    "KIND_BILL_PAID",
    "KIND_FAILURE",
    "KIND_OFFER",
    "KIND_REJECTION",
    "KIND_REMINDER",
    "KIND_UNFINISHED",
    "KIND_VIEW",
    "ClientContext",
    "ContextView",
    "Record",
]
