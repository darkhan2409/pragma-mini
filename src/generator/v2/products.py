from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from ..products import Holding, ProductEvent, ProductState, add_months, open_contract
from .config import CARD_BLOCK_MAX_DAYS, EARLY_CLOSE_MIN_MONTHS


# ============================================================
# СОСТОЯНИЕ ПРОДУКТОВ V2
# ============================================================
#
# В v1 договор закрывается только по сроку, а карта не имеет
# состояния вовсе. В v2 успешная операция обязана менять то,
# что она по смыслу меняет:
#
#     deposit_close, loan_early_repay -> договор закрыт раньше
#     card_block / card_unblock       -> интервал блокировки
#
# Блокировка карты это НЕ договор: она не видна в реестре,
# но видна по последствиям (оплаты падают, клиент идёт
# в раздел карт). В RAW она не пишется.
# ============================================================


class ProductStateV2(ProductState):

    def __init__(self, client_id: int) -> None:

        super().__init__(client_id)

        # Интервалы блокировки карты: (начало, конец | None).
        self.card_blocks: list[tuple[datetime, datetime | None]] = []

    # --------------------------------------------------------
    # ОТКРЫТИЕ ПОСЛЕ СВОЕЙ ПРИЧИНЫ
    # --------------------------------------------------------

    def open_after(
        self,
        product_type: str,
        ts: datetime,
        not_before: datetime,
    ) -> ProductEvent:
        """
        Открывает договор так, чтобы запись о нём не оказалась
        РАНЬШЕ события, которое его вызвало.

        У депозита timestamp_quality почти всегда date_only, и время
        открытия теряется: договор падает в полночь своего дня. Для
        договора, открытого днём, это дало бы запись раньше причины.
        Реестр в таком случае учитывает его следующим днём.

        not_before это сама причина: решение по заявке или успешная
        операция. Сравнение идёт с ней, а не с моментом открытия,
        иначе договор с точным временем уезжал бы на сутки без нужды.
        """

        event, holding = open_contract(self.client_id, product_type, ts)

        if event.ts <= not_before:

            later = (ts + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(hours=12)

            event, holding = open_contract(self.client_id, product_type, later)

        self._add(event, holding)

        return event

    # --------------------------------------------------------
    # ДОСРОЧНОЕ ЗАКРЫТИЕ
    # --------------------------------------------------------

    def can_close_early(self, product_type: str, ts: datetime) -> bool:
        """
        Закрыть можно только то, что открыто достаточно давно.
        """

        return self._closable_index(product_type, ts) is not None

    def _closable_index(self, product_type: str, ts: datetime) -> int | None:

        for index, holding in enumerate(self.holdings):

            if holding.product_type != product_type:
                continue

            if not holding.is_open_at(ts):
                continue

            if add_months(holding.opened_at, EARLY_CLOSE_MIN_MONTHS) > ts:
                continue

            return index

        return None

    def close_early(self, product_type: str, ts: datetime) -> bool:
        """
        Закрывает договор моментом ts. Возвращает, получилось ли.
        """

        index = self._closable_index(product_type, ts)

        if index is None:
            return False

        holding = self.holdings[index]

        self.holdings[index] = replace(holding, closed_at=ts)

        return True

    # --------------------------------------------------------
    # БЛОКИРОВКА КАРТЫ
    # --------------------------------------------------------

    def card_blocked_at(self, ts: datetime) -> bool:

        for started, ended in self.card_blocks:
            if started <= ts and (ended is None or ts < ended):
                return True

        return False

    def block_card(self, ts: datetime) -> bool:

        if self.card_blocked_at(ts):
            return False

        # Долгая блокировка снимается сама: клиент перевыпускает
        # карту в отделении, и это не событие приложения.
        self.card_blocks.append((ts, ts + timedelta(days=CARD_BLOCK_MAX_DAYS)))

        return True

    def unblock_card(self, ts: datetime) -> bool:

        for index, (started, ended) in enumerate(self.card_blocks):

            if started <= ts and (ended is None or ts < ended):
                self.card_blocks[index] = (started, ts)
                return True

        return False

    def blocked_intervals(self) -> tuple[tuple[datetime, datetime | None], ...]:
        return tuple(self.card_blocks)

    # --------------------------------------------------------
    # ДОСТУПНОСТЬ ДЕЙСТВИЙ
    # --------------------------------------------------------

    def blocking_products(self, ts: datetime) -> frozenset[str]:
        """
        Продукты, второй договор по которым сейчас невозможен,
        включая уже запланированное на будущее открытие.

        Нужно, чтобы успешная операция не была потом молча
        отброшена: недоступное действие просто не предлагается.
        """

        return frozenset(
            holding.product_type
            for holding in self.holdings
            if holding.closed_at is None or holding.closed_at > ts
        )

    def closable_products(self, ts: datetime) -> frozenset[str]:
        """
        Что можно закрыть досрочно прямо сейчас.
        """

        return frozenset(
            holding.product_type
            for holding in self.holdings
            if holding.is_open_at(ts)
            and add_months(holding.opened_at, EARLY_CLOSE_MIN_MONTHS) <= ts
        )


def holding_opened_at(state: ProductState, product_type: str, ts: datetime) -> datetime | None:

    for holding in state.holdings:
        if holding.product_type == product_type and holding.is_open_at(ts):
            return holding.opened_at

    return None


__all__ = ["ProductStateV2", "Holding", "holding_opened_at"]
