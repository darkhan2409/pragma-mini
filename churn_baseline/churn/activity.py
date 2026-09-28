from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# АКТИВНОСТЬ САМОГО КЛИЕНТА
# ============================================================
#
# Событие — действие клиента, если его начал сам клиент. Инициатор
# определён по коду генератора и полям RAW, а не по объявленному
# генератором списку CLIENT_ACTION_EVENT_TYPES: список расходится с тем,
# что код делает на деле (автоплатёж лежит в bill_payment, автоматическая
# активация карты — в card_activated).
#
# Двойные проводки считаются по стороне, которую начал клиент: у
# пополнения вклада — дебетовая нога, у перевода между своими счетами —
# transfer_out.
#
# Не действие клиента: исходящие коммуникации банка, начисления,
# списания и автоплатежи, зачисления от третьих лиц, учёт кредита,
# решения банка и антифрода, показ баннера (сессию клиента и так видно по
# app_operation и app_screen).
# ============================================================


# Всякое событие этого типа — действие клиента.
ALWAYS: frozenset[str] = frozenset(
    {
        "app_screen",
        "app_operation",
        "transfer_out",
        "p2p_out",
        "cash_withdrawal",
        "cash_deposit",
        "early_repayment",
        "application_submitted",
        "banner_clicked",
        "case_opened",
    }
)

# Причины блокировки и разблокировки карты, которые исходят от клиента.
CLIENT_CARD_BLOCK = ("client_freeze", "lost_or_stolen")
CLIENT_CARD_UNBLOCK = ("client_request",)

# Деньги с собственного счёта в другом банке: клиент заводит их сам.
OWN_ACCOUNT = "Own account"


def is_client_action(events: pd.DataFrame) -> np.ndarray:
    """
    Маска событий, которые начал сам клиент.
    """
    kind = events["type"]
    reason = events["reason"]
    channel = events["channel"]

    action = kind.isin(ALWAYS)
    action |= (kind == "purchase") & (reason == "purchase")
    action |= kind.isin(["bill_payment", "loan_payment"]) & channel.notna() & (channel != "system")
    action |= (kind == "deposit_topup") & (events["direction"] == "debit")
    action |= (kind == "deposit_withdrawal") & (reason == "early_closure") & (events["direction"] == "debit")
    action |= (kind == "transfer_in") & (reason == "transfer") & (events["counterparty"] == OWN_ACCOUNT)
    action |= (kind == "card_blocked") & reason.isin(CLIENT_CARD_BLOCK)
    action |= (kind == "card_unblocked") & reason.isin(CLIENT_CARD_UNBLOCK)
    action |= (kind == "product_closed") & (reason == "early_closure")
    action |= (kind == "product_migrated") & (events["migration_reason"] == "successor_offer")
    action |= (kind == "profile_change") & (events["change_source"] == "client")
    return action.fillna(False).to_numpy(dtype=bool)
