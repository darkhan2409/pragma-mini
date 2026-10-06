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
#
# Обязательства — тоже не действие (решение владельца 2026-10-06): взнос
# по кредиту, пополнение под платёж (reason payment_topup) и под выписку
# кредитной карты (card_statement). Ушедший клиент продолжает
# обслуживать долг. Досрочное погашение остаётся действием: его клиент
# выбирает сам.
#
# Действие, визит и целевое действие — три разных понятия:
#   is_client_action   всё, что начал клиент (метка churn, act_*);
#   is_visit           успешный вход в приложение: так визит считает
#                      банк (SQL мобильной команды, авторизации со
#                      status successful) — MAU, WAU, CORE;
#   is_target_action   первое полезное действие после регистрации:
#                      действие клиента или открытие продукта по его
#                      заявке (решение владельца 2026-10-06).
# ============================================================


# Всякое событие этого типа — действие клиента.
ALWAYS: frozenset[str] = frozenset(
    {
        "app_screen",
        "app_operation",
        "transfer_out",
        "p2p_out",
        "cash_withdrawal",
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

# Пополнение под платёж по кредиту и под выписку кредитной карты.
OBLIGATION_TOPUP = ("payment_topup", "card_statement")

# Вход в приложение. Сессия начинается ровно с одного входа.
LOGIN_OPERATIONS = ("login", "biometry_login")
SUCCESS = "success"


def is_client_action(events: pd.DataFrame) -> np.ndarray:
    """
    Маска событий, которые начал сам клиент.
    """
    kind = events["type"]
    reason = events["reason"]
    channel = events["channel"]

    action = kind.isin(ALWAYS)
    action |= (kind == "purchase") & (reason == "purchase")
    action |= (kind == "cash_deposit") & ~reason.isin(OBLIGATION_TOPUP)
    action |= (kind == "bill_payment") & channel.notna() & (channel != "system")
    action |= (kind == "deposit_topup") & (events["direction"] == "debit")
    action |= (kind == "deposit_withdrawal") & (reason == "early_closure") & (events["direction"] == "debit")
    action |= (kind == "transfer_in") & (reason == "transfer") & (events["counterparty"] == OWN_ACCOUNT)
    action |= (kind == "card_blocked") & reason.isin(CLIENT_CARD_BLOCK)
    action |= (kind == "card_unblocked") & reason.isin(CLIENT_CARD_UNBLOCK)
    action |= (kind == "product_closed") & (reason == "early_closure")
    action |= (kind == "product_migrated") & (events["migration_reason"] == "successor_offer")
    action |= (kind == "profile_change") & (events["change_source"] == "client")
    return action.fillna(False).to_numpy(dtype=bool)


def is_visit(events: pd.DataFrame) -> np.ndarray:
    """
    Маска визитов: успешный вход в приложение.
    """
    visit = (
        (events["type"] == "app_operation")
        & events["operation"].isin(LOGIN_OPERATIONS)
        & (events["status"] == SUCCESS)
    )
    return visit.fillna(False).to_numpy(dtype=bool)


def is_target_action(events: pd.DataFrame, action: np.ndarray) -> np.ndarray:
    """
    Маска целевых действий: действие клиента (action — is_client_action
    тех же событий) или продукт, открытый по его заявке. Автоматическая
    активация карты — не целевое действие: её делает банк.
    """
    opened = (events["type"] == "product_opened") & (events["reason"] == "application_approved")
    return action | opened.fillna(False).to_numpy(dtype=bool)
