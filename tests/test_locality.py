"""
Локальность RNG: изменение одного дня не пересеивает другие.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.generator import app as app_module
from src.generator import communications as comm_module
from src.generator import transactions as tx_module
from src.generator.config import FEATURE_END, HISTORY_START
from src.generator.coverage import app_adoption
from src.generator.noise import apply_noise
from src.generator.transactions import TransactionEvent, generate_transaction_history


CLIENT = 5

DAY = datetime(2025, 3, 15)


def other_days(events):
    return [event for event in events if event.ts.date() != DAY.date()]


def same_day(events):
    return [event for event in events if event.ts.date() == DAY.date()]


def test_changing_one_day_keeps_other_days(monkeypatch):
    """
    Если в один день добавить транзакций, остальные дни
    обязаны остаться байт в байт теми же.
    """

    baseline = generate_transaction_history(CLIENT, HISTORY_START, FEATURE_END)

    original = tx_module.daily_purchase_rate

    def bumped(persona, ts):
        rate = original(persona, ts)
        return rate + 3.0 if ts.date() == DAY.date() else rate

    monkeypatch.setattr(tx_module, "daily_purchase_rate", bumped)

    perturbed = generate_transaction_history(CLIENT, HISTORY_START, FEATURE_END)

    assert other_days(perturbed) == other_days(baseline)
    assert len(same_day(perturbed)) > len(same_day(baseline))


def test_extra_sessions_do_not_change_other_days(monkeypatch):
    client_id = next(c for c in range(200) if app_adoption(c) is not None)

    day_before = DAY - timedelta(days=1)

    baseline = app_module.sessions_for_day(client_id, day_before)

    original = app_module.daily_session_rate

    def bumped(persona, ts):
        rate = original(persona, ts)
        return rate + 5.0 if ts.date() == DAY.date() else rate

    monkeypatch.setattr(app_module, "daily_session_rate", bumped)

    assert app_module.sessions_for_day(client_id, day_before) == baseline


def test_communication_content_is_keyed_by_event_identity():
    """
    Содержимое отправки зависит от дня и индекса в дне,
    а не от порядкового номера в истории клиента.
    """

    owned = frozenset({"debit_card"})

    first = comm_module.generate_communications_for_day(
        CLIENT, DAY, HISTORY_START, FEATURE_END, owned, True
    )

    second = comm_module.generate_communications_for_day(
        CLIENT, DAY, HISTORY_START, FEATURE_END, owned, True
    )

    assert first == second


def test_ownership_changes_only_affect_campaign_choice():
    """
    Смена владения меняет кампанию, но не время отправки:
    время разыгрывается отдельным компонентом RNG.
    """

    poor = comm_module.generate_communications_for_day(
        CLIENT, DAY, HISTORY_START, FEATURE_END, frozenset({"debit_card"}), True
    )

    rich = comm_module.generate_communications_for_day(
        CLIENT,
        DAY,
        HISTORY_START,
        FEATURE_END,
        frozenset({"debit_card", "cash_loan", "deposit"}),
        True,
    )

    assert [o.event.ts for o in poor] == [o.event.ts for o in rich]


def purchase(ts) -> TransactionEvent:
    return TransactionEvent(
        client_id=CLIENT,
        ts=ts,
        amount=1000,
        direction="debit",
        mcc="5411",
        merchant_city="Almaty",
        merchant_country="KZ",
        is_online=False,
        is_subscription=False,
    )


def test_noise_depends_on_event_identity_not_row_order():
    event = purchase(DAY.replace(hour=10, minute=5, second=7))

    assert apply_noise("transactions", event, CLIENT) == apply_noise(
        "transactions", event, CLIENT
    )


def test_noise_varies_across_events():
    cities = {
        apply_noise("transactions", purchase(DAY + timedelta(seconds=step)), CLIENT).merchant_city
        for step in range(4000)
    }

    assert cities == {"Almaty", None}


def test_noise_is_a_no_op_for_untouched_sources():
    event = purchase(DAY)

    # Баннеры наблюдаются без потерь: событие возвращается как есть.
    assert apply_noise("banners", event, CLIENT) is event
