"""
Коммуникации: контракт, доставка, право на кампанию.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pytest

from src.generator.communications import (
    CAMPAIGNS,
    campaign_weights,
    generate_communications_for_day,
)
from src.generator.config import FEATURE_END, HISTORY_START
from src.generator.coverage import app_adoption
from src.generator.history import generate_client_history
from src.generator.persona import draw_persona
from src.generator.products import add_months
from src.generator.world import (
    CAMPAIGN_PRODUCT,
    CAMPAIGN_TEMPLATES,
    COMM_CHANNELS,
    DELIVERY_RATE,
)


ALL_TEMPLATES = {
    template for templates in CAMPAIGN_TEMPLATES.values() for template in templates
}

TEMPLATE_CAMPAIGN = {
    template: campaign
    for campaign, templates in CAMPAIGN_TEMPLATES.items()
    for template in templates
}

POPULATION = 60


@pytest.fixture(scope="module")
def histories():
    return {
        client_id: generate_client_history(client_id) for client_id in range(POPULATION)
    }


def all_events(histories):
    return [event for history in histories.values() for event in history.communications]


# ============================================================
# КОНТРАКТ
# ============================================================


def test_contract_fields(histories):
    events = all_events(histories)

    assert events

    for event in events:
        assert event.channel in COMM_CHANNELS
        assert event.template in ALL_TEMPLATES
        assert event.day_of_week == event.ts.weekday()
        assert event.hour == event.ts.hour
        assert isinstance(event.delivered, bool)


def test_campaign_is_not_in_contract(histories):
    event = all_events(histories)[0]

    assert not hasattr(event, "campaign")
    assert not hasattr(event, "outcome")
    assert not hasattr(event, "product")


def test_all_three_channels_used(histories):
    channels = Counter(event.channel for event in all_events(histories))

    assert set(channels) == set(COMM_CHANNELS)


# ============================================================
# ДОСТАВКА
# ============================================================


def test_delivery_rate_by_channel(histories):
    events = all_events(histories)

    for channel in COMM_CHANNELS:

        sent = [event for event in events if event.channel == channel]

        assert len(sent) > 30, channel

        delivered = sum(1 for event in sent if event.delivered) / len(sent)

        # push дополнительно теряется на недостижимых клиентах.
        upper = DELIVERY_RATE[channel] * 1.25
        lower = DELIVERY_RATE[channel] * (0.35 if channel == "push" else 0.75)

        assert lower <= delivered <= upper, (channel, delivered)


def test_push_never_delivered_without_app():
    client_id = next(c for c in range(200) if app_adoption(c) is None)

    history = generate_client_history(client_id)

    pushes = [e for e in history.communications if e.channel == "push"]

    assert all(not event.delivered for event in pushes)


# ============================================================
# ПРАВО НА КАМПАНИЮ
# ============================================================


def test_payment_reminder_requires_credit_product():
    persona = draw_persona(0)

    weights = dict(
        zip(CAMPAIGNS, campaign_weights(persona, 0.0, frozenset({"debit_card"})))
    )

    assert weights["payment_reminder"] == 0.0

    with_credit = dict(
        zip(
            CAMPAIGNS,
            campaign_weights(persona, 0.0, frozenset({"debit_card", "cash_loan"})),
        )
    )

    assert with_credit["payment_reminder"] > 0.0


def test_cashback_requires_a_card():
    persona = draw_persona(0)

    weights = dict(zip(CAMPAIGNS, campaign_weights(persona, 0.0, frozenset())))

    assert weights["cashback"] == 0.0


def test_offer_for_owned_product_is_suppressed():
    persona = draw_persona(0)

    without = dict(zip(CAMPAIGNS, campaign_weights(persona, 0.0, frozenset({"debit_card"}))))
    with_loan = dict(
        zip(CAMPAIGNS, campaign_weights(persona, 0.0, frozenset({"debit_card", "cash_loan"})))
    )

    assert with_loan["cash_loan_offer"] < 0.2 * without["cash_loan_offer"]


def test_stress_raises_payment_reminders():
    persona = draw_persona(0)

    owned = frozenset({"debit_card", "cash_loan"})

    calm = dict(zip(CAMPAIGNS, campaign_weights(persona, 0.0, owned)))
    stressed = dict(zip(CAMPAIGNS, campaign_weights(persona, 0.9, owned)))

    assert stressed["payment_reminder"] > 2.0 * calm["payment_reminder"]


def test_reminders_in_history_only_for_credit_owners(histories):
    """
    В сгенерированной истории напоминание приходит только тем,
    у кого на этот день есть кредитный продукт.
    """

    checked = 0

    for client_id, history in histories.items():

        # Владение восстанавливаем из самой истории: договоры
        # открываются в том числе внутри окна.
        contracts = [
            (
                event.ts,
                add_months(event.ts, event.term) if event.term else None,
                event.product_type,
            )
            for event in history.product_events
        ]

        for event in history.communications:

            if TEMPLATE_CAMPAIGN[event.template] != "payment_reminder":
                continue

            day = event.ts.replace(hour=0, minute=0, second=0, microsecond=0)

            owned = {
                product
                for opened, closed, product in contracts
                if opened <= day and (closed is None or day < closed)
            }

            assert owned & {"credit_card", "cash_loan"}, (client_id, event.ts)

            checked += 1

    assert checked > 0


# ============================================================
# ШАБЛОНЫ
# ============================================================


def test_template_distribution_has_a_long_tail(histories):
    counts = Counter(event.template for event in all_events(histories))

    assert len(counts) > 15

    ordered = [value for _, value in counts.most_common()]

    top = ordered[0]
    median = ordered[len(ordered) // 2]
    tail = ordered[-1]

    # Распределение скошенное: у частого шаблона на порядок
    # больше отправок, чем у хвостового.
    assert top > 4 * median
    assert top > 10 * tail


def test_day_hour_consistency_on_disk(raw_tables):
    communications = raw_tables["communications"]

    assert (communications.day_of_week == communications.ts.dt.weekday).all()
    assert (communications.hour == communications.ts.dt.hour).all()
