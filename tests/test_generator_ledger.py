from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime

import pyarrow.parquet as pq
import pytest

from src.generator import emit
from src.generator import engine


# ============================================================
# ИДЕЯ
# ============================================================
#
# Деньги и договоры в ленте согласованы сами с собой:
#
#   - остаток после операции продолжает предыдущий остаток счёта:
#     balance_after = прежний ± сумма;
#   - снимок остатка равен текущему остатку счёта;
#   - у перевода между клиентами банка две ноги с одним
#     transfer_id и одной суммой;
#   - закрытой (перевыпущенной) картой не платят после закрытия;
#   - покупки в точке (POS) без карты не бывает.
# ============================================================


START = datetime(2024, 1, 1)
END = datetime(2024, 9, 1)


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    """
    Выгрузка и состояния клиентов той же симуляции: _finish ищется
    как глобал модуля, обёртка ничего не меняет в розыгрышах.
    """

    out = tmp_path_factory.mktemp("ledger") / "raw"

    captured: list = []
    original = engine._finish

    def watching(sim):
        captured.extend(sim.clients[ordinal] for ordinal in sorted(sim.clients))
        return original(sim)

    engine._finish = watching

    try:
        emit.generate_dataset(
            total_clients=24, out_dir=out, seed=91, world_seed=42, history_start=START,
            history_end=END, workers=1, community_size=4, quiet=True,
        )
    finally:
        engine._finish = original

    table = pq.read_table(out / "events.parquet")

    rows = [
        (client, when, json.loads(payload))
        for client, when, payload in zip(
            table.column("client_id").to_pylist(),
            table.column("event_time").to_pylist(),
            table.column("payload").to_pylist(),
        )
    ]

    return {"rows": rows, "states": captured}


def test_the_balance_after_continues_the_previous_balance(world):

    last: dict[str, int] = {}
    checked = 0

    for _, _, payload in world["rows"]:

        account = payload.get("account_id")
        balance = payload.get("balance_after")

        if account is None or balance is None:
            continue

        if payload["type"] == "balance_snapshot":
            if account in last:
                assert balance == last[account], payload
                checked += 1
            last[account] = balance
            continue

        if payload.get("status") != "approved":
            continue

        if account in last:
            sign = 1 if payload["direction"] == "credit" else -1
            assert balance == last[account] + sign * payload["amount"], payload
            checked += 1

        last[account] = balance

    assert checked > 1000, "проверка вырождена"


def test_both_legs_of_a_transfer_between_clients_agree(world):

    legs: dict[str, dict] = defaultdict(dict)

    for _, _, payload in world["rows"]:
        if payload["type"] in ("p2p_out", "p2p_in") and payload.get("status") == "approved":
            legs[payload["transfer_id"]][payload["type"]] = payload["amount"]

    paired = [item for item in legs.values() if len(item) == 2]

    assert paired, "в выборке есть внутрибанковские переводы"

    for item in legs.values():
        # Нога получателя могла остаться за окном выгрузки только
        # вместе с ногой отправителя: зачисление идёт через секунду.
        assert set(item) == {"p2p_out", "p2p_in"}, item
        assert item["p2p_out"] == item["p2p_in"], item


def test_a_closed_card_is_not_used_after_its_closure(world):

    for state in world["states"]:
        for event in state.events:

            card_id = event.payload.get("card_id")

            if card_id is None or event.event_type in ("card_reissued", "card_blocked", "card_unblocked"):
                continue

            card = state.cards.get(card_id)

            assert card is not None, (state.client_id, event.event_type, card_id)

            if card.closed_at is not None and event.payload.get("status") == "approved":
                assert event.event_time < card.closed_at, (state.client_id, event.event_type, event.event_time)


def test_a_purchase_in_a_shop_always_has_a_card(world):

    # QR платят со счёта через приложение, карта там не участвует.
    shop = [
        payload for _, _, payload in world["rows"]
        if payload["type"] == "purchase" and payload.get("channel") == "pos"
        and payload.get("is_online") is False
    ]

    assert shop

    for payload in shop:
        assert payload.get("card_id"), payload
