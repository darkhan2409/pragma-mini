from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest

from fraud.labels import attach_rows
from fraud.replay import FALSE_POSITIVE, row_digest_bytes
from world import LOCAL, event, login, purchase, transfer, write_group

END = datetime(2026, 4, 1, tzinfo=LOCAL)
NOW = datetime(2025, 6, 10, 14, 0, tzinfo=LOCAL)


def tape(tmp_path):
    rows = [
        purchase("a", NOW - timedelta(days=1)),
        # Как пишет мошеннический путь генератора: card_id сразу за channel.
        event("a", NOW, "transactions", type="purchase", channel="ecom", card_id="crd_1", mcc="5732", amount=90000, reason="purchase", status="approved", account_id="acc_1"),
        login("b", NOW - timedelta(hours=1)),
        transfer("b", NOW, amount=40000, transfer_id=None, account_id="acc_2", counterparty="X. Y"),
        transfer("b", NOW + timedelta(hours=1), amount=1000),
    ]
    out = write_group(tmp_path, "val", END, rows, [], row_group_size=2)
    raw = pq.read_table(out / "events.parquet").to_pandas()
    digests = {}
    for row in raw.itertuples(index=False):
        digests.setdefault(row.client_id, hashlib.sha256()).update(row_digest_bytes(row.client_id, row.event_time, row.source, row.payload))
    return out / "events.parquet", raw, {client: value.hexdigest() for client, value in digests.items()}


def captured(raw: pd.DataFrame, positions: list[int], kinds: list[str]) -> pd.DataFrame:
    part = raw.iloc[positions][["client_id", "event_time", "source", "payload"]].reset_index(drop=True)
    part["episode_kind"] = kinds
    part["step_kind"] = "strike"
    return part


def test_marked_rows_get_their_raw_row_numbers(tmp_path) -> None:
    path, raw, digests = tape(tmp_path)
    labels, check = attach_rows(path, captured(raw, [1, 3], ["card_compromise", FALSE_POSITIVE]), digests)
    assert labels["raw_row"].tolist() == [1, 3]
    assert labels["type"].tolist() == ["purchase", "transfer_out"]
    assert labels["amount"].tolist() == [90000, 40000]
    assert check["client_digests_equal"] and check["signature_equals_marked"]


def test_a_tape_that_differs_from_raw_is_rejected(tmp_path) -> None:
    path, raw, digests = tape(tmp_path)
    digests["b"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(RuntimeError, match="не совпала"):
        attach_rows(path, captured(raw, [1], ["card_compromise"]), digests)


def test_a_mark_without_its_raw_row_is_rejected(tmp_path) -> None:
    path, raw, digests = tape(tmp_path)
    ghost = captured(raw, [1], ["card_compromise"])
    body = json.loads(ghost.loc[0, "payload"])
    body["amount"] = 1
    ghost.loc[0, "payload"] = json.dumps(body)
    with pytest.raises(RuntimeError, match="без единственной пары"):
        attach_rows(path, ghost, digests)
