from __future__ import annotations

from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from churn.raw import RawContractError, client_blocks
from world import LOCAL, UTC, event, login, purchase, write_group


def test_blocks_are_whole_clients_across_row_groups(tmp_path) -> None:
    start = datetime(2025, 5, 1, 10, tzinfo=LOCAL)
    events = []
    for index, client in enumerate(["c1", "c2", "c3"]):
        for step in range(5 + index):
            events.append(purchase(client, start + timedelta(hours=step), amount=100 * (step + 1)))
    out = write_group(tmp_path, "val", datetime(2026, 4, 1, tzinfo=LOCAL), events, [], row_group_size=3)

    blocks = list(client_blocks(out / "events.parquet"))
    rows = [row for block in blocks for row in block.itertuples()]
    assert len(rows) == len(events)
    seen = [set(block["client_id"]) for block in blocks]
    assert all(not (a & b) for i, a in enumerate(seen) for b in seen[i + 1 :])
    assert sorted(row.raw_row for row in rows) == list(range(len(events)))


def test_time_is_parsed_to_utc_with_and_without_milliseconds(tmp_path) -> None:
    moment = datetime(2025, 3, 18, 19, 32, tzinfo=LOCAL)
    precise = moment + timedelta(microseconds=250000)
    events = [purchase("c1", moment), login("c1", precise)]
    out = write_group(tmp_path, "val", datetime(2026, 4, 1, tzinfo=LOCAL), events, [])
    (block,) = client_blocks(out / "events.parquet")
    assert block["t"].iloc[0].to_pydatetime() == moment.astimezone(UTC)
    assert block["t"].iloc[1].to_pydatetime() == precise.astimezone(UTC)
    assert block["type"].tolist() == ["purchase", "app_operation"]
    assert block["amount"].iloc[0] == 1000


def test_other_offset_breaks_the_contract(tmp_path) -> None:
    row = event("c1", datetime(2025, 3, 18, 19, 32, tzinfo=LOCAL), "app_screens", type="app_screen")
    row["event_time"] = "2025-03-18T14:32:00+00:00"
    out = write_group(tmp_path, "val", datetime(2026, 4, 1, tzinfo=LOCAL), [row], [])
    with pytest.raises(RawContractError):
        list(client_blocks(out / "events.parquet"))


def test_client_split_into_two_runs_breaks_the_contract(tmp_path) -> None:
    moment = datetime(2025, 3, 18, 19, 32, tzinfo=LOCAL)
    rows = [purchase("c1", moment), purchase("c2", moment), purchase("c1", moment + timedelta(hours=1))]
    table = pa.Table.from_pylist(rows)
    path = tmp_path / "events.parquet"
    pq.write_table(table, path, row_group_size=2)
    with pytest.raises(RawContractError):
        list(client_blocks(path))


def test_an_empty_export_yields_no_blocks(tmp_path) -> None:
    out = write_group(tmp_path, "val", datetime(2026, 4, 1, tzinfo=LOCAL), [], [])
    assert list(client_blocks(out / "events.parquet")) == []

