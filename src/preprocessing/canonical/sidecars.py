from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pyarrow as pa

from ..rawdata import RawDataset
from .schema import coverage_schema, profile_schema


# ============================================================
# ИДЕЯ
# ============================================================
#
# Профиль и покрытие переносятся в canonical как есть: версии не
# схлопываются, «дыры» не заполняются, состояние на дату здесь
# не считается — это работа этапа истории.
#
# opening_state разбирается в числа, но остаётся ЗАЯВЛЕННЫМ
# начальным состоянием: ни счетов, ни договоров, ни прошлых
# событий из него не создаётся.
# ============================================================


OPENING_OK = "parsed"
OPENING_ABSENT = "absent"
OPENING_UNPARSEABLE = "unparseable"
OPENING_UNEXPECTED = "unexpected_values"


def _trace(rows: int, group_index: int, offset: int, name: str) -> dict[str, pa.Array]:
    return {
        "raw_file": pa.array([name] * rows, pa.string()),
        "raw_row_group": pa.array(np.full(rows, group_index, dtype=np.int32)),
        "raw_row": pa.array(np.arange(offset, offset + rows, dtype=np.int64)),
    }


def build_profile(raw: RawDataset, client_index: dict[str, int]) -> tuple[pa.Table, dict]:
    """
    Версии профиля с внутренним индексом клиента и трассировкой.
    """

    schema = profile_schema()

    pieces: list[pa.Table] = []
    offset = 0
    unknown_clients: set[str] = set()

    for group_index, chunk in raw.iter_row_groups("profile"):

        rows = chunk.num_rows

        client_id = chunk.column("client_id").to_pylist()

        idx = []
        for value in client_id:
            if value not in client_index:
                unknown_clients.add(value)
            idx.append(client_index.get(value))

        columns = {name: chunk.column(name) for name in chunk.column_names}
        columns["client_idx"] = pa.array(idx, pa.int64())
        columns.update(_trace(rows, group_index, offset, "profile.parquet"))

        pieces.append(pa.table(columns).select(schema.names).cast(schema))

        offset += rows

    table = pa.concat_tables(pieces) if pieces else schema.empty_table()

    report = {
        "rows": table.num_rows,
        "clients": len(set(table.column("client_id").to_pylist())),
        "clients_unknown_to_index": sorted(unknown_clients),
        "rule": "одна итоговая строка на клиента: версий у профиля нет",
    }

    return table, report


def build_coverage(raw: RawDataset, client_index: dict[str, int]) -> tuple[pa.Table, dict]:
    """
    Покрытие источников с разобранным opening_state.
    """

    schema = coverage_schema()

    pieces: list[pa.Table] = []
    offset = 0
    statuses: dict[str, int] = {}
    observed_keys: dict[str, int] = {}
    unknown_clients: set[str] = set()

    for group_index, chunk in raw.iter_row_groups("source_coverage"):

        rows = chunk.num_rows

        client_id = chunk.column("client_id").to_pylist()

        idx = []
        for value in client_id:
            if value not in client_index:
                unknown_clients.add(value)
            idx.append(client_index.get(value))

        values: list[list[tuple[str, int]] | None] = []
        status: list[str] = []

        for value in chunk.column("opening_state").to_pylist():

            if value is None:
                values.append(None)
                status.append(OPENING_ABSENT)
                continue

            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                values.append(None)
                status.append(OPENING_UNPARSEABLE)
                continue

            if not isinstance(parsed, dict):
                values.append(None)
                status.append(OPENING_UNPARSEABLE)
                continue

            pairs = [(str(key), _as_int(item)) for key, item in sorted(parsed.items())]

            kept = [(key, item) for key, item in pairs if item is not None]

            values.append(kept or None)
            status.append(OPENING_OK if len(kept) == len(pairs) else OPENING_UNEXPECTED)

            for key, _ in pairs:
                observed_keys[key] = observed_keys.get(key, 0) + 1

        for value in status:
            statuses[value] = statuses.get(value, 0) + 1

        columns = {name: chunk.column(name) for name in chunk.column_names}
        columns["client_idx"] = pa.array(idx, pa.int64())
        columns["opening_state_values"] = pa.array(values, pa.map_(pa.string(), pa.int64()))
        columns["opening_state_status"] = pa.array(status, pa.string())
        columns.update(_trace(rows, group_index, offset, "source_coverage.parquet"))

        pieces.append(pa.table(columns).select(schema.names).cast(schema))

        offset += rows

    table = pa.concat_tables(pieces) if pieces else schema.empty_table()

    report = {
        "rows": table.num_rows,
        "clients": len(set(table.column("client_id").to_pylist())),
        "clients_unknown_to_index": sorted(unknown_clients),
        "opening_state": dict(sorted(statuses.items())),
        "opening_state_keys": dict(sorted(observed_keys.items())),
        "rule": (
            "opening_state это заявленное начальное состояние: события и сущности из него не создаются. "
            "Форма зависит от источника, поэтому все объявленные ключи сохраняются картой, а исходная строка лежит рядом"
        ),
    }

    return table, report


def _as_int(value) -> int | None:

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    return None


__all__ = [
    "OPENING_ABSENT",
    "OPENING_OK",
    "OPENING_UNEXPECTED",
    "OPENING_UNPARSEABLE",
    "build_coverage",
    "build_profile",
]
