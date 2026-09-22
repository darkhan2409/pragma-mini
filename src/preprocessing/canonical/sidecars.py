from __future__ import annotations


import numpy as np
import pyarrow as pa

from ..rawdata import RawDataset
from .schema import profile_schema


# ============================================================
# ИДЕЯ
# ============================================================
#
# Профиль переносится в canonical как есть: «дыры» не
# заполняются, состояние на дату здесь не считается — это работа
# этапа истории.
# ============================================================


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


__all__ = ["build_profile"]
