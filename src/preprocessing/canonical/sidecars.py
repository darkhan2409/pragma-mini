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


def build_profile(raw: RawDataset, client_index: dict[str, int]) -> pa.Table:
    """
    Версии профиля с внутренним индексом клиента и трассировкой.
    """

    schema = profile_schema()

    pieces: list[pa.Table] = []
    offset = 0

    for group_index, chunk in raw.iter_row_groups("profile"):

        rows = chunk.num_rows

        client_id = chunk.column("client_id").to_pylist()

        idx = [client_index.get(value) for value in client_id]

        columns = {name: chunk.column(name) for name in chunk.column_names}
        columns["client_idx"] = pa.array(idx, pa.int64())
        columns.update(_trace(rows, group_index, offset, "profile.parquet"))

        pieces.append(pa.table(columns).select(schema.names).cast(schema))

        offset += rows

    return pa.concat_tables(pieces) if pieces else schema.empty_table()


__all__ = ["build_profile"]
