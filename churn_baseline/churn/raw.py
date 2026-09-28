from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pj
import pyarrow.parquet as pq

from .config import RAW_OFFSET_SUFFIX


# ============================================================
# ЧТЕНИЕ ВЫГРУЗКИ ГЕНЕРАТОРА
# ============================================================
#
# RAW — конверт client_id, event_time, source, payload, все четыре
# колонки строковые; payload — JSON. Строки одного клиента в файле
# идут подряд одним блоком, поэтому таблица читается по row group, а
# хвостовой клиент группы строк переносится в следующую: наружу уходят
# только клиенты целиком.
#
# Время — строка ISO 8601 со смещением Казахстана; здесь она
# становится моментом UTC. Другое смещение — нарушение контракта
# выгрузки, и чтение останавливается: календарный день клиента
# считается от местной полуночи по этому смещению.
# ============================================================


ENVELOPE = ["client_id", "event_time", "source", "payload"]

# Поля payload, которые читает проект. Остальные ключи пропускаются.
PAYLOAD = pa.schema(
    [
        ("type", pa.string()),
        ("channel", pa.string()),
        ("reason", pa.string()),
        ("status", pa.string()),
        ("direction", pa.string()),
        ("is_subscription", pa.bool_()),
        ("is_online", pa.bool_()),
        ("amount", pa.int64()),
        ("balance_after", pa.int64()),
        ("account_id", pa.string()),
        ("card_id", pa.string()),
        ("contract_id", pa.string()),
        ("product_id", pa.string()),
        ("mcc", pa.string()),
        ("merchant_name", pa.string()),
        ("counterparty", pa.string()),
        ("decline_reason", pa.string()),
        ("days_past_due", pa.int64()),
        ("decision", pa.string()),
        ("delivered", pa.bool_()),
        ("template", pa.string()),
        ("operation", pa.string()),
        ("session_id", pa.string()),
        ("change_source", pa.string()),
        ("field_name", pa.string()),
        ("old_value", pa.string()),
        ("new_value", pa.string()),
        ("migration_reason", pa.string()),
    ]
)


class RawContractError(ValueError):
    """
    Выгрузка не соответствует контракту, на который опирается проект.
    """


def decode_payload(payload: pa.Array, schema: pa.Schema = PAYLOAD) -> pa.Table:
    """
    JSON-строки payload в таблицу нужных полей одним разбором pyarrow.

    Строки склеиваются через перевод строки в один буфер: JSON-строка не
    содержит сырого перевода строки, поэтому граница записи однозначна.
    """
    if len(payload) == 0:
        return schema.empty_table()
    if payload.null_count:
        raise RawContractError("пустой payload в выгрузке")

    lines = pc.binary_join_element_wise(payload, pa.scalar("\n"), "")
    offsets = np.frombuffer(lines.buffers()[1], dtype=np.int32)[lines.offset : lines.offset + len(lines) + 1]
    data = lines.buffers()[2].slice(int(offsets[0]), int(offsets[-1] - offsets[0]))

    table = pj.read_json(
        pa.BufferReader(data),
        read_options=pj.ReadOptions(block_size=1 << 24, use_threads=True),
        parse_options=pj.ParseOptions(explicit_schema=schema, unexpected_field_behavior="ignore"),
    )
    if table.num_rows != len(payload):
        raise RawContractError(f"разобрано {table.num_rows} payload из {len(payload)}")
    return table.select(schema.names)


def decode(envelope: pa.Table, first_row: int, schema: pa.Schema = PAYLOAD) -> pd.DataFrame:
    """
    Строки конверта в плоскую таблицу: client_id, t (UTC), source, raw_row
    и поля payload. raw_row — номер строки в файле выгрузки.
    """
    times = envelope.column("event_time").combine_chunks()
    if pc.sum(pc.invert(pc.ends_with(times, RAW_OFFSET_SUFFIX))).as_py():
        raise RawContractError(f"время RAW не со смещением {RAW_OFFSET_SUFFIX}")

    frame = decode_payload(envelope.column("payload").combine_chunks(), schema).to_pandas()
    frame.insert(0, "client_id", envelope.column("client_id").to_pandas())
    frame.insert(1, "t", pd.to_datetime(times.to_pandas(), format="ISO8601", utc=True))
    frame.insert(2, "source", envelope.column("source").to_pandas())
    frame.insert(3, "raw_row", np.arange(first_row, first_row + envelope.num_rows, dtype=np.int64))
    return frame


def client_blocks(path: Path, schema: pa.Schema = PAYLOAD) -> Iterator[pd.DataFrame]:
    """
    События выгрузки блоками целых клиентов, внутри клиента по времени.

    Клиент, встретившийся во втором несмежном куске файла, нарушает
    контракт выгрузки: его агрегаты были бы посчитаны по половине истории.
    """
    parquet = pq.ParquetFile(path)
    seen: set[str] = set()
    carry: pd.DataFrame | None = None
    first_row = 0

    for index in range(parquet.metadata.num_row_groups):
        envelope = parquet.read_row_group(index, columns=ENVELOPE)
        frame = decode(envelope, first_row, schema)
        first_row += envelope.num_rows

        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)

        ids = frame["client_id"].to_numpy()
        tail = ids == ids[-1]
        carry = frame[tail]
        block = frame[~tail]
        if len(block):
            yield _ordered(block, seen)

    if carry is not None and len(carry):
        yield _ordered(carry, seen)


def _ordered(block: pd.DataFrame, seen: set[str]) -> pd.DataFrame:
    ids = block["client_id"].to_numpy()
    runs = 1 + int((ids[1:] != ids[:-1]).sum())
    clients = set(ids)
    if runs != len(clients) or clients & seen:
        raise RawContractError("строки клиента в выгрузке идут не одним блоком")
    seen |= clients
    return block.sort_values(["client_id", "t", "raw_row"], kind="stable", ignore_index=True)


def read_profile(path: Path) -> pd.DataFrame:
    """
    Анкета: одна строка на клиента, снимок на границу выгрузки as_of.
    Списки employment и lifelong остаются списками словарей.
    """
    table = pq.read_table(path)
    frame = table.drop_columns(["employment", "lifelong"]).to_pandas()
    frame["employment"] = table.column("employment").to_pylist()
    frame["lifelong"] = table.column("lifelong").to_pylist()
    if frame["client_id"].duplicated().any():
        raise RawContractError("в анкете клиент повторяется")
    return frame
