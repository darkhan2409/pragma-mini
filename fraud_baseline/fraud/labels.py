from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .raw import ENVELOPE


# ============================================================
# ПОВТОР → СТРОКИ RAW
# ============================================================
#
# Повтор отдаёт помеченные строки ровно в том виде, в каком генератор
# пишет их в файл: client_id, строка времени, source, строка payload. Здесь
# каждая из них находит свою строку RAW по точному совпадению всех четырёх
# полей и получает её номер raw_row. Помеченная строка без пары или с
# несколькими парами — остановка: метке нельзя доверять.
#
# Там же, одним проходом по файлу, считается хеш ленты каждого клиента.
# Он обязан совпасть с хешем повтора у всех клиентов группы.
#
# Независимая сверка: у мошеннической операции генератора есть следы
# сериализации — у покупки card_id идёт сразу за channel, у перевода нет
# transfer_id при заполненном account_id. Эти следы в признаки не идут; здесь
# по ним только проверяется, что повтор пометил те же строки.
# ============================================================


SEPARATOR = "\x1f"
TERMINATOR = "\x1e"

PURCHASE_TRACE = r'^\{"type":"purchase","channel":"[^"]*","card_id"'


def _joined(table: pa.Table) -> pa.Array:
    return pc.binary_join_element_wise(*[table.column(name).combine_chunks() for name in ENVELOPE], SEPARATOR)


def _signature(payload: pa.Array) -> np.ndarray:
    purchase = pc.match_substring_regex(payload, PURCHASE_TRACE)
    transfer = pc.and_(
        pc.and_(pc.starts_with(payload, '{"type":"transfer_out"'), pc.invert(pc.match_substring(payload, '"transfer_id"'))),
        pc.match_substring(payload, '"account_id"'),
    )
    return pc.or_(purchase, transfer).to_numpy(zero_copy_only=False)


def attach_rows(raw_path: Path, captured: pd.DataFrame, digests: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """
    Номера строк RAW для помеченных строк повтора и сверка хешей лент.
    """
    wanted = pa.array(
        [SEPARATOR.join(values) for values in captured[ENVELOPE].itertuples(index=False, name=None)],
        type=pa.string(),
    )
    found: list[tuple[int, int]] = []
    hashes: dict[str, "hashlib._Hash"] = {}
    signature: list[int] = []

    parquet = pq.ParquetFile(raw_path)
    first_row = 0
    for index in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(index, columns=ENVELOPE)
        joined = _joined(table)

        position = pc.index_in(joined, value_set=wanted).to_numpy(zero_copy_only=False)
        rows = np.flatnonzero(~np.isnan(position.astype(float)))
        found += [(first_row + int(row), int(position[row])) for row in rows]
        signature += (first_row + np.flatnonzero(_signature(table.column("payload").combine_chunks()))).tolist()

        lines = pc.binary_join_element_wise(joined, pa.scalar(TERMINATOR), "")
        offsets = np.frombuffer(lines.buffers()[1], dtype=np.int32)[lines.offset : lines.offset + len(lines) + 1]
        data = lines.buffers()[2]
        ids = table.column("client_id").to_numpy()
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ends = np.r_[starts[1:], len(ids)]
        for start, end in zip(starts, ends):
            digest = hashes.get(ids[start])
            if digest is None:
                digest = hashes[ids[start]] = hashlib.sha256()
            digest.update(data[int(offsets[start]) : int(offsets[end])].to_pybytes())
        first_row += table.num_rows

    raw_digests = {client: digest.hexdigest() for client, digest in hashes.items()}
    if set(raw_digests) != set(digests):
        raise RuntimeError(
            f"клиенты повтора и RAW расходятся: {len(set(digests) - set(raw_digests))} лишних, "
            f"{len(set(raw_digests) - set(digests))} недостающих"
        )
    mismatched = sorted(client for client, value in raw_digests.items() if digests[client] != value)
    if mismatched:
        raise RuntimeError(f"лента {len(mismatched)} клиентов в повторе не совпала с RAW, например {mismatched[:3]}")

    matches = pd.DataFrame(found, columns=["raw_row", "captured"])
    per_mark = matches.groupby("captured").size().reindex(range(len(captured)), fill_value=0)
    if (per_mark != 1).any():
        raise RuntimeError(f"помеченных строк без единственной пары в RAW: {int((per_mark != 1).sum())}")

    labels = captured.iloc[matches["captured"].to_numpy()].reset_index(drop=True)
    labels.insert(2, "raw_row", matches["raw_row"].to_numpy())
    payloads = [json.loads(text) for text in labels["payload"]]
    labels["type"] = [body["type"] for body in payloads]
    labels["amount"] = [body.get("amount") for body in payloads]
    labels["event_time"] = pd.to_datetime(labels["event_time"], format="ISO8601", utc=True)
    labels = labels.drop(columns=["source", "payload"]).sort_values("raw_row", ignore_index=True)

    marked = set(labels["raw_row"])
    traced = set(signature)
    check = {
        "clients": len(raw_digests),
        "client_digests_equal": True,
        "marked_rows": len(marked),
        "signature_rows": len(traced),
        "signature_equals_marked": marked == traced,
        "marked_without_signature": len(marked - traced),
        "signature_without_mark": len(traced - marked),
    }
    return labels, check
