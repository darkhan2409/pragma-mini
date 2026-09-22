from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ..rawdata import RawDataset, RawManifest, event_types_of, iter_event_types, parse_payloads
from ..settings import PreprocessingConfig
from .schema import (
    DERIVED_NAMES,
    ENVELOPE_NAMES,
    PAYLOAD_NULL,
    PAYLOAD_OK,
    PAYLOAD_UNPARSEABLE,
    canonical_column,
    events_schema,
    payload_columns,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Разбор ленты в типизированный canonical. Одна строка RAW —
# одна строка canonical, ничего не исчезает.
#
# Работа идёт пачками целых клиентов: все версии, дубли и
# ссылки одного события лежат у одного клиента, поэтому пачка
# самодостаточна, а память ограничена её размером. Клиент,
# попавший на границу row group, переносится в следующую пачку
# целиком.
#
# Разбор payload векторный: для каждого типа события строится
# таблица полной ширины, затем все они складываются и
# возвращаются в исходный порядок одной перестановкой.
# ============================================================


class CanonicalError(ValueError):
    """
    Вход необрабатываем: строка не даёт построить порядок,
    версию или связь. Этап останавливается с названием поля и
    номером строки, а не падает исключением Python где-то
    глубже и не выбрасывает строку молча.
    """


# Структурно обязательные поля конверта. Паспорт блокирует
# выгрузку с пустым значением здесь, но canonical не полагается
# на то, что его запускали: проверка повторяется до сортировки,
# индексов и связей.
REQUIRED_ENVELOPE_FIELDS: tuple[str, ...] = (
    "client_id",
    "event_time",
    "source",
)


def require_envelope(batch: pa.Table) -> None:
    """
    Останавливает разбор на первой строке с пустым обязательным
    полем конверта.
    """

    for name in REQUIRED_ENVELOPE_FIELDS:

        column = batch.column(name)

        if not column.null_count:
            continue

        position = pc.indices_nonzero(pc.is_null(column))[0].as_py()
        row = batch.column("raw_row")[position].as_py() if "raw_row" in batch.column_names else position

        raise CanonicalError(
            f"строка {row}: пустое обязательное поле конверта {name}. "
            f"Всего таких строк в пачке: {column.null_count}. "
            "Выгрузка необрабатываема: выполните этап passport, он называет все такие строки"
        )


# Тип события, чья сумма это сам остаток, а не движение по счёту.
# Объявляется явно: по каталогу ключей такое свойство не видно.
SNAPSHOT_EVENT_TYPES: frozenset[str] = frozenset({"balance_snapshot"})

TEXT_NORMALIZATION = {
    "version": "1.0.0",
    "rule": "Unicode NFKC, схлопывание пробелов, единый регистр casefold; исходное значение сохраняется рядом",
}

def normalize_text(value: str | None) -> str | None:
    """
    Нормализованная копия текста. Исходное значение остаётся в
    своей колонке: транслитерации и переименования здесь нет.
    """

    if value is None:
        return None

    text = unicodedata.normalize("NFKC", value)
    text = " ".join(text.split())

    return text.casefold() or None


# ============================================================
# ПАЧКИ КЛИЕНТОВ
# ============================================================


def _client_runs(client_id: np.ndarray) -> list[tuple[int, int]]:
    """
    Границы [lo, hi) непрерывных блоков клиента.
    """

    if client_id.size == 0:
        return []

    change = np.flatnonzero(client_id[1:] != client_id[:-1]) + 1

    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [client_id.size]])

    return [(int(lo), int(hi)) for lo, hi in zip(starts, ends)]


def iter_client_batches(raw: RawDataset, batch_clients: int) -> Iterator[pa.Table]:
    """
    Пачки строк ленты, содержащие только целых клиентов.

    Клиент, чей блок обрывается на границе row group, переносится
    в следующую пачку: делить его нельзя, иначе версии и ссылки
    события окажутся в разных пачках.

    Клиент, встретившийся второй раз после другого, это уже не
    перенос, а разорванная лента. Молчать об этом нельзя: второй
    блок перезаписал бы первый в индексе, и часть истории
    исчезла бы без следа.
    """

    carry: pa.Table | None = None
    offset = 0
    seen: set = set()

    for group_index, chunk in raw.iter_row_groups("events"):

        rows = chunk.num_rows

        chunk = chunk.append_column("raw_row_group", pa.array(np.full(rows, group_index, dtype=np.int32)))
        chunk = chunk.append_column("raw_row", pa.array(np.arange(offset, offset + rows, dtype=np.int64)))

        offset += rows

        table = chunk if carry is None else pa.concat_tables([carry, chunk])

        client_id = np.asarray(table.column("client_id").to_pylist(), dtype=object)

        runs = _client_runs(client_id)

        if not runs:
            carry = None
            continue

        # Последний блок может продолжиться в следующем row group.
        last_start = runs[-1][0]

        head = table.slice(0, last_start)
        carry = table.slice(last_start)

        if head.num_rows == 0:
            continue

        head_runs = _client_runs(np.asarray(head.column("client_id").to_pylist(), dtype=object))

        for start in range(0, len(head_runs), batch_clients):
            window = head_runs[start : start + batch_clients]
            lo = window[0][0]
            hi = window[-1][1]
            piece = head.slice(lo, hi - lo)
            _refuse_split_clients(piece, seen)
            yield piece

    if carry is not None and carry.num_rows:
        _refuse_split_clients(carry, seen)
        yield carry


def _refuse_split_clients(batch: pa.Table, seen: set) -> None:
    """
    Клиент не может начаться заново после другого клиента.
    """

    client_id = np.asarray(batch.column("client_id").to_pylist(), dtype=object)

    for lo, _hi in _client_runs(client_id):

        client = client_id[lo]

        if client in seen:
            raise CanonicalError(
                f"строки клиента {client} лежат в ленте не подряд: "
                "пачка целых клиентов собирается только из непрерывного блока"
            )

        seen.add(client)


# ============================================================
# РАЗБОР PAYLOAD
# ============================================================


@dataclass
class ParsedBatch:
    table: pa.Table
    status: list[str]
    violations: dict[int, list[str]]
    rejects: list[dict]
    counts: dict[str, int] = field(default_factory=dict)


def parse_batch(manifest: RawManifest, batch: pa.Table, payload_names: list[str]) -> ParsedBatch:
    """
    Типизированные колонки payload для всей пачки, в исходном
    порядке строк.
    """

    rows = batch.num_rows

    columns = {name: field_type for name, field_type in payload_columns(manifest)}

    status = [PAYLOAD_OK] * rows
    violations: dict[int, list[str]] = {}
    rejects: list[dict] = []
    counts: dict[str, int] = {}

    if rows == 0:
        empty = pa.table(
            {canonical_column(name): pa.nulls(0, columns[canonical_column(name)]) for name in payload_names}
        )
        return ParsedBatch(empty, status, violations, rejects, counts)

    order: list[np.ndarray] = []
    pieces: list[pa.Table] = []

    # Тип события читается из payload: колонки event_type в
    # выгрузке нет. Строка без разобранного типа уходит в
    # unknown_event_type, а не получает выдуманный тип.
    event_type_column = event_types_of(batch.column("payload"))

    for event_type, _ in iter_event_types(batch):

        mask = (
            pc.equal(event_type_column, event_type)
            if event_type is not None
            else pc.is_null(event_type_column)
        )
        indices = np.asarray(pc.indices_nonzero(mask).to_pylist(), dtype=np.int64)

        rows_of_type = batch.filter(mask)

        info = manifest.catalogue.get(event_type) if event_type is not None else None

        if info is None:
            # Тип вне каталога ключей: строка сохраняется, поля пустые.
            for local, position in enumerate(indices):
                status[int(position)] = PAYLOAD_UNPARSEABLE
                violations.setdefault(int(position), []).append(f"unknown_event_type:{event_type}")
                rejects.append(
                    {
                        "client_id": rows_of_type.column("client_id")[local].as_py(),
                        "event_type": event_type,
                        "reason": "unknown_event_type",
                        "detail": "типа нет в каталоге ключей манифеста",
                        "payload": rows_of_type.column("payload")[local].as_py(),
                        "raw_file": "events.parquet",
                        "raw_row_group": rows_of_type.column("raw_row_group")[local].as_py(),
                        "raw_row": rows_of_type.column("raw_row")[local].as_py(),
                    }
                )
            counts["unknown_event_type"] = counts.get("unknown_event_type", 0) + len(indices)
            piece = pa.table(
                {
                    canonical_column(name): pa.nulls(len(indices), columns[canonical_column(name)])
                    for name in payload_names
                }
            )
            order.append(indices)
            pieces.append(piece)
            continue

        parsed = parse_payloads(info, rows_of_type.column("payload"))

        for kind, count in parsed.counts.items():
            counts[kind] = counts.get(kind, 0) + count

        for local, items in parsed.by_row.items():
            violations.setdefault(int(indices[local]), []).extend(items)

        for local, items in parsed.by_row.items():
            if any(item.startswith(("unparseable", "null_payload")) for item in items):
                position = int(indices[local])
                status[position] = PAYLOAD_UNPARSEABLE if items[0].startswith("unparseable") else PAYLOAD_NULL
                rejects.append(
                    {
                        "client_id": rows_of_type.column("client_id")[local].as_py(),
                        "event_type": event_type,
                        "reason": items[0].split(":")[0],
                        "detail": "; ".join(items),
                        "payload": rows_of_type.column("payload")[local].as_py(),
                        "raw_file": "events.parquet",
                        "raw_row_group": rows_of_type.column("raw_row_group")[local].as_py(),
                        "raw_row": rows_of_type.column("raw_row")[local].as_py(),
                    }
                )

        piece = pa.table(
            {
                canonical_column(name): (
                    parsed.table.column(name)
                    if name in parsed.table.column_names
                    else pa.nulls(len(indices), columns[canonical_column(name)])
                )
                for name in payload_names
            }
        )

        order.append(indices)
        pieces.append(piece)

    stacked = pa.concat_tables(pieces)

    positions = np.concatenate(order)
    inverse = np.empty(rows, dtype=np.int64)
    inverse[positions] = np.arange(positions.size, dtype=np.int64)

    return ParsedBatch(stacked.take(pa.array(inverse)), status, violations, rejects, counts)


# ============================================================
# РАЗРЫВ ЦЕПОЧКИ ОСТАТКОВ
# ============================================================
#
# Наблюдаемый остаток счёта обязан продолжать предыдущий
# наблюдаемый остаток того же счёта. Если не продолжает, между
# строками потеряно движение денег: сбой источника не донёс
# запись до выгрузки.
#
# Это признак КАЧЕСТВА НАБЛЮДЕНИЯ, а не ошибка арифметики, и
# считается он по самой выгрузке: ничего, кроме неё, у
# canonical нет.
#
# Строки одного счёта с одинаковым event_time считаются группой:
# порядок внутри секунды в ленте задаёт приоритет типа события,
# а не очередь проводок, поэтому сверяется итог группы.
# ============================================================


def balance_chain_gaps(
    runs: list[tuple[int, int]],
    event_type,
    event_time,
    stable_index,
    raw_row,
    account_id,
    amount,
    direction,
    status,
    balance_after,
) -> np.ndarray:
    """
    Отмечает строки, чей остаток не продолжает предыдущий.

    Участвуют только одобренные денежные строки с известным
    счётом и остатком.
    """

    rows = len(event_type)

    gap = np.zeros(rows, dtype=bool)

    if not rows:
        return gap

    for lo, hi in runs:

        order = sorted(range(lo, hi), key=lambda index: (stable_index[index], raw_row[index]))

        per_account: dict[object, list[int]] = {}

        for index in order:

            # Снимок остатка статуса не несёт: он не операция, а
            # сообщение о том, сколько на счёте лежит. Из цепочки
            # остатков его выбрасывать нельзя — именно он её и
            # подтверждает.
            if (
                status[index] != "approved"
                and event_type[index] not in SNAPSHOT_EVENT_TYPES
            ):
                continue

            if account_id[index] is None or balance_after[index] is None:
                continue

            per_account.setdefault(account_id[index], []).append(index)

        for positions in per_account.values():

            previous: int | None = None
            start = 0

            while start < len(positions):

                stop = start + 1

                while stop < len(positions) and event_time[positions[stop]] == event_time[positions[start]]:
                    stop += 1

                group = positions[start:stop]
                start = stop

                delta = 0

                for index in group:
                    if event_type[index] in SNAPSHOT_EVENT_TYPES:
                        continue
                    value = int(amount[index] or 0)
                    delta += value if direction[index] == "credit" else -value

                final = int(balance_after[group[-1]])

                if previous is not None and previous + delta != final:
                    for index in group:
                        gap[index] = True

                previous = final

    return gap


# ============================================================
# СБОРКА ПАЧКИ
# ============================================================


@dataclass
class BatchResult:
    table: pa.Table
    rejects: list[dict]
    clients: list[dict]
    counts: dict[str, int]


def build_batch(
    raw: RawDataset,
    config: PreprocessingConfig,
    batch: pa.Table,
    payload_names: list[str],
    schema: pa.Schema,
    client_index: dict[str, int],
    row_group: int,
    row_start: int,
) -> BatchResult:

    manifest = raw.manifest
    rows = batch.num_rows

    # До разбора, сортировки, версий и связей: дальше по коду
    # пустое обязательное поле превращается либо в исключение
    # сравнения, либо в невидимую строку.
    require_envelope(batch)

    parsed = parse_batch(manifest, batch, payload_names)

    client_id = np.asarray(batch.column("client_id").to_pylist(), dtype=object)
    runs = _client_runs(client_id)

    # --- клиентский индекс ---

    client_idx = np.empty(rows, dtype=np.int64)
    clients: list[dict] = []

    event_time = batch.column("event_time").to_numpy(zero_copy_only=False).astype("datetime64[us]")
    raw_row = np.asarray(batch.column("raw_row").to_pylist(), dtype=np.int64)

    # Причинный порядок внутри одной секунды задаёт приоритет типа
    # события из контракта: договор открыт, потом график,
    # потом выдача. Приоритет в данные не пишется и токеном не
    # становится; неизвестный тип уходит в конец секунды.
    priority_of = manifest.event_type_priority
    last_priority = len(priority_of)
    event_type = parsed.table.column("event_type").to_pylist()
    priority = np.asarray(
        [priority_of.get(name, last_priority) for name in event_type],
        dtype=np.int64,
    )

    stable_index = np.empty(rows, dtype=np.int64)

    for lo, hi in runs:

        # Индекс клиента берётся из общего реестра: он не зависит
        # от порядка пачек и одинаков во всех таблицах слоя.
        index = client_index[str(client_id[lo])]
        client_idx[lo:hi] = index

        # Номер события в истории клиента: строки упорядочены
        # по времени, приоритету типа и своему месту в RAW.
        # Идентификатора записи нет, версий и дублей тоже: каждая
        # строка это отдельное событие, и место у неё своё.
        positions = sorted(
            range(lo, hi),
            key=lambda position: (
                event_time[position],
                priority[position],
                raw_row[position],
            ),
        )

        for rank, position in enumerate(positions):
            stable_index[position] = rank

        clients.append(
            {
                "client_idx": index,
                "client_id": str(client_id[lo]),
                # Адрес уточняется по фактической разбивке файла
                # после записи: писатель волен резать row group
                # иначе, чем легли пачки.
                "row_group": None,
                "row_offset": None,
                "global_row_start": row_start + lo,
                "row_count": hi - lo,
                "spans_row_groups": None,
                "event_time_min": event_time[lo:hi].min().astype("datetime64[us]").astype(datetime),
                "event_time_max": event_time[lo:hi].max().astype("datetime64[us]").astype(datetime),
            }
        )

    # --- наблюдаемость ---

    period_start = np.datetime64(manifest.period_start, "us")
    period_end = np.datetime64(manifest.period_end, "us")

    before_window = event_time < period_start
    after_extract = event_time >= period_end

    # --- неоднозначное местное время ---

    ambiguous = np.zeros(rows, dtype=bool)
    for interval in config.ambiguous_local_intervals:
        start = np.datetime64(interval.start, "us")
        end = np.datetime64(interval.end, "us")
        ambiguous |= (event_time >= start) & (event_time < end)

    # --- разрыв цепочки остатков ---

    def payload_column(name: str) -> list:
        if name not in parsed.table.column_names:
            return [None] * rows
        return parsed.table.column(name).to_pylist()

    chain_gap = balance_chain_gaps(
        runs,
        event_type,
        event_time,
        stable_index,
        raw_row,
        payload_column("account_id"),
        payload_column("amount"),
        payload_column("direction"),
        payload_column("status"),
        payload_column("balance_after"),
    )

    # --- пропуски с известной причиной ---
    #
    # Правило схемы говорит, С КАКОЙ даты источник начал
    # собирать поле. Пусто ДО этой даты — известная причина;
    # пусто после — обычный пропуск без объяснения. Раньше знак
    # сравнения стоял наоборот, и причина приписывалась ровно
    # тем строкам, у которых её нет.

    known_missing: list[list[str] | None] = [None] * rows

    source = np.asarray(batch.column("source").to_pylist(), dtype=object)

    for rule in manifest.schema_changes:
        name = rule.get("field")
        if name not in parsed.table.column_names:
            continue
        since = np.datetime64(datetime.fromisoformat(str(rule["from"])), "us")
        is_null = pc.is_null(parsed.table.column(name)).to_numpy(zero_copy_only=False)
        mask = (source == rule.get("source")) & (event_time < since) & is_null
        for position in np.flatnonzero(mask):
            entry = f"{name}={rule.get('reason', 'not_collected')}"
            if known_missing[int(position)] is None:
                known_missing[int(position)] = [entry]
            else:
                known_missing[int(position)].append(entry)

    # --- нормализованный текст ---

    def normalized(name: str) -> pa.Array:
        if name not in parsed.table.column_names:
            return pa.nulls(rows, pa.string())
        values = parsed.table.column(name).to_pylist()
        return pa.array([normalize_text(value) for value in values], pa.string())

    # --- сборка ---

    columns: dict[str, pa.Array | pa.ChunkedArray] = {
        name: batch.column(name) for name in ENVELOPE_NAMES
    }

    derived = {
        "client_idx": pa.array(client_idx),
        "stable_event_index": pa.array(stable_index),
        "before_window": pa.array(before_window),
        "at_or_after_extract": pa.array(after_extract),
        "ambiguous_local_time": pa.array(ambiguous),
        "balance_chain_gap": pa.array(chain_gap),
        "payload_status": pa.array(parsed.status, pa.string()),
        "payload_violations": pa.array(
            [parsed.violations.get(index) or None for index in range(rows)], pa.list_(pa.string())
        ),
        "known_missing": pa.array(known_missing, pa.list_(pa.string())),
        "merchant_name_norm": normalized("merchant_name"),
        "counterparty_norm": normalized("counterparty"),
        "raw_file": pa.array(["events.parquet"] * rows, pa.string()),
        "raw_row_group": batch.column("raw_row_group"),
        "raw_row": batch.column("raw_row"),
    }

    assert set(derived) == set(DERIVED_NAMES), "производные колонки разошлись со схемой"

    columns.update(derived)

    # parsed уже назвал колонки по-канонически: тип события
    # приехал из payload["type"] колонкой event_type.
    for name in payload_names:
        column = canonical_column(name)
        columns[column] = parsed.table.column(column)

    table = pa.table(columns).select(schema.names).cast(schema)

    counts = dict(parsed.counts)
    counts["rows"] = rows

    return BatchResult(table, parsed.rejects, clients, counts)


def canonical_schema(manifest: RawManifest) -> pa.Schema:
    return events_schema(manifest)


__all__ = [
    "SNAPSHOT_EVENT_TYPES",
    "CanonicalError",
    "TEXT_NORMALIZATION",
    "BatchResult",
    "balance_chain_gaps",
    "build_batch",
    "canonical_schema",
    "iter_client_batches",
    "normalize_text",
    "parse_batch",
]
