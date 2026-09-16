from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.generator.config import EVENT_TYPES
from src.preprocessing.config import KIND_NUMERIC, FieldSpec, payload_fields
from src.preprocessing.stats import value_key

from .config import EVT_ID, MISSING_ID, UNK_ID, USR_ID
from .vocab import (
    KEY_FORMAT,
    NO_FIELD,
    FieldEntry,
    Vocab,
    event_key_specs,
    events_column,
    field_ids_by_key,
    profile_column,
    profile_key_specs,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Событие это четвёрки (field_id, key_id, value_id, position).
#
#   позиция 0   [EVT] одинаковым ID в key_ids и в value_ids
#   позиция 1   timeline__event_type
#   позиция 2+  поля payload в порядке реестра preprocessing
#
# field_id это идентичность ФИЗИЧЕСКОГО поля, одна и та же во
# всех режимах словаря; key_id и value_id это токены ЭТОГО
# режима. У специальной позиции поля нет: она несёт NO_FIELD.
#
# Порядок полей внутри записи задаётся field_id, а НЕ key_id:
# в semantic-режиме два поля делят один key token, и сортировка
# по нему переставила бы поля профиля. Раскладка записи обязана
# быть одинаковой во всех режимах, иначе арки сравнивали бы
# разные ленты.
#
# Позиция это номер значения ВНУТРИ события и сбрасывается на
# каждом событии; к позиции события в истории она отношения не
# имеет. Профиль кодируется теми же правилами, но отдельно и
# начинается с [USR].
#
# Два пути дают один результат: encode_pairs это эталон на
# списке пар (поддерживает повторяющиеся ключи и ключи вне
# реестра), encode_*_table это векторная реализация для
# широких таблиц processed. Совпадение проверяется тестом.
# ============================================================


KEY_IDS_TYPE = pa.list_(pa.int32())
VALUE_IDS_TYPE = pa.list_(pa.int32())
POSITIONS_TYPE = pa.list_(pa.int16())
FIELD_IDS_TYPE = pa.list_(pa.int16())

TS = pa.timestamp("us")

EVENT_FORMAT = {
    "lead": "[EVT]",
    "lead_position": 0,
    "event_type_position": 1,
    "field_positions": "поля payload в порядке реестра preprocessing, начиная с позиции 2",
    "position_scope": "позиция считается внутри события и сбрасывается на каждом событии",
    "limit": "лимит токенов события считается включая [EVT]",
}

PROFILE_FORMAT = {
    "lead": "[USR]",
    "lead_position": 0,
    "field_positions": "20 полей профиля в порядке PROFILE_FIELDS, начиная с позиции 1",
    "namespace": "profile: отдельное пространство ключей, не profile_snapshot",
}


def event_field_specs(event_type: str) -> list[FieldSpec]:
    """
    Поля payload типа события, которые попадают в поток.
    Metadata отсеивается.
    """

    by_key = {spec.field: spec for spec in event_key_specs() if spec.namespace == event_type}

    return [by_key[name] for name in payload_fields(event_type) if name in by_key]


EVENT_FIELDS: dict[str, list[FieldSpec]] = {
    event_type: event_field_specs(event_type) for event_type in EVENT_TYPES
}

# Ширина события: [EVT] + event_type + поля payload.
EVENT_WIDTH: dict[str, int] = {
    event_type: 2 + len(specs) for event_type, specs in EVENT_FIELDS.items()
}

PROFILE_SPECS: list[FieldSpec] = profile_key_specs()

# Ширина профиля: [USR] + 20 полей.
PROFILE_WIDTH = 1 + len(PROFILE_SPECS)

EVENT_TYPE_KEY = KEY_FORMAT.format(namespace="timeline", field="event_type")


# ------------------------------------------------------------
# РАСКЛАДКА ПОЛЕЙ ЗАПИСИ
# ------------------------------------------------------------
#
# field_id на каждой позиции записи выводится из реестра, а не
# из словаря: раскладка одинакова во всех четырёх режимах.
# Именно это делает маски и цели сравнимыми между арками.
# ------------------------------------------------------------

_FIELD_ID = field_ids_by_key()

EVENT_FIELD_IDS: dict[str, np.ndarray] = {
    event_type: np.array(
        [NO_FIELD, _FIELD_ID[EVENT_TYPE_KEY], *(_FIELD_ID[spec.column] for spec in specs)],
        dtype=np.int16,
    )
    for event_type, specs in EVENT_FIELDS.items()
}

PROFILE_FIELD_IDS: np.ndarray = np.array(
    [NO_FIELD, *(_FIELD_ID[spec.column] for spec in PROFILE_SPECS)],
    dtype=np.int16,
)


# ============================================================
# ЗАПИСЬ
# ============================================================


@dataclass(frozen=True)
class Record:
    """
    Одно событие или один профиль: четыре массива одной длины.
    """

    key_ids: np.ndarray
    value_ids: np.ndarray
    positions: np.ndarray
    field_ids: np.ndarray

    def __post_init__(self) -> None:
        if not (
            len(self.key_ids) == len(self.value_ids) == len(self.positions) == len(self.field_ids)
        ):
            raise ValueError("key_ids, value_ids, positions и field_ids должны быть одной длины")

    def __len__(self) -> int:
        return len(self.key_ids)

    @property
    def n_tokens(self) -> int:
        return len(self.key_ids)

def _record(
    key_ids: Sequence[int],
    value_ids: Sequence[int],
    field_ids: Sequence[int],
) -> Record:
    return Record(
        key_ids=np.asarray(key_ids, dtype=np.int32),
        value_ids=np.asarray(value_ids, dtype=np.int32),
        positions=np.arange(len(key_ids), dtype=np.int16),
        field_ids=np.asarray(field_ids, dtype=np.int16),
    )


# ============================================================
# ЭТАЛОН: СПИСОК ПАР
# ============================================================


def resolve_value(vocab: Vocab, entry: FieldEntry, value: Any) -> int:
    """
    Значение одного поля в value token.

    Поиск идёт ПО ПОЛЮ, а не по key token: в semantic-режиме
    два поля делят key token, но множества их значений разные, и
    поиск по общему ключу нашёл бы чужое значение.

    Пропуск это [MISSING], непустое значение вне frozen vocab
    это [UNK]; словарь не расширяется.
    """

    if value is None:
        return MISSING_ID

    if isinstance(value, float) and value != value:
        return MISSING_ID

    found = vocab.value_token(entry.field_id, value_key(value))

    return UNK_ID if found is None else found


def encode_pairs(vocab: Vocab, pairs: Iterable[tuple[str, Any]], lead: int) -> Record:
    """
    Эталонное кодирование списка пар (ключ, значение).

    Порядок берётся из реестра; сортировка стабильная, поэтому
    повторяющиеся ключи сохраняют исходный порядок значений и не
    схлопываются. Ключ вне реестра ставится после известных
    полей, в порядке ввода, и кодируется [UNK]/[UNK].
    """

    items = list(pairs)

    order: list[tuple[int, int]] = []

    for index, (key, _) in enumerate(items):
        entry = vocab.field_entry(key)
        # Ранг это field_id, а не key token: порядок записи не
        # имеет права зависеть от режима словаря.
        rank = entry.field_id if entry is not None else NO_FIELD + index
        order.append((rank, index))

    order.sort()

    key_ids = [lead]
    value_ids = [lead]
    field_ids = [NO_FIELD]

    for _, index in order:

        key, value = items[index]

        entry = vocab.field_entry(key)

        if entry is None:
            key_ids.append(UNK_ID)
            value_ids.append(UNK_ID)
            field_ids.append(NO_FIELD)
            continue

        key_ids.append(entry.key_token_id)
        value_ids.append(resolve_value(vocab, entry, value))
        field_ids.append(entry.field_id)

    return _record(key_ids, value_ids, field_ids)


def encode_event(vocab: Vocab, event_type: str, pairs: Iterable[tuple[str, Any]]) -> Record:
    """
    Событие: [EVT], затем event_type, затем поля payload.
    """

    return encode_pairs(vocab, [(EVENT_TYPE_KEY, event_type), *pairs], EVT_ID)


def encode_profile(vocab: Vocab, pairs: Iterable[tuple[str, Any]]) -> Record:
    return encode_pairs(vocab, pairs, USR_ID)


# ------------------------------------------------------------
# ПАРЫ ИЗ СТРОКИ PROCESSED
# ------------------------------------------------------------


def event_pairs(event_type: str, row: dict) -> list[tuple[str, Any]]:
    """
    Пары (ключ, значение) строки широкой таблицы событий.
    Numeric берётся уже bucketized.
    """

    return [(spec.column, row.get(events_column(spec))) for spec in EVENT_FIELDS[event_type]]


def profile_pairs(row: dict) -> list[tuple[str, Any]]:
    return [(spec.column, row.get(profile_column(spec))) for spec in PROFILE_SPECS]


# ============================================================
# ВЕКТОРНЫЙ ПУТЬ
# ============================================================


def tokenized_events_schema() -> pa.Schema:
    return pa.schema(
        [
            ("client_id", pa.int64()),
            ("seq", pa.int64()),
            ("ts", TS),
            ("event_type", pa.string()),
            ("n_tokens", pa.int16()),
            ("key_ids", KEY_IDS_TYPE),
            ("value_ids", VALUE_IDS_TYPE),
            ("positions", POSITIONS_TYPE),
            ("field_ids", FIELD_IDS_TYPE),
        ]
    )


def tokenized_profile_schema() -> pa.Schema:
    return pa.schema(
        [
            ("client_id", pa.int64()),
            ("ts", TS),
            ("snapshot_month", TS),
            ("n_tokens", pa.int16()),
            ("key_ids", KEY_IDS_TYPE),
            ("value_ids", VALUE_IDS_TYPE),
            ("positions", POSITIONS_TYPE),
            ("field_ids", FIELD_IDS_TYPE),
        ]
    )


def column_value_ids(vocab: Vocab, entry: FieldEntry, column) -> np.ndarray:
    """
    Колонка значений одного поля в value token.

    Пропуск это [MISSING], значение вне словаря это [UNK].

    Numeric приходит колонкой __bucket и переводится арифметикой:
    корзины не делятся между полями ни в одном режиме, поэтому
    диапазон поля непрерывен, и это проверено в _check_layout.

    У остального index_in даёт ЛОКАЛЬНЫЙ индекс в порядке поля,
    а token берётся выборкой из кандидатов: в shared-режиме
    кандидаты поля не обязаны идти подряд.
    """

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()

    length = len(column)

    null_mask = pc.is_null(column).to_numpy(zero_copy_only=False)

    if entry.n_values == 0:
        return np.where(null_mask, MISSING_ID, UNK_ID).astype(np.int32)

    if entry.kind == KIND_NUMERIC:

        buckets = pc.fill_null(column, -1).to_numpy(zero_copy_only=False).astype(np.int64)

        inside = (buckets >= 0) & (buckets < entry.n_values)

        ids = np.where(inside, entry.value_start + buckets, UNK_ID)

    else:

        index = pc.index_in(column, value_set=vocab.typed_values(entry.field_id))

        found = pc.fill_null(index, -1).to_numpy(zero_copy_only=False).astype(np.int64)

        candidates = np.asarray(entry.candidates, dtype=np.int64)

        ids = np.where(found >= 0, candidates[np.clip(found, 0, entry.n_values - 1)], UNK_ID)

    result = np.where(null_mask, MISSING_ID, ids).astype(np.int32)

    assert len(result) == length

    return result


def _list_column(flat: np.ndarray, offsets: np.ndarray, value_type: pa.DataType) -> pa.ListArray:
    return pa.ListArray.from_arrays(
        pa.array(offsets, pa.int32()),
        pa.array(flat, value_type),
    )


def encode_events_table(table: pa.Table, vocab: Vocab) -> pa.Table:
    """
    Батч широкой таблицы событий в токены.
    """

    schema = tokenized_events_schema()

    length = table.num_rows

    if length == 0:
        return schema.empty_table()

    event_type_column = table.column("event_type").combine_chunks()

    codes = pc.index_in(event_type_column, value_set=pa.array(list(EVENT_TYPES), pa.string()))

    codes_np = pc.fill_null(codes, -1).to_numpy(zero_copy_only=False).astype(np.int64)

    if (codes_np < 0).any():
        unknown = sorted({event_type_column[int(i)].as_py() for i in np.flatnonzero(codes_np < 0)[:5]})
        raise ValueError(f"неизвестный тип события в processed: {unknown}")

    widths = np.array([EVENT_WIDTH[name] for name in EVENT_TYPES], dtype=np.int64)[codes_np]

    offsets = np.zeros(length + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    total = int(offsets[-1])

    flat_keys = np.zeros(total, dtype=np.int32)
    flat_values = np.zeros(total, dtype=np.int32)
    flat_positions = np.zeros(total, dtype=np.int16)
    flat_fields = np.zeros(total, dtype=np.int16)

    event_type_entry = vocab.field_entry(EVENT_TYPE_KEY)

    for code, event_type in enumerate(EVENT_TYPES):

        rows = np.flatnonzero(codes_np == code)

        if rows.size == 0:
            continue

        specs = EVENT_FIELDS[event_type]
        width = EVENT_WIDTH[event_type]

        typed = table.take(pa.array(rows))

        places = offsets[rows][:, None] + np.arange(width, dtype=np.int64)[None, :]

        key_row = np.empty(width, dtype=np.int32)
        key_row[0] = EVT_ID
        key_row[1] = event_type_entry.key_token_id

        field_row = EVENT_FIELD_IDS[event_type]

        value_matrix = np.empty((rows.size, width), dtype=np.int32)
        value_matrix[:, 0] = EVT_ID
        value_matrix[:, 1] = column_value_ids(vocab, event_type_entry, typed.column("event_type"))

        for offset, spec in enumerate(specs):

            entry = vocab.field_entry(spec.column)

            key_row[offset + 2] = entry.key_token_id
            value_matrix[:, offset + 2] = column_value_ids(vocab, entry, typed.column(events_column(spec)))

            # Раскладка выведена из реестра до словаря: сверяем,
            # что словарь согласен, а не верим на слово.
            assert field_row[offset + 2] == entry.field_id

        flat_keys[places] = key_row
        flat_values[places] = value_matrix
        flat_positions[places] = np.arange(width, dtype=np.int16)
        flat_fields[places] = field_row

    return pa.table(
        {
            "client_id": table.column("client_id").combine_chunks(),
            "seq": table.column("seq").combine_chunks(),
            "ts": table.column("ts").combine_chunks(),
            "event_type": event_type_column,
            "n_tokens": pa.array(widths.astype(np.int16), pa.int16()),
            "key_ids": _list_column(flat_keys, offsets, pa.int32()),
            "value_ids": _list_column(flat_values, offsets, pa.int32()),
            "positions": _list_column(flat_positions, offsets, pa.int16()),
            "field_ids": _list_column(flat_fields, offsets, pa.int16()),
        },
        schema=schema,
    )


def encode_profile_table(table: pa.Table, vocab: Vocab) -> pa.Table:
    """
    Снимки профиля в токены: фиксированная ширина.
    """

    schema = tokenized_profile_schema()

    length = table.num_rows

    if length == 0:
        return schema.empty_table()

    width = PROFILE_WIDTH

    offsets = (np.arange(length + 1, dtype=np.int64) * width)

    flat_keys = np.zeros(length * width, dtype=np.int32)
    flat_values = np.zeros(length * width, dtype=np.int32)

    key_row = np.empty(width, dtype=np.int32)
    key_row[0] = USR_ID

    value_matrix = np.empty((length, width), dtype=np.int32)
    value_matrix[:, 0] = USR_ID

    for offset, spec in enumerate(PROFILE_SPECS):

        entry = vocab.field_entry(spec.column)

        key_row[offset + 1] = entry.key_token_id
        value_matrix[:, offset + 1] = column_value_ids(vocab, entry, table.column(profile_column(spec)))

        assert PROFILE_FIELD_IDS[offset + 1] == entry.field_id

    flat_keys[:] = np.tile(key_row, length)
    flat_values[:] = value_matrix.reshape(-1)

    flat_positions = np.tile(np.arange(width, dtype=np.int16), length)
    flat_fields = np.tile(PROFILE_FIELD_IDS, length)

    return pa.table(
        {
            "client_id": table.column("client_id").combine_chunks(),
            "ts": table.column("ts").combine_chunks(),
            "snapshot_month": table.column("snapshot_month").combine_chunks(),
            "n_tokens": pa.array(np.full(length, width, dtype=np.int16), pa.int16()),
            "key_ids": _list_column(flat_keys, offsets, pa.int32()),
            "value_ids": _list_column(flat_values, offsets, pa.int32()),
            "positions": _list_column(flat_positions, offsets, pa.int16()),
            "field_ids": _list_column(flat_fields, offsets, pa.int16()),
        },
        schema=schema,
    )
