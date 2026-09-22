from __future__ import annotations

import pyarrow as pa

from ..rawdata import DTYPE_MAP, ENVELOPE_SCHEMA, RawManifest


# ============================================================
# ИДЕЯ
# ============================================================
#
# Схемы слоя canonical. Одна строка canonical это одна строка
# RAW: ни одна не исчезает, дубли и конфликты остаются с
# пометкой, неразобранный payload остаётся с причиной.
#
# У записи одно время — время события. Времени поступления в
# хранилище у выгрузки нет, поэтому нет и полей, выведенных из
# него: задержек, опозданий, момента закрытия версии.
#
# Колонки payload называются по имени поля, а принадлежность
# даёт event_type строки: dtype одного имени совпадает во всех
# типах событий (проверяется, а не предполагается), поэтому
# физические поля всех типов укладываются в общий набор колонок.
# ============================================================


SCHEMA_VERSION = 9

TS = pa.timestamp("us")

PAYLOAD_OK = "ok"
PAYLOAD_UNPARSEABLE = "unparseable"
PAYLOAD_NULL = "null_payload"


# ------------------------------------------------------------
# ПРОИЗВОДНЫЕ КОЛОНКИ
# ------------------------------------------------------------
#
# (имя, тип, dtype реестра, описание). Порядок фиксирован.
# ------------------------------------------------------------

DERIVED_COLUMNS: tuple[tuple[str, pa.DataType, str, str], ...] = (
    ("client_idx", pa.int64(), "int", "плотный внутренний индекс клиента внутри группы"),
    (
        "stable_event_index",
        pa.int64(),
        "int",
        "номер логического события клиента по (event_time, приоритет типа, event_id)",
    ),
    (
        "is_repeated_event_id",
        pa.bool_(),
        "bool",
        "event_id этой строки уже встречался в выгрузке: запись обязана приходить один раз",
    ),
    (
        "before_window",
        pa.bool_(),
        "bool",
        "событие произошло раньше period_start выгрузки: по контракту таких строк нет, "
        "и колонка это проверка, а не описание",
    ),
    ("at_or_after_extract", pa.bool_(), "bool", "событие произошло на границе period_end или позже"),
    ("ambiguous_local_time", pa.bool_(), "bool", "местное время попадает в объявленный неоднозначный интервал"),
    (
        "balance_chain_gap",
        pa.bool_(),
        "bool",
        "остаток счёта не продолжает предыдущий наблюдаемый остаток: "
        "между строками потеряно движение денег",
    ),
    ("payload_status", pa.string(), "str", "ok, unparseable или null_payload"),
    (
        "payload_violations",
        pa.list_(pa.string()),
        "str",
        "нарушения контракта payload в этой строке: вид и поле",
    ),
    (
        "known_missing",
        pa.list_(pa.string()),
        "str",
        "пропуски с известной причиной по датированному правилу схемы",
    ),
    ("merchant_name_norm", pa.string(), "str", "нормализованная копия merchant_name"),
    ("counterparty_norm", pa.string(), "str", "нормализованная копия counterparty"),
    ("raw_file", pa.string(), "str", "файл RAW, из которого взята строка"),
    ("raw_row_group", pa.int32(), "int", "номер row group в файле RAW"),
    ("raw_row", pa.int64(), "int", "номер строки в файле RAW"),
)

DERIVED_NAMES: tuple[str, ...] = tuple(name for name, _, _, _ in DERIVED_COLUMNS)

ENVELOPE_NAMES: tuple[str, ...] = tuple(name for name in ENVELOPE_SCHEMA.names if name != "payload")


def payload_columns(manifest: RawManifest) -> list[tuple[str, pa.DataType]]:
    """
    Колонки payload в порядке первого появления при обходе типов
    событий каталога. Расхождение dtype у одного имени это ошибка
    контракта, а не повод придумать общий тип.
    """

    columns: dict[str, str] = {}

    for info in manifest.catalogue.values():
        for item in info.fields:
            known = columns.get(item.name)
            if known is None:
                columns[item.name] = item.dtype
            elif known != item.dtype:
                raise ValueError(
                    f"поле {item.name} объявлено и как {known}, и как {item.dtype}: "
                    "общая колонка невозможна"
                )

    return [(name, DTYPE_MAP[dtype]) for name, dtype in columns.items()]


def events_schema(manifest: RawManifest) -> pa.Schema:

    fields: list[tuple[str, pa.DataType]] = [
        (name, ENVELOPE_SCHEMA.field(name).type) for name in ENVELOPE_NAMES
    ]

    fields += [(name, field_type) for name, field_type, _, _ in DERIVED_COLUMNS]

    fields += payload_columns(manifest)

    return pa.schema(fields)


def profile_schema() -> pa.Schema:

    from src.generator.profile import PROFILE_SCHEMA

    fields = [(field.name, field.type) for field in PROFILE_SCHEMA]

    fields += [
        ("client_idx", pa.int64()),
        ("raw_file", pa.string()),
        ("raw_row_group", pa.int32()),
        ("raw_row", pa.int64()),
    ]

    return pa.schema(fields)


# Адрес клиента в файле. Два разных числа, и путать их нельзя:
# row_offset отсчитывается внутри своего row group, а
# global_row_start — от начала файла.
CLIENT_INDEX_SCHEMA = pa.schema(
    [
        ("client_idx", pa.int64()),
        ("client_id", pa.string()),
        ("row_group", pa.int32()),
        ("row_offset", pa.int64()),
        ("global_row_start", pa.int64()),
        ("row_count", pa.int64()),
        ("spans_row_groups", pa.bool_()),
        ("event_time_min", TS),
        ("event_time_max", TS),
    ]
)


MENTIONS_SCHEMA = pa.schema(
    [
        ("entity_kind", pa.string()),
        ("entity_id", pa.string()),
        ("client_idx", pa.int64()),
        ("client_id", pa.string()),
        ("event_id", pa.string()),
        ("stable_event_index", pa.int64()),
        ("event_time", TS),
        ("event_type", pa.string()),
        ("source", pa.string()),
        ("field_name", pa.string()),
        ("is_transition", pa.bool_()),
        ("transition", pa.string()),
        ("raw_row", pa.int64()),
    ]
)


# Стороны перевода как они наблюдаются. Парности и контрагента
# здесь нет намеренно: кто с кем сошёлся на дату, решает этап
# истории среди видимых строк.
#
# transfer_id приходит из payload: в конверте связи нет, вид её
# задаёт имя ключа.
TRANSFERS_SCHEMA = pa.schema(
    [
        ("transfer_id", pa.string()),
        ("side", pa.string()),
        ("client_idx", pa.int64()),
        ("client_id", pa.string()),
        ("event_id", pa.string()),
        ("event_type", pa.string()),
        ("stable_event_index", pa.int64()),
        ("event_time", TS),
        ("amount", pa.int64()),
        ("direction", pa.string()),
        ("status", pa.string()),
        ("counterparty", pa.string()),
        ("raw_row", pa.int64()),
    ]
)


# Повторы идентификатора: запись обязана приходить в выгрузку
# ровно один раз, и повтор это поломка контракта, а не дефект
# доставки. Строка сохраняется с пометкой, чтобы расхождение
# было видно, а не исчезло молча.
REPEATS_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("client_id", pa.string()),
        ("raw_row", pa.int64()),
        ("first_raw_row", pa.int64()),
        ("reason", pa.string()),
    ]
)


REJECTS_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("client_id", pa.string()),
        ("event_type", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
        ("payload", pa.string()),
        ("raw_file", pa.string()),
        ("raw_row_group", pa.int32()),
        ("raw_row", pa.int64()),
    ]
)


__all__ = [
    "CLIENT_INDEX_SCHEMA",
    "DERIVED_COLUMNS",
    "DERIVED_NAMES",
    "ENVELOPE_NAMES",
    "MENTIONS_SCHEMA",
    "PAYLOAD_NULL",
    "PAYLOAD_OK",
    "PAYLOAD_UNPARSEABLE",
    "REJECTS_SCHEMA",
    "REPEATS_SCHEMA",
    "SCHEMA_VERSION",
    "TRANSFERS_SCHEMA",
    "events_schema",
    "payload_columns",
    "profile_schema",
]
