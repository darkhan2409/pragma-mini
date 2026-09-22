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
# строки даёт её тип — ключ type: dtype одного имени совпадает
# во всех типах событий (проверяется, а не предполагается),
# поэтому физические поля всех типов укладываются в общий набор
# колонок. Переименований между выгрузкой и слоем нет: как поле
# названо в payload, так называется и колонка.
# ============================================================


SCHEMA_VERSION = 13

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
        "номер логического события клиента по (event_time, приоритет типа, номер строки RAW)",
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


__all__ = [
    "DERIVED_COLUMNS",
    "DERIVED_NAMES",
    "ENVELOPE_NAMES",
    "SCHEMA_VERSION",
    "events_schema",
    "payload_columns",
]
