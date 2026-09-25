from __future__ import annotations

import pyarrow as pa

from ..projection import model_role
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


# 16 — колонка lifelong_source: пометка события-источника вехи.
SCHEMA_VERSION = 16

# Время события в слое УЖЕ нормализовано: в выгрузке это
# строка ISO 8601 со смещением, здесь — момент в UTC.
# Второго перевода пояса ниже по конвейеру нет.
TS_UTC = pa.timestamp("us", tz="UTC")


# ------------------------------------------------------------
# СОСТАВ СЛОЯ
# ------------------------------------------------------------
#
# Только конверт и смысловые поля событий. Производных
# и служебных колонок нет ни одной: ни внутреннего номера
# клиента, ни номера события, ни трассировки к RAW, ни
# признаков качества. Идентификаторов сущностей тоже нет:
# модель строит вектор клиента по событиям, их типам,
# значениям и времени, а не по точным связям между договорами
# и счетами.
#
# Нормализованный текст ложится в само поле: двух записей
# одного значения рядом не держим.
#
# Одна служебная колонка всё же есть — lifelong_source: тип вехи
# анкеты, чей источник записан этой строкой, иначе null. Её
# находит ссылка source_id вехи на карту или договор, пока
# идентификаторы ещё в payload. Модели она не отдаётся: по ней
# датасет только исключает строку из целей MLM.
# ------------------------------------------------------------

ENVELOPE_NAMES: tuple[str, ...] = tuple(name for name in ENVELOPE_SCHEMA.names if name != "payload")

LIFELONG_SOURCE_COLUMN = "lifelong_source"

# Поля, текст которых нормализуется на месте.
NORMALIZED_FIELDS: tuple[str, ...] = ("merchant_name", "counterparty")


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

    # В слой проходят только смысловые поля. Фильтр позитивный:
    # решает модельная проекция, а не список исключений здесь.
    return [
        (name, DTYPE_MAP[dtype])
        for name, dtype in columns.items()
        if model_role(name) == "semantic_field"
    ]


def events_schema(manifest: RawManifest) -> pa.Schema:

    fields: list[tuple[str, pa.DataType]] = [
        (name, TS_UTC if name == "event_time" else ENVELOPE_SCHEMA.field(name).type)
        for name in ENVELOPE_NAMES
    ]

    fields += payload_columns(manifest)

    fields.append((LIFELONG_SOURCE_COLUMN, pa.string()))

    return pa.schema(fields)


__all__ = [
    "ENVELOPE_NAMES",
    "LIFELONG_SOURCE_COLUMN",
    "NORMALIZED_FIELDS",
    "TS_UTC",
    "SCHEMA_VERSION",
    "events_schema",
    "payload_columns",
]
