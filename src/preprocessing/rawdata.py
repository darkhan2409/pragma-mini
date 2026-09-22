from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pj
import pyarrow.parquet as pq

from src.generator.config import (
    EVENT_TYPE_PRIORITY,
    SCHEMA_CHANGES,
    SOURCE_LAUNCH,
    key_catalogue,
)
from src.generator.profile import PROFILE_SCHEMA

from .artifacts import sha256_file


# ============================================================
# ИДЕЯ
# ============================================================
#
# Доступ к RAW v10 без pandas и без загрузки всего датасета:
# лента читается по одному row group, payload разбирается
# pyarrow.json по каталогу ключей.
#
# Манифест это технический паспорт выгрузки: версия контракта,
# окно, число строк и sha256 двух основных файлов. Всё это
# здесь сверяется.
#
# Статическая часть контракта — каталог ключей payload,
# приоритет типов событий, даты запуска источников, смены
# схемы — живёт в config генератора и читается из кода: копии
# рядом с каждой выгрузкой у неё больше нет.
#
# Справочники мерчантов, продуктов и географии в выгрузку тоже
# не попадают. Событие несёт поля выбранного объекта, и
# расшифровывать по справочнику больше нечего.
# ============================================================


MANIFEST_NAME = "manifest.json"

EXPECTED_SCHEMA_VERSION = 10

# Файлы, без которых группа не обрабатывается. Справочников
# рядом с выгрузкой нет: они остались входом генератора.
REQUIRED_FILES: tuple[str, ...] = (
    "events.parquet",
    "profile.parquet",
)

TABLE_FILES: dict[str, str] = {
    "events": "events.parquet",
    "profile": "profile.parquet",
}

# Таблицы выгрузки: их подписывает манифест и читает
# препроцессинг. Других в выгрузке нет.
MAIN_TABLES: tuple[str, ...] = ("events", "profile")

REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "schema_version",
    "period_start",
    "period_end",
    "events_rows",
    "profile_rows",
    "events_sha256",
    "profile_sha256",
)

ENVELOPE_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("client_id", pa.string()),
        ("event_type", pa.string()),
        ("source", pa.string()),
        ("event_time", pa.timestamp("us")),
        ("payload", pa.string()),
    ]
)

EXPECTED_SCHEMAS: dict[str, pa.Schema] = {
    "events": ENVELOPE_SCHEMA,
    "profile": PROFILE_SCHEMA,
}

# dtype каталога ключей -> тип pyarrow.
DTYPE_MAP: dict[str, pa.DataType] = {
    "str": pa.string(),
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
}

# Какие типы Python допустимы в значении поля каждого dtype
# (медленный путь разбора). bool исключается из int: JSON true
# в числовом поле это ошибка типа, а не единица.
PY_TYPES: dict[str, tuple[type, ...]] = {
    "str": (str,),
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
}


class RawContractError(ValueError):
    """
    RAW не соответствует контракту: нет файла, ключа манифеста,
    неверная версия схемы.
    """


# ============================================================
# МАНИФЕСТ
# ============================================================


@dataclass(frozen=True)
class FieldInfo:
    name: str
    dtype: str
    nullable: bool
    level: str
    description: str

    @property
    def arrow_type(self) -> pa.DataType:
        if self.dtype not in DTYPE_MAP:
            raise RawContractError(f"неизвестный dtype каталога ключей: {self.dtype!r} у поля {self.name}")
        return DTYPE_MAP[self.dtype]


@dataclass(frozen=True)
class EventTypeInfo:
    event_type: str
    source: str
    description: str
    fields: tuple[FieldInfo, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.fields)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.fields if not item.nullable)

    def schema(self) -> pa.Schema:
        return pa.schema([(item.name, item.arrow_type) for item in self.fields])


@dataclass(frozen=True)
class SourceInfo:
    source: str
    available_from: datetime


@dataclass(frozen=True)
class RawManifest:
    """
    Технический паспорт выгрузки: версия контракта, окно, число
    строк и sha256 двух основных файлов.

    Статическая часть контракта сюда не копируется, а берётся из
    кода генератора: каталог ключей payload, приоритет типов
    событий, даты запуска источников и смены схемы.
    """

    schema_version: int
    period_start: datetime
    period_end: datetime
    events_rows: int
    profile_rows: int
    events_sha256: str
    profile_sha256: str
    sha256: str

    @property
    def catalogue(self) -> dict[str, EventTypeInfo]:
        return _static_catalogue()

    @property
    def event_types(self) -> tuple[str, ...]:
        return tuple(self.catalogue)

    @property
    def event_type_priority(self) -> dict[str, int]:
        return dict(EVENT_TYPE_PRIORITY)

    @property
    def sources(self) -> dict[str, SourceInfo]:
        """
        Источник доступен с даты запуска своей системы, а тот,
        что существует с начала наблюдения, — с начала окна.
        """

        return {
            name: SourceInfo(
                source=name,
                available_from=self.period_start if launch is None else launch,
            )
            for name, launch in SOURCE_LAUNCH.items()
        }

    @property
    def schema_changes(self) -> tuple[dict, ...]:
        return tuple(dict(item) for item in SCHEMA_CHANGES)

    @property
    def rows(self) -> dict[str, int]:
        return {"events": self.events_rows, "profile": self.profile_rows}

    @property
    def file_sha256(self) -> dict[str, str]:
        return {
            TABLE_FILES["events"]: self.events_sha256,
            TABLE_FILES["profile"]: self.profile_sha256,
        }

    def echo(self) -> dict:
        """
        Манифест целиком: он и так короткий.
        """

        return {
            "schema_version": self.schema_version,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "events_rows": self.events_rows,
            "profile_rows": self.profile_rows,
            "events_sha256": self.events_sha256,
            "profile_sha256": self.profile_sha256,
            "manifest_sha256": self.sha256,
        }


def _parse_catalogue(data: dict) -> dict[str, EventTypeInfo]:

    catalogue: dict[str, EventTypeInfo] = {}

    for event_type, spec in data.items():

        fields = tuple(
            FieldInfo(
                name=str(item["name"]),
                dtype=str(item["dtype"]),
                nullable=bool(item["nullable"]),
                level=str(item.get("level", "")),
                description=str(item.get("description", "")),
            )
            for item in spec["fields"]
        )

        names = [item.name for item in fields]
        if len(names) != len(set(names)):
            raise RawContractError(f"каталог ключей {event_type}: повторяющиеся имена полей")

        for item in fields:
            item.arrow_type  # проверка dtype

        catalogue[event_type] = EventTypeInfo(
            event_type=event_type,
            source=str(spec["source"]),
            description=str(spec.get("description", "")),
            fields=fields,
        )

    return catalogue


_CATALOGUE: dict[str, EventTypeInfo] = {}


def _static_catalogue() -> dict[str, EventTypeInfo]:
    """
    Каталог ключей payload из config генератора. Разбирается
    один раз: он статичен и от выгрузки не зависит.
    """

    if not _CATALOGUE:
        _CATALOGUE.update(_parse_catalogue(key_catalogue()))

    return _CATALOGUE


def read_manifest(raw_dir: Path) -> RawManifest:

    path = Path(raw_dir) / MANIFEST_NAME

    if not path.exists():
        raise RawContractError(f"нет {MANIFEST_NAME} в {raw_dir}")

    # Битый или нечитаемый манифест это тоже нарушение контракта
    # RAW, а не поломка препроцессинга: паспорт обязан ответить
    # «данные непригодны», а не упасть с чужим исключением.
    try:
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawContractError(f"{MANIFEST_NAME} нечитаем: {error}") from error

    if not isinstance(data, dict):
        raise RawContractError(f"{MANIFEST_NAME} не объект: {type(data).__name__}")

    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in data]
    if missing:
        raise RawContractError(f"в {MANIFEST_NAME} нет ключей: {missing}")

    schema_version = int(data["schema_version"])
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise RawContractError(
            f"schema_version манифеста {schema_version}, поддерживается {EXPECTED_SCHEMA_VERSION}"
        )

    period_start = datetime.fromisoformat(data["period_start"])
    period_end = datetime.fromisoformat(data["period_end"])

    if period_start >= period_end:
        raise RawContractError(
            f"окно выгрузки пусто: period_start {period_start} не раньше period_end {period_end}"
        )

    return RawManifest(
        schema_version=schema_version,
        period_start=period_start,
        period_end=period_end,
        events_rows=int(data["events_rows"]),
        profile_rows=int(data["profile_rows"]),
        events_sha256=str(data["events_sha256"]),
        profile_sha256=str(data["profile_sha256"]),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


# ============================================================
# КОНТРОЛЬНАЯ СУММА СОДЕРЖИМОГО
# ============================================================
#
# Сумма blake2b-отпечатков строк по модулю 2**128 плюс число
# строк: порядок строк не важен. Нужна там, где сравниваются
# НАБОРЫ строк, а не файлы: каталоги мира у разных групп и
# корпус срезов.
# ============================================================


class ContentDigest:

    MODULUS = 2 ** 128

    def __init__(self) -> None:
        self.total = 0
        self.rows = 0

    def add(self, row: dict) -> None:
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
        self.total = (self.total + int.from_bytes(digest, "little")) % self.MODULUS
        self.rows += 1

    def extend(self, rows: list) -> None:
        for row in rows:
            self.add(row)

    def value(self) -> str:
        return hashlib.sha256(f"{self.total}:{self.rows}".encode("utf-8")).hexdigest()


# ============================================================
# ДАТАСЕТ
# ============================================================


class RawDataset:
    """
    Ленивый доступ к таблицам RAW. При создании читается только
    манифест.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.manifest = read_manifest(self.raw_dir)

    # --- файлы ---

    def path(self, table: str) -> Path:
        return self.raw_dir / TABLE_FILES[table]

    def exists(self, table: str) -> bool:
        return self.path(table).exists()

    def missing_required_files(self) -> list[str]:
        return [name for name in REQUIRED_FILES if not (self.raw_dir / name).exists()]

    def listed_files(self) -> list[str]:
        """
        Все parquet-файлы каталога относительными путями с прямыми
        слэшами.
        """

        return sorted(
            path.relative_to(self.raw_dir).as_posix()
            for path in self.raw_dir.rglob("*.parquet")
        )

    # --- таблицы ---

    def schema(self, table: str) -> pa.Schema:
        return pq.read_schema(self.path(table)).remove_metadata()

    def metadata(self, table: str) -> pq.FileMetaData:
        return pq.read_metadata(self.path(table))

    def num_rows(self, table: str) -> int:
        return int(self.metadata(table).num_rows)

    def parquet_file(self, table: str) -> pq.ParquetFile:
        return pq.ParquetFile(self.path(table))

    def read(self, table: str, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.path(table), columns=columns)

    def iter_row_groups(self, table: str, columns: list[str] | None = None) -> Iterator[tuple[int, pa.Table]]:
        """
        Таблица по одному row group в порядке файла. Память
        ограничена размером row group, а не файла.
        """

        parquet = self.parquet_file(table)

        for index in range(parquet.num_row_groups):
            yield index, parquet.read_row_group(index, columns=columns)

    # --- сверка с манифестом ---

    def verify_files(self) -> dict[str, dict]:
        """
        sha256 двух основных файлов: манифест подписывает только
        их. Каталоги мира сверяет отдельная проверка корпуса.
        """

        result: dict[str, dict] = {}

        for name, expected in sorted(self.manifest.file_sha256.items()):

            path = self.raw_dir / name

            if not path.exists():
                result[name] = {"status": "missing", "expected": expected, "actual": None}
                continue

            actual = sha256_file(path)

            result[name] = {
                "status": "ok" if actual == expected else "mismatch",
                "expected": expected,
                "actual": actual,
            }

        return result

    def verify_rows(self) -> dict[str, dict]:

        result: dict[str, dict] = {}

        for table in MAIN_TABLES:

            expected = self.manifest.rows.get(table)

            if not self.exists(table):
                result[table] = {"status": "missing", "expected": expected, "actual": None}
                continue

            try:
                actual = self.num_rows(table)
            except Exception as error:  # noqa: BLE001 — повреждённый файл это результат проверки
                result[table] = {"status": "unreadable", "expected": expected, "actual": None, "error": type(error).__name__}
                continue

            result[table] = {
                "status": "ok" if expected == actual else "mismatch",
                "expected": expected,
                "actual": actual,
            }

        return result

    def verify_schemas(self) -> dict[str, dict]:

        result: dict[str, dict] = {}

        for table, expected in EXPECTED_SCHEMAS.items():

            if not self.exists(table):
                result[table] = {"status": "missing", "differences": []}
                continue

            try:
                actual = self.schema(table)
            except Exception as error:  # noqa: BLE001
                result[table] = {"status": "unreadable", "differences": [type(error).__name__]}
                continue

            differences = schema_differences(expected, actual)

            result[table] = {"status": "ok" if not differences else "mismatch", "differences": differences}

        return result


def schema_differences(expected: pa.Schema, actual: pa.Schema) -> list[str]:

    differences: list[str] = []

    expected_names = expected.names
    actual_names = actual.names

    for name in expected_names:
        if name not in actual_names:
            differences.append(f"нет колонки {name}")
        elif not expected.field(name).type.equals(actual.field(name).type):
            differences.append(
                f"{name}: ожидался {expected.field(name).type}, получен {actual.field(name).type}"
            )

    for name in actual_names:
        if name not in expected_names:
            differences.append(f"лишняя колонка {name}")

    if not differences and expected_names != actual_names:
        differences.append("другой порядок колонок")

    return differences


# ============================================================
# РАЗБОР PAYLOAD
# ============================================================


@dataclass
class ParsedPayload:
    """
    Типизированная таблица полей одного типа события плюс
    нарушения контракта. Строка с нарушением не выбрасывается:
    лишний ключ отбрасывается, значение не того типа становится
    null, и каждое такое решение записано.
    """

    table: pa.Table
    counts: Counter = field(default_factory=Counter)
    # Нарушения по конкретному полю: ключ "вид:поле". Нужен, чтобы
    # отчёт называл поле, а не только вид нарушения.
    by_field: Counter = field(default_factory=Counter)
    # Нарушения по строке: canonical обязан объяснить каждую
    # обнулённую или отброшенную ячейку, а не только их число.
    by_row: dict = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    # Ячейки, уже обнулённые из-за несовпадения типа: пропуском
    # обязательного ключа они не считаются второй раз.
    nulled: set = field(default_factory=set)

    SAMPLE_LIMIT = 20

    def record(self, kind: str, row: int, field_name: str | None, detail: str = "") -> None:
        self.counts[kind] += 1
        self.by_field[f"{kind}:{field_name}"] += 1
        self.by_row.setdefault(int(row), []).append(f"{kind}:{field_name}")
        if kind == "type_mismatch":
            self.nulled.add((int(row), field_name))
        if len(self.samples) < self.SAMPLE_LIMIT:
            self.samples.append({"kind": kind, "row": int(row), "field": field_name, "detail": detail})


def _joined_buffer(payload: pa.Array | pa.ChunkedArray) -> pa.Buffer:
    """
    Все payload одним NDJSON-буфером без прохода по Python.
    """

    if isinstance(payload, pa.ChunkedArray):
        payload = payload.combine_chunks()

    offsets = pa.array([0, len(payload)], pa.int32())
    as_list = pa.ListArray.from_arrays(offsets, payload)
    joined = pc.binary_join(as_list, "\n")

    return joined[0].as_buffer()


def _fast_parse(schema: pa.Schema, payload: pa.Array | pa.ChunkedArray) -> pa.Table:

    parsed = pj.read_json(
        pa.BufferReader(_joined_buffer(payload)),
        parse_options=pj.ParseOptions(explicit_schema=schema, unexpected_field_behavior="error"),
    )

    if parsed.num_rows != len(payload):
        raise pa.ArrowInvalid(f"разобрано {parsed.num_rows} строк из {len(payload)}")

    return parsed.select(schema.names).combine_chunks()


def _slow_parse(info: EventTypeInfo, payload: pa.Array | pa.ChunkedArray, result: ParsedPayload) -> pa.Table:
    """
    Построчный разбор, когда быстрый путь отказал: ищет и
    записывает конкретные нарушения.
    """

    by_name = {item.name: item for item in info.fields}

    cleaned: list[dict] = []

    for row, text in enumerate(payload.to_pylist()):

        record: dict[str, Any] = {}

        if text is None:
            result.record("null_payload", row, None)
            cleaned.append(record)
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            result.record("unparseable", row, None, str(error)[:120])
            cleaned.append(record)
            continue

        if not isinstance(data, dict):
            result.record("unparseable", row, None, "payload не объект")
            cleaned.append(record)
            continue

        for key, value in data.items():

            spec = by_name.get(key)

            if spec is None:
                result.record("unexpected_key", row, key)
                continue

            if value is None:
                record[key] = None
                continue

            if not isinstance(value, PY_TYPES[spec.dtype]) or (spec.dtype != "bool" and isinstance(value, bool)):
                result.record("type_mismatch", row, key, f"{type(value).__name__} вместо {spec.dtype}")
                record[key] = None
                continue

            record[key] = value

        cleaned.append(record)

    return pa.Table.from_pylist(cleaned, schema=info.schema())


def parse_payloads(info: EventTypeInfo, payload: pa.Array | pa.ChunkedArray) -> ParsedPayload:
    """
    Все строки одного типа события. Отсутствующий необязательный
    ключ — допустимый пропуск; отсутствующий обязательный —
    нарушение missing_required; лишний ключ и значение не того
    типа — нарушения, строка сохраняется.
    """

    schema = info.schema()

    if len(payload) == 0:
        return ParsedPayload(schema.empty_table())

    result = ParsedPayload(schema.empty_table())

    try:
        table = _fast_parse(schema, payload)
    except pa.ArrowInvalid:
        table = _slow_parse(info, payload, result)

    result.table = table

    for name in info.required:
        nulls = pc.is_null(table.column(name))
        if not int(pc.sum(nulls).as_py() or 0):
            continue
        positions = [
            int(row) for row in pc.indices_nonzero(nulls).to_pylist() if (int(row), name) not in result.nulled
        ]
        # Образцы ограничены, а журнал по строкам ведётся целиком:
        # каждая пустая обязательная ячейка должна быть объяснима.
        for position, row in enumerate(positions):
            if position < ParsedPayload.SAMPLE_LIMIT:
                result.record("missing_required", row, name)
            else:
                result.counts["missing_required"] += 1
                result.by_field[f"missing_required:{name}"] += 1
                result.by_row.setdefault(int(row), []).append(f"missing_required:{name}")

    return result


def iter_event_types(table: pa.Table) -> Iterator[tuple[str, pa.Table]]:
    """
    Строки row group по типам событий, в порядке первого
    появления типа. Порядок строк внутри типа сохраняется.
    """

    types = table.column("event_type")

    seen: list[str] = []
    for value in pc.unique(types).to_pylist():
        seen.append(value)

    for event_type in seen:
        mask = pc.equal(types, event_type)
        yield event_type, table.filter(mask)


__all__ = [
    "ContentDigest",
    "DTYPE_MAP",
    "ENVELOPE_SCHEMA",
    "EXPECTED_SCHEMAS",
    "EventTypeInfo",
    "FieldInfo",
    "MAIN_TABLES",
    "MANIFEST_NAME",
    "ParsedPayload",
    "REQUIRED_FILES",
    "RawContractError",
    "RawDataset",
    "RawManifest",
    "SourceInfo",
    "TABLE_FILES",
    "iter_event_types",
    "parse_payloads",
    "read_manifest",
    "schema_differences",
]
