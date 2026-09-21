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

from src.generator.profile import PROFILE_SCHEMA

from .artifacts import sha256_file


# ============================================================
# ИДЕЯ
# ============================================================
#
# Доступ к RAW v5 без pandas и без загрузки всего датасета:
# лента читается по одному row group, payload разбирается
# pyarrow.json по каталогу ключей из manifest.json.
#
# Контракт берётся из манифеста, а не из config генератора:
# препроцессинг должен работать и на выгрузке, которую делал не
# этот генератор. Всё, что манифест обещает (строки, контрольные
# суммы файлов и содержимого), здесь сверяется.
#
# truth/* не читается никогда: файлы лишь перечисляются как
# присутствующие.
# ============================================================


MANIFEST_NAME = "manifest.json"

EXPECTED_SCHEMA_VERSION = 7

# Файлы, без которых группа не обрабатывается.
REQUIRED_FILES: tuple[str, ...] = (
    "events.parquet",
    "profile.parquet",
    "source_coverage.parquet",
    "catalog/products.parquet",
    "catalog/merchants.parquet",
    "catalog/geography.parquet",
)

# Контрольный слой генератора: присутствие отмечается, содержимое
# не читается.
TRUTH_PREFIX = "truth/"

TABLE_FILES: dict[str, str] = {
    "events": "events.parquet",
    "profile": "profile.parquet",
    "source_coverage": "source_coverage.parquet",
    "products": "catalog/products.parquet",
    "merchants": "catalog/merchants.parquet",
    "geography": "catalog/geography.parquet",
}

# Таблицы, чьё содержимое манифест подписывает content_sha256 и
# которые препроцессинг читает.
CONTENT_TABLES: tuple[str, ...] = ("events", "profile", "source_coverage")

REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "schema_version",
    "seed",
    "total_clients",
    "history_start",
    "history_end",
    "extract_time",
    "sources",
    "key_catalogue",
    "rows",
    "content_sha256",
    "file_sha256",
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

COVERAGE_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("source", pa.string()),
        ("first_available_at", pa.timestamp("us")),
        ("last_available_at", pa.timestamp("us")),
        ("first_seen", pa.timestamp("us")),
        ("coverage_status", pa.string()),
        ("coverage_reason", pa.string()),
        ("opening_state", pa.string()),
        ("outage_days", pa.string()),
    ]
)

EXPECTED_SCHEMAS: dict[str, pa.Schema] = {
    "events": ENVELOPE_SCHEMA,
    "profile": PROFILE_SCHEMA,
    "source_coverage": COVERAGE_SCHEMA,
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
    defect_profile: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RawManifest:
    schema_version: int
    generator_version: str | None
    seed: int
    world_seed: int | None
    total_clients: int
    history_start: datetime
    history_end: datetime
    registry_start: datetime | None
    extract_time: datetime
    sources: dict[str, SourceInfo]
    catalogue: dict[str, EventTypeInfo]
    schema_changes: tuple[dict, ...]
    event_type_priority: dict[str, int]
    rows: dict[str, int]
    content_sha256: dict[str, str]
    file_sha256: dict[str, str]
    catalog_rows: dict[str, int]
    generation_config_sha256: str | None
    product_timeline_sha256: str | None
    sha256: str

    @property
    def event_types(self) -> tuple[str, ...]:
        return tuple(self.catalogue)

    def echo(self) -> dict:
        """
        Часть манифеста, которая переписывается в артефакты. Без
        путей и без полного каталога.
        """

        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "world_seed": self.world_seed,
            "total_clients": self.total_clients,
            "history_start": self.history_start.isoformat(),
            "history_end": self.history_end.isoformat(),
            "registry_start": None if self.registry_start is None else self.registry_start.isoformat(),
            "extract_time": self.extract_time.isoformat(),
            "sources": {
                name: {"available_from": info.available_from.isoformat()}
                for name, info in sorted(self.sources.items())
            },
            "event_types": list(self.event_types),
            "schema_changes": list(self.schema_changes),
            "rows": dict(sorted(self.rows.items())),
            "content_sha256": dict(sorted(self.content_sha256.items())),
            "catalog_rows": dict(sorted(self.catalog_rows.items())),
            "generation_config_sha256": self.generation_config_sha256,
            "product_timeline_sha256": self.product_timeline_sha256,
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

    sources = {
        name: SourceInfo(
            source=name,
            available_from=datetime.fromisoformat(info["available_from"]),
            defect_profile=dict(info.get("defect_profile", {})),
        )
        for name, info in data["sources"].items()
    }

    catalogue = _parse_catalogue(data["key_catalogue"])

    unknown_sources = sorted({info.source for info in catalogue.values()} - set(sources))
    if unknown_sources:
        raise RawContractError(f"каталог ключей ссылается на источники вне manifest.sources: {unknown_sources}")

    return RawManifest(
        schema_version=schema_version,
        generator_version=None if data.get("generator_version") is None else str(data["generator_version"]),
        seed=int(data["seed"]),
        world_seed=None if data.get("world_seed") is None else int(data["world_seed"]),
        total_clients=int(data["total_clients"]),
        history_start=datetime.fromisoformat(data["history_start"]),
        history_end=datetime.fromisoformat(data["history_end"]),
        registry_start=(
            None if data.get("registry_start") is None else datetime.fromisoformat(data["registry_start"])
        ),
        extract_time=datetime.fromisoformat(data["extract_time"]),
        sources=sources,
        catalogue=catalogue,
        schema_changes=tuple(dict(item) for item in data.get("schema_changes", [])),
        event_type_priority={name: int(value) for name, value in data.get("event_type_priority", {}).items()},
        rows={name: int(value) for name, value in data["rows"].items()},
        content_sha256={name: str(value) for name, value in data["content_sha256"].items()},
        file_sha256={name: str(value) for name, value in data["file_sha256"].items()},
        catalog_rows={name: int(value) for name, value in data.get("catalog_rows", {}).items()},
        generation_config_sha256=data.get("generation_config_sha256"),
        product_timeline_sha256=data.get("product_timeline_sha256"),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


# ============================================================
# КОНТРОЛЬНАЯ СУММА СОДЕРЖИМОГО
# ============================================================
#
# Тот же алгоритм, что у генератора: сумма blake2b-отпечатков
# строк по модулю 2**128 плюс число строк, порядок строк не
# важен. Повторён здесь, чтобы препроцессинг не зависел от
# внутренностей emit; равенство с оригиналом проверяет тест.
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
        слэшами, как их называет манифест.
        """

        return sorted(
            path.relative_to(self.raw_dir).as_posix()
            for path in self.raw_dir.rglob("*.parquet")
        )

    def truth_files(self) -> list[str]:
        return [name for name in self.listed_files() if name.startswith(TRUTH_PREFIX)]

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
        sha256 каждого файла, названного манифестом, кроме truth/*:
        контрольный слой не читается даже ради хэша.
        """

        result: dict[str, dict] = {}

        for name, expected in sorted(self.manifest.file_sha256.items()):

            if name.startswith(TRUTH_PREFIX):
                result[name] = {"status": "not_read", "expected": expected, "actual": None}
                continue

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

        for name in self.listed_files():
            if name not in result:
                result[name] = {"status": "unlisted", "expected": None, "actual": None}

        return result

    def verify_rows(self) -> dict[str, dict]:

        result: dict[str, dict] = {}

        for table in CONTENT_TABLES:

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
                "status": "ok" if expected == actual else ("unlisted" if expected is None else "mismatch"),
                "expected": expected,
                "actual": actual,
            }

        return result

    def verify_content(self) -> dict[str, dict]:
        """
        content_sha256 таблиц по строкам, потоково по row group.
        """

        result: dict[str, dict] = {}

        for table in CONTENT_TABLES:

            expected = self.manifest.content_sha256.get(table)

            if not self.exists(table):
                result[table] = {"status": "missing", "expected": expected, "actual": None}
                continue

            digest = ContentDigest()

            for _, chunk in self.iter_row_groups(table):
                digest.extend(chunk.to_pylist())

            actual = digest.value()

            result[table] = {
                "status": "ok" if expected == actual else ("unlisted" if expected is None else "mismatch"),
                "expected": expected,
                "actual": actual,
                "rows": digest.rows,
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
    "CONTENT_TABLES",
    "COVERAGE_SCHEMA",
    "ContentDigest",
    "DTYPE_MAP",
    "ENVELOPE_SCHEMA",
    "EXPECTED_SCHEMAS",
    "EventTypeInfo",
    "FieldInfo",
    "MANIFEST_NAME",
    "ParsedPayload",
    "REQUIRED_FILES",
    "RawContractError",
    "RawDataset",
    "RawManifest",
    "SourceInfo",
    "TABLE_FILES",
    "TRUTH_PREFIX",
    "iter_event_types",
    "parse_payloads",
    "read_manifest",
    "schema_differences",
]
