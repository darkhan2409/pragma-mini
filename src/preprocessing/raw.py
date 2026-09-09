from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pj
import pyarrow.parquet as pq

from src.generator.emit import SCHEMAS
from src.generator.version import manifest_revision

from .config import payload_schema


# ============================================================
# ИДЕЯ
# ============================================================
#
# Доступ к RAW без pandas: nullable int64 у pandas превращается
# в float64, nullable bool в object, и типы контракта теряются.
# Всё читается через pyarrow, лента по одному row group,
# payload разбирается pyarrow.json с явной схемой.
#
# Границы горизонта берутся из manifest.json, а не из config
# генератора: preprocessing должен работать и на реальных RAW.
# ============================================================


MANIFEST_NAME = "manifest.json"

REQUIRED_MANIFEST_KEYS = (
    "seed",
    "total_clients",
    "history_start",
    "feature_end",
    "label_end",
    "source_availability",
    "event_type_priority",
    "max_tokens_per_event",
    "max_events_per_history",
    "rows",
)


@dataclass(frozen=True)
class RawManifest:
    seed: int
    total_clients: int
    history_start: datetime
    feature_end: datetime
    label_end: datetime
    source_availability: dict[str, datetime]
    event_type_priority: dict[str, int]
    max_tokens_per_event: int
    max_events_per_history: int
    rows: dict[str, int]
    sha256: str

    # Ревизия схемы RAW. Ключа нет значит 1: манифесты первого
    # выпуска лежат на диске и обязаны читаться как есть.
    revision: int = 1

    def echo(self) -> dict:
        """
        Часть манифеста, которая переписывается в artifacts.
        Без путей и времени запуска.
        """

        return {
            "seed": self.seed,
            "total_clients": self.total_clients,
            "history_start": self.history_start.isoformat(),
            "feature_end": self.feature_end.isoformat(),
            "label_end": self.label_end.isoformat(),
            "source_availability": {
                source: ts.isoformat() for source, ts in sorted(self.source_availability.items())
            },
            "event_type_priority": dict(sorted(self.event_type_priority.items())),
            "max_tokens_per_event": self.max_tokens_per_event,
            "max_events_per_history": self.max_events_per_history,
            "rows": dict(sorted(self.rows.items())),
            "raw_schema_revision": self.revision,
            "manifest_sha256": self.sha256,
        }


def read_manifest(raw_dir: Path) -> RawManifest:

    path = raw_dir / MANIFEST_NAME

    if not path.exists():
        raise FileNotFoundError(f"нет manifest.json в {raw_dir}")

    raw_bytes = path.read_bytes()

    data = json.loads(raw_bytes.decode("utf-8"))

    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in data]

    if missing:
        raise ValueError(f"в manifest.json нет ключей: {missing}")

    return RawManifest(
        seed=int(data["seed"]),
        total_clients=int(data["total_clients"]),
        history_start=datetime.fromisoformat(data["history_start"]),
        feature_end=datetime.fromisoformat(data["feature_end"]),
        label_end=datetime.fromisoformat(data["label_end"]),
        source_availability={
            source: datetime.fromisoformat(value)
            for source, value in data["source_availability"].items()
        },
        event_type_priority={name: int(value) for name, value in data["event_type_priority"].items()},
        max_tokens_per_event=int(data["max_tokens_per_event"]),
        max_events_per_history=int(data["max_events_per_history"]),
        rows={name: int(value) for name, value in data["rows"].items()},
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        revision=manifest_revision(data),
    )


# ============================================================
# ДАТАСЕТ
# ============================================================


class RawDataset:
    """
    Ленивый доступ к таблицам RAW. Ничего не читает при создании.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.manifest = read_manifest(self.raw_dir)

    def path(self, name: str) -> Path:
        return self.raw_dir / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def schema(self, name: str) -> pa.Schema:
        return pq.read_schema(self.path(name)).remove_metadata()

    def num_rows(self, name: str) -> int:
        return int(pq.read_metadata(self.path(name)).num_rows)

    def read(self, name: str, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.path(name), columns=columns)

    def parquet_file(self, name: str) -> pq.ParquetFile:
        return pq.ParquetFile(self.path(name))

    def iter_row_groups(self, name: str, columns: list[str] | None = None) -> Iterator[pa.Table]:
        """
        Таблица по одному row group в порядке файла. У генератора
        один row group это один чанк клиентов, поэтому память
        ограничена размером чанка.
        """

        parquet = self.parquet_file(name)

        for index in range(parquet.num_row_groups):
            yield parquet.read_row_group(index, columns=columns)

    def client_ids(self) -> list[int]:
        """
        Все клиенты датасета по таблице покрытия: она есть у каждого.
        """

        column = self.read("source_coverage", ["client_id"]).column("client_id")

        return sorted(set(column.to_pylist()))


# ============================================================
# РАЗБОР PAYLOAD
# ============================================================


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


def parse_payloads(event_type: str, payload: pa.Array | pa.ChunkedArray) -> pa.Table:
    """
    Типизированная таблица полей payload в порядке контракта.
    Лишний ключ это ошибка; отсутствующий ключ становится null.
    Порядок строк сохраняется.
    """

    schema = payload_schema(event_type)

    if len(payload) == 0:
        return schema.empty_table()

    parsed = pj.read_json(
        pa.BufferReader(_joined_buffer(payload)),
        parse_options=pj.ParseOptions(
            explicit_schema=schema,
            unexpected_field_behavior="error",
        ),
    )

    if parsed.num_rows != len(payload):
        raise ValueError(
            f"{event_type}: разобрано {parsed.num_rows} строк payload из {len(payload)}"
        )

    return parsed.select(schema.names).combine_chunks()


def payload_keys(payload_text: str) -> list[str]:
    """
    Ключи одного payload в порядке записи (для проверки контракта).
    """

    return list(json.loads(payload_text).keys())


def timeline_schema() -> pa.Schema:
    return SCHEMAS["timeline"]
