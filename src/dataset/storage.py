from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa

from src.preprocessing.artifacts import TableWriter, dumps_json, sha256_file, write_json

from .sample import Sample
from .version import FORMAT_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Набор лежит частями, читается частями и публикуется целиком.
#
# Три вида файлов, и граница между ними та же, что между кругами
# колонок примера:
#
#   samples.parquet          указатель и служебные сведения о
#                            примере целиком;
#   shards/<g>-N.parquet     входы модели и границы для Masker;
#   shards/<g>-N.events...   строка на КАЖДОЕ видимое событие,
#                            включая исключённые отбором.
#
# Горячий файл держится маленьким нарочно: обучение читает из
# него миллионы раз, и служебные колонки в нём были бы платой за
# то, что модели не нужно.
#
# Незавершённая сборка не выглядит готовой. Манифест пишется
# последним и только в каталоге `<id>.building`; готовым набор
# становится одним переименованием.
# ============================================================


MANIFEST_FILE = "dataset_manifest.json"
INDEX_FILE = "samples.parquet"
SHARDS_DIR = "shards"
REPORT_JSON_FILE = "dataset_report.json"
REPORT_MD_FILE = "dataset_report.md"
GOLDEN_JSON_FILE = "golden_examples.json"
GOLDEN_MD_FILE = "golden_examples.md"

BUILDING_SUFFIX = ".building"
REPLACED_SUFFIX = ".replaced"

# Предел числа элементов в списочной колонке одной группы строк:
# смещения pyarrow это int32.
LIST_LIMIT = 2 ** 31 - 1


# Колонки, объявленные модели, Masker и трассировке. Список
# едет в манифест: потребитель не должен угадывать, что ему
# разрешено читать.
MODEL_COLUMNS: tuple[str, ...] = (
    "key_ids",
    "value_ids",
    "positions",
    "event_starts",
    "event_lengths",
    "calendar",
    "hour_known",
    "hours_to_cutoff",
    "coverage_at_cutoff",
    "history_age_days",
    "profile_key_ids",
    "profile_value_ids",
    "profile_positions",
    "profile_state",
)

MASKER_COLUMNS: tuple[str, ...] = (
    "value_event",
    "value_start",
    "value_length",
    "value_key_id",
    "profile_value_start",
    "profile_value_length",
    "profile_value_key_id",
    "event_eligible",
    "has_targets",
    "n_eligible_events",
    "weight",
    "sample_seed",
    "dep_value",
    "dep_source_event",
    "dep_source_value",
    "dep_status",
)

SERVICE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "group",
    "client_id",
    "cutoff",
    "profile_version",
    "valid_from",
    "has_profile",
    "history_age_reason",
    "truncated",
    "selection",
    "coverage",
    "relationship",
    "limitations",
    "dep_key",
)


SHARD_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("group", pa.string()),
        ("client_id", pa.string()),
        ("cutoff", pa.timestamp("us")),
        ("weight", pa.float64()),
        ("sample_seed", pa.int64()),
        ("n_events", pa.int32()),
        ("n_tokens", pa.int32()),
        ("n_values", pa.int32()),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("event_starts", pa.list_(pa.int32())),
        ("event_lengths", pa.list_(pa.int32())),
        ("calendar", pa.list_(pa.float32())),
        ("hour_known", pa.list_(pa.bool_())),
        ("hours_to_cutoff", pa.list_(pa.float64())),
        ("event_eligible", pa.list_(pa.bool_())),
        ("value_event", pa.list_(pa.int32())),
        ("value_start", pa.list_(pa.int32())),
        ("value_length", pa.list_(pa.int32())),
        ("value_key_id", pa.list_(pa.int32())),
        ("profile_key_ids", pa.list_(pa.int32())),
        ("profile_value_ids", pa.list_(pa.int32())),
        ("profile_positions", pa.list_(pa.int32())),
        ("profile_value_start", pa.list_(pa.int32())),
        ("profile_value_length", pa.list_(pa.int32())),
        ("profile_value_key_id", pa.list_(pa.int32())),
        ("profile_state", pa.string()),
        ("profile_version", pa.int64()),
        ("valid_from", pa.timestamp("us")),
        ("has_profile", pa.bool_()),
        ("coverage_at_cutoff", pa.list_(pa.int8())),
        ("history_age_days", pa.float64()),
        ("n_eligible_events", pa.int32()),
        ("has_targets", pa.bool_()),
        ("truncated", pa.bool_()),
        ("dep_value", pa.list_(pa.int32())),
        ("dep_source_event", pa.list_(pa.int32())),
        ("dep_source_value", pa.list_(pa.int32())),
        ("dep_status", pa.list_(pa.int8())),
    ]
)


INDEX_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("group", pa.string()),
        ("client_id", pa.string()),
        ("cutoff", pa.timestamp("us")),
        ("shard", pa.string()),
        ("row", pa.int32()),
        ("n_events", pa.int32()),
        ("n_tokens", pa.int32()),
        ("n_values", pa.int32()),
        ("profile_tokens", pa.int32()),
        ("weight", pa.float64()),
        ("has_targets", pa.bool_()),
        ("n_eligible_events", pa.int32()),
        ("truncated", pa.bool_()),
        ("has_profile", pa.bool_()),
        ("excluded_events", pa.int32()),
        ("excluded_tokens", pa.int32()),
        ("excluded_eligible", pa.int32()),
        ("excluded_milestones", pa.int32()),
        ("budget_binding", pa.string()),
        ("history_age_days", pa.float64()),
        ("history_age_reason", pa.string()),
        ("profile_state", pa.string()),
        ("profile_version", pa.int64()),
        ("valid_from", pa.timestamp("us")),
        ("selection", pa.string()),
        ("coverage", pa.string()),
        ("relationship", pa.string()),
        ("limitations", pa.list_(pa.string())),
        ("dep_key", pa.list_(pa.string())),
    ]
)


EVENTS_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("group", pa.string()),
        ("client_id", pa.string()),
        ("cutoff", pa.timestamp("us")),
        ("position", pa.int32()),
        ("event_index", pa.int32()),
        ("kept", pa.bool_()),
        ("selection_reason", pa.string()),
        ("exclusion_reason", pa.string()),
        ("event_id", pa.string()),
        ("event_version", pa.int32()),
        ("stable_event_index", pa.int64()),
        ("event_time", pa.timestamp("us")),
        ("source", pa.string()),
        ("event_type", pa.string()),
        ("time_precision", pa.string()),
        ("hour_known", pa.bool_()),
        ("n_values", pa.int32()),
        ("n_tokens", pa.int32()),
        ("eligible", pa.bool_()),
        ("value_keys", pa.list_(pa.string())),
        ("unknown_keys", pa.list_(pa.string())),
        ("refs", pa.string()),
        ("provenance", pa.string()),
        ("absent_reasons", pa.string()),
    ]
)


class StorageError(ValueError):
    """
    Набор записать или прочитать нельзя.
    """


def shard_name(group: str, number: int) -> str:
    return f"{group}-{number:05d}"


def _json(value) -> str:
    return dumps_json(value).strip()


def shard_row(sample: Sample) -> dict:
    return {
        "sample_id": sample.sample_id,
        "group": sample.group,
        "client_id": sample.client_id,
        "cutoff": sample.cutoff,
        "weight": sample.weight,
        "sample_seed": sample.sample_seed,
        "n_events": sample.n_events,
        "n_tokens": sample.n_tokens,
        "n_values": sample.n_values,
        "key_ids": sample.key_ids.tolist(),
        "value_ids": sample.value_ids.tolist(),
        "positions": sample.positions.tolist(),
        "event_starts": sample.event_starts.tolist(),
        "event_lengths": sample.event_lengths.tolist(),
        "calendar": sample.calendar.tolist(),
        "hour_known": sample.hour_known.tolist(),
        "hours_to_cutoff": sample.hours_to_cutoff.tolist(),
        "event_eligible": sample.event_eligible.tolist(),
        "value_event": sample.value_event.tolist(),
        "value_start": sample.value_start.tolist(),
        "value_length": sample.value_length.tolist(),
        "value_key_id": sample.value_key_id.tolist(),
        "profile_key_ids": sample.profile_key_ids.tolist(),
        "profile_value_ids": sample.profile_value_ids.tolist(),
        "profile_positions": sample.profile_positions.tolist(),
        "profile_value_start": sample.profile_value_start.tolist(),
        "profile_value_length": sample.profile_value_length.tolist(),
        "profile_value_key_id": sample.profile_value_key_id.tolist(),
        "profile_state": sample.profile_state,
        "profile_version": sample.profile_version,
        "valid_from": sample.valid_from,
        "has_profile": sample.has_profile,
        "coverage_at_cutoff": sample.coverage_at_cutoff.tolist(),
        "history_age_days": sample.history_age_days,
        "n_eligible_events": sample.n_eligible_events,
        "has_targets": sample.has_targets,
        "truncated": sample.truncated,
        "dep_value": list(sample.dependencies.value),
        "dep_source_event": list(sample.dependencies.source_event),
        "dep_source_value": list(sample.dependencies.source_value),
        "dep_status": list(sample.dependencies.status),
    }


def index_row(sample: Sample, shard: str, row: int) -> dict:

    out = sample.as_index_row(shard, row)

    out.update(
        {
            "history_age_days": sample.history_age_days,
            "history_age_reason": sample.history_age_reason,
            "profile_state": sample.profile_state,
            "profile_version": sample.profile_version,
            "valid_from": sample.valid_from,
            "selection": _json(sample.selection),
            "coverage": _json(sample.coverage),
            "relationship": _json(sample.relationship),
            "limitations": list(sample.limitations),
            "dep_key": list(sample.dependencies.key),
        }
    )

    return out


def event_rows(sample: Sample) -> list[dict]:

    out: list[dict] = []

    for item in sample.events:

        row = dict(item)

        row.update(
            {
                "sample_id": sample.sample_id,
                "group": sample.group,
                "client_id": sample.client_id,
                "cutoff": sample.cutoff,
                "refs": _json(row.get("refs") or {}),
                "provenance": _json(row.get("provenance") or {}),
                "absent_reasons": _json(row.get("absent_reasons") or {}),
            }
        )

        out.append(row)

    return out


@dataclass
class ShardWriter:
    """
    Примеры одной группы: файл на shard_samples примеров, группа
    строк на row_group_samples.

    Размер группы строк это единица чтения. Он выбран небольшим
    нарочно: пример длинной истории занимает сотни тысяч чисел, и
    читатель обязан уметь взять один кусок, а не файл целиком.
    """

    directory: Path
    group: str
    shard_samples: int
    row_group_samples: int

    shards: dict[str, int] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    index: list[dict] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)

    _number: int = 0
    _in_shard: int = 0
    _writer: TableWriter | None = None
    _events_writer: TableWriter | None = None

    @property
    def shard(self) -> str:
        return shard_name(self.group, self._number)

    def add(self, sample: Sample) -> None:

        if self._writer is None:
            self._open()

        self.index.append(index_row(sample, self.shard, self._in_shard))

        self.rows.append(shard_row(sample))
        self.events.extend(event_rows(sample))

        self._in_shard += 1

        if len(self.rows) >= self.row_group_samples:
            self._flush()

        if self._in_shard >= self.shard_samples:
            self._close()

    def close(self) -> None:
        self._close()

    # --- внутреннее ---

    def _open(self) -> None:

        directory = self.directory / SHARDS_DIR

        self._writer = TableWriter(directory / f"{self.shard}.parquet", SHARD_SCHEMA)
        self._events_writer = TableWriter(directory / f"{self.shard}.events.parquet", EVENTS_SCHEMA)

    def _flush(self) -> None:

        if not self.rows:
            return

        _check_lists(self.rows)

        self._writer.write(pa.Table.from_pylist(self.rows, schema=SHARD_SCHEMA))

        if self.events:
            self._events_writer.write(pa.Table.from_pylist(self.events, schema=EVENTS_SCHEMA))

        self.rows = []
        self.events = []

    def _close(self) -> None:

        if self._writer is None:
            return

        self._flush()

        written = self._writer.close()
        self._events_writer.close()

        self.shards[self.shard] = written

        self.outputs.append(self.directory / SHARDS_DIR / f"{self.shard}.parquet")
        self.outputs.append(self.directory / SHARDS_DIR / f"{self.shard}.events.parquet")

        self._writer = None
        self._events_writer = None
        self._number += 1
        self._in_shard = 0


def _check_lists(rows: list[dict]) -> None:
    """
    Списочная колонка группы строк не переполняет смещения
    pyarrow.

    Проверка дешёвая, а отказ без неё выглядел бы как порча
    файла на чтении.
    """

    for name in ("key_ids", "value_ids", "positions"):

        total = sum(len(row[name]) for row in rows)

        if total > LIST_LIMIT:
            raise StorageError(
                f"в группе строк {total} элементов колонки {name} при пределе {LIST_LIMIT}: "
                "уменьшите row_group_samples"
            )


# ------------------------------------------------------------
# ПУБЛИКАЦИЯ
# ------------------------------------------------------------


def building_dir(root: Path, dataset_id: str) -> Path:
    return Path(root) / f"{dataset_id}{BUILDING_SUFFIX}"


def target_dir(root: Path, dataset_id: str) -> Path:
    return Path(root) / dataset_id


def prepare_build(root: Path, dataset_id: str, force: bool) -> Path:
    """
    Готовит каталог сборки, не трогая готовый набор.

    Незавершённый каталог с прошлого раза удаляется: завершённая
    сборка в нём не остаётся никогда, поэтому найденный `.building`
    это по определению обрубок.
    """

    root = Path(root)

    target = target_dir(root, dataset_id)

    if target.exists() and not force:
        raise StorageError(
            f"набор {dataset_id} уже собран в {target}. Пересборка стирает его целиком: "
            "укажите --force, если это то, чего вы хотите"
        )

    building = building_dir(root, dataset_id)

    if building.exists():
        shutil.rmtree(building)

    building.mkdir(parents=True)

    return building


def publish(root: Path, dataset_id: str, attempts: int = 10, pause: float = 0.5) -> Path:
    """
    Делает собранный каталог готовым набором одним движением.

    Прежний набор сначала отходит в сторону, и если подмена не
    удалась, он возвращается на место: исправный набор не должен
    исчезать из-за чужого открытого файла.
    """

    root = Path(root)

    building = building_dir(root, dataset_id)
    target = target_dir(root, dataset_id)

    if not (building / MANIFEST_FILE).exists():
        raise StorageError(
            f"в {building} нет манифеста: незавершённая сборка готовой не объявляется"
        )

    replaced = Path(f"{target}{REPLACED_SUFFIX}")

    if replaced.exists():
        shutil.rmtree(replaced)

    moved = False

    if target.exists():
        _rename(target, replaced, attempts, pause)
        moved = True

    try:
        _rename(building, target, attempts, pause)
    except StorageError:
        if moved:
            # Прежний набор был исправен, и оставлять пользователя
            # без него из-за неудачной подмены нельзя.
            _rename(replaced, target, attempts, pause)
        raise

    if moved:
        shutil.rmtree(replaced, ignore_errors=True)

    return target


def _rename(source: Path, destination: Path, attempts: int, pause: float) -> None:
    """
    Переименование каталога с повторами.

    На Windows чужой открытый дескриптор внутри каталога делает
    переименование невозможным на время: это не поломка набора, и
    падать с первой попытки незачем.
    """

    last: OSError | None = None

    for attempt in range(attempts):

        try:
            os.replace(source, destination)
            return
        except OSError as error:
            last = error
            if attempt + 1 < attempts:
                time.sleep(pause)

    raise StorageError(
        f"не удалось переименовать {source} в {destination} за {attempts} попыток: {last}. "
        "Каталог сборки сохранён"
    )


def write_manifest(directory: Path, manifest: dict, outputs: list[Path]) -> Path:
    """
    Манифест пишется последним и сам перечисляет все файлы.
    """

    directory = Path(directory)

    manifest = dict(manifest)

    manifest["schema_version"] = SCHEMA_VERSION
    manifest["format_version"] = FORMAT_VERSION

    manifest["files_sha256"] = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(outputs)
    }

    path = directory / MANIFEST_FILE

    write_json(path, manifest)

    return path


__all__ = [
    "BUILDING_SUFFIX",
    "EVENTS_SCHEMA",
    "GOLDEN_JSON_FILE",
    "GOLDEN_MD_FILE",
    "INDEX_FILE",
    "INDEX_SCHEMA",
    "LIST_LIMIT",
    "MANIFEST_FILE",
    "MASKER_COLUMNS",
    "MODEL_COLUMNS",
    "REPLACED_SUFFIX",
    "REPORT_JSON_FILE",
    "REPORT_MD_FILE",
    "SERVICE_COLUMNS",
    "SHARDS_DIR",
    "SHARD_SCHEMA",
    "ShardWriter",
    "StorageError",
    "building_dir",
    "event_rows",
    "index_row",
    "prepare_build",
    "publish",
    "shard_name",
    "shard_row",
    "target_dir",
    "write_manifest",
]
