from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.generator.config import PROFILE_FIELDS
from src.preprocessing.artifacts import read_json, sha256_ints, write_json
from src.preprocessing.buckets import STATUS_NO_FIT_DATA, load_specs
from src.preprocessing.config import (
    KIND_BOOLEAN,
    KIND_NUMERIC,
    SCHEMA_VERSION,
    FieldSpec,
    feature_specs,
)
from src.preprocessing.cutoffs import CUTOFF_INDEX_SCHEMA
from src.preprocessing.fit import FitScope, fit_scope
from src.preprocessing.stats import value_key

from .config import (
    FIELD_VALUE_IDS_FILE,
    KEY_VOCAB_FILE,
    N_SPECIAL,
    SPECIAL_IDS,
    SPECIAL_NOTES,
    SPECIAL_TOKENS,
    SPECIAL_TOKENS_FILE,
    VALUE_VOCAB_FILE,
    IncompatibleArtifactsError,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Словарь обучается ТОЛЬКО на train и ровно на том же наборе
# записей, на котором preprocessing считал свои статистики:
# fit_scope переиспользуется как есть. Поэтому повторение одной
# записи в нескольких cutoff-примерах не увеличивает её частоту:
# fit-набор это префикс ленты клиента, а не объединение историй.
#
# Порядок ID не зависит от частот: ключи идут в порядке реестра
# preprocessing, значения внутри ключа по возрастанию самого
# значения. Значит value-ID одного ключа образуют непрерывный
# диапазон [value_start, value_end), а local = global - start.
# ============================================================


KEY_FORMAT = "{namespace}__{field}"

ORDER_RULES = {
    "ids": "сначала special, затем ключи, затем значения",
    "keys": "порядок реестра preprocessing: timeline.event_type, поля событий по EVENT_TYPES, затем profile",
    "values": {
        "numeric": "корзины 0..actual_bucket_count-1 из bucket_edges.json по возрастанию",
        "boolean": "false, затем true (среди встреченных на train)",
        "categorical": "по возрастанию типизированного значения: целые численно, строки по кодпоинтам",
    },
    "frequency": "count хранится рядом со значением, но на порядок ID не влияет",
    "repeated_keys": "повторяющиеся ключи внутри события сохраняют исходный порядок значений (стабильная сортировка)",
    "unknown_key": "пара с ключом вне реестра ставится после известных полей, в порядке ввода",
}

VALUE_RULES = {
    "missing": "null значение -> [MISSING] в value_ids; key_id при этом настоящий ключ поля",
    "unknown": "непустое значение вне frozen vocab -> [UNK]; словарь не расширяется",
    "numeric": "numeric берётся уже bucketized из preprocessing: значение это номер корзины",
    "no_bpe": "остальные значения кодируются целиком, без BPE",
    "metadata": "metadata (client_id, ts, seq, session_id, snapshot_month, payload) в словарь не входит",
    "special_position": "[EVT], [USR] и неизвестный ключ несут один и тот же special ID в key_ids и в value_ids",
}


ARROW_TYPES: dict[str, pa.DataType] = {
    "int64": pa.int64(),
    "int32": pa.int32(),
    "int16": pa.int16(),
    "double": pa.float64(),
    "float": pa.float32(),
    "string": pa.string(),
    "bool": pa.bool_(),
}


# ============================================================
# ЗАПИСИ СЛОВАРЯ
# ============================================================


@dataclass(frozen=True)
class KeyEntry:
    id: int
    key: str
    namespace: str
    field: str
    kind: str
    predictable: bool
    arrow_type: str
    value_start: int
    value_end: int

    @property
    def n_values(self) -> int:
        return self.value_end - self.value_start

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "key": self.key,
            "namespace": self.namespace,
            "field": self.field,
            "kind": self.kind,
            "predictable": self.predictable,
            "arrow_type": self.arrow_type,
            "value_start": self.value_start,
            "value_end": self.value_end,
            "n_values": self.n_values,
        }

    @staticmethod
    def from_json(data: dict) -> "KeyEntry":
        return KeyEntry(
            id=int(data["id"]),
            key=data["key"],
            namespace=data["namespace"],
            field=data["field"],
            kind=data["kind"],
            predictable=bool(data["predictable"]),
            arrow_type=data["arrow_type"],
            value_start=int(data["value_start"]),
            value_end=int(data["value_end"]),
        )


@dataclass(frozen=True)
class ValueEntry:
    id: int
    key_id: int
    key: str
    value: str
    count: int

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "key_id": self.key_id,
            "key": self.key,
            "value": self.value,
            "count": self.count,
        }

    @staticmethod
    def from_json(data: dict) -> "ValueEntry":
        return ValueEntry(
            id=int(data["id"]),
            key_id=int(data["key_id"]),
            key=data["key"],
            value=data["value"],
            count=int(data["count"]),
        )


def parse_value(kind: str, arrow_type: str, text: str) -> Any:
    """
    Строка из value_vocab обратно в типизированное значение.
    """

    if kind == KIND_BOOLEAN or arrow_type == "bool":
        return text == "true"

    if arrow_type.startswith("int"):
        return int(text)

    if arrow_type in ("double", "float"):
        return float(text)

    return text


# ============================================================
# СЛОВАРЬ
# ============================================================


class Vocab:
    """
    Frozen словарь: special-токены, ключи полей и значения в
    едином пространстве ID.
    """

    def __init__(self, keys: Iterable[KeyEntry], values: Iterable[ValueEntry]):

        self.specials: dict[str, int] = dict(SPECIAL_IDS)

        self.keys: tuple[KeyEntry, ...] = tuple(keys)
        self.values: tuple[ValueEntry, ...] = tuple(values)

        self._by_key_name: dict[str, KeyEntry] = {entry.key: entry for entry in self.keys}
        self._by_key_id: dict[int, KeyEntry] = {entry.id: entry for entry in self.keys}

        self._value_id: dict[tuple[int, str], int] = {
            (entry.key_id, entry.value): entry.id for entry in self.values
        }

        self._check_layout()

        self._typed_cache: dict[int, pa.Array] = {}

        size = self.size

        self.predictable_by_id = np.zeros(size, dtype=bool)
        self.value_start_by_id = np.full(size, -1, dtype=np.int64)
        self.n_candidates_by_id = np.zeros(size, dtype=np.int64)

        for entry in self.keys:
            self.predictable_by_id[entry.id] = entry.predictable
            self.value_start_by_id[entry.id] = entry.value_start
            self.n_candidates_by_id[entry.id] = entry.n_values

    # --------------------------------------------------------

    def _check_layout(self) -> None:
        """
        ID непрерывны и сгруппированы: special, ключи, значения.
        """

        for index, entry in enumerate(self.keys):
            if entry.id != N_SPECIAL + index:
                raise IncompatibleArtifactsError(
                    f"ключ {entry.key} имеет ID {entry.id}, ожидался {N_SPECIAL + index}"
                )

        expected = self.first_value_id

        for entry in self.keys:
            if entry.value_start != expected or entry.value_end < entry.value_start:
                raise IncompatibleArtifactsError(
                    f"диапазон значений ключа {entry.key} разрывен: {entry.value_start}..{entry.value_end}"
                )
            expected = entry.value_end

        if expected != self.size:
            raise IncompatibleArtifactsError("диапазоны значений не покрывают словарь целиком")

        for index, entry in enumerate(self.values):
            if entry.id != self.first_value_id + index:
                raise IncompatibleArtifactsError(f"значение {entry.value} имеет разрывный ID {entry.id}")

    # --------------------------------------------------------

    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def n_values(self) -> int:
        return len(self.values)

    @property
    def first_value_id(self) -> int:
        return N_SPECIAL + self.n_keys

    @property
    def size(self) -> int:
        return N_SPECIAL + self.n_keys + self.n_values

    # --------------------------------------------------------

    def key_entry(self, key: str) -> KeyEntry | None:
        return self._by_key_name.get(key)

    def key_entry_by_id(self, key_id: int) -> KeyEntry | None:
        return self._by_key_id.get(key_id)

    def key_id(self, namespace: str, field: str) -> int | None:
        entry = self._by_key_name.get(KEY_FORMAT.format(namespace=namespace, field=field))
        return None if entry is None else entry.id

    def value_id(self, key_id: int, value: str) -> int | None:
        return self._value_id.get((key_id, value))

    def decode(self, token_id: int) -> str:
        """
        Человекочитаемое имя токена: для отчётов и golden-векторов.
        """

        if 0 <= token_id < N_SPECIAL:
            return SPECIAL_TOKENS[token_id]

        entry = self._by_key_id.get(token_id)

        if entry is not None:
            return entry.key

        index = token_id - self.first_value_id

        if 0 <= index < self.n_values:
            return self.values[index].value

        raise KeyError(f"ID {token_id} вне словаря")

    def typed_values(self, key_id: int) -> pa.Array:
        """
        Значения ключа в типе исходного поля: для pc.index_in.
        Для numeric не используется: там колонка __bucket.
        """

        cached = self._typed_cache.get(key_id)

        if cached is not None:
            return cached

        entry = self._by_key_id[key_id]

        arrow_type = ARROW_TYPES.get(entry.arrow_type, pa.string())

        low = entry.value_start - self.first_value_id
        high = entry.value_end - self.first_value_id

        parsed = [
            parse_value(entry.kind, entry.arrow_type, value.value)
            for value in self.values[low:high]
        ]

        array = pa.array(parsed, type=arrow_type)

        self._typed_cache[key_id] = array

        return array

    # --------------------------------------------------------

    def field_value_ids(self) -> dict:
        return {
            "rule": (
                "поле -> отсортированные допустимые value ID; numeric это все корзины train-artifact, "
                "остальные поля это значения, встреченные на train; special токены в кандидаты не входят"
            ),
            "fields": {
                entry.key: {
                    "key_id": entry.id,
                    "kind": entry.kind,
                    "predictable": entry.predictable,
                    "n_candidates": entry.n_values,
                    "value_ids": list(range(entry.value_start, entry.value_end)),
                }
                for entry in self.keys
            },
        }

    def candidates(self) -> "CandidateIndex":
        return CandidateIndex(self)

    # --------------------------------------------------------

    def save(self, directory: Path) -> None:

        directory = Path(directory)

        write_json(
            directory / SPECIAL_TOKENS_FILE,
            {
                "n_special": N_SPECIAL,
                "ids": dict(SPECIAL_IDS),
                "tokens": [
                    {"id": SPECIAL_IDS[name], "token": name, "note": SPECIAL_NOTES[name]}
                    for name in SPECIAL_TOKENS
                ],
            },
        )

        write_json(
            directory / KEY_VOCAB_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "key_format": KEY_FORMAT,
                "first_key_id": N_SPECIAL,
                "n_keys": self.n_keys,
                "keys": [entry.to_json() for entry in self.keys],
            },
        )

        write_json(
            directory / VALUE_VOCAB_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "first_value_id": self.first_value_id,
                "n_values": self.n_values,
                "values": [entry.to_json() for entry in self.values],
            },
        )

        write_json(directory / FIELD_VALUE_IDS_FILE, self.field_value_ids())

    @staticmethod
    def load(directory: Path) -> "Vocab":

        directory = Path(directory)

        specials = read_json(directory / SPECIAL_TOKENS_FILE)

        if specials["ids"] != SPECIAL_IDS:
            raise IncompatibleArtifactsError("special-токены словаря не совпадают с контрактом")

        keys = read_json(directory / KEY_VOCAB_FILE)
        values = read_json(directory / VALUE_VOCAB_FILE)

        return Vocab(
            keys=[KeyEntry.from_json(item) for item in keys["keys"]],
            values=[ValueEntry.from_json(item) for item in values["values"]],
        )


class CandidateIndex:
    """
    Перевод между глобальным value ID и локальным индексом
    кандидата внутри поля.
    """

    def __init__(self, vocab: Vocab):
        self.value_start = vocab.value_start_by_id
        self.n_candidates = vocab.n_candidates_by_id

    def to_local(self, key_ids, value_ids) -> np.ndarray:
        return np.asarray(value_ids, dtype=np.int64) - self.value_start[np.asarray(key_ids, dtype=np.int64)]

    def to_global(self, key_ids, local) -> np.ndarray:
        return np.asarray(local, dtype=np.int64) + self.value_start[np.asarray(key_ids, dtype=np.int64)]

    def size_of(self, key_ids) -> np.ndarray:
        return self.n_candidates[np.asarray(key_ids, dtype=np.int64)]


# ============================================================
# КОЛОНКИ PROCESSED
# ============================================================


def events_column(spec: FieldSpec) -> str:
    """
    Из какой колонки events.parquet берётся значение поля.
    """

    if spec.namespace == "timeline":
        return spec.field

    return spec.bucket_column if spec.is_numeric else spec.column


def profile_column(spec: FieldSpec) -> str:
    return f"{spec.field}__bucket" if spec.is_numeric else spec.field


def key_specs() -> list[FieldSpec]:
    """
    Ключи словаря: все содержательные поля реестра, metadata нет.
    """

    return list(feature_specs())


def event_key_specs() -> list[FieldSpec]:
    return [spec for spec in key_specs() if spec.namespace != "profile"]


def profile_key_specs() -> list[FieldSpec]:
    order = {name: index for index, name in enumerate(PROFILE_FIELDS)}

    specs = [spec for spec in key_specs() if spec.namespace == "profile"]

    return sorted(specs, key=lambda spec: order[spec.field])


# ============================================================
# FIT
# ============================================================


@dataclass(frozen=True)
class FitReport:
    n_clients: int
    n_events: int
    n_snapshots: int
    cutoff_max: str | None

    def as_dict(self) -> dict:
        return {
            "dataset": "train",
            "n_fit_clients": self.n_clients,
            "n_fit_events": self.n_events,
            "n_fit_snapshots": self.n_snapshots,
            "fit_cutoff_max": self.cutoff_max,
            "rule": (
                "fit-набор словаря это fit-набор preprocessing: события train-клиентов с "
                "ts < последний валидный train-cutoff клиента и as-of снимки валидных train-примеров, "
                "каждая запись один раз"
            ),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompatibleArtifactsError(message)


def load_fit_scope(processed_dir: Path, split_manifest: dict) -> FitScope:
    """
    Fit-набор словаря это fit-набор preprocessing: тот же код,
    та же выборка. Состав сверяется со split manifest.
    """

    index = pq.read_table(Path(processed_dir) / "cutoff_index.parquet")

    _require(
        index.schema.equals(CUTOFF_INDEX_SCHEMA),
        "cutoff_index.parquet имеет чужую схему: preprocessing другой версии",
    )

    scope = fit_scope(index)

    train_ids = sorted(
        {
            int(client_id)
            for client_id, group in zip(
                index.column("client_id").to_pylist(), index.column("client_group").to_pylist()
            )
            if group == "train"
        }
    )

    clients = split_manifest.get("clients", {})

    _require(
        sha256_ints(train_ids) == clients.get("sha256", {}).get("train"),
        "состав train-клиентов не совпадает со split_manifest.json",
    )

    _require(
        scope.n_clients == clients.get("fit_clients"),
        f"fit-клиентов {scope.n_clients}, в split_manifest {clients.get('fit_clients')}",
    )

    return scope


def _count_column(counter: Counter, column) -> None:
    """
    Считает непустые значения колонки типизированными ключами:
    строка появляется только при записи словаря.
    """

    present = pc.drop_null(column)

    if len(present) == 0:
        return

    for item in pc.value_counts(present).to_pylist():
        counter[item["values"]] += int(item["counts"])


def count_events(processed_dir: Path, scope: FitScope) -> tuple[dict[tuple[str, str], Counter], int]:
    """
    Значения полей событий на fit-наборе. Колонка чужого типа
    события всегда null, поэтому фильтр по типу не нужен.
    """

    specs = event_key_specs()

    counters: dict[tuple[str, str], Counter] = {spec.key: Counter() for spec in specs}

    parquet = pq.ParquetFile(Path(processed_dir) / "clients" / "train_clients" / "events.parquet")

    columns = sorted({"client_id", "ts", *(events_column(spec) for spec in specs)})

    total = 0

    for index in range(parquet.num_row_groups):

        batch = parquet.read_row_group(index, columns=columns)

        if batch.num_rows == 0:
            continue

        mask = scope.mask_for(batch.column("client_id").to_numpy(), batch.column("ts").to_numpy())

        if not mask.any():
            continue

        rows = batch.filter(pa.array(mask))

        total += rows.num_rows

        for spec in specs:
            _count_column(counters[spec.key], rows.column(events_column(spec)))

    return counters, total


def count_profile(processed_dir: Path, scope: FitScope) -> tuple[dict[tuple[str, str], Counter], int]:
    """
    Значения профиля на тех снимках, которые реально выбраны
    как as-of валидными train-примерами.
    """

    specs = profile_key_specs()

    counters: dict[tuple[str, str], Counter] = {spec.key: Counter() for spec in specs}

    profile = pq.read_table(Path(processed_dir) / "clients" / "train_clients" / "profile.parquet")

    if profile.num_rows == 0 or not scope.snapshots:
        return counters, 0

    wanted = {
        (int(client_id), int(np.datetime64(stamp, "us").astype(np.int64)))
        for client_id, stamp in scope.snapshots
    }

    client_id = profile.column("client_id").to_numpy()
    ts = profile.column("ts").to_numpy().astype("datetime64[us]").astype(np.int64)

    mask = np.array(
        [(int(cid), int(stamp)) in wanted for cid, stamp in zip(client_id, ts)],
        dtype=bool,
    )

    rows = profile.filter(pa.array(mask))

    for spec in specs:
        _count_column(counters[spec.key], rows.column(profile_column(spec)))

    return counters, rows.num_rows


def _ordered_values(spec: FieldSpec, counter: Counter, bucket_count: int | None) -> list[tuple[str, int]]:
    """
    Значения ключа в порядке назначения ID.
    """

    if spec.kind == KIND_NUMERIC:

        if bucket_count is None:
            return []

        empty = [index for index in range(bucket_count) if counter.get(index, 0) == 0]

        if empty:
            raise IncompatibleArtifactsError(
                f"{spec.namespace}.{spec.field}: корзины {empty} пусты на train, "
                "это противоречит инварианту preprocessing"
            )

        return [(value_key(index), int(counter[index])) for index in range(bucket_count)]

    return [(value_key(value), int(count)) for value, count in sorted(counter.items())]


def build_vocab(
    counters: dict[tuple[str, str], Counter],
    bucket_counts: dict[tuple[str, str], int | None],
) -> Vocab:

    keys: list[KeyEntry] = []
    values: list[ValueEntry] = []

    specs = key_specs()

    next_value_id = N_SPECIAL + len(specs)

    for index, spec in enumerate(specs):

        key_id = N_SPECIAL + index

        key_name = KEY_FORMAT.format(namespace=spec.namespace, field=spec.field)

        ordered = _ordered_values(spec, counters.get(spec.key, Counter()), bucket_counts.get(spec.key))

        start = next_value_id

        for value, count in ordered:
            values.append(
                ValueEntry(id=next_value_id, key_id=key_id, key=key_name, value=value, count=count)
            )
            next_value_id += 1

        keys.append(
            KeyEntry(
                id=key_id,
                key=key_name,
                namespace=spec.namespace,
                field=spec.field,
                kind=spec.kind,
                predictable=spec.predictable,
                arrow_type=str(spec.arrow_type),
                value_start=start,
                value_end=next_value_id,
            )
        )

    return Vocab(keys=keys, values=values)


def fit_vocab(processed_dir: Path, artifacts_dir: Path) -> tuple[Vocab, FitReport]:
    """
    Полный fit словаря: только train, каждая запись один раз.
    """

    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)

    split_manifest = read_json(artifacts_dir / "split_manifest.json")
    field_stats = read_json(artifacts_dir / "field_stats.json")
    bucket_edges = read_json(artifacts_dir / "bucket_edges.json")

    _require(
        split_manifest.get("schema_version") == SCHEMA_VERSION,
        "split_manifest.json другой версии схемы preprocessing",
    )

    scope = load_fit_scope(processed_dir, split_manifest)

    event_counters, n_events = count_events(processed_dir, scope)
    profile_counters, n_snapshots = count_profile(processed_dir, scope)

    counters = {**event_counters, **profile_counters}

    expected_events = field_stats["fields"]["timeline"]["event_type"]["n_total"]
    expected_snapshots = field_stats["fields"]["profile"]["age"]["n_total"]

    _require(
        n_events == expected_events,
        f"fit-событий {n_events}, в field_stats {expected_events}",
    )

    _require(
        n_snapshots == expected_snapshots,
        f"fit-снимков {n_snapshots}, в field_stats {expected_snapshots}",
    )

    specs = load_specs(bucket_edges)

    bucket_counts: dict[tuple[str, str], int | None] = {}

    for spec in key_specs():

        if spec.kind != KIND_NUMERIC:
            continue

        bucket = specs.get(spec.key)

        _require(bucket is not None, f"в bucket_edges.json нет поля {spec.namespace}.{spec.field}")

        bucket_counts[spec.key] = None if bucket.status == STATUS_NO_FIT_DATA else bucket.actual_bucket_count

    vocab = build_vocab(counters, bucket_counts)

    cutoff_max = scope.max_cutoff

    report = FitReport(
        n_clients=scope.n_clients,
        n_events=n_events,
        n_snapshots=n_snapshots,
        cutoff_max=cutoff_max.isoformat() if isinstance(cutoff_max, datetime) else None,
    )

    return vocab, report
