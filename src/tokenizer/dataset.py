from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.artifacts import read_json, sha256_file

from .build import iter_client_blocks, tokenized_examples_schema
from .config import (
    CONFIG_FILE,
    DATASET_MANIFEST_FILE,
    EVT_ID,
    USR_ID,
    IncompatibleArtifactsError,
)
from .encode import Record, tokenized_events_schema
from .vocab import Vocab


# ============================================================
# ИДЕЯ
# ============================================================
#
# Контракт чтения для будущей модели.
#
# Пример это профиль (начинается с [USR]) плюс история событий
# (каждое начинается с [EVT]), уже в ID. Три массива токенов
# одной длины, смещения по событиям, и рядом ts, seq и
# event_type: время не превращалось в словарные категории и
# остаётся доступным для будущего временного кодирования.
#
# collate отдаёт плоский batch со смещениями, без padding:
# [PAD] зарезервирован, но не используется.
# ============================================================


@dataclass(frozen=True)
class Events:
    """
    История примера: плоские токены плюс смещения по событиям.
    """

    key_ids: np.ndarray
    value_ids: np.ndarray
    positions: np.ndarray
    offsets: np.ndarray
    event_type: np.ndarray
    ts: np.ndarray
    seq: np.ndarray

    @property
    def n_events(self) -> int:
        return int(self.offsets.size - 1)

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.size)

    def event(self, index: int) -> Record:
        lo, hi = int(self.offsets[index]), int(self.offsets[index + 1])
        return Record(self.key_ids[lo:hi], self.value_ids[lo:hi], self.positions[lo:hi])


@dataclass(frozen=True)
class Example:
    client_id: int
    cutoff: datetime
    dataset: str
    client_group: str
    seq_end: int
    snapshot_ts: datetime
    profile: Record
    events: Events

    @property
    def n_tokens(self) -> int:
        return self.profile.n_tokens + self.events.n_tokens


@dataclass(frozen=True)
class TokenBatch:
    """
    Вход будущего Event и History Encoder.

    Токены событий лежат плоско; event_ids говорит, какому
    событию принадлежит токен, example_ids — какому примеру.
    """

    key_ids: np.ndarray
    value_ids: np.ndarray
    positions: np.ndarray
    event_ids: np.ndarray
    example_ids: np.ndarray

    event_offsets: np.ndarray
    example_of_event: np.ndarray
    event_type: np.ndarray
    ts: np.ndarray
    seq: np.ndarray

    profile_key_ids: np.ndarray
    profile_value_ids: np.ndarray
    profile_positions: np.ndarray
    profile_example_ids: np.ndarray

    n_examples: int

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.size)

    @property
    def n_events(self) -> int:
        return int(self.event_offsets.size - 1)


# ============================================================
# ЧТЕНИЕ
# ============================================================


def _row_group_ranges(parquet: pq.ParquetFile, column: str) -> list[tuple[int, int]]:
    """
    Диапазон client_id каждой группы строк.

    Берётся из статистик parquet; если их нет, читается сама
    колонка — она одна и дешёвая по сравнению со списками.
    """

    position = parquet.schema_arrow.get_field_index(column)

    ranges: list[tuple[int, int]] = []

    for index in range(parquet.num_row_groups):

        stats = parquet.metadata.row_group(index).column(position).statistics

        if stats is not None and stats.has_min_max:
            ranges.append((int(stats.min), int(stats.max)))
            continue

        values = parquet.read_row_group(index, columns=[column]).column(column).to_numpy()

        ranges.append((int(values.min()), int(values.max())) if values.size else (1, 0))

    return ranges


def _record_of(table: pa.Table, index: int) -> Record:

    row = table.slice(index, 1)

    return Record(
        key_ids=np.asarray(row.column("key_ids")[0].as_py(), dtype=np.int32),
        value_ids=np.asarray(row.column("value_ids")[0].as_py(), dtype=np.int32),
        positions=np.asarray(row.column("positions")[0].as_py(), dtype=np.int16),
    )


def _events_of(table: pa.Table, lo: int, hi: int) -> Events:
    """
    Строки [lo, hi) таблицы событий в плоскую историю.
    """

    rows = table.slice(lo, hi - lo)

    widths = rows.column("n_tokens").to_numpy().astype(np.int64)

    offsets = np.zeros(widths.size + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    def flat(name: str, dtype) -> np.ndarray:
        column = rows.column(name).combine_chunks()
        if len(column) == 0:
            return np.zeros(0, dtype=dtype)
        return column.flatten().to_numpy(zero_copy_only=False).astype(dtype)

    return Events(
        key_ids=flat("key_ids", np.int32),
        value_ids=flat("value_ids", np.int32),
        positions=flat("positions", np.int16),
        offsets=offsets,
        event_type=np.asarray(rows.column("event_type").to_pylist(), dtype=object),
        ts=rows.column("ts").to_numpy().astype("datetime64[us]"),
        seq=rows.column("seq").to_numpy().astype(np.int64),
    )


class TokenizedDataset:
    """
    Один датасет: примеры и записи их группы клиентов.
    """

    def __init__(
        self,
        root: Path,
        dataset: str | None,
        vocab_dir: Path | None = None,
        group: str | None = None,
    ):
        """
        dataset=None открывает записи группы клиентов без своего
        каталога примеров. Так читаются наборы, которых в
        разбиении нет: downstream строит свои строки примеров
        сам, из cutoff_index, и каталога под них не заводит.
        """

        self.root = Path(root)
        self.dataset = dataset

        if dataset is None:

            if group is None:
                raise ValueError("без каталога примеров нужно указать группу клиентов")

            self.examples = tokenized_examples_schema().empty_table()
            self.group = group

        else:

            self.examples = pq.read_table(self.root / dataset / "examples.parquet")

            groups = set(self.examples.column("client_group").to_pylist())

            if len(groups) > 1:
                raise ValueError(
                    f"{dataset}: примеры из разных групп клиентов {sorted(groups)}"
                )

            self.group = groups.pop() if groups else "train"

        self.events_path = self.root / "clients" / f"{self.group}_clients" / "events.parquet"
        self.profile_path = self.root / "clients" / f"{self.group}_clients" / "profile.parquet"

        self._parquet = pq.ParquetFile(self.events_path)

        self._row_group_range = _row_group_ranges(self._parquet, "client_id")

        self._cached_groups: tuple[int, int] | None = None
        self._cached_table: pa.Table | None = None

        self._profile = pq.read_table(self.profile_path)
        self._profile_index = {
            (int(client_id), int(np.datetime64(stamp, "us").astype(np.int64))): index
            for index, (client_id, stamp) in enumerate(
                zip(self._profile.column("client_id").to_pylist(), self._profile.column("ts").to_pylist())
            )
        }

        if vocab_dir is not None:
            self.check_vocab(vocab_dir)

    # --------------------------------------------------------

    def check_vocab(self, vocab_dir: Path) -> None:
        """
        Датасет закодирован именно этим словарём.

        Перезапуск preprocessing перестраивает словарь; старый
        data/tokenized не должен открываться молча.
        """

        manifest_path = self.root / DATASET_MANIFEST_FILE

        if not manifest_path.exists():
            raise IncompatibleArtifactsError(f"нет {DATASET_MANIFEST_FILE} в {self.root}")

        manifest = read_json(manifest_path)

        actual = sha256_file(Path(vocab_dir) / CONFIG_FILE)

        if manifest.get("tokenizer_config_sha256") != actual:
            raise IncompatibleArtifactsError(
                "датасет закодирован другим словарём: tokenizer_config.json не совпадает"
            )

    # --------------------------------------------------------

    def __len__(self) -> int:
        return self.examples.num_rows

    def _events_of_client(self, client_id: int, seq_end: int) -> Events:
        """
        Только те группы строк, где лежит этот клиент.

        Читать файл целиком нельзя: на dev в ленте train больше
        15 млн событий, и три колонки списков дают гигабайты.
        """

        groups = [
            index
            for index, (low, high) in enumerate(self._row_group_range)
            if low <= client_id <= high
        ]

        if not groups:
            return _events_of(tokenized_events_schema().empty_table(), 0, 0)

        span = (groups[0], groups[-1])

        if self._cached_groups != span:
            self._cached_table = self._parquet.read_row_groups(list(range(span[0], span[1] + 1)))
            self._cached_groups = span

        table = self._cached_table

        ids = table.column("client_id").to_numpy()

        lo = int(np.searchsorted(ids, client_id, side="left"))

        return _events_of(table, lo, lo + seq_end)

    def _profile_record(self, client_id: int, snapshot: datetime) -> Record:

        stamp = int(np.datetime64(snapshot, "us").astype(np.int64))

        position = self._profile_index.get((int(client_id), stamp))

        if position is None:
            raise KeyError(f"нет снимка профиля клиента {client_id} на {snapshot}")

        return _record_of(self._profile, position)

    def _example(self, row: dict, events: Events) -> Example:
        return Example(
            client_id=int(row["client_id"]),
            cutoff=row["cutoff"],
            dataset=row["dataset"],
            client_group=row["client_group"],
            seq_end=int(row["seq_end"]),
            snapshot_ts=row["snapshot_ts"],
            profile=self._profile_record(int(row["client_id"]), row["snapshot_ts"]),
            events=events,
        )

    # --------------------------------------------------------

    def load(self, index: int) -> Example:
        """
        Один пример: история это префикс seq < seq_end.
        """

        row = self.examples.slice(index, 1).to_pylist()[0]

        return self._example(row, self._events_of_client(int(row["client_id"]), int(row["seq_end"])))

    def iter_examples(self) -> Iterator[Example]:
        """
        Потоковый обход: события читаются блоками клиентов, для
        каждого клиента отдаются все его примеры.
        """

        by_client: dict[int, list[dict]] = {}

        for row in self.examples.to_pylist():
            by_client.setdefault(int(row["client_id"]), []).append(row)

        for block in iter_client_blocks(self.events_path):

            client_id = block.column("client_id").to_numpy()

            if client_id.size == 0:
                continue

            change = np.flatnonzero(np.diff(client_id)) + 1
            starts = np.concatenate([[0], change])
            ends = np.concatenate([change, [client_id.size]])

            for lo, hi in zip(starts, ends):

                rows = by_client.pop(int(client_id[lo]), [])

                for row in rows:
                    yield self._example(row, _events_of(block, int(lo), int(lo) + int(row["seq_end"])))

        if by_client:
            raise ValueError(f"примеры без событий: клиенты {sorted(by_client)[:5]}")


# ============================================================
# BATCH
# ============================================================


def collate(examples: Sequence[Example]) -> TokenBatch:
    """
    Плоский batch без padding: смещения вместо выравнивания.
    """

    key_ids: list[np.ndarray] = []
    value_ids: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []
    example_ids: list[np.ndarray] = []

    widths: list[np.ndarray] = []
    example_of_event: list[np.ndarray] = []
    event_type: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    seq: list[np.ndarray] = []

    profile_keys: list[np.ndarray] = []
    profile_values: list[np.ndarray] = []
    profile_positions: list[np.ndarray] = []
    profile_examples: list[np.ndarray] = []

    event_base = 0

    for index, example in enumerate(examples):

        events = example.events

        n_events = events.n_events

        key_ids.append(events.key_ids)
        value_ids.append(events.value_ids)
        positions.append(events.positions)

        per_event = np.diff(events.offsets)

        event_ids.append(np.repeat(np.arange(n_events, dtype=np.int64) + event_base, per_event))
        example_ids.append(np.full(events.n_tokens, index, dtype=np.int64))

        widths.append(per_event)
        example_of_event.append(np.full(n_events, index, dtype=np.int64))
        event_type.append(events.event_type)
        ts.append(events.ts)
        seq.append(events.seq)

        event_base += n_events

        profile_keys.append(example.profile.key_ids)
        profile_values.append(example.profile.value_ids)
        profile_positions.append(example.profile.positions)
        profile_examples.append(np.full(example.profile.n_tokens, index, dtype=np.int64))

    def join(chunks: list[np.ndarray], dtype) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=dtype)
        return np.concatenate(chunks).astype(dtype)

    per_event_all = join(widths, np.int64)

    event_offsets = np.zeros(per_event_all.size + 1, dtype=np.int64)
    np.cumsum(per_event_all, out=event_offsets[1:])

    return TokenBatch(
        key_ids=join(key_ids, np.int32),
        value_ids=join(value_ids, np.int32),
        positions=join(positions, np.int16),
        event_ids=join(event_ids, np.int64),
        example_ids=join(example_ids, np.int64),
        event_offsets=event_offsets,
        example_of_event=join(example_of_event, np.int64),
        event_type=np.concatenate(event_type) if event_type else np.zeros(0, dtype=object),
        ts=np.concatenate(ts) if ts else np.zeros(0, dtype="datetime64[us]"),
        seq=join(seq, np.int64),
        profile_key_ids=join(profile_keys, np.int32),
        profile_value_ids=join(profile_values, np.int32),
        profile_positions=join(profile_positions, np.int16),
        profile_example_ids=join(profile_examples, np.int64),
        n_examples=len(examples),
    )


# ============================================================
# GOLDEN
# ============================================================


def decode_record(vocab: Vocab, record: Record) -> list[list]:
    return [
        [int(position), vocab.decode(int(key)), vocab.decode(int(value)), int(key), int(value)]
        for key, value, position in zip(record.key_ids, record.value_ids, record.positions)
    ]


def golden_example(vocab: Vocab, example: Example) -> dict:
    """
    Человекочитаемый снимок примера для фиксации в тестах.
    """

    events = example.events

    picked = list(range(min(3, events.n_events)))

    if events.n_events and events.n_events - 1 not in picked:
        picked.append(events.n_events - 1)

    return {
        "client_id": example.client_id,
        "cutoff": example.cutoff.isoformat(),
        "dataset": example.dataset,
        "client_group": example.client_group,
        "seq_end": example.seq_end,
        "snapshot_ts": example.snapshot_ts.isoformat(),
        "n_events": events.n_events,
        "n_event_tokens": events.n_tokens,
        "n_tokens": example.n_tokens,
        "profile": {
            "lead": vocab.decode(int(example.profile.value_ids[0])),
            "n_tokens": example.profile.n_tokens,
            "tokens": decode_record(vocab, example.profile),
        },
        "events": [
            {
                "index": index,
                "seq": int(events.seq[index]),
                "ts": events.ts[index].astype("datetime64[us]").astype(datetime).isoformat(),
                "event_type": str(events.event_type[index]),
                "lead": vocab.decode(int(events.event(index).value_ids[0])),
                "tokens": decode_record(vocab, events.event(index)),
            }
            for index in picked
        ],
    }


def first_example(dataset: TokenizedDataset) -> Example | None:
    """
    Первый пример в порядке (client_id, cutoff): стабильный
    выбор, не зависящий от порядка строк в файле.
    """

    if len(dataset) == 0:
        return None

    keys = list(
        zip(
            dataset.examples.column("client_id").to_pylist(),
            dataset.examples.column("cutoff").to_pylist(),
            range(len(dataset)),
        )
    )

    keys.sort()

    return dataset.load(keys[0][2])


assert EVT_ID != USR_ID
