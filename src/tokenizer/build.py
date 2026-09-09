from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.artifacts import TableWriter, write_table
from src.preprocessing.config import CLIENT_GROUPS, DATASETS
from src.preprocessing.cutoffs import EXAMPLES_SCHEMA
from src.preprocessing.stats import numeric_summary

from .config import MASK_ID, MISSING_ID, UNK_ID, ResolvedLimits
from .encode import (
    PROFILE_WIDTH,
    encode_events_table,
    encode_profile_table,
    tokenized_events_schema,
    tokenized_profile_schema,
)
from .vocab import Vocab


# ============================================================
# ИДЕЯ
# ============================================================
#
# Записи хранятся один раз на группу клиентов, пример это
# ссылка (client_id, cutoff, seq_end, snapshot_ts), как в
# preprocessing. История примера это префикс seq < seq_end,
# поэтому все счётчики примера берутся из накопительных сумм по
# клиенту, а не пересчётом события в каждом примере.
#
# Ничего не обрезается: превышение лимита события или истории
# только считается и попадает в статистику.
# ============================================================


DATASETS_BY_GROUP: dict[str, list[str]] = {
    group: [dataset for (grp, _), dataset in DATASETS.items() if grp == group]
    for group in CLIENT_GROUPS
}


def tokenized_examples_schema() -> pa.Schema:
    """
    Ссылки preprocessing плюс то, что считается по токенам.
    """

    return pa.schema(
        list(EXAMPLES_SCHEMA)
        + [
            ("n_tokens", pa.int64()),
            ("n_profile_tokens", pa.int64()),
            ("n_event_tokens", pa.int64()),
            ("max_event_tokens", pa.int16()),
            ("events_over_token_limit", pa.int64()),
            ("history_over_limit", pa.bool_()),
            ("n_unknown", pa.int64()),
            ("n_missing", pa.int64()),
        ]
    )


# ============================================================
# ПОТОК ПО КЛИЕНТАМ
# ============================================================


def iter_client_blocks(path: Path, columns: list[str] | None = None) -> Iterator[pa.Table]:
    """
    Таблица блоками, внутри которых ни один клиент не разрезан.

    Раскладка row group не предполагается: хвост последнего
    клиента переносится в следующий блок. На реальных данных
    row group нарезаны иначе, чем у генератора.
    """

    parquet = pq.ParquetFile(path)

    carry: pa.Table | None = None

    for index in range(parquet.num_row_groups):

        group = parquet.read_row_group(index, columns=columns)

        if group.num_rows == 0:
            continue

        table = group if carry is None else pa.concat_tables([carry, group])

        carry = None

        client_id = table.column("client_id").to_numpy()

        cut = int(np.searchsorted(client_id, client_id[-1], side="left"))

        if cut == 0:
            # Клиент занимает весь блок: ждём его продолжения.
            carry = table
            continue

        carry = table.slice(cut)

        yield table.slice(0, cut)

    if carry is not None and carry.num_rows:
        yield carry


def client_runs(client_id: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Границы [lo, hi) блока каждого клиента в отсортированном массиве.
    """

    if client_id.size == 0:
        return []

    change = np.flatnonzero(np.diff(client_id)) + 1

    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [client_id.size]])

    return [(int(client_id[lo]), int(lo), int(hi)) for lo, hi in zip(starts, ends)]


def check_order(client_id: np.ndarray, ts: np.ndarray, seq: np.ndarray) -> None:
    """
    Внутри клиента строки идут по (ts, seq).

    Порядок не предполагается, а проверяется: чужой processed
    должен падать громко, а не давать перепутанную историю.
    """

    for value, lo, hi in client_runs(client_id):

        order = np.lexsort((seq[lo:hi], ts[lo:hi]))

        if not np.array_equal(order, np.arange(hi - lo)):
            raise ValueError(f"события клиента {value} не отсортированы по (ts, seq)")


# ============================================================
# СЧЁТЧИКИ ПО ТОКЕНАМ
# ============================================================


def flat_arrays(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Плоские key_ids, value_ids и длины строк.
    """

    keys = table.column("key_ids").combine_chunks()
    values = table.column("value_ids").combine_chunks()

    widths = table.column("n_tokens").to_numpy().astype(np.int64)

    return (
        keys.flatten().to_numpy(zero_copy_only=False).astype(np.int64),
        values.flatten().to_numpy(zero_copy_only=False).astype(np.int64),
        widths,
    )


def per_row_counts(flat_values: np.ndarray, widths: np.ndarray, token_id: int) -> np.ndarray:

    rows = widths.size

    if rows == 0:
        return np.zeros(0, dtype=np.int64)

    row_index = np.repeat(np.arange(rows, dtype=np.int64), widths)

    return np.bincount(row_index[flat_values == token_id], minlength=rows).astype(np.int64)


class TokenStats:
    """
    Счётчики записей группы клиентов: каждая запись один раз.
    """

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.n_records = 0
        self.n_tokens = 0
        self.n_unknown = 0
        self.n_missing = 0
        self.by_type: Counter[str] = Counter()
        self.tokens_by_type: Counter[str] = Counter()
        self.unknown_by_key: Counter[str] = Counter()
        self.missing_by_key: Counter[str] = Counter()
        self.over_token_limit = 0

    def update(self, table: pa.Table, limits: ResolvedLimits, type_column: str | None) -> None:

        if table.num_rows == 0:
            return

        flat_keys, flat_values, widths = flat_arrays(table)

        self.n_records += table.num_rows
        self.n_tokens += int(widths.sum())
        self.over_token_limit += int((widths > limits.max_tokens_per_event).sum())

        for token_id, total, by_key in (
            (UNK_ID, "n_unknown", self.unknown_by_key),
            (MISSING_ID, "n_missing", self.missing_by_key),
        ):
            hit = flat_values == token_id

            setattr(self, total, getattr(self, total) + int(hit.sum()))

            if hit.any():
                for key_id, count in Counter(flat_keys[hit].tolist()).items():
                    entry = self.vocab.key_entry_by_id(int(key_id))
                    by_key[entry.key if entry else "[UNK]"] += int(count)

        if type_column is not None:
            names = table.column(type_column).to_pylist()
            for name, width in zip(names, widths.tolist()):
                self.by_type[name] += 1
                self.tokens_by_type[name] += int(width)

    def as_dict(self) -> dict:
        return {
            "n_records": self.n_records,
            "n_tokens": self.n_tokens,
            "n_unknown": self.n_unknown,
            "n_missing": self.n_missing,
            "records_over_token_limit": self.over_token_limit,
            "records_by_type": dict(sorted(self.by_type.items())),
            "tokens_by_type": dict(sorted(self.tokens_by_type.items())),
            "unknown_by_key": dict(sorted(self.unknown_by_key.items())),
            "missing_by_key": dict(sorted(self.missing_by_key.items())),
        }


# ============================================================
# ПРОФИЛЬ
# ============================================================


def profile_index(table: pa.Table) -> dict[tuple[int, int], tuple[int, int]]:
    """
    (client_id, ts) -> (unknown, missing) для строки профиля.
    """

    if table.num_rows == 0:
        return {}

    _, flat_values, widths = flat_arrays(table)

    unknown = per_row_counts(flat_values, widths, UNK_ID)
    missing = per_row_counts(flat_values, widths, MISSING_ID)

    client_id = table.column("client_id").to_numpy()
    ts = table.column("ts").to_numpy().astype("datetime64[us]").astype(np.int64)

    return {
        (int(cid), int(stamp)): (int(unk), int(mis))
        for cid, stamp, unk, mis in zip(client_id, ts, unknown, missing)
    }


# ============================================================
# СБОРКА
# ============================================================


class ExampleBuilder:
    """
    Накапливает по клиенту то, что нужно каждому его примеру.
    """

    def __init__(self, processed_dir: Path, group: str, limits: ResolvedLimits):

        self.limits = limits

        self.tables: dict[str, pa.Table] = {}
        self.columns: dict[str, dict[str, np.ndarray]] = {}
        self.pending: dict[int, list[tuple[str, int, int]]] = {}

        for dataset in sorted(DATASETS_BY_GROUP[group]):

            table = pq.read_table(Path(processed_dir) / dataset / "examples.parquet")

            self.tables[dataset] = table

            rows = table.num_rows

            self.columns[dataset] = {
                "n_tokens": np.zeros(rows, dtype=np.int64),
                "n_profile_tokens": np.zeros(rows, dtype=np.int64),
                "n_event_tokens": np.zeros(rows, dtype=np.int64),
                "max_event_tokens": np.zeros(rows, dtype=np.int16),
                "events_over_token_limit": np.zeros(rows, dtype=np.int64),
                "history_over_limit": np.zeros(rows, dtype=bool),
                "n_unknown": np.zeros(rows, dtype=np.int64),
                "n_missing": np.zeros(rows, dtype=np.int64),
            }

            client_id = table.column("client_id").to_pylist()
            seq_end = table.column("seq_end").to_pylist()

            for index, (cid, end) in enumerate(zip(client_id, seq_end)):
                self.pending.setdefault(int(cid), []).append((dataset, index, int(end)))

        self.profile: dict[tuple[int, int], tuple[int, int]] = {}

    # --------------------------------------------------------

    def add_client(self, client_id: int, widths: np.ndarray, unknown: np.ndarray, missing: np.ndarray) -> None:
        """
        widths, unknown, missing идут по возрастанию seq.
        """

        requests = self.pending.pop(client_id, None)

        if not requests:
            return

        size = widths.size

        cum_tokens = np.zeros(size + 1, dtype=np.int64)
        cum_unknown = np.zeros(size + 1, dtype=np.int64)
        cum_missing = np.zeros(size + 1, dtype=np.int64)
        cum_over = np.zeros(size + 1, dtype=np.int64)
        running_max = np.zeros(size + 1, dtype=np.int64)

        if size:
            np.cumsum(widths, out=cum_tokens[1:])
            np.cumsum(unknown, out=cum_unknown[1:])
            np.cumsum(missing, out=cum_missing[1:])
            np.cumsum(widths > self.limits.max_tokens_per_event, out=cum_over[1:])
            np.maximum.accumulate(widths, out=running_max[1:])

        for dataset, index, seq_end in requests:

            if seq_end > size:
                raise ValueError(f"клиент {client_id}: seq_end {seq_end} больше числа событий {size}")

            table = self.tables[dataset]

            snapshot = table.column("snapshot_ts")[index].as_py()

            stamp = int(np.datetime64(snapshot, "us").astype(np.int64)) if snapshot is not None else None

            profile_unknown, profile_missing = self.profile.get((client_id, stamp), (0, 0))

            columns = self.columns[dataset]

            columns["n_event_tokens"][index] = cum_tokens[seq_end]
            columns["n_profile_tokens"][index] = PROFILE_WIDTH
            columns["n_tokens"][index] = cum_tokens[seq_end] + PROFILE_WIDTH
            columns["max_event_tokens"][index] = running_max[seq_end]
            columns["events_over_token_limit"][index] = cum_over[seq_end]
            columns["history_over_limit"][index] = seq_end > self.limits.max_events_per_history
            columns["n_unknown"][index] = cum_unknown[seq_end] + profile_unknown
            columns["n_missing"][index] = cum_missing[seq_end] + profile_missing

    # --------------------------------------------------------

    def finish(self, out_dir: Path) -> tuple[dict[str, int], dict[str, dict]]:

        if self.pending:
            raise ValueError(f"остались примеры без событий: клиенты {sorted(self.pending)[:5]}")

        schema = tokenized_examples_schema()

        counts: dict[str, int] = {}
        summaries: dict[str, dict] = {}

        for dataset, table in sorted(self.tables.items()):

            columns = {name: table.column(name).combine_chunks() for name in EXAMPLES_SCHEMA.names}

            for name, values in self.columns[dataset].items():
                columns[name] = pa.array(values, schema.field(name).type)

            rows = pa.table({name: columns[name] for name in schema.names}, schema=schema)

            write_table(out_dir / dataset / "examples.parquet", rows, schema)

            counts[f"{dataset}/examples"] = rows.num_rows

            summaries[dataset] = self._summary(rows)

        return counts, summaries

    def _summary(self, rows: pa.Table) -> dict:

        history = rows.column("n_events").to_numpy().astype(np.float64)
        tokens = rows.column("n_tokens").to_numpy().astype(np.float64)

        return {
            "n_examples": rows.num_rows,
            "history_length": numeric_summary(history),
            "tokens_per_example": numeric_summary(tokens),
            "histories_over_limit": int(rows.column("history_over_limit").to_numpy().sum()),
            "events_over_token_limit": int(rows.column("events_over_token_limit").to_numpy().sum()),
            "n_unknown": int(rows.column("n_unknown").to_numpy().sum()),
            "n_missing": int(rows.column("n_missing").to_numpy().sum()),
        }


def write_group(
    processed_dir: Path,
    out_dir: Path,
    group: str,
    vocab: Vocab,
    limits: ResolvedLimits,
) -> tuple[dict[str, int], dict[str, dict], TokenStats, TokenStats]:
    """
    Одна группа клиентов: события, профиль и её датасеты.
    """

    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)

    builder = ExampleBuilder(processed_dir, group, limits)

    # --------------------------------------------------------
    # ПРОФИЛЬ
    # --------------------------------------------------------

    profile_stats = TokenStats(vocab)

    profile = pq.read_table(processed_dir / "clients" / f"{group}_clients" / "profile.parquet")

    tokenized_profile = encode_profile_table(profile, vocab)

    write_table(
        out_dir / "clients" / f"{group}_clients" / "profile.parquet",
        tokenized_profile,
        tokenized_profile_schema(),
    )

    profile_stats.update(tokenized_profile, limits, None)

    builder.profile = profile_index(tokenized_profile)

    # --------------------------------------------------------
    # СОБЫТИЯ
    # --------------------------------------------------------

    event_stats = TokenStats(vocab)

    schema = tokenized_events_schema()

    writer = TableWriter(out_dir / "clients" / f"{group}_clients" / "events.parquet", schema)

    for block in iter_client_blocks(processed_dir / "clients" / f"{group}_clients" / "events.parquet"):

        client_id = block.column("client_id").to_numpy()
        ts = block.column("ts").to_numpy()
        seq = block.column("seq").to_numpy()

        check_order(client_id, ts, seq)

        tokenized = encode_events_table(block, vocab)

        writer.write(tokenized)

        event_stats.update(tokenized, limits, "event_type")

        _, flat_values, widths = flat_arrays(tokenized)

        unknown = per_row_counts(flat_values, widths, UNK_ID)
        missing = per_row_counts(flat_values, widths, MISSING_ID)

        for value, lo, hi in client_runs(client_id):
            builder.add_client(value, widths[lo:hi], unknown[lo:hi], missing[lo:hi])

    counts = {f"clients/{group}_clients/events": writer.close()}

    counts[f"clients/{group}_clients/profile"] = tokenized_profile.num_rows

    example_counts, summaries = builder.finish(out_dir)

    counts.update(example_counts)

    return counts, summaries, event_stats, profile_stats


def assert_no_mask(out_dir: Path) -> None:
    """
    Сохранённые датасеты содержат исходные ID: [MASK] ставит
    только runtime-masker и только на копии.
    """

    for path in sorted(Path(out_dir).rglob("*.parquet")):

        if "value_ids" not in pq.read_schema(path).names:
            continue

        parquet = pq.ParquetFile(path)

        # По одной группе строк: на dev в ленте 15 млн событий,
        # целиком колонка списков не помещается разумно.
        for index in range(parquet.num_row_groups):

            table = parquet.read_row_group(index, columns=["value_ids"])

            if table.num_rows == 0:
                continue

            flat = table.column("value_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False)

            if (flat == MASK_ID).any():
                raise ValueError(f"в сохранённом датасете есть [MASK]: {path}")
