from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from src.generator import config as generator_config
from src.generator.emit import SCHEMAS, schemas_for
from src.generator.profile import ALWAYS_PRESENT, FIELD_GROUPS

from .config import (
    EVENT_TYPE_PROFILE,
    EVENT_TYPES,
    LATENT_NAMES,
    NAMESPACE_TABLE,
    PROFILE_DYNAMIC_FIELDS,
    SCHEMA_VERSION,
    SOURCES,
    payload_fields,
    payload_schema,
)
from .raw import RawDataset, parse_payloads


# ============================================================
# ИДЕЯ
# ============================================================
#
# Проверка RAW перед любой обработкой. Каждая проверка это
# запись в отчёте; hard-проверки при провале останавливают
# запуск, но отчёт формируется полностью, чтобы было видно всё.
#
# Лента читается ОДИН раз и ПОТОКОВО, по одному row group.
# Ничего размером с датасет в памяти не держится:
#
#   - порядок и tie-break проверяются на соседних строках
#     с переносом хвоста между батчами;
#   - лента сверяется с типизированными таблицами блоками
#     клиентов через курсор по исходной таблице;
#   - длины историй копятся счётчиком на клиента, а не
#     массивом на событие.
#
# Ничего не исправляется и не удаляется: дубликаты и равные ts
# только считаются.
# ============================================================


STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_INFO = "info"

TABLES_WITH_TS = (
    "profile",
    "transactions",
    "product_events",
    "communications",
    "app_screens",
    "app_operations",
    "banners",
    "timeline",
)

# Таблицы, в которых считаются полные дубликаты строк.
DUPLICATE_TABLES = (
    "profile",
    "transactions",
    "product_events",
    "communications",
    "app_screens",
    "app_operations",
    "banners",
)

KEY_ORDER_SAMPLE = 5000


class ValidationError(Exception):

    def __init__(self, report: dict):
        self.report = report
        failed = [check["name"] for check in report["checks"] if check["status"] == STATUS_FAILED]
        super().__init__(f"RAW не прошёл проверки: {failed}")


@dataclass
class Check:
    name: str
    status: str
    hard: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"name": self.name, "status": self.status, "hard": self.hard, "details": self.details}


def ok(name: str, hard: bool = True, **details: Any) -> Check:
    return Check(name, STATUS_OK, hard, details)


def failed(name: str, hard: bool = True, **details: Any) -> Check:
    return Check(name, STATUS_FAILED, hard, details)


def info(name: str, **details: Any) -> Check:
    return Check(name, STATUS_INFO, False, details)


def verdict(name: str, violations: int, hard: bool = True, **details: Any) -> Check:
    details = {"violations": int(violations), **details}
    return ok(name, hard, **details) if violations == 0 else failed(name, hard, **details)


# ============================================================
# КУРСОР ПО ИСХОДНОЙ ТАБЛИЦЕ
# ============================================================


class TableCursor:
    """
    Отдаёт строки таблицы блоками клиентов в том же порядке,
    в котором клиенты идут в ленте.

    Клиент целиком лежит внутри одного row group, а row group
    идут по возрастанию client_id, поэтому достаточно тянуть
    группы, пока не наберётся весь запрошенный диапазон.
    """

    def __init__(self, raw: RawDataset, name: str):
        self._name = name
        self._schema = schemas_for(raw.manifest.revision)[name]
        self._groups: Iterator[pa.Table] = raw.iter_row_groups(name)
        self._buffer: pa.Table | None = None
        self._exhausted = False

    def _last_client(self) -> int | None:

        if self._buffer is None or self._buffer.num_rows == 0:
            return None

        return int(self._buffer.column("client_id")[-1].as_py())

    def take(self, max_client: int) -> pa.Table:
        """
        Все строки с client_id <= max_client.
        """

        while not self._exhausted:

            last = self._last_client()

            if last is not None and last > max_client:
                break

            try:
                group = next(self._groups)
            except StopIteration:
                self._exhausted = True
                break

            self._buffer = group if self._buffer is None else pa.concat_tables([self._buffer, group])

        if self._buffer is None:
            return self._schema.empty_table()

        client_id = self._buffer.column("client_id")

        head = self._buffer.filter(pc.less_equal(client_id, max_client))
        self._buffer = self._buffer.filter(pc.greater(client_id, max_client))

        return head

    def rest(self) -> pa.Table:
        """
        Всё, что осталось после последнего блока: хвост, которого
        в ленте не было.
        """

        return self.take(2 ** 62)


def canonical(table: pa.Table) -> pa.Table:
    """
    Строки в детерминированном порядке по всем колонкам: два
    равных мультимножества дают равные таблицы.
    """

    if table.num_rows == 0:
        return table

    keys = [(name, "ascending") for name in table.schema.names]

    return table.take(pc.sort_indices(table, sort_keys=keys)).combine_chunks()


# ============================================================
# ПОТОКОВЫЙ ПРОХОД ПО ЛЕНТЕ
# ============================================================


@dataclass
class TimelineScan:
    # Ревизия схемы RAW: от неё зависит ожидаемый список
    # ключей payload, и сверять его надо с контрактом того
    # датасета, который читаем, а не последнего известного.
    revision: int = 1

    rows: int = 0
    clients: int = 0

    clients_out_of_order: int = 0
    seq_not_dense: int = 0
    ts_decreasing: int = 0

    equal_ts_pairs: int = 0
    priority_inversions: int = 0
    unknown_event_types: list[str] = field(default_factory=list)

    shared_ts_events: int = 0

    latent_matches: int = 0
    nan_matches: int = 0

    key_order_violations: dict[str, int] = field(default_factory=dict)
    key_order_sampled: dict[str, int] = field(default_factory=dict)
    parse_errors: dict[str, str] = field(default_factory=dict)

    parsed_rows: dict[str, int] = field(default_factory=dict)
    table_rows: dict[str, int] = field(default_factory=dict)
    mismatched: dict[str, Any] = field(default_factory=dict)

    duplicate_rows: dict[str, int] = field(default_factory=dict)

    history_lengths: dict[int, int] = field(default_factory=dict)


def _latent_pattern() -> str:
    return '"(' + "|".join(sorted(LATENT_NAMES)) + ')":'


def _regex_count(column, pattern: str) -> int:
    matched = pc.match_substring_regex(column, pattern)
    return int(pc.sum(pc.cast(matched, pa.int64())).as_py() or 0)


def _source_view(event_type: str, timeline_rows: pa.Table, parsed: pa.Table) -> pa.Table:
    """
    Разобранный payload плюс ключи события: та же форма, что у
    исходной таблицы.
    """

    return pa.table(
        {
            "client_id": timeline_rows.column("client_id"),
            "ts": timeline_rows.column("ts"),
            **{name: parsed.column(name) for name in parsed.schema.names},
        }
    )


def scan_timeline(raw: RawDataset) -> TimelineScan:

    scan = TimelineScan(
        revision=raw.manifest.revision,
        key_order_violations={event_type: 0 for event_type in EVENT_TYPES},
        key_order_sampled={event_type: 0 for event_type in EVENT_TYPES},
        parsed_rows={event_type: 0 for event_type in EVENT_TYPES},
        table_rows={event_type: 0 for event_type in EVENT_TYPES},
    )

    latent_pattern = _latent_pattern()

    priority = raw.manifest.event_type_priority

    ranks = pa.array([priority.get(name, -1) for name in EVENT_TYPES], pa.int64())

    cursors = {event_type: TableCursor(raw, NAMESPACE_TABLE[event_type]) for event_type in EVENT_TYPES}

    unknown: set[str] = set()

    # Хвост предыдущего батча: последняя строка и текущий забег
    # одинаковых (client_id, ts).
    previous: tuple[int, np.datetime64, int] | None = None
    run: tuple[int, np.datetime64, int] | None = None

    for batch in raw.iter_row_groups("timeline"):

        if batch.num_rows == 0:
            continue

        client_id = batch.column("client_id").to_numpy()
        ts = batch.column("ts").to_numpy()
        seq = batch.column("seq").to_numpy()

        event_type_column = batch.column("event_type")

        # Код типа вместо строки: массив на 18 млн Python-строк
        # весит больше гигабайта.
        codes = pc.index_in(event_type_column, value_set=pa.array(EVENT_TYPES)).to_numpy(zero_copy_only=False)

        if np.isnan(codes.astype(np.float64)).any():
            unknown |= set(event_type_column.to_pylist()) - set(EVENT_TYPES)
            codes = np.nan_to_num(codes.astype(np.float64), nan=-1).astype(np.int64)

        rank = np.where(codes >= 0, np.asarray(ranks)[np.clip(codes, 0, len(EVENT_TYPES) - 1)], -1)

        scan.rows += batch.num_rows

        # ----------------------------------------------------
        # ПОРЯДОК
        # ----------------------------------------------------

        if previous is not None:
            head_client, head_ts, head_rank = previous
            ext_client = np.concatenate([[head_client], client_id])
            ext_ts = np.concatenate([[head_ts], ts])
            ext_rank = np.concatenate([[head_rank], rank])
        else:
            ext_client, ext_ts, ext_rank = client_id, ts, rank

        same_client = ext_client[1:] == ext_client[:-1]

        scan.clients_out_of_order += int((ext_client[1:] < ext_client[:-1]).sum())
        scan.ts_decreasing += int(((ext_ts[1:] < ext_ts[:-1]) & same_client).sum())

        equal_ts = (ext_ts[1:] == ext_ts[:-1]) & same_client

        scan.equal_ts_pairs += int(equal_ts.sum())
        scan.priority_inversions += int((equal_ts & (ext_rank[1:] < ext_rank[:-1])).sum())

        # ----------------------------------------------------
        # ПЛОТНОСТЬ SEQ И ДЛИНЫ ИСТОРИЙ
        # ----------------------------------------------------

        change = np.ones(client_id.size, dtype=bool)
        change[1:] = client_id[1:] != client_id[:-1]

        starts = np.flatnonzero(change)
        run_id = np.cumsum(change) - 1

        offsets = np.array(
            [scan.history_lengths.get(int(client_id[position]), 0) for position in starts],
            dtype=np.int64,
        )

        expected = np.arange(client_id.size) - starts[run_id] + offsets[run_id]

        scan.seq_not_dense += int((seq != expected).sum())

        lengths = np.diff(np.concatenate([starts, [client_id.size]]))

        for position, length in zip(starts, lengths):
            key = int(client_id[position])
            scan.history_lengths[key] = scan.history_lengths.get(key, 0) + int(length)

        # ----------------------------------------------------
        # ЗАБЕГИ ОДИНАКОВЫХ TS
        # ----------------------------------------------------

        run = _count_shared_ts(scan, client_id, ts, run)

        previous = (int(client_id[-1]), ts[-1], int(rank[-1]))

        # ----------------------------------------------------
        # PAYLOAD
        # ----------------------------------------------------

        payload = batch.column("payload")

        scan.latent_matches += _regex_count(payload, latent_pattern)
        scan.nan_matches += _regex_count(payload, r"\b(NaN|Infinity)\b")

        max_client = int(client_id[-1])

        for event_type in EVENT_TYPES:

            rows = batch.filter(pc.equal(event_type_column, event_type))

            source = cursors[event_type].take(max_client)

            scan.parsed_rows[event_type] += rows.num_rows
            scan.table_rows[event_type] += source.num_rows

            if rows.num_rows == 0 and source.num_rows == 0:
                continue

            _sample_key_order(scan, event_type, rows)

            try:
                parsed = parse_payloads(event_type, rows.column("payload"))
            except Exception as error:  # noqa: BLE001
                scan.parse_errors.setdefault(event_type, f"{type(error).__name__}: {error}")
                continue

            _compare_block(scan, event_type, _source_view(event_type, rows, parsed), source)

    # Хвост незакрытого забега.
    if run is not None and run[2] >= 2:
        scan.shared_ts_events += run[2]

    # Строки таблиц, которых в ленте не оказалось вовсе.
    for event_type, cursor in cursors.items():
        rest = cursor.rest()
        if rest.num_rows:
            scan.table_rows[event_type] += rest.num_rows
            scan.mismatched.setdefault(event_type, {})["rows_after_timeline"] = rest.num_rows

    scan.clients = len(scan.history_lengths)
    scan.unknown_event_types = sorted(unknown)

    return scan


def _count_shared_ts(scan: TimelineScan, client_id: np.ndarray, ts: np.ndarray, run) -> tuple:
    """
    Считает события, у которых внутри клиента есть сосед с тем же
    ts. Забег может переходить через границу батча.
    """

    change = np.ones(client_id.size, dtype=bool)
    change[1:] = (client_id[1:] != client_id[:-1]) | (ts[1:] != ts[:-1])

    starts = np.flatnonzero(change)
    lengths = np.diff(np.concatenate([starts, [client_id.size]]))

    for index, (position, length) in enumerate(zip(starts, lengths)):

        key = (int(client_id[position]), ts[position])

        if index == 0 and run is not None and (run[0], run[1]) == key:
            run = (run[0], run[1], run[2] + int(length))
            continue

        if run is not None and run[2] >= 2:
            scan.shared_ts_events += run[2]

        run = (key[0], key[1], int(length))

    return run


def _sample_key_order(scan: TimelineScan, event_type: str, rows: pa.Table) -> None:

    budget = KEY_ORDER_SAMPLE - scan.key_order_sampled[event_type]

    if budget <= 0 or rows.num_rows == 0:
        return

    expected = list(payload_fields(event_type, scan.revision))

    for text in rows.column("payload").slice(0, budget).to_pylist():
        scan.key_order_sampled[event_type] += 1
        if list(json.loads(text).keys()) != expected:
            scan.key_order_violations[event_type] += 1


def _compare_block(scan: TimelineScan, event_type: str, from_timeline: pa.Table, source: pa.Table) -> None:
    """
    Сверяет один блок клиентов: мультимножества строк обязаны
    совпасть.
    """

    if event_type in scan.mismatched:
        return

    if event_type == EVENT_TYPE_PROFILE:
        source = source.select(["client_id", "ts", *PROFILE_DYNAMIC_FIELDS])

    if from_timeline.num_rows != source.num_rows:
        scan.mismatched[event_type] = {
            "timeline_rows_in_block": from_timeline.num_rows,
            "table_rows_in_block": source.num_rows,
        }
        return

    if source.num_rows == 0:
        return

    left = canonical(from_timeline.select(source.schema.names).cast(source.schema))

    if not left.equals(canonical(source)):
        scan.mismatched[event_type] = {"reason": "содержимое блока отличается при равном числе строк"}


# ============================================================
# ПРОВЕРКИ ТАБЛИЦ
# ============================================================


def check_schemas(raw: RawDataset) -> Check:

    expected = schemas_for(raw.manifest.revision)

    missing = [name for name in expected if not raw.exists(name)]

    mismatched = {}

    for name, schema in expected.items():

        if name in missing:
            continue

        actual = raw.schema(name)

        if not actual.equals(schema):
            mismatched[name] = {"expected": schema.names, "actual": actual.names}

    return verdict(
        "schemas_match",
        len(missing) + len(mismatched),
        revision=raw.manifest.revision,
        tables=len(expected),
        missing=missing,
        mismatched=mismatched,
    )


def check_manifest_rows(raw: RawDataset) -> Check:

    mismatched = {}

    for name in SCHEMAS:
        expected = raw.manifest.rows.get(name)
        actual = raw.num_rows(name)
        if expected != actual:
            mismatched[name] = {"manifest": expected, "actual": actual}

    return verdict("manifest_rows_match", len(mismatched), mismatched=mismatched)


def check_latent_columns(raw: RawDataset) -> Check:

    leaked = {}

    for name in SCHEMAS:
        found = sorted(set(raw.schema(name).names) & LATENT_NAMES)
        if found:
            leaked[name] = found

    return verdict("latent_columns_absent", len(leaked), leaked=leaked)


def check_timestamps(raw: RawDataset) -> Check:

    feature_end = np.datetime64(raw.manifest.feature_end, "us")

    violations = 0
    per_table = {}

    for name in TABLES_WITH_TS:

        late = 0
        nulls = 0
        rows = 0

        for batch in raw.iter_row_groups(name, ["ts"]):

            ts = batch.column("ts")

            rows += len(ts)
            nulls += ts.null_count

            values = pc.drop_null(ts).to_numpy()

            if values.size:
                late += int((values >= feature_end).sum())

        violations += late + nulls

        per_table[name] = {"rows": rows, "null_ts": nulls, "at_or_after_feature_end": late}

    return verdict("timestamps_before_feature_end", violations, tables=per_table)


def check_availability(raw: RawDataset) -> Check:

    problems = {}

    for source in SOURCES:

        availability = raw.manifest.source_availability.get(source)

        if availability is None:
            problems[source] = "нет в manifest.source_availability"
            continue

        boundary = np.datetime64(availability, "us")

        earliest = None

        for batch in raw.iter_row_groups(source, ["ts"]):
            values = pc.drop_null(batch.column("ts")).to_numpy()
            if values.size:
                current = values.min()
                earliest = current if earliest is None else min(earliest, current)

        if earliest is not None and earliest < boundary:
            problems[source] = {"min_ts": str(earliest), "availability_start": availability.isoformat()}

    return verdict("availability_respected", len(problems), sources=problems)


def check_first_seen(raw: RawDataset) -> Check:

    coverage = raw.read("source_coverage")

    problems = {}

    for source in SOURCES:

        table = raw.read(source, ["client_id", "ts"])

        grouped = table.group_by("client_id").aggregate([("ts", "min"), ("ts", "count")])

        rows = coverage.filter(pc.equal(coverage.column("source"), source)).select(["client_id", "first_seen"])

        joined = rows.join(grouped, keys="client_id", join_type="left outer")

        first_seen = joined.column("first_seen").to_numpy(zero_copy_only=False)
        ts_min = joined.column("ts_min").to_numpy(zero_copy_only=False)
        ts_count = pc.fill_null(joined.column("ts_count"), 0).to_numpy(zero_copy_only=False)

        never_seen = np.isnat(first_seen)
        seen_with_rows = (~never_seen) & (~np.isnat(ts_min))

        early = int((ts_min[seen_with_rows] < first_seen[seen_with_rows]).sum())
        ghosts = int((ts_count[never_seen] > 0).sum())

        known = set(rows.column("client_id").to_pylist())
        unknown = len(set(grouped.column("client_id").to_pylist()) - known)

        if early or ghosts or unknown:
            problems[source] = {
                "events_before_first_seen": early,
                "never_seen_with_rows": ghosts,
                "unknown_clients": unknown,
            }

    return verdict("first_seen_respected", len(problems), sources=problems)


def check_coverage_rows(raw: RawDataset) -> Check:

    coverage = raw.read("source_coverage")

    per_client = coverage.group_by("client_id").aggregate([("source", "count_distinct"), ("source", "count")])

    distinct = per_client.column("source_count_distinct").to_numpy()
    total = per_client.column("source_count").to_numpy()

    bad_shape = int(((distinct != len(SOURCES)) | (total != len(SOURCES))).sum())

    unknown_sources = sorted(set(coverage.column("source").to_pylist()) - set(SOURCES))

    availability = coverage.column("availability_start").to_numpy(zero_copy_only=False)
    first_seen = coverage.column("first_seen").to_numpy(zero_copy_only=False)
    sources = np.asarray(coverage.column("source").to_pylist(), dtype=object)

    expected = (
        np.array(
            [
                np.datetime64(raw.manifest.source_availability.get(source, raw.manifest.history_start), "us")
                for source in sources
            ],
            dtype="datetime64[us]",
        )
        if len(sources)
        else np.array([], dtype="datetime64[us]")
    )

    wrong_availability = int((availability != expected).sum()) if len(sources) else 0

    seen = ~np.isnat(first_seen)
    before_availability = int((first_seen[seen] < availability[seen]).sum())

    return verdict(
        "coverage_rows_complete",
        bad_shape + len(unknown_sources) + wrong_availability + before_availability,
        clients=per_client.num_rows,
        bad_shape=bad_shape,
        unknown_sources=unknown_sources,
        wrong_availability=wrong_availability,
        first_seen_before_availability=before_availability,
    )


def check_profile_grid(raw: RawDataset) -> Check:

    profile = raw.read("profile", ["client_id", "ts", "snapshot_month"])

    distinct = profile.group_by(["client_id", "snapshot_month"]).aggregate([]).num_rows

    duplicates = profile.num_rows - distinct

    month = profile.column("snapshot_month").to_numpy()
    ts = profile.column("ts").to_numpy()

    month_floor = month.astype("datetime64[M]")

    not_month_start = int((month_floor.astype("datetime64[us]") != month).sum())

    expected_ts = (month_floor + np.timedelta64(1, "M")).astype("datetime64[us]") - np.timedelta64(1, "s")

    wrong_ts = int((expected_ts != ts).sum())

    return verdict(
        "profile_monthly_grid",
        duplicates + not_month_start + wrong_ts,
        rows=profile.num_rows,
        duplicate_client_months=duplicates,
        snapshot_month_not_month_start=not_month_start,
        ts_not_month_end=wrong_ts,
    )


def check_timestamp_quality(raw: RawDataset) -> Check:

    allowed = {"exact", "date_only"}

    unknown = 0
    not_midnight = 0
    date_only = 0
    rows = 0

    for batch in raw.iter_row_groups("product_events", ["ts", "timestamp_quality"]):

        quality = np.asarray(batch.column("timestamp_quality").to_pylist(), dtype=object)
        ts = batch.column("ts").to_numpy()

        rows += batch.num_rows
        unknown += int(sum(1 for value in quality if value not in allowed))

        mask = quality == "date_only"
        date_only += int(mask.sum())

        midnight = ts.astype("datetime64[D]").astype("datetime64[us]") == ts

        not_midnight += int((mask & ~midnight).sum())

    return verdict(
        "timestamp_quality_consistent",
        unknown + not_midnight,
        rows=rows,
        date_only=date_only,
        unknown_values=unknown,
        date_only_not_midnight=not_midnight,
    )


def check_labels(raw: RawDataset) -> Check:

    labels = raw.read("labels")

    client_ids = labels.column("client_id")

    duplicates = len(client_ids) - len(pc.unique(client_ids))

    wrong_count = 0 if len(client_ids) == raw.manifest.total_clients else 1

    start = labels.column("label_start").to_numpy(zero_copy_only=False)
    end = labels.column("label_end").to_numpy(zero_copy_only=False)

    wrong_start = int((start != np.datetime64(raw.manifest.feature_end, "us")).sum())
    wrong_end = int((end != np.datetime64(raw.manifest.label_end, "us")).sum())

    return verdict(
        "labels_shape",
        duplicates + wrong_count + wrong_start + wrong_end,
        rows=labels.num_rows,
        duplicate_clients=duplicates,
        wrong_client_count=wrong_count,
        label_start_mismatch=wrong_start,
        label_end_mismatch=wrong_end,
    )


def check_manifest_vs_generator(raw: RawDataset) -> Check:
    """
    Информационно: RAW мог быть собран другой версией config.
    """

    manifest = raw.manifest

    mismatches = {}

    if manifest.history_start != generator_config.HISTORY_START:
        mismatches["history_start"] = manifest.history_start.isoformat()

    if manifest.feature_end != generator_config.FEATURE_END:
        mismatches["feature_end"] = manifest.feature_end.isoformat()

    if manifest.max_events_per_history != generator_config.MAX_EVENTS_PER_HISTORY:
        mismatches["max_events_per_history"] = manifest.max_events_per_history

    for source, ts in generator_config.SOURCE_AVAILABILITY.items():
        if manifest.source_availability.get(source) != ts:
            mismatches[f"source_availability.{source}"] = str(manifest.source_availability.get(source))

    return info("manifest_matches_generator_config", mismatches=mismatches, matches=not mismatches)


# ============================================================
# ПРОВЕРКИ ЛЕНТЫ
# ============================================================


def check_timeline_order(scan: TimelineScan) -> Check:

    return verdict(
        "timeline_sorted_and_dense",
        scan.clients_out_of_order + scan.seq_not_dense + scan.ts_decreasing,
        rows=scan.rows,
        clients=scan.clients,
        clients_out_of_order=scan.clients_out_of_order,
        seq_not_dense=scan.seq_not_dense,
        ts_decreasing_within_client=scan.ts_decreasing,
    )


def check_tie_break(scan: TimelineScan) -> Check:

    return verdict(
        "timeline_tie_break_priority",
        scan.priority_inversions + len(scan.unknown_event_types),
        equal_ts_pairs=scan.equal_ts_pairs,
        priority_inversions=scan.priority_inversions,
        unknown_event_types=scan.unknown_event_types,
    )


def check_payload_parses(scan: TimelineScan) -> Check:
    """
    Разбор с явной схемой: лишний ключ в payload это ошибка.
    """

    return verdict("payload_parses", len(scan.parse_errors), errors=scan.parse_errors)


def check_payload_keys(scan: TimelineScan) -> Check:

    return verdict(
        "payload_keys_match_contract",
        sum(scan.key_order_violations.values()),
        sampled=scan.key_order_sampled,
        violations_by_type=scan.key_order_violations,
        note="полный разбор с explicit_schema дополнительно запрещает лишние ключи во всех строках",
    )


def check_payload_latent(scan: TimelineScan) -> Check:
    return verdict("latent_payload_keys_absent", scan.latent_matches, matches=scan.latent_matches)


def check_payload_nan(scan: TimelineScan) -> Check:
    return verdict("payload_no_nan", scan.nan_matches, matches=scan.nan_matches)


def check_timeline_matches_tables(scan: TimelineScan) -> Check:

    return verdict(
        "timeline_matches_tables",
        len(scan.mismatched),
        rows=scan.parsed_rows,
        table_rows=scan.table_rows,
        mismatched=scan.mismatched,
    )


# ============================================================
# СВОДКА
# ============================================================


def count_duplicates(raw: RawDataset, name: str) -> int:
    """
    Полные дубликаты строк. Считаются внутри row group, и это
    точно: одинаковые строки делят client_id, а клиент целиком
    лежит в одном чанке.
    """

    duplicates = 0

    for batch in raw.iter_row_groups(name):

        if batch.num_rows == 0:
            continue

        distinct = batch.group_by(batch.schema.names).aggregate([]).num_rows

        duplicates += batch.num_rows - distinct

    return duplicates


def summarize(raw: RawDataset, scan: TimelineScan) -> dict[str, Any]:

    summary: dict[str, Any] = {}

    summary["rows"] = {name: raw.num_rows(name) for name in SCHEMAS}

    summary["exact_duplicate_rows"] = {name: count_duplicates(raw, name) for name in DUPLICATE_TABLES}

    if scan.rows:

        summary["equal_ts"] = {
            "events_with_shared_ts_share": scan.shared_ts_events / scan.rows,
            "adjacent_pairs_share": scan.equal_ts_pairs / scan.rows,
        }

        lengths = np.array(sorted(scan.history_lengths.values()), dtype=np.int64)

        limit = raw.manifest.max_events_per_history

        summary["history_length"] = {
            "clients_with_events": int(lengths.size),
            "min": int(lengths.min()),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
            "max_events_per_history": limit,
            "clients_over_limit": int((lengths > limit).sum()),
        }

    coverage = raw.read("source_coverage")

    seen = {}

    for source in SOURCES:
        rows = coverage.filter(pc.equal(coverage.column("source"), source))
        first_seen = rows.column("first_seen")
        seen[source] = float((len(first_seen) - first_seen.null_count) / len(first_seen)) if len(first_seen) else 0.0

    summary["source_coverage_share"] = seen

    profile = raw.read("profile")

    if profile.num_rows:

        null_share = {
            name: float(profile.column(name).null_count / profile.num_rows)
            for name in profile.schema.names[3:]
        }

        group_missing = {}

        for group, fields in FIELD_GROUPS.items():
            all_null = np.ones(profile.num_rows, dtype=bool)
            for name in fields:
                all_null &= pc.is_null(profile.column(name)).to_numpy(zero_copy_only=False)
            group_missing[group] = float(all_null.mean())

        summary["profile_missing"] = {
            "field_null_share": null_share,
            "group_all_null_share": group_missing,
            "always_present_null_share": {name: null_share[name] for name in ALWAYS_PRESENT},
        }

    return summary


# ============================================================
# ЗАПУСК
# ============================================================


def _guarded(name: str, function: Callable[[], Check]) -> Check:

    try:
        return function()
    except Exception as error:  # noqa: BLE001 — отчёт должен быть полным
        return failed(name, error=f"{type(error).__name__}: {error}")


def validate_raw(raw: RawDataset) -> dict:
    """
    Полный отчёт о проверках. При провале hard-проверки
    поднимает ValidationError с этим же отчётом.
    """

    checks: list[Check] = []

    checks.append(_guarded("schemas_match", lambda: check_schemas(raw)))
    checks.append(_guarded("manifest_rows_match", lambda: check_manifest_rows(raw)))
    checks.append(_guarded("latent_columns_absent", lambda: check_latent_columns(raw)))
    checks.append(_guarded("timestamps_before_feature_end", lambda: check_timestamps(raw)))
    checks.append(_guarded("availability_respected", lambda: check_availability(raw)))
    checks.append(_guarded("first_seen_respected", lambda: check_first_seen(raw)))
    checks.append(_guarded("coverage_rows_complete", lambda: check_coverage_rows(raw)))
    checks.append(_guarded("profile_monthly_grid", lambda: check_profile_grid(raw)))
    checks.append(_guarded("timestamp_quality_consistent", lambda: check_timestamp_quality(raw)))
    checks.append(_guarded("labels_shape", lambda: check_labels(raw)))
    checks.append(_guarded("manifest_matches_generator_config", lambda: check_manifest_vs_generator(raw)))

    summary: dict[str, Any] = {}

    try:
        scan = scan_timeline(raw)
    except Exception as error:  # noqa: BLE001
        checks.append(failed("timeline_scan", error=f"{type(error).__name__}: {error}"))
        scan = None

    if scan is not None:
        checks.append(_guarded("timeline_sorted_and_dense", lambda: check_timeline_order(scan)))
        checks.append(_guarded("timeline_tie_break_priority", lambda: check_tie_break(scan)))
        checks.append(_guarded("payload_parses", lambda: check_payload_parses(scan)))
        checks.append(_guarded("payload_keys_match_contract", lambda: check_payload_keys(scan)))
        checks.append(_guarded("latent_payload_keys_absent", lambda: check_payload_latent(scan)))
        checks.append(_guarded("payload_no_nan", lambda: check_payload_nan(scan)))
        checks.append(_guarded("timeline_matches_tables", lambda: check_timeline_matches_tables(scan)))

        try:
            summary = summarize(raw, scan)
        except Exception as error:  # noqa: BLE001
            checks.append(failed("summary", hard=False, error=f"{type(error).__name__}: {error}"))

    hard_failures = [check for check in checks if check.hard and check.status == STATUS_FAILED]

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_FAILED if hard_failures else STATUS_OK,
        "checks": [check.to_json() for check in checks],
        "summary": summary,
    }

    if hard_failures:
        raise ValidationError(report)

    return report
