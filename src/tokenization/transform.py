from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from src.preprocessing.artifacts import TableWriter, dumps_json, read_json, sha256_file, write_json, write_text
from src.preprocessing.history import HistoryError
from src.preprocessing.semantic.chains import ChainsError
from src.preprocessing.semantic.keys import KeysError

from .corpus import CorpusError, GroupCorpus
from .encode import (
    EncodeError,
    absent_reasons,
    decode_record,
    encode_event,
    encode_profile,
    event_identity,
    profile_known,
    provenance,
    references,
)
from .layout import EMPTY, INVALID, MISSING, UNK, FrozenArtifacts
from .contract import COMPATIBILITY_REPORT_FILE, FIT_MANIFEST_FILE
from .report import render_compatibility_md, render_golden_md, render_tokenization_md
from .settings import TokenizerConfig
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Замороженный transform: словари уже построены, и этот этап их
# только применяет. Ни одного нового токена, ни одной новой
# границы, ни одного пересчитанного частотного порога.
#
# Срезы задаются явно. Расписание срезов, доступная история и
# периоды целей это работа будущего построителя датасета, и
# здесь её нет: сохранённая токенизация ещё не обучающий набор.
#
# Клиент без событий остаётся в индексе, а событий на дату может
# быть ноль: выдумывать покупки и сессии токенизатор не вправе.
# ============================================================


STAGE = "encode"

EVENTS_FILE = "events.parquet"
PROFILES_FILE = "profiles.parquet"
CLIENTS_FILE = "clients.parquet"
MANIFEST_FILE = "tokenized_manifest.json"
REPORT_JSON_FILE = "tokenization_report.json"
REPORT_MD_FILE = "tokenization_report.md"
GOLDEN_JSON_FILE = "golden_examples.json"
GOLDEN_MD_FILE = "golden_examples.md"

# Колонки, которые идут в модель, и колонки, которые едут рядом.
MODEL_INPUT_COLUMNS: tuple[str, ...] = ("key_ids", "value_ids", "positions")
SPAN_COLUMNS: tuple[str, ...] = ("value_starts", "value_lengths", "value_keys")
CHANNEL_COLUMNS: tuple[str, ...] = ("calendar",)
METADATA_COLUMNS: tuple[str, ...] = (
    "client_id",
    "cutoff",
    "event_id",
    "stable_event_index",
    "event_time",
    "source",
    "event_type",
    "n_values",
    "n_tokens",
    "refs",
    "provenance",
    "absent_reasons",
    "unknown_keys",
)

EVENTS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("cutoff", pa.timestamp("us")),
        ("event_id", pa.string()),
        ("stable_event_index", pa.int64()),
        ("event_time", pa.timestamp("us")),
        ("source", pa.string()),
        ("event_type", pa.string()),
        ("n_values", pa.int32()),
        ("n_tokens", pa.int32()),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("value_starts", pa.list_(pa.int32())),
        ("value_lengths", pa.list_(pa.int32())),
        ("value_keys", pa.list_(pa.string())),
        ("refs", pa.string()),
        ("provenance", pa.string()),
        ("absent_reasons", pa.string()),
        ("unknown_keys", pa.list_(pa.string())),
        ("calendar", pa.list_(pa.float64())),
    ]
)

PROFILES_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("cutoff", pa.timestamp("us")),
        ("profile_state", pa.string()),
        ("n_values", pa.int32()),
        ("n_tokens", pa.int32()),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("value_starts", pa.list_(pa.int32())),
        ("value_lengths", pa.list_(pa.int32())),
        ("value_keys", pa.list_(pa.string())),
    ]
)

CLIENTS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("cutoff", pa.timestamp("us")),
        ("n_events", pa.int64()),
        ("n_values", pa.int64()),
        ("n_tokens", pa.int64()),
        ("has_profile", pa.bool_()),
        ("limitations", pa.list_(pa.string())),
    ]
)


class TransformError(ValueError):
    """
    Кодирование выполнить нельзя.
    """


@dataclass
class Counters:
    events: int = 0
    # values и tokens это ИТОГ по записям обоих видов; рядом
    # лежат доли, чтобы было видно, из чего он сложился.
    values: int = 0
    tokens: int = 0
    event_values: int = 0
    event_tokens_total: int = 0
    profile_values: int = 0
    profile_tokens: int = 0
    # Строка это пара «клиент, срез»: у одного клиента их
    # столько, сколько срезов заказали.
    slices: int = 0
    clients: set[str] = field(default_factory=set)
    silent_clients: int = 0
    without_profile: int = 0
    profiles: int = 0
    missing: int = 0
    unknown: int = 0
    invalid: int = 0
    empty: int = 0
    unknown_keys: dict[str, int] = field(default_factory=dict)
    edge_buckets: dict[str, int] = field(default_factory=dict)
    event_tokens: list[int] = field(default_factory=list)
    text_pieces: list[int] = field(default_factory=list)


@dataclass
class TransformResult:
    report: dict
    outputs: list[Path] = field(default_factory=list)
    directory: Path | None = None


def _count_specials(artifacts: FrozenArtifacts, record, counters: Counters) -> None:

    specials = {
        artifacts.special(MISSING): "missing",
        artifacts.special(UNK): "unknown",
        artifacts.special(INVALID): "invalid",
        artifacts.special(EMPTY): "empty",
    }

    for value_id in record.value_ids:
        name = specials.get(value_id)
        if name is not None:
            setattr(counters, name, getattr(counters, name) + 1)


def _count_edges(artifacts: FrozenArtifacts, record, counters: Counters) -> None:
    """
    Значения, попавшие в крайние диапазоны: их доля говорит,
    насколько шкала подходит данным.
    """

    first_value = artifacts.first_value_id

    for key, start in zip(record.value_keys, record.value_starts):

        value_id = record.value_ids[start]

        if value_id < first_value or key not in artifacts.encoders:
            continue

        ids = artifacts.candidates.get(key, {}).get("value_ids") or []

        if not ids:
            continue

        if value_id in (ids[0], ids[-1]):
            counters.edge_buckets[key] = counters.edge_buckets.get(key, 0) + 1


def transform_group(
    artifacts: FrozenArtifacts,
    corpus: GroupCorpus,
    cutoffs: list[datetime],
    directory: Path,
    config: TokenizerConfig,
    clients: list[str] | None = None,
    golden_limit: int = 6,
) -> TransformResult:
    """
    Кодирует выбранные срезы группы замороженными артефактами.
    """

    directory = Path(directory)

    # Конфигурация это часть замороженного комплекта: она решает,
    # в том числе, предел кусков в значении. Кодировать одним
    # словарём по правилам другого нельзя.
    if config.sha256() != artifacts.manifest["config_sha256"]:
        raise TransformError(
            "конфигурация не та, которой заморожен словарь: кодирование пошло бы по одним "
            "правилам, а словарь построен по другим. Возьмите tokenizer_config.json из каталога "
            "артефактов"
        )

    wanted = list(clients or corpus.client_ids)

    unknown_clients = sorted(set(wanted) - set(corpus.client_ids))

    if unknown_clients:
        raise TransformError(
            "клиентов нет в группе " + corpus.group + ": " + ", ".join(unknown_clients[:5])
        )

    limitations: set[str] = set()
    skipped: list[dict] = []

    usable = []

    for cutoff in cutoffs:
        if cutoff > corpus.period_end:
            skipped.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "reason": "срез позже границы выгрузки: такой истории ещё не существует",
                }
            )
        else:
            usable.append(cutoff)

    counters = Counters()

    events_writer = TableWriter(directory / EVENTS_FILE, EVENTS_SCHEMA)
    profiles_writer = TableWriter(directory / PROFILES_FILE, PROFILES_SCHEMA)
    clients_writer = TableWriter(directory / CLIENTS_FILE, CLIENTS_SCHEMA)

    golden: list[dict] = []

    try:
        for cutoff in usable:

            for client_id in wanted:

                try:
                    history = corpus.history(client_id, cutoff)
                except (HistoryError, ChainsError, KeysError) as error:
                    raise TransformError(
                        f"клиент {client_id} на срезе {cutoff.isoformat()}: {error}"
                    ) from error

                identity = event_identity(history)

                reason_of_event = {
                    item.stable_event_index: item.reason
                    for item in history.relations
                    if getattr(item, "reason", None)
                }

                rows: list[dict] = []
                client_values = 0
                client_tokens = 0

                for event in history.events:

                    record = encode_event(artifacts, event, config.max_pieces_per_value)

                    _count_specials(artifacts, record, counters)
                    _count_edges(artifacts, record, counters)

                    counters.events += 1
                    counters.values += record.n_values
                    counters.tokens += record.n_tokens
                    counters.event_values += record.n_values
                    counters.event_tokens_total += record.n_tokens
                    counters.event_tokens.append(record.n_tokens)

                    for key, length in zip(record.value_keys, record.value_lengths):
                        if artifacts.key_info.get(key, {}).get("value_kind") == "text":
                            counters.text_pieces.append(length)

                    for key in record.unknown_keys:
                        counters.unknown_keys[key] = counters.unknown_keys.get(key, 0) + 1

                    client_values += record.n_values
                    client_tokens += record.n_tokens

                    rows.append(
                        {
                            "client_id": history.client_id,
                            "cutoff": cutoff,
                            "event_id": event.event_id,
                            "stable_event_index": event.stable_event_index,
                            "event_time": event.event_time,
                            "source": event.source,
                            "event_type": event.values.get("event_type"),
                            "n_values": record.n_values,
                            "n_tokens": record.n_tokens,
                            "key_ids": record.key_ids,
                            "value_ids": record.value_ids,
                            "positions": record.positions,
                            "value_starts": record.value_starts,
                            "value_lengths": record.value_lengths,
                            "value_keys": record.value_keys,
                            "refs": dumps_json(references(artifacts, event)).strip(),
                            "provenance": dumps_json(provenance(event, identity)).strip(),
                            "absent_reasons": dumps_json(
                                absent_reasons(event, reason_of_event.get(event.stable_event_index))
                            ).strip(),
                            "unknown_keys": record.unknown_keys,
                            "calendar": list(event.calendar),
                        }
                    )

                if len(rows) != history.n_events:
                    raise TransformError(
                        f"клиент {client_id}: закодировано {len(rows)} событий из {history.n_events}"
                    )

                if rows:
                    events_writer.write(pa.Table.from_pylist(rows, schema=EVENTS_SCHEMA))
                else:
                    counters.silent_clients += 1

                # --- профиль ---

                record = encode_profile(artifacts, history, config.max_pieces_per_value)

                _count_specials(artifacts, record, counters)

                meta = history.profile_meta or {}

                # Анкета есть или её нет — это СОСТОЯНИЕ, а не
                # пустота словаря значений: у известной версии
                # все поля могут оказаться незаполненными.
                known_profile = profile_known(history)

                if known_profile:
                    counters.profiles += 1
                else:
                    counters.without_profile += 1

                # Профиль это тоже закодированная запись. Раньше
                # его значения попадали в строку клиента, но не в
                # итог группы, и две величины, обязанные сходиться,
                # молча расходились.
                counters.values += record.n_values
                counters.tokens += record.n_tokens
                counters.profile_values += record.n_values
                counters.profile_tokens += record.n_tokens

                client_values += record.n_values
                client_tokens += record.n_tokens

                profiles_writer.write(
                    pa.Table.from_pylist(
                        [
                            {
                                "client_id": history.client_id,
                                "cutoff": cutoff,
                                "profile_state": meta.get("state"),
                                "n_values": record.n_values,
                                "n_tokens": record.n_tokens,
                                "key_ids": record.key_ids,
                                "value_ids": record.value_ids,
                                "positions": record.positions,
                                "value_starts": record.value_starts,
                                "value_lengths": record.value_lengths,
                                "value_keys": record.value_keys,
                            }
                        ],
                        schema=PROFILES_SCHEMA,
                    )
                )

                counters.slices += 1
                counters.clients.add(history.client_id)

                clients_writer.write(
                    pa.Table.from_pylist(
                        [
                            {
                                "client_id": history.client_id,
                                "cutoff": cutoff,
                                "n_events": history.n_events,
                                "n_values": client_values,
                                "n_tokens": client_tokens,
                                "has_profile": known_profile,
                                "limitations": list(history.limitations),
                            }
                        ],
                        schema=CLIENTS_SCHEMA,
                    )
                )

                limitations.update(history.limitations)

                if len(golden) < golden_limit and history.events and cutoff == usable[-1]:
                    golden.extend(
                        _golden(artifacts, history, config, golden_limit - len(golden), cutoff)
                    )

    finally:
        rows_events = events_writer.close()
        rows_profiles = profiles_writer.close()
        rows_clients = clients_writer.close()

    # --- отчёт ---

    report = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "artifact_id": artifacts.manifest["artifact_id"],
        "group": corpus.group,
        "cutoffs": [cutoff.isoformat() for cutoff in usable],
        "skipped_cutoffs": skipped,
        "readiness": artifacts.manifest["readiness"],
        "rows": {"events": rows_events, "profiles": rows_profiles, "clients": rows_clients},
        "counts": {
            "clients": len(counters.clients),
            "client_slices": counters.slices,
            "events": counters.events,
            "values": counters.values,
            "tokens": counters.tokens,
            "event_values": counters.event_values,
            "event_tokens": counters.event_tokens_total,
            "profile_values": counters.profile_values,
            "profile_tokens": counters.profile_tokens,
            "profiles": counters.profiles,
            "clients_without_profile": counters.without_profile,
            "silent_clients": counters.silent_clients,
        },
        "specials": {
            "missing": counters.missing,
            "unknown": counters.unknown,
            "invalid": counters.invalid,
            "empty": counters.empty,
        },
        "unknown_keys": dict(sorted(counters.unknown_keys.items())),
        "edge_buckets": dict(sorted(counters.edge_buckets.items())),
        "lengths": {
            "event_tokens": _lengths(counters.event_tokens),
            "text_pieces": _lengths(counters.text_pieces),
        },
        "limitations": sorted(limitations),
        "contract": {
            "marker_owner": "tokenizer",
            "pair_order": "ведущий маркер, затем пары по возрастанию key_id; порядок смысла не несёт",
            "positions": "номер куска внутри значения, а не порядок полей события",
            "model_input_columns": list(MODEL_INPUT_COLUMNS),
            "span_columns": list(SPAN_COLUMNS),
            "channel_columns": list(CHANNEL_COLUMNS),
            "metadata_columns": list(METADATA_COLUMNS),
            "truncation": "ничего не обрезается: значение сверх предела кусков это ошибка",
            "dataset": "это ещё не обучающий датасет: расписание срезов назначает построитель",
        },
    }

    outputs = [directory / EVENTS_FILE, directory / PROFILES_FILE, directory / CLIENTS_FILE]

    manifest = {
        **{key: report[key] for key in ("stage", "schema_version", "format_version",
                                        "implementation_version", "artifact_id", "group",
                                        "cutoffs", "readiness", "rows", "contract")},
        "vocab_directory": artifacts.directory.name,
        "artifacts_sha256": artifacts.manifest["artifacts"],
        "files_sha256": {path.name: sha256_file(path) for path in outputs},
        "fit_content_sha256": artifacts.manifest["fit_content_sha256"],
        "config_sha256": artifacts.manifest["config_sha256"],
    }

    path = directory / MANIFEST_FILE
    write_json(path, manifest)
    outputs.append(path)

    path = directory / REPORT_JSON_FILE
    write_json(path, report)
    outputs.append(path)

    path = directory / REPORT_MD_FILE
    write_text(path, render_tokenization_md(report))
    outputs.append(path)

    path = directory / GOLDEN_JSON_FILE
    write_json(path, golden)
    outputs.append(path)

    path = directory / GOLDEN_MD_FILE
    write_text(path, render_golden_md(golden))
    outputs.append(path)

    # Окончательная редакция отчёта совместимости: она опирается
    # на реальные записи, а не на намерение.
    path = directory / COMPATIBILITY_REPORT_FILE
    write_text(path, render_compatibility_md(read_json(artifacts.directory / FIT_MANIFEST_FILE), report))
    outputs.append(path)

    # Артефакты словаря обязаны быть теми же и ПОСЛЕ работы:
    # transform ничего не дообучает, и это проверяется, а не
    # обещается.
    artifacts.verify()

    return TransformResult(report=report, outputs=outputs, directory=directory)


def _lengths(values: list[int]) -> dict:

    if not values:
        return {"count": 0, "mean": 0, "median": 0, "max": 0}

    ordered = sorted(values)

    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 2),
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def _golden(artifacts: FrozenArtifacts, history, config: TokenizerConfig, limit: int,
            cutoff: datetime) -> list[dict]:
    """
    Читаемые примеры: исходное значение, смысл, ID, расшифровка.
    """

    out: list[dict] = []

    identity = event_identity(history)

    reason_of_event = {
        item.stable_event_index: item.reason
        for item in history.relations
        if getattr(item, "reason", None)
    }

    # Разные типы событий интереснее, чем подряд идущие покупки.
    seen: set[str] = set()

    for event in history.events:

        event_type = event.values.get("event_type")

        if event_type in seen:
            continue

        seen.add(event_type)

        record = encode_event(artifacts, event, config.max_pieces_per_value)

        values = event.model_values()

        out.append(
            {
                "client_id": history.client_id,
                "cutoff": cutoff.isoformat(),
                "event_id": event.event_id,
                "event_type": event_type,
                "event_time": event.event_time.isoformat(),
                "n_tokens": record.n_tokens,
                "pairs": [
                    {**item, "source_value": _short(values.get(item["key"]))}
                    for item in decode_record(artifacts, record)
                ],
                "refs": references(artifacts, event),
                "absent_reasons": absent_reasons(event, reason_of_event.get(event.stable_event_index)),
                "provenance": provenance(event, identity),
            }
        )

        if len(out) >= limit:
            break

    return out


def _short(value: object) -> object:

    if isinstance(value, str) and len(value) > 60:
        return value[:57] + "..."

    return value


def open_artifacts(target: Path) -> FrozenArtifacts:
    return FrozenArtifacts.load(target)


def group_corpus(processed_dir: Path, raw_dir: Path, group: str) -> GroupCorpus:

    try:
        return GroupCorpus.open(processed_dir, raw_dir, group)
    except CorpusError as error:
        raise TransformError(str(error)) from error


def resolve_cutoffs(values: list[str] | None, default: datetime) -> list[datetime]:

    if not values:
        return [default]

    return [datetime.fromisoformat(item) for item in values]


__all__ = [
    "CLIENTS_FILE",
    "EVENTS_FILE",
    "GOLDEN_JSON_FILE",
    "GOLDEN_MD_FILE",
    "MANIFEST_FILE",
    "PROFILES_FILE",
    "REPORT_JSON_FILE",
    "REPORT_MD_FILE",
    "STAGE",
    "TransformError",
    "TransformResult",
    "group_corpus",
    "open_artifacts",
    "resolve_cutoffs",
    "transform_group",
]
