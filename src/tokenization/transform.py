from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from src.preprocessing.artifacts import TableWriter, dumps_json
from src.preprocessing.keys import KeysError
from src.preprocessing.read import Group, ReadError
from src.preprocessing.settings import PreprocessingConfig

from .encode import EncodeError, encode_event, encode_profile, profile_known, references
from .layout import FrozenArtifacts
from .settings import TokenizerConfig, tokenized_dir
from .specials import EMPTY, INVALID, MISSING, UNK


# ============================================================
# ЭТАП 6: КОДИРОВАНИЕ ГРУППЫ
# ============================================================
#
# Словарь уже построен на train, и здесь он только применяется:
# ни одного нового токена, ни одной новой границы, ни одного
# пересчитанного порога. Значение val или test, которого на
# train не было, кодируется специальным токеном — это и есть
# честный ответ «такого мы не видели», а не повод дообучить
# словарь.
#
# Результат группы это ровно два файла:
#
#   data/tokenized/<group>/events.parquet
#   data/tokenized/<group>/profile.parquet
#
# client_id и event_time сохраняются: по ним датасет группирует
# и сортирует. Границы значений внутри записи сохраняются тоже —
# без них последовательность ID это просто числа.
#
# Клиент без событий остаётся в профиле, а событий у него может
# быть ноль: выдумывать покупки и сессии токенизатор не вправе.
# ============================================================


EVENTS_FILE = "events.parquet"
PROFILE_FILE = "profile.parquet"

# Момент, до которого взята история группы, и имя группы едут
# метаданными файла: колонкой одно и то же значение повторять по
# всем строкам незачем.
CUTOFF_KEY = b"cutoff"
GROUP_KEY = b"group"

EVENTS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("event_time", pa.timestamp("us")),
        ("stable_event_index", pa.int64()),
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
        ("unknown_keys", pa.list_(pa.string())),
        ("calendar", pa.list_(pa.float64())),
    ]
)

PROFILE_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("has_profile", pa.bool_()),
        ("n_events", pa.int64()),
        ("n_values", pa.int32()),
        ("n_tokens", pa.int32()),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("value_starts", pa.list_(pa.int32())),
        ("value_lengths", pa.list_(pa.int32())),
        ("value_keys", pa.list_(pa.string())),
        ("limitations", pa.list_(pa.string())),
    ]
)


class TransformError(ValueError):
    """
    Кодирование выполнить нельзя.
    """


@dataclass
class Counters:
    clients: int = 0
    events: int = 0
    values: int = 0
    tokens: int = 0
    profiles: int = 0
    without_profile: int = 0
    silent_clients: int = 0
    missing: int = 0
    unknown: int = 0
    invalid: int = 0
    empty: int = 0
    unknown_keys: dict[str, int] = field(default_factory=dict)
    max_event_tokens: int = 0


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


def group_cutoff(group: str) -> datetime:
    """
    Конечный cutoff группы: дальше её истории не существует.
    """

    window = PreprocessingConfig.load(None).windows.get(group)

    if window is None:
        raise TransformError(f"для группы {group} не объявлено окно наблюдения")

    return window.final_cutoff


def encode_group(
    artifacts: FrozenArtifacts,
    group: str,
    config: TokenizerConfig,
    directory: Path | None = None,
) -> dict:
    """
    Кодирует группу замороженным словарём в два файла.
    """

    directory = Path(directory) if directory is not None else tokenized_dir(group)

    cutoff = group_cutoff(group)

    try:
        source = Group(group)
    except ReadError as error:
        raise TransformError(str(error)) from error

    metadata = {
        CUTOFF_KEY: cutoff.isoformat().encode("utf-8"),
        GROUP_KEY: group.encode("utf-8"),
    }

    _clear(directory)

    counters = Counters()

    events_writer = TableWriter(directory / EVENTS_FILE, EVENTS_SCHEMA.with_metadata(metadata))
    profile_writer = TableWriter(directory / PROFILE_FILE, PROFILE_SCHEMA.with_metadata(metadata))

    limitations: set[str] = set()

    try:
        for client_id in source.client_ids:

            try:
                history = source.history(client_id, cutoff)
            except (ReadError, KeysError) as error:
                raise TransformError(f"клиент {client_id}: {error}") from error

            counters.clients += 1

            rows: list[dict] = []

            for event in history.events:

                try:
                    record = encode_event(artifacts, event, config.max_pieces_per_value)
                except EncodeError as error:
                    raise TransformError(
                        f"клиент {client_id}, событие {event.stable_event_index}: {error}"
                    ) from error

                _count_specials(artifacts, record, counters)

                counters.events += 1
                counters.values += record.n_values
                counters.tokens += record.n_tokens
                counters.max_event_tokens = max(counters.max_event_tokens, record.n_tokens)

                for key in record.unknown_keys:
                    counters.unknown_keys[key] = counters.unknown_keys.get(key, 0) + 1

                rows.append(
                    {
                        "client_id": history.client_id,
                        "event_time": event.event_time,
                        "stable_event_index": event.stable_event_index,
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
                        "unknown_keys": record.unknown_keys,
                        "calendar": list(event.calendar),
                    }
                )

            if rows:
                events_writer.write(pa.Table.from_pylist(rows, schema=EVENTS_SCHEMA))
            else:
                counters.silent_clients += 1

            # --- профиль ---

            record = encode_profile(artifacts, history, config.max_pieces_per_value)

            _count_specials(artifacts, record, counters)

            known = profile_known(history)

            if known:
                counters.profiles += 1
            else:
                counters.without_profile += 1

            counters.values += record.n_values
            counters.tokens += record.n_tokens

            profile_writer.write(
                pa.Table.from_pylist(
                    [
                        {
                            "client_id": history.client_id,
                            "has_profile": known,
                            "n_events": history.n_events,
                            "n_values": record.n_values,
                            "n_tokens": record.n_tokens,
                            "key_ids": record.key_ids,
                            "value_ids": record.value_ids,
                            "positions": record.positions,
                            "value_starts": record.value_starts,
                            "value_lengths": record.value_lengths,
                            "value_keys": record.value_keys,
                            "limitations": list(history.limitations),
                        }
                    ],
                    schema=PROFILE_SCHEMA,
                )
            )

            limitations.update(history.limitations)

    finally:
        events_rows = events_writer.close()
        profile_rows = profile_writer.close()

    # Словарь обязан быть тем же и ПОСЛЕ работы: кодирование
    # ничего не дообучает, и это проверяется, а не обещается.
    artifacts.verify()

    return {
        "group": group,
        "cutoff": cutoff.isoformat(),
        "rows": {"events": events_rows, "profile": profile_rows},
        "counts": {
            "clients": counters.clients,
            "events": counters.events,
            "values": counters.values,
            "tokens": counters.tokens,
            "profiles": counters.profiles,
            "clients_without_profile": counters.without_profile,
            "silent_clients": counters.silent_clients,
            "max_event_tokens": counters.max_event_tokens,
        },
        "specials": {
            "missing": counters.missing,
            "unknown": counters.unknown,
            "invalid": counters.invalid,
            "empty": counters.empty,
        },
        "unknown_keys": dict(sorted(counters.unknown_keys.items())),
        "limitations": sorted(limitations),
    }


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только два файла: прежний результат
    стирается целиком, чтобы половина старого не осталась рядом
    с половиной нового.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "CUTOFF_KEY",
    "EVENTS_FILE",
    "EVENTS_SCHEMA",
    "GROUP_KEY",
    "PROFILE_FILE",
    "PROFILE_SCHEMA",
    "TransformError",
    "encode_group",
    "group_cutoff",
]
