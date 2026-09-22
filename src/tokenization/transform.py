from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from src.preprocessing.artifacts import TableWriter
from src.preprocessing.keys import KeysError
from src.preprocessing.read import Group, ReadError
from src.preprocessing.settings import PreprocessingConfig

from .encode import EncodeError, encode_event, encode_profile
from .finalvocab import FrozenArtifacts
from .schema import SemanticSchema
from .settings import TokenizerConfig, tokenized_dir
from .specials import UNK


# ============================================================
# ЭТАП 7: КОДИРОВАНИЕ ГРУППЫ
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
# Профиль здесь уже закодирован: это представление анкеты
# токенами, а не копия выгрузки.
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

# В файлах лежит только то, что нужно модели. Число токенов
# и значений не хранится: это длины массивов. Названия
# ключей рядом с их номерами тоже: их восстанавливает словарь.
# Тип события и его источник уже лежат внутри пар ключ/значение.
EVENTS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("event_time", pa.timestamp("us")),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("value_starts", pa.list_(pa.int32())),
        ("value_lengths", pa.list_(pa.int32())),
        ("calendar", pa.list_(pa.float32())),
    ]
)

PROFILE_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("value_starts", pa.list_(pa.int32())),
        ("value_lengths", pa.list_(pa.int32())),
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
    empty_profiles: int = 0
    silent_clients: int = 0
    unknown: int = 0
    unknown_keys: dict[str, int] = field(default_factory=dict)
    max_event_tokens: int = 0


def _count_unknown(artifacts: FrozenArtifacts, record, counters: Counters) -> None:
    """
    Сколько значений словарь не знает.
    """

    unknown = artifacts.special(UNK)

    counters.unknown += sum(1 for value_id in record.value_ids if value_id == unknown)


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

    _clear(directory)

    # Ссылки на сущности кода не получают: их состав знает
    # смысловой реестр, а не словарь.
    links = frozenset(SemanticSchema.open().link_keys)

    counters = Counters()

    events_writer = TableWriter(directory / EVENTS_FILE, EVENTS_SCHEMA)
    profile_writer = TableWriter(directory / PROFILE_FILE, PROFILE_SCHEMA)

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
                    record = encode_event(artifacts, event, config.max_pieces_per_value, links)
                except EncodeError as error:
                    raise TransformError(
                        f"клиент {client_id}, событие {event.stable_event_index}: {error}"
                    ) from error

                _count_unknown(artifacts, record, counters)

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
                        "key_ids": record.key_ids,
                        "value_ids": record.value_ids,
                        "positions": record.positions,
                        "value_starts": record.value_starts,
                        "value_lengths": record.value_lengths,
                        "calendar": list(event.calendar),
                    }
                )

            if rows:
                events_writer.write(pa.Table.from_pylist(rows, schema=EVENTS_SCHEMA))
            else:
                counters.silent_clients += 1

            # --- профиль ---

            record = encode_profile(artifacts, history, config.max_pieces_per_value, links)

            _count_unknown(artifacts, record, counters)

            if record.n_values:
                counters.profiles += 1
            else:
                counters.empty_profiles += 1

            counters.values += record.n_values
            counters.tokens += record.n_tokens

            profile_writer.write(
                pa.Table.from_pylist(
                    [
                        {
                            "client_id": history.client_id,
                            "key_ids": record.key_ids,
                            "value_ids": record.value_ids,
                            "positions": record.positions,
                            "value_starts": record.value_starts,
                            "value_lengths": record.value_lengths,
                        }
                    ],
                    schema=PROFILE_SCHEMA,
                )
            )

    finally:
        events_rows = events_writer.close()
        profile_rows = profile_writer.close()

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
            "empty_profiles": counters.empty_profiles,
            "silent_clients": counters.silent_clients,
            "max_event_tokens": counters.max_event_tokens,
        },
        "unknown_values": counters.unknown,
        "unknown_keys": dict(sorted(counters.unknown_keys.items())),
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
    "EVENTS_FILE",
    "EVENTS_SCHEMA",
    "PROFILE_FILE",
    "PROFILE_SCHEMA",
    "TransformError",
    "encode_group",
    "group_cutoff",
]
