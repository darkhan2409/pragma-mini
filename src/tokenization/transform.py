from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa

from src.preprocessing.artifacts import TableWriter, write_json
from src.preprocessing.keys import PROFILE_KEYS, PROFILE_LIFELONG_KEY, KeysError
from src.preprocessing.profile_state import (
    EXCLUDED_FIELDS,
    INCLUDED_FIELDS,
    LIFELONG_TYPES,
    PROFILE_SEMANTICS,
)
from src.temporal.position import TIME_TRANSFORM
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
#   data/04_tokenized/<group>/events.parquet
#   data/04_tokenized/<group>/profile.parquet
#
# Профиль здесь уже закодирован: это представление анкеты
# токенами, а не копия выгрузки. Он описывает клиента на cutoff
# событий группы: Attributes на cutoff и вехи Lifelong строго
# раньше него (PROFILE_SEMANTICS). У каждого токена анкеты есть
# время: у вех — момент вехи, у [USR] и Attributes — null.
#
# Отсюда третий файл — meta.json. В нём записаны cutoff, смысл
# анкеты, набор вех и версия формата: каталог, собранный прежним
# кодом, несёт анкету другого смысла, и смешивать его с новым
# нельзя. Читатель (05_dataset) это проверяет и на старом
# каталоге останавливается.
#
# client_id и event_time сохраняются: по ним датасет группирует
# и сортирует. Границы значений отдельными массивами не лежат:
# их задаёт positions, где ноль начинает значение, а 1, 2, …
# продолжают его кусками BPE.
#
# Клиент без событий остаётся в профиле, а событий у него может
# быть ноль: выдумывать покупки и сессии токенизатор не вправе.
# ============================================================


EVENTS_FILE = "events.parquet"
PROFILE_FILE = "profile.parquet"
META_FILE = "meta.json"

# 1 — анкета на конец выгрузки (прежний формат, читателю не
#     годится);
# 2 — анкета на начало периода целей;
# 3 — анкета на cutoff событий (PROFILE_SEMANTICS), возраст и
#     признак пенсионера посчитаны от даты рождения;
# 4 — анкета это Attributes на cutoff и вехи Lifelong раньше
#     него; у токенов анкеты есть время (колонка time);
# 5 — признака пенсионера среди Attributes нет; возраст — только
#     от birth_date на cutoff и точным числом полных лет: каждый
#     возраст своё значение словаря, а не диапазон;
# 6 — в Attributes есть стаж на месте работы полугодиями; вехи
#     Lifelong —
#     bank_registered, app_registered, first_card_activated,
#     first_loan_opened, first_deposit_opened;
# 7 — у события есть колонка lifelong_source: тип вехи, чей
#     источник записан этим событием (ссылка source_id вехи,
#     найденная препроцессингом), иначе null; стаж есть только при
#     наёмном виде дохода на cutoff.
TOKENIZED_FORMAT = 7

# В файлах лежит только то, что нужно модели. Число токенов
# и значений не хранится: это длины массивов. Названия
# ключей рядом с их номерами тоже: их восстанавливает словарь.
# Тип события и его источник уже лежат внутри пар ключ/значение.
#
# lifelong_source токеном не является: это пометка события-
# источника вехи анкеты, по которой датасет не делает его целью.
# Она едет вместе со строкой события, поэтому переживает и отбор
# контекста, и раскладку по batch.
EVENTS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("event_time", pa.timestamp("us", tz="UTC")),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("calendar", pa.list_(pa.float32())),
        ("lifelong_source", pa.string()),
    ]
)

# time — момент каждого токена анкеты: у вехи её время, у [USR]
# и Attributes null.
PROFILE_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("time", pa.list_(pa.timestamp("us", tz="UTC"))),
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


def group_window(group: str):
    """
    Окно группы: конец истории и начало периода целей.
    """

    window = PreprocessingConfig.load(None).windows.get(group)

    if window is None:
        raise TransformError(f"для группы {group} не объявлено окно наблюдения")

    return window


def group_cutoff(group: str) -> datetime:
    """
    Конечный cutoff группы: дальше её истории не существует.
    """

    return group_window(group).final_cutoff


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

    check_vocab(artifacts)

    window = group_window(group)

    # Один момент на всё: события строго раньше cutoff, анкета —
    # Attributes на тот же cutoff и вехи строго раньше него.
    cutoff = window.final_cutoff

    try:
        source = Group(group)
    except ReadError as error:
        raise TransformError(str(error)) from error

    _clear(directory)

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
                    record = encode_event(artifacts, event, config.max_pieces_per_value)
                except EncodeError as error:
                    raise TransformError(
                        f"клиент {client_id}, событие {event.event_time.isoformat()}: {error}"
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
                        "calendar": list(event.calendar),
                        "lifelong_source": event.lifelong_source,
                    }
                )

            if rows:
                events_writer.write(pa.Table.from_pylist(rows, schema=EVENTS_SCHEMA))
            else:
                counters.silent_clients += 1

            # --- профиль ---

            try:
                record, times = encode_profile(artifacts, history, config.max_pieces_per_value)
            except EncodeError as error:
                raise TransformError(f"клиент {client_id}, анкета: {error}") from error

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
                            "time": times,
                        }
                    ],
                    schema=PROFILE_SCHEMA,
                )
            )

    finally:
        events_rows = events_writer.close()
        profile_rows = profile_writer.close()

    meta = {
        "format": TOKENIZED_FORMAT,
        "group": group,
        "events_cutoff": cutoff.isoformat(),
        "profile_semantics": PROFILE_SEMANTICS,
        "profile_fields": list(INCLUDED_FIELDS),
        "profile_fields_excluded": dict(EXCLUDED_FIELDS),
        "profile_lifelong_types": list(LIFELONG_TYPES),
        "profile_lifelong_time": {"anchor": "cutoff", "transform": TIME_TRANSFORM},
        "clients": counters.clients,
        "clients_with_profile": counters.profiles,
        "clients_with_empty_profile": counters.empty_profiles,
    }

    write_json(directory / META_FILE, meta)

    return {
        "group": group,
        "cutoff": cutoff.isoformat(),
        "meta": meta,
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


def check_vocab(artifacts: FrozenArtifacts) -> None:
    """
    Словарь собран тем же кодом, что кодирует.

    Своей версии у словаря нет, поэтому сверяется содержание:
    словарь прежнего кода кодировал бы анкету молча и неверно —
    поля без ключа ушли бы в неизвестные ключи, возраст стал бы
    диапазоном, а вехи прежнего набора — известными значениями.
    """

    schema = SemanticSchema()

    command = "словарь собран прежним кодом — выполните python -m src.tokenization.run fit заново"

    features = {key for key, info in schema.keys.items() if info.is_model_feature}

    gone = sorted(set(artifacts.keys) - features)

    if gone:
        raise TransformError(f"в словаре ключи, которых больше нет: {gone}; {command}")

    needed = [PROFILE_KEYS[name].key for name in INCLUDED_FIELDS] + [PROFILE_LIFELONG_KEY.key]

    missing = sorted(key for key in needed if artifacts.key_id(key) is None)

    if missing:
        raise TransformError(f"в словаре нет ключей анкеты {missing}; {command}")

    scaled = sorted(key for key in schema.categorical_keys if key in artifacts.buckets)

    if scaled:
        raise TransformError(f"у категориальных ключей {scaled} в словаре числовая шкала; {command}")

    stale = sorted(set(artifacts.values.get(PROFILE_LIFELONG_KEY.key, {})) - set(LIFELONG_TYPES))

    if stale:
        raise TransformError(f"в словаре вехи прежнего набора {stale}; {command}")


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
    "check_vocab",
    "encode_group",
    "group_cutoff",
]
