from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from operator import attrgetter
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from src.tokenization.finalvocab import FrozenArtifacts, VALUE_PREFIX, vocabulary_digest
from src.tokenization.settings import tokenized_dir
from src.preprocessing.artifacts import read_json
from src.tokenization.transform import (
    EVENTS_FILE,
    META_FILE,
    PROFILE_FILE,
    TOKENIZED_FORMAT,
)
from src.preprocessing.profile_state import LIFELONG_TYPES, PROFILE_SEMANTICS


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственный вход датасета: закодированная группа.
#
#   data/04_tokenized/<group>/events.parquet
#   data/04_tokenized/<group>/profile.parquet
#
# Читается она потоково и по клиентам: строки одного клиента
# лежат подряд, поэтому держать в памяти нужно ровно одного.
#
# События клиента отдаются в порядке event_time. Препроцессинг
# уложил их так же, но датасет не подразумевает чужой порядок, а
# задаёт свой явно. Сортировка устойчива, поэтому причинный
# порядок событий с одинаковым временем сохраняется.
#
# Тип события отдельной колонкой не хранится: он уже лежит парой
# ключ/значение внутри самого события, и восстанавливается через
# словарь. Второй записи одного и того же факта рядом нет.
#
# Границы значений тоже не хранятся: значение открывает
# positions == 0, а 1, 2, … продолжают его кусками BPE. Нулевая
# позиция записи это маркер, и в разбор она не входит.
#
# Ничего не кодируется и не пересчитывается: токены уже готовы.
# ============================================================


# Ключ, по которому читается тип события: по типу решается, может
# ли событие стать целью MLM (targets.can_be_target).
EVENT_TYPE_KEY = "event_type"


class TokenizedError(ValueError):
    """
    Закодированную группу прочитать нельзя.
    """


@dataclass(frozen=True)
class TokenizedEvent:
    """
    Одно закодированное событие клиента.
    """

    event_time: datetime
    event_type: str | None
    key_ids: list[int]
    value_ids: list[int]
    positions: list[int]
    calendar: list[float]

    # Тип вехи анкеты, чей источник записан этим событием, иначе
    # None: такое событие остаётся контекстом, но не целью.
    lifelong_source: str | None = None

    @property
    def n_tokens(self) -> int:
        return len(self.key_ids)


@dataclass
class TokenizedClient:
    """
    Один клиент группы: его события во времени и его профиль.
    """

    client_id: str
    events: list[TokenizedEvent]

    profile_key_ids: list[int] = field(default_factory=list)
    profile_value_ids: list[int] = field(default_factory=list)
    profile_positions: list[int] = field(default_factory=list)

    # Время каждого токена анкеты: у вехи её момент, у [USR] и
    # Attributes None.
    profile_time: list[datetime | None] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def profile_tokens(self) -> int:
        return len(self.profile_key_ids)


def event_type_of(artifacts: FrozenArtifacts, row: dict, event_type_key: int | None,
                  names: dict[int, str] | None = None) -> str | None:
    """
    Тип события из его же пар ключ/значение.

    Имя токена в финальном словаре это `value:<ключ>=<значение>`,
    поэтому тип читается без второй колонки рядом с данными.
    event_type_key — номер ключа типа (artifacts.key_id), None —
    словарь его не знает. names — готовый event_type_names(artifacts):
    тот же ответ без разбора имени на каждом событии.

    Строка это одно событие, значит её нулевая позиция это
    маркер [EVT]. Разбор начинается со следующей: значение
    открывает positions == 0, а 1, 2, … это куски того же
    значения, и на них смотреть незачем.
    """

    return _event_type(artifacts, row["key_ids"], row["value_ids"], row["positions"], event_type_key, names)


def event_type_names(artifacts: FrozenArtifacts) -> dict[int, str]:
    """
    Номер значения типа события -> сам тип: все `value:event_type=…`
    словаря.
    """

    prefix = f"{VALUE_PREFIX}{EVENT_TYPE_KEY}="

    return {token_id: name[len(prefix):] for token_id, name in artifacts.name_of.items() if name.startswith(prefix)}


def _event_type(artifacts: FrozenArtifacts, key_ids: list[int], value_ids: list[int], positions: list[int],
                event_type_key: int | None, names: dict[int, str] | None) -> str | None:

    if event_type_key is None:
        return None

    # Первое место после маркера, где ключ — тип, а позиция открывает
    # значение: то же, что обход всех позиций, но поиск ключа в C.
    end = len(positions)
    index = 0

    while True:

        try:
            index = key_ids.index(event_type_key, index + 1, end)
        except ValueError:
            return None

        if positions[index] == 0:
            break

    value_id = value_ids[index]

    if names is not None:
        found = names.get(value_id)
        if found is not None:
            return found

    # Номер не тип события: describe остановит неизвестный номер, а
    # имя другого вида типом не станет.
    name = artifacts.describe(value_id)

    prefix = f"{VALUE_PREFIX}{EVENT_TYPE_KEY}="

    return name[len(prefix):] if name.startswith(prefix) else None


class TokenizedGroup:
    """
    Закодированная группа по стандартному пути.
    """

    def __init__(self, group: str, artifacts: FrozenArtifacts, directory: Path | None = None):

        self.group = group
        self.artifacts = artifacts
        self.directory = Path(directory) if directory is not None else tokenized_dir(group)

        for name in (EVENTS_FILE, PROFILE_FILE, META_FILE):
            if not (self.directory / name).exists():
                raise TokenizedError(
                    f"нет {self.directory / name}: выполните "
                    f"python -m src.tokenization.run encode {group}"
                )

        # Версия формата, а не только наличие файлов. Каталог
        # прежней сборки несёт анкету другого смысла или другой
        # набор вех.
        self.meta = read_json(self.directory / META_FILE)

        found = (
            self.meta.get("format"),
            self.meta.get("profile_semantics"),
            self.meta.get("profile_lifelong_types"),
        )

        needed = (TOKENIZED_FORMAT, PROFILE_SEMANTICS, list(LIFELONG_TYPES))

        if found != needed:
            raise TokenizedError(
                f"{self.directory / META_FILE}: формат, смысл анкеты и вехи {found!r}, а нужны "
                f"{needed!r}. Каталог собран прежним кодом — выполните "
                f"python -m src.tokenization.run encode {group} заново"
            )

        # Номера значений принадлежат словарю, под который группа
        # закодирована: другой словарь прочёл бы их как другие токены.
        if self.meta.get("vocabulary") != vocabulary_digest():
            raise TokenizedError(
                f"{self.directory / META_FILE}: группа закодирована не под текущий словарь "
                f"data/03_vocab — выполните python -m src.tokenization.run encode {group} заново"
            )

        self._events = pq.ParquetFile(self.directory / EVENTS_FILE)

        profile = pq.read_table(self.directory / PROFILE_FILE)

        self._profiles: dict[str, dict] = {row["client_id"]: row for row in profile.to_pylist()}

        self.client_ids: list[str] = sorted(self._profiles)

        self._event_type_key = artifacts.key_id(EVENT_TYPE_KEY)
        self._event_types = event_type_names(artifacts)

    # --- чтение ---

    def clients(self) -> Iterator[TokenizedClient]:
        """
        Клиенты по одному, в порядке client_id.

        Состав задаёт профиль, а не файл событий: клиент без
        событий тоже приходит, и порядок не зависит от того, у
        кого события есть. Пустая история это факт о клиенте, а
        не повод его пропустить.
        """

        stream = self._client_rows()

        pending = next(stream, None)

        for client_id in self.client_ids:

            if pending is not None and pending[0] == client_id:
                rows = pending[1]
                pending = next(stream, None)
            else:
                rows = None

            yield self._client(client_id, rows)

        if pending is not None:
            raise TokenizedError(
                f"события группы {self.group} лежат не в порядке профиля: на "
                f"{pending[0]} порядок разошёлся. Выполните "
                f"python -m src.tokenization.run encode {self.group}"
            )

    def _client_rows(self) -> Iterator[tuple[str, dict[str, list]]]:
        """
        Строки событий по клиентам, колонками: строки одного клиента
        лежат подряд, поэтому буфер нужен ровно на одного. Колонки, а
        не словарь на строку: to_pylist строил бы его на каждое событие.
        """

        current: str | None = None
        buffer: dict[str, list] | None = None

        for number in range(self._events.num_row_groups):

            columns = self._events.read_row_group(number).to_pydict()

            ids = columns["client_id"]

            start = 0

            for index in range(1, len(ids) + 1):

                if index < len(ids) and ids[index] == ids[start]:
                    continue

                if ids[start] != current:

                    if current is not None:
                        yield current, buffer

                    current = ids[start]
                    buffer = None

                if buffer is None:
                    buffer = {name: [] for name in columns}

                for name, values in columns.items():
                    buffer[name].extend(values[start:index])

                start = index

        if current is not None:
            yield current, buffer

    def _client(self, client_id: str, rows: dict[str, list] | None) -> TokenizedClient:

        profile = self._profiles.get(client_id)

        if profile is None:
            raise TokenizedError(
                f"клиент {client_id} есть в событиях группы {self.group}, но не в профиле: "
                "закодированная группа собрана не за один проход"
            )

        # Списки берутся как прочитаны: to_pydict отдаёт свои, и больше
        # их не держит никто.
        events = [
            TokenizedEvent(
                event_time=event_time,
                event_type=_event_type(self.artifacts, key_ids, value_ids, positions, self._event_type_key,
                                       self._event_types),
                key_ids=key_ids,
                value_ids=value_ids,
                positions=positions,
                calendar=calendar,
                lifelong_source=source,
            )
            for event_time, key_ids, value_ids, positions, calendar, source in zip(
                rows["event_time"], rows["key_ids"], rows["value_ids"], rows["positions"], rows["calendar"],
                rows["lifelong_source"],
            )
        ] if rows else []

        # Порядок примера временной, и sort устойчив: события с
        # одинаковым временем остаются в том причинном порядке,
        # в каком их уложил препроцессинг.
        events.sort(key=attrgetter("event_time"))

        return TokenizedClient(
            client_id=client_id,
            events=events,
            profile_key_ids=list(profile["key_ids"]),
            profile_value_ids=list(profile["value_ids"]),
            profile_positions=list(profile["positions"]),
            profile_time=list(profile["time"]),
        )


__all__ = [
    "EVENT_TYPE_KEY",
    "TokenizedClient",
    "TokenizedError",
    "TokenizedEvent",
    "TokenizedGroup",
    "event_type_names",
    "event_type_of",
]
