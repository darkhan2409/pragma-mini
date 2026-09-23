from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from src.tokenization.finalvocab import FrozenArtifacts, VALUE_PREFIX
from src.tokenization.settings import tokenized_dir
from src.tokenization.transform import EVENTS_FILE, PROFILE_FILE


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


# Ключ, по которому читается тип события: политика отбора
# контекста смотрит на него, когда решает, что такое веха.
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

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def profile_tokens(self) -> int:
        return len(self.profile_key_ids)


class TokenizedGroup:
    """
    Закодированная группа по стандартному пути.
    """

    def __init__(self, group: str, artifacts: FrozenArtifacts, directory: Path | None = None):

        self.group = group
        self.artifacts = artifacts
        self.directory = Path(directory) if directory is not None else tokenized_dir(group)

        for name in (EVENTS_FILE, PROFILE_FILE):
            if not (self.directory / name).exists():
                raise TokenizedError(
                    f"нет {self.directory / name}: выполните "
                    f"python -m src.tokenization.run encode {group}"
                )

        self._events = pq.ParquetFile(self.directory / EVENTS_FILE)

        profile = pq.read_table(self.directory / PROFILE_FILE)

        self._profiles: dict[str, dict] = {row["client_id"]: row for row in profile.to_pylist()}

        self.client_ids: list[str] = sorted(self._profiles)

        self._event_type_key = artifacts.key_id(EVENT_TYPE_KEY)

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
                rows = []

            yield self._client(client_id, rows)

        if pending is not None:
            raise TokenizedError(
                f"события группы {self.group} лежат не в порядке профиля: на "
                f"{pending[0]} порядок разошёлся. Выполните "
                f"python -m src.tokenization.run encode {self.group}"
            )

    def _client_rows(self) -> Iterator[tuple[str, list[dict]]]:
        """
        Строки событий по клиентам: строки одного клиента лежат
        подряд, поэтому буфер нужен ровно на одного.
        """

        current: str | None = None
        buffer: list[dict] = []

        for number in range(self._events.num_row_groups):

            for row in self._events.read_row_group(number).to_pylist():

                if row["client_id"] != current:

                    if current is not None:
                        yield current, buffer

                    current = row["client_id"]
                    buffer = []

                buffer.append(row)

        if current is not None:
            yield current, buffer

    def _event_type(self, row: dict) -> str | None:
        """
        Тип события из его же пар ключ/значение.

        Имя токена в финальном словаре это `value:<ключ>=<значение>`,
        поэтому тип читается без второй колонки рядом с данными.

        Строка это одно событие, значит её нулевая позиция это
        маркер [EVT]. Разбор начинается со следующей: значение
        открывает positions == 0, а 1, 2, … это куски того же
        значения, и на них смотреть незачем.
        """

        if self._event_type_key is None:
            return None

        positions = row["positions"]

        for index in range(1, len(positions)):

            if positions[index] != 0:
                continue

            if row["key_ids"][index] != self._event_type_key:
                continue

            name = self.artifacts.describe(row["value_ids"][index])

            prefix = f"{VALUE_PREFIX}{EVENT_TYPE_KEY}="

            return name[len(prefix):] if name.startswith(prefix) else None

        return None

    def _client(self, client_id: str, rows: list[dict]) -> TokenizedClient:

        profile = self._profiles.get(client_id)

        if profile is None:
            raise TokenizedError(
                f"клиент {client_id} есть в событиях группы {self.group}, но не в профиле: "
                "закодированная группа собрана не за один проход"
            )

        events = [
            TokenizedEvent(
                event_time=row["event_time"],
                event_type=self._event_type(row),
                key_ids=list(row["key_ids"]),
                value_ids=list(row["value_ids"]),
                positions=list(row["positions"]),
                calendar=list(row["calendar"]),
            )
            for row in rows
        ]

        # Порядок примера временной, и sort устойчив: события с
        # одинаковым временем остаются в том причинном порядке,
        # в каком их уложил препроцессинг.
        events.sort(key=lambda item: item.event_time)

        return TokenizedClient(
            client_id=client_id,
            events=events,
            profile_key_ids=list(profile["key_ids"]),
            profile_value_ids=list(profile["value_ids"]),
            profile_positions=list(profile["positions"]),
        )


__all__ = [
    "EVENT_TYPE_KEY",
    "TokenizedClient",
    "TokenizedError",
    "TokenizedEvent",
    "TokenizedGroup",
]
