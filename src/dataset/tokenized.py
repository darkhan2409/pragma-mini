from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from src.tokenization.transform import CUTOFF_KEY, EVENTS_FILE, PROFILE_FILE
from src.tokenization.settings import tokenized_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственный вход датасета: закодированная группа.
#
#   data/tokenized/<group>/events.parquet
#   data/tokenized/<group>/profile.parquet
#
# Читается она потоково и по клиентам: строки одного клиента
# лежат подряд, поэтому держать в памяти нужно ровно одного.
#
# События клиента отдаются в порядке event_time. Порядок в файле
# задал препроцессинг, и он деловой; датасету нужен временной, и
# сортировка делается здесь явно, а не подразумевается.
#
# Ничего не кодируется и не пересчитывается: токены уже готовы.
# ============================================================


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
    stable_event_index: int
    event_type: str | None
    key_ids: list[int]
    value_ids: list[int]
    positions: list[int]
    value_starts: list[int]
    value_lengths: list[int]
    calendar: list[float]

    @property
    def n_tokens(self) -> int:
        return len(self.key_ids)

    @property
    def n_values(self) -> int:
        return len(self.value_starts)


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
    profile_value_starts: list[int] = field(default_factory=list)
    profile_value_lengths: list[int] = field(default_factory=list)

    has_profile: bool = False
    limitations: list[str] = field(default_factory=list)

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

    def __init__(self, group: str, directory: Path | None = None):

        self.group = group
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

        self.cutoff = self._cutoff()

    def _cutoff(self) -> datetime:
        """
        Момент, до которого взята история группы.

        Записан кодированием в метаданные файла: второй раз
        вычислять его из конфигурации значило бы завести два
        ответа на один вопрос.
        """

        metadata = self._events.schema_arrow.metadata or {}

        stored = metadata.get(CUTOFF_KEY)

        if stored is None:
            raise TokenizedError(
                f"в {self.directory / EVENTS_FILE} нет границы среза: файл собран старым кодом, "
                f"выполните python -m src.tokenization.run encode {self.group}"
            )

        return datetime.fromisoformat(stored.decode("utf-8"))

    # --- чтение ---

    def clients(self) -> Iterator[TokenizedClient]:
        """
        Клиенты по одному, в порядке client_id.

        Состав задаёт профиль, а не файл событий: клиент без
        событий тоже приходит, и порядок не зависит от того,
        у кого события есть. Пустая история это факт о клиенте,
        а не повод его пропустить.
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
                stable_event_index=row["stable_event_index"],
                event_type=row["event_type"],
                key_ids=list(row["key_ids"]),
                value_ids=list(row["value_ids"]),
                positions=list(row["positions"]),
                value_starts=list(row["value_starts"]),
                value_lengths=list(row["value_lengths"]),
                calendar=list(row["calendar"]),
            )
            for row in rows
        ]

        # Порядок примера временной. Событие-исправление может
        # быть датировано позже своего исходного, и деловой
        # порядок файла этого не отражает.
        events.sort(key=lambda item: (item.event_time, item.stable_event_index))

        return TokenizedClient(
            client_id=client_id,
            events=events,
            profile_key_ids=list(profile["key_ids"]),
            profile_value_ids=list(profile["value_ids"]),
            profile_positions=list(profile["positions"]),
            profile_value_starts=list(profile["value_starts"]),
            profile_value_lengths=list(profile["value_lengths"]),
            has_profile=bool(profile["has_profile"]),
            limitations=list(profile["limitations"] or ()),
        )


__all__ = [
    "TokenizedClient",
    "TokenizedError",
    "TokenizedEvent",
    "TokenizedGroup",
]
