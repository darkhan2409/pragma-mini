from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.preprocessing.settings import GroupWindow
from src.tokenization.layout import FrozenArtifacts
from src.tokenization.specials import PAD

from .context import EventStub, Selection, select
from .settings import ContextPolicy
from .targets import eligible, hours_to_cutoff
from .tokenized import TokenizedClient


# ============================================================
# ИДЕЯ
# ============================================================
#
# Один пример это один клиент группы на её конечный срез.
#
# Внутри примера три круга колонок, и граница между ними не
# декоративная:
#
#   model    входы модели: токены событий и профиля, каналы
#            времени;
#   masker   границы значений, признак допустимой цели и вес;
#   service  трассировка: клиент, срез, усечение, ограничения.
#            В embedding ничего из этого не входит.
#
# Адрес значения это пара «событие, столбец внутри него»: Masker
# прячет значение по этому адресу, не заглядывая в словарь.
#
# Клиент без событий остаётся примером. Никакой выдуманной
# покупки: пустая история это факт о клиенте, а не пустое место.
# ============================================================


class SampleError(ValueError):
    """
    Пример собрать нельзя.
    """


@dataclass
class Sample:
    """
    Готовый пример: один клиент целиком.
    """

    # --- тождество ---
    client_id: str
    group: str
    cutoff: datetime
    weight: float
    sample_seed: int

    # --- события: входы модели ---
    key_ids: np.ndarray
    value_ids: np.ndarray
    positions: np.ndarray
    event_starts: np.ndarray
    event_lengths: np.ndarray

    # --- события: каналы времени ---
    calendar: np.ndarray
    hours_to_cutoff: np.ndarray
    event_time: np.ndarray

    # --- цели ---
    event_eligible: np.ndarray

    # --- значения событий ---
    value_event: np.ndarray
    value_start: np.ndarray
    value_length: np.ndarray
    value_key_id: np.ndarray

    # --- профиль ---
    profile_key_ids: np.ndarray
    profile_value_ids: np.ndarray
    profile_positions: np.ndarray
    profile_value_start: np.ndarray
    profile_value_length: np.ndarray
    profile_value_key_id: np.ndarray
    has_profile: bool

    # --- отбор истории ---
    n_eligible_events: int
    has_targets: bool
    truncated: bool
    excluded_events: int
    excluded_tokens: int
    excluded_eligible: int

    limitations: list[str] = field(default_factory=list)

    # --------------------------------------------------------

    @property
    def n_events(self) -> int:
        return int(self.event_starts.size)

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.size)

    @property
    def n_values(self) -> int:
        return int(self.value_event.size)

    @property
    def profile_tokens(self) -> int:
        return int(self.profile_key_ids.size)

    def check(self, artifacts: FrozenArtifacts) -> None:
        """
        Инварианты примера.

        Проверяется то, на что будет опираться Masker и модель:
        согласованность длин, покрытие токенов границами, один
        маркер на запись и принадлежность каждого ID словарю.
        """

        if not (self.key_ids.size == self.value_ids.size == self.positions.size):
            raise SampleError(f"{self.client_id}: массивы токенов разной длины")

        if self.event_starts.size != self.event_lengths.size:
            raise SampleError(f"{self.client_id}: границы событий разной длины")

        if self.calendar.size != self.n_events * 6:
            raise SampleError(f"{self.client_id}: календарь не по шесть чисел на событие")

        for name in ("hours_to_cutoff", "event_eligible", "event_time"):
            if getattr(self, name).size != self.n_events:
                raise SampleError(f"{self.client_id}: канал {name} не по одному значению на событие")

        # События лежат подряд и покрывают все токены.
        covered = 0

        for start, length in zip(self.event_starts.tolist(), self.event_lengths.tolist()):

            if start != covered:
                raise SampleError(f"{self.client_id}: событие начинается в {start}, а покрыто {covered}")

            if length < 1:
                raise SampleError(f"{self.client_id}: событие без единого токена")

            # Маркер: одинаковый код в обоих слотах на нулевой
            # позиции внутри события.
            if self.key_ids[start] != self.value_ids[start] or self.positions[start] != 0:
                raise SampleError(f"{self.client_id}: событие начинается не с маркера")

            covered += length

        if covered != self.n_tokens:
            raise SampleError(f"{self.client_id}: границы покрывают {covered} токенов из {self.n_tokens}")

        _check_record(self.client_id, "профиль", self.profile_key_ids, self.profile_value_ids,
                      self.profile_positions)

        # Границы значений: внутри своей записи, без нахлёста, и
        # маркер под них не попадает.
        _check_spans(self.client_id, self.value_event, self.value_start, self.value_length,
                     self.event_lengths.tolist())

        _check_spans(self.client_id, np.zeros(self.profile_value_start.size, dtype=np.int64),
                     self.profile_value_start, self.profile_value_length, [self.profile_tokens])

        # ID в пространстве словаря, [PAD] нигде не написан.
        pad = artifacts.special(PAD)
        size = artifacts.size

        for name in ("key_ids", "value_ids", "profile_key_ids", "profile_value_ids"):

            values = getattr(self, name)

            if values.size and (values.min() < 0 or values.max() >= size):
                raise SampleError(f"{self.client_id}: {name} выходит за пространство ID {size}")

            if values.size and bool((values == pad).any()):
                raise SampleError(
                    f"{self.client_id}: в {name} встретился [PAD]. Он существует только для "
                    "выравнивания batch и в сохранённом примере невозможен"
                )


def _check_record(client_id: str, what: str, key_ids: np.ndarray, value_ids: np.ndarray,
                  positions: np.ndarray) -> None:

    if not (key_ids.size == value_ids.size == positions.size):
        raise SampleError(f"{client_id}: {what} — массивы разной длины")

    if key_ids.size == 0:
        raise SampleError(f"{client_id}: {what} без единого токена: маркер обязан быть всегда")

    if key_ids[0] != value_ids[0] or positions[0] != 0:
        raise SampleError(f"{client_id}: {what} начинается не с маркера")


def _check_spans(client_id: str, owner: np.ndarray, start: np.ndarray, length: np.ndarray,
                 lengths: list[int]) -> None:

    if not (owner.size == start.size == length.size):
        raise SampleError(f"{client_id}: границы значений разной длины")

    covered: dict[int, int] = {}

    for index in range(owner.size):

        which = int(owner[index])
        first = int(start[index])
        size = int(length[index])

        # Маркер занимает нулевой столбец записи и значением не
        # является: span на него наложиться не может.
        expected = covered.get(which, 1)

        if first != expected:
            raise SampleError(
                f"{client_id}: значение записи {which} начинается в столбце {first}, "
                f"а покрыто до {expected}"
            )

        covered[which] = first + size

    for which, size in covered.items():
        if size != lengths[which]:
            raise SampleError(
                f"{client_id}: значения записи {which} покрывают {size} токенов из {lengths[which]}"
            )


def sample_seed_of(group: str, client_id: str, cutoff: datetime) -> int:
    """
    Устойчивый номер примера для будущего Masker.

    Считается от тождества примера, а не от его места в файле:
    иначе маска зависела бы от порядка чтения и от размера
    batch.
    """

    text = f"{group}\x1f{client_id}\x1f{cutoff.isoformat()}"

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()

    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def build_sample(
    artifacts: FrozenArtifacts,
    client: TokenizedClient,
    group: str,
    window: GroupWindow,
    cutoff: datetime,
    policy: ContextPolicy,
    weight: float = 1.0,
) -> Sample:
    """
    Пример из закодированной истории клиента.
    """

    flags = [eligible(item.event_time, window) for item in client.events]

    stubs = [
        EventStub(index=number, event_type=item.event_type, n_tokens=item.n_tokens,
                  eligible=flags[number])
        for number, item in enumerate(client.events)
    ]

    selection: Selection = select(stubs, policy)

    if client.profile_tokens > policy.max_profile_tokens:
        raise SampleError(
            f"клиент {client.client_id}: представление профиля занимает "
            f"{client.profile_tokens} токенов при пределе {policy.max_profile_tokens}"
        )

    key_ids: list[int] = []
    value_ids: list[int] = []
    positions: list[int] = []
    event_starts: list[int] = []
    event_lengths: list[int] = []
    calendar: list[float] = []
    to_cutoff: list[float] = []
    moments: list[datetime] = []
    event_eligible: list[bool] = []

    value_event: list[int] = []
    value_start: list[int] = []
    value_length: list[int] = []
    value_key_id: list[int] = []

    for slot, position in enumerate(selection.kept):

        item = client.events[position]

        event_starts.append(len(key_ids))
        event_lengths.append(item.n_tokens)

        for local in range(item.n_values):
            value_event.append(slot)
            value_start.append(item.value_starts[local])
            value_length.append(item.value_lengths[local])
            value_key_id.append(item.key_ids[item.value_starts[local]])

        key_ids.extend(item.key_ids)
        value_ids.extend(item.value_ids)
        positions.extend(item.positions)

        if len(item.calendar) != 6:
            raise SampleError(
                f"клиент {client.client_id}, событие {item.stable_event_index}: "
                f"календарь из {len(item.calendar)} чисел вместо шести"
            )

        calendar.extend(item.calendar)
        to_cutoff.append(hours_to_cutoff(cutoff, item.event_time))
        moments.append(item.event_time)
        event_eligible.append(flags[position])

    sample = Sample(
        client_id=client.client_id,
        group=group,
        cutoff=cutoff,
        weight=weight,
        sample_seed=sample_seed_of(group, client.client_id, cutoff),
        key_ids=_ints(key_ids),
        value_ids=_ints(value_ids),
        positions=_ints(positions),
        event_starts=_ints(event_starts),
        event_lengths=_ints(event_lengths),
        calendar=np.asarray(calendar, dtype=np.float32),
        hours_to_cutoff=np.asarray(to_cutoff, dtype=np.float64),
        event_time=np.asarray(moments, dtype="datetime64[us]"),
        event_eligible=np.asarray(event_eligible, dtype=bool),
        value_event=_ints(value_event),
        value_start=_ints(value_start),
        value_length=_ints(value_length),
        value_key_id=_ints(value_key_id),
        profile_key_ids=_ints(client.profile_key_ids),
        profile_value_ids=_ints(client.profile_value_ids),
        profile_positions=_ints(client.profile_positions),
        profile_value_start=_ints(client.profile_value_starts),
        profile_value_length=_ints(client.profile_value_lengths),
        profile_value_key_id=_ints(
            [client.profile_key_ids[start] for start in client.profile_value_starts]
        ),
        has_profile=client.has_profile,
        n_eligible_events=sum(1 for slot in selection.kept if flags[slot]),
        has_targets=any(flags[slot] for slot in selection.kept),
        truncated=selection.truncated,
        excluded_events=selection.n_excluded,
        excluded_tokens=selection.excluded_tokens,
        excluded_eligible=selection.excluded_eligible,
        limitations=list(client.limitations),
    )

    sample.check(artifacts)

    return sample


def _ints(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


__all__ = [
    "Sample",
    "SampleError",
    "build_sample",
    "sample_seed_of",
]
