from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.preprocessing.settings import GroupWindow
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import PAD

from .context import EventStub, Selection, select
from .settings import ContextPolicy
from .targets import eligible
from .tokenized import TokenizedClient


# ============================================================
# ИДЕЯ
# ============================================================
#
# Один пример это один клиент группы: его события во времени и
# его профиль.
#
# В примере лежит только то, что нужно модели и маскированию:
# сама последовательность, границы событий и значений, время и
# маска допустимых целей. Всё, что считается по этим массивам,
# в файле не хранится — число событий это длина event_starts, а
# принадлежность значения событию видна по границам.
#
# Адрес значения это его начало в ОБЩЕЙ последовательности
# клиента: Masker прячет значение по паре начало/длина, не
# заглядывая в словарь и не пересчитывая смещения событий.
#
# Маска целей это период целей своей группы: старая история
# validation и test остаётся видимым контекстом, но целью чужой
# группы не становится.
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

    client_id: str

    # --- события ---
    key_ids: np.ndarray
    value_ids: np.ndarray
    positions: np.ndarray
    event_starts: np.ndarray
    event_lengths: np.ndarray
    event_time: np.ndarray
    calendar: np.ndarray

    # --- границы значений в общей последовательности ---
    value_starts: np.ndarray
    value_lengths: np.ndarray

    # --- цели ---
    target_event_mask: np.ndarray

    # --- профиль ---
    profile_key_ids: np.ndarray
    profile_value_ids: np.ndarray
    profile_positions: np.ndarray
    profile_value_starts: np.ndarray
    profile_value_lengths: np.ndarray

    # --- не попадает в файл ---
    #
    # Отбор контекста нужен сборщику: усечение оценочной группы
    # запрещено, и об этом обязан узнать человек. В строке
    # примера этих чисел нет.
    truncated: bool = False
    excluded_events: int = 0
    excluded_eligible: int = 0

    # --------------------------------------------------------

    @property
    def n_events(self) -> int:
        return int(self.event_starts.size)

    @property
    def n_tokens(self) -> int:
        return int(self.key_ids.size)

    @property
    def n_values(self) -> int:
        return int(self.value_starts.size)

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

        for name in ("event_time", "target_event_mask"):
            if getattr(self, name).size != self.n_events:
                raise SampleError(f"{self.client_id}: канал {name} не по одному значению на событие")

        if self.value_starts.size != self.value_lengths.size:
            raise SampleError(f"{self.client_id}: границы значений разной длины")

        # События лежат подряд и покрывают все токены, а значения
        # внутри события идут следом за его маркером.
        covered = 0
        value = 0

        for start, length in zip(self.event_starts.tolist(), self.event_lengths.tolist()):

            if start != covered:
                raise SampleError(f"{self.client_id}: событие начинается в {start}, а покрыто {covered}")

            if length < 1:
                raise SampleError(f"{self.client_id}: событие без единого токена")

            # Маркер: одинаковый код в обоих слотах на нулевой
            # позиции внутри события.
            if self.key_ids[start] != self.value_ids[start] or self.positions[start] != 0:
                raise SampleError(f"{self.client_id}: событие начинается не с маркера")

            value = _check_spans(
                self.client_id, self.value_starts, self.value_lengths, value, start + 1, start + length
            )

            covered += length

        if covered != self.n_tokens:
            raise SampleError(f"{self.client_id}: границы покрывают {covered} токенов из {self.n_tokens}")

        if value != self.n_values:
            raise SampleError(
                f"{self.client_id}: {self.n_values - value} значений лежат вне своих событий"
            )

        _check_record(self.client_id, "профиль", self.profile_key_ids, self.profile_value_ids,
                      self.profile_positions)

        _check_spans(
            self.client_id, self.profile_value_starts, self.profile_value_lengths, 0, 1,
            self.profile_tokens
        )

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


def _check_spans(client_id: str, starts: np.ndarray, lengths: np.ndarray, first: int,
                 begin: int, end: int) -> int:
    """
    Значения записи идут подряд от begin до end без дыр и
    нахлёстов. Возвращает номер следующего непроверенного
    значения.

    Маркер занимает нулевой столбец записи и значением не
    является: span на него наложиться не может.
    """

    covered = begin

    index = first

    while index < starts.size and int(starts[index]) < end:

        start = int(starts[index])
        length = int(lengths[index])

        if start != covered:
            raise SampleError(
                f"{client_id}: значение начинается в {start}, а покрыто до {covered}"
            )

        if length < 1:
            raise SampleError(f"{client_id}: значение без единого токена")

        covered = start + length
        index += 1

    if covered != end:
        raise SampleError(f"{client_id}: значения покрывают до {covered} вместо {end}")

    return index


def build_sample(
    artifacts: FrozenArtifacts,
    client: TokenizedClient,
    window: GroupWindow,
    policy: ContextPolicy,
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
    moments: list[datetime] = []
    target_mask: list[bool] = []

    value_starts: list[int] = []
    value_lengths: list[int] = []

    for position in selection.kept:

        item = client.events[position]

        start = len(key_ids)

        event_starts.append(start)
        event_lengths.append(item.n_tokens)

        # Границы значения считаются от начала ОБЩЕЙ
        # последовательности: по ним же видно, какому событию
        # значение принадлежит.
        for local in range(item.n_values):
            value_starts.append(start + item.value_starts[local])
            value_lengths.append(item.value_lengths[local])

        key_ids.extend(item.key_ids)
        value_ids.extend(item.value_ids)
        positions.extend(item.positions)

        if len(item.calendar) != 6:
            raise SampleError(
                f"клиент {client.client_id}, событие {item.event_time.isoformat()}: "
                f"календарь из {len(item.calendar)} чисел вместо шести"
            )

        calendar.extend(item.calendar)
        moments.append(item.event_time)
        target_mask.append(flags[position])

    sample = Sample(
        client_id=client.client_id,
        key_ids=_ints(key_ids),
        value_ids=_ints(value_ids),
        positions=_ints(positions),
        event_starts=_ints(event_starts),
        event_lengths=_ints(event_lengths),
        event_time=np.asarray(moments, dtype="datetime64[us]"),
        calendar=np.asarray(calendar, dtype=np.float32),
        value_starts=_ints(value_starts),
        value_lengths=_ints(value_lengths),
        target_event_mask=np.asarray(target_mask, dtype=bool),
        profile_key_ids=_ints(client.profile_key_ids),
        profile_value_ids=_ints(client.profile_value_ids),
        profile_positions=_ints(client.profile_positions),
        profile_value_starts=_ints(client.profile_value_starts),
        profile_value_lengths=_ints(client.profile_value_lengths),
        truncated=selection.truncated,
        excluded_events=selection.n_excluded,
        excluded_eligible=selection.excluded_eligible,
    )

    sample.check(artifacts)

    return sample


def _ints(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


__all__ = [
    "Sample",
    "SampleError",
    "build_sample",
]
