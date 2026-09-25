from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.preprocessing.keys import PROFILE_LIFELONG_KEY
from src.preprocessing.settings import GroupWindow
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import PAD

from .context import EventStub, Selection, select
from .settings import ContextPolicy
from .targets import can_be_target, eligible
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
# Границы значения задаёт positions: ноль открывает значение, а
# 1, 2, … продолжают его кусками BPE. Masker разбирает их внутри
# окна события, не заглядывая в словарь. Нулевая позиция окна
# это маркер события, и значением она не становится.
#
# Маска целей это период целей своей группы: старая история
# validation и test остаётся видимым контекстом, но целью чужой
# группы не становится.
#
# Анкета это Attributes на cutoff и вехи Lifelong строго раньше
# него. Время есть только у вех — profile_time, у остальных
# токенов анкеты NaT.
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

    # --- цели ---
    target_event_mask: np.ndarray

    # --- профиль ---
    profile_key_ids: np.ndarray
    profile_value_ids: np.ndarray
    profile_positions: np.ndarray
    profile_time: np.ndarray

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
        # Значений столько, сколько нулей вне маркеров событий.
        return int((self.positions == 0).sum()) - self.n_events

    @property
    def profile_n_values(self) -> int:
        # Тот же счёт для анкеты: её маркер ровно один.
        return int((self.profile_positions == 0).sum()) - 1

    @property
    def profile_tokens(self) -> int:
        return int(self.profile_key_ids.size)

    def check(self, artifacts: FrozenArtifacts, cutoff: datetime) -> None:
        """
        Инварианты примера.

        Проверяется то, на что будет опираться Masker и модель:
        согласованность длин, покрытие токенов границами, один
        маркер на запись, принадлежность каждого ID словарю и
        время анкеты — только у вех и строго раньше cutoff.
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

        # События лежат подряд и покрывают все токены, а значения
        # внутри события разбираются по positions следом за его
        # маркером.
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

            _check_positions(self.client_id, "событие", self.positions, start + 1, start + length)

            covered += length

        if covered != self.n_tokens:
            raise SampleError(f"{self.client_id}: границы покрывают {covered} токенов из {self.n_tokens}")

        _check_record(self.client_id, "профиль", self.profile_key_ids, self.profile_value_ids,
                      self.profile_positions)

        _check_positions(self.client_id, "профиль", self.profile_positions, 1, self.profile_tokens)

        _check_profile_time(self, artifacts, cutoff)

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


def _check_profile_time(sample: Sample, artifacts: FrozenArtifacts, cutoff: datetime) -> None:
    """
    Время анкеты: есть ровно у токенов вех, строго раньше cutoff
    и не убывает — вехи идут по времени.
    """

    if sample.profile_time.size != sample.profile_tokens:
        raise SampleError(f"{sample.client_id}: время анкеты не по одному на токен")

    dated = ~np.isnat(sample.profile_time)

    lifelong = sample.profile_key_ids == artifacts.key_id(PROFILE_LIFELONG_KEY.key)

    if not np.array_equal(dated, lifelong):
        raise SampleError(
            f"{sample.client_id}: время анкеты есть не ровно у вех — у Attributes и [USR] "
            "его быть не может, а у вехи оно обязательно"
        )

    moments = sample.profile_time[dated]

    if moments.size == 0:
        return

    anchor = np.datetime64(cutoff.replace(tzinfo=None), "us")

    if bool((moments >= anchor).any()):
        raise SampleError(f"{sample.client_id}: веха анкеты не раньше cutoff {cutoff.isoformat()}")

    if bool((moments[1:] < moments[:-1]).any()):
        raise SampleError(f"{sample.client_id}: вехи анкеты идут не по времени")


def _check_record(client_id: str, what: str, key_ids: np.ndarray, value_ids: np.ndarray,
                  positions: np.ndarray) -> None:

    if not (key_ids.size == value_ids.size == positions.size):
        raise SampleError(f"{client_id}: {what} — массивы разной длины")

    if key_ids.size == 0:
        raise SampleError(f"{client_id}: {what} без единого токена: маркер обязан быть всегда")

    if key_ids[0] != value_ids[0] or positions[0] != 0:
        raise SampleError(f"{client_id}: {what} начинается не с маркера")


def _check_positions(client_id: str, what: str, positions: np.ndarray, begin: int,
                     end: int) -> None:
    """
    Значения записи восстанавливаются по positions: ноль
    начинает новое значение, дальше его куски идут подряд.

    Маркер записи лежит в begin - 1 и значением не является: его
    ноль в разбор не входит. Два соседних значения с одним
    ключом остаются двумя: каждое начинает свой ноль.
    """

    # Содержимое обязано начаться с нуля: токен с positions == 3
    # на первом месте не пройдёт ни одну из двух веток.
    expected = 0

    for index in range(begin, end):

        position = int(positions[index])

        if position != 0 and position != expected:
            raise SampleError(
                f"{client_id}: {what} — в позиции {index} стоит {position}, а куски "
                f"значения идут подряд от нуля (ожидалось {expected})"
            )

        expected = position + 1


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

    for position in selection.kept:

        item = client.events[position]

        start = len(key_ids)

        event_starts.append(start)
        event_lengths.append(item.n_tokens)

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
        # Период целей решает, где цели разрешены; тип — может ли
        # событие ей быть. Изменение анкеты остаётся контекстом.
        target_mask.append(flags[position] and can_be_target(item.event_type))

    sample = Sample(
        client_id=client.client_id,
        key_ids=_ints(key_ids),
        value_ids=_ints(value_ids),
        positions=_ints(positions),
        event_starts=_ints(event_starts),
        event_lengths=_ints(event_lengths),
        event_time=_utc_moments(moments),
        calendar=np.asarray(calendar, dtype=np.float32),
        target_event_mask=np.asarray(target_mask, dtype=bool),
        profile_key_ids=_ints(client.profile_key_ids),
        profile_value_ids=_ints(client.profile_value_ids),
        profile_positions=_ints(client.profile_positions),
        profile_time=_utc_moments(client.profile_time),
        truncated=selection.truncated,
        excluded_events=selection.n_excluded,
        excluded_eligible=selection.excluded_eligible,
    )

    sample.check(artifacts, window.final_cutoff)

    return sample


def _ints(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


def _utc_moments(moments) -> np.ndarray:
    """
    Моменты UTC как datetime64[us]; None становится NaT.

    Время уже в UTC, и numpy хранит его без пояса: пояс снимается
    явно, чтобы никто не пересчитал его вторично.
    """

    return np.asarray(
        [None if moment is None else moment.replace(tzinfo=None) for moment in moments],
        dtype="datetime64[us]",
    )


__all__ = [
    "Sample",
    "SampleError",
    "build_sample",
]
