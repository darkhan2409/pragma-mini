from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dependencies import DEP_PROFILE
from .sample import Sample


# ============================================================
# ИДЕЯ
# ============================================================
#
# Примеры переменной длины в прямоугольный batch.
#
# События лежат ПЛОСКО по всему batch: [N, T], где N это сумма
# событий примеров, а T длина самого длинного события. Так
# энкодер события проходит по настоящим событиям и ни одного
# выдуманного: раскладка [B, E, T] заставила бы дополнять
# историю пустыми событиями, а пустое событие это не то же
# самое, что отсутствие события.
#
# Обратный адрес даёт пара: event_sample и event_slot говорят,
# чьё это событие и какое по счёту, а history_rows наоборот — по
# примеру и номеру в истории даёт строку в N. Оба направления
# дешёвые, и держать их оба честнее, чем заставлять потребителя
# строить второе самому.
#
# Правило масок одно на весь формат: True это НАСТОЯЩЕЕ. Прежний
# слой модели держал противоположное соглашение для padding, и
# молча унаследовать его было бы ошибкой на ровном месте.
#
# target_candidate_mask это ОБЛАСТЬ допустимых целей, а не
# выбранные цели. Датасет не маскирует ничего: выбор делает
# Masker, и его маска появится отдельно.
#
# Пустая история занимает свою строку batch и не даёт ни одного
# события. Ничего не выдумывается, до единицы ничего не
# дополняется.
# ============================================================


PAD_ID = 0


@dataclass
class Batch:
    """
    Прямоугольный batch: numpy, без torch.
    """

    # --- примеры ---
    sample_ids: list[str] = field(default_factory=list)
    client_ids: list[str] = field(default_factory=list)
    cutoffs: list = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    sample_seeds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    has_targets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    n_events: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    event_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))

    # --- события ---
    event_key_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    event_value_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    event_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    event_token_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    target_candidate_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    event_lengths: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    event_sample: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    event_slot: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    event_eligible: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    # --- история ---
    history_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    history_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))

    # --- каналы ---
    calendar: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), dtype=np.float32))
    hours_to_cutoff: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    coverage_at_cutoff: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    history_age_days: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    history_age_known: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    # --- значения ---
    value_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    value_event: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    value_start: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    value_length: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    value_key_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # --- профиль ---
    profile_key_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    profile_value_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    profile_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int64))
    profile_token_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    profile_known: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    profile_value_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    profile_value_sample: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    profile_value_start: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    profile_value_length: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    profile_value_key_id: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # --- происхождение ---
    dep_value: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dep_source_event: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dep_source_value: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    dep_status: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_rows(self) -> int:
        return int(self.event_key_ids.shape[0])

    @property
    def width(self) -> int:
        return int(self.event_key_ids.shape[1])

    @property
    def padded_tokens(self) -> int:
        return self.n_rows * self.width


class CollateError(ValueError):
    """
    Batch собрать нельзя.
    """


def collate(samples: list[Sample]) -> Batch:
    """
    Список примеров в один batch.
    """

    if not samples:
        raise CollateError("batch из нуля примеров не бывает")

    groups = {item.group for item in samples}

    if len(groups) > 1:
        # У групп разные окна целей и разные клиенты: смешивать
        # их в одном batch значит мерить двумя линейками сразу.
        raise CollateError(f"в одном batch группы {sorted(groups)}: они не смешиваются")

    count = len(samples)

    n_events = np.asarray([item.n_events for item in samples], dtype=np.int64)

    offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(n_events, out=offsets[1:])

    rows = int(offsets[-1])

    lengths = (
        np.concatenate([item.event_lengths for item in samples]).astype(np.int64)
        if rows
        else np.zeros(0, dtype=np.int64)
    )

    width = int(lengths.max()) if rows else 0

    key_ids = np.full((rows, width), PAD_ID, dtype=np.int64)
    value_ids = np.full((rows, width), PAD_ID, dtype=np.int64)
    positions = np.zeros((rows, width), dtype=np.int64)

    event_sample = np.zeros(rows, dtype=np.int64)
    event_slot = np.zeros(rows, dtype=np.int64)
    eligible = np.zeros(rows, dtype=bool)

    calendar = np.zeros((rows, 6), dtype=np.float32)
    to_cutoff = np.zeros(rows, dtype=np.float64)

    history_width = int(n_events.max()) if count else 0

    history_mask = np.zeros((count, history_width), dtype=bool)
    history_rows = np.full((count, history_width), -1, dtype=np.int64)

    value_counts = np.asarray([item.n_values for item in samples], dtype=np.int64)
    value_offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(value_counts, out=value_offsets[1:])

    value_event = np.zeros(int(value_offsets[-1]), dtype=np.int64)

    for number, sample in enumerate(samples):

        base = int(offsets[number])

        for slot in range(sample.n_events):

            row = base + slot
            length = int(sample.event_lengths[slot])
            start = int(sample.event_starts[slot])

            key_ids[row, :length] = sample.key_ids[start : start + length]
            value_ids[row, :length] = sample.value_ids[start : start + length]
            positions[row, :length] = sample.positions[start : start + length]

            event_sample[row] = number
            event_slot[row] = slot

            history_mask[number, slot] = True
            history_rows[number, slot] = row

        if sample.n_events:

            span = slice(base, base + sample.n_events)

            eligible[span] = sample.event_eligible
            calendar[span] = sample.calendar.reshape(sample.n_events, 6)
            to_cutoff[span] = sample.hours_to_cutoff

        # Адрес значения переезжает в координаты batch: номер
        # события внутри примера становится строкой в N, а
        # столбец внутри события не меняется вовсе.
        if sample.n_values:
            first = int(value_offsets[number])
            value_event[first : first + sample.n_values] = sample.value_event.astype(np.int64) + base

    token_mask = np.arange(width, dtype=np.int64)[None, :] < lengths[:, None] if rows else (
        np.zeros((0, 0), dtype=bool)
    )

    # Область допустимых целей: настоящий токен, не ведущий
    # маркер и событие в периоде целей своей группы. Что именно
    # спрятать, решит Masker.
    candidate = token_mask.copy()

    if rows:
        candidate[:, 0] = False
        candidate &= eligible[:, None]

    profile_lengths = np.asarray([item.profile_tokens for item in samples], dtype=np.int64)
    profile_width = int(profile_lengths.max()) if count else 0

    profile_key_ids = np.full((count, profile_width), PAD_ID, dtype=np.int64)
    profile_value_ids = np.full((count, profile_width), PAD_ID, dtype=np.int64)
    profile_positions = np.zeros((count, profile_width), dtype=np.int64)

    profile_counts = np.asarray(
        [int(item.profile_value_start.size) for item in samples], dtype=np.int64
    )
    profile_value_offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(profile_counts, out=profile_value_offsets[1:])

    profile_value_sample = np.zeros(int(profile_value_offsets[-1]), dtype=np.int64)

    for number, sample in enumerate(samples):

        length = int(profile_lengths[number])

        profile_key_ids[number, :length] = sample.profile_key_ids
        profile_value_ids[number, :length] = sample.profile_value_ids
        profile_positions[number, :length] = sample.profile_positions

        first = int(profile_value_offsets[number])
        profile_value_sample[first : first + int(profile_counts[number])] = number

    profile_mask = (
        np.arange(profile_width, dtype=np.int64)[None, :] < profile_lengths[:, None]
        if profile_width
        else np.zeros((count, 0), dtype=bool)
    )

    age = np.asarray(
        [np.nan if item.history_age_days is None else item.history_age_days for item in samples],
        dtype=np.float64,
    )

    return Batch(
        sample_ids=[item.sample_id for item in samples],
        client_ids=[item.client_id for item in samples],
        cutoffs=[item.cutoff for item in samples],
        groups=[item.group for item in samples],
        sample_seeds=np.asarray([item.sample_seed for item in samples], dtype=np.int64),
        weights=np.asarray([item.weight for item in samples], dtype=np.float32),
        has_targets=np.asarray([item.has_targets for item in samples], dtype=bool),
        n_events=n_events,
        event_offsets=offsets,
        event_key_ids=key_ids,
        event_value_ids=value_ids,
        event_positions=positions,
        event_token_mask=token_mask,
        target_candidate_mask=candidate,
        event_lengths=lengths,
        event_sample=event_sample,
        event_slot=event_slot,
        event_eligible=eligible,
        history_mask=history_mask,
        history_rows=history_rows,
        calendar=calendar,
        hours_to_cutoff=to_cutoff,
        coverage_at_cutoff=_stack([item.coverage_at_cutoff for item in samples], np.int8),
        history_age_days=age,
        history_age_known=~np.isnan(age),
        value_offsets=value_offsets,
        value_event=value_event,
        value_start=_join([item.value_start for item in samples]),
        value_length=_join([item.value_length for item in samples]),
        value_key_id=_join([item.value_key_id for item in samples]),
        profile_key_ids=profile_key_ids,
        profile_value_ids=profile_value_ids,
        profile_positions=profile_positions,
        profile_token_mask=profile_mask,
        profile_known=np.asarray([item.has_profile for item in samples], dtype=bool),
        profile_value_offsets=profile_value_offsets,
        profile_value_sample=profile_value_sample,
        profile_value_start=_join([item.profile_value_start for item in samples]),
        profile_value_length=_join([item.profile_value_length for item in samples]),
        profile_value_key_id=_join([item.profile_value_key_id for item in samples]),
        **_dependencies(samples, value_offsets, offsets, profile_value_offsets),
    )


def _join(arrays: list[np.ndarray]) -> np.ndarray:

    if not arrays:
        return np.zeros(0, dtype=np.int64)

    return np.concatenate(arrays).astype(np.int64)


def _stack(arrays: list[np.ndarray], dtype) -> np.ndarray:

    width = max((item.size for item in arrays), default=0)

    out = np.zeros((len(arrays), width), dtype=dtype)

    for number, item in enumerate(arrays):
        if item.size != width:
            raise CollateError(
                "доступность источников разной ширины: список источников обязан быть общим"
            )
        out[number] = item

    return out


def _dependencies(samples: list[Sample], value_offsets: np.ndarray,
                  event_offsets: np.ndarray, profile_offsets: np.ndarray) -> dict:
    """
    Происхождение в координатах batch.

    Индекс значения сдвигается на начало примера, номер
    события-источника — на начало его событий. Пустой адрес
    (-1) остаётся пустым: сдвигать «нет адреса» нельзя.

    Значения анкеты живут в СВОЁМ пространстве индексов, и сдвиг
    у них свой. Сложить их с индексами событий значило бы дать
    Masker адрес, указывающий в чужую запись.
    """

    value: list[int] = []
    source_event: list[int] = []
    source_value: list[int] = []
    status: list[int] = []

    for number, sample in enumerate(samples):

        base_value = int(value_offsets[number])
        base_event = int(event_offsets[number])

        for index in range(len(sample.dependencies)):

            value.append(sample.dependencies.value[index] + base_value)

            event = sample.dependencies.source_event[index]
            source_event.append(-1 if event < 0 else event + base_event)

            code = sample.dependencies.status[index]

            target = sample.dependencies.source_value[index]

            if target < 0:
                source_value.append(-1)
            elif code == DEP_PROFILE:
                source_value.append(target + int(profile_offsets[number]))
            else:
                source_value.append(target + base_value)

            status.append(code)

    return {
        "dep_value": np.asarray(value, dtype=np.int64),
        "dep_source_event": np.asarray(source_event, dtype=np.int64),
        "dep_source_value": np.asarray(source_value, dtype=np.int64),
        "dep_status": np.asarray(status, dtype=np.int8),
    }


__all__ = [
    "PAD_ID",
    "Batch",
    "CollateError",
    "collate",
]
