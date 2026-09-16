from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch

from src.tokenizer.build import check_order
from src.tokenizer.dataset import Example, TokenBatch

from .batching import BatchError, PaddedRecords, check_batch, events_from_batch, profiles_from_batch
from .config import ModelConfig
from .time_features import (
    INACTIVITY_NORM_HOURS,
    age_hours,
    calendar_features,
    inactivity_feature,
    squash_np,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Пример это пара (client_id, cutoff), а не клиент: у одного
# клиента несколько срезов, и cutoff у них разный. В TokenBatch
# cutoff нет, поэтому он приходит отдельной metadata.
#
# История обрезается ДО Event Encoder: кодировать выброшенные
# события бессмысленно. Обрезка строит новый корректный
# TokenBatch, поэтому runtime-masker работает на нём без
# изменений, а mapping к исходному batch сохраняется.
#
# Возраст считается по ПОЛНОЙ истории до обрезки: cutoff у
# примера один, и отбрасывание старых событий его не двигает.
#
# Вся эта подготовка это NumPy. Тензоры появляются только в
# to_model_inputs, а forward модели про parquet не знает.
# ============================================================




# ============================================================
# METADATA
# ============================================================


@dataclass(frozen=True)
class HistoryMeta:
    """
    Cutoff и дата as-of снимка каждого примера.

    Сверить их с токенами невозможно: в профильных массивах
    TokenBatch нет ts. Поэтому единственный санкционированный
    конструктор это metadata_from_examples, и строить её нужно
    из того же списка examples, который ушёл в collate.
    """

    client_ids: np.ndarray
    cutoffs: np.ndarray
    snapshot_ts: np.ndarray

    def __len__(self) -> int:
        return int(self.client_ids.size)

    @property
    def window_start(self) -> np.ndarray:
        """
        Начало месяца наблюдения каждого примера.

        Пример это пара (клиент, cutoff), а cutoff по контракту
        preprocessing это начало СЛЕДУЮЩЕГО месяца. Значит месяц
        наблюдения задан самим cutoff и отдельного поля не
        требует.

        Берётся месяц последнего момента, который вообще может
        попасть в историю, то есть cutoff минус микросекунда.
        Определение тотальное: на настоящем cutoff (начало
        месяца) оно даёт ровно observation_month, а на любом
        другом моменте всё равно даёт месяц, непосредственно
        предшествующий cutoff, а не съехавшее окно.

        События с ts >= window_start это то новое, что месяц
        добавил. Метрика по ним отвечает на другой вопрос, чем
        метрика по всей истории: как модель читает свежее, а не
        как она в среднем помнит прошлое.
        """

        last = np.asarray(self.cutoffs).astype("datetime64[us]") - np.timedelta64(1, "us")

        return last.astype("datetime64[M]")


def metadata_from_examples(examples: Sequence[Example]) -> HistoryMeta:
    """
    Metadata строго в том же порядке, в каком collate сложил batch.
    """

    if len(examples) == 0:
        raise BatchError("нет примеров: metadata собирать не из чего")

    return HistoryMeta(
        client_ids=np.array([example.client_id for example in examples], dtype=np.int64),
        cutoffs=np.array([np.datetime64(example.cutoff, "us") for example in examples]),
        snapshot_ts=np.array([np.datetime64(example.snapshot_ts, "us") for example in examples]),
    )


# ============================================================
# ПРОВЕРКА
# ============================================================


def example_lengths(example_of_event: np.ndarray, n_examples: int) -> np.ndarray:
    return np.bincount(example_of_event, minlength=n_examples).astype(np.int64)


def validate_history_batch(batch: TokenBatch, meta: HistoryMeta) -> None:
    """
    Batch согласован сам с собой и с metadata.
    """

    check_batch(batch)

    if len(meta) != batch.n_examples:
        raise BatchError(
            f"metadata на {len(meta)} примеров, а в batch их {batch.n_examples}"
        )

    owner = np.asarray(batch.profile_example_ids, dtype=np.int64)

    profiles = np.unique(owner)

    if profiles.size != batch.n_examples or profiles[0] != 0 or profiles[-1] != batch.n_examples - 1:
        raise BatchError(
            f"профиль обязан быть ровно у каждого примера, найдено {profiles.size} на {batch.n_examples}"
        )

    example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)

    if example_of_event.size and (np.diff(example_of_event) < 0).any():
        raise BatchError("события не сгруппированы по примерам: example_of_event убывает")

    ts = np.asarray(batch.ts)
    seq = np.asarray(batch.seq, dtype=np.int64)

    # Тот же порядок (ts, seq), что в ленте; ключ группировки
    # здесь пример, а не клиент.
    try:
        check_order(example_of_event, ts, seq)
    except ValueError as error:
        raise BatchError(f"история не упорядочена по (ts, seq): {error}") from error

    if example_of_event.size:

        limit = np.asarray(meta.cutoffs)[example_of_event]

        late = np.flatnonzero(ts >= limit)

        if late.size:
            position = int(late[0])
            index = int(example_of_event[position])
            raise BatchError(
                f"пример {index} (клиент {int(meta.client_ids[index])}): событие {ts[position]} "
                f"не раньше cutoff {meta.cutoffs[index]}"
            )

    bad = np.flatnonzero(np.asarray(meta.snapshot_ts) >= np.asarray(meta.cutoffs))

    if bad.size:
        index = int(bad[0])
        raise BatchError(
            f"пример {index} (клиент {int(meta.client_ids[index])}): снимок профиля "
            f"{meta.snapshot_ts[index]} не раньше cutoff {meta.cutoffs[index]}"
        )


# ============================================================
# ОБРЕЗКА
# ============================================================


@dataclass(frozen=True)
class TruncationInfo:
    original_history_length: np.ndarray
    used_history_length: np.ndarray
    truncated: np.ndarray
    kept_events: np.ndarray
    kept_tokens: np.ndarray
    slot_of_event: np.ndarray

    @property
    def truncated_share(self) -> float:
        return float(self.truncated.mean()) if self.truncated.size else 0.0


def keep_everything(batch: TokenBatch) -> tuple[TokenBatch, TruncationInfo]:
    """
    Лимита нет: история идёт целиком.

    Отдельная ветвь, а не max_events = бесконечность. Общий путь
    собирает kept_tokens списковым включением по событиям, и на
    полных историях это миллионы итераций Python ради
    тождественного отображения.
    """

    example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)

    original = example_lengths(example_of_event, batch.n_examples)

    n_events = batch.n_events

    starts = np.concatenate([[0], np.cumsum(original)[:-1]])

    info = TruncationInfo(
        original_history_length=original,
        used_history_length=original,
        truncated=np.zeros(batch.n_examples, dtype=bool),
        kept_events=np.arange(n_events, dtype=np.int64),
        kept_tokens=np.arange(batch.n_tokens, dtype=np.int64),
        slot_of_event=np.arange(n_events, dtype=np.int64) - starts[example_of_event] + 1,
    )

    return batch, info


def truncate_recent(batch: TokenBatch, max_events: int | None) -> tuple[TokenBatch, TruncationInfo]:
    """
    Оставляет последние max_events событий каждого примера.

    Короткая история сохраняется целиком. Профиль не трогается:
    [USR] в лимит не входит, это отдельная позиция истории.

    max_events = None означает отсутствие лимита.
    """

    if max_events is None:
        return keep_everything(batch)

    if max_events < 1:
        raise BatchError("max_events должен быть положительным")

    example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)

    original = example_lengths(example_of_event, batch.n_examples)

    # Номер события с конца внутри своего примера.
    ends = np.cumsum(original)

    from_end = ends[example_of_event] - np.arange(example_of_event.size, dtype=np.int64)

    keep = from_end <= max_events

    kept_events = np.flatnonzero(keep)

    used = np.minimum(original, max_events)

    widths = np.diff(np.asarray(batch.event_offsets, dtype=np.int64))

    kept_tokens = np.concatenate(
        [
            np.arange(start, start + width, dtype=np.int64)
            for start, width in zip(
                np.asarray(batch.event_offsets, dtype=np.int64)[kept_events], widths[kept_events]
            )
        ]
    ) if kept_events.size else np.zeros(0, dtype=np.int64)

    new_widths = widths[kept_events]

    offsets = np.zeros(kept_events.size + 1, dtype=np.int64)
    np.cumsum(new_widths, out=offsets[1:])

    new_example_of_event = example_of_event[kept_events]

    slot = np.arange(kept_events.size, dtype=np.int64) - np.concatenate(
        [[0], np.cumsum(used)[:-1]]
    )[new_example_of_event] + 1

    truncated = TokenBatch(
        key_ids=batch.key_ids[kept_tokens],
        value_ids=batch.value_ids[kept_tokens],
        positions=batch.positions[kept_tokens],
        field_ids=batch.field_ids[kept_tokens],
        event_ids=np.repeat(np.arange(kept_events.size, dtype=np.int64), new_widths),
        example_ids=np.repeat(new_example_of_event, new_widths),
        event_offsets=offsets,
        example_of_event=new_example_of_event,
        event_type=np.asarray(batch.event_type)[kept_events],
        ts=np.asarray(batch.ts)[kept_events],
        seq=np.asarray(batch.seq, dtype=np.int64)[kept_events],
        profile_key_ids=batch.profile_key_ids,
        profile_value_ids=batch.profile_value_ids,
        profile_positions=batch.profile_positions,
        profile_example_ids=batch.profile_example_ids,
        profile_field_ids=batch.profile_field_ids,
        n_examples=batch.n_examples,
    )

    info = TruncationInfo(
        original_history_length=original,
        used_history_length=used,
        truncated=original > max_events,
        kept_events=kept_events,
        kept_tokens=kept_tokens,
        slot_of_event=slot,
    )

    return truncated, info


# ============================================================
# ПОДГОТОВЛЕННЫЙ BATCH
# ============================================================


@dataclass(frozen=True)
class HistoryBatch:
    tokens: TokenBatch
    meta: HistoryMeta
    info: TruncationInfo
    age_hours: np.ndarray
    targets: np.ndarray | None = None
    mask: np.ndarray | None = None

    # Что именно выбрал masker: режим, доступные позиции, счётчики
    # стратегий до объединения и итоговая доля.
    masking: dict | None = None

    @property
    def n_examples(self) -> int:
        return self.tokens.n_examples

    @property
    def n_events(self) -> int:
        return self.tokens.n_events


def prepare_history_batch(
    batch: TokenBatch,
    meta: HistoryMeta,
    max_events: int | None,
    masker=None,
    step: int = 0,
) -> HistoryBatch:
    """
    Проверка, временные признаки, обрезка и маски.

    Порядок важен: masker применяется ПОСЛЕ обрезки, чтобы не
    тратить маски на события, которых модель не увидит.
    """

    validate_history_batch(batch, meta)

    ages = age_hours(batch, meta.cutoffs)

    tokens, info = truncate_recent(batch, max_events)

    kept = info.kept_events

    targets = None
    mask = None
    diagnostics = None

    if masker is not None:

        # Идентичность примера это пара (клиент, cutoff): ею
        # засевается поток маски. Обрезка число примеров не
        # меняет, поэтому metadata остаётся выровненной.
        identities = np.stack(
            [
                np.asarray(meta.client_ids, dtype=np.int64),
                np.asarray(meta.cutoffs).astype("datetime64[us]").astype(np.int64),
            ],
            axis=1,
        )

        masked = masker.apply(tokens, step, identities)

        tokens = replace(tokens, value_ids=masked.value_ids)

        targets = masked.targets
        mask = masked.mask
        diagnostics = masked.diagnostics()

    return HistoryBatch(
        tokens=tokens,
        meta=meta,
        info=info,
        age_hours=ages[kept],
        targets=targets,
        mask=mask,
        masking=diagnostics,
    )


# ============================================================
# ТЕНЗОРЫ
# ============================================================


@dataclass(frozen=True)
class TemporalInputs:
    """
    Признаки времени: координаты поворота и то, чего поворот
    выразить не может.

    calendar относится к КАЖДОМУ событию, поэтому нумерация
    строк та же, что у batch событий.

    Координаты это часы до ПОСЛЕДНЕГО элемента истории примера,
    сжатые squash. У последнего элемента координата ноль, дальше
    в прошлое она растёт. Внимание видит разность координат, то
    есть время между парой элементов.

    inactivity хранится готовым к подаче в FeatureMLP, то есть
    [B, 1], а не [B].
    """

    calendar: torch.Tensor                  # [n_events, 6]
    event_coords: torch.Tensor              # [n_events]
    inactivity: torch.Tensor                # [B, 1]

    @property
    def n_examples(self) -> int:
        return int(self.inactivity.shape[0])

    def to(self, device) -> "TemporalInputs":
        return TemporalInputs(
            calendar=self.calendar.to(device),
            event_coords=self.event_coords.to(device),
            inactivity=self.inactivity.to(device),
        )


@dataclass(frozen=True)
class ModelInputs:
    """
    Всё, что нужно forward. Targets сюда не попадают: это цель,
    а не признак.

    slot_of_event это позиция события в истории примера; слот 0
    занят профилем.
    """

    events: PaddedRecords
    profiles: PaddedRecords
    example_of_event: torch.Tensor
    slot_of_event: torch.Tensor
    used_history_length: torch.Tensor
    n_examples: int
    kept_events: torch.Tensor
    kept_tokens: torch.Tensor

    temporal: TemporalInputs | None = None

    @property
    def n_events(self) -> int:
        return int(self.example_of_event.numel())

    @property
    def max_length(self) -> int:
        return 1 + int(self.used_history_length.max().item())

    def to(self, device) -> "ModelInputs":
        return ModelInputs(
            events=self.events.to(device),
            profiles=self.profiles.to(device),
            example_of_event=self.example_of_event.to(device),
            slot_of_event=self.slot_of_event.to(device),
            used_history_length=self.used_history_length.to(device),
            n_examples=self.n_examples,
            kept_events=self.kept_events.to(device),
            kept_tokens=self.kept_tokens.to(device),
            temporal=None if self.temporal is None else self.temporal.to(device),
        )


def _long(values) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.int64))


def _temporal_inputs(history: HistoryBatch) -> TemporalInputs:
    """
    Календарь событий, координаты элементов и простой клиента.

    Координата это возраст события МИНУС возраст последнего
    события того же примера. Вычитание обязано идти по примеру,
    а не по batch: у соседа по batch свой cutoff и своя лента.
    """

    tokens = history.tokens

    n_examples = int(tokens.n_examples)

    example_of_event = np.asarray(tokens.example_of_event, dtype=np.int64)

    ages = np.asarray(history.age_hours, dtype=np.float64)

    # Возраст самого свежего события примера.
    last_age = np.full(n_examples, np.inf, dtype=np.float64)

    np.minimum.at(last_age, example_of_event, ages)

    # Пример без единого элемента невозможен по контракту
    # preprocessing; если он всё же случится, простой берётся
    # максимальным, а не бесконечным.
    empty = ~np.isfinite(last_age)

    if empty.any():
        last_age[empty] = INACTIVITY_NORM_HOURS

    def coords_of(ages: np.ndarray, owner: np.ndarray) -> np.ndarray:

        elapsed = ages - last_age[owner]

        negative = np.flatnonzero(elapsed < 0.0)

        if negative.size:
            position = int(negative[0])
            raise BatchError(
                f"событие {position} примера {int(owner[position])} старше самого свежего "
                f"на {elapsed[position]:.6f} ч: история повреждена"
            )

        return squash_np(elapsed).astype(np.float32)

    return TemporalInputs(
        calendar=torch.from_numpy(calendar_features(tokens.ts)),
        event_coords=torch.from_numpy(coords_of(ages, example_of_event)),
        inactivity=torch.from_numpy(inactivity_feature(last_age).reshape(n_examples, 1)),
    )


def to_model_inputs(history: HistoryBatch, config: ModelConfig, device=None) -> ModelInputs:

    tokens = history.tokens

    slot_of_event = np.asarray(history.info.slot_of_event, dtype=np.int64)
    used = np.asarray(history.info.used_history_length, dtype=np.int64)

    inputs = ModelInputs(
        events=events_from_batch(tokens, config),
        profiles=profiles_from_batch(tokens, config),
        example_of_event=_long(tokens.example_of_event),
        slot_of_event=torch.from_numpy(slot_of_event),
        used_history_length=torch.from_numpy(used),
        n_examples=tokens.n_examples,
        kept_events=_long(history.info.kept_events),
        kept_tokens=_long(history.info.kept_tokens),
        temporal=_temporal_inputs(history),
    )

    return inputs if device is None else inputs.to(device)
