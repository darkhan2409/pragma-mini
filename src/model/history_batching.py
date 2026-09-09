from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch

from src.tokenizer.build import check_order
from src.tokenizer.dataset import Example, TokenBatch

from .batching import BatchError, PaddedRecords, check_batch, events_from_batch, profiles_from_batch
from .config import STRUCTURE_EVENT, STRUCTURE_SESSION, ModelConfig
from .session_batching import HistoryLayout, group_events
from .time_encoding import age_hours, gap_hours


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
# Временные признаки считаются по ПОЛНОЙ истории до обрезки:
# у первого оставшегося события gap настоящий, а не ноль.
#
# Вся эта подготовка это NumPy. Тензоры появляются только в
# to_model_inputs, а forward модели про parquet не знает.
# ============================================================


POLICY_RECENT = "recent"
POLICY_NONE = "none"


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
    policy: str
    max_events: int | None
    original_history_length: np.ndarray
    used_history_length: np.ndarray
    truncated: np.ndarray
    kept_events: np.ndarray
    kept_tokens: np.ndarray
    slot_of_event: np.ndarray

    @property
    def any_truncated(self) -> bool:
        return bool(self.truncated.any())

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
        policy=POLICY_NONE,
        max_events=None,
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
        n_examples=batch.n_examples,
    )

    info = TruncationInfo(
        policy=POLICY_RECENT,
        max_events=max_events,
        original_history_length=original,
        used_history_length=used,
        truncated=original > max_events,
        kept_events=kept_events,
        kept_tokens=kept_tokens,
        slot_of_event=slot,
    )

    return truncated, info


def _previous_ts(batch: TokenBatch, info: "TruncationInfo") -> np.ndarray:
    """
    Время последнего ОТБРОШЕННОГО обрезкой события примера.

    NaT означает, что до уцелевшего окна ничего не было. Без
    этого первый элемент истории получил бы gap ноль и выглядел
    бы началом жизни клиента, чем он не является.
    """

    n_examples = int(batch.n_examples)

    previous = np.full(n_examples, np.datetime64("NaT", "us"), dtype="datetime64[us]")

    dropped = np.asarray(info.original_history_length, dtype=np.int64) - np.asarray(
        info.used_history_length, dtype=np.int64
    )

    if not bool((dropped > 0).any()):
        return previous

    example_of_event = np.asarray(batch.example_of_event, dtype=np.int64)
    ts = np.asarray(batch.ts)

    starts = np.zeros(n_examples + 1, dtype=np.int64)
    np.cumsum(np.bincount(example_of_event, minlength=n_examples), out=starts[1:])

    for index in np.flatnonzero(dropped > 0):
        previous[index] = ts[starts[index] + dropped[index] - 1]

    return previous


# ============================================================
# ПОДГОТОВЛЕННЫЙ BATCH
# ============================================================


@dataclass(frozen=True)
class HistoryBatch:
    tokens: TokenBatch
    meta: HistoryMeta
    info: TruncationInfo
    gap_hours: np.ndarray
    age_hours: np.ndarray
    targets: np.ndarray | None = None
    mask: np.ndarray | None = None

    # Что именно выбрал masker: режим, доступные позиции, счётчики
    # стратегий до объединения и итоговая доля.
    masking: dict | None = None

    # Раскладка истории по сессиям. None означает прежнюю
    # структуру: каждое событие занимает свою позицию.
    layout: HistoryLayout | None = None

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
    *,
    session_keys: np.ndarray | None = None,
    structure: str = STRUCTURE_EVENT,
) -> HistoryBatch:
    """
    Проверка, временные признаки, обрезка, раскладка и маски.

    Порядок важен трижды. gap и age считаются ДО обрезки, чтобы
    у первого уцелевшего события был настоящий разрыв. Masker
    применяется ПОСЛЕ обрезки, чтобы не тратить маски на события,
    которых модель не увидит. Раскладка сессий строится МЕЖДУ
    ними: она обязана видеть только уцелевшие события и не имеет
    права зависеть от того, что замаскировано.
    """

    validate_history_batch(batch, meta)

    gaps = gap_hours(batch)
    ages = age_hours(batch, meta.cutoffs)

    tokens, info = truncate_recent(batch, max_events)

    kept = info.kept_events

    layout = None

    if structure == STRUCTURE_SESSION:

        if session_keys is None:
            raise BatchError(
                "структура session требует ключей сессий: "
                "ClientStore должен быть открыт с sessions=True"
            )

        keys = np.asarray(session_keys, dtype=np.int64)

        if keys.size != batch.n_events:
            raise BatchError(
                f"ключей сессий {keys.size}, а событий до обрезки {batch.n_events}"
            )

        layout = group_events(
            example_of_event=tokens.example_of_event,
            ts=tokens.ts,
            seq=tokens.seq,
            session_keys=keys[kept],
            cutoffs=meta.cutoffs,
            previous_ts=_previous_ts(batch, info),
            n_examples=tokens.n_examples,
        )

    targets = None
    mask = None
    diagnostics = None

    if masker is not None:

        # Идентичность примера это пара (клиент, cutoff): схема
        # example засевает ею свой поток. Обрезка число примеров
        # не меняет, поэтому metadata по-прежнему выровнена.
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
        gap_hours=gaps[kept],
        age_hours=ages[kept],
        targets=targets,
        mask=mask,
        masking=diagnostics,
        layout=layout,
    )


# ============================================================
# ТЕНЗОРЫ
# ============================================================


@dataclass(frozen=True)
class SessionInputs:
    """
    Сессии одного batch в тензорах.

    member_rows уже без -1: padding заменён нулём, а правда о
    нём живёт в member_valid. Индекс -1 в torch не ошибка, он
    молча берёт последнюю строку, и такую подмену не видно
    ни в forward, ни в loss.
    """

    session_example: torch.Tensor
    session_slot: torch.Tensor
    session_hours: torch.Tensor

    member_rows: torch.Tensor
    member_valid: torch.Tensor
    member_gap_minutes: torch.Tensor

    session_of_event: torch.Tensor
    position_in_session: torch.Tensor

    n_sessions: int
    max_session_length: int

    def to(self, device) -> "SessionInputs":
        return SessionInputs(
            session_example=self.session_example.to(device),
            session_slot=self.session_slot.to(device),
            session_hours=self.session_hours.to(device),
            member_rows=self.member_rows.to(device),
            member_valid=self.member_valid.to(device),
            member_gap_minutes=self.member_gap_minutes.to(device),
            session_of_event=self.session_of_event.to(device),
            position_in_session=self.position_in_session.to(device),
            n_sessions=self.n_sessions,
            max_session_length=self.max_session_length,
        )


@dataclass(frozen=True)
class ModelInputs:
    """
    Всё, что нужно forward. Targets сюда не попадают: это цель,
    а не признак.

    slot_of_event это слот НЕСУЩЕГО элемента: своего у отдельного
    события и слота сессии у её члена. Поэтому адрес контекстного
    вектора события один и тот же в обеих структурах.

    time_hours относится к отдельным элементам истории. В прежней
    структуре standalone_rows равен None, а отдельные элементы это
    все события, поэтому массив совпадает с прежним.
    """

    events: PaddedRecords
    profiles: PaddedRecords
    example_of_event: torch.Tensor
    slot_of_event: torch.Tensor
    time_hours: torch.Tensor
    used_history_length: torch.Tensor
    n_examples: int
    kept_events: torch.Tensor
    kept_tokens: torch.Tensor

    standalone_rows: torch.Tensor | None = None
    sessions: SessionInputs | None = None

    @property
    def n_events(self) -> int:
        return int(self.example_of_event.numel())

    @property
    def n_standalone(self) -> int:
        return self.n_events if self.standalone_rows is None else int(self.standalone_rows.numel())

    @property
    def max_length(self) -> int:
        return 1 + int(self.used_history_length.max().item())

    def to(self, device) -> "ModelInputs":
        return ModelInputs(
            events=self.events.to(device),
            profiles=self.profiles.to(device),
            example_of_event=self.example_of_event.to(device),
            slot_of_event=self.slot_of_event.to(device),
            time_hours=self.time_hours.to(device),
            used_history_length=self.used_history_length.to(device),
            n_examples=self.n_examples,
            kept_events=self.kept_events.to(device),
            kept_tokens=self.kept_tokens.to(device),
            standalone_rows=None if self.standalone_rows is None else self.standalone_rows.to(device),
            sessions=None if self.sessions is None else self.sessions.to(device),
        )


def _long(values) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.int64))


def _session_inputs(layout: HistoryLayout) -> SessionInputs:

    member_rows = np.asarray(layout.member_rows, dtype=np.int64)

    valid = member_rows >= 0

    # Padding заменяется нулём ДО torch: отрицательный индекс
    # в gather не ошибка, он берёт последнюю строку.
    safe = np.where(valid, member_rows, 0)

    return SessionInputs(
        session_example=_long(layout.session_example),
        session_slot=_long(layout.session_slot),
        session_hours=torch.from_numpy(np.asarray(layout.session_hours, dtype=np.float32)),
        member_rows=_long(safe),
        member_valid=torch.from_numpy(valid),
        member_gap_minutes=torch.from_numpy(
            np.asarray(layout.member_gap_minutes, dtype=np.float32)
        ),
        session_of_event=_long(layout.session_of_event),
        position_in_session=_long(layout.position_in_session),
        n_sessions=layout.n_sessions,
        max_session_length=layout.max_session_length,
    )


def to_model_inputs(history: HistoryBatch, config: ModelConfig, device=None) -> ModelInputs:

    tokens = history.tokens

    layout = history.layout

    if layout is None:

        time_hours = np.stack(
            [
                np.asarray(history.gap_hours, dtype=np.float32),
                np.asarray(history.age_hours, dtype=np.float32),
            ],
            axis=1,
        )

        slot_of_event = np.asarray(history.info.slot_of_event, dtype=np.int64)
        used = np.asarray(history.info.used_history_length, dtype=np.int64)

        standalone_rows = None
        sessions = None

    else:

        if layout.n_events != tokens.n_events:
            raise BatchError(
                f"раскладка построена на {layout.n_events} событиях, а в batch их {tokens.n_events}"
            )

        time_hours = np.asarray(layout.standalone_hours, dtype=np.float32)

        slot_of_event = np.asarray(layout.slot_of_event, dtype=np.int64)
        used = np.asarray(layout.used_history_length, dtype=np.int64)

        standalone_rows = _long(layout.standalone_rows)
        sessions = _session_inputs(layout)

    inputs = ModelInputs(
        events=events_from_batch(tokens, config),
        profiles=profiles_from_batch(tokens, config),
        example_of_event=_long(tokens.example_of_event),
        slot_of_event=torch.from_numpy(slot_of_event),
        time_hours=torch.from_numpy(time_hours),
        used_history_length=torch.from_numpy(used),
        n_examples=tokens.n_examples,
        kept_events=_long(history.info.kept_events),
        kept_tokens=_long(history.info.kept_tokens),
        standalone_rows=standalone_rows,
        sessions=sessions,
    )

    return inputs if device is None else inputs.to(device)
