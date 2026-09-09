from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batching import BatchError
from .time_encoding import hours_between


# ============================================================
# ИДЕЯ
# ============================================================
#
# Раскладка истории: что станет отдельной позицией History
# Encoder, а что сначала свернётся в сессию.
#
# Сессия это события приложения с одним session_id внутри
# ОДНОГО примера: экраны, операции и баннеры, рождённые в этой
# сессии. Отдельным событием остаётся то, у чего принадлежность
# неизвестна: восстанавливать её по близости во времени значит
# придумывать метаданные, и этого здесь нет.
#
# Функция по сигнатуре не получает ни value_ids, ни targets:
# состав сессии не может зависеть от того, что замаскировано.
# Это гарантия интерфейсом, а не дисциплиной.
#
# Cutoff уже применён раньше: пример это префикс ленты клиента.
# Поэтому сессия на границе содержит ровно доступную часть, а
# её границы и длительность не знают о будущем.
#
# Порядок элементов истории задаётся временем ПОСЛЕДНЕГО
# события элемента: сессия занимает место там, где она
# закончилась. Следствие: транзакция, случившаяся между
# экранами, стоит в истории раньше сессии.
# ============================================================


NO_SESSION = -1

APP_SCREEN = "app_screen"


@dataclass(frozen=True)
class HistoryLayout:
    """
    Кто где стоит в истории и что из чего собрано.

    Индексы событий это строки ОБРЕЗАННОГО TokenBatch: раскладка
    строится после truncate_recent и ничего не переупорядочивает.
    """

    # На событие.
    session_of_event: np.ndarray        # [n_events] int64, -1 = отдельное
    position_in_session: np.ndarray     # [n_events] int64, -1 = отдельное
    slot_of_event: np.ndarray           # [n_events] int64, слот НЕСУЩЕГО элемента

    # Отдельные события.
    standalone_rows: np.ndarray         # [n_standalone] int64, возрастает
    standalone_hours: np.ndarray        # [n_standalone, 2] float64 (gap, age)

    # Сессии.
    session_example: np.ndarray         # [n_sessions] int64
    session_slot: np.ndarray            # [n_sessions] int64, 1-based
    session_hours: np.ndarray           # [n_sessions, 2] float64
    member_rows: np.ndarray             # [n_sessions, S] int64, -1 = padding
    member_gap_minutes: np.ndarray      # [n_sessions, S] float64
    session_length: np.ndarray          # [n_sessions] int64
    session_key: np.ndarray             # [n_sessions] int64
    session_start: np.ndarray           # [n_sessions] datetime64[us]
    session_end: np.ndarray             # [n_sessions] datetime64[us]

    # На пример.
    used_history_length: np.ndarray     # [B] int64 = отдельные + сессии

    @property
    def n_events(self) -> int:
        return int(self.session_of_event.size)

    @property
    def n_sessions(self) -> int:
        return int(self.session_slot.size)

    @property
    def n_grouped(self) -> int:
        return int((self.session_of_event >= 0).sum())

    @property
    def max_session_length(self) -> int:
        return int(self.member_rows.shape[1]) if self.n_sessions else 0

    def as_dict(self) -> dict:
        return {
            "n_events": self.n_events,
            "n_sessions": self.n_sessions,
            "n_grouped": self.n_grouped,
            "n_standalone": int(self.standalone_rows.size),
            "max_session_length": self.max_session_length,
            "used_history_length": self.used_history_length.tolist(),
        }


# ============================================================
# ПОСТРОЕНИЕ
# ============================================================


def group_events(
    example_of_event: np.ndarray,
    ts: np.ndarray,
    seq: np.ndarray,
    session_keys: np.ndarray,
    cutoffs: np.ndarray,
    previous_ts: np.ndarray | None = None,
    n_examples: int | None = None,
) -> HistoryLayout:
    """
    Раскладка истории по событиям одного batch.

    previous_ts это время последнего ОТБРОШЕННОГО обрезкой
    события каждого примера (NaT, если ничего не отброшено).
    Оно нужно, чтобы у первого уцелевшего элемента был
    настоящий gap, как и у событий в v1.

    Тип события сюда не передаётся вовсе: принадлежность к
    сессии несёт ключ, и только он. Что ключ бывает лишь у
    событий приложения, проверяет sidecar при сборке, где
    известна колонка каждого типа.
    """

    example_of_event = np.asarray(example_of_event, dtype=np.int64)
    seq = np.asarray(seq, dtype=np.int64)
    ts = np.asarray(ts)
    keys = np.asarray(session_keys, dtype=np.int64)
    cutoffs = np.asarray(cutoffs)

    n_events = int(example_of_event.size)

    if keys.size != n_events:
        raise BatchError(
            f"ключей сессий {keys.size}, а событий {n_events}: "
            "sidecar должен быть срезан теми же индексами, что и события"
        )

    if ts.size != n_events or seq.size != n_events:
        raise BatchError("массивы событий разной длины: раскладку строить не из чего")

    batch = int(cutoffs.size if n_examples is None else n_examples)

    if n_events and int(example_of_event.max()) >= batch:
        raise BatchError("event ссылается на пример вне batch")

    # --------------------------------------------------------
    # КТО В СЕССИИ
    # --------------------------------------------------------
    #
    # Ключ уникален только внутри клиента, а пример это префикс
    # ленты одного клиента, поэтому группировать надо по паре
    # (пример, ключ): один и тот же ключ у разных примеров это
    # разные сессии.
    # --------------------------------------------------------

    # Тип события не проверяется: ключ и есть признак
    # принадлежности. Проверку "только экраны" делает sidecar,
    # где известна колонка каждого типа; здесь она означала бы
    # выбрасывать настоящую метаданную.
    grouped = keys >= 0

    session_of_event = np.full(n_events, NO_SESSION, dtype=np.int64)
    position_in_session = np.full(n_events, NO_SESSION, dtype=np.int64)

    rows = np.flatnonzero(grouped)

    if rows.size:

        pairs = np.stack([example_of_event[rows], keys[rows]], axis=1)

        _, inverse = np.unique(pairs, axis=0, return_inverse=True)

        session_of_event[rows] = np.asarray(inverse, dtype=np.int64).ravel()

    n_sessions = int(session_of_event.max()) + 1 if rows.size else 0

    standalone_rows = np.flatnonzero(session_of_event < 0).astype(np.int64)

    # --------------------------------------------------------
    # ПОРЯДОК ЭЛЕМЕНТОВ
    # --------------------------------------------------------

    if n_sessions:

        order = np.argsort(session_of_event[rows], kind="stable")
        ordered = rows[order]

        owner = session_of_event[ordered]

        starts = np.flatnonzero(np.concatenate([[True], owner[1:] != owner[:-1]]))

        position_in_session[ordered] = np.arange(ordered.size, dtype=np.int64) - np.repeat(
            starts, np.diff(np.concatenate([starts, [ordered.size]]))
        )

        session_length = np.diff(np.concatenate([starts, [ordered.size]])).astype(np.int64)

        session_example = example_of_event[ordered[starts]]
        session_key = keys[ordered[starts]]
        session_start = ts[ordered[starts]]

        last = np.concatenate([starts[1:], [ordered.size]]) - 1
        session_end = ts[ordered[last]]
        session_seq = seq[ordered[last]]

        width = int(session_length.max())

        member_rows = np.full((n_sessions, width), NO_SESSION, dtype=np.int64)
        member_rows[owner, position_in_session[ordered]] = ordered

    else:
        session_length = np.zeros(0, dtype=np.int64)
        session_example = np.zeros(0, dtype=np.int64)
        session_key = np.zeros(0, dtype=np.int64)
        session_start = np.zeros(0, dtype=ts.dtype if ts.size else "datetime64[us]")
        session_end = np.zeros(0, dtype=ts.dtype if ts.size else "datetime64[us]")
        session_seq = np.zeros(0, dtype=np.int64)
        member_rows = np.zeros((0, 0), dtype=np.int64)

    # Элементы истории: отдельные события и сессии в одном
    # порядке по (ts, seq) внутри примера.
    item_example = np.concatenate([example_of_event[standalone_rows], session_example])
    item_ts = np.concatenate([ts[standalone_rows], session_end])
    item_seq = np.concatenate([seq[standalone_rows], session_seq])

    item_order = np.lexsort((item_seq, item_ts, item_example))

    used_history_length = np.bincount(item_example.astype(np.int64), minlength=batch).astype(np.int64)

    slot = np.empty(item_order.size, dtype=np.int64)

    if item_order.size:
        ordered_example = item_example[item_order]
        item_starts = np.flatnonzero(
            np.concatenate([[True], ordered_example[1:] != ordered_example[:-1]])
        )
        offsets = np.repeat(item_starts, np.diff(np.concatenate([item_starts, [item_order.size]])))
        slot[item_order] = np.arange(item_order.size, dtype=np.int64) - offsets + 1

    n_standalone = int(standalone_rows.size)

    slot_of_event = np.zeros(n_events, dtype=np.int64)
    slot_of_event[standalone_rows] = slot[:n_standalone]

    session_slot = slot[n_standalone:]

    if n_sessions:
        slot_of_event[rows] = session_slot[session_of_event[rows]]

    # --------------------------------------------------------
    # ВРЕМЯ
    # --------------------------------------------------------

    standalone_hours, session_hours = _item_hours(
        item_example=item_example,
        item_ts=item_ts,
        item_order=item_order,
        n_standalone=n_standalone,
        cutoffs=cutoffs,
        previous_ts=previous_ts,
        batch=batch,
    )

    member_gap_minutes = _member_gaps(member_rows, ts)

    layout = HistoryLayout(
        session_of_event=session_of_event,
        position_in_session=position_in_session,
        slot_of_event=slot_of_event,
        standalone_rows=standalone_rows,
        standalone_hours=standalone_hours,
        session_example=np.asarray(session_example, dtype=np.int64),
        session_slot=np.asarray(session_slot, dtype=np.int64),
        session_hours=session_hours,
        member_rows=member_rows,
        member_gap_minutes=member_gap_minutes,
        session_length=session_length,
        session_key=np.asarray(session_key, dtype=np.int64),
        session_start=session_start,
        session_end=session_end,
        used_history_length=used_history_length,
    )

    _check_layout(layout, batch)

    return layout


def _item_hours(
    item_example: np.ndarray,
    item_ts: np.ndarray,
    item_order: np.ndarray,
    n_standalone: int,
    cutoffs: np.ndarray,
    previous_ts: np.ndarray | None,
    batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    gap до предыдущего элемента истории и age до cutoff.

    Считается по элементам, а не по событиям: после свёртки
    предыдущий сосед у события другой.
    """

    total = int(item_order.size)

    gaps = np.zeros(total, dtype=np.float64)
    ages = np.zeros(total, dtype=np.float64)

    if total:

        ordered_ts = item_ts[item_order]
        ordered_example = item_example[item_order]

        first = np.concatenate([[True], ordered_example[1:] != ordered_example[:-1]])

        ordered_gaps = np.zeros(total, dtype=np.float64)
        ordered_gaps[1:] = hours_between(ordered_ts[1:], ordered_ts[:-1])
        ordered_gaps[first] = 0.0

        if previous_ts is not None:

            previous = np.asarray(previous_ts)

            starts = np.flatnonzero(first)
            known = ~np.isnat(previous[ordered_example[starts]])

            if known.any():
                ordered_gaps[starts[known]] = hours_between(
                    ordered_ts[starts[known]], previous[ordered_example[starts[known]]]
                )

        negative = np.flatnonzero(ordered_gaps < 0)

        if negative.size:
            raise BatchError(
                f"элемент {int(negative[0])} истории раньше предыдущего: "
                f"gap {ordered_gaps[negative[0]]:.6f} ч"
            )

        gaps[item_order] = ordered_gaps

        ages = hours_between(np.asarray(cutoffs)[item_example], item_ts)

        late = np.flatnonzero(ages <= 0)

        if late.size:
            raise BatchError(
                f"элемент {int(late[0])} истории не раньше cutoff: age {ages[late[0]]:.6f} ч"
            )

    standalone_hours = np.stack([gaps[:n_standalone], ages[:n_standalone]], axis=1)
    session_hours = np.stack([gaps[n_standalone:], ages[n_standalone:]], axis=1)

    return standalone_hours, session_hours


def _member_gaps(member_rows: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    Паузы внутри сессии в МИНУТАХ.

    В часах пауза в полминуты это 0.008, и после squash признак
    неотличим от нуля. Единица зафиксирована в ModelConfig.
    """

    if member_rows.size == 0:
        return np.zeros(member_rows.shape, dtype=np.float64)

    valid = member_rows >= 0

    safe = np.where(valid, member_rows, 0)

    member_ts = np.asarray(ts)[safe]

    gaps = np.zeros(member_rows.shape, dtype=np.float64)

    if member_rows.shape[1] > 1:
        gaps[:, 1:] = hours_between(member_ts[:, 1:], member_ts[:, :-1]) * 60.0

    gaps[~valid] = 0.0
    gaps[:, 0] = 0.0

    # Padding идёт только хвостом, поэтому «следующий за
    # padding» невозможен; отрицательных пауз быть не должно.
    negative = np.flatnonzero(gaps.ravel() < 0)

    if negative.size:
        raise BatchError("отрицательная пауза внутри сессии: события не упорядочены по (ts, seq)")

    return gaps


def _check_layout(layout: HistoryLayout, batch: int) -> None:
    """
    Инварианты, без которых ошибка адресации была бы невидимой.
    """

    grouped = np.flatnonzero(layout.session_of_event >= 0)

    members = layout.member_rows[layout.member_rows >= 0]

    if members.size != grouped.size:
        raise BatchError(
            f"в сессиях {members.size} членов, а сгруппированных событий {grouped.size}"
        )

    if members.size and np.unique(members).size != members.size:
        raise BatchError("одно событие попало в сессию дважды")

    if members.size and not np.array_equal(np.sort(members), grouped):
        raise BatchError("состав сессий не совпадает со списком сгруппированных событий")

    if int(layout.standalone_rows.size) + int(grouped.size) != layout.n_events:
        raise BatchError("события потерялись: отдельные плюс сгруппированные не дают всех")

    if layout.used_history_length.size != batch:
        raise BatchError("длина истории посчитана не для всех примеров")

    if layout.n_events:

        if int(layout.slot_of_event.min()) < 1:
            raise BatchError("слот события меньше единицы: позиция 0 занята профилем")

    if layout.n_sessions:

        if int(layout.session_slot.min()) < 1:
            raise BatchError("слот сессии меньше единицы")

        if int(layout.session_length.sum()) != int(grouped.size):
            raise BatchError("сумма длин сессий не равна числу сгруппированных событий")

        limit = layout.used_history_length[layout.session_example]

        if bool((layout.session_slot > limit).any()):
            raise BatchError("слот сессии выходит за длину истории своего примера")


# ============================================================
# СТАТИСТИКА
# ============================================================


def summarize_layout(layout: HistoryLayout, original_lengths: np.ndarray) -> dict:
    """
    Сводка одной раскладки для отчёта.
    """

    original = np.asarray(original_lengths, dtype=np.int64)
    used = np.asarray(layout.used_history_length, dtype=np.int64)

    return {
        "n_events": layout.n_events,
        "n_sessions": layout.n_sessions,
        "n_grouped": layout.n_grouped,
        "n_standalone": int(layout.standalone_rows.size),
        "session_lengths": layout.session_length.tolist(),
        "before": original.tolist(),
        "after": used.tolist(),
    }


__all__ = [
    "APP_SCREEN",
    "NO_SESSION",
    "HistoryLayout",
    "group_events",
    "summarize_layout",
]
