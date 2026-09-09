from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .history_batching import HistoryBatch
from .mlm_head import FieldTable, TargetError


# ============================================================
# ИДЕЯ
# ============================================================
#
# Цели живут отдельно от входа модели. ModelInputs не знает о
# них ничего: иначе достаточно было бы одной ошибки в сборке,
# чтобы правильный ответ попал во вход и метрики стали
# бессмысленными.
#
# Для каждой замаскированной позиции нужно ЧЕТЫРЕ адреса:
#
#   event_row  строка padded batch событий  → h_local и h_event
#   col        позиция токена внутри события → h_local
#   example    номер примера                 → h_usr
#   key_id     поле                          → своя голова
#
# Плюс обратное соответствие исходному batch (до обрезки), уже
# посчитанное truncate_recent: kept_tokens и kept_events.
#
# Позиции вырожденных полей отбрасываются здесь, а не в голове:
# тогда их скрытые состояния даже не собираются.
# ============================================================


@dataclass(frozen=True)
class MaskedTargets:
    """
    Адреса замаскированных позиций и их правильные значения.
    """

    flat: np.ndarray
    event_row: np.ndarray
    col: np.ndarray
    example: np.ndarray
    key_ids: np.ndarray
    global_targets: np.ndarray
    local_targets: np.ndarray

    original_tokens: np.ndarray
    original_events: np.ndarray

    # Сколько позиций выбрал masker до отсева вырожденных полей.
    n_masked: int
    n_degenerate: int

    # Цель на событии месяца наблюдения, а не более раннего.
    # Массив numpy и в tensors() не входит: тот словарь едет
    # в representations и within_session, и лишний ключ там
    # был бы связью, которой нет.
    recent: np.ndarray | None = None

    @property
    def n_recent(self) -> int:
        return 0 if self.recent is None else int(self.recent.sum())

    @property
    def n(self) -> int:
        return int(self.flat.size)

    @property
    def fields_present(self) -> tuple[int, ...]:
        return tuple(int(key) for key in np.unique(self.key_ids)) if self.n else ()

    @property
    def n_fields(self) -> int:
        return len(self.fields_present)

    # --------------------------------------------------------

    def gather(self, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Пара (строка события, позиция токена) для Event Encoder.
        """

        rows = torch.from_numpy(np.ascontiguousarray(self.event_row))
        cols = torch.from_numpy(np.ascontiguousarray(self.col))

        return (rows, cols) if device is None else (rows.to(device), cols.to(device))

    def tensors(self, device=None) -> dict[str, torch.Tensor]:

        made = {
            "event_row": torch.from_numpy(np.ascontiguousarray(self.event_row)),
            "col": torch.from_numpy(np.ascontiguousarray(self.col)),
            "example": torch.from_numpy(np.ascontiguousarray(self.example)),
            "key_ids": torch.from_numpy(np.ascontiguousarray(self.key_ids)),
            "local_targets": torch.from_numpy(np.ascontiguousarray(self.local_targets)),
        }

        return made if device is None else {name: value.to(device) for name, value in made.items()}

    def digest(self) -> str:
        """
        Отпечаток набора целей: им сверяется фиксированная validation.
        """

        parts = [
            np.asarray(self.flat, dtype=np.int64),
            np.asarray(self.key_ids, dtype=np.int64),
            np.asarray(self.global_targets, dtype=np.int64),
        ]

        digest = hashlib.sha256()

        for part in parts:
            digest.update(part.tobytes())

        return digest.hexdigest()


# ============================================================
# ПОСТРОЕНИЕ
# ============================================================


def build_targets(history: HistoryBatch, table: FieldTable) -> MaskedTargets:
    """
    Замаскированные позиции обрезанного batch в адреса и цели.
    """

    if history.mask is None or history.targets is None:
        raise TargetError("batch подготовлен без masker: целей нет")

    tokens = history.tokens

    mask = np.asarray(history.mask, dtype=bool)

    if mask.size != tokens.key_ids.size:
        raise TargetError(
            f"маска на {mask.size} токенов, а в batch их {tokens.key_ids.size}"
        )

    flat = np.flatnonzero(mask)

    n_masked = int(flat.size)

    key_ids = np.asarray(tokens.key_ids, dtype=np.int64)[flat]
    global_targets = np.asarray(history.targets, dtype=np.int64)[flat]

    # Вырожденные поля отсеиваются до сбора представлений.
    keep = ~table.degenerate[key_ids]

    n_degenerate = int((~keep).sum())

    flat = flat[keep]
    key_ids = key_ids[keep]
    global_targets = global_targets[keep]

    event_ids = np.asarray(tokens.event_ids, dtype=np.int64)
    offsets = np.asarray(tokens.event_offsets, dtype=np.int64)

    event_row = event_ids[flat]

    col = flat - offsets[event_row]

    example = np.asarray(tokens.example_of_event, dtype=np.int64)[event_row]

    local_targets = table.to_local(key_ids, global_targets)

    kept_tokens = np.asarray(history.info.kept_tokens, dtype=np.int64)
    kept_events = np.asarray(history.info.kept_events, dtype=np.int64)

    # Событие свежее, если оно случилось в месяце наблюдения
    # примера. Граница берётся из cutoff, а не из отдельного
    # поля: cutoff это начало следующего месяца по построению.
    window = np.asarray(history.meta.window_start).astype("datetime64[us]")

    recent = np.asarray(tokens.ts).astype("datetime64[us]")[event_row] >= window[example]

    return MaskedTargets(
        flat=flat,
        event_row=event_row,
        col=col,
        example=example,
        key_ids=key_ids,
        global_targets=global_targets,
        local_targets=local_targets,
        original_tokens=kept_tokens[flat],
        original_events=kept_events[event_row],
        n_masked=n_masked,
        n_degenerate=n_degenerate,
        recent=recent,
    )


def representations(out, targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Три представления одной замаскированной позиции из одного прохода.

        h_local  скрытое состояние позиции после Event Encoder
        h_event  вектор события после History Encoder
        h_usr    вектор клиента после History Encoder
    """

    if out.local_hidden is None:
        raise TargetError("backbone вызван без gather: локальных представлений нет")

    rows = targets["event_row"]

    if int(out.local_hidden.shape[0]) != int(rows.numel()):
        raise TargetError(
            f"локальных представлений {int(out.local_hidden.shape[0])}, а целей {int(rows.numel())}"
        )

    return (
        out.local_hidden,
        out.event_embeddings[rows],
        out.client_embedding[targets["example"]],
    )


def within_session(out, targets: dict[str, torch.Tensor]):
    """
    Состояние события внутри его сессии для сгруппированных целей.

    Возвращает (индексы целей, состояния) либо None, когда
    сессий нет вовсе. Позиция сдвинута на единицу: нулевую
    занимает [SES].
    """

    if out.session_hidden is None or out.session_of_event is None:
        return None

    rows = targets["event_row"]

    owner = out.session_of_event[rows]

    index = torch.nonzero(owner >= 0, as_tuple=True)[0]

    if int(index.numel()) == 0:
        return None

    position = out.position_in_session[rows[index]]

    if bool((position < 0).any()):
        raise TargetError(
            "у сгруппированной цели нет позиции внутри сессии: раскладка повреждена"
        )

    return index, out.session_hidden[owner[index], position + 1]
