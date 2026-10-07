from __future__ import annotations

import numpy as np

from .choose import REASONS, Choice, NONE, Selection, Value


# ============================================================
# ПОДСТАНОВКА
# ============================================================
#
# Единственное место, где используется словарь, и единственное,
# где меняются value_ids. Случайности здесь нет: что прятать,
# решено раньше.
#
# Все куски выбранного значения получают ОДНО решение, потому
# что подстановка идёт по диапазону значения, а не по токенам:
#
#   [MASK]  во все куски, а в labels ложатся исходные value_ids
#           каждого куска — это и есть задача модели;
#   [UNK]   во все куски, labels остаются -100, причина не
#           пишется. Значение испорчено, но целью не является:
#           модель видит незнакомое значение и не получает за
#           него ни награды, ни штрафа.
#
# key_ids сюда не приходят вовсе, поэтому изменить их нельзя:
# ключ виден, предсказывается значение.
#
# corrupted — значения выбранного механизмом key ключа вне целей,
# испорченные ради ключа (key_context_corruption_probability):
# [UNK] во все куски, labels -100 и без причины — это вход, а не
# задача.
#
# Длина последовательности не меняется: ни один токен не
# добавляется и не выбрасывается.
# ============================================================


# Позиция вне loss. Это соглашение PyTorch: ignore_index у
# перекрёстной энтропии по умолчанию равен -100.
IGNORE = -100


class MaskError(ValueError):
    """
    Маску применить нельзя.
    """


def apply(client_id: str, row: dict, choices: list[Choice], mask: int,
          unknown: int, corrupted: tuple[Value, ...] = ()) -> dict:
    """
    Четыре массива по длине строки клиента.
    """

    source = list(row["value_ids"])
    width = len(source)

    value_ids = list(source)
    labels = [IGNORE] * width
    reason = [NONE] * width

    # Занятость отслеживается отдельно: у [UNK] ни labels, ни
    # причина не пишутся, и по ним наложение было бы не видно.
    taken = [False] * width

    closed = [(choice.value, choice) for choice in choices] + [(value, None) for value in corrupted]

    for value, choice in closed:

        if value.start < 0 or value.start + value.length > width:
            raise MaskError(
                f"{client_id}: значение [{value.start}, {value.start + value.length}) "
                f"не помещается в строку шириной {width}"
            )

        for index in range(value.start, value.start + value.length):

            if taken[index]:
                raise MaskError(
                    f"{client_id}: позиция {index} выбрана дважды — значения наложились"
                )

            taken[index] = True

            # Испорченный контекст выбранного ключа: вход, не цель.
            if choice is None:
                value_ids[index] = unknown
                continue

            if choice.unknown:
                value_ids[index] = unknown
                continue

            value_ids[index] = mask
            labels[index] = source[index]
            reason[index] = choice.reason

    return {
        "client_id": client_id,
        "value_ids_source": source,
        "value_ids": value_ids,
        "labels": labels,
        "reason": reason,
    }


# Код причины -> её имя (те же строки, что у Choice.reason).
_REASON_NAMES = np.array(REASONS, dtype=object)


def apply_selection(client_id: str, row: dict, selection: Selection, mask: int, unknown: int) -> dict:
    """
    То же, что apply по selection.choices и selection.corrupted, но
    массивами: value_ids и labels — int64, reason — список строк.

    Значение, вышедшее за строку, или наложение — ошибка apply: тогда
    её называет apply обходом, по тем же значениям в том же порядке.
    """

    source = np.asarray(row["value_ids"], dtype=np.int64)
    width = source.size

    found, spoiled = selection.found, selection.spoiled

    # Как в apply: сначала выбранные, затем испорченный контекст.
    starts = np.concatenate([found.start[selection.picked], spoiled.start])
    lengths = np.concatenate([found.length[selection.picked], spoiled.length])

    total = int(lengths.sum())

    index = np.repeat(starts, lengths) + (np.arange(total, dtype=np.int64)
                                          - np.repeat(np.cumsum(lengths) - lengths, lengths))

    if (starts.size and (int(starts.min()) < 0 or int((starts + lengths).max()) > width)) or (
            total and int(np.bincount(index, minlength=width).max()) > 1):
        closed = apply(client_id, row, selection.choices, mask, unknown, selection.corrupted)
        return {"value_ids": np.asarray(closed["value_ids"], dtype=np.int64),
                "labels": np.asarray(closed["labels"], dtype=np.int64), "reason": closed["reason"]}

    item = np.repeat(np.arange(starts.size, dtype=np.int64), lengths)

    # [UNK] — у выбранного с unknown и у всего испорченного контекста.
    spoils = np.concatenate([selection.unknown, np.ones(len(spoiled), dtype=bool)])[item]
    reasons = np.concatenate([selection.reasons, np.zeros(len(spoiled), dtype=np.uint8)])[item]

    value_ids = source.copy()
    labels = np.full(width, IGNORE, dtype=np.int64)
    codes = np.zeros(width, dtype=np.uint8)

    value_ids[index[spoils]] = unknown

    closed = index[~spoils]

    value_ids[closed] = mask
    labels[closed] = source[closed]
    codes[closed] = reasons[~spoils]

    return {"value_ids": value_ids, "labels": labels, "reason": _REASON_NAMES[codes].tolist()}


__all__ = [
    "IGNORE",
    "MaskError",
    "apply",
    "apply_selection",
]
