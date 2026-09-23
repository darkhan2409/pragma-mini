from __future__ import annotations

from .choose import Choice, NONE


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
          unknown: int) -> dict:
    """
    Четыре массива по ширине строки батча.
    """

    source = list(row["value_ids"])
    width = len(source)

    value_ids = list(source)
    labels = [IGNORE] * width
    reason = [NONE] * width

    # Занятость отслеживается отдельно: у [UNK] ни labels, ни
    # причина не пишутся, и по ним наложение было бы не видно.
    taken = [False] * width

    for choice in choices:

        value = choice.value

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


__all__ = [
    "IGNORE",
    "MaskError",
    "apply",
]
