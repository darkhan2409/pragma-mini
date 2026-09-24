from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ============================================================
# ПЛОСКИЕ ПОСЛЕДОВАТЕЛЬНОСТИ И КОРЗИНЫ ДЛИНЫ
# ============================================================
#
# Последовательности разной длины лежат подряд в одном плоском
# массиве, а их границы — в cu_seqlens:
#
#   длины       [20, 3, 5, 2]
#   cu_seqlens  [0, 20, 23, 28, 30]
#   сегмент i   flat[cu_seqlens[i] : cu_seqlens[i + 1]]
#
# Трансформерам нужен прямоугольник, а ядра внимания без
# заполнителя (varlen) здесь нет. Поэтому сегменты раскладываются
# по КОРЗИНАМ близкой длины, и заполнитель появляется только
# внутри корзины — до её собственной наибольшей длины:
#
#   корзина = ceil(log2(длина)):  1 -> 0, 2 -> 1, 3..4 -> 2,
#                                 5..8 -> 3, 9..16 -> 4, ...
#
# Сегмент длиной 6500 попадает в корзину 13 и не растягивает до
# 6500 сегменты длиной 3. Внутри корзины заполнитель занимает
# меньше половины ширины каждого сегмента.
#
# Это плоское представление с запасным путём через корзины, а не
# внимание без заполнителя: сами ядра по-прежнему считают
# прямоугольник. Заменить корзины на varlen-ядро можно, не меняя
# ни плоских массивов, ни cu_seqlens.
#
# Раскладка строится один раз на micro-batch, на CPU по NumPy-
# длинам: в прямом проходе нет ни одной синхронизации с
# устройством ради формы корзины.
# ============================================================


@dataclass(frozen=True)
class Bucket:
    """
    Одна корзина: несколько сегментов близкой длины.

    index указывает в плоский массив. У хвоста короткого сегмента
    он смотрит на начало самого сегмента: индекс остаётся в
    границах, а значение там всё равно закрыто маской.
    """

    segments: torch.Tensor   # [N] номера сегментов, в исходном порядке
    index: torch.Tensor      # [N, L] позиции в плоском массиве
    mask: torch.Tensor       # [N, L] True у настоящих позиций

    @property
    def size(self) -> int:
        return int(self.segments.numel())


@dataclass(frozen=True)
class VarlenLayout:
    """
    Границы сегментов плоского массива и их корзины.

    bucket_of и row_of говорят, где сегмент лежит в раскладке:
    номер корзины и строка внутри неё.
    """

    cu_seqlens: torch.Tensor   # [S + 1]
    lengths: torch.Tensor      # [S]
    max_seqlen: int
    buckets: tuple
    bucket_of: np.ndarray      # [S]
    row_of: np.ndarray         # [S]

    @property
    def segments(self) -> int:
        return int(self.lengths.numel())

    @staticmethod
    def build(lengths: np.ndarray, device: torch.device, what: str) -> "VarlenLayout":
        """
        Раскладка по длинам сегментов.

        Пустой сегмент — ошибка данных: у события всегда есть
        маркер [EVT], у анкеты — [USR], у истории — слот анкеты.
        """

        lengths = np.asarray(lengths, dtype=np.int64)

        if lengths.size and int(lengths.min()) < 1:
            raise ValueError(f"{what}: пустой сегмент — у каждого должен быть хотя бы маркер")

        cu = np.zeros(lengths.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=cu[1:])

        keys = np.ceil(np.log2(lengths)).astype(np.int64) if lengths.size else lengths

        bucket_of = np.zeros(lengths.size, dtype=np.int64)
        row_of = np.zeros(lengths.size, dtype=np.int64)

        buckets: list[Bucket] = []

        for number, key in enumerate(np.unique(keys)):

            segments = np.nonzero(keys == key)[0]

            width = int(lengths[segments].max())

            steps = np.arange(width, dtype=np.int64)[None, :]

            mask = steps < lengths[segments][:, None]

            starts = cu[segments][:, None]

            index = np.where(mask, starts + steps, starts)

            bucket_of[segments] = number
            row_of[segments] = np.arange(segments.size)

            buckets.append(
                Bucket(
                    segments=torch.as_tensor(segments, device=device),
                    index=torch.as_tensor(index, device=device),
                    mask=torch.as_tensor(mask, device=device),
                )
            )

        return VarlenLayout(
            cu_seqlens=torch.as_tensor(cu, device=device),
            lengths=torch.as_tensor(lengths, device=device),
            max_seqlen=int(lengths.max()) if lengths.size else 0,
            buckets=tuple(buckets),
            bucket_of=bucket_of,
            row_of=row_of,
        )


def assemble(parts: list[torch.Tensor], indices: list[torch.Tensor], count: int,
             empty: torch.Tensor) -> torch.Tensor:
    """
    Результаты корзин обратно в исходный порядок: строка i выхода
    — строка с индексом i.

    Индексы корзин вместе покрывают 0..count-1 ровно по разу.
    index_copy вне места: градиент идёт к частям, а результат
    детерминирован. empty — связанная с графом пустая заготовка
    на случай, когда частей нет.
    """

    if not parts:
        return empty

    values = torch.cat(parts, dim=0)

    return values.new_zeros((count,) + tuple(values.shape[1:])).index_copy(
        0, torch.cat(indices, dim=0), values
    )


__all__ = ["Bucket", "VarlenLayout", "assemble"]
