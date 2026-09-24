from __future__ import annotations

import numpy as np
import pytest
import torch

from src.mlm.varlen import VarlenLayout, assemble


# ============================================================
# ИДЕЯ
# ============================================================
#
# Раскладка — единственное место, где плоский массив без
# заполнителя превращается в прямоугольники. Ошибка здесь тихая:
# потерянный сегмент останется нулём после assemble, а лишний
# заполнитель укажет не на ноль, а на НАЧАЛО своего сегмента, то
# есть на настоящее значение.
#
# Поэтому проверяются точные числа, а не формы: границы корзин
# ceil(log2), плотная нумерация корзин, ширина корзины по её
# собственному максимуму и покрытие сегментов ровно по разу.
# ============================================================


CPU = torch.device("cpu")


def layout(lengths: list[int]) -> VarlenLayout:
    return VarlenLayout.build(np.asarray(lengths, dtype=np.int64), CPU, "сегменты")


# ============================================================
# ГРАНИЦЫ И ТОЧНЫЕ ЧИСЛА
# ============================================================


def test_cumulative_bounds_are_exact():
    """
    cu_seqlens — исключающая сумма длин, и сегмент i лежит между
    соседними значениями.
    """

    built = layout([20, 3, 5, 2])

    assert built.cu_seqlens.tolist() == [0, 20, 23, 28, 30]
    assert built.cu_seqlens_int32.tolist() == [0, 20, 23, 28, 30]
    assert built.cu_seqlens_int32.dtype == torch.int32
    assert built.lengths.tolist() == [20, 3, 5, 2]
    assert built.max_seqlen == 20
    assert built.segments == 4


@pytest.mark.parametrize(
    "length, key",
    [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (16, 4), (17, 5)],
)
def test_bucket_key_follows_ceil_log2(length: int, key: int):
    """
    Корзина это ceil(log2(длина)) — на самих границах степеней
    двойки, где off-by-one был бы незаметен.
    """

    assert int(np.ceil(np.log2(length))) == key


def test_bucket_numbers_are_dense_ranks_not_log2_keys():
    """
    bucket_of — номер корзины в кортеже buckets, а не сам ключ
    ceil(log2). Пропущенные ключи номеров не занимают.
    """

    built = layout([20, 3, 5, 2])

    # ключи 5, 2, 3, 1 -> отсортированные уникальные 1, 2, 3, 5
    assert built.bucket_of.tolist() == [3, 1, 2, 0]
    assert len(built.buckets) == 4

    for number, bucket in enumerate(built.buckets):
        assert built.bucket_of[bucket.segments.numpy()].tolist() == [number] * bucket.size


def test_row_follows_ascending_segment_number():
    """
    row_of — место внутри корзины, и порядок в ней тот же, что в
    исходном массиве.
    """

    built = layout([1, 2, 1, 2, 1])

    assert built.bucket_of.tolist() == [0, 1, 0, 1, 0]
    assert built.row_of.tolist() == [0, 0, 1, 1, 2]

    for bucket in built.buckets:
        assert bucket.segments.tolist() == sorted(bucket.segments.tolist())


# ============================================================
# ШИРИНА КОРЗИНЫ
# ============================================================


def test_one_huge_segment_does_not_stretch_the_short_ones():
    """
    Сегмент в 6500 лежит в своей корзине, и короткие остаются при
    своей ширине. Это и есть причина, по которой корзины вообще
    заведены.
    """

    built = layout([20, 30, 31, 6500])

    assert built.bucket_of.tolist() == [0, 0, 0, 1]

    short, huge = built.buckets

    assert short.index.shape == (3, 31)
    assert huge.index.shape == (1, 6500)

    padded = sum(int(bucket.index.numel()) for bucket in built.buckets)

    # Один прямоугольник на всех стоил бы 4 * 6500 = 26000.
    assert padded == 3 * 31 + 6500


def test_bucket_width_is_its_own_maximum():

    built = layout([5, 6, 7, 8, 9])

    widths = {int(bucket.index.shape[1]) for bucket in built.buckets}

    assert widths == {8, 9}


# ============================================================
# ПОКРЫТИЕ И ЗАПОЛНИТЕЛЬ
# ============================================================


def test_every_segment_appears_exactly_once():
    """
    Корзины вместе покрывают все сегменты ровно по разу. Без этой
    проверки потерянный сегмент остался бы нулём после assemble и
    ошибки не вызвал.
    """

    lengths = [1, 2, 3, 4, 5, 8, 9, 16, 17]

    built = layout(lengths)

    seen = np.concatenate([bucket.segments.numpy() for bucket in built.buckets])

    assert sorted(seen.tolist()) == list(range(len(lengths)))


def test_real_positions_point_at_their_own_segment():
    """
    index настоящих позиций — это в точности отрезок плоского
    массива, принадлежащий сегменту.
    """

    lengths = [20, 3, 5, 2]

    built = layout(lengths)

    starts = built.cu_seqlens.tolist()

    for bucket in built.buckets:
        for row, segment in enumerate(bucket.segments.tolist()):

            length = lengths[segment]
            index = bucket.index[row][bucket.mask[row]].tolist()

            assert index == list(range(starts[segment], starts[segment] + length))


def test_padding_points_at_the_segment_start_and_is_masked():
    """
    Хвост короткого сегмента указывает на его же начало, а не на
    ноль: индекс остаётся в границах, а закрывает его маска.

    Проверяется именно это, потому что при потере маски туда
    попало бы настоящее значение, а не ноль, и ошибка была бы
    тихой.
    """

    # Длины 3 и 4 дают один ключ ceil(log2) = 2, значит одну
    # корзину шириной 4: у первого сегмента один хвостовой слот.
    built = layout([3, 4])

    assert len(built.buckets) == 1

    bucket = built.buckets[0]

    assert bucket.index.shape == (2, 4)
    assert bucket.mask.tolist() == [[True, True, True, False], [True] * 4]

    # Хвост первого сегмента смотрит на его собственное начало,
    # то есть на позицию 0, а не на ноль-значение.
    assert bucket.index.tolist() == [[0, 1, 2, 0], [3, 4, 5, 6]]


def test_length_one_segment_survives():
    """
    Сегмент из одного токена — обычное дело: событие бывает из
    одного [EVT].
    """

    built = layout([1, 1, 1])

    assert built.max_seqlen == 1
    assert len(built.buckets) == 1
    assert built.buckets[0].mask.tolist() == [[True], [True], [True]]
    assert built.buckets[0].index.tolist() == [[0], [1], [2]]


# ============================================================
# КРАЯ
# ============================================================


def test_empty_segment_is_refused():
    """
    Пустого сегмента не бывает: у события есть [EVT], у анкеты
    [USR], у истории слот анкеты.
    """

    with pytest.raises(ValueError, match="пустой сегмент"):
        VarlenLayout.build(np.asarray([3, 0, 2], dtype=np.int64), CPU, "события")

    with pytest.raises(ValueError, match="события: пустой сегмент"):
        VarlenLayout.build(np.asarray([0], dtype=np.int64), CPU, "события")


def test_no_segments_at_all_is_not_an_error():

    built = VarlenLayout.build(np.asarray([], dtype=np.int64), CPU, "события")

    assert built.cu_seqlens.tolist() == [0]
    assert built.max_seqlen == 0
    assert built.buckets == ()
    assert built.segments == 0


# ============================================================
# СБОРКА ОБРАТНО
# ============================================================


def test_assemble_restores_the_original_order():
    """
    Результаты корзин возвращаются на свои места: строка i выхода
    — это сегмент i, а не порядок обхода корзин.
    """

    lengths = [20, 3, 5, 2, 31, 1]

    built = layout(lengths)

    # Пометка сегмента — его номер, чтобы перестановка была видна.
    marks = torch.arange(len(lengths), dtype=torch.float32)[:, None]

    parts = [marks[bucket.segments] for bucket in built.buckets]
    indices = [bucket.segments for bucket in built.buckets]

    out = assemble(parts, indices, len(lengths), marks[:0])

    assert torch.equal(out, marks)


def test_assemble_without_parts_returns_the_graph_connected_blank():
    """
    Пустая заготовка возвращается как есть: она связана с графом,
    и backward на batch без целей не оборвётся.
    """

    blank = torch.zeros(0, 4, requires_grad=True)

    out = assemble([], [], 5, blank)

    assert out is blank


def test_assemble_leaves_an_uncovered_row_at_zero():
    """
    Непокрытая строка молча остаётся нулём — именно поэтому
    покрытие проверяется отдельным тестом, а не считается само
    собой разумеющимся.
    """

    values = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    out = assemble([values[:2]], [torch.tensor([0, 2])], 4, values[:0])

    assert out.tolist() == [[0.0, 1.0], [0.0, 0.0], [2.0, 3.0], [0.0, 0.0]]
