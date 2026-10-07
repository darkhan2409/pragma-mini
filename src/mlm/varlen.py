from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
# Запасному пути (SDPA) нужен прямоугольник: ядро без заполнителя
# там не участвует. Поэтому на нём сегменты раскладываются по
# КОРЗИНАМ близкой длины, и заполнитель появляется только внутри
# корзины — до её собственной наибольшей длины:
#
#   корзина = ceil(log2(длина)):  1 -> 0, 2 -> 1, 3..4 -> 2,
#                                 5..8 -> 3, 9..16 -> 4, ...
#
# Сегмент длиной 6500 попадает в корзину 13 и не растягивает до
# 6500 сегменты длиной 3. Внутри корзины заполнитель занимает
# меньше половины ширины каждого сегмента.
#
# Корзины и varlen-ядро читают одни и те же плоские массивы и
# cu_seqlens: путь меняется, представление — нет.
#
# Раскладка строится один раз на micro-batch, на CPU по NumPy-
# длинам: в прямом проходе нет ни одной синхронизации с
# устройством ради формы корзины.
#
# Основной путь на CUDA — varlen-ядро flash_attn_varlen_func: оно
# берёт плоские Q/K/V [N, головы, размер головы] и cu_seqlens и
# считает внимание каждого сегмента без единой позиции
# заполнителя. Корзины остаются запасным путём: CPU, fp32, нет
# библиотеки flash-attn.
#
# Сегменты идут в ядро ГРУППАМИ подряд, а не все одним вызовом.
# Backward flash-attn (mha_varlen_bwd) заводит буфер dq_accum fp32
# на total_q + 128·(число сегментов) строк: 128 строк запаса на
# сегмент, сколько бы в нём ни было позиций. У энкодера события
# сегмент — событие, их до 12 000 на клиента по 8 токенов в
# среднем, и один вызов просил бы до 780 МиБ, из которых 96% —
# запас. Группа держит не больше FLASH_ROWS строк вместе с
# запасом. Результат тот же бит в бит: ядро считает каждый
# сегмент само по себе, а генератор dropout сдвигается на группу
# ровно так, как на её часть одного большого вызова.
#
# Varlen-слои ниже не заводят своих весов: они считают формулу
# уже существующих модулей на их же параметрах —
# nn.TransformerEncoderLayer энкодера события и Block с TimeRoPE,
# общий для энкодеров анкеты и истории. Поэтому начальные веса
# backbone и чекпойнт грузятся без изменений.
# ============================================================


BACKENDS = ("auto", "flash", "sdpa")

# Строк запаса на сегмент в буферах backward flash-attn.
FLASH_PAD = 128

# Предел одного вызова ядра: строки сегментов плюс их запас. При 4
# головах по 32 буфер dq_accum не больше 64 МиБ.
FLASH_ROWS = 1 << 17


class BackendError(RuntimeError):
    """
    Выбранный бэкенд внимания здесь недоступен.
    """


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
class Group:
    """
    Сегменты подряд, которые идут в flash-attn одним вызовом.
    """

    rows: int                  # строк плоского массива у группы
    cu_seqlens: torch.Tensor   # [S_g + 1], int32, от начала группы
    max_seqlen: int


@dataclass(frozen=True)
class VarlenLayout:
    """
    Границы сегментов плоского массива, их корзины и группы.

    bucket_of и row_of говорят, где сегмент лежит в раскладке:
    номер корзины и строка внутри неё.

    Корзины нужны только проходу без flash-attn (корзины SDPA), а
    flash-attn берёт группы. Поэтому матрицы корзин (index, mask) и
    их перенос на устройство строятся при первом обращении к buckets:
    на пути flash их нет вовсе. Сами корзины от этого не меняются.
    """

    cu_seqlens: torch.Tensor   # [S + 1]
    lengths: torch.Tensor      # [S]
    groups: tuple              # вызовы flash-attn, по порядку сегментов

    # Длины и границы сегментов на CPU и устройство — для корзин.
    sizes: np.ndarray = field(repr=False, compare=False)
    edges: np.ndarray = field(repr=False, compare=False)
    device: torch.device = field(repr=False, compare=False)

    @property
    def segments(self) -> int:
        return int(self.lengths.numel())

    @cached_property
    def _keys(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Корзина сегмента — ceil(log2(длина)); сегменты каждой корзины
        по возрастанию корзины.
        """

        keys = np.ceil(np.log2(self.sizes)).astype(np.int64) if self.sizes.size else self.sizes

        return keys, [np.nonzero(keys == key)[0] for key in np.unique(keys)]

    @cached_property
    def bucket_of(self) -> np.ndarray:  # [S]

        bucket_of = np.zeros(self.sizes.size, dtype=np.int64)

        for number, segments in enumerate(self._keys[1]):
            bucket_of[segments] = number

        return bucket_of

    @cached_property
    def row_of(self) -> np.ndarray:  # [S]

        row_of = np.zeros(self.sizes.size, dtype=np.int64)

        for segments in self._keys[1]:
            row_of[segments] = np.arange(segments.size)

        return row_of

    @cached_property
    def buckets(self) -> tuple:

        buckets: list[Bucket] = []

        for segments in self._keys[1]:

            width = int(self.sizes[segments].max())

            steps = np.arange(width, dtype=np.int64)[None, :]

            mask = steps < self.sizes[segments][:, None]

            starts = self.edges[segments][:, None]

            index = np.where(mask, starts + steps, starts)

            buckets.append(
                Bucket(
                    segments=torch.as_tensor(segments, device=self.device),
                    index=torch.as_tensor(index, device=self.device),
                    mask=torch.as_tensor(mask, device=self.device),
                )
            )

        return tuple(buckets)

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

        groups = tuple(
            Group(
                rows=int(cu[last] - cu[first]),
                cu_seqlens=torch.as_tensor((cu[first:last + 1] - cu[first]).astype(np.int32), device=device),
                max_seqlen=int(lengths[first:last].max(initial=0)),
            )
            for first, last in _group_edges(lengths)
        )

        return VarlenLayout(
            cu_seqlens=torch.as_tensor(cu, device=device),
            lengths=torch.as_tensor(lengths, device=device),
            groups=groups,
            sizes=lengths,
            edges=cu,
            device=device,
        )


def _group_edges(lengths: np.ndarray) -> list[tuple[int, int]]:
    """
    Сегменты [first, last) каждой группы: подряд, пока строки с
    запасом FLASH_PAD на сегмент помещаются в FLASH_ROWS.

    Сегмент больше предела идёт отдельной группой. Без сегментов —
    одна пустая группа: вызов остаётся прежним.
    """

    if not lengths.size:
        return [(0, 0)]

    price = np.cumsum(lengths + FLASH_PAD)

    edges = [0]

    while edges[-1] < lengths.size:

        spent = int(price[edges[-1] - 1]) if edges[-1] else 0

        fits = int(np.searchsorted(price, spent + FLASH_ROWS, side="right"))

        edges.append(max(fits, edges[-1] + 1))

    return list(zip(edges, edges[1:]))


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


def flash_available() -> bool:
    """
    Есть ли библиотека flash-attn. Сама библиотека необязательна.
    """

    # Ловится любая ошибка, а не только ImportError: CUDA-расширение
    # несовместимой сборки падает при загрузке с OSError или
    # RuntimeError. Сломанная библиотека — то же, что её нет: auto
    # уходит на корзины SDPA, а явный flash получает BackendError.
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except Exception:
        return False

    return True


def resolve_backend(name: str, device: torch.device) -> str:
    """
    Бэкенд внимания: "flash" или "sdpa".

    auto выбирает flash, когда есть CUDA и библиотека, иначе молча
    берёт корзины. Явный flash без них — ошибка: тихая подмена
    спрятала бы, что обучение идёт не тем путём, который просили.
    Тип активаций проверяется позже, на каждом проходе.
    """

    if name not in BACKENDS:
        raise BackendError(f"attention_backend обязан быть одним из {list(BACKENDS)}, получено {name!r}")

    if name == "sdpa":
        return "sdpa"

    available = device.type == "cuda" and flash_available()

    if name == "flash" and not available:
        raise BackendError(
            "attention_backend=flash требует CUDA и библиотеку flash-attn "
            "(pip install -e .[flash]); здесь "
            + ("нет CUDA" if device.type != "cuda" else "flash-attn не установлена")
        )

    return "flash" if available else "sdpa"


def autocast(device: torch.device):
    """
    Смешанная точность прохода: bf16 на CUDA, которая его умеет,
    иначе прежний fp32.

    Веса не переводятся: autocast считает матричные операции в
    bf16, а параметры, градиенты и состояние AdamW остаются fp32.
    Для bf16 масштабирование потерь не нужно.
    """

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)

    return nullcontext()


def attend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: VarlenLayout,
    dropout: float,
) -> torch.Tensor:
    """
    Внимание внутри сегментов: [N, головы, размер] -> то же.

    Сегмент видит только себя: границы задаёт cu_seqlens группы.
    Масштаб 1/sqrt(размер головы) — по умолчанию, как у SDPA и
    MultiheadAttention.

    Группы вызываются строго по порядку и без других обращений к
    генератору CUDA между ними: только так маски dropout совпадают
    с одним большим вызовом. split, а не срезы: backward собирает
    градиент групп одним cat, а не нулевым тензором на группу.
    """

    from flash_attn import flash_attn_varlen_func

    def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, group: Group) -> torch.Tensor:
        return flash_attn_varlen_func(
            q, k, v,
            group.cu_seqlens, group.cu_seqlens,
            group.max_seqlen, group.max_seqlen,
            dropout_p=dropout,
            causal=False,
        )

    if len(layout.groups) == 1:
        return kernel(query, key, value, layout.groups[0])

    rows = [group.rows for group in layout.groups]

    return torch.cat([
        kernel(q, k, v, group)
        for q, k, v, group in zip(query.split(rows), key.split(rows), value.split(rows), layout.groups)
    ])


def encoder_layer_varlen(
    layer: nn.TransformerEncoderLayer, x: torch.Tensor, layout: VarlenLayout
) -> torch.Tensor:
    """
    Слой nn.TransformerEncoderLayer (norm_first) на плоских
    последовательностях.

    Формула та же, что у слоя: x + dropout1(out_proj(внимание(
    norm1(x)))), затем x + dropout2(linear2(dropout(activation(
    linear1(norm2(x)))))). Q, K и V режутся из in_proj так же,
    как внутри MultiheadAttention: первые d строк — Q, внутри них
    головы подряд.
    """

    attention = layer.self_attn

    count, dim = x.shape

    heads = attention.num_heads

    qkv = F.linear(
        layer.norm1(x), attention.in_proj_weight, attention.in_proj_bias
    ).view(count, 3, heads, dim // heads)

    mixed = attend(
        qkv[:, 0], qkv[:, 1], qkv[:, 2], layout,
        attention.dropout if layer.training else 0.0,
    )

    x = x + layer.dropout1(attention.out_proj(mixed.reshape(count, dim)))

    return x + layer.dropout2(
        layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
    )


def history_block_varlen(
    block: nn.Module,
    rope: nn.Module,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layout: VarlenLayout,
) -> torch.Tensor:
    """
    Блок с TimeRoPE (энкодеры анкеты и истории) на плоских
    последовательностях.

    Формула та же, что у Block.forward: поворачиваются только Q и
    K тем же TimeRoPE.rotate. cos и sin [N, размер] считаются в
    fp32 один раз на все блоки и приводятся к типу активаций
    только внутри поворота.
    """

    count, dim = x.shape

    qkv = block.qkv(block.norm1(x)).view(count, 3, block.heads, block.head_dim)

    # Ось головы добирается бродкастом: у всех голов позиция одна.
    angle_cos, angle_sin = cos[:, None], sin[:, None]

    query = rope.rotate(qkv[:, 0], angle_cos, angle_sin)
    key = rope.rotate(qkv[:, 1], angle_cos, angle_sin)

    mixed = attend(
        query, key, qkv[:, 2], layout, block.dropout if block.training else 0.0
    )

    x = x + block.drop(block.out(mixed.reshape(count, dim)))

    return x + block.drop(block.ffn(block.norm2(x)))


__all__ = [
    "BACKENDS",
    "FLASH_PAD",
    "FLASH_ROWS",
    "BackendError",
    "Bucket",
    "Group",
    "VarlenLayout",
    "assemble",
    "attend",
    "autocast",
    "encoder_layer_varlen",
    "flash_available",
    "history_block_varlen",
    "resolve_backend",
]
