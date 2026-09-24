from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

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
#
# Основной путь на CUDA — varlen-ядро flash_attn_varlen_func: оно
# берёт плоские Q/K/V [N, головы, размер головы] и cu_seqlens и
# считает внимание каждого сегмента без единой позиции
# заполнителя. Корзины остаются запасным путём: CPU, fp32, нет
# библиотеки flash-attn.
#
# Varlen-слои ниже не заводят своих весов: они считают формулу
# уже существующих модулей (nn.TransformerEncoderLayer и Block
# энкодера истории) на их же параметрах. Поэтому веса этапов
# 10-12 грузятся без изменений.
# ============================================================


BACKENDS = ("auto", "flash", "sdpa")


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
class VarlenLayout:
    """
    Границы сегментов плоского массива и их корзины.

    bucket_of и row_of говорят, где сегмент лежит в раскладке:
    номер корзины и строка внутри неё.
    """

    cu_seqlens: torch.Tensor   # [S + 1]
    cu_seqlens_int32: torch.Tensor  # [S + 1], тот же, для flash-attn
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
            cu_seqlens_int32=torch.as_tensor(cu.astype(np.int32), device=device),
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


def flash_available() -> bool:
    """
    Есть ли библиотека flash-attn. Сама библиотека необязательна.
    """

    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError:
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

    Сегмент видит только себя: границы задаёт cu_seqlens.
    Масштаб 1/sqrt(размер головы) — по умолчанию, как у SDPA и
    MultiheadAttention.
    """

    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        query, key, value,
        layout.cu_seqlens_int32, layout.cu_seqlens_int32,
        layout.max_seqlen, layout.max_seqlen,
        dropout_p=dropout,
        causal=False,
    )


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
    Блок энкодера истории на плоских последовательностях.

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
    "BackendError",
    "Bucket",
    "VarlenLayout",
    "assemble",
    "attend",
    "autocast",
    "encoder_layer_varlen",
    "flash_available",
    "history_block_varlen",
    "resolve_backend",
]
