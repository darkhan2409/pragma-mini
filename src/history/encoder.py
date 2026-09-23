from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# ЭНКОДЕР ИСТОРИИ
# ============================================================
#
# История клиента это последовательность [z_a, события...], где
# z_a — вектор анкеты из этапа 11, положенный прямо в слот [USR].
# Второй маркер не заводится и заново не эмбеддится: вектор уже
# посчитан.
#
# События идут от старых к новым, а их временные позиции —
# log-секунды до последнего события — наоборот убывают к нулю.
# [USR] якорится на нуле, то есть в тот же момент, что и самое
# свежее событие. Для двунаправленного внимания порядок в массиве
# сам по себе ничего не значит: значение имеет только позиция.
#
# ВРЕМЯ ВХОДИТ ЧЕРЕЗ ПОВОРОТ, а не слагаемым. TimeRoPE вращает Q
# и K на угол, пропорциональный непрерывной позиции, поэтому
# скалярное произведение зависит от РАЗНОСТИ позиций — то есть от
# того, сколько времени прошло между двумя событиями, а не от их
# порядкового расстояния. Для ленты с неравными промежутками это
# и нужно: секунда и месяц между соседями обязаны отличаться.
#
# Своё внимание, а не nn.TransformerEncoderLayer, по двум
# причинам. Первая: RoPE обязан примениться к Q и K ВНУТРИ
# внимания, а готовый слой этого не позволяет. Вторая измерена:
# путь MATH материализует матрицу L x L целиком, и на истории в
# 23 320 событий это 8.7 ГиБ и отказ по памяти, тогда как
# F.scaled_dot_product_attention выбирает блочное ядро и тратит
# 12 МиБ на то же самое точное вычисление.
#
# Маски внимания здесь нет, и она не нужна: клиенты считаются по
# одному, заполнителя во входе не существует.
# ============================================================


class TimeRoPE(nn.Module):
    """
    Поворот по непрерывной позиции.

    В отличие от обычной RoPE позиция здесь вещественная — это
    само временное расстояние в сжатых логарифмом секундах, а не
    порядковый номер.
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10000.0):

        super().__init__()

        if head_dim % 2:
            raise ValueError(f"TimeRoPE требует чётного head_dim, получено {head_dim}")

        self.head_dim = int(head_dim)
        self.base = float(base)

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

        # Буфер непостоянный: в файл весов лестница частот не
        # пишется, она однозначно восстанавливается из rope_base.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def angles(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (cos, sin) формы [L, head_dim] по позициям [L].

        Углы считаются в float32 независимо от типа активаций:
        разрешение log-секунд терять нельзя.
        """

        freqs = position[:, None].float() * self.inv_freq[None, :]

        # Частоты ДУБЛИРУЮТСЯ, а не чередуются: пара это i и
        # i + head_dim/2. Это схема LLaMA, и поворот ниже устроен
        # под неё.
        angles = torch.cat([freqs, freqs], dim=-1)

        return angles.cos(), angles.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:

        left, right = x.chunk(2, dim=-1)

        return torch.cat([-right, left], dim=-1)

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Поворот [..., L, head_dim] при cos/sin формы [L, head_dim].

        Оси головы и батча добираются бродкастом по хвостовым
        размерностям, поэтому отдельного None здесь нет.
        """

        return x * cos.to(x.dtype) + self._rotate_half(x) * sin.to(x.dtype)


class Block(nn.Module):
    """
    Один блок: внимание с поворотом и FFN, оба через pre-norm.
    """

    def __init__(self, dim: int, heads: int, feedforward: int, dropout: float):

        super().__init__()

        self.heads = int(heads)
        self.head_dim = dim // heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Одна слитая проекция на Q, K и V, без bias — как в
        # эталоне.
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, dim),
        )

        self.drop = nn.Dropout(dropout)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: TimeRoPE,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:

        batch, length, dim = x.shape

        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim)

        # [B, L, H, hd] -> [B, H, L, hd]
        query = qkv[:, :, 0].transpose(1, 2)
        key = qkv[:, :, 1].transpose(1, 2)
        value = qkv[:, :, 2].transpose(1, 2)

        # Поворачиваются только Q и K. V остаётся как есть: иначе
        # повернулось бы само содержимое, а не то, с чем его
        # сравнивают.
        query = rope.rotate(query, cos, sin)
        key = rope.rotate(key, cos, sin)

        # Масштаб 1/sqrt(head_dim) внутри SDPA, руками не пишется.
        attention = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0
        )

        x = x + self.drop(self.out(attention.transpose(1, 2).reshape(batch, length, dim)))

        return x + self.drop(self.ffn(self.norm2(x)))


class HistoryEncoder(nn.Module):
    """
    Стек блоков по истории одного клиента.
    """

    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float,
        rope_base: float,
        seed: int,
    ):

        super().__init__()

        self.dim = int(dim)

        # Веса разыгрываются от известного состояния и не трогают
        # чужое. Тот же приём есть в этапах 10 и 11; повтор
        # намеренный — этапы держатся отдельно. Когда энкодеры
        # будут сводиться в один класс, его стоит вынести.
        with _seeded(seed):

            # Слои поштучно: nn.TransformerEncoder копирует один
            # слой, и все блоки стартовали бы с одинаковых весов.
            self.layers = nn.ModuleList(
                [Block(self.dim, heads, feedforward, dropout) for _ in range(layers)]
            )

            self.norm = nn.LayerNorm(self.dim)

            self.rope = TimeRoPE(self.dim // heads, base=rope_base)

    def forward(self, sequence: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        [1, L, d] и [L] -> [1, L, d].

        Углы считаются один раз на всю историю и переиспользуются
        всеми блоками: лестница частот у них общая.
        """

        cos, sin = self.rope.angles(positions)

        x = sequence

        for layer in self.layers:
            x = layer(x, self.rope, cos, sin)

        return self.norm(x)


@contextmanager
def _seeded(seed: int):
    """
    Известное состояние генератора на время сборки весов.
    """

    state = torch.get_rng_state()

    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(state)


__all__ = [
    "Block",
    "HistoryEncoder",
    "TimeRoPE",
]
