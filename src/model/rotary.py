from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batching import BatchError
from .config import ROPE_BASE, ModelConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Время истории живёт во внимании, а не во входном векторе.
#
# Каждой позиции соответствует координата t, и q с k
# поворачиваются на угол, пропорциональный t. Скалярное
# произведение повёрнутых векторов зависит только от РАЗНОСТИ
# координат, поэтому внимание видит «сколько времени между
# этими двумя элементами», а сам вектор события остаётся
# нетронутым.
#
# Это и есть ответ на дефект additive: там временное слагаемое
# имело норму 26..143 против нормы вектора события 9, и
# содержание было малой добавкой к направлению «возраст».
# Поворот норму не меняет вовсе.
#
# Поворачиваются ТОЛЬКО q и k. v несёт содержание, и его
# вращение исказило бы то, что внимание переносит.
#
# Углы, таблицы и сам поворот считаются во float32. Под autocast
# bf16 у угла порядка 60 радиан шаг был бы 0.25 радиана, то есть
# сутки не отличались бы от недели.
#
# Слой наследуется от nn.TransformerEncoderLayer ради его
# параметров, их инициализации и имён в state_dict, но forward
# переопределён целиком и родительский НИКОГДА не вызывается:
# fused fast path не должен получить шанс посчитать внимание в
# обход поворота.
# ============================================================


def rotary_tables(
    coords: torch.Tensor, head_dim: int, base: float = ROPE_BASE
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Косинусы и синусы углов поворота: [B, 1, L, head_dim / 2].

    Размерность головы вставляется единицей: у всех голов угол
    один и тот же, различаются только сами векторы.
    """

    if coords.ndim != 2:
        raise BatchError(f"ожидались координаты [B, L], получено {tuple(coords.shape)}")

    if head_dim % 2 != 0:
        raise ValueError(f"head_dim={head_dim} нечётный: поворачиваются пары измерений")

    steps = torch.arange(0, head_dim, 2, device=coords.device, dtype=torch.float32)

    frequency = torch.exp(-math.log(base) * steps / head_dim)

    angles = coords.float().unsqueeze(-1) * frequency

    return torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Поворот пар измерений головы: [B, H, L, head_dim].

    Пара это (i, i + head_dim/2), а не (2i, 2i+1): вариант
    rotate-half, тот же, что в реализациях RoPE. Выбор пар на
    результат не влияет, но обязан совпадать между q и k.
    """

    if x.ndim != 4:
        raise BatchError(f"ожидалось [B, H, L, head_dim], получено {tuple(x.shape)}")

    half = x.shape[-1] // 2

    values = x.float()

    left, right = values[..., :half], values[..., half:]

    rotated = torch.cat([left * cos - right * sin, left * sin + right * cos], dim=-1)

    return rotated.to(x.dtype)


class RotaryHistoryLayer(nn.TransformerEncoderLayer):
    """
    Слой History Encoder с поворотом q и k по времени.

    Параметры, их инициализация и имена в state_dict те же, что у
    обычного слоя: наследование нужно именно ради этого. При одном
    seed стек rope-слоёв стартует с тех же весов, что стек
    обычных слоёв: поворот не трогает ни число параметров, ни
    порядок их розыгрыша.
    """

    def __init__(self, config: ModelConfig):

        super().__init__(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pre-norm блок: внимание с поворотом, затем FFN.

        allowed это bool-маска в соглашении SDPA: True означает
        «ключ доступен». Полярность переворачивается один раз в
        HistoryEncoder.allowed_keys, сюда приходит уже готовая.

        Сигнатура намеренно не совпадает с родительской: вызов с
        забытым поворотом должен быть ошибкой, а не тихо считать
        внимание без времени.
        """

        batch, length, _ = x.shape

        normed = self.norm1(x)

        qkv = F.linear(normed, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.self_attn.dropout if self.training else 0.0,
        )

        attended = attended.transpose(1, 2).reshape(batch, length, -1)

        x = x + self.dropout1(self.self_attn.out_proj(attended))

        # FFN достаётся от родителя как есть: он ничем не
        # отличается от обычного слоя.
        return x + self._ff_block(self.norm2(x))


def rotary_layers(config: ModelConfig, n_layers: int) -> nn.ModuleList:
    """
    Стек независимо инициализированных rope-слоёв.
    """

    return nn.ModuleList([RotaryHistoryLayer(config) for _ in range(n_layers)])


__all__ = ["RotaryHistoryLayer", "apply_rotary", "rotary_layers", "rotary_tables"]
