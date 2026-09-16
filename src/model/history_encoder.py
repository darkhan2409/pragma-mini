from __future__ import annotations

import torch
import torch.nn as nn

from .batching import BatchError
from .config import ModelConfig
from .rotary import rotary_layers, rotary_tables


# ============================================================
# ИДЕЯ
# ============================================================
#
# История клиента это последовательность
#
#     [профиль, событие_1, ..., событие_N]
#
# где каждый элемент уже вектор d_model. Attention
# двусторонний, causal mask нет: цель не предсказание
# следующего события, а представление среза целиком. Разные
# примеры лежат разными строками и друг друга не видят.
#
# Слои здесь свои (rotary.py): у них собственный forward поверх
# scaled_dot_product_attention. Это не только про время, но и
# про память: fast path nn.TransformerEncoderLayer материализует
# полную матрицу внимания, и на истории в 3637 событий это
# 3.2 ГБ вместо 47 МБ. Время приходит отдельным аргументом
# coords и поворачивает q и k, а не прибавляется к входному
# вектору.
# ============================================================


class HistoryEncoder(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.config = config

        # Свой forward, поэтому и хук, и fast path не при чём.
        self.layers = rotary_layers(config, config.n_history_layers)

        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    # --------------------------------------------------------

    def allowed_keys(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Маска padding в соглашении SDPA: True означает «ключ доступен».

        Полярность обратна соглашению nn.Transformer, где True
        это запрет, и переворачивается она ровно здесь. Два места
        с инверсией рано или поздно разошлись бы, и половина
        запретов потерялась бы молча.

        Хватает [B, 1, 1, L]: запрет один и тот же у всех голов и
        всех запросов, и SDPA разошлёт его сам, не материализуя
        [B·heads, L, L].
        """

        return ~padding_mask.unsqueeze(1).unsqueeze(1)

    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise BatchError(f"HistoryEncoder: ожидался [B, L, d], получено {tuple(x.shape)}")

        if padding_mask.shape != x.shape[:2]:
            raise BatchError(
                f"HistoryEncoder: маска {tuple(padding_mask.shape)} не совпадает с {tuple(x.shape[:2])}"
            )

        if bool(padding_mask.all(dim=1).any()):
            raise BatchError("HistoryEncoder: есть пример целиком из padding, кодировать нечего")

        if coords is None:
            raise BatchError(
                "нет координат времени: без них внимание не знало бы о времени вовсе"
            )

        if coords.shape != x.shape[:2]:
            raise BatchError(
                f"HistoryEncoder: координаты {tuple(coords.shape)} не совпадают "
                f"с {tuple(x.shape[:2])}"
            )

        cos, sin = rotary_tables(coords, self.config.head_dim, self.config.rope_base)

        allowed = self.allowed_keys(padding_mask)

        for layer in self.layers:
            x = layer(x, cos, sin, allowed)

        hidden = self.final_norm(x)

        # LayerNorm нулевой строки даёт bias, а не ноль.
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
