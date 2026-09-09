from __future__ import annotations

import torch
import torch.nn as nn

from .batching import BatchError
from .config import ModelConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна таблица E на всё пространство ID tokenizer: ключи и
# значения живут в одном пространстве, поэтому и embedding у
# них общий.
#
#   обычное поле:   x = E(key_id) + E(value_id) + P(position)
#   special-позиция: x = E(special_id) + P(position)
#
# Special-позиция узнаётся по key_id == value_id. Это работает,
# потому что диапазоны ключей [6, first_value) и значений
# [first_value, size) не пересекаются: равенство возможно только
# у [EVT], [USR], неизвестного ключа [UNK]/[UNK] и padding.
#
# А вот [MISSING] и [UNK] на месте ЗНАЧЕНИЯ обычные видимые
# токены: ключ у них настоящий, поэтому складываются оба.
# ============================================================


class SharedEmbeddings(nn.Module):
    """
    Общие таблицы токенов и позиций для обоих энкодеров.
    """

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.config = config

        self.token = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position = nn.Embedding(config.max_position_embeddings, config.d_model)

    # --------------------------------------------------------

    def _check(self, key_ids: torch.Tensor, value_ids: torch.Tensor, positions: torch.Tensor) -> None:
        """
        Дублирует проверку адаптера, но настоящим исключением:
        assert исчезает под python -O, и тогда таблица позиций
        читалась бы за границей невнятной ошибкой устройства.
        """

        limit = self.config.max_position_embeddings

        if int(positions.max()) >= limit or int(positions.min()) < 0:
            raise BatchError(
                f"позиция вне таблицы размера {limit}: диапазон "
                f"{int(positions.min())}..{int(positions.max())}"
            )

        size = self.config.vocab_size

        for name, tensor in (("key_ids", key_ids), ("value_ids", value_ids)):
            if int(tensor.max()) >= size or int(tensor.min()) < 0:
                raise BatchError(
                    f"{name} вне словаря [0, {size}): диапазон {int(tensor.min())}..{int(tensor.max())}"
                )

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        self._check(key_ids, value_ids, positions)

        # Special на позиции учитывается один раз.
        paired = (key_ids != value_ids).unsqueeze(-1)

        x = self.token(key_ids) + self.token(value_ids) * paired + self.position(positions)

        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
