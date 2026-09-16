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


# Соль потока таблицы токенов. Отдельный поток нужен, чтобы
# размер словаря не двигал инициализацию остальных параметров:
# Сравнение режимов словаря обязано менять словарь, а не
# стартовые веса.
VOCAB_SEED_SALT = 0x5EED_0CAB


def draw_token_weight(config: ModelConfig, seed: int) -> torch.Tensor:
    """
    Веса таблицы токенов из собственного генератора.

    Глобальный поток не трогается вовсе: Generator у нас свой, и
    ни одно число из torch.manual_seed(seed) здесь не тратится.
    Значит Transformer, History Encoder и MLM head стартуют
    одинаково при любом vocab_size.

    Инициализация та же, что у nn.Embedding: N(0, 1) и нулевая
    строка padding_idx.
    """

    generator = torch.Generator().manual_seed(int(seed) ^ VOCAB_SEED_SALT)

    weight = torch.empty(config.vocab_size, config.d_model)

    weight.normal_(mean=0.0, std=1.0, generator=generator)

    with torch.no_grad():
        weight[config.pad_id].fill_(0.0)

    return weight


class SharedEmbeddings(nn.Module):
    """
    Общие таблицы токенов и позиций для обоих энкодеров.
    """

    def __init__(self, config: ModelConfig, token_weight: torch.Tensor | None = None):
        """
        token_weight приходит готовым, если его разыграли
        заранее из ОТДЕЛЬНОГО потока.

        Причина в сравнении режимов словаря: nn.Embedding тянет
        vocab_size * d_model чисел из глобального генератора, и
        словарь другого размера сдвинул бы инициализацию
        Transformer, History Encoder и MLM head. Тогда арки
        различались бы не только словарём.

        skip_init создаёт модуль, ничего не разыгрывая, поэтому
        глобальный поток не расходуется вовсе.
        """

        super().__init__()

        self.config = config

        if token_weight is None:
            self.token = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        else:

            if tuple(token_weight.shape) != (config.vocab_size, config.d_model):
                raise BatchError(
                    f"таблица токенов {tuple(token_weight.shape)}, а конфиг требует "
                    f"({config.vocab_size}, {config.d_model})"
                )

            self.token = torch.nn.utils.skip_init(
                nn.Embedding,
                config.vocab_size,
                config.d_model,
                padding_idx=config.pad_id,
            )

            with torch.no_grad():
                self.token.weight.copy_(token_weight)

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
