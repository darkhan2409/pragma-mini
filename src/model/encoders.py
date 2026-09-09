from __future__ import annotations

import torch
import torch.nn as nn

from src.tokenizer.dataset import TokenBatch

from .batching import PaddedRecords, events_from_batch, profiles_from_batch
from .config import ModelConfig
from .embeddings import SharedEmbeddings
from .event_encoder import EventEncoder
from .profile_encoder import ProfileEncoder


# ============================================================
# ИДЕЯ
# ============================================================
#
# Два энкодера с общими таблицами E и P и полностью
# независимыми Transformer-блоками.
#
# Таблицы регистрируются только здесь, у пары. Энкодеры держат
# на них обычную ссылку, поэтому state_dict не содержит трёх
# копий одних весов, а parameters() отдаёт каждый тензор один
# раз.
#
# Seed задаётся один раз при сборке и только на CPU: если
# строить сразу на CUDA, числа берутся из другого генератора, и
# «один seed — одни веса» тихо перестало бы выполняться.
# ============================================================


class EncoderPair(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.config = config

        self.embeddings = SharedEmbeddings(config)

        self.event = EventEncoder(config, self.embeddings)
        self.profile = ProfileEncoder(config, self.embeddings)

    # --------------------------------------------------------

    def encode_events(
        self,
        records: PaddedRecords,
        microbatch_size: int | None = None,
        gather: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        return self.event.encode(records, microbatch_size, gather)

    def encode_profiles(self, records: PaddedRecords, microbatch_size: int | None = None) -> torch.Tensor:
        return self.profile.encode(records, microbatch_size)

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_encoders(config: ModelConfig, seed: int = 42, device=None) -> EncoderPair:
    """
    Сборка всегда на CPU, затем перенос: веса не зависят от того,
    на каком устройстве модель будет считать.
    """

    torch.manual_seed(seed)

    pair = EncoderPair(config)

    if device is not None:
        pair = pair.to(device)

    return pair


# ============================================================
# ПОВЕРХ TokenBatch
# ============================================================


def encode_events(
    pair: EncoderPair,
    batch: TokenBatch,
    microbatch_size: int | None = None,
    device=None,
) -> torch.Tensor:
    """
    Все события batch в [n_events, d_model].
    """

    records = events_from_batch(batch, pair.config)

    if device is not None:
        records = records.to(device)

    return pair.encode_events(records, microbatch_size)


def encode_profiles(
    pair: EncoderPair,
    batch: TokenBatch,
    microbatch_size: int | None = None,
    device=None,
) -> torch.Tensor:
    """
    Все профили batch в [n_profiles, d_model].
    """

    records = profiles_from_batch(batch, pair.config)

    if device is not None:
        records = records.to(device)

    return pair.encode_profiles(records, microbatch_size)
