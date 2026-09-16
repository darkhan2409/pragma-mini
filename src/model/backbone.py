from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .batching import BatchError
from .config import ModelConfig
from .embeddings import draw_token_weight
from .encoders import EncoderPair
from .history_batching import ModelInputs
from .history_encoder import HistoryEncoder
from .time_features import CALENDAR_FEATURES, FeatureMLP


# ============================================================
# ИДЕЯ
# ============================================================
#
# Backbone соединяет три энкодера в один проход:
#
#   события  → Event Encoder    → вектор на событие
#   профиль  → Profile Encoder  → вектор профиля
#   сборка   → [профиль, события...] + позиция + время
#            → History Encoder  → вектор клиента и контекстные
#                                 векторы событий
#
# Позиция 0 это УЖЕ выход Profile Encoder: повторно доставать
# [USR] из таблицы токенов не нужно и нельзя, это был бы другой
# вектор.
#
# Embeddings событий не кэшируются между вызовами. Одно и то же
# событие может войти дважды под разными runtime-масками и с
# разным dropout, и его вектор обязан считаться заново.
#
# Ко времени к вектору элемента не прибавляется ничего: элемент
# несёт координату времени, по которой поворачиваются q и k
# внутри History Encoder. Календарь события и простой клиента
# приходят отдельными признаками.
# ============================================================


@dataclass(frozen=True)
class BackboneOutput:
    """
    Вектор клиента плюс всё, что понадобится MLM head.
    """

    client_embedding: torch.Tensor
    contextualized: torch.Tensor
    padding_mask: torch.Tensor

    event_embeddings: torch.Tensor
    event_example: torch.Tensor
    event_slot: torch.Tensor

    kept_events: torch.Tensor
    kept_tokens: torch.Tensor

    # Скрытые состояния запрошенных позиций внутри событий:
    # вход MLM head. None, когда голова не нужна.
    local_hidden: torch.Tensor | None = None


class Backbone(nn.Module):

    def __init__(self, config: ModelConfig, token_weight: torch.Tensor | None = None):

        super().__init__()

        self.config = config

        self.pair = EncoderPair(config, token_weight)
        self.history = HistoryEncoder(config)

        # Признаки времени строятся ПОСЛЕДНИМИ: тогда при одном
        # seed веса pair и history от них не зависят.
        self.calendar = FeatureMLP(CALENDAR_FEATURES, config)
        self.inactivity = FeatureMLP(1, config)

    # --------------------------------------------------------

    def assemble(
        self,
        inputs: ModelInputs,
        event_vectors: torch.Tensor,
        profile_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Последовательность истории из готовых векторов.

        Ко времени тут не прибавляется ничего: порядок и
        расстояние несёт поворот внутри внимания, а календарь и
        простой приходят отдельными признаками.
        """

        batch = inputs.n_examples
        width = inputs.max_length

        device = profile_vectors.device
        dtype = profile_vectors.dtype

        used = inputs.used_history_length

        rows = torch.arange(batch, device=device)

        x = torch.zeros(batch, width, self.config.d_model, device=device, dtype=dtype)

        # Профиль на позиции 0. Он получает простой клиента:
        # координаты истории относительны, и без этого признака
        # никто не знал бы, когда история кончилась.
        lead = profile_vectors + self.inactivity(inputs.temporal.inactivity).to(dtype)

        x = x.index_put((rows, torch.zeros_like(rows)), lead)

        if inputs.n_events:
            x = x.index_put(
                (inputs.example_of_event, inputs.slot_of_event),
                event_vectors,
            )

        padding_mask = torch.arange(width, device=device).unsqueeze(0) >= (used + 1).unsqueeze(1)

        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask

    # --------------------------------------------------------

    def history_coords(self, inputs: ModelInputs) -> torch.Tensor:
        """
        Координата времени каждого слота истории: [B, width].

        Слот 0 это профиль, и координата у него ноль — та же, что
        у самого свежего элемента: профиль это состояние НА
        cutoff, и отставать от последнего события ему незачем.

        У padding координата тоже ноль. Его выход всё равно
        зануляется после финальной нормы, а произвольный угол
        мешал бы читать промежуточные величины.
        """

        temporal = inputs.temporal

        if temporal is None:
            raise BatchError("признаков времени во входе нет")

        device = temporal.event_coords.device

        coords = torch.zeros(
            inputs.n_examples, inputs.max_length, device=device, dtype=torch.float32
        )

        if inputs.n_events:
            coords = coords.index_put(
                (inputs.example_of_event, inputs.slot_of_event),
                temporal.event_coords,
            )

        return coords

    # --------------------------------------------------------

    def encode_history(
        self,
        inputs: ModelInputs,
        event_vectors: torch.Tensor,
        profile_vectors: torch.Tensor,
    ) -> BackboneOutput:

        x, padding_mask = self.assemble(inputs, event_vectors, profile_vectors)

        coords = self.history_coords(inputs)

        hidden = self.history(x, padding_mask, coords)

        return BackboneOutput(
            client_embedding=hidden[:, 0],
            contextualized=hidden,
            padding_mask=padding_mask,
            event_embeddings=hidden[inputs.example_of_event, inputs.slot_of_event],
            event_example=inputs.example_of_event,
            event_slot=inputs.slot_of_event,
            kept_events=inputs.kept_events,
            kept_tokens=inputs.kept_tokens,
        )

    def forward(
        self,
        inputs: ModelInputs,
        event_microbatch: int | None = 1024,
        gather: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> BackboneOutput:
        """
        Один проход.

        gather задаёт позиции внутри событий, чьи скрытые
        состояния нужны MLM head. Они берутся из этого же прохода
        Event Encoder: повторный запуск ради них считал бы другое.
        """

        if inputs.temporal is None:
            raise BatchError("во входе нет признаков времени")

        local_hidden = None

        if gather is None:
            event_vectors = self.pair.encode_events(inputs.events, event_microbatch)
        else:
            event_vectors, local_hidden = self.pair.encode_events(
                inputs.events, event_microbatch, gather
            )

        # h_local календарём не трогается: он снят с Event
        # Encoder выше, и вход MLM head от календаря не зависит.
        event_vectors = event_vectors + self.calendar(inputs.temporal.calendar).to(
            event_vectors.dtype
        )

        profile_vectors = self.pair.encode_profiles(inputs.profiles)

        out = self.encode_history(inputs, event_vectors, profile_vectors)

        return out if gather is None else replace(out, local_hidden=local_hidden)

    # --------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_backbone(config: ModelConfig, seed: int = 42, device=None) -> Backbone:
    """
    Seed задаётся один раз; сборка на CPU, затем перенос.

    Таблица токенов разыгрывается ДО и из своего потока, чтобы
    размер словаря не сдвигал инициализацию всего остального:
    иначе сравнение режимов словаря мерило бы ещё и разные
    стартовые веса.
    """

    token_weight = draw_token_weight(config, seed)

    torch.manual_seed(seed)

    backbone = Backbone(config, token_weight)

    if device is not None:
        backbone = backbone.to(device)

    return backbone
