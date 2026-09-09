from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .batching import BatchError
from .config import ModelConfig
from .encoders import EncoderPair
from .history_batching import ModelInputs
from .history_encoder import HistoryEncoder
from .session_encoder import SessionEncoder
from .time_encoding import TimeEncoding, sinusoidal_positions


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

    # Сессии. None в прежней структуре истории.
    session_pooled: torch.Tensor | None = None
    session_hidden: torch.Tensor | None = None
    session_embeddings: torch.Tensor | None = None
    session_of_event: torch.Tensor | None = None
    position_in_session: torch.Tensor | None = None

    @property
    def n_examples(self) -> int:
        return int(self.client_embedding.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.contextualized.shape[1])


class Backbone(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.config = config

        self.pair = EncoderPair(config)
        self.history = HistoryEncoder(config)
        self.time = TimeEncoding(config)

        # Session Encoder строится ПОСЛЕДНИМ: тогда при одном
        # seed общие веса обеих структур совпадают, и равенство
        # выходов на batch без сессий можно проверить.
        self.session = SessionEncoder(config) if config.uses_sessions else None

    # --------------------------------------------------------

    def assemble(
        self,
        inputs: ModelInputs,
        event_vectors: torch.Tensor,
        profile_vectors: torch.Tensor,
        session_vectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Последовательность истории из готовых векторов.

        Элемент истории это отдельное событие либо сессия. В
        прежней структуре отдельные события это все события,
        и ветка та же самая.
        """

        batch = inputs.n_examples
        width = inputs.max_length

        device = profile_vectors.device
        dtype = profile_vectors.dtype

        used = inputs.used_history_length

        rows = torch.arange(batch, device=device)

        x = torch.zeros(batch, width, self.config.d_model, device=device, dtype=dtype)

        # Профиль на позиции 0, без временного слагаемого.
        x = x.index_put((rows, torch.zeros_like(rows)), profile_vectors)

        standalone = inputs.standalone_rows

        if standalone is None:

            if inputs.n_events:
                x = x.index_put(
                    (inputs.example_of_event, inputs.slot_of_event),
                    event_vectors + self.time(inputs.time_hours).to(dtype),
                )

        elif int(standalone.numel()):

            x = x.index_put(
                (inputs.example_of_event[standalone], inputs.slot_of_event[standalone]),
                event_vectors[standalone] + self.time(inputs.time_hours).to(dtype),
            )

        if session_vectors is not None and inputs.sessions is not None:

            sessions = inputs.sessions

            if int(sessions.n_sessions):
                x = x.index_put(
                    (sessions.session_example, sessions.session_slot),
                    session_vectors + self.time(sessions.session_hours).to(dtype),
                )

        x = x + sinusoidal_positions(width, self.config.d_model, device=device, dtype=dtype)

        padding_mask = torch.arange(width, device=device).unsqueeze(0) >= (used + 1).unsqueeze(1)

        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask

    # --------------------------------------------------------

    def encode_history(
        self,
        inputs: ModelInputs,
        event_vectors: torch.Tensor,
        profile_vectors: torch.Tensor,
        attention_rule: str | None = None,
        session_out=None,
    ) -> BackboneOutput:

        session_vectors = None if session_out is None else session_out.pooled

        x, padding_mask = self.assemble(inputs, event_vectors, profile_vectors, session_vectors)

        hidden = self.history(x, padding_mask, attention_rule)

        sessions = inputs.sessions

        session_embeddings = None

        if sessions is not None and int(sessions.n_sessions):
            session_embeddings = hidden[sessions.session_example, sessions.session_slot]

        return BackboneOutput(
            client_embedding=hidden[:, 0],
            contextualized=hidden,
            padding_mask=padding_mask,
            # Вектор НЕСУЩЕГО элемента: своего события или его
            # сессии. Один адрес в обеих структурах.
            event_embeddings=hidden[inputs.example_of_event, inputs.slot_of_event],
            event_example=inputs.example_of_event,
            event_slot=inputs.slot_of_event,
            kept_events=inputs.kept_events,
            kept_tokens=inputs.kept_tokens,
            session_pooled=None if session_out is None else session_out.pooled,
            session_hidden=None if session_out is None else session_out.hidden,
            session_embeddings=session_embeddings,
            session_of_event=None if sessions is None else sessions.session_of_event,
            position_in_session=None if sessions is None else sessions.position_in_session,
        )

    def forward(
        self,
        inputs: ModelInputs,
        event_microbatch: int | None = 1024,
        gather: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_rule: str | None = None,
    ) -> BackboneOutput:
        """
        Один проход.

        gather задаёт позиции внутри событий, чьи скрытые
        состояния нужны MLM head. Они берутся из этого же прохода
        Event Encoder: повторный запуск ради них считал бы другое.

        attention_rule ограничивает внимание History Encoder и
        нужен только диагностике; Event и Profile Encoder он не
        трогает по построению.
        """

        has_module = self.session is not None
        has_inputs = inputs.sessions is not None

        if has_module != has_inputs:
            raise BatchError(
                "структура модели и структура входа не совпадают: "
                f"Session Encoder в модели {has_module}, сессии во входе {has_inputs}"
            )

        local_hidden = None

        if gather is None:
            event_vectors = self.pair.encode_events(inputs.events, event_microbatch)
        else:
            event_vectors, local_hidden = self.pair.encode_events(
                inputs.events, event_microbatch, gather
            )

        profile_vectors = self.pair.encode_profiles(inputs.profiles)

        session_out = None

        # Сессий может не быть и в session-структуре: у клиента
        # без приложения их нет вовсе.
        if has_module and int(inputs.sessions.n_sessions):
            session_out = self.session(event_vectors, inputs.sessions)

        out = self.encode_history(
            inputs, event_vectors, profile_vectors, attention_rule, session_out
        )

        return out if gather is None else replace(out, local_hidden=local_hidden)

    # --------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_backbone(config: ModelConfig, seed: int = 42, device=None) -> Backbone:
    """
    Seed задаётся один раз; сборка на CPU, затем перенос.
    """

    torch.manual_seed(seed)

    backbone = Backbone(config)

    if device is not None:
        backbone = backbone.to(device)

    return backbone
