from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .batching import BatchError
from .config import ModelConfig
from .history_batching import SessionInputs
from .time_encoding import sinusoidal_positions, squash
from .transformer import transformer_layers


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сессия приложения это одно действие клиента, а не десяток
# независимых экранов. Session Encoder сворачивает её в один
# вектор:
#
#     [SES], событие_1, ..., событие_N
#         -> двусторонний Transformer
#         -> hidden[SES] = вектор сессии
#
# [SES] это обучаемый вектор МОДЕЛИ, а не токен словаря:
# tokenizer про сессии не знает и знать не должен, иначе
# пришлось бы менять словарь и все уже собранные датасеты.
#
# Контекстные состояния событий сохраняются: MLM предсказывает
# поле экрана, видя и остальные экраны своей сессии.
#
# Паузы внутри сессии идут в МИНУТАХ через ту же squash, что и
# время истории, но своей проекцией: у сессии другой масштаб,
# и делить с историей одну матрицу значило бы утверждать, что
# час между покупками и час внутри сессии это одно и то же.
#
# Fast path здесь не отключается, в отличие от History Encoder:
# длина сессии это десятки, и матрица внимания ничтожна.
# ============================================================


@dataclass(frozen=True)
class SessionOutput:
    """
    Вектор сессии и контекстные состояния её событий.
    """

    pooled: torch.Tensor            # [n_sessions, d]
    hidden: torch.Tensor            # [n_sessions, 1 + S, d]
    padding_mask: torch.Tensor      # [n_sessions, 1 + S]

    @property
    def n_sessions(self) -> int:
        return int(self.pooled.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.hidden.shape[1])


class SessionEncoder(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.config = config

        # Как строка таблицы embedding: [EVT] и [USR] приходят
        # в свои энкодеры такими же по масштабу.
        self.ses = nn.Parameter(torch.randn(config.d_model))

        self.layers = transformer_layers(config, config.n_session_layers)

        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Одна пауза на событие: [n, 1] -> [n, d]. Без bias,
        # чтобы нулевая пауза первого события и padding не
        # получали временного слагаемого.
        self.time = nn.Linear(1, config.d_model, bias=False)

    # --------------------------------------------------------

    def assemble(
        self,
        event_vectors: torch.Tensor,
        sessions: SessionInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Последовательность одной сессии из готовых векторов событий.
        """

        if event_vectors.ndim != 2 or event_vectors.shape[1] != self.config.d_model:
            raise BatchError(
                f"ожидались векторы событий [n, {self.config.d_model}], "
                f"получено {tuple(event_vectors.shape)}"
            )

        rows = sessions.member_rows
        valid = sessions.member_valid

        if rows.shape != valid.shape:
            raise BatchError(
                f"состав сессий {tuple(rows.shape)} и его маска {tuple(valid.shape)} разной формы"
            )

        if rows.numel() and int(rows.max()) >= int(event_vectors.shape[0]):
            raise BatchError("состав сессии ссылается на событие вне batch")

        device = event_vectors.device
        dtype = event_vectors.dtype

        keep = valid.unsqueeze(-1)

        members = event_vectors[rows] * keep

        # Пауза до предыдущего события сессии, в минутах.
        gaps = self.time(squash(sessions.member_gap_minutes.float()).unsqueeze(-1))

        members = members + gaps.to(dtype) * keep

        lead = self.ses.to(dtype).view(1, 1, -1).expand(rows.shape[0], 1, -1)

        x = torch.cat([lead, members], dim=1)

        x = x + sinusoidal_positions(
            x.shape[1], self.config.d_model, device=device, dtype=dtype
        )

        padding_mask = torch.cat(
            [torch.zeros(rows.shape[0], 1, dtype=torch.bool, device=device), ~valid], dim=1
        )

        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0), padding_mask

    # --------------------------------------------------------

    def forward(
        self,
        event_vectors: torch.Tensor,
        sessions: SessionInputs,
    ) -> SessionOutput:

        if sessions.n_sessions == 0:
            raise BatchError("сессий нет: вызывать Session Encoder не на чем")

        x, padding_mask = self.assemble(event_vectors, sessions)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        hidden = self.final_norm(x)

        # Повторное обнуление после LayerNorm: нормировка нулевой
        # строки возвращает bias, а не ноль.
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return SessionOutput(pooled=hidden[:, 0], hidden=hidden, padding_mask=padding_mask)

    # --------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = ["SessionEncoder", "SessionOutput"]
