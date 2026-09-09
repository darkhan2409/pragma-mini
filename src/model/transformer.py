from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .batching import BatchError, PaddedRecords
from .config import ModelConfig
from .embeddings import SharedEmbeddings


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна запись это одна строка batch: событие не видит соседние
# события, потому что attention работает внутри строки.
#
# Стек собирается ModuleList из отдельно созданных слоёв, а не
# nn.TransformerEncoder: тот делает deepcopy одного слоя, и все
# блоки стартуют с одинаковых весов.
#
# Pooling это скрытое состояние позиции 0 ПОСЛЕ финального
# LayerNorm. Mean pooling не используется: у записей разная
# длина, и усреднение размывало бы ведущий токен.
#
# Padded позиции зануляются дважды: до слоёв и после финальной
# нормы. Второе не перестраховка — LayerNorm нулевой строки
# даёт ненулевой вектор из bias.
# ============================================================


def transformer_layers(config: ModelConfig, n_layers: int) -> nn.ModuleList:
    """
    Стек независимо инициализированных слоёв.

    Каждый слой создаётся отдельно. nn.TransformerEncoder так
    нельзя: он делает deepcopy одного слоя, и все блоки
    стартуют с одинаковых весов.
    """

    return nn.ModuleList(
        [
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ]
    )


@dataclass(frozen=True)
class EncoderOutput:
    pooled: torch.Tensor
    hidden: torch.Tensor
    padding_mask: torch.Tensor


class RecordEncoder(nn.Module):
    """
    Кодировщик одной записи в вектор d_model.
    """

    def __init__(
        self,
        config: ModelConfig,
        embeddings: SharedEmbeddings,
        n_layers: int,
        lead_id: int,
        name: str,
    ):

        super().__init__()

        self.config = config
        self.n_layers = n_layers
        self.lead_id = lead_id
        self.name = name

        # Ссылка, а не подмодуль: таблицы регистрирует пара,
        # иначе state_dict нёс бы три копии одних весов.
        object.__setattr__(self, "embeddings", embeddings)

        self.layers = transformer_layers(config, n_layers)

        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    # --------------------------------------------------------

    def contextualize(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Стек слоёв без финальной нормы.

        Маска идёт в каждый слой; causal mask не используется,
        attention двусторонний.
        """

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        return x

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
        return_hidden: bool = False,
    ):

        if key_ids.ndim != 2:
            raise BatchError(f"{self.name}: ожидался batch [n, L], получено {tuple(key_ids.shape)}")

        if bool(padding_mask.all(dim=1).any()):
            raise BatchError(f"{self.name}: есть запись целиком из padding, кодировать нечего")

        lead = key_ids[:, 0]

        if not bool(((lead == self.lead_id) & (value_ids[:, 0] == self.lead_id)).all()):
            raise BatchError(
                f"{self.name}: не каждая запись начинается с ведущего токена {self.lead_id}"
            )

        x = self.embeddings(key_ids, value_ids, positions, padding_mask)

        hidden = self.final_norm(self.contextualize(x, padding_mask))

        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        pooled = hidden[:, 0]

        if return_hidden:
            return EncoderOutput(pooled=pooled, hidden=hidden, padding_mask=padding_mask)

        return pooled

    # --------------------------------------------------------

    def _check_gather(
        self, records: PaddedRecords, gather: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Запрошенные позиции существуют и идут по возрастанию записей.
        """

        rows, cols = gather

        if rows.ndim != 1 or cols.ndim != 1 or rows.shape != cols.shape:
            raise BatchError(
                f"{self.name}: gather ждёт два одномерных массива одной длины, "
                f"получено {tuple(rows.shape)} и {tuple(cols.shape)}"
            )

        device = records.key_ids.device

        rows = rows.to(device)
        cols = cols.to(device)

        if rows.numel() == 0:
            return rows, cols

        # Неубывание обязательно: иначе разбиение на microbatch
        # переставляло бы выходы.
        if bool((rows[1:] < rows[:-1]).any()):
            raise BatchError(f"{self.name}: gather требует неубывающих номеров записей")

        if int(rows.min()) < 0 or int(rows.max()) >= len(records):
            raise BatchError(
                f"{self.name}: gather указывает на запись вне batch [0, {len(records)}): "
                f"диапазон {int(rows.min())}..{int(rows.max())}"
            )

        limits = records.lengths.to(device)[rows]

        if int(cols.min()) < 0 or bool((cols >= limits).any()):
            raise BatchError(f"{self.name}: gather указывает на padding или за границу записи")

        return rows, cols

    @staticmethod
    def _gather_edges(rows: torch.Tensor, total: int, microbatch_size: int) -> list[int]:
        """
        Границы кусков в массиве запрошенных позиций.
        """

        bounds = torch.arange(
            0, total + microbatch_size, microbatch_size, device=rows.device, dtype=torch.long
        ).clamp(max=total)

        return torch.searchsorted(rows, bounds).tolist()

    def encode(
        self,
        records: PaddedRecords,
        microbatch_size: int | None = None,
        gather: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Кодирует записи кусками, сохраняя порядок выходов.

        Ширина куска обрезается до самой длинной записи в нём:
        результат от количества padding не зависит.

        gather=(rows, cols) дополнительно отдаёт скрытые состояния
        отдельных позиций ИЗ ЭТОГО ЖЕ прохода. Второй проход ради
        них считал бы другое: dropout и маска у него свои.
        """

        total = len(records)

        if total == 0:
            raise BatchError(f"{self.name}: нечего кодировать")

        if microbatch_size is not None and microbatch_size < 1:
            raise ValueError("microbatch_size должен быть положительным")

        whole = microbatch_size is None or microbatch_size >= total

        if gather is None:

            if whole:
                return self.forward(
                    records.key_ids, records.value_ids, records.positions, records.padding_mask
                )

            chunks = []

            for start in range(0, total, microbatch_size):

                piece = records.slice(start, min(start + microbatch_size, total))

                chunks.append(
                    self.forward(piece.key_ids, piece.value_ids, piece.positions, piece.padding_mask)
                )

            return torch.cat(chunks, dim=0)

        rows, cols = self._check_gather(records, gather)

        if whole:

            out = self.forward(
                records.key_ids,
                records.value_ids,
                records.positions,
                records.padding_mask,
                return_hidden=True,
            )

            return out.pooled, out.hidden[rows, cols]

        edges = self._gather_edges(rows, total, microbatch_size)

        chunks: list[torch.Tensor] = []
        picked: list[torch.Tensor] = []

        for index, start in enumerate(range(0, total, microbatch_size)):

            piece = records.slice(start, min(start + microbatch_size, total))

            out = self.forward(
                piece.key_ids,
                piece.value_ids,
                piece.positions,
                piece.padding_mask,
                return_hidden=True,
            )

            chunks.append(out.pooled)

            lo, hi = edges[index], edges[index + 1]

            if hi > lo:
                picked.append(out.hidden[rows[lo:hi] - start, cols[lo:hi]])

        pooled = torch.cat(chunks, dim=0)

        if not picked:
            return pooled, pooled.new_zeros((0, self.config.d_model))

        return pooled, torch.cat(picked, dim=0)
