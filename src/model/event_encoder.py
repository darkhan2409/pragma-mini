from __future__ import annotations

from .config import ModelConfig
from .embeddings import SharedEmbeddings
from .transformer import RecordEncoder


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вход это одно событие целиком:
#
#     [EVT], event_type, поля payload
#
# event_type уже закодирован tokenizer как обычная тройка на
# позиции 1, повторно его добавлять нельзя.
#
# Выход это скрытое состояние [EVT] после финального LayerNorm:
# один вектор на событие.
# ============================================================


class EventEncoder(RecordEncoder):

    def __init__(self, config: ModelConfig, embeddings: SharedEmbeddings):
        super().__init__(
            config=config,
            embeddings=embeddings,
            n_layers=config.n_event_layers,
            lead_id=config.evt_id,
            name="EventEncoder",
        )
