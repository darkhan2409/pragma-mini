from __future__ import annotations

from .config import ModelConfig
from .embeddings import SharedEmbeddings
from .transformer import RecordEncoder


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вход это снимок профиля as-of:
#
#     [USR], 20 полей профиля
#
# Профиль шире лимита события и это нормально: лимит токенов
# события к профилю не относится.
#
# Выход это скрытое состояние [USR] после финального LayerNorm.
# ============================================================


class ProfileEncoder(RecordEncoder):

    def __init__(self, config: ModelConfig, embeddings: SharedEmbeddings):
        super().__init__(
            config=config,
            embeddings=embeddings,
            n_layers=config.n_profile_layers,
            lead_id=config.usr_id,
            name="ProfileEncoder",
        )
