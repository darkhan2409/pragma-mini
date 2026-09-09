from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import FEATURE_END, LABEL_END
from .history import ClientHistory


# ============================================================
# ИДЕЯ
# ============================================================
#
# Метка это следствие БУДУЩЕГО поведения, а не скрытой персоны.
#
#     history.before(FEATURE_END)  -> признаки (RAW)
#     history.since(FEATURE_END)   -> окно метки
#
# В V1 метка одна: открыл ли клиент хоть один продукт
# в 90-дневном окне после среза. Никакого RNG здесь нет.
# ============================================================


@dataclass(frozen=True)
class ClientLabels:

    client_id: int

    label_start: datetime
    label_end: datetime

    product_open_90d: bool


def derive_labels(history: ClientHistory) -> ClientLabels:

    opened = any(
        FEATURE_END <= event.ts < LABEL_END for event in history.product_events
    )

    return ClientLabels(
        client_id=history.client_id,
        label_start=FEATURE_END,
        label_end=LABEL_END,
        product_open_90d=opened,
    )
