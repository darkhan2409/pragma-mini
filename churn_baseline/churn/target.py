from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .config import HORIZON


# ============================================================
# TARGET
# ============================================================
#
#   churn = 1   ни одного действия клиента в (T, T + 60 дней]
#   churn = 0   хотя бы одно
#
# Популяция — клиенты, у которых до T было хотя бы одно действие:
# клиент без действий уже неактивен, и ярлык для него тривиален.
#
# Событие ровно в T не входит ни в признаки (они строго раньше T), ни
# в окно target (оно открыто слева).
# ============================================================


def labels(events: pd.DataFrame, cutoff: datetime, action: np.ndarray) -> pd.DataFrame:
    """
    По клиенту блока: было ли действие до T и отток в окне после T.
    Индекс client_id — те же клиенты и в том же порядке, что у признаков.
    """
    clients = pd.Index(pd.unique(events["client_id"]), name="client_id")
    code = clients.get_indexer(events["client_id"])
    t = events["t"]
    before = action & (t < cutoff).to_numpy()
    window = action & ((t > cutoff) & (t <= cutoff + HORIZON)).to_numpy()
    n = len(clients)
    return pd.DataFrame(
        {
            "has_action_before": np.bincount(code[before], minlength=n) > 0,
            "has_history_before": np.bincount(code[(t < cutoff).to_numpy()], minlength=n) > 0,
            "churn": (np.bincount(code[window], minlength=n) == 0).astype(np.int8),
        },
        index=clients,
    )
