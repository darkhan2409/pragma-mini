from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .config import HORIZON, RECENT


# ============================================================
# TARGET И ЗАДАЧИ
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
#
# Популяция и active90 — только история до T. Target — события окна:
# у val и test из той же выгрузки, у train из продолжения
# (sources.acting_clients) — выгрузка train кончается ровно в T.
#
# Задача — на строках популяции:
#
#   churn_active90  клиенты с действием за 90 дней до T
#                   (T − 90 дней ≤ t < T): уйдёт ли недавно активный
#                   клиент в следующие 60 дней. Давно замолчавший
#                   клиент почти наверняка не вернётся, и задачу о нём
#                   во многом решала бы давность последнего действия.
# ============================================================


TASKS: tuple[str, ...] = ("churn_active90",)


def labels(
    events: pd.DataFrame, cutoff: datetime, action: np.ndarray, acting: set[str] | None = None
) -> pd.DataFrame:
    """
    По клиенту блока: было ли действие до T, было ли оно за последние
    RECENT до T и отток в окне после T. Индекс client_id — те же
    клиенты и в том же порядке, что у признаков.

    acting — клиенты с действием в окне по продолжению группы; без
    него окно берётся из тех же событий блока.
    """
    clients = pd.Index(pd.unique(events["client_id"]), name="client_id")
    code = clients.get_indexer(events["client_id"])
    t = events["t"]
    before = action & (t < cutoff).to_numpy()
    recent = before & (t >= cutoff - RECENT).to_numpy()
    n = len(clients)

    if acting is None:
        window = action & ((t > cutoff) & (t <= cutoff + HORIZON)).to_numpy()
        churn = np.bincount(code[window], minlength=n) == 0
    else:
        # Выгрузка такой группы кончается в T: событие не раньше T —
        # не та выгрузка.
        if (t >= cutoff).any():
            raise ValueError(f"в выгрузке есть события не раньше T {cutoff.isoformat()}: метка ушла бы в признаки")
        churn = ~clients.isin(list(acting))

    return pd.DataFrame(
        {
            "has_action_before": np.bincount(code[before], minlength=n) > 0,
            "has_history_before": np.bincount(code[(t < cutoff).to_numpy()], minlength=n) > 0,
            "active90": np.bincount(code[recent], minlength=n) > 0,
            "churn": churn.astype(np.int8),
        },
        index=clients,
    )


def task_rows(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    """
    Строки задачи среди строк популяции группы.
    """
    if task == "churn_active90":
        return frame[frame["active90"]]
    raise ValueError(f"задача {task!r} не из {TASKS}")
