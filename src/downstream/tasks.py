from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.preprocessing.canonical.build import EVENTS_FILE
from src.preprocessing.settings import group_dir

from .settings import CHURN_REPORTS, DOWNSTREAM_DIR, HORIZON_DAYS, cutoff


# ============================================================
# ЗАДАЧИ
# ============================================================
#
# Метки строятся только из наблюдаемой ленты 02_preprocessed —
# скрытое состояние генератора (truth) не читается. Признаки —
# только события строго раньше T, метка — только события в
# (T, T + HORIZON_DAYS]. Событие ровно в T не входит ни туда, ни
# туда.
#
#   churn  строки, метки и прогнозы churn_baseline: те же клиенты,
#          тот же T и та же метка, что у CatBoost.
#   ndq    новая просрочка. Популяция — клиенты с плановым платежом
#          или оплатой кредита за 90 дней до T и без пропуска или
#          просрочки за 45 дней до T; метка — просрочка в окне.
#          Без второго условия задача свелась бы к «просрочка уже
#          идёт».
#   a1     заявка на продукт в окне; популяция — все клиенты с
#          событиями до T.
#
# Рядом — счётчики событий каждого типа за 30, 90 и 365 дней до T,
# число событий и давность последнего: обычный агрегатный бейзлайн
# для задач, у которых своего CatBoost нет.
# ============================================================


COUNT_WINDOWS = (30, 90, 365)

NDQ_ACTIVE = ("installment_due", "loan_payment")
NDQ_ACTIVE_DAYS = 90
NDQ_BAD = ("installment_missed", "delinquency_registered")
NDQ_BAD_DAYS = 45
NDQ_LABEL = "delinquency_registered"

A1_LABEL = "application_submitted"

TASKS_DIR = "tasks"


def _counts(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Клиенты × типы: число событий.
    """

    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name="client_id"))

    table = frame.groupby(["client_id", "type"], observed=True).size().unstack(fill_value=0)
    table.columns = [f"{prefix}{name}" for name in table.columns]

    return table


def _partial(table: pd.DataFrame, moment: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Суммы и максимумы одной группы строк 02. Клиент может лежать в
    двух соседних группах строк: суммы и максимумы складываются.
    """

    horizon = moment + pd.Timedelta(days=HORIZON_DAYS)

    before = table[table["event_time"] < moment]
    after = table[(table["event_time"] > moment) & (table["event_time"] <= horizon)]

    sums = [
        before.groupby("client_id", observed=True).size().rename("n_events").to_frame(),
        *[
            _counts(before[before["event_time"] >= moment - pd.Timedelta(days=days)], f"n_{days}d_")
            for days in COUNT_WINDOWS
        ],
        before[
            before["type"].isin(NDQ_ACTIVE)
            & (before["event_time"] >= moment - pd.Timedelta(days=NDQ_ACTIVE_DAYS))
        ].groupby("client_id", observed=True).size().rename("ndq_active").to_frame(),
        before[
            before["type"].isin(NDQ_BAD)
            & (before["event_time"] >= moment - pd.Timedelta(days=NDQ_BAD_DAYS))
        ].groupby("client_id", observed=True).size().rename("ndq_bad").to_frame(),
        after[after["type"] == NDQ_LABEL].groupby("client_id", observed=True).size()
        .rename("ndq_after").to_frame(),
        after[after["type"] == A1_LABEL].groupby("client_id", observed=True).size()
        .rename("a1_after").to_frame(),
    ]

    last = before.groupby("client_id", observed=True)["event_time"].max().rename("last_event").to_frame()

    return pd.concat(sums, axis=1), last


def build_table(group: str, moment: datetime) -> pd.DataFrame:
    """
    По клиенту группы с событиями до T: счётчики, давность, метки.
    """

    stamp = pd.Timestamp(moment)

    events = pq.ParquetFile(group_dir(group) / EVENTS_FILE)

    sums, lasts = [], []

    for index in range(events.num_row_groups):

        table = events.read_row_group(index, columns=["client_id", "event_time", "type"]).to_pandas()

        part, last = _partial(table, stamp)

        sums.append(part)
        lasts.append(last)

    total = pd.concat(sums).fillna(0).groupby(level=0).sum()
    last = pd.concat(lasts).groupby(level=0).max()

    total = total[total["n_events"] > 0].join(last, how="left")

    windows = tuple(f"n_{days}d_" for days in COUNT_WINDOWS)
    counts = sorted(name for name in total.columns if name.startswith(windows))

    def column(name: str) -> pd.Series:
        return total[name] if name in total else pd.Series(0, index=total.index)

    out = pd.concat(
        [
            total["n_events"].astype(np.int64),
            (stamp - total["last_event"]).dt.total_seconds().rename("gap_seconds"),
            total[counts].astype(np.int64),
            ((column("ndq_active") > 0) & (column("ndq_bad") == 0)).rename("ndq_population"),
            (column("ndq_after") > 0).rename("ndq"),
            (column("a1_after") > 0).rename("a1"),
        ],
        axis=1,
    )
    out.index.name = "client_id"

    return out.sort_index()


def table(group: str) -> pd.DataFrame:
    """
    Таблица задач группы на её T — из кэша, если он собран по тому
    же файлу 02 и тому же T.
    """

    moment = cutoff(group)
    source = group_dir(group) / EVENTS_FILE
    stat = source.stat()

    stamp = {"cutoff": moment.isoformat(), "source": str(source), "size": stat.st_size,
             "mtime_ns": stat.st_mtime_ns}

    directory = DOWNSTREAM_DIR / TASKS_DIR
    path = directory / f"{group}.parquet"
    meta = directory / f"{group}.json"

    if path.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == stamp:
        return pd.read_parquet(path)

    built = build_table(group, moment)

    directory.mkdir(parents=True, exist_ok=True)
    built.to_parquet(path)
    meta.write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")

    return built


def churn_rows(group: str) -> pd.DataFrame:
    """
    Строки churn-бейзлайна группы: client_id, churn и прогноз
    CatBoost. T строк обязан совпасть с T группы.
    """

    name = "train_rows.parquet" if group == "train" else "eval_rows.parquet"

    rows = pd.read_parquet(CHURN_REPORTS / name)
    rows = rows[rows["group"] == group]

    moment = pd.Timestamp(cutoff(group))

    found = pd.to_datetime(rows["T"], utc=True).unique()

    if len(found) != 1 or found[0] != moment:
        raise ValueError(
            f"churn-бейзлайн группы {group} построен на T {list(found)}, а T группы здесь "
            f"{moment.isoformat()}: сравнение было бы на разных моментах"
        )

    return rows.set_index("client_id")[["churn", "score"]].sort_index()


__all__ = [
    "A1_LABEL",
    "COUNT_WINDOWS",
    "NDQ_LABEL",
    "build_table",
    "churn_rows",
    "table",
]
