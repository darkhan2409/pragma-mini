from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.preprocessing.canonical.build import EVENTS_FILE
from src.preprocessing.rawdata import read_manifest
from src.preprocessing.settings import group_dir, raw_group_dir

from .settings import CHURN_FUTURE, CHURN_REPORTS, DOWNSTREAM_DIR, FUTURE_LABEL_GROUPS, cutoff


# ============================================================
# ЗАДАЧИ
# ============================================================
#
#   churn_active90  строки, метки и прогнозы churn_baseline: те же
#                   клиенты, тот же T и та же метка, что у CatBoost.
#                   Клиенты с действием за 90 дней до T
#                   (T − 90 дней ≤ t < T). Строки отбирает и CatBoost
#                   учит churn_baseline: «действие клиента» определено
#                   только там.
#
# Рядом — агрегатные признаки из наблюдаемой ленты 02_preprocessed,
# только по событиям строго раньше T: счётчики событий каждого типа
# за 30, 90 и 365 дней, число событий и давность последнего.
# ============================================================


COUNT_WINDOWS = (30, 90, 365)

CHURN_TASKS = ("churn_active90",)

# Прогнозы CatBoost на [USR] в отчётах churn-бейзлайна: набор →
# каталог <каталог>/<тег> и ключ python -m churn.plus_usr.
#   catboost_plus_usr  полный X бейзлайна и [USR]
#   catboost_usr       только [USR] — диагностика головы
USR_CATBOOST = {"catboost_plus_usr": ("plus_usr", ""), "catboost_usr": ("usr_only", " --usr-only")}

TASKS_DIR = "tasks"

# Состав таблицы: кэш прежнего состава пересчитывается.
TABLE_FORMAT = 2


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

    before = table[table["event_time"] < moment]

    sums = [
        before.groupby("client_id", observed=True).size().rename("n_events").to_frame(),
        *[
            _counts(before[before["event_time"] >= moment - pd.Timedelta(days=days)], f"n_{days}d_")
            for days in COUNT_WINDOWS
        ],
    ]

    last = before.groupby("client_id", observed=True)["event_time"].max().rename("last_event").to_frame()

    return pd.concat(sums, axis=1), last


def build_table(group: str, moment: datetime) -> pd.DataFrame:
    """
    По клиенту группы с событиями до T: счётчики и давность.
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

    out = pd.concat(
        [
            total["n_events"].astype(np.int64),
            (stamp - total["last_event"]).dt.total_seconds().rename("gap_seconds"),
            total[counts].astype(np.int64),
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

    stamp = {"format": TABLE_FORMAT, "cutoff": moment.isoformat(), "source": str(source),
             "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

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


def churn_sources(group: str) -> dict:
    """
    Источники строк churn-бейзлайна группы, сверенные с текущими:

      история признаков   текущая выгрузка группы (sha256 событий);
      анкета признаков    её же анкета (sha256);
      метка               та же выгрузка (val, test) или продолжение
                          (train) — то самое, что прочёл бейзлайн, и
                          продолжающее именно текущую выгрузку.

    Каждый источник сверяется отдельно: одна выгрузка и одно
    продолжение — разные наборы, и один sha256 их не описывает.
    """

    metrics = json.loads((CHURN_REPORTS / "metrics.json").read_text(encoding="utf-8"))

    built = metrics.get("sources", {}).get(group)

    if built is None:
        raise ValueError(
            f"в churn-бейзлайне нет группы {group}: "
            + ("python -m churn.build test --final-test и python -m churn.train --final-test"
               if group == "test" else f"python -m churn.build {group} и python -m churn.train")
        )

    exported = read_manifest(raw_group_dir(group))

    problems = []

    if built["feature_history_events_sha256"] != exported.events_sha256:
        problems.append("история признаков")

    if built["feature_profile_sha256"] != exported.profile_sha256:
        problems.append("анкета признаков")

    expected = "future" if group in FUTURE_LABEL_GROUPS else "export"

    if built["target_source"] != expected:
        problems.append(f"метка из {built['target_source']}, а нужна из {expected}")
    elif expected == "export":
        if built["target_events_sha256"] != exported.events_sha256:
            problems.append("источник метки")
    else:
        path = CHURN_FUTURE / group / "future.json"
        future = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if future is None:
            problems.append(f"нет {path}")
        elif future["events_sha256"] != built["target_events_sha256"]:
            problems.append("продолжение — источник метки")
        elif (future["source"]["events_sha256"], future["source"]["profile_sha256"]) != (
            exported.events_sha256, exported.profile_sha256
        ):
            problems.append("продолжение другой выгрузки")

    if problems:
        raise ValueError(
            f"churn-бейзлайн группы {group} собран на других источниках ({'; '.join(problems)}): "
            "пересоберите его в churn_baseline"
        )

    return built


def churn_thresholds() -> dict[str, float]:
    """
    Порог CatBoost-бейзлайна по задаче: выбран на inner_holdout train.
    """

    metrics = json.loads((CHURN_REPORTS / "metrics.json").read_text(encoding="utf-8"))

    return {task: float(block["threshold"]) for task, block in metrics["tasks"].items()}


def usr_catboost(name: str, tag: str, used: tuple[str, ...], embedded: dict) -> dict | None:
    """
    Прогнозы CatBoost churn-бейзлайна на [USR] векторов тега — набор
    name из USR_CATBOOST: порог по задаче и строки (churn, score) по
    задаче и группе. None — для этих векторов он не обучен.

    embedded — meta.json векторов. Годится только CatBoost, обученный
    ровно на них: тот же тег и чекпойнт и те же записи групп (у снятых
    заново векторов запись другая, хотя бы по времени съёма), — и на
    тех же источниках строк, что и сам бейзлайн.
    """

    folder, flag = USR_CATBOOST[name]
    directory = CHURN_REPORTS / folder / tag

    if not (directory / "metrics.json").exists():
        return None

    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    baseline = json.loads((CHURN_REPORTS / "metrics.json").read_text(encoding="utf-8"))
    recorded = metrics["embeddings"]

    problems = []

    if recorded["tag"] != tag or recorded["checkpoint"] != embedded.get("checkpoint"):
        problems.append(f"обучен на векторах {recorded['tag']} ({recorded['checkpoint']})")

    for group in used:
        if group not in metrics["sources"]:
            problems.append(f"нет группы {group}" + (" — churn.plus_usr --final-test" if group == "test" else ""))
        elif metrics["sources"][group] != baseline["sources"].get(group):
            problems.append(f"источники {group} не те, что у CatBoost-бейзлайна")
        elif recorded.get("groups", {}).get(group) != embedded.get("groups", {}).get(group):
            problems.append(f"векторы {group} сняты заново после его обучения")

    if problems:
        raise ValueError(f"{name} {tag}: {'; '.join(problems)} — python -m churn.plus_usr{flag} заново")

    rows = pd.concat(
        [pd.read_parquet(directory / f"{name}.parquet") for name in ("train_rows", "eval_rows")], ignore_index=True
    )

    return {
        "thresholds": {task: float(block["threshold"]) for task, block in metrics["tasks"].items()},
        "rows": {
            (task, group): part.set_index("client_id")[["churn", "score"]].sort_index()
            for (task, group), part in rows.groupby(["task", "group"])
        },
    }


def churn_rows(task: str, group: str) -> pd.DataFrame:
    """
    Строки задачи churn-бейзлайна в группе: client_id, churn и прогноз
    CatBoost этой задачи. Бейзлайн обязан быть собран на текущей
    выгрузке группы, и T строк — совпасть с T группы: отчёт прежней
    генерации для сравнения не годится.
    """

    if task not in CHURN_TASKS:
        raise ValueError(f"задача {task!r} не из {CHURN_TASKS}")

    churn_sources(group)

    name = "train_rows.parquet" if group == "train" else "eval_rows.parquet"

    rows = pd.read_parquet(CHURN_REPORTS / name)
    rows = rows[(rows["task"] == task) & (rows["group"] == group)]

    moment = pd.Timestamp(cutoff(group))

    found = pd.to_datetime(rows["T"], utc=True).unique()

    if len(found) != 1 or found[0] != moment:
        raise ValueError(
            f"churn-бейзлайн группы {group} построен на T {list(found)}, а T группы здесь "
            f"{moment.isoformat()}: сравнение было бы на разных моментах"
        )

    return rows.set_index("client_id")[["churn", "score"]].sort_index()


__all__ = [
    "CHURN_TASKS",
    "COUNT_WINDOWS",
    "USR_CATBOOST",
    "build_table",
    "churn_rows",
    "churn_sources",
    "churn_thresholds",
    "table",
    "usr_catboost",
]
