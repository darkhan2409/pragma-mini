from __future__ import annotations

import json

import pandas as pd

from src.preprocessing.rawdata import read_manifest
from src.preprocessing.settings import raw_group_dir

from .settings import CHURN_FUTURE, CHURN_REPORTS, FUTURE_LABEL_GROUPS, cutoff


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
# Там же — прогноз CatBoost на полном X и [USR] этих векторов
# (catboost_plus_usr, python -m churn.plus_usr).
# ============================================================


CHURN_TASKS = ("churn_active90",)

# Каталог catboost_plus_usr в отчётах churn-бейзлайна: plus_usr/<тег>.
PLUS_USR = "plus_usr"


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


def plus_usr(tag: str, used: tuple[str, ...], embedded: dict) -> dict | None:
    """
    catboost_plus_usr churn-бейзлайна (полный X + [USR] векторов тега →
    CatBoost): порог по задаче и строки (churn, score) по задаче и
    группе. None — для этих векторов он не обучен.

    embedded — meta.json векторов. Годится только CatBoost, обученный
    ровно на них: тот же тег и чекпойнт и те же записи групп (у снятых
    заново векторов запись другая, хотя бы по времени съёма), — и на
    тех же источниках строк, что и сам бейзлайн.
    """

    directory = CHURN_REPORTS / PLUS_USR / tag

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
        raise ValueError(f"catboost_plus_usr {tag}: {'; '.join(problems)} — python -m churn.plus_usr заново")

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
    "PLUS_USR",
    "churn_rows",
    "churn_sources",
    "churn_thresholds",
    "plus_usr",
]
