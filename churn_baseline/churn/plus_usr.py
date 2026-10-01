from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

from .build import KEYS
from .config import DATA_DIR, FUTURE_DIR, MODELS_DIR, RAW_DIR, REPORTS_DIR, groups
from .profile import CATEGORICAL
from .target import TASKS, task_rows
from .train import PARAMS, check_fresh, check_groups, evaluate, fit_task, load, sha256, show


# ============================================================
# ПОЛНЫЙ X + [USR] PRAGMA → CATBOOST
# ============================================================
#
#   python -m churn.plus_usr --embeddings ../data/13_downstream/w4-b0
#   python -m churn.plus_usr --embeddings ../data/13_downstream/w4-b0 --usr-only
#
# Третий вариант сравнения: тот же полный X, на котором учится
# CatBoost-бейзлайн, плюс вектор [USR] модели PRAGMA на T строки —
# usr_0 … usr_{d−1}, числами, без стандартизации. Даёт ли [USR]
# сигнал сверх handcrafted-признаков.
#
# --usr-only — диагностика: CatBoost только на usr_*, без единого
# handcrafted-признака и без категориальных. Нет ли в [USR] сигнала,
# который линейная проба не извлекает.
#
# Всё остальное — как у бейзлайна: те же строки, T и метки задач,
# те же категориальные признаки, тот же fit_task (ранняя остановка
# и порог на inner_holdout train, обучение на всём train). val только
# оценивается, test — только с --final-test.
#
# Векторы — файл данных data/13_downstream/<тег>/<группа>.parquet
# (python -m src.downstream.embed); код PRAGMA не импортируется.
# Они обязаны быть сняты с той же выгрузки, из которой собраны
# признаки, и на тот же T, а у каждой строки — ровно один вектор.
# ============================================================


USR = "usr"

PLUS_USR = "plus_usr"

USR_ONLY = "usr_only"


def with_usr(directory: Path, group: str, sources: dict, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Строки группы с колонками usr_* из векторов PRAGMA, в прежнем
    порядке. Любое расхождение источников — отказ.
    """
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    recorded = meta.get("groups", {}).get(group)
    if recorded is None:
        raise ValueError(f"{directory}: нет векторов группы {group}")

    exported = (sources["feature_history_events_sha256"], sources["feature_profile_sha256"])
    if (recorded.get("raw_events_sha256"), recorded.get("raw_profile_sha256")) != exported:
        raise ValueError(f"векторы {group} сняты не с той выгрузки, из которой собраны признаки")

    moment = pd.Timestamp(sources["T"])
    if pd.Timestamp(recorded["cutoff"]) != moment or not (pd.to_datetime(frame["T"], utc=True) == moment).all():
        raise ValueError(f"векторы {group} сняты на {recorded['cutoff']}, а T строк — {sources['T']}")

    table = pd.read_parquet(directory / f"{group}.parquet", columns=["client_id", "cutoff", USR])
    if table["client_id"].duplicated().any():
        raise ValueError(f"в векторах {group} клиент повторяется")
    if not (pd.to_datetime(table["cutoff"], utc=True) == moment).all():
        raise ValueError(f"в векторах {group} есть строки не на T {sources['T']}")

    dims = table[USR].map(len).unique()
    if len(dims) != 1:
        raise ValueError(f"в векторах {group} разная длина [USR]: {sorted(dims)}")

    names = [f"usr_{index}" for index in range(int(dims[0]))]
    wide = pd.DataFrame(np.stack(table[USR].to_numpy()), columns=names)
    wide.insert(0, "client_id", table["client_id"].to_numpy())

    joined = frame.merge(wide, on="client_id", how="left", validate="one_to_one")
    missing = int(joined[names[0]].isna().sum())
    if missing or len(joined) != len(frame) or not joined["client_id"].equals(frame["client_id"]):
        raise ValueError(f"у {missing} строк {group} нет вектора [USR]")

    return joined, names


def train_plus_usr(
    embeddings: Path,
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    raw_dir: Path = RAW_DIR,
    future_dir: Path = FUTURE_DIR,
    final_test: bool = False,
    usr_only: bool = False,
) -> dict:
    embeddings = Path(embeddings)
    used = groups(final_test)

    sources = {group: check_fresh(group, data_dir, raw_dir, future_dir) for group in used}
    frames = {group: load(group, data_dir) for group in used}
    check_groups(frames)

    columns = [name for name in frames["train"].columns if name not in KEYS]
    categorical = [name for name in columns if name in CATEGORICAL]
    for group, frame in frames.items():
        if [name for name in frame.columns if name not in KEYS] != columns:
            raise ValueError(f"признаки группы {group} не совпадают с train")

    joined, usr = {}, None
    for group, frame in frames.items():
        joined[group], names = with_usr(embeddings, group, sources[group], frame)
        if usr is not None and names != usr:
            raise ValueError(f"у векторов {group} другая длина [USR]")
        usr = names

    # Строки задач отбираются по полному X и в --usr-only, модель его
    # не видит.
    handcrafted, categorical = ([], []) if usr_only else (columns, categorical)
    features = handcrafted + usr
    kind = USR_ONLY if usr_only else PLUS_USR
    meta = json.loads((embeddings / "meta.json").read_text(encoding="utf-8"))
    tag = meta["tag"]
    evaluated = tuple(group for group in used if group != "train")

    reports = reports_dir / kind / tag
    models = models_dir / kind / tag
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"final_test": final_test, "tasks": {}}
    train_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []

    for task in TASKS:
        rows = {group: task_rows(task, frame) for group, frame in joined.items()}

        model, fitted = fit_task(rows["train"], features, categorical)
        model.save_model(models / f"catboost_{task}.cbm")

        scores = {
            group: model.predict_proba(Pool(frame[features], cat_features=categorical))[:, 1]
            for group, frame in rows.items()
        }

        fitted["groups"] = {
            group: evaluate(rows[group]["churn"].to_numpy(), scores[group], fitted["threshold"])
            for group in evaluated
        }
        metrics["tasks"][task] = fitted

        def part(group: str) -> pd.DataFrame:
            frame = rows[group][list(KEYS)].assign(score=scores[group])
            frame.insert(0, "task", task)
            return frame

        train_parts.append(part("train"))
        eval_parts += [part(group) for group in evaluated]

    pd.concat(eval_parts, ignore_index=True).to_parquet(reports / "eval_rows.parquet", index=False)
    pd.concat(train_parts, ignore_index=True).to_parquet(reports / "train_rows.parquet", index=False)

    metrics.update(
        {
            "params": PARAMS,
            "features": {"handcrafted": len(handcrafted), "usr": len(usr), "total": len(features)},
            "categorical": categorical,
            "embeddings": {
                "tag": tag,
                "checkpoint": meta.get("checkpoint"),
                "epoch": meta.get("epoch"),
                "groups": {group: meta["groups"][group] for group in used},
            },
            "eval_rows_sha256": sha256(reports / "eval_rows.parquet"),
            "train_rows_sha256": sha256(reports / "train_rows.parquet"),
            "sources": sources,
        }
    )
    (reports / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CatBoost на полном X бейзлайна и [USR] PRAGMA: обучение на train, оценка на val"
    )
    parser.add_argument("--embeddings", type=Path, required=True, help="каталог векторов, data/13_downstream/<тег>")
    parser.add_argument("--final-test", action="store_true", help="финальная оценка: оценить и test")
    parser.add_argument("--usr-only", action="store_true", help="диагностика: CatBoost только на [USR]")
    args = parser.parse_args(argv)
    metrics = train_plus_usr(args.embeddings, final_test=args.final_test, usr_only=args.usr_only)
    print(f"признаков: handcrafted {metrics['features']['handcrafted']} + usr {metrics['features']['usr']}")
    print(show(metrics))


if __name__ == "__main__":
    main()
