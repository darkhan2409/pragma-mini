"""
Шаг 0 плана 2026-11-03: разброс CatBoost по seed на churn_active90 —
шумовой пол для правила принятия (README §13).

Только читает и печатает; ничего не пишет, test не читает. CatBoost есть
только в окружении churn_baseline, запуск из его каталога:

    cd churn_baseline
    PYTHONPATH=.. .venv/bin/python ../audit/2026-11-03-pretrain-fixes/catboost_seeds.py --tag ref-2026-10-09

Повторяет два потока бейзлайна, меняя только PARAMS["random_seed"] (разбиение
inner_train / inner_holdout остаётся на seed 42): строки и признаки как в
churn.train.train, признаки с [USR] как в churn.plus_usr.train_plus_usr, обучение
тем же churn.train.fit_task. Seed 42 печатается первым как контроль: на выгрузке
2026-10-09 он обязан дать PR-AUC val 0.570 (catboost) и 0.586 (catboost_plus_usr);
в среднее, размах и разброс Δ контроль не входит.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from catboost import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "churn_baseline"))

from churn.build import KEYS  # noqa: E402
from churn.config import DATA_DIR, FUTURE_DIR, RAW_DIR, SEED, groups  # noqa: E402
from churn.plus_usr import with_usr  # noqa: E402
from churn.profile import CATEGORICAL  # noqa: E402
from churn.target import task_rows  # noqa: E402
from churn.train import PARAMS, check_fresh, check_groups, evaluate, fit_task, load  # noqa: E402

TASK = "churn_active90"
GROUPS = groups(final_test=False)  # train и val: test здесь не читается


def pr_auc_val(rows: dict, features: list[str], categorical: list[str], seed: int) -> float:
    """PR-AUC val модели, обученной fit_task с данным seed CatBoost."""
    PARAMS["random_seed"] = seed
    model, fitted = fit_task(rows["train"], features, categorical)
    val = rows["val"]
    score = model.predict_proba(Pool(val[features], cat_features=categorical))[:, 1]
    return evaluate(val["churn"].to_numpy(), score, fitted["threshold"])["pr_auc"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Разброс PR-AUC val CatBoost по seed: catboost и catboost_plus_usr")
    parser.add_argument("--tag", required=True, help="тег векторов PRAGMA: data/13_downstream/<тег>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46, 47], help="seed CatBoost")
    args = parser.parse_args(argv)

    directory = ROOT / "data" / "13_downstream" / args.tag
    if SEED in args.seeds:
        raise SystemExit(f"seed {SEED} — контроль, он печатается первым и в --seeds не нужен")

    sources = {group: check_fresh(group, DATA_DIR, RAW_DIR, FUTURE_DIR) for group in GROUPS}
    frames = {group: load(group, DATA_DIR) for group in GROUPS}
    check_groups(frames)

    columns = [name for name in frames["train"].columns if name not in KEYS]
    categorical = [name for name in columns if name in CATEGORICAL]
    for group, frame in frames.items():
        if [name for name in frame.columns if name not in KEYS] != columns:
            raise ValueError(f"признаки группы {group} не совпадают с train")

    joined, usr = {}, None
    for group, frame in frames.items():
        joined[group], names = with_usr(directory, group, sources[group], frame)
        if usr is not None and names != usr:
            raise ValueError(f"у векторов {group} другая длина [USR]")
        usr = names

    base = {group: task_rows(TASK, frame) for group, frame in frames.items()}
    plus = {group: task_rows(TASK, frame) for group, frame in joined.items()}

    print(f"{TASK}, PR-AUC val; векторы {args.tag}")
    print(f"{'seed':>5} {'catboost':>9} {'plus_usr':>9} {'Δ':>8}")
    deltas = []
    for seed in [SEED, *args.seeds]:
        first = pr_auc_val(base, columns, categorical, seed)
        second = pr_auc_val(plus, columns + usr, categorical, seed)
        label = f"{seed}*" if seed == SEED else str(seed)
        print(f"{label:>5} {first:>9.3f} {second:>9.3f} {second - first:>+8.3f}", flush=True)
        if seed != SEED:
            deltas.append(second - first)

    delta = np.array(deltas)
    print(f"\n* seed {SEED} — контроль (ожидается 0.570 и 0.586), в статистику не входит")
    print(f"Δ по seed {args.seeds}: среднее {delta.mean():+.3f}, размах {delta.max() - delta.min():.3f}, "
          f"стандартное отклонение {delta.std(ddof=1) if len(delta) > 1 else float('nan'):.3f} (ddof=1)")


if __name__ == "__main__":
    main()
