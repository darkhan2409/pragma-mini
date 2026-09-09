from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.artifacts import write_json, write_text

from .ablation import evaluate_rules, rule_tables
from .compare import build_trainer, excluded_fields
from .data import FixedSplit
from .diagnostics import LEVEL_CLIENT, LEVEL_EXAMPLE, Levels, _field_support
from .history_encoder import ATTENTION_RULES, RULE_FULL
from .trainer import (
    Environment,
    TrainConfig,
    Trainer,
    VALIDATION_SPLITS,
    benchmark,
    run_training,
    store_for,
    truncation_check,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Прежние запуски видели последние 128 событий, то есть около
# полутора месяцев ленты и меньше десятой её части. Здесь лимит
# снят целиком: модель получает всю доступную на cutoff историю,
# набор расширен до 1000 клиентов, а бюджет в шагах заменён
# одной полной эпохой.
#
# Меняются три вещи сразу: объём данных, длина истории и число
# шагов. Разницу нельзя приписывать одной длине контекста, и
# отчёт обязан это говорить, а не подразумевать.
#
# Baseline'ы обучались на своих масках (128 событий, 64
# клиента). Здесь они оцениваются на наборе НОВОГО эксперимента,
# то есть вне своих условий, и это тоже сказано прямо.
# ============================================================


CLIENT_UNIVERSE = 1000

ABLATION_SPLITS: tuple[str, ...] = ("val_time",)

CAVEAT = (
    "относительно прежнего запуска изменились три вещи сразу: объём обучающих данных, "
    "длина истории и число шагов. Разницу нельзя приписывать одной только длине контекста"
)

BASELINE_NOTE = (
    "baseline обучался на своих масках (128 событий, 64 клиента) и здесь оценивается на "
    "наборе нового эксперимента, то есть вне условий, в которых обучался"
)


# ============================================================
# НАБОР КЛИЕНТОВ
# ============================================================


def client_list(env: Environment, config: TrainConfig) -> dict:
    """
    Фактический состав набора и его распределение по сплитам.
    """

    from src.tokenizer.dataset import TokenizedDataset

    from .data import select_clients

    universe = config.client_universe

    wanted = None if universe is None else range(int(universe))

    by_split: dict[str, list[int]] = {}

    for name in ("train", "val_client", "test_client"):

        table = TokenizedDataset(env.root, name, vocab_dir=env.vocab_dir).examples

        by_split[name] = select_clients(table, None, wanted)

    every = sorted({value for values in by_split.values() for value in values})

    return {
        "requested": universe,
        "rule": "клиенты с ID меньше client_universe; распределение по сплитам из существующего разбиения",
        "n_clients": len(every),
        "by_split": {name: len(values) for name, values in by_split.items()},
        "clients": every,
        "train_clients": by_split["train"],
    }


# ============================================================
# ЗАПУСК
# ============================================================


def run_full_history(
    env: Environment,
    out_dir: Path,
    config: TrainConfig,
    baselines: dict[str, Path],
    device: str = "cpu",
    ablation_splits: tuple[str, ...] = ABLATION_SPLITS,
    rules: tuple[str, ...] = ATTENTION_RULES,
    n_boot: int = 2000,
    warmup: int = 5,
    measured: int = 20,
    skip_benchmark: bool = False,
    quiet: bool = False,
) -> dict:
    """
    Полные истории, одна эпоха, сравнение с прежними checkpoint'ами.
    """

    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if config.max_events_per_history is not None:
        raise ValueError(
            "полный контекст требует max_events_per_history=None; "
            f"получено {config.max_events_per_history}"
        )

    if config.epochs != 1:
        raise ValueError(f"этот запуск делает ровно одну эпоху, получено epochs={config.epochs}")

    # --------------------------------------------------------
    # СОСТАВ НАБОРА И PREFLIGHT
    # --------------------------------------------------------

    clients = client_list(env, config)

    write_json(out_dir / "clients.json", clients)

    if not quiet:
        print(f"клиентов в наборе: {clients['n_clients']}  " +
              ", ".join(f"{name} {value}" for name, value in clients["by_split"].items()))

    started = time.perf_counter()

    preflight = {}

    for name in ("train", *VALIDATION_SPLITS):

        store = store_for(env, config, name)

        preflight[name] = {**truncation_check(store, config), **store.summary()}

        if not quiet:
            item = preflight[name]
            print(f"  {name:<12s} клиентов {item['clients']:>5}  примеров {item['n_examples']:>7}  "
                  f"обрезано {item['n_truncated']}  длина p50 {item['lengths']['p50']:>5} "
                  f"p90 {item['lengths']['p90']:>5} p95 {item['lengths']['p95']:>5} "
                  f"p99 {item['lengths']['p99']:>5} max {item['lengths']['max']:>5}")

    preflight_seconds = time.perf_counter() - started

    broken = [name for name, item in preflight.items() if not item["passed"]]

    if broken:
        raise ValueError(f"обрезка не отключена на сплитах {broken}")

    write_json(out_dir / "preflight.json", {"seconds": preflight_seconds, "splits": preflight})

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    measurement = None

    if not skip_benchmark:

        if not quiet:
            print()
            print("замер полного шага на отдельной модели")
            print()

        measurement = benchmark(
            env,
            config,
            out_dir / "benchmark",
            warmup=warmup,
            measured=measured,
            device=device,
            quiet=quiet,
        )

    # --------------------------------------------------------
    # ЭПОХА
    # --------------------------------------------------------

    if not quiet:
        print()
        print("обучение: одна полная эпоха")
        print()

    training = run_training(env, config, out_dir / "run", device=device, quiet=quiet, preflight=False)

    trained = out_dir / "run" / "last.pt"

    # --------------------------------------------------------
    # ОЦЕНКА
    # --------------------------------------------------------

    if not quiet:
        print()
        print("оценка на полных историях")
        print()

    exclude = excluded_fields(env.table)

    splits, _ = _build_splits(env, config)

    levels = {name: Levels(split, n_boot, config.val_seed) for name, split in splits.items()}

    models: dict[str, Trainer] = {
        "epoch": build_trainer(env, config, trained, device),
    }

    for label, path in baselines.items():

        payload = torch.load(path, map_location="cpu", weights_only=False)

        models[label] = build_trainer(
            env, TrainConfig.from_dict(payload["train_config"]), path, device
        )

    # Модель до обучения это та же архитектура с тем же seed и
    # тем же способом инициализации, без загрузки весов.
    models["untrained"] = Trainer(config, env.tokenizer, env.table, env.unigram, device)

    order = ["untrained", "epoch", *baselines]

    evaluated: dict[str, dict] = {}

    for label in order:

        reports = models[label].evaluate(splits, exclude=exclude, keep_units=True)

        reports.pop("_seconds")

        evaluated[label] = reports

        if not quiet:
            for name in splits:
                item = reports[name]
                print(f"  {label:<18s}{name:<12s} CE {item['field_balanced_ce']:.4f}  "
                      f"NCE {_text(item['mean_nce_gain'])}  Acc {_text(item['accuracy'])}  "
                      f"целей {item['n_targets']:,}".replace(",", " "))

    # --------------------------------------------------------
    # ИНТЕРВАЛЫ
    # --------------------------------------------------------

    contrasts: dict[str, dict] = {}

    for name in splits:

        pair = levels[name]

        samples = {
            (label, scope): pair.samples(evaluated[label]["_accumulators"][name], subset)
            for label in order
            for scope, subset in (("all_fields", frozenset()), ("subset", exclude))
        }

        contrasts[name] = {
            f"epoch_vs_{label}": {
                scope: pair.combine(
                    [(-1.0, samples[(label, scope)]), (1.0, samples[("epoch", scope)])]
                )
                for scope in ("all_fields", "subset")
            }
            for label in order
            if label != "epoch"
        }

    # Поля сравниваются с первым baseline: у него тот же
    # objective, и от новой модели он отличается только длиной
    # контекста и объёмом данных.
    field_baseline = next(iter(baselines), "untrained")

    fields = {
        name: _field_support(
            evaluated[field_baseline][name],
            evaluated["epoch"][name],
            evaluated["epoch"]["_accumulators"][name],
            levels[name].clusters,
            30,
        )
        for name in splits
    }

    # --------------------------------------------------------
    # ДИАГНОСТИКА ВНИМАНИЯ
    # --------------------------------------------------------

    ablation = _ablation(
        models["epoch"], splits, levels, list(rules), exclude, ablation_splits, n_boot, quiet
    )

    # --------------------------------------------------------

    report = {
        "mode": "full_history",
        "device": str(models["epoch"].device),
        "precision": models["epoch"].precision,
        "config": config.as_dict(),
        "clients": {key: value for key, value in clients.items() if key != "clients"},
        "preflight": preflight,
        "benchmark": measurement,
        "training": {
            "counters": training["counters"],
            "budget": training["budget"],
            "train_seconds": training["train_seconds"],
            "peak_gpu_mb": training["peak_gpu_mb"],
            "best": training["best"],
            "masking": training["masking_diagnostics"],
        },
        "splits": {name: split.description() for name, split in splits.items()},
        "units": {
            name: {"n_clients": levels[name].n_clients, "n_examples": levels[name].n_examples}
            for name in splits
        },
        "models": order,
        "baselines": {label: str(path) for label, path in baselines.items()},
        "metrics": {
            label: {name: evaluated[label][name] for name in splits} for label in order
        },
        "contrast": contrasts,
        "field_baseline": field_baseline,
        "fields": fields,
        "ablation": ablation,
        "excluded_fields": sorted(exclude),
        "checkpoints": {
            "last": str(trained),
            "best": str(out_dir / "run" / "best.pt"),
        },
        "caveat": CAVEAT,
        "baseline_note": BASELINE_NOTE,
    }

    write_json(out_dir / "full_history.json", report)
    write_text(out_dir / "full_history.md", render_full_history(report))
    write_text(out_dir / "ablation.md", render_full_ablation(report))

    if not quiet:
        print()
        print(render_full_history(report))

    return report


def _build_splits(env: Environment, config: TrainConfig):
    """
    Фиксированные наборы на полных историях.
    """

    masking = config.masking(seed=config.val_seed)

    splits: dict[str, FixedSplit] = {}
    stores: dict[str, dict] = {}

    for name in VALIDATION_SPLITS:

        store = store_for(env, config, name)

        splits[name] = FixedSplit.build(
            name=name,
            store=store,
            vocab=env.vocab,
            table=env.table,
            model_config=Trainer(config, env.tokenizer, env.table, env.unigram, "cpu").model_config,
            masking=masking,
            max_events=config.max_events_per_history,
            batch_size=config.eval_batch_size,
        )

        stores[name] = store.summary()

    return splits, stores


def _ablation(trainer, splits, levels, rules, exclude, wanted, n_boot, quiet) -> dict:
    """
    Правила внимания на полных историях, интервалы по клиентам.
    """

    chosen = {name: splits[name] for name in wanted if name in splits}

    if not chosen:
        return {"splits": [], "reason": "ни один сплит не выбран для диагностики внимания"}

    results = evaluate_rules(trainer, chosen, rules, exclude=exclude, keep_units=True, quiet=quiet)

    aggregates, fields, silent = rule_tables(results, rules, chosen)

    contrast: dict[str, dict] = {}

    for name in chosen:

        pair = levels[name]

        samples = {
            (rule, scope): pair.samples(results[rule]["accumulators"][name], subset)
            for rule in rules
            for scope, subset in (("all_fields", frozenset()), ("subset", exclude))
        }

        contrast[name] = {
            rule: {
                scope: pair.combine(
                    [(-1.0, samples[(RULE_FULL, scope)]), (1.0, samples[(rule, scope)])]
                )
                for scope in ("all_fields", "subset")
            }
            for rule in rules
            if rule != RULE_FULL
        }

    return {
        "splits": list(chosen),
        "skipped_splits": [name for name in splits if name not in chosen],
        "reason": (
            "структурная маска внимания на полной длине стоит 468 МБ при batch 2 и почти "
            "гигабайт при batch 4, поэтому диагностика идёт на одном сплите"
        ),
        "aggregates": aggregates,
        "fields": fields,
        "fields_without_targets": silent,
        "contrast": contrast,
    }


# ============================================================
# ОТЧЁТЫ
# ============================================================


def _text(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def _interval(item: dict, digits: int = 4) -> str:
    return f"{item['estimate']:+.{digits}f} [{item['ci_low']:+.{digits}f}, {item['ci_high']:+.{digits}f}]"


ROWS: tuple[tuple[str, str], ...] = (
    ("field-balanced CE", "field_balanced_ce"),
    ("та же CE у unigram", "field_balanced_ce_unigram"),
    ("средний NCE gain", "mean_nce_gain"),
    ("accuracy", "accuracy"),
    ("accuracy unigram", "unigram_accuracy"),
    ("средний macro F1", "macro_f1_mean"),
)

TITLES = {
    "untrained": "новая до обучения",
    "epoch": "новая после эпохи",
}


def _title(label: str) -> str:
    return TITLES.get(label, f"baseline {label}")


def render_full_history(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Полные истории: одна эпоха на 1000 клиентах")
    lines.append("")
    lines.append(
        f"Устройство {report['device']}, precision {report['precision']}. "
        f"Masking {report['config']['masking_mode']}: token {report['config']['token_rate']}, "
        f"event {report['config']['event_rate']}, key {report['config']['key_rate']}. "
        f"Лимит истории {report['config']['max_events_per_history']}."
    )
    lines.append("")
    lines.append(f"**Оговорка:** {report['caveat']}.")
    lines.append("")

    # --------------------------------------------------------

    lines.append("## Набор и отсутствие обрезки")
    lines.append("")
    lines.append(
        f"Клиентов {report['clients']['n_clients']}: "
        + ", ".join(f"{name} {value}" for name, value in report["clients"]["by_split"].items())
        + f". Правило отбора: {report['clients']['rule']}."
    )
    lines.append("")
    lines.append("| сплит | клиентов | примеров | обрезано | p50 | p90 | p95 | p99 | max | событий |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for name, item in report["preflight"].items():
        lengths = item["lengths"]
        lines.append(
            f"| {name} | {item['clients']} | {item['n_examples']} | {item['n_truncated']} | "
            f"{lengths['p50']} | {lengths['p90']} | {lengths['p95']} | {lengths['p99']} | "
            f"{lengths['max']} | {lengths['total_events']:,} |".replace(",", " ")
        )

    lines.append("")
    lines.append(
        "У каждого примера использованная длина равна исходной, обрезанных нет ни одного."
    )
    lines.append("")

    # --------------------------------------------------------

    budget = report["training"]["budget"]
    counters = report["training"]["counters"]

    lines.append("## Обучение")
    lines.append("")
    lines.append(
        f"Одна эпоха: {budget['epoch_examples']} примеров, {budget['micro_batches_done']} batch'ей, "
        f"{counters['n_steps']} шагов оптимизатора, пропущено {counters['n_skipped']}. "
        f"Время {report['training']['train_seconds'] / 60:.1f} мин"
        + (f", пик GPU {report['training']['peak_gpu_mb']:.0f} МБ." if report["training"]["peak_gpu_mb"] else ".")
    )

    if report["training"]["masking"]:
        masking = report["training"]["masking"]
        lines.append("")
        lines.append(
            f"Скрыто {masking['masked_fraction'] * 100:.1f} % доступных значений, "
            f"в среднем {masking['masked']:.0f} целей на шаг из {masking['eligible']:.0f} доступных."
        )

    lines.append("")

    # --------------------------------------------------------

    lines.append("## Метрики на одних целях")
    lines.append("")
    lines.append(f"Каждый {report['baseline_note']}.")
    lines.append("")

    for name in report["splits"]:

        description = report["splits"][name]
        units = report["units"][name]

        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"Клиентов {units['n_clients']}, примеров {units['n_examples']}, "
            f"целей {description['n_targets']}, скрыто "
            f"{description['masked_fraction'] * 100:.1f} % доступных значений."
        )
        lines.append("")

        header = "| метрика | " + " | ".join(_title(label) for label in report["models"]) + " |"

        lines.append(header)
        lines.append("|" + "---|" * (len(report["models"]) + 1))

        for label, key in ROWS:
            cells = " | ".join(_text(report["metrics"][model][name][key]) for model in report["models"])
            lines.append(f"| {label} | {cells} |")

        subset = " | ".join(
            _text(report["metrics"][model][name]["subset"]["field_balanced_ce"])
            for model in report["models"]
        )

        lines.append(f"| CE без исключённых полей | {subset} |")

        subset_gain = " | ".join(
            _text(report["metrics"][model][name]["subset"]["mean_nce_gain"])
            for model in report["models"]
        )

        lines.append(f"| NCE без исключённых полей | {subset_gain} |")

        lines.append("")

        lines.append("Парный bootstrap по клиентам, ΔCE относительно модели после эпохи:")
        lines.append("")
        lines.append("| сравнение | все поля | без исключённых |")
        lines.append("|---|---|---|")

        for key, item in report["contrast"][name].items():
            lines.append(
                f"| {key} | {_interval(item['all_fields'][LEVEL_CLIENT])} | "
                f"{_interval(item['subset'][LEVEL_CLIENT])} |"
            )

        lines.append("")

    # --------------------------------------------------------

    baseline = report.get("field_baseline", "baseline")

    lines.append("## По полям после эпохи")
    lines.append("")
    lines.append(f"Сравнение с `{baseline}`.")
    lines.append("")

    for name, rows in report["fields"].items():

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"| поле | целей | клиентов | CE {baseline} | CE после эпохи | ΔCE | поддержка |")
        lines.append("|---|---|---|---|---|---|---|")

        for row in rows:
            lines.append(
                f"| {row['field']} | {row['n_targets']} | {row['n_clients_with_targets']} | "
                f"{row['ce_old']:.3f} | {row['ce_new']:.3f} | {row['delta_ce']:+.3f} | "
                f"{'мало' if row['support'] == 'low_support' else ''} |"
            )

        lines.append("")

    lines.append(f"Исключаемые из второго агрегата поля: {', '.join(report['excluded_fields'])}.")

    return "\n".join(lines) + "\n"


def render_full_ablation(report: dict) -> str:

    ablation = report["ablation"]

    lines: list[str] = []

    lines.append("# Роль истории на полном контексте")
    lines.append("")

    if not ablation["splits"]:
        lines.append(ablation["reason"] + ".")
        return "\n".join(lines) + "\n"

    lines.append(
        f"Диагностика на {', '.join(ablation['splits'])}. "
        f"Пропущено: {', '.join(ablation['skipped_splits']) or 'ничего'}. "
        f"Причина: {ablation['reason']}."
    )
    lines.append("")

    for name, rules in ablation["contrast"].items():

        lines.append(f"## {name}")
        lines.append("")
        lines.append("| запрет | ΔCE, все поля | ΔCE без исключённых |")
        lines.append("|---|---|---|")

        for rule, item in rules.items():
            lines.append(
                f"| `{rule}` | {_interval(item['all_fields'][LEVEL_CLIENT])} | "
                f"{_interval(item['subset'][LEVEL_CLIENT])} |"
            )

        lines.append("")

        by_rule = {row["rule"]: row for row in ablation["aggregates"][name]}

        lines.append(
            "Прямое event-to-event внимание это потеря от `events_via_profile`, обмен через "
            "профиль это разница между `events_isolated` и `events_via_profile`."
        )
        lines.append("")
        lines.append("| вариант | field-balanced CE | ΔCE к full |")
        lines.append("|---|---|---|")

        for rule, row in by_rule.items():
            lines.append(
                f"| `{rule}` | {_text(row['field_balanced_ce'])} | "
                f"{_signed(row['delta_field_balanced_ce'])} |"
            )

        lines.append("")

    return "\n".join(lines) + "\n"
