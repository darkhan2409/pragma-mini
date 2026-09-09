from __future__ import annotations

import time
from pathlib import Path

import torch

from src.preprocessing.artifacts import write_json, write_text
from src.tokenizer.config import IncompatibleArtifactsError

from .checkpoint import load_checkpoint
from .history_encoder import ATTENTION_RULES, RULE_FULL, RULE_NOTES
from .trainer import Environment, TrainConfig, Trainer, build_validation


# ============================================================
# ИДЕЯ
# ============================================================
#
# Обученная модель лучше unigram, и вопрос в том, за счёт чего.
# Голова видит три представления: локальное состояние токена из
# Event Encoder и два выхода History Encoder. Только History
# Encoder смешивает информацию между событиями и с профилем,
# поэтому именно его внимание и ограничивается.
#
# Ничего не обучается. Тот же checkpoint, те же фиксированные
# validation-маски (сверяются по sha256 целей), та же точность.
# Baseline пересчитывается в этом же запуске: сравнивать
# сегодняшние числа со вчерашней строкой отчёта нельзя, у bf16
# свои разряды.
#
# Все дельты считаются к пересчитанному full. Отрицательная
# дельта CE означала бы, что запрет внимания помог, и её тоже
# надо показывать как есть.
# ============================================================


def _aggregate_row(rule: str, metrics: dict, baseline: dict | None) -> dict:

    row = {
        "rule": rule,
        "n_targets": metrics["n_targets"],
        "field_balanced_ce": metrics["field_balanced_ce"],
        "token_weighted_ce": metrics["token_weighted_ce"],
        "mean_nce_gain": metrics["mean_nce_gain"],
        "accuracy": metrics["accuracy"],
        "macro_f1_mean": metrics["macro_f1_mean"],
    }

    for name in ("field_balanced_ce", "token_weighted_ce", "mean_nce_gain", "accuracy", "macro_f1_mean"):

        mine, theirs = row[name], None if baseline is None else baseline[name]

        row[f"delta_{name}"] = None if mine is None or theirs is None else mine - theirs

    return row


def _field_rows(reports: dict[str, dict], rules: list[str]) -> tuple[list[dict], list[str]]:
    """
    По полю: метрики каждого варианта и дельты к full.
    """

    baseline = reports[RULE_FULL]["fields"]

    order = [item["key_id"] for item in baseline]

    rows: list[dict] = []
    quiet: list[str] = []

    by_rule = {rule: {item["key_id"]: item for item in reports[rule]["fields"]} for rule in rules}

    for key_id in order:

        anchor = by_rule[RULE_FULL][key_id]

        if anchor["n_targets"] == 0:
            quiet.append(f"{anchor['field']} ({anchor['status']})")
            continue

        row = {
            "field": anchor["field"],
            "key_id": key_id,
            "kind": anchor["kind"],
            "n_targets": anchor["n_targets"],
            "n_candidates": anchor["n_candidates"],
            "ce_unigram": anchor["ce_unigram"],
        }

        for name in ("ce_model", "nce_gain", "accuracy", "macro_f1"):

            row[name] = {rule: by_rule[rule][key_id][name] for rule in rules}

            reference = row[name][RULE_FULL]

            row[f"delta_{name}"] = {
                rule: None
                if row[name][rule] is None or reference is None
                else row[name][rule] - reference
                for rule in rules
            }

        rows.append(row)

    return rows, quiet


def evaluate_rules(
    trainer,
    splits: dict,
    rules: list[str],
    exclude: frozenset[str] = frozenset(),
    keep_units: bool = False,
    quiet: bool = True,
    label: str = "",
) -> dict[str, dict]:
    """
    Один checkpoint под каждым правилом на одних и тех же масках.

    full считается обычным путём, без маски: baseline обязан
    быть тем же вычислением, которым модель обучалась.
    """

    results: dict[str, dict] = {}

    for rule in rules:

        reports = trainer.evaluate(
            splits,
            attention_rule=None if rule == RULE_FULL else rule,
            exclude=exclude,
            keep_units=keep_units,
        )

        seconds = reports.pop("_seconds")

        accumulators = reports.pop("_accumulators", None)

        results[rule] = {"seconds": seconds, "metrics": reports, "accumulators": accumulators}

        if not quiet:
            summary = "  ".join(
                f"{name} CE {reports[name]['field_balanced_ce']:.4f}" for name in sorted(reports)
            )
            print(f"  {label}{rule:<20s}{summary}   {seconds:.1f} с")

    return results


def rule_tables(results: dict[str, dict], rules: list[str], split_names) -> tuple[dict, dict, dict]:
    """
    Агрегаты и таблицы по полям с дельтами к full.
    """

    aggregates: dict[str, list[dict]] = {}
    fields: dict[str, list[dict]] = {}
    silent: dict[str, list[str]] = {}

    for name in split_names:

        reports = {rule: results[rule]["metrics"][name] for rule in rules}

        aggregates[name] = [
            _aggregate_row(rule, reports[rule], reports[RULE_FULL]) for rule in rules
        ]

        rows, quiet_fields = _field_rows(reports, rules)

        rows.sort(key=_sort_key(rules))

        fields[name] = rows
        silent[name] = quiet_fields

    return aggregates, fields, silent


def _sort_key(rules: list[str]):
    """
    Сортировка по потере от самого строгого доступного запрета.
    """

    probe = next((rule for rule in rules if rule != RULE_FULL), RULE_FULL)

    def key(row: dict):
        value = row["delta_ce_model"].get(probe)
        return -(value if value is not None else 0.0)

    return key


# ============================================================
# ЗАПУСК
# ============================================================


def run_ablation(
    env: Environment,
    checkpoint: Path,
    out_dir: Path,
    device: str = "cpu",
    rules: tuple[str, ...] = ATTENTION_RULES,
    quiet: bool = False,
) -> dict:
    """
    Один checkpoint, несколько правил внимания, одни и те же цели.
    """

    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    rules = list(rules)

    if RULE_FULL not in rules:
        raise ValueError("без full сравнивать не с чем")

    unknown = [rule for rule in rules if rule not in ATTENTION_RULES]

    if unknown:
        raise ValueError(f"неизвестные правила {unknown}, ожидались из {ATTENTION_RULES}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    config = TrainConfig.from_dict(payload["train_config"])

    if abs(config.epsilon - env.unigram.epsilon) > 0:
        raise IncompatibleArtifactsError(
            f"unigram собран с epsilon {env.unigram.epsilon}, а checkpoint обучался с {config.epsilon}"
        )

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, device)

    started = time.perf_counter()

    splits, stores = build_validation(env, config, trainer.model_config)

    load_seconds = time.perf_counter() - started

    descriptions = {name: split.description() for name, split in splits.items()}

    # Проверяет и словарь, и то, что маски validation те же самые.
    load_checkpoint(
        checkpoint,
        backbone=trainer.backbone,
        head=trainer.head,
        model_config=trainer.model_config.as_dict(),
        artifacts=env.hashes,
        splits=descriptions,
        restore_random=False,
    )

    # --------------------------------------------------------

    results = evaluate_rules(trainer, splits, rules, quiet=quiet)

    # --------------------------------------------------------
    # СВЕРКА С СОХРАНЁННЫМ BASELINE
    # --------------------------------------------------------

    stored = (payload.get("metrics") or {}).get("metrics") or {}

    baseline_check = {
        "step": (payload.get("metrics") or {}).get("step"),
        "splits": {
            name: {
                "saved": stored.get(name, {}).get("field_balanced_ce"),
                "recomputed": results[RULE_FULL]["metrics"][name]["field_balanced_ce"],
                "difference": (
                    None
                    if stored.get(name, {}).get("field_balanced_ce") is None
                    else results[RULE_FULL]["metrics"][name]["field_balanced_ce"]
                    - stored[name]["field_balanced_ce"]
                ),
            }
            for name in results[RULE_FULL]["metrics"]
        },
    }

    # --------------------------------------------------------

    aggregates, fields, silent = rule_tables(results, rules, splits)

    report = {
        "mode": "ablation",
        "checkpoint": str(checkpoint),
        "device": str(trainer.device),
        "precision": trainer.precision,
        "parameters": trainer.n_parameters(),
        "rules": rules,
        "rule_notes": {rule: RULE_NOTES[rule] for rule in rules},
        "config": config.as_dict(),
        "model": trainer.model_config.as_dict(),
        "data": stores,
        "splits": descriptions,
        "load_seconds": load_seconds,
        "seconds": {rule: results[rule]["seconds"] for rule in rules},
        "baseline_check": baseline_check,
        "aggregates": aggregates,
        "fields": fields,
        "fields_without_targets": silent,
        "note": (
            "дельты считаются к пересчитанному full; CE unigram от правила не зависит, "
            "поэтому delta_nce_gain = -delta_ce_model / ce_unigram"
        ),
    }

    write_json(out_dir / "ablation.json", report)
    write_text(out_dir / "ablation.md", render_ablation(report))

    if not quiet:
        print()
        print(render_ablation(report))

    return report


# ============================================================
# ОТЧЁТ
# ============================================================


def _text(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _signed(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


AGGREGATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("field-balanced CE", "field_balanced_ce"),
    ("token-weighted CE", "token_weighted_ce"),
    ("средний NCE gain", "mean_nce_gain"),
    ("accuracy", "accuracy"),
    ("средний macro F1", "macro_f1_mean"),
)


def render_ablation(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Диагностика внимания History Encoder")
    lines.append("")
    lines.append(
        f"Checkpoint `{report['checkpoint']}`, устройство {report['device']}, "
        f"precision {report['precision']}, параметров {report['parameters']:,}".replace(",", " ")
    )
    lines.append("")
    lines.append(
        "Модель не обучается и не меняется. Один и тот же checkpoint прогоняется по одним и тем же "
        "зафиксированным validation-маскам, меняется только то, кому позиция History Encoder может внимать."
    )
    lines.append("")

    # --------------------------------------------------------

    lines.append("## Варианты")
    lines.append("")
    lines.append("| вариант | что запрещено |")
    lines.append("|---|---|")

    for rule in report["rules"]:
        lines.append(f"| `{rule}` | {report['rule_notes'][rule]} |")

    lines.append("")

    # --------------------------------------------------------

    check = report["baseline_check"]

    lines.append("## Сверка baseline")
    lines.append("")
    lines.append(f"Шаг checkpoint: {check['step']}. Маски validation совпали с сохранёнными в checkpoint.")
    lines.append("")
    lines.append("| набор | field-balanced CE в checkpoint | пересчитано сейчас | разница |")
    lines.append("|---|---|---|---|")

    for name, item in check["splits"].items():
        lines.append(
            f"| {name} | {_text(item['saved'])} | {_text(item['recomputed'])} | {_signed(item['difference'], 6)} |"
        )

    lines.append("")

    # --------------------------------------------------------

    lines.append("## Агрегаты")
    lines.append("")

    for name, rows in report["aggregates"].items():

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Целей {rows[0]['n_targets']:,}.".replace(",", " "))
        lines.append("")

        header = "| вариант | " + " | ".join(label for label, _ in AGGREGATE_COLUMNS) + " |"

        lines.append(header)
        lines.append("|" + "---|" * (len(AGGREGATE_COLUMNS) + 1))

        for row in rows:

            cells = []

            for _, key in AGGREGATE_COLUMNS:

                value = _text(row[key])

                if row["rule"] != "full":
                    value = f"{value} ({_signed(row['delta_' + key])})"

                cells.append(value)

            lines.append(f"| `{row['rule']}` | " + " | ".join(cells) + " |")

        lines.append("")

    # --------------------------------------------------------

    lines.append("## По полям")
    lines.append("")
    lines.append(
        "Показана CE варианта `full` и прирост CE каждого запрета. Больше прирост это сильнее "
        "поле зависит от закрытого канала. NCE gain движется зеркально: "
        "`delta_nce = -delta_ce / CE unigram`, а CE unigram от варианта не зависит."
    )
    lines.append("")

    others = [rule for rule in report["rules"] if rule != "full"]

    for name, rows in report["fields"].items():

        lines.append(f"### {name}")
        lines.append("")

        header = (
            "| поле | целей | канд | CE full | NCE full | CE unigram | "
            + " | ".join(f"ΔCE `{rule}`" for rule in others)
            + " |"
        )

        lines.append(header)
        lines.append("|" + "---|" * (6 + len(others)))

        for row in rows:
            deltas = " | ".join(_signed(row["delta_ce_model"][rule], 3) for rule in others)
            lines.append(
                f"| {row['field']} | {row['n_targets']} | {row['n_candidates']} | "
                f"{_text(row['ce_model']['full'], 3)} | {_text(row['nce_gain']['full'], 3)} | "
                f"{_text(row['ce_unigram'], 3)} | {deltas} |"
            )

        lines.append("")

        if report["fields_without_targets"].get(name):
            lines.append("Без целей: " + ", ".join(report["fields_without_targets"][name]) + ".")
            lines.append("")

    return "\n".join(lines) + "\n"
