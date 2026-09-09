from __future__ import annotations

import time
from pathlib import Path

import torch

from src.preprocessing.artifacts import write_json, write_text
from src.tokenizer.config import IncompatibleArtifactsError
from src.tokenizer.masking import MaskingConfig

from .ablation import evaluate_rules, rule_tables
from .checkpoint import load_checkpoint
from .data import ClientStore, FixedSplit
from .history_encoder import ATTENTION_RULES, RULE_EVENTS_VIA_PROFILE, RULE_FULL, RULE_SELF_ONLY
from .metrics import bootstrap_combination, bootstrap_contrast
from .targets import excluded_fields as _excluded_fields
from .trainer import Environment, TrainConfig, Trainer, VALIDATION_SPLITS, store_for


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сравниваются две модели, обученные разными masking objective.
# CE у разных режимов маскирования несравнима: скрыть одно
# значение и скрыть событие целиком это разные задачи. Поэтому
# режим оценки фиксируется, и внутри него сравниваются два
# checkpoint'а на ОДНИХ примерах, масках и целях.
#
# Наборы масок строятся один раз и хэшируются. Набор
# field_balanced обязан совпасть с тем, на котором оценивался
# старый run: иначе «стало лучше» означало бы в том числе
# «маски стали проще».
#
# Малая разница это не результат. Разница между двумя моделями
# сопровождается парным bootstrap по примерам, и вывод делается
# только когда интервал не накрывает ноль.
# ============================================================


EVAL_MODES: tuple[str, ...] = ("field_balanced", "token", "event", "key")

# Режимы, на которых повторяется диагностика внимания.
ABLATION_MODES: tuple[str, ...] = ("token", "event", "key")

# Настройки, при расхождении которых наборы масок или цели
# перестали бы совпадать, и сравнение потеряло бы смысл.
MUST_MATCH: tuple[str, ...] = (
    "max_val_clients",
    "max_events_per_history",
    "eval_batch_size",
    "val_seed",
    "epsilon",
    "top_k",
)

MIN_TARGETS = 30

STATUS_LOW = "low_support"


def evaluation_masking(mode: str, stored: dict, val_seed: int) -> MaskingConfig:
    """
    Маски оценки. Ставки заданы спецификацией, seed общий.

    field_balanced воспроизводит набор старого run вплоть до
    balanced_share, поэтому доля берётся из его masking_config.

    key это combined с нулевыми token и event: режим key в
    tokenizer маскирует ровно keys_per_example ключей на пример,
    а здесь нужен независимый розыгрыш каждого ключа с
    вероятностью 10 %.
    """

    if mode == "field_balanced":
        return MaskingConfig(
            mode="field_balanced",
            seed=val_seed,
            balanced_share=float(stored.get("balanced_share", 0.15)),
        )

    if mode == "token":
        return MaskingConfig(mode="token", seed=val_seed, token_rate=0.15)

    if mode == "event":
        return MaskingConfig(mode="event", seed=val_seed, event_rate=0.10)

    if mode == "key":
        return MaskingConfig(
            mode="combined", seed=val_seed, token_rate=0.0, event_rate=0.0, key_rate=0.10
        )

    raise ValueError(f"неизвестный режим оценки {mode!r}, ожидался один из {EVAL_MODES}")


# Определение переехало в src/model/targets.py: там же живёт
# политика целей, которая убирает эти поля не только из
# агрегата отчёта, но и из самой задачи. Имя оставлено, чтобы
# места вызова и тесты не менялись.
excluded_fields = _excluded_fields


# ============================================================
# ЗАГРУЗКА ДВУХ CHECKPOINT'ОВ
# ============================================================


def _config_differences(left: dict, right: dict) -> dict:
    names = sorted(set(left) | set(right))

    return {
        name: {"old": left.get(name), "new": right.get(name)}
        for name in names
        if left.get(name) != right.get(name)
    }


def check_comparable(old: dict, new: dict) -> None:

    broken = {
        name: {"old": old.get(name), "new": new.get(name)}
        for name in MUST_MATCH
        if old.get(name) != new.get(name)
    }

    if broken:
        raise IncompatibleArtifactsError(
            "checkpoint'ы несравнимы: наборы масок или цели построились бы разными. "
            f"Расходятся {broken}"
        )


def build_trainer(env: Environment, config: TrainConfig, checkpoint: Path, device: str) -> Trainer:

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, device)

    load_checkpoint(
        checkpoint,
        backbone=trainer.backbone,
        head=trainer.head,
        model_config=trainer.model_config.as_dict(),
        artifacts=env.hashes,
        restore_random=False,
    )

    return trainer


# ============================================================
# СРАВНЕНИЕ
# ============================================================


def run_comparison(
    env: Environment,
    old: Path,
    new: Path,
    out_dir: Path,
    device: str = "cpu",
    modes: tuple[str, ...] = EVAL_MODES,
    ablation_modes: tuple[str, ...] = ABLATION_MODES,
    rules: tuple[str, ...] = ATTENTION_RULES,
    min_targets: int = MIN_TARGETS,
    n_boot: int = 2000,
    quiet: bool = False,
) -> dict:
    """
    Два checkpoint'а, четыре набора масок, повтор диагностики внимания.
    """

    old, new, out_dir = Path(old), Path(new), Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    old_payload = torch.load(old, map_location="cpu", weights_only=False)
    new_payload = torch.load(new, map_location="cpu", weights_only=False)

    old_config = TrainConfig.from_dict(old_payload["train_config"])
    new_config = TrainConfig.from_dict(new_payload["train_config"])

    check_comparable(old_payload["train_config"], new_payload["train_config"])

    if abs(old_config.epsilon - env.unigram.epsilon) > 0:
        raise IncompatibleArtifactsError(
            f"unigram собран с epsilon {env.unigram.epsilon}, а checkpoint'ы с {old_config.epsilon}"
        )

    old_trainer = build_trainer(env, old_config, old, device)
    new_trainer = build_trainer(env, new_config, new, device)

    if old_trainer.precision != new_trainer.precision:
        raise IncompatibleArtifactsError(
            f"разная точность оценки: {old_trainer.precision} против {new_trainer.precision}"
        )

    exclude = excluded_fields(env.table)

    # --------------------------------------------------------
    # НАБОРЫ МАСОК
    # --------------------------------------------------------

    started = time.perf_counter()

    stores = {
        name: store_for(env, old_config, name)
        for name in VALIDATION_SPLITS
    }

    masks: dict[str, dict[str, FixedSplit]] = {}

    for mode in modes:

        masking = evaluation_masking(mode, old_payload.get("masking_config") or {}, old_config.val_seed)

        masks[mode] = {
            name: FixedSplit.build(
                name=name,
                store=store,
                vocab=env.vocab,
                table=env.table,
                model_config=old_trainer.model_config,
                masking=masking,
                max_events=old_config.max_events_per_history,
                batch_size=old_config.eval_batch_size,
            )
            for name, store in stores.items()
        }

    build_seconds = time.perf_counter() - started

    # Набор field_balanced обязан быть тем же, на котором
    # оценивался старый run.
    reference = old_payload.get("splits") or {}

    for name, split in masks.get("field_balanced", {}).items():

        saved = (reference.get(name) or {}).get("targets_sha256")

        if saved is not None and saved != split.digest:
            raise IncompatibleArtifactsError(
                f"набор field_balanced для {name} не совпал с сохранённым в старом checkpoint: "
                f"{split.digest} против {saved}"
            )

    # --------------------------------------------------------
    # ОЦЕНКА
    # --------------------------------------------------------

    comparisons: dict[str, dict] = {}

    for mode in modes:

        if not quiet:
            print(f"  режим {mode}")

        splits = masks[mode]

        reports: dict[str, dict] = {}
        contrasts: dict[str, dict] = {}

        for label, trainer in (("old", old_trainer), ("new", new_trainer)):

            evaluated = trainer.evaluate(splits, exclude=exclude, keep_units=True)

            evaluated.pop("_seconds")

            reports[label] = {
                "metrics": {name: evaluated[name] for name in splits},
                "accumulators": evaluated["_accumulators"],
            }

        for name in splits:

            left = reports["old"]["accumulators"][name]
            right = reports["new"]["accumulators"][name]

            contrasts[name] = {
                "all_fields": bootstrap_contrast(
                    left.unit_losses(), right.unit_losses(), n_boot=n_boot, seed=old_config.val_seed
                ),
                "subset": bootstrap_contrast(
                    left.unit_losses(exclude),
                    right.unit_losses(exclude),
                    n_boot=n_boot,
                    seed=old_config.val_seed,
                ),
            }

        comparisons[mode] = {
            "masking": masks[mode][VALIDATION_SPLITS[0]].settings["masking"],
            "splits": {name: split.description() for name, split in splits.items()},
            "old": reports["old"]["metrics"],
            "new": reports["new"]["metrics"],
            "contrast": contrasts,
            "fields": {
                name: _field_pairs(
                    reports["old"]["metrics"][name], reports["new"]["metrics"][name], min_targets
                )
                for name in splits
            },
        }

        if not quiet:
            for name in splits:
                item = comparisons[mode]["contrast"][name]["all_fields"]
                print(
                    f"    {name:<12s} old {reports['old']['metrics'][name]['field_balanced_ce']:.4f}  "
                    f"new {reports['new']['metrics'][name]['field_balanced_ce']:.4f}  "
                    f"Δ {item['estimate']:+.4f} [{item['ci_low']:+.4f}, {item['ci_high']:+.4f}]"
                )

    # --------------------------------------------------------
    # ДИАГНОСТИКА ВНИМАНИЯ
    # --------------------------------------------------------

    ablations: dict[str, dict] = {}

    for mode in ablation_modes:

        if not quiet:
            print(f"  диагностика внимания, режим {mode}")

        splits = masks[mode]

        per_checkpoint: dict[str, dict] = {}

        for label, trainer in (("old", old_trainer), ("new", new_trainer)):

            results = evaluate_rules(
                trainer,
                splits,
                list(rules),
                exclude=exclude,
                keep_units=True,
                quiet=quiet,
                label=f"{label} ",
            )

            aggregates, fields, silent = rule_tables(results, list(rules), splits)

            per_checkpoint[label] = {
                "aggregates": aggregates,
                "fields": fields,
                "fields_without_targets": silent,
                "accumulators": {rule: results[rule]["accumulators"] for rule in rules},
            }

        ablations[mode] = {
            "old": {key: value for key, value in per_checkpoint["old"].items() if key != "accumulators"},
            "new": {key: value for key, value in per_checkpoint["new"].items() if key != "accumulators"},
            "contrast": _ablation_contrasts(per_checkpoint, list(rules), splits, exclude, n_boot, old_config.val_seed),
        }

    # --------------------------------------------------------

    report = {
        "mode": "comparison",
        "device": str(old_trainer.device),
        "precision": old_trainer.precision,
        "checkpoints": {"old": str(old), "new": str(new)},
        "steps": {
            "old": (old_payload.get("metrics") or {}).get("step"),
            "new": (new_payload.get("metrics") or {}).get("step"),
        },
        "counters": {"old": old_payload["counters"], "new": new_payload["counters"]},
        "config": {"old": old_config.as_dict(), "new": new_config.as_dict()},
        "config_differences": _config_differences(
            old_payload["train_config"], new_payload["train_config"]
        ),
        "training_masking": {
            "old": old_payload.get("masking_config"),
            "new": new_payload.get("masking_config"),
        },
        "excluded_fields": sorted(exclude),
        "min_targets": min_targets,
        "n_boot": n_boot,
        "modes": list(modes),
        "ablation_modes": list(ablation_modes),
        "rules": list(rules),
        "data": {name: store.summary() for name, store in stores.items()},
        "build_seconds": build_seconds,
        "comparisons": comparisons,
        "ablations": ablations,
        "answers": _answers(ablations, comparisons),
        "note": (
            "CE разных режимов маскирования между собой не сравнивается: скрыть одно значение "
            "и скрыть событие целиком это разные задачи. Сравниваются два checkpoint'а внутри "
            "одного режима на одних примерах, масках и целях"
        ),
    }

    write_json(out_dir / "comparison.json", report)
    write_json(out_dir / "masks.json", _masks_manifest(masks, report))
    write_text(out_dir / "comparison.md", render_comparison(report))
    write_text(out_dir / "ablation_comparison.md", render_ablation_comparison(report))

    if not quiet:
        print()
        print(render_comparison(report))
        print()
        print(render_ablation_comparison(report))

    return report


# ============================================================
# ТАБЛИЦЫ
# ============================================================


def _field_pairs(old: dict, new: dict, min_targets: int) -> list[dict]:
    """
    Поле за полем: старый checkpoint против нового.
    """

    right = {item["key_id"]: item for item in new["fields"]}

    rows: list[dict] = []

    for item in old["fields"]:

        if item["n_targets"] == 0:
            continue

        other = right[item["key_id"]]

        rows.append(
            {
                "field": item["field"],
                "key_id": item["key_id"],
                "n_targets": item["n_targets"],
                "n_candidates": item["n_candidates"],
                "ce_unigram": item["ce_unigram"],
                "support": STATUS_LOW if item["n_targets"] < min_targets else "ok",
                "old": {
                    key: item[key] for key in ("ce_model", "nce_gain", "accuracy", "macro_f1")
                },
                "new": {
                    key: other[key] for key in ("ce_model", "nce_gain", "accuracy", "macro_f1")
                },
                "delta": {
                    key: None
                    if item[key] is None or other[key] is None
                    else other[key] - item[key]
                    for key in ("ce_model", "nce_gain", "accuracy", "macro_f1")
                },
            }
        )

    rows.sort(key=lambda row: row["delta"]["ce_model"] if row["delta"]["ce_model"] is not None else 0.0)

    return rows


def _ablation_contrasts(per_checkpoint, rules, splits, exclude, n_boot, seed) -> dict:
    """
    Насколько сильнее новая модель теряет от запрета внимания.

    Сравнивается не CE, а ПОТЕРЯ от запрета: у моделей разные
    базовые уровни, и разность потерь это то, что отвечает на
    вопрос о роли истории.
    """

    out: dict[str, dict] = {}

    for name in splits:

        out[name] = {}

        for rule in rules:

            if rule == RULE_FULL:
                continue

            entry: dict[str, dict] = {}

            for scope, subset in (("all_fields", frozenset()), ("subset", exclude)):

                deltas = {}
                samples = {}

                for label in ("old", "new"):

                    accumulators = per_checkpoint[label]["accumulators"]

                    samples[label] = {
                        "full": accumulators[RULE_FULL][name].unit_losses(subset),
                        "rule": accumulators[rule][name].unit_losses(subset),
                    }

                    deltas[label] = bootstrap_contrast(
                        samples[label]["full"], samples[label]["rule"], n_boot=n_boot, seed=seed
                    )

                entry[scope] = {
                    "old": deltas["old"],
                    "new": deltas["new"],
                    # Разность разностей на общей выборке
                    # примеров: два независимых интервала
                    # ответа на вопрос «стало ли сильнее» не дают.
                    "difference": bootstrap_combination(
                        [
                            (-1.0, samples["new"]["full"]),
                            (1.0, samples["new"]["rule"]),
                            (1.0, samples["old"]["full"]),
                            (-1.0, samples["old"]["rule"]),
                        ],
                        n_boot=n_boot,
                        seed=seed,
                    ),
                }

            out[name][rule] = entry

    return out


def _grew(item: dict) -> str:
    """
    Вердикт о ПРИРОСТЕ зависимости, а не о самой зависимости.

    Значимая потеря от запрета у новой модели ещё не значит,
    что зависимость выросла: у старой она могла быть такой же
    или больше. Отвечает только интервал разности разностей.
    """

    difference = item["difference"]

    if not difference["significant"]:
        return "не подтверждено: интервал разности накрывает ноль"

    return "да, выросла" if difference["estimate"] > 0 else "нет, зависимость снизилась"


def _present(item: dict) -> str:
    """
    Есть ли вклад у новой модели вообще.
    """

    if not item["new"]["significant"]:
        return "нет: потеря от запрета неотличима от нуля"

    return "да" if item["new"]["estimate"] > 0 else "нет, запрет улучшает результат"


def _answers(ablations: dict, comparisons: dict) -> dict:
    """
    Три вопроса спецификации с числами вместо впечатлений.
    """

    out: dict[str, dict] = {}

    for mode, section in ablations.items():

        for name, rules in section["contrast"].items():

            if RULE_SELF_ONLY not in rules or RULE_EVENTS_VIA_PROFILE not in rules:
                continue

            self_only = rules[RULE_SELF_ONLY]
            via_profile = rules[RULE_EVENTS_VIA_PROFILE]

            out[f"{mode}/{name}"] = {
                "history_exchange": {
                    "question": "выросла ли зависимость от обмена между позициями истории",
                    "old_delta": self_only["all_fields"]["old"]["estimate"],
                    "new_delta": self_only["all_fields"]["new"]["estimate"],
                    "difference": self_only["all_fields"]["difference"],
                    "answer": _grew(self_only["all_fields"]),
                },
                "direct_event_to_event": {
                    "question": "появился ли вклад прямого event-to-event attention",
                    "old_delta": via_profile["all_fields"]["old"]["estimate"],
                    "new_delta": via_profile["all_fields"]["new"]["estimate"],
                    "difference": via_profile["all_fields"]["difference"],
                    "answer": (
                        f"вклад у новой модели: {_present(via_profile['all_fields'])}; "
                        f"прирост: {_grew(via_profile['all_fields'])}"
                    ),
                },
                "without_leaky_fields": {
                    "question": "сохраняется ли картина без event_type и profile_snapshot",
                    "old_delta": self_only["subset"]["old"]["estimate"],
                    "new_delta": self_only["subset"]["new"]["estimate"],
                    "difference": self_only["subset"]["difference"],
                    "answer": _grew(self_only["subset"]),
                },
            }

    return out


def _masks_manifest(masks: dict, report: dict) -> dict:
    return {
        "note": (
            "наборы масок воспроизводятся по этим настройкам: тот же словарь, те же клиенты, "
            "тот же val_seed и номер batch как шаг masker"
        ),
        "val_seed": report["config"]["old"]["val_seed"],
        "max_val_clients": report["config"]["old"]["max_val_clients"],
        "max_events_per_history": report["config"]["old"]["max_events_per_history"],
        "eval_batch_size": report["config"]["old"]["eval_batch_size"],
        "modes": {
            mode: {name: split.description() for name, split in splits.items()}
            for mode, splits in masks.items()
        },
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


SUMMARY_ROWS: tuple[tuple[str, str], ...] = (
    ("field-balanced CE", "field_balanced_ce"),
    ("средний NCE gain", "mean_nce_gain"),
    ("accuracy", "accuracy"),
    ("средний macro F1", "macro_f1_mean"),
)


def render_comparison(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Combined masking против field_balanced")
    lines.append("")
    lines.append(
        f"Устройство {report['device']}, precision {report['precision']}. "
        f"Старый checkpoint `{report['checkpoints']['old']}` (шаг {report['steps']['old']}), "
        f"новый `{report['checkpoints']['new']}` (шаг {report['steps']['new']})."
    )
    lines.append("")
    lines.append(report["note"] + ".")
    lines.append("")

    differences = report["config_differences"]

    lines.append("## Что различается в конфигурациях")
    lines.append("")

    if differences:
        lines.append("| настройка | старый | новый |")
        lines.append("|---|---|---|")
        for name, item in differences.items():
            lines.append(f"| {name} | {item['old']} | {item['new']} |")
    else:
        lines.append("Ничего: конфигурации совпадают.")

    lines.append("")
    lines.append(
        "Исключаемые из второго агрегата поля: "
        + ", ".join(f"`{name}`" for name in report["excluded_fields"])
        + "."
    )
    lines.append("")

    # --------------------------------------------------------

    for mode in report["modes"]:

        section = report["comparisons"][mode]

        lines.append(f"## Режим оценки `{mode}`")
        lines.append("")

        settings = section["masking"]

        lines.append(
            f"Маски: {settings['mode']}, token {settings['token_rate']}, event {settings['event_rate']}, "
            f"key {settings['key_rate']}, balanced_share {settings['balanced_share']}, seed {settings['seed']}."
        )
        lines.append("")

        for name in section["splits"]:

            description = section["splits"][name]

            lines.append(f"### {name}")
            lines.append("")
            lines.append(
                f"Клиентов {report['data'][name]['clients']}, примеров {description['n_examples']}, "
                f"целей {description['n_targets']}, скрыто {description['masked_fraction'] * 100:.1f} % "
                f"доступных значений. Хэш целей `{description['targets_sha256'][:16]}…`."
            )
            lines.append("")

            old, new = section["old"][name], section["new"][name]

            lines.append("| метрика | старый | новый | Δ |")
            lines.append("|---|---|---|---|")

            for label, key in SUMMARY_ROWS:
                delta = None if old[key] is None or new[key] is None else new[key] - old[key]
                lines.append(f"| {label} | {_text(old[key])} | {_text(new[key])} | {_signed(delta)} |")

            subset_old, subset_new = old.get("subset"), new.get("subset")

            if subset_old and subset_new:
                lines.append(
                    f"| field-balanced CE без {len(report['excluded_fields'])} полей | "
                    f"{_text(subset_old['field_balanced_ce'])} | {_text(subset_new['field_balanced_ce'])} | "
                    f"{_signed(subset_new['field_balanced_ce'] - subset_old['field_balanced_ce'])} |"
                )
                lines.append(
                    f"| средний NCE gain без них | {_text(subset_old['mean_nce_gain'])} | "
                    f"{_text(subset_new['mean_nce_gain'])} | "
                    f"{_signed(subset_new['mean_nce_gain'] - subset_old['mean_nce_gain'])} |"
                )

            lines.append("")

            contrast = section["contrast"][name]

            lines.append(
                f"Парный bootstrap по примерам, {report['n_boot']} выборок, ΔCE (новый − старый): "
                f"все поля {_interval(contrast['all_fields'])}, "
                f"без исключённых {_interval(contrast['subset'])}."
            )
            lines.append("")

            # ----------------------------------------------

            lines.append("| поле | целей | CE старый | CE новый | ΔCE | NCE старый | NCE новый | поддержка |")
            lines.append("|---|---|---|---|---|---|---|---|")

            for row in section["fields"][name]:
                lines.append(
                    f"| {row['field']} | {row['n_targets']} | {_text(row['old']['ce_model'], 3)} | "
                    f"{_text(row['new']['ce_model'], 3)} | {_signed(row['delta']['ce_model'], 3)} | "
                    f"{_text(row['old']['nce_gain'], 3)} | {_text(row['new']['nce_gain'], 3)} | "
                    f"{'мало целей' if row['support'] == STATUS_LOW else ''} |"
                )

            lines.append("")

    return "\n".join(lines) + "\n"


def render_ablation_comparison(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Роль истории: диагностика внимания у обоих checkpoint'ов")
    lines.append("")
    lines.append(
        "Сравнивается не CE, а потеря от запрета внимания. У моделей разные базовые уровни, "
        "и именно разность потерь отвечает на вопрос о роли истории."
    )
    lines.append("")

    others = [rule for rule in report["rules"] if rule != RULE_FULL]

    for mode in report["ablation_modes"]:

        section = report["ablations"][mode]

        lines.append(f"## Маски `{mode}`")
        lines.append("")

        for name in section["contrast"]:

            lines.append(f"### {name}")
            lines.append("")
            lines.append("| запрет | ΔCE старый | ΔCE новый | разность разностей |")
            lines.append("|---|---|---|---|")

            for rule in others:

                item = section["contrast"][name][rule]["all_fields"]

                lines.append(
                    f"| `{rule}` | {_interval(item['old'], 4)} | {_interval(item['new'], 4)} | "
                    f"{_interval(item['difference'], 4)} |"
                )

            lines.append("")
            lines.append("Без `timeline__event_type` и `profile_snapshot__*`:")
            lines.append("")
            lines.append("| запрет | ΔCE старый | ΔCE новый | разность разностей |")
            lines.append("|---|---|---|---|")

            for rule in others:

                item = section["contrast"][name][rule]["subset"]

                lines.append(
                    f"| `{rule}` | {_interval(item['old'], 4)} | {_interval(item['new'], 4)} | "
                    f"{_interval(item['difference'], 4)} |"
                )

            lines.append("")

    # --------------------------------------------------------

    lines.append("## Ответы")
    lines.append("")

    for key, answers in report["answers"].items():

        lines.append(f"### {key}")
        lines.append("")

        for item in answers.values():
            lines.append(
                f"- {item['question']}: **{item['answer']}**. "
                f"Потеря от запрета: старый {_signed(item['old_delta'])}, "
                f"новый {_signed(item['new_delta'])}, разность {_interval(item['difference'])}."
            )

        lines.append("")

    lines.append(
        "Вывод о качестве client embedding по этим числам не делается: они меряют только "
        "восстановление замаскированных значений."
    )

    return "\n".join(lines) + "\n"
