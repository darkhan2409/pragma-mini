from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.artifacts import write_json, write_text
from src.tokenizer.config import IncompatibleArtifactsError

from .ablation import evaluate_rules
from .compare import (
    ABLATION_MODES,
    EVAL_MODES,
    MIN_TARGETS,
    STATUS_LOW,
    build_trainer,
    check_comparable,
    evaluation_masking,
    excluded_fields,
)
from .data import ClientStore, FixedSplit
from .history_encoder import ATTENTION_RULES, RULE_FULL
from .metrics import bootstrap_combination, cluster_draws, group_sample
from .trainer import Environment, TrainConfig, VALIDATION_SPLITS, store_for


# ============================================================
# ИДЕЯ
# ============================================================
#
# Прежний bootstrap пересэмплировал месячные примеры
# независимо. В val_client одному клиенту принадлежат двадцать
# cutoff'ов: его истории пересекаются, ошибки коррелированы, и
# интервал получался уже настоящего.
#
# Единица пересэмплирования это клиент. Реализуется
# суммированием строк примеров одного клиента ДО розыгрыша:
# field-balanced CE сначала складывает NLL и counts по полю и
# только потом усредняет CE по полям, поэтому сумма столбца от
# группировки не меняется. «Взять клиента с кратностью k вместе
# со всеми его примерами» тождественно «взять строку клиента с
# весом k», и точечные оценки остаются прежними до разряда.
#
# Оба уровня считаются в одном проходе и печатаются рядом:
# вопрос был не «какой интервал правильный», а «изменился ли
# вывод».
# ============================================================


LEVEL_EXAMPLE = "by_example"
LEVEL_CLIENT = "by_client"

LEVELS: tuple[str, ...] = (LEVEL_EXAMPLE, LEVEL_CLIENT)


# ============================================================
# ДВА УРОВНЯ ОДНИМ КОДОМ
# ============================================================


class Levels:
    """
    Пара розыгрышей на один набор: по примерам и по клиентам.

    Матрицы кратностей строятся один раз и переиспользуются для
    обеих моделей, всех режимов масок и всех правил внимания.
    """

    def __init__(self, split: FixedSplit, n_boot: int, seed: int):

        self.clusters = np.asarray(split.client_of_example, dtype=np.int64)

        self.n_examples = int(split.n_examples)
        self.n_clients = int(np.unique(self.clusters).size)

        self.draws = {
            LEVEL_EXAMPLE: cluster_draws(self.n_examples, n_boot, seed),
            LEVEL_CLIENT: cluster_draws(self.n_clients, n_boot, seed),
        }

    def samples(self, accumulator, exclude: frozenset[str]) -> dict[str, tuple]:

        universe = np.arange(self.n_examples, dtype=np.int64)

        by_example = accumulator.unit_losses(exclude, universe=universe)

        return {
            LEVEL_EXAMPLE: by_example,
            LEVEL_CLIENT: group_sample(by_example, self.clusters),
        }

    def combine(self, terms: list[tuple[float, dict[str, tuple]]]) -> dict:
        """
        Одна линейная комбинация, посчитанная на обоих уровнях.
        """

        out: dict[str, dict] = {}

        for level in LEVELS:
            out[level] = bootstrap_combination(
                [(weight, sample[level]) for weight, sample in terms],
                draws=self.draws[level],
            )

        out["significance_changed"] = bool(
            out[LEVEL_EXAMPLE]["significant"] != out[LEVEL_CLIENT]["significant"]
        )

        out["ci_width_ratio"] = _width_ratio(out[LEVEL_EXAMPLE], out[LEVEL_CLIENT])

        return out


def _width_ratio(example: dict, client: dict) -> float | None:

    width = example["ci_high"] - example["ci_low"]

    if width <= 0:
        return None

    return (client["ci_high"] - client["ci_low"]) / width


# ============================================================
# ЗАПУСК
# ============================================================


def run_cluster_bootstrap(
    env: Environment,
    old: Path,
    new: Path,
    out_dir: Path,
    device: str = "cpu",
    modes: tuple[str, ...] = EVAL_MODES,
    ablation_modes: tuple[str, ...] = ABLATION_MODES,
    rules: tuple[str, ...] = ATTENTION_RULES,
    n_boot: int = 2000,
    min_targets: int = MIN_TARGETS,
    quiet: bool = False,
) -> dict:
    """
    Тот же эксперимент, две единицы пересэмплирования.
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

    trainers = {
        "old": build_trainer(env, old_config, old, device),
        "new": build_trainer(env, new_config, new, device),
    }

    if trainers["old"].precision != trainers["new"].precision:
        raise IncompatibleArtifactsError("разная точность оценки у двух checkpoint'ов")

    exclude = excluded_fields(env.table)

    seed = old_config.val_seed

    # --------------------------------------------------------
    # ТЕ ЖЕ НАБОРЫ МАСОК
    # --------------------------------------------------------

    started = time.perf_counter()

    stores = {
        name: store_for(env, old_config, name)
        for name in VALIDATION_SPLITS
    }

    masks: dict[str, dict[str, FixedSplit]] = {}

    for mode in modes:

        masking = evaluation_masking(mode, old_payload.get("masking_config") or {}, seed)

        masks[mode] = {
            name: FixedSplit.build(
                name=name,
                store=store,
                vocab=env.vocab,
                table=env.table,
                model_config=trainers["old"].model_config,
                masking=masking,
                max_events=old_config.max_events_per_history,
                batch_size=old_config.eval_batch_size,
            )
            for name, store in stores.items()
        }

    build_seconds = time.perf_counter() - started

    reference = old_payload.get("splits") or {}

    for name, split in masks.get("field_balanced", {}).items():

        saved = (reference.get(name) or {}).get("targets_sha256")

        if saved is not None and saved != split.digest:
            raise IncompatibleArtifactsError(
                f"набор field_balanced для {name} не совпал с сохранённым: {split.digest} против {saved}"
            )

    # Один набор розыгрышей на сплит; маски у режимов разные, но
    # клиенты и порядок примеров одни и те же.
    levels = {
        name: Levels(masks[modes[0]][name], n_boot, seed) for name in VALIDATION_SPLITS
    }

    # --------------------------------------------------------
    # ОЦЕНКА
    # --------------------------------------------------------

    results: dict[str, dict] = {}

    for mode in modes:

        if not quiet:
            print(f"  режим {mode}")

        splits = masks[mode]

        wanted = list(rules) if mode in ablation_modes else [RULE_FULL]

        evaluated = {
            label: evaluate_rules(
                trainer, splits, wanted, exclude=exclude, keep_units=True, quiet=True
            )
            for label, trainer in trainers.items()
        }

        contrasts: dict[str, dict] = {}
        ablations: dict[str, dict] = {}
        fields: dict[str, list[dict]] = {}

        for name in splits:

            pair = levels[name]

            samples = {
                (label, rule, scope): pair.samples(
                    evaluated[label][rule]["accumulators"][name], subset
                )
                for label in trainers
                for rule in wanted
                for scope, subset in (("all_fields", frozenset()), ("subset", exclude))
            }

            contrasts[name] = {
                scope: pair.combine(
                    [
                        (-1.0, samples[("old", RULE_FULL, scope)]),
                        (1.0, samples[("new", RULE_FULL, scope)]),
                    ]
                )
                for scope in ("all_fields", "subset")
            }

            ablations[name] = {
                rule: {
                    scope: {
                        "old": pair.combine(
                            [
                                (-1.0, samples[("old", RULE_FULL, scope)]),
                                (1.0, samples[("old", rule, scope)]),
                            ]
                        ),
                        "new": pair.combine(
                            [
                                (-1.0, samples[("new", RULE_FULL, scope)]),
                                (1.0, samples[("new", rule, scope)]),
                            ]
                        ),
                        "difference": pair.combine(
                            [
                                (-1.0, samples[("new", RULE_FULL, scope)]),
                                (1.0, samples[("new", rule, scope)]),
                                (1.0, samples[("old", RULE_FULL, scope)]),
                                (-1.0, samples[("old", rule, scope)]),
                            ]
                        ),
                    }
                    for scope in ("all_fields", "subset")
                }
                for rule in wanted
                if rule != RULE_FULL
            }

            fields[name] = _field_support(
                evaluated["old"][RULE_FULL]["metrics"][name],
                evaluated["new"][RULE_FULL]["metrics"][name],
                evaluated["old"][RULE_FULL]["accumulators"][name],
                pair.clusters,
                min_targets,
            )

        results[mode] = {
            "masking": splits[VALIDATION_SPLITS[0]].settings["masking"],
            "splits": {name: split.description() for name, split in splits.items()},
            "metrics": {
                label: {
                    name: evaluated[label][RULE_FULL]["metrics"][name] for name in splits
                }
                for label in trainers
            },
            "contrast": contrasts,
            "ablation": ablations,
            "fields": fields,
        }

        if not quiet:
            for name in splits:
                item = contrasts[name]["all_fields"]
                print(
                    f"    {name:<12s} ΔCE {item[LEVEL_EXAMPLE]['estimate']:+.4f}   "
                    f"по примерам [{item[LEVEL_EXAMPLE]['ci_low']:+.4f}, {item[LEVEL_EXAMPLE]['ci_high']:+.4f}]"
                    f"   по клиентам [{item[LEVEL_CLIENT]['ci_low']:+.4f}, {item[LEVEL_CLIENT]['ci_high']:+.4f}]"
                    + ("   вывод изменился" if item["significance_changed"] else "")
                )

    # --------------------------------------------------------

    report = {
        "mode": "cluster_bootstrap",
        "device": str(trainers["old"].device),
        "precision": trainers["old"].precision,
        "checkpoints": {"old": str(old), "new": str(new)},
        "steps": {
            "old": (old_payload.get("metrics") or {}).get("step"),
            "new": (new_payload.get("metrics") or {}).get("step"),
        },
        "n_boot": int(n_boot),
        "seed": int(seed),
        "min_targets": int(min_targets),
        "excluded_fields": sorted(exclude),
        "modes": list(modes),
        "ablation_modes": list(ablation_modes),
        "rules": list(rules),
        "build_seconds": build_seconds,
        "units": {
            name: {
                "n_clients": levels[name].n_clients,
                "n_examples": levels[name].n_examples,
                "examples_per_client": levels[name].n_examples / levels[name].n_clients,
            }
            for name in VALIDATION_SPLITS
        },
        "results": results,
        "summary": _summary(results, modes),
        "note": (
            "единица пересэмплирования это клиент: примеры одного клиента входят в выборку "
            "вместе и с его кратностью. Точечные оценки от этого не меняются, меняется только "
            "ширина интервала"
        ),
    }

    write_json(out_dir / "cluster_bootstrap.json", report)
    write_text(out_dir / "cluster_bootstrap.md", render_cluster_bootstrap(report))

    if not quiet:
        print()
        print(render_cluster_bootstrap(report))

    return report


# ============================================================
# ПОДДЕРЖКА ПОЛЕЙ
# ============================================================


def _field_support(
    old: dict, new: dict, accumulator, clusters: np.ndarray, min_targets: int
) -> list[dict]:
    """
    По полю: цели, клиенты с целями и разница двух моделей.
    """

    clients = accumulator.clients_with_targets(clusters)

    right = {item["key_id"]: item for item in new["fields"]}

    rows: list[dict] = []

    for item in old["fields"]:

        if item["n_targets"] == 0:
            continue

        other = right[item["key_id"]]

        n_clients = int(clients.get(item["key_id"], 0))

        rows.append(
            {
                "field": item["field"],
                "key_id": item["key_id"],
                "n_targets": item["n_targets"],
                "n_clients_with_targets": n_clients,
                "targets_per_client": item["n_targets"] / n_clients if n_clients else None,
                "support": STATUS_LOW if item["n_targets"] < min_targets or n_clients < 2 else "ok",
                "ce_old": item["ce_model"],
                "ce_new": other["ce_model"],
                "delta_ce": other["ce_model"] - item["ce_model"],
            }
        )

    rows.sort(key=lambda row: row["n_clients_with_targets"])

    return rows


def _summary(results: dict, modes) -> dict:
    """
    Сколько выводов пережило смену единицы пересэмплирования.
    """

    changed: list[str] = []
    kept: list[str] = []

    for mode in modes:

        for name, item in results[mode]["contrast"].items():

            entry = item["all_fields"]

            key = f"{mode}/{name}"

            (changed if entry["significance_changed"] else kept).append(key)

    return {
        "contrasts_total": len(changed) + len(kept),
        "significance_changed": changed,
        "significance_kept": kept,
    }


# ============================================================
# ОТЧЁТ
# ============================================================


def _interval(item: dict, digits: int = 4) -> str:
    return f"[{item['ci_low']:+.{digits}f}, {item['ci_high']:+.{digits}f}]"


def _verdict(item: dict) -> str:
    return "значимо" if item["significant"] else "не значимо"


def render_cluster_bootstrap(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Bootstrap по клиентам вместо месячных примеров")
    lines.append("")
    lines.append(
        f"Устройство {report['device']}, precision {report['precision']}. "
        f"Старый checkpoint шага {report['steps']['old']}, новый шага {report['steps']['new']}. "
        f"{report['n_boot']} выборок, seed {report['seed']}."
    )
    lines.append("")
    lines.append(report["note"] + ".")
    lines.append("")

    lines.append("## Единицы пересэмплирования")
    lines.append("")
    lines.append("| набор | клиентов | примеров | примеров на клиента |")
    lines.append("|---|---|---|---|")

    for name, item in report["units"].items():
        lines.append(
            f"| {name} | {item['n_clients']} | {item['n_examples']} | {item['examples_per_client']:.1f} |"
        )

    lines.append("")

    summary = report["summary"]

    lines.append(
        f"Сравнений моделей: {summary['contrasts_total']}. "
        f"Вывод о значимости изменился у {len(summary['significance_changed'])}: "
        + (", ".join(f"`{key}`" for key in summary["significance_changed"]) or "ни у одного")
        + "."
    )
    lines.append("")

    # --------------------------------------------------------

    lines.append("## Разница моделей: ΔCE (новый − старый)")
    lines.append("")

    for mode in report["modes"]:

        section = report["results"][mode]

        lines.append(f"### Маски `{mode}`")
        lines.append("")
        lines.append(
            "| набор | поля | ΔCE | по примерам | вывод | по клиентам | вывод | во сколько раз шире |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        for name in section["contrast"]:

            for scope, label in (("all_fields", "все"), ("subset", "без исключённых")):

                item = section["contrast"][name][scope]

                ratio = item["ci_width_ratio"]

                lines.append(
                    f"| {name} | {label} | {item[LEVEL_EXAMPLE]['estimate']:+.4f} | "
                    f"{_interval(item[LEVEL_EXAMPLE])} | {_verdict(item[LEVEL_EXAMPLE])} | "
                    f"{_interval(item[LEVEL_CLIENT])} | {_verdict(item[LEVEL_CLIENT])} | "
                    f"{'—' if ratio is None else f'{ratio:.1f}'} |"
                )

        lines.append("")

    # --------------------------------------------------------

    lines.append("## Потеря от запрета внимания")
    lines.append("")
    lines.append(
        "Разность разностей это `(CE нового с запретом − CE нового без) − (то же у старого)`; "
        "она и отвечает на вопрос, стала ли зависимость от истории сильнее."
    )
    lines.append("")

    for mode in report["ablation_modes"]:

        section = report["results"].get(mode)

        if section is None or not section["ablation"]:
            continue

        lines.append(f"### Маски `{mode}`")
        lines.append("")

        for name, rules in section["ablation"].items():

            lines.append(f"**{name}**")
            lines.append("")
            lines.append(
                "| запрет | ΔΔCE | по примерам | вывод | по клиентам | вывод |"
            )
            lines.append("|---|---|---|---|---|---|")

            for rule, item in rules.items():

                entry = item["all_fields"]["difference"]

                lines.append(
                    f"| `{rule}` | {entry[LEVEL_EXAMPLE]['estimate']:+.4f} | "
                    f"{_interval(entry[LEVEL_EXAMPLE])} | {_verdict(entry[LEVEL_EXAMPLE])} | "
                    f"{_interval(entry[LEVEL_CLIENT])} | {_verdict(entry[LEVEL_CLIENT])} |"
                )

            lines.append("")

    # --------------------------------------------------------

    lines.append("## Поддержка полей")
    lines.append("")
    lines.append(
        "Число целей не говорит о числе клиентов: у редкого поля все цели могут принадлежать "
        "одному клиенту, и тогда клиентский bootstrap честно показывает, что оценивать нечего."
    )
    lines.append("")

    for mode in report["modes"]:

        section = report["results"][mode]

        for name, rows in section["fields"].items():

            if name != VALIDATION_SPLITS[-1]:
                continue

            lines.append(f"### Маски `{mode}`, {name}")
            lines.append("")
            lines.append("| поле | целей | клиентов | целей на клиента | CE старый | CE новый | ΔCE | поддержка |")
            lines.append("|---|---|---|---|---|---|---|---|")

            for row in rows:
                per = row["targets_per_client"]
                lines.append(
                    f"| {row['field']} | {row['n_targets']} | {row['n_clients_with_targets']} | "
                    f"{'—' if per is None else f'{per:.1f}'} | {row['ce_old']:.3f} | "
                    f"{row['ce_new']:.3f} | {row['delta_ce']:+.3f} | "
                    f"{'мало' if row['support'] == STATUS_LOW else ''} |"
                )

            lines.append("")

    return "\n".join(lines) + "\n"
