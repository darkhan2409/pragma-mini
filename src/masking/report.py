from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group
from src.temporal.position import TemporalError
from src.temporal.samples import SamplesError, TemporalGroup
from src.tokenization.finalvocab import FrozenArtifacts, VocabError
from src.tokenization.names import Names

from .choose import EVENT, KEY, VALUE, choose, value_chance
from .settings import ConfigError, MaskingConfig
from .weights import WeightsError, load_value_weights


# ============================================================
# ОТЧЁТ О ЦЕЛЯХ МАСКИ
# ============================================================
#
#   python -m src.masking.report train|val|test [--masking-config путь] [--out файл.json]
#
# Куда уходят цели маски группы — без модели и без записи в data/.
# Маска та же, что у обучения: набор 05 читается по группам строк,
# и для каждого клиента зовётся тот же choose. Считается на уровне
# значений (значение из нескольких кусков BPE — одно), по ключам:
#
#   values, data_share      допустимые значения ключа и его доля в данных;
#   targets, target_share   цели ключа (без испорченных в [UNK]) и его
#                           доля среди целей; by_reason — по механизмам;
#   unknown                 выбранные значения, ушедшие в [UNK];
#   top_*                   самое частое значение ключа: его доля в
#                           данных и среди целей ключа;
#   mean_value_probability  средняя вероятность механизма value;
#   value_tv                расстояние полной вариации между значениями
#                           ключа у целей и в данных.
#
# key_tv — то же расстояние между долями ключей у целей и в данных.
# Высокая точность MLM на почти константном поле видна здесь как
# большая доля его целей при маленькой информативности.
# ============================================================


# Колонки набора, которые читает маскер: время не нужно.
COLUMNS = [
    "client_id",
    "key_ids",
    "value_ids",
    "positions",
    "event_starts",
    "event_lengths",
    "target_event_mask",
]


def _tokens(row: dict, value) -> tuple[int, ...]:
    return tuple(row["value_ids"][value.start:value.start + value.length])


def _tv(targets: Counter, data: Counter) -> float:
    """
    Расстояние полной вариации между двумя распределениями счётчиков.
    """

    hits = sum(targets.values())
    total = sum(data.values())

    return 0.5 * sum(abs(targets[value] / hits - data[value] / total) for value in set(data) | set(targets))


def masking_report(group: str, masking: MaskingConfig) -> dict:
    """
    Цели маски группы по ключам.
    """

    samples = TemporalGroup(group)

    weights = load_value_weights() if masking.informativeness_weighted_masking else None

    names = Names(FrozenArtifacts.load())

    data: dict[int, Counter] = {}
    targets: dict[int, Counter] = {}
    reasons: dict[int, Counter] = {}
    chances: dict[int, float] = {}
    spoiled: Counter = Counter()

    for number in range(samples.count):

        for row in samples.row_group(number, columns=COLUMNS).to_pylist():

            selection = choose(group, row, masking, weights)

            for value in selection.values:
                data.setdefault(value.key_id, Counter())[_tokens(row, value)] += 1
                chances[value.key_id] = chances.get(value.key_id, 0.0) + value_chance(value, row, masking, weights)

            for choice in selection.choices:

                key_id = choice.value.key_id

                if choice.unknown:
                    spoiled[key_id] += 1
                    continue

                targets.setdefault(key_id, Counter())[_tokens(row, choice.value)] += 1
                reasons.setdefault(key_id, Counter())[choice.reason] += 1

    total = sum(sum(counter.values()) for counter in data.values())
    hits = sum(sum(counter.values()) for counter in targets.values())

    keys = []

    for key_id, counter in data.items():

        present = sum(counter.values())
        found = targets.get(key_id, Counter())
        chosen = sum(found.values())
        top, top_count = counter.most_common(1)[0]

        keys.append(
            {
                "key": names.short(key_id),
                "values": present,
                "data_share": present / total,
                "targets": chosen,
                "target_share": chosen / hits if hits else 0.0,
                "by_reason": {reason: reasons.get(key_id, Counter())[reason] for reason in (EVENT, KEY, VALUE)},
                "unknown": spoiled[key_id],
                "top_value": names.text(list(top)),
                "top_data_share": top_count / present,
                "top_target_share": found[top] / chosen if chosen else None,
                "mean_value_probability": chances[key_id] / present,
                "value_tv": _tv(found, counter) if chosen else None,
            }
        )

    keys.sort(key=lambda item: (-item["targets"], item["key"]))

    return {
        "group": group,
        "masking": masking.as_dict(),
        "values": total,
        "targets": hits,
        "unknown": sum(spoiled.values()),
        "mean_value_probability": sum(chances.values()) / total if total else None,
        "key_tv": 0.5 * sum(abs(item["target_share"] - item["data_share"]) for item in keys),
        "keys": keys,
    }


def show(report: dict) -> None:

    masking = report["masking"]

    kind = (
        f"взвешенная, value_probability {masking['value_probability']} в "
        f"[{masking['min_value_probability']}, {masking['max_value_probability']}]"
        if masking["informativeness_weighted_masking"]
        else f"равномерная, value_probability {masking['value_probability']}"
    )

    mean = report["mean_value_probability"]

    print(f"[mask-report] группа {report['group']}, маска {kind}")
    print(
        f"    значений {report['values']}, целей {report['targets']}, в [UNK] {report['unknown']}; "
        f"средняя P value {mean:.3f}; TV ключей {report['key_tv']:.3f}"
        if mean is not None
        else "    допустимых значений нет"
    )
    print(
        f"    {'ключ':28} {'данные':>8} {'доля':>6} {'цели':>7} {'доля':>6} {'value':>6} "
        f"{'P value':>7} {'частое: данные/цели':>20} {'TV':>6}  частое значение"
    )

    for item in report["keys"]:

        top = item["top_target_share"]

        print(
            f"    {item['key'][:28]:28} {item['values']:>8} {item['data_share']:>6.3f} "
            f"{item['targets']:>7} {item['target_share']:>6.3f} {item['by_reason'][VALUE]:>6} "
            f"{item['mean_value_probability']:>7.3f} "
            f"{item['top_data_share']:>9.3f}/{'—' if top is None else f'{top:.3f}':>10} "
            f"{'—' if item['value_tv'] is None else format(item['value_tv'], '.3f'):>6}  "
            f"{item['top_value'][:40]}"
        )


def run(args) -> int:

    group = normalize_group(args.group)

    try:
        masking = MaskingConfig.load(args.masking_config)
        report = masking_report(group, masking)
    except (ConfigError, SamplesError, TemporalError, VocabError, WeightsError) as error:
        print(f"[mask-report] группа {group}: {error}")
        return EXIT_BLOCKED

    if args.out is not None:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    show(report)

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.masking.report")
    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument("--masking-config", type=Path, default=None, help="JSON конфига маскирования")
    parser.add_argument("--out", type=Path, default=None, help="куда записать отчёт JSON")

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raise SystemExit(run(build_parser().parse_args(argv)))


__all__ = ["COLUMNS", "masking_report", "show"]


if __name__ == "__main__":
    main()
