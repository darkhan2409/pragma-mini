from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group
from src.temporal.position import TemporalError
from src.temporal.samples import SamplesError, TemporalGroup
from src.tokenization.finalvocab import FrozenArtifacts, VocabError
from src.tokenization.keyvocab import KeyVocabError, load_key_vocab
from src.tokenization.names import Names
from src.tokenization.schema import SemanticSchema

from .choose import EVENT, KEY, VALUE, FieldGroups, Selection, choose, field_groups, value_chance
from .settings import ConfigError, MaskingConfig
from .weights import WeightsError, load_value_weights


# ============================================================
# ОТЧЁТ О ЦЕЛЯХ МАСКИ
# ============================================================
#
#   python -m src.masking.report train|val|test [--masking-config путь] [--out файл.json] [--pairs-row-groups N]
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
#
# partners — цели, чьё значение видно по другому полю того же события.
# Пара (B, A) — ключи одного события с H(B|A)/H(B) ниже порога: по A
# значение B почти определено. Цель ключа B считается, если в её
# событии значение A видно (не [MASK] и не [UNK]); covered говорит,
# лежат ли оба ключа пары в одной группе конфига, то есть закрывает ли
# её замыкание группы. Пары ищутся ТОЛЬКО по train, для любой группы
# отчёта: статистика обучается на train и не подстраивается под val и test.
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


# Пара считается зависимой при H(B|A)/H(B) ниже порога.
PAIR_THRESHOLD = 0.05

# Условная энтропия, посчитанная по частотам, занижена, когда на значение
# A приходится мало событий: у A, уникального в каждом событии, она ноль
# при любом B. Пара без этого запаса событий на значение A не оценивается.
EVENTS_PER_VALUE = 20


def _events_by_key(row_groups: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Значения train по ключам: номера событий и коды значений, по первым
    row_groups группам строк. Значение — кортеж его токенов, код один на
    кортеж. Все события, не только допустимые для целей: зависимость
    полей свойство данных. Повтор ключа в событии оставляет последнее
    значение, как в аудите.
    """

    samples = TemporalGroup("train")

    codes: dict[tuple[int, ...], int] = {}
    found: dict[int, tuple[list[int], list[int]]] = {}

    event_number = 0

    for number in range(min(row_groups, samples.count)):

        for row in samples.row_group(number, columns=["key_ids", "value_ids", "positions", "event_starts", "event_lengths"]).to_pylist():

            key_ids, value_ids, positions = row["key_ids"], row["value_ids"], row["positions"]

            for start, length in zip(row["event_starts"], row["event_lengths"]):

                end = start + length
                values: dict[int, tuple[int, ...]] = {}

                index = start + 1

                while index < end:

                    # Значение открывает позиция 0, закрывает следующее открытие.
                    stop = index + 1

                    while stop < end and positions[stop] != 0:
                        stop += 1

                    values[key_ids[index]] = tuple(value_ids[index:stop])
                    index = stop

                for key_id, tokens in values.items():
                    events, ids = found.setdefault(key_id, ([], []))
                    events.append(event_number)
                    ids.append(codes.setdefault(tokens, len(codes)))

                event_number += 1

    return {key_id: (np.asarray(events, dtype=np.int64), np.asarray(ids, dtype=np.int64))
            for key_id, (events, ids) in found.items()}


def _entropy(counts: np.ndarray) -> float:

    share = counts / counts.sum()

    return float(-(share * np.log(share)).sum())


def dependent_pairs(row_groups: int = 8, threshold: float = PAIR_THRESHOLD) -> list[tuple[str, str, float]]:
    """
    Пары (B, A, H(B|A)/H(B)) ключей одного события с отношением ниже
    threshold, по первым row_groups группам строк train. Всегда train,
    для какой бы группы ни строился отчёт: статистику нельзя брать по val
    и test. Порядок: по отношению, затем по именам.
    """

    names = {number: name for name, number in load_key_vocab().items()}

    events = _events_by_key(row_groups)

    pairs = []

    for b in sorted(events):
        for a in sorted(events):

            if a == b:
                continue

            both, in_a, in_b = np.intersect1d(events[a][0], events[b][0], assume_unique=True, return_indices=True)

            if not both.size:
                continue

            _, code_a, count_a = np.unique(events[a][1][in_a], return_inverse=True, return_counts=True)
            _, code_b, count_b = np.unique(events[b][1][in_b], return_inverse=True, return_counts=True)

            # Постоянное B нечего предсказывать.
            if count_b.size < 2 or both.size < EVENTS_PER_VALUE * count_a.size:
                continue

            joint = np.unique(code_a * count_b.size + code_b, return_counts=True)[1]

            ratio = max(0.0, _entropy(joint) - _entropy(count_a)) / _entropy(count_b)

            if ratio < threshold:
                pairs.append((names[b], names[a], ratio))

    return sorted(pairs, key=lambda pair: (pair[2], pair[0], pair[1]))


def _with_visible_partner(selection: Selection, partners: dict[int, set[int]]) -> int:
    """
    Сколько целей (без испорченных в [UNK]) имеют в том же событии видимое
    значение ключа-партнёра. Скрыто то, что выбрано: оно стало [MASK] или [UNK].
    """

    if not partners or not selection.picked.size:
        return 0

    found = selection.found

    hidden = set(found.start[selection.picked].tolist())
    needed = set().union(*partners.values())

    visible: dict[int, set[int]] = {}

    for event, key_id, start in zip(found.event.tolist(), found.key_id.tolist(), found.start.tolist()):
        if key_id in needed and start not in hidden:
            visible.setdefault(event, set()).add(key_id)

    chosen = found.take(selection.picked)

    return sum(
        1
        for event, key_id, unknown in zip(chosen.event.tolist(), chosen.key_id.tolist(), selection.unknown.tolist())
        if not unknown and partners.get(key_id, set()) & visible.get(event, set())
    )


def _covered(pair_ids: list[tuple[int, int]], fields: FieldGroups | None) -> list[bool]:
    """
    Лежат ли оба ключа каждой пары в одной группе конфига.
    """

    if fields is None:
        return [False] * len(pair_ids)

    return [bool(fields.of_key[b] >= 0 and fields.of_key[b] == fields.of_key[a]) for b, a in pair_ids]


def _positive(text: str) -> int:

    try:
        number = int(text)
    except ValueError:
        number = 0

    if number < 1:
        raise argparse.ArgumentTypeError(f"нужно целое число групп строк не меньше 1, получено {text!r}")

    return number


def masking_report(group: str, masking: MaskingConfig, pairs_row_groups: int = 8) -> dict:
    """
    Цели маски группы по ключам.
    """

    samples = TemporalGroup(group)

    weights = load_value_weights() if masking.informativeness_weighted_masking else None

    # Группы полей — один раз на отчёт, по словарю ключей текущего словаря.
    fields = None

    vocab = load_key_vocab()

    if masking.field_groups or masking.event_type_event_only:
        fields = field_groups(masking, vocab, SemanticSchema.open())

    pairs = dependent_pairs(pairs_row_groups)

    # Ключ B -> ключи A, по которым его значение видно.
    partners: dict[int, set[int]] = {}

    for b, a, _ in pairs:
        partners.setdefault(vocab[b], set()).add(vocab[a])

    visible_partner = 0

    names = Names(FrozenArtifacts.load())

    data: dict[int, Counter] = {}
    targets: dict[int, Counter] = {}
    reasons: dict[int, Counter] = {}
    chances: dict[int, float] = {}
    spoiled: Counter = Counter()

    for number in range(samples.count):

        for row in samples.row_group(number, columns=COLUMNS).to_pylist():

            selection = choose(group, row, masking, weights, fields)

            visible_partner += _with_visible_partner(selection, partners)

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
        "partners": {
            "pairs": pairs,
            "covered": _covered([(vocab[b], vocab[a]) for b, a, _ in pairs], fields),
            "targets_with_visible_partner": visible_partner,
            "share": visible_partner / hits if hits else 0.0,
        },
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

    partners = report["partners"]

    print(f"    целей с видимым партнёром по паре: {partners['targets_with_visible_partner']} ({partners['share']:.3f})")

    open_pairs = [pair for pair, covered in zip(partners["pairs"], partners["covered"]) if not covered]

    if open_pairs:
        print("    пары вне групп (ключ ← партнёр, H(B|A)/H(B)):")

    for b, a, ratio in open_pairs:
        print(f"        {b} ← {a}  {ratio:.3f}")


def run(args) -> int:

    group = normalize_group(args.group)

    try:
        masking = MaskingConfig.load(args.masking_config)
        report = masking_report(group, masking, args.pairs_row_groups)
    except (ConfigError, KeyVocabError, SamplesError, TemporalError, VocabError, WeightsError) as error:
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
    parser.add_argument("--pairs-row-groups", type=_positive, default=8,
                        help="сколько групп строк train читать при поиске зависимых пар")

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raise SystemExit(run(build_parser().parse_args(argv)))


__all__ = ["COLUMNS", "dependent_pairs", "masking_report", "show"]


if __name__ == "__main__":
    main()
