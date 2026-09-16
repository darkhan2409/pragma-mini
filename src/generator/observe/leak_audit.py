from __future__ import annotations

import json
import math
from collections import defaultdict

from ..config import FORBIDDEN_RAW_FIELDS
from ..rng import stable_hash


# ============================================================
# АУДИТ УТЕЧЕК
# ============================================================
#
# Скрытые характеристики, сценарии и готовые ответы не должны
# попадать в наблюдаемые данные ни колонкой, ни ключом payload,
# ни значением.
#
# Отдельно ищутся PROXY-утечки: наблюдаемый признак, который
# восстанавливает скрытую характеристику почти однозначно.
# Такой признак не запрещён сам по себе, но обязан быть назван.
# ============================================================


def forbidden_names(columns: list, payload_keys: set) -> list:
    """
    Запрещённые имена среди колонок и ключей payload.
    """

    found = []

    for name in sorted(set(columns) | set(payload_keys)):
        if name in FORBIDDEN_RAW_FIELDS:
            found.append(name)

    return found


def forbidden_values(rows: list, truth: dict, sample: int = 2000) -> list:
    """
    Значение скрытой характеристики не должно встречаться
    в payload как есть.
    """

    problems = []

    for index, row in enumerate(rows[:sample]):

        client = truth.get(row.get("client_id"))

        if not client:
            continue

        payload = row.get("payload")

        if isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload, ensure_ascii=False, default=str)

        for name, value in client.items():

            if not name.startswith("trait_"):
                continue

            token = f"{round(float(value), 4)}"

            if len(token) >= 6 and token in text:
                problems.append(f"{row['client_id']}: {name} встречается в payload")

    return sorted(set(problems))


def _bucket(value: float, buckets: int = 5) -> int:
    return min(buckets - 1, max(0, int(value * buckets)))


def _entropy(counts: dict) -> float:

    total = sum(counts.values())

    if total <= 0:
        return 0.0

    value = 0.0

    for count in counts.values():
        if count <= 0:
            continue
        share = count / total
        value -= share * math.log(share, 2)

    return value


def mutual_information(pairs: list) -> float:
    """
    Взаимная информация между скрытой корзиной и наблюдаемым
    признаком, в битах.
    """

    joint: dict = defaultdict(int)
    left: dict = defaultdict(int)
    right: dict = defaultdict(int)

    for hidden, observed in pairs:
        joint[(hidden, observed)] += 1
        left[hidden] += 1
        right[observed] += 1

    total = len(pairs)

    if total <= 0:
        return 0.0

    value = 0.0

    for (hidden, observed), count in joint.items():
        share = count / total
        value += share * math.log(
            share / ((left[hidden] / total) * (right[observed] / total)), 2
        )

    return value


def proxy_report(
    features_by_client: dict,
    truth: dict,
    threshold: float = 0.35,
) -> list:
    """
    Пары «скрытая характеристика — наблюдаемый признак» с
    подозрительно высокой взаимной информацией.

    Признак, восстанавливающий скрытое почти однозначно, это
    потенциальная proxy-утечка, и отчёт обязан её назвать.
    """

    if not features_by_client:
        return []

    names = sorted(
        {name for client in truth.values() for name in client if name.startswith("trait_")
         and not name.startswith("trait_final_")}
    )

    feature_names = sorted(
        {name for features in features_by_client.values() for name in features}
    )

    rows = []

    for trait in names:

        for feature in feature_names:

            pairs = []

            for client_id, features in features_by_client.items():

                hidden = truth.get(client_id, {}).get(trait)
                observed = features.get(feature)

                if hidden is None or observed is None:
                    continue

                pairs.append((_bucket(float(hidden)), _bucket_feature(observed)))

            if len(pairs) < 30:
                continue

            info = mutual_information(pairs)

            reference = _entropy({value: count for value, count in _counts(pairs)}) or 1.0

            share = info / reference if reference > 0 else 0.0

            if share >= threshold:
                rows.append(
                    {
                        "trait": trait,
                        "feature": feature,
                        "mutual_information_bits": round(info, 4),
                        "share_of_entropy": round(share, 4),
                        "clients": len(pairs),
                    }
                )

    rows.sort(key=lambda item: -item["share_of_entropy"])

    return rows


def _counts(pairs: list) -> list:

    counts: dict = defaultdict(int)

    for hidden, _ in pairs:
        counts[hidden] += 1

    return list(counts.items())


def _bucket_feature(value) -> int:

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if value <= 0:
            return 0
        return min(5, int(math.log10(value + 1) * 2))

    return stable_hash(str(value)) % 7


__all__ = [
    "forbidden_names",
    "forbidden_values",
    "mutual_information",
    "proxy_report",
]
