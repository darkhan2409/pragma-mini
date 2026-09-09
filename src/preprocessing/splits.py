from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Iterable

from .config import CLIENT_GROUPS, DATASETS, MONTH_ROLES


# ============================================================
# ИДЕЯ
# ============================================================
#
# Группа клиента это функция от (seed, client_id) через SHA-256.
# Добавление новых клиентов не меняет группы старых, а Python
# hash() не используется: он рандомизирован между процессами.
#
# Роли месяцев: последний полный месяц RAW это test, предыдущий
# это val, остальные это train_period.
# ============================================================


SPLIT_METHOD = "sha256"
KEY_FORMAT = "{seed}:{client_id}"
DIGEST_BYTES = 8


def unit_interval(client_id: int, seed: int) -> float:
    """
    Детерминированное число в [0, 1) для клиента.
    """

    key = KEY_FORMAT.format(seed=seed, client_id=client_id).encode("utf-8")

    digest = hashlib.sha256(key).digest()[:DIGEST_BYTES]

    return int.from_bytes(digest, "big") / float(1 << (8 * DIGEST_BYTES))


def boundaries(shares: tuple[float, float, float]) -> dict[str, tuple[float, float]]:

    if len(shares) != len(CLIENT_GROUPS):
        raise ValueError("нужны доли для train, val и test")

    if abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError(f"доли должны суммироваться в 1, получено {sum(shares)}")

    result: dict[str, tuple[float, float]] = {}

    low = 0.0

    for group, share in zip(CLIENT_GROUPS, shares):
        high = low + share
        result[group] = (low, high)
        low = high

    # Замыкаем последний интервал до 1, чтобы u=0.9999 не выпал.
    last = CLIENT_GROUPS[-1]
    result[last] = (result[last][0], 1.0)

    return result


def client_group(client_id: int, seed: int, shares: tuple[float, float, float]) -> str:

    u = unit_interval(client_id, seed)

    for group, (low, high) in boundaries(shares).items():
        if low <= u < high:
            return group

    raise AssertionError("u вне [0, 1)")


def assign_groups(client_ids: Iterable[int], seed: int, shares: tuple[float, float, float]) -> dict[int, str]:
    return {client_id: client_group(client_id, seed, shares) for client_id in sorted(client_ids)}


# ============================================================
# РОЛИ МЕСЯЦЕВ
# ============================================================


def month_roles(months: list[datetime]) -> dict[datetime, str]:
    """
    months: начала полных месяцев наблюдения по возрастанию.
    """

    if len(months) < 3:
        raise ValueError("нужно минимум три полных месяца: train, val и test")

    if months != sorted(months):
        raise ValueError("месяцы должны идти по возрастанию")

    roles = {month: "train_period" for month in months[:-2]}
    roles[months[-2]] = "val_month"
    roles[months[-1]] = "test_month"

    assert set(roles.values()) <= set(MONTH_ROLES)

    return roles


def dataset_for(group: str, role: str) -> str | None:
    return DATASETS.get((group, role))


def split_description(seed: int, shares: tuple[float, float, float]) -> dict:
    return {
        "method": SPLIT_METHOD,
        "seed": seed,
        "key_format": KEY_FORMAT,
        "digest_bytes": DIGEST_BYTES,
        "shares": dict(zip(CLIENT_GROUPS, shares)),
        "boundaries": {group: list(bounds) for group, bounds in boundaries(shares).items()},
    }
