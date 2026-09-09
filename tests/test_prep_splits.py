"""
Сплиты: группа клиента зависит только от seed и client_id.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from src.preprocessing.config import CLIENT_GROUPS, DATASETS
from src.preprocessing.splits import (
    assign_groups,
    boundaries,
    client_group,
    dataset_for,
    month_roles,
    unit_interval,
)


ROOT = Path(__file__).resolve().parents[1]

SEED = 20240601


# ============================================================
# ХЭШ
# ============================================================


def test_unit_interval_is_in_range():
    for client_id in range(1000):
        u = unit_interval(client_id, SEED)
        assert 0.0 <= u < 1.0


def test_seed_changes_assignment():
    a = assign_groups(range(200), SEED, (0.8, 0.1, 0.1))
    b = assign_groups(range(200), SEED + 1, (0.8, 0.1, 0.1))

    assert a != b


def test_shares_are_approximate():
    groups = assign_groups(range(10000), SEED, (0.8, 0.1, 0.1))

    counts = {group: sum(1 for value in groups.values() if value == group) for group in CLIENT_GROUPS}

    assert abs(counts["train"] / 10000 - 0.8) < 0.02
    assert abs(counts["val"] / 10000 - 0.1) < 0.02
    assert abs(counts["test"] / 10000 - 0.1) < 0.02


def test_adding_clients_keeps_existing_assignment():
    """
    Главное свойство: новый клиент не переселяет старых.
    """

    small = assign_groups(range(100), SEED, (0.8, 0.1, 0.1))
    large = assign_groups(range(500), SEED, (0.8, 0.1, 0.1))

    for client_id, group in small.items():
        assert large[client_id] == group


def test_groups_are_disjoint_and_complete():
    groups = assign_groups(range(300), SEED, (0.8, 0.1, 0.1))

    assert set(groups.values()) <= set(CLIENT_GROUPS)
    assert len(groups) == 300


def test_boundaries_cover_unit_interval():
    bounds = boundaries((0.8, 0.1, 0.1))

    assert bounds["train"][0] == 0.0
    assert bounds["test"][1] == 1.0

    for previous, current in zip(CLIENT_GROUPS, CLIENT_GROUPS[1:]):
        assert bounds[previous][1] == bounds[current][0]


def test_shares_must_sum_to_one():
    with pytest.raises(ValueError):
        boundaries((0.5, 0.1, 0.1))


def test_assignment_is_stable_across_processes():
    """
    SHA-256, а не hash(): при другом PYTHONHASHSEED результат
    обязан совпасть.
    """

    script = (
        "from src.preprocessing.splits import assign_groups;"
        "g = assign_groups(range(50), 20240601, (0.8, 0.1, 0.1));"
        "print(','.join(g[i] for i in range(50)))"
    )

    expected = ",".join(assign_groups(range(50), SEED, (0.8, 0.1, 0.1))[i] for i in range(50))

    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "12345"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == expected


# ============================================================
# МЕСЯЦЫ И ДАТАСЕТЫ
# ============================================================


def months(count: int) -> list[datetime]:
    return [datetime(2024, 6, 1).replace(year=2024 + (5 + index) // 12, month=(5 + index) % 12 + 1) for index in range(count)]


def test_month_roles_pick_last_two():
    grid = months(24)

    roles = month_roles(grid)

    assert roles[grid[-1]] == "test_month"
    assert roles[grid[-2]] == "val_month"
    assert all(roles[month] == "train_period" for month in grid[:-2])
    assert sum(1 for role in roles.values() if role == "train_period") == 22


def test_month_roles_need_three_months():
    with pytest.raises(ValueError):
        month_roles(months(2))


def test_month_roles_reject_unsorted():
    grid = months(5)

    with pytest.raises(ValueError):
        month_roles(list(reversed(grid)))


def test_dataset_matrix():
    assert dataset_for("train", "train_period") == "train"
    assert dataset_for("val", "train_period") == "val_client"
    assert dataset_for("test", "train_period") == "test_client"
    assert dataset_for("train", "val_month") == "val_time"
    assert dataset_for("train", "test_month") == "test_time"


def test_unused_combinations_have_no_dataset():
    for group in ("val", "test"):
        for role in ("val_month", "test_month"):
            assert dataset_for(group, role) is None


def test_dataset_matrix_has_five_entries():
    assert len(DATASETS) == 5
    assert len(set(DATASETS.values())) == 5
