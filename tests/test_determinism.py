"""
Детерминизм: одинаковые входы дают одинаковую историю
в одном процессе, в разных процессах и на диске.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.generator.config import FEATURE_END, HISTORY_START, LABEL_END
from src.generator.derive import derive_labels
from src.generator.history import STREAM_FIELDS, generate_client_history
from src.generator.persona import draw_persona
from src.generator.timeline import build_timeline


ROOT = Path(__file__).resolve().parents[1]

CLIENT = 3


HASH_SCRIPT = """
import hashlib, os, sys
sys.path.insert(0, os.getcwd())

from src.generator.config import HISTORY_START, LABEL_END
from src.generator.derive import derive_labels
from src.generator.history import generate_client_history
from src.generator.timeline import build_timeline

def digest(client_id):
    history = generate_client_history(client_id, start=HISTORY_START, end=LABEL_END)
    h = hashlib.sha256()
    h.update(repr(history).encode("utf-8"))
    h.update(repr(build_timeline(history)).encode("utf-8"))
    h.update(repr(derive_labels(history)).encode("utf-8"))
    return h.hexdigest()
"""


def test_persona_is_deterministic():
    assert draw_persona(CLIENT) == draw_persona(CLIENT)
    assert draw_persona(CLIENT) != draw_persona(CLIENT + 1)


def test_history_is_deterministic_in_process():
    first = generate_client_history(CLIENT, start=HISTORY_START, end=LABEL_END)
    second = generate_client_history(CLIENT, start=HISTORY_START, end=LABEL_END)

    assert first == second
    assert build_timeline(first) == build_timeline(second)
    assert derive_labels(first) == derive_labels(second)


def test_history_is_deterministic_across_processes():
    namespace: dict = {}
    exec(HASH_SCRIPT, namespace)
    expected = namespace["digest"](CLIENT)

    env = dict(os.environ)
    # Другой hash seed ловит зависимость от порядка обхода set/dict.
    env["PYTHONHASHSEED"] = "12345"

    result = subprocess.run(
        [sys.executable, "-c", HASH_SCRIPT + f"\nprint(digest({CLIENT}))"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == expected


def test_every_stream_is_non_empty_somewhere(raw_tables):
    for name in STREAM_FIELDS:
        assert not raw_tables[name].empty, f"{name}: пустая таблица"


def test_raw_matches_in_memory_history(raw_tables):
    """
    RAW не зависит от соседей по чанку: строки клиента на диске
    совпадают с отдельной генерацией того же клиента.
    """

    history = generate_client_history(CLIENT, start=HISTORY_START, end=LABEL_END)

    feature = history.before(FEATURE_END)

    for name in STREAM_FIELDS:

        on_disk = raw_tables[name]
        on_disk = on_disk[on_disk.client_id == CLIENT]

        expected = sorted(event.ts for event in feature.events(name))
        actual = sorted(value.to_pydatetime() for value in on_disk.ts)

        assert actual == expected, name


def test_transaction_content_matches_in_memory(raw_tables):
    history = generate_client_history(CLIENT, start=HISTORY_START, end=LABEL_END)

    feature = history.before(FEATURE_END)

    on_disk = raw_tables["transactions"]
    on_disk = on_disk[on_disk.client_id == CLIENT]

    disk_rows = sorted(
        zip(
            (value.to_pydatetime() for value in on_disk.ts),
            on_disk.amount.tolist(),
            on_disk.direction.tolist(),
            on_disk.mcc.tolist(),
        )
    )

    memory_rows = sorted(
        (event.ts, event.amount, event.direction, event.mcc)
        for event in feature.transactions
    )

    assert disk_rows == memory_rows
