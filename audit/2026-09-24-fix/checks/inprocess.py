"""
F-2: несколько выгрузок в одном процессе равны отдельным.

    python audit/2026-09-24-fix/checks/inprocess.py --clients 48 --seed 100

generate_dataset ставит три процессных глобала: горизонт,
параметры и состояние розыгрыша (emit.py). Кэши розыгрыша живут
рядом с ними. Поэтому вторая выгрузка в том же процессе могла
получить величины, посчитанные под прежним окном.

Проверка идёт обоими порядками — короткое→длинное→короткое и
длинное→короткое→длинное — и сравнивает каждую выгрузку с
эталоном, снятым СВЕЖИМ процессом. Сравнение по содержимому
таблиц, а не по байтам файла: parquet вправе отличаться служебными
полями.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))


def content_digest(directory: Path) -> dict:
    """
    Отпечаток содержимого выгрузки, не зависящий от того, как
    parquet разложил его по файлу.
    """

    result = {}

    for name in ("events", "profile"):

        table = pq.read_table(directory / f"{name}.parquet")

        sha = hashlib.sha256()

        for column in table.column_names:
            sha.update(column.encode("utf-8"))
            for value in table.column(column).to_pylist():
                sha.update(repr(value).encode("utf-8"))
            sha.update(b"\x1e")

        result[name] = {"rows": table.num_rows, "sha256": sha.hexdigest()}

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    result["manifest"] = {
        key: manifest.get(key)
        for key in sorted(manifest)
        if key not in ("created_at", "elapsed_seconds")
    }

    return result


def fresh(name: str, clients: int, seed: int, start: str, end: str) -> Path:
    """
    Эталон: отдельный процесс, один прогон и ничего больше.
    """

    out = RUNS / name

    command = [
        sys.executable,
        str(AUDIT / "harness" / "run_gen.py"),
        "--name", name,
        "--clients", str(clients),
        "--seed", str(seed),
        "--start", start,
        "--end", end,
        "--workers", "1",
    ]

    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    if done.returncode != 0:
        raise SystemExit(f"эталон {name} не собран:\n{done.stdout}\n{done.stderr}")

    return out


def sequence(label: str, clients: int, seed: int, start: str, windows: list[str]) -> dict:
    """
    Несколько выгрузок подряд В ОДНОМ процессе.
    """

    from src.generator import emit

    result = {}

    for index, end in enumerate(windows):

        out = RUNS / f"{label}-{index}-{end}"

        emit.generate_dataset(
            total_clients=clients,
            out_dir=out,
            seed=seed,
            world_seed=42,
            history_start=datetime.fromisoformat(start),
            history_end=datetime.fromisoformat(end),
            workers=1,
            quiet=True,
        )

        result[f"{index}:{end}"] = content_digest(out)

    return result


def main() -> int:

    parser = argparse.ArgumentParser(prog="inprocess")
    parser.add_argument("--clients", type=int, default=48)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--short", default="2025-01-01")
    parser.add_argument("--long", default="2026-01-01")
    parser.add_argument("--out", default="evidence/f2-inprocess.json")

    args = parser.parse_args()

    baseline = {
        args.short: content_digest(
            fresh(f"f2-fresh-{args.short}", args.clients, args.seed, args.start, args.short)
        ),
        args.long: content_digest(
            fresh(f"f2-fresh-{args.long}", args.clients, args.seed, args.start, args.long)
        ),
    }

    orders = {
        "короткое-длинное-короткое": [args.short, args.long, args.short],
        "длинное-короткое-длинное": [args.long, args.short, args.long],
    }

    report = {"baseline": baseline, "orders": {}, "failures": []}

    for label, windows in orders.items():

        measured = sequence(
            label.replace("ё", "e").replace("-", "_"), args.clients, args.seed, args.start, windows
        )

        report["orders"][label] = {}

        for key, digest in measured.items():

            end = key.split(":", 1)[1]

            same = digest == baseline[end]

            report["orders"][label][key] = {
                "matches_fresh_process": same,
                "rows": digest["events"]["rows"],
                "sha256": digest["events"]["sha256"][:16],
                "expected_sha256": baseline[end]["events"]["sha256"][:16],
            }

            if not same:
                report["failures"].append(f"{label} / {key}")

    report["verdict"] = "PASS" if not report["failures"] else "FAIL"

    text = json.dumps(report, ensure_ascii=False, indent=2)

    (AUDIT / args.out).write_text(text, encoding="utf-8")

    for label, items in report["orders"].items():
        print(f"--- {label}")
        for key, item in items.items():
            mark = "совпадает" if item["matches_fresh_process"] else "РАСХОДИТСЯ"
            print(f"    {key}: строк {item['rows']}, {mark} со свежим процессом")

    print("ИТОГ:", report["verdict"])

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
