"""
Сравнение двух прогонов: где именно они совпадают и где нет.

    python audit/2026-09-24-gen/checks/determinism.py --left p0-pilot-c --right d1-repeat

Сравнение послойное, потому что «файлы равны побайтово» и «данные
равны» — разные утверждения, и при расхождении важно знать, на
каком слое оно появилось:

    files      байты файлов
    schema     имена колонок, типы, nullable
    rows       число строк
    order      порядок строк (client_id, event_time, source)
    values     значения колонок как текст
    payload    разобранный JSON: те же ключи и значения
    text       текстовое представление payload (порядок ключей)

Порядок ключей JSON контрактом не обещан, поэтому payload и text
разделены: расхождение только в text — это не расхождение данных.

Ключ --prefix сравнивает только общий временной префикс: нужен
для D6, где окна разной длины и хвост длинного прогона сравнивать
не с чем.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"


def sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def load(run: str, table: str):
    return pq.read_table(RUNS / run / f"{table}.parquet")


def rows_of(table) -> list[tuple]:
    """
    Строки как кортежи колонок, в порядке файла.
    """

    columns = [table.column(name).to_pylist() for name in table.column_names]

    return list(zip(*columns))


def compare_table(left_run: str, right_run: str, table: str, cutoff: str | None) -> dict:

    left = load(left_run, table)
    right = load(right_run, table)

    result: dict = {"table": table}

    result["schema"] = left.schema.equals(right.schema, check_metadata=False)

    left_rows = rows_of(left)
    right_rows = rows_of(right)

    if cutoff is not None and table == "events":
        # Общий префикс: только события строго раньше границы.
        when = left.column_names.index("event_time")
        left_rows = [row for row in left_rows if row[when] < cutoff]
        right_rows = [row for row in right_rows if row[when] < cutoff]
        result["prefix_cutoff"] = cutoff

    result["rows_left"] = len(left_rows)
    result["rows_right"] = len(right_rows)
    result["rows"] = len(left_rows) == len(right_rows)

    if table == "events":

        key = lambda row: (row[0], row[1], row[2])  # noqa: E731

        result["order"] = [key(row) for row in left_rows] == [
            key(row) for row in right_rows
        ]

        payload_at = left.column_names.index("payload")

        result["text"] = [row[payload_at] for row in left_rows] == [
            row[payload_at] for row in right_rows
        ]

        if len(left_rows) == len(right_rows):
            result["payload"] = all(
                json.loads(one[payload_at]) == json.loads(other[payload_at])
                for one, other in zip(left_rows, right_rows)
            )
        else:
            result["payload"] = False

        result["values"] = left_rows == right_rows

    else:
        result["order"] = [row[0] for row in left_rows] == [row[0] for row in right_rows]
        result["values"] = left_rows == right_rows

    return result


def main() -> int:

    parser = argparse.ArgumentParser(prog="determinism")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument(
        "--prefix",
        default=None,
        help="сравнить только события раньше этой границы, например 2026-01-01",
    )
    parser.add_argument("--expect", choices=["same", "differ"], default="same")
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    left, right = RUNS / args.left, RUNS / args.right

    cutoff = None

    if args.prefix:
        # Строки event_time сравнимы лексикографически: один формат
        # и одно смещение у всех записей.
        cutoff = f"{args.prefix}T00:00:00+05:00"

    files = {}

    for name in ("events.parquet", "profile.parquet", "manifest.json"):
        files[name] = sha256(left / name) == sha256(right / name)

    tables = [
        compare_table(args.left, args.right, "events", cutoff),
        compare_table(args.left, args.right, "profile", None),
    ]

    same_data = all(
        item["schema"] and item["rows"] and item["order"] and item["values"]
        for item in tables
    )

    report = {
        "left": args.left,
        "right": args.right,
        "prefix": args.prefix,
        "expect": args.expect,
        "files_identical": files,
        "tables": tables,
        "data_identical": same_data,
        "verdict": "PASS"
        if (same_data == (args.expect == "same"))
        else "FAIL",
    }

    destination = (
        Path(args.out)
        if args.out
        else AUDIT / "evidence" / f"determinism-{args.left}-vs-{args.right}.json"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{args.left} vs {args.right}: {report['verdict']} (ожидалось {args.expect})")
    print(f"  байты: {files}")

    for item in tables:
        marks = {
            name: item[name]
            for name in ("schema", "rows", "order", "values", "payload", "text")
            if name in item
        }
        print(f"  {item['table']}: строк {item['rows_left']}/{item['rows_right']} {marks}")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
