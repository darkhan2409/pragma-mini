"""
Влияет ли конечная дата генерации на уже случившуюся историю.

    python audit/2026-09-24-gen/checks/horizon.py --short p0-pilot-c --long d6-late \
        --boundary 2026-01-01

Два прогона отличаются ТОЛЬКО концом окна. Начало окна, seed,
world_seed, число клиентов и параметры одинаковы. Поэтому общий
временной префикс обязан совпадать: то, что уже случилось, от
будущей границы выгрузки зависеть не может.

Проверка не ограничивается сравнением: она ищет ПЕРВОЕ
расхождение и показывает, глобальное оно или начинается позже.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"


def tape(run: str, boundary: str) -> list[tuple]:

    table = pq.read_table(RUNS / run / "events.parquet")

    rows = list(
        zip(
            table.column("client_id").to_pylist(),
            table.column("event_time").to_pylist(),
            table.column("source").to_pylist(),
            table.column("payload").to_pylist(),
        )
    )

    return [row for row in rows if row[1] < boundary]


def main() -> int:

    parser = argparse.ArgumentParser(prog="horizon")
    parser.add_argument("--short", required=True)
    parser.add_argument("--long", required=True)
    parser.add_argument("--boundary", required=True)

    args = parser.parse_args()

    boundary = f"{args.boundary}T00:00:00+05:00"

    left = tape(args.short, boundary)
    right = tape(args.long, boundary)

    left_clients = {row[0] for row in left}
    right_clients = {row[0] for row in right}

    # Первое расхождение по общей ленте.
    first_diff = None

    for number, (one, other) in enumerate(zip(left, right)):
        if one != other:
            first_diff = {
                "index": number,
                "short": {"client": one[0], "time": one[1], "source": one[2],
                          "type": json.loads(one[3])["type"]},
                "long": {"client": other[0], "time": other[1], "source": other[2],
                         "type": json.loads(other[3])["type"]},
            }
            break

    # Расхождение по клиентам: у кого история префикса отличается.
    def per_client(rows):
        counts: Counter = Counter()
        for row in rows:
            counts[row[0]] += 1
        return counts

    left_counts = per_client(left)
    right_counts = per_client(right)

    differing = sorted(
        client for client in left_clients | right_clients
        if left_counts.get(client, 0) != right_counts.get(client, 0)
    )

    # Самое раннее событие у клиента, чья история разошлась.
    earliest_change = None

    if differing:
        sample = differing[0]
        mine = [row for row in left if row[0] == sample]
        theirs = [row for row in right if row[0] == sample]

        for one, other in zip(mine, theirs):
            if one != other:
                earliest_change = {
                    "client": sample,
                    "short": {"time": one[1], "type": json.loads(one[3])["type"]},
                    "long": {"time": other[1], "type": json.loads(other[3])["type"]},
                }
                break

        if earliest_change is None:
            earliest_change = {
                "client": sample,
                "note": "общий префикс совпал, различается только длина истории",
                "short_events": len(mine),
                "long_events": len(theirs),
            }

    report = {
        "short": args.short,
        "long": args.long,
        "boundary": args.boundary,
        "rows_short": len(left),
        "rows_long": len(right),
        "clients_short": len(left_clients),
        "clients_long": len(right_clients),
        "same_client_set": left_clients == right_clients,
        "clients_with_identical_prefix": len(left_clients) - len(differing),
        "clients_differing": len(differing),
        "first_difference": first_diff,
        "earliest_client_change": earliest_change,
        "verdict": "PASS" if not differing and len(left) == len(right) else "FAIL",
    }

    destination = AUDIT / "evidence" / f"horizon-{args.short}-vs-{args.long}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"общий префикс до {args.boundary}: {report['verdict']}")
    print(f"  строк: {len(left)} против {len(right)}")
    print(f"  клиентов: {len(left_clients)}, множество совпало: {report['same_client_set']}")
    print(f"  с одинаковым префиксом: {report['clients_with_identical_prefix']}, "
          f"разошлись: {len(differing)}")

    if first_diff:
        print(f"  первое расхождение на строке {first_diff['index']}:")
        print(f"    короткий: {first_diff['short']}")
        print(f"    длинный:  {first_diff['long']}")

    if earliest_change:
        print(f"  пример клиента: {json.dumps(earliest_change, ensure_ascii=False)}")

    print(f"-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
