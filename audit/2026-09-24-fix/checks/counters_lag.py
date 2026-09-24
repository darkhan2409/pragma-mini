"""
Почему счётчики договоров нельзя откатить на дату.

    python audit/2026-09-24-fix/checks/counters_lag.py --runs regression/train regression/val

Счётчики анкеты пересчитываются раз в месяц, а не при каждом
открытии и закрытии договора. Значит снимок описывает последний
пересчёт анкеты, а не конец выгрузки, и вычесть из него события
периода целей нельзя: неизвестно, какие из них в него уже вошли.

Утверждение проверяется арифметикой по самой выгрузке, без
обращения к коду генератора. Если снимок относился бы к концу
выгрузки, то для каждого клиента выполнялось бы

    contracts_count - открытий_в_окне >= 0
    active_contracts - открытий + закрытий >= 0
    и первое не меньше второго,

потому что договор нельзя закрыть, не открыв. Нарушение хотя бы
у одного клиента опровергает посылку.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"


def main() -> int:

    parser = argparse.ArgumentParser(prog="counters_lag")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="evidence/profile-counters-lag.json")

    args = parser.parse_args()

    report: dict = {"runs": {}}

    for run in args.runs:

        directory = RUNS / run

        table = pq.read_table(directory / "events.parquet")

        opens: Counter = Counter()
        closes: Counter = Counter()

        for client, payload in zip(
            table.column("client_id").to_pylist(), table.column("payload").to_pylist()
        ):
            kind = json.loads(payload)["type"]
            if kind == "product_opened":
                opens[client] += 1
            elif kind == "product_closed":
                closes[client] += 1

        snapshots = pq.read_table(directory / "profile.parquet").to_pylist()

        broken: list[dict] = []

        for row in snapshots:

            client = row["client_id"]

            total = row.get("contracts_count")
            active = row.get("active_contracts")

            if total is None or active is None:
                continue

            left_total = int(total) - opens[client]
            left_active = int(active) - opens[client] + closes[client]

            if left_total >= 0 and left_active >= 0 and left_active <= left_total:
                continue

            broken.append(
                {
                    "client_id": client,
                    "snapshot_total": int(total),
                    "snapshot_active": int(active),
                    "opened_in_window": opens[client],
                    "closed_in_window": closes[client],
                    "rolled_total": left_total,
                    "rolled_active": left_active,
                }
            )

        report["runs"][run] = {
            "clients": len(snapshots),
            "impossible_rollbacks": len(broken),
            "examples": broken[:3],
            "conclusion": (
                "снимок счётчиков не относится к концу выгрузки: откат даёт "
                "невозможное состояние"
                if broken
                else "противоречий не найдено; это не доказывает, что снимок "
                "относится к концу выгрузки"
            ),
        }

        print(
            f"{run}: клиентов {len(snapshots)}, невозможных откатов "
            f"{len(broken)} — {report['runs'][run]['conclusion']}"
        )

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
