"""
Путь от решения антифрода до наблюдаемого исхода.

    python audit/2026-09-24-fix/checks/fraud_path.py --run s1-observed

Постоянство блокировки вычисляется из скрытого вида эпизода и
разыгранного заранее ответа клиента
(engine_products.py:1158). Само по себе это не утечка: скрытым
остаётся то, что банк узнаёт позже. Утечкой было бы, если бы
исход становился виден в ленте РАНЬШЕ, чем банк успел бы его
узнать.

Поэтому измеряется задержка: сколько проходит от решения до
разблокировки и до перевыпуска, и успевает ли клиент обратиться
раньше.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"


def main() -> int:

    parser = argparse.ArgumentParser(prog="fraud_path")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="evidence/fraud-path.json")

    args = parser.parse_args()

    table = pq.read_table(RUNS / args.run / "events.parquet")

    rows = [
        (client, datetime.fromisoformat(when), json.loads(raw))
        for client, when, raw in zip(
            table.column("client_id").to_pylist(),
            table.column("event_time").to_pylist(),
            table.column("payload").to_pylist(),
        )
    ]

    by_client: dict[str, list] = {}

    for client, when, payload in rows:
        by_client.setdefault(client, []).append((when, payload))

    outcomes: Counter = Counter()
    unblock_hours: list[float] = []
    reissue_days: list[float] = []
    case_first = 0
    pairs = 0

    for client, items in by_client.items():

        items.sort(key=lambda item: item[0])

        decisions = [item for item in items if item[1]["type"] == "fraud_decision"]

        for when, _ in decisions:

            later = [item for item in items if item[0] > when]

            unblock = next(
                (item for item in later
                 if item[1]["type"] == "card_unblocked"
                 and item[1].get("reason") == "fraud_check_closed"),
                None,
            )
            reissue = next(
                (item for item in later if item[1]["type"] == "card_reissued"), None
            )
            case = next(
                (item for item in later if item[1]["type"] == "case_opened"), None
            )

            if unblock is None and reissue is None:
                outcomes["исход в окно не попал"] += 1
                continue

            pairs += 1

            if unblock is not None and (reissue is None or unblock[0] < reissue[0]):
                outcomes["разблокировка"] += 1
                unblock_hours.append((unblock[0] - when).total_seconds() / 3600)
                edge = unblock[0]
            else:
                outcomes["перевыпуск"] += 1
                reissue_days.append((reissue[0] - when).total_seconds() / 86400)
                edge = reissue[0]

            if case is not None and case[0] < edge:
                case_first += 1

    def spread(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "min": round(ordered[0], 2),
            "median": round(ordered[len(ordered) // 2], 2),
            "max": round(ordered[-1], 2),
        }

    report = {
        "run": args.run,
        "decisions_with_outcome": pairs,
        "outcomes": dict(outcomes),
        "unblock_delay_hours": spread(unblock_hours),
        "reissue_delay_days": spread(reissue_days),
        "cases_before_outcome": case_first,
        "note": (
            "исход становится виден только действием банка и не раньше "
            "измеренной задержки; обращение клиента успевает встать перед "
            f"исходом в {case_first} случаях из {pairs}"
        ),
    }

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
