"""
F-12: ответ клиента не выгружается ни одним полем.

    python audit/2026-09-24-fix/checks/f12.py --runs regression/train regression/val

Скрытый исход эпизода разыгрывается при подготовке клиента, до
самой операции. Банк в момент своего решения его не знает.

Проверяется по готовой ленте, без обращения к внутреннему
состоянию: ни одно поле ни одного события не имеет права нести
ни само значение ответа, ни поле, которое прежде его несло.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"

# Слова, которыми назывался скрытый ответ клиента
# (life/fraud.py, прежние значения fraud_decision.resolution и
# причины разблокировки карты).
FORBIDDEN = {"confirmed_by_client", "no_response", "denied_by_client"}


def main() -> int:

    parser = argparse.ArgumentParser(prog="f12")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="evidence/f12.json")

    args = parser.parse_args()

    report = {}

    for run in args.runs:

        table = pq.read_table(RUNS / run / "events.parquet")

        payloads = [json.loads(raw) for raw in table.column("payload").to_pylist()]

        decisions = [item for item in payloads if item["type"] == "fraud_decision"]

        with_field = [item for item in decisions if "resolution" in item]

        guilty = [
            (item["type"], name, value)
            for item in payloads
            for name, value in item.items()
            if isinstance(value, str) and value in FORBIDDEN
        ]

        unblock = Counter(
            item.get("reason") for item in payloads if item["type"] == "card_unblocked"
        )

        report[run] = {
            "events": len(payloads),
            "fraud_decision": len(decisions),
            "decisions_with_resolution": len(with_field),
            "fields_naming_the_answer": len(guilty),
            "examples": guilty[:3],
            "card_unblocked_reasons": dict(unblock),
            "verdict": "PASS" if not with_field and not guilty else "FAIL",
        }

    report["verdict"] = (
        "PASS"
        if all(item["verdict"] == "PASS" for key, item in report.items() if key != "verdict")
        else "FAIL"
    )

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for run, item in report.items():
        if run == "verdict":
            continue
        print(
            f"{run}: решений антифрода {item['fraud_decision']}, из них с resolution "
            f"{item['decisions_with_resolution']}; полей с ответом клиента "
            f"{item['fields_naming_the_answer']}; причины разблокировки "
            f"{item['card_unblocked_reasons']} — {item['verdict']}"
        )

    print("ИТОГ:", report["verdict"])

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
