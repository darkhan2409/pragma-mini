"""
Сравнение планов, снятых при разных окнах выгрузки.

    python audit/2026-09-24-fix/checks/plan_diff.py \
        --left evidence/plan-2025-01-05.json \
        --right evidence/plan-2026-01-01.json

Планы клиента не имеют права зависеть от конца окна. Выплаты
сравниваются только до меньшей из двух границ: их список по
построению идёт до горизонта планирования, а вот попавшее в
общий отрезок обязано совпасть.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]


def walk(left, right, path=""):

    if type(left) is not type(right):
        return [(path, type(left).__name__, type(right).__name__)]

    if isinstance(left, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            out += walk(left.get(key), right.get(key), f"{path}.{key}")
        return out

    if isinstance(left, list):
        out = []
        if len(left) != len(right):
            out.append((path + "[длина]", len(left), len(right)))
        for index, (one, other) in enumerate(zip(left, right)):
            out += walk(one, other, f"{path}[{index}]")
        return out

    return [] if left == right else [(path, left, right)]


def main() -> int:

    parser = argparse.ArgumentParser(prog="plan_diff")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--boundary", default=None, help="до какой даты сравнивать выплаты")
    parser.add_argument("--out", default="evidence/f1-plans.json")

    args = parser.parse_args()

    left = json.loads((AUDIT / args.left).read_text(encoding="utf-8"))
    right = json.loads((AUDIT / args.right).read_text(encoding="utf-8"))

    boundary = args.boundary

    differing: dict[str, list] = {}
    payout_differing: dict[str, list] = {}

    for client in sorted(set(left) & set(right)):

        one = dict(left[client])
        other = dict(right[client])

        mine = one.pop("payouts", [])
        theirs = other.pop("payouts", [])

        found = walk(one, other)

        if found:
            differing[client] = found[:5]

        if boundary is not None:
            mine = [item for item in mine if item["ts"] < boundary]
            theirs = [item for item in theirs if item["ts"] < boundary]
            trouble = walk(mine, theirs)
            if trouble:
                payout_differing[client] = trouble[:5]

    report = {
        "left": args.left,
        "right": args.right,
        "clients": len(set(left) & set(right)),
        "same_client_set": sorted(left) == sorted(right),
        "clients_with_plan_differences": len(differing),
        "examples": {key: value for key, value in list(differing.items())[:3]},
        "payout_boundary": boundary,
        "clients_with_payout_differences": len(payout_differing) if boundary else None,
        "verdict": "PASS" if not differing and not payout_differing else "FAIL",
    }

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
