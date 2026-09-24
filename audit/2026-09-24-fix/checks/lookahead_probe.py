"""
Что видит решение о трате в момент расхождения.

    python audit/2026-09-24-fix/checks/lookahead_probe.py \
        --clients 48 --seed 100 --end 2025-01-05 --client c248404220627 \
        --moment 2025-01-01T09:01:53 --out evidence/lookahead-2025-01-05.json

Оборачивается Ledger.available_at: записываются проводки, которые
на этот момент уже лежат в книге и датированы будущим. Обёртка
ничего не меняет — только читает, поэтому прогон остаётся тем же.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

sys.path.insert(0, str(ROOT))


def main() -> int:

    parser = argparse.ArgumentParser(prog="lookahead_probe")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, default=42)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--client", required=True)
    parser.add_argument("--moment", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    from src.generator import config, engine
    from src.generator import params as params_module
    from src.generator import rng as rng_module
    from src.generator.finance import ledger as ledger_module
    from src.generator.world import communities

    settings = params_module.load(None)

    config.activate_horizon(
        datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    )
    params_module.activate(settings)
    rng_module.configure(args.seed, settings.fingerprint(), args.world_seed)

    target = datetime.fromisoformat(args.moment).replace(tzinfo=None)

    captured: list = []

    original = ledger_module.Ledger.available_at

    def watching(self, account_id, ts):

        result = original(self, account_id, ts)

        if self.client_id == args.client and ts.replace(tzinfo=None) == target:
            recent = self.recent.get(account_id, [])
            captured.append(
                {
                    "account_id": account_id,
                    "ts": ts.isoformat(),
                    "available_at": result,
                    "account_available": self.accounts[account_id].available,
                    "recent": [
                        {
                            "ts": item.ts.isoformat(),
                            "amount": item.amount,
                            "debit": item.debit,
                            "credit": item.credit,
                            "reason": getattr(item, "reason", None),
                        }
                        for item in recent
                    ],
                }
            )

        return result

    ledger_module.Ledger.available_at = watching

    try:
        for community_id in range(communities.community_count(args.clients)):
            members = communities.members(community_id, args.clients)
            engine.run_community(community_id, members)
    finally:
        ledger_module.Ledger.available_at = original

    path = AUDIT / args.out

    path.write_text(json.dumps(captured, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"вызовов в этот момент: {len(captured)} -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
