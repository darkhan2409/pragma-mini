"""
Последовательность ключей розыгрыша одного сообщества.

    python audit/2026-09-24-fix/checks/key_probe.py \
        --clients 48 --seed 100 --end 2025-01-05 --community 0 \
        --out evidence/keys-2025-01-05.txt

Два прогона с разным концом окна обязаны дать один и тот же
поток ключей до тех пор, пока не кончится короткое окно. Первое
расхождение указывает на розыгрыш, который зависит от границы
выгрузки.

С --trace N дополнительно печатается место вызова N-го ключа.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

sys.path.insert(0, str(ROOT))


def main() -> int:

    parser = argparse.ArgumentParser(prog="key_probe")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, default=42)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--community", type=int, default=0)
    parser.add_argument("--trace", type=int, default=None)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    from src.generator import config, engine
    from src.generator import params as params_module
    from src.generator import rng as rng_module
    from src.generator.world import communities

    settings = params_module.load(None)

    config.activate_horizon(
        datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    )
    params_module.activate(settings)
    rng_module.configure(args.seed, settings.fingerprint(), args.world_seed)

    keys: list[str] = []

    original = rng_module.KeyedRandom.__init__

    def watching(self, key):
        keys.append(",".join(str(int(item)) for item in key))
        if args.trace is not None and len(keys) - 1 == args.trace:
            print(f"--- ключ {args.trace}: {keys[-1]}")
            traceback.print_stack()
        original(self, key)

    rng_module.KeyedRandom.__init__ = watching

    try:
        engine.run_community(
            args.community, communities.members(args.community, args.clients)
        )
    finally:
        rng_module.KeyedRandom.__init__ = original

    path = AUDIT / args.out

    path.write_text("\n".join(keys), encoding="utf-8")

    print(f"ключей {len(keys)} -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
