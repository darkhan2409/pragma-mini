"""
Снимок ПЛАНОВ клиента при заданном окне выгрузки.

    python audit/2026-09-24-fix/checks/plan_probe.py \
        --clients 48 --seed 100 --end 2025-01-05 --out evidence/plan-2025-01-05.json

Печатает то, что решено ДО симуляции дней: жизненные события,
стресс, паузы, мошенничество, потоки дохода, привычки, дату
прихода в банк, дату согласия. По контракту F-1 всё это не имеет
права зависеть от конца окна.

Выплаты берутся только до самой ранней сравниваемой границы:
их список по построению обрывается концом окна, и сравнивать
имеет смысл общий отрезок.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

sys.path.insert(0, str(ROOT))


def plain(value):
    """
    Любое плановое значение в сравнимый вид, без знания полей.

    Список полей не перечисляется нарочно: перечисление молча
    пропустило бы то, что как раз и уехало.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return {name: plain(getattr(value, name)) for name in sorted(f.name for f in fields(value))}

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "__dict__"):
        return {name: plain(item) for name, item in sorted(vars(value).items())}

    return repr(value)


# Планы клиента: всё, что решено ДО симуляции дней. payouts сюда
# не входит — их список по построению обрывается концом окна.
PLANNED = (
    "life_events",
    "stress_episodes",
    "pauses",
    "fraud_episodes",
    "income_streams",
    "habits",
    "traits",
    "app_adopted_at",
    "consent_at",
    "profile_values",
)


def describe(state) -> dict:

    record = {
        "client_id": state.persona.client_id,
        "persona": plain(state.persona),
    }

    for name in PLANNED:
        record[name] = plain(getattr(state, name))

    # Выплаты до общей границы: за ней короткое окно их и не
    # обязано иметь.
    record["payouts"] = plain(state.payouts)

    return record


def main() -> int:

    parser = argparse.ArgumentParser(prog="plan_probe")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, default=42)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    from src.generator import config, simulate
    from src.generator import params as params_module
    from src.generator import rng as rng_module

    settings = params_module.load(None)

    config.activate_horizon(
        datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    )
    params_module.activate(settings)
    rng_module.configure(args.seed, settings.fingerprint(), args.world_seed)

    from src.generator.world import communities

    result = {}

    for community_id in range(communities.community_count(args.clients)):

        ordinals = communities.members(community_id, args.clients)

        sim = simulate.CommunitySimulation(community_id, ordinals)

        for ordinal in ordinals:
            record = describe(sim.clients[ordinal])
            result[record["client_id"]] = record

    path = AUDIT / args.out

    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"клиентов {len(result)} -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
