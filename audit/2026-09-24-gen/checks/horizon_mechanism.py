"""
Почему конец окна меняет уже случившееся: что именно разошлось.

    python audit/2026-09-24-gen/checks/horizon_mechanism.py

Одно и то же сообщество готовится дважды — с окном до 2026-01-01
и до 2026-09-01, при одном seed и одном world_seed. Сравнивается
то, что разыгрывается ДО дня симуляции: паузы, жизненные события,
потоки дохода и заранее рассчитанные выплаты.

Если расходится уже подготовка, значит конец окна входит в
розыгрыш планов, а не только обрезает ленту.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT / "harness"))

from sources import verify  # noqa: E402

from src.generator import config, params as params_module, rng as rng_module  # noqa: E402
from src.generator.simulate import CommunitySimulation  # noqa: E402
from src.generator.world import communities  # noqa: E402


def prepare(end: str, clients: int = 64) -> dict:
    """
    Подготовка первого сообщества при заданном конце окна.

    Повторяет активацию из emit._worker_init: горизонт, параметры,
    состояние RNG.
    """

    config.activate_horizon(
        datetime.fromisoformat("2024-01-01"), datetime.fromisoformat(end)
    )

    settings = params_module.load(None)
    params_module.activate(settings)
    rng_module.configure(100, settings.fingerprint(), 42)

    members = communities.members(0, clients)

    sim = CommunitySimulation(0, members)

    out = {}

    for ordinal in sorted(sim.clients):

        state = sim.clients[ordinal]

        out[state.client_id] = {
            "pauses": [
                (str(getattr(item, "start", item)), str(getattr(item, "end", "")))
                for item in getattr(state, "pauses", [])
            ],
            "life_events": [
                (str(getattr(item, "kind", "?")), str(getattr(item, "ts", getattr(item, "at", ""))))
                for item in getattr(state, "life_events", [])
            ],
            "income_streams": [
                (
                    str(getattr(item, "kind", "?")),
                    str(getattr(item, "payer", "")),
                    str(getattr(item, "landing", "")),
                )
                for item in getattr(state, "income_streams", [])
            ],
            "payouts": [
                (str(getattr(item, "ts", "")), int(getattr(item, "amount", 0)))
                for item in getattr(state, "payouts", [])
            ],
            "stress_episodes": len(getattr(state, "stress_episodes", [])),
            "fraud_episodes": len(getattr(state, "fraud_episodes", [])),
        }

    return out


def main() -> int:

    code_state = verify("до сравнения")

    short = prepare("2026-01-01")
    long = prepare("2026-09-01")

    # Вернуть горизонт по умолчанию.
    config.activate_horizon(
        datetime.fromisoformat("2024-01-01"), datetime.fromisoformat("2026-09-01")
    )

    shared = sorted(set(short) & set(long))

    fields = ("pauses", "life_events", "income_streams", "stress_episodes", "fraud_episodes")

    differing: dict[str, list[str]] = {name: [] for name in fields}
    payout_prefix_differs: list[str] = []
    payout_count: list[str] = []

    for client in shared:

        for name in fields:
            if short[client][name] != long[client][name]:
                differing[name].append(client)

        mine, theirs = short[client]["payouts"], long[client]["payouts"]

        if len(mine) != len(theirs):
            payout_count.append(client)

        # Общий префикс выплат: до конца КОРОТКОГО окна.
        limit = "2026-01-01"
        head_mine = [item for item in mine if item[0] < limit]
        head_theirs = [item for item in theirs if item[0] < limit]

        if head_mine != head_theirs:
            payout_prefix_differs.append(client)

    example = None

    if payout_prefix_differs:
        client = payout_prefix_differs[0]
        mine = [item for item in short[client]["payouts"] if item[0] < "2026-01-01"][:3]
        theirs = [item for item in long[client]["payouts"] if item[0] < "2026-01-01"][:3]
        example = {"client": client, "short": mine, "long": theirs}

    report = {
        "code_state": code_state,
        "clients_compared": len(shared),
        "same_client_set": set(short) == set(long),
        "differing": {name: len(items) for name, items in differing.items()},
        "payouts_count_differs": len(payout_count),
        "payouts_prefix_differs": len(payout_prefix_differs),
        "example": example,
        "verdict": "подготовка расходится"
        if any(differing.values()) or payout_prefix_differs
        else "подготовка совпадает",
    }

    destination = AUDIT / "evidence" / "horizon-mechanism.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"клиентов сравнено: {len(shared)}, множество совпало: {report['same_client_set']}")
    print(f"расходится подготовка: {report['differing']}")
    print(f"выплат разное число у {len(payout_count)} клиентов")
    print(f"общий префикс выплат (до 2026-01-01) различается у {len(payout_prefix_differs)}")

    if example:
        print(f"пример: {json.dumps(example, ensure_ascii=False)}")

    print(f"вердикт: {report['verdict']}")
    print(f"-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
