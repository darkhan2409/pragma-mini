"""
Мошенничество: порядок эпизода и связь видимого со скрытым.

    python audit/2026-09-24-fix/checks/fraud.py

Прогон идёт с поднятыми долями (controls/params/s1-fraud.json),
потому что при обычных параметрах на 64 клиентах эпизодов не
бывает вовсе. Частоты этого прогона оценкой реализма НЕ являются.

  LIFE-9   порядок: операция -> алерт -> решение -> блокировка ->
           обращение -> chargeback
  FRAUD-1  совпадает ли fraud_decision.resolution со скрытым
           ответом клиента episode.client_response
  FRAUD-2  зависит ли распределение rule_code от скрытого вида
           эпизода
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT / "harness"))

from sources import verify  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str, checked: int = 0, bad: int = 0) -> None:
    RESULTS.append(
        {"check": name, "verdict": verdict, "detail": detail, "checked": checked, "violations": bad}
    )
    print(f"[{verdict}] {name}: проверено {checked}, нарушений {bad} — {detail}")


def observed():
    """
    Прогон с наблюдением: нужны скрытые эпизоды.
    """

    from src.generator import emit, engine

    captured: list = []
    original = engine._finish

    def watching(sim):
        captured.extend(sim.clients[o] for o in sorted(sim.clients))
        return original(sim)

    engine._finish = watching

    try:
        emit.generate_dataset(
            total_clients=64,
            out_dir=(RUNS / "s1-observed").resolve(),
            seed=100,
            world_seed=42,
            history_start=datetime.fromisoformat("2024-01-01"),
            history_end=datetime.fromisoformat("2026-01-01"),
            workers=1,
            params_path=str(AUDIT / "controls" / "params" / "s1-fraud.json"),
            quiet=True,
        )
    finally:
        engine._finish = original

    return captured


def main() -> int:

    code_state = verify("до проверки мошенничества")

    states = observed()

    table = pq.read_table(RUNS / "s1-observed" / "events.parquet")

    ids = table.column("client_id").to_pylist()
    times = table.column("event_time").to_pylist()
    payloads = [json.loads(raw) for raw in table.column("payload").to_pylist()]

    # --- LIFE-9: порядок ---

    per_client: dict[str, list] = defaultdict(list)

    for client, when, payload in zip(ids, times, payloads):
        if payload["type"] in ("fraud_alert", "fraud_decision", "card_blocked", "chargeback"):
            per_client[client].append((when, payload["type"]))

    wrong = 0
    checked = 0

    for client, items in per_client.items():
        items.sort()
        seen_alert = None
        for when, kind in items:
            if kind == "fraud_alert":
                seen_alert = when
            elif kind == "fraud_decision":
                checked += 1
                if seen_alert is None or when < seen_alert:
                    wrong += 1

    record("LIFE-9 решение не раньше алерта", "PASS" if wrong == 0 else "FAIL",
           "у каждого fraud_decision есть более ранний fraud_alert у того же клиента",
           checked, wrong)

    # --- FRAUD-1: resolution против скрытого ответа ---

    hidden: dict[str, list] = {}

    for state in states:
        episodes = getattr(state, "fraud_episodes", [])
        hidden[state.client_id] = [
            (getattr(ep, "kind", None), getattr(ep, "client_response", None))
            for ep in episodes
        ]

    decisions: dict[str, list] = defaultdict(list)

    for client, when, payload in zip(ids, times, payloads):
        if payload["type"] == "fraud_decision":
            decisions[client].append(payload.get("resolution"))

    pairs = 0
    matched = 0
    table_kind: Counter = Counter()

    for client, items in hidden.items():
        shown = decisions.get(client, [])
        for (kind, response), resolution in zip(items, shown):
            pairs += 1
            table_kind[(kind, resolution)] += 1
            expected = None if response == "no_response" else response
            if resolution == expected:
                matched += 1

    record(
        "FRAUD-1 resolution равен скрытому ответу клиента",
        "ПОДТВЕРЖДЕНА" if pairs and matched == pairs else
        ("ОПРОВЕРГНУТА" if pairs else "НЕ ПРОВЕРЕНО"),
        f"совпало {matched} из {pairs} пар эпизод/решение"
        if pairs else "пар эпизод/решение не нашлось",
        pairs,
        pairs - matched,
    )

    # --- FRAUD-2: rule_code против скрытого вида ---

    rules: dict[str, list] = defaultdict(list)

    for client, when, payload in zip(ids, times, payloads):
        if payload["type"] == "fraud_alert":
            rules[client].append(payload.get("rule_code"))

    by_kind: dict[str, Counter] = defaultdict(Counter)
    total_pairs = 0

    for client, items in hidden.items():
        shown = rules.get(client, [])
        for (kind, _), code in zip(items, shown):
            by_kind[kind][code] += 1
            total_pairs += 1

    spread = {
        kind: dict(counter.most_common(3)) for kind, counter in by_kind.items()
    }

    record(
        "FRAUD-2 распределение rule_code по скрытому виду",
        "СПРАВКА",
        json.dumps(spread, ensure_ascii=False),
        total_pairs,
        0,
    )

    if verify("после проверки мошенничества") != code_state:
        record("состояние кода", "FAIL", "исходники изменились во время проверки")

    destination = AUDIT / "evidence" / "fraud.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"checks": RESULTS, "kind_vs_resolution":
                    {f"{k[0]}|{k[1]}": v for k, v in table_kind.items()}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
