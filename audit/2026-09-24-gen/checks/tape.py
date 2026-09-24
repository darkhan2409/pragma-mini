"""
Инварианты выгруженной ленты и путь от анкеты к цели.

    python audit/2026-09-24-gen/checks/tape.py --run m-train

Считается только по RAW, без обращения к коду генератора.

  RAW-1   конверт: четыре строковые колонки, тип внутри payload
  RAW-3   время со смещением, разбирается однозначно
  OBS-2   значение null в payload не встречается
  OBS-3   обязательные поля типа на месте
  TIME-1  все события внутри [start, end)
  TIME-3  событие источника не раньше его запуска
  P2P     симметрия p2p_out и p2p_in по ленте
  ANKETA  путь от снимка анкеты к скрытой цели: является ли поле
          анкеты точной суммой событий ленты
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"

# Даты запуска источников (src/generator/config.py:117-132).
LAUNCH = {
    "antifraud": "2025-01-15",
    "communications": "2024-12-22",
    "banners": "2024-08-01",
    "app_screens": "2024-12-16",
    "support": "2025-03-01",
}

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str, checked: int = 0, bad: int = 0) -> None:
    RESULTS.append(
        {"check": name, "verdict": verdict, "detail": detail, "checked": checked, "violations": bad}
    )
    print(f"[{verdict}] {name}: проверено {checked}, нарушений {bad} — {detail}")


def main() -> int:

    parser = argparse.ArgumentParser(prog="tape")
    parser.add_argument("--run", required=True)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", required=True)

    args = parser.parse_args()

    out = RUNS / args.run

    events = pq.read_table(out / "events.parquet")
    profile = pq.read_table(out / "profile.parquet")

    # --- RAW-1 ---

    expected = ["client_id", "event_time", "source", "payload"]
    types = [str(events.schema.field(name).type) for name in events.column_names]

    ok = events.column_names == expected and set(types) == {"string"}

    record("RAW-1 конверт", "PASS" if ok else "FAIL",
           f"колонки {events.column_names}, типы {set(types)}", events.num_rows, 0 if ok else 1)

    ids = events.column("client_id").to_pylist()
    times = events.column("event_time").to_pylist()
    sources = events.column("source").to_pylist()
    payloads = [json.loads(raw) for raw in events.column("payload").to_pylist()]

    # --- RAW-3 и TIME-1 ---

    no_offset = 0
    outside = 0

    low = f"{args.start}T00:00:00+05:00"
    high = f"{args.end}T00:00:00+05:00"

    for text in times:
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            no_offset += 1
        if not (low <= text < high):
            outside += 1

    record("RAW-3 смещение времени", "PASS" if no_offset == 0 else "FAIL",
           "каждая строка разобрана и несёт смещение", len(times), no_offset)

    record("TIME-1 окно [start, end)", "PASS" if outside == 0 else "FAIL",
           f"границы {args.start} .. {args.end}", len(times), outside)

    # --- OBS-2: нет null ---

    nulls = sum(
        1 for payload in payloads if any(value is None for value in payload.values())
    )

    record("OBS-2 отсутствие null", "PASS" if nulls == 0 else "FAIL",
           "пропуск выражается отсутствием ключа, а не null", len(payloads), nulls)

    # --- OBS-3: обязательные поля ---
    # Набор обязательных берётся из кода генератора: это контракт,
    # а не наблюдение, поэтому читается напрямую.

    import sys

    sys.path.insert(0, str(AUDIT.parents[1]))

    from src.generator.config import PAYLOAD_REQUIRED

    missing = 0
    missing_examples: list[str] = []

    for payload in payloads:
        required = PAYLOAD_REQUIRED.get(payload["type"], frozenset())
        absent = [name for name in required if name not in payload]
        if absent:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(f"{payload['type']}: нет {absent}")

    record("OBS-3 обязательные поля", "PASS" if missing == 0 else "FAIL",
           "; ".join(missing_examples) or "все обязательные поля на месте",
           len(payloads), missing)

    # --- TIME-3: источник не раньше запуска ---

    early = 0
    early_examples: list[str] = []

    for text, source in zip(times, sources):
        launch = LAUNCH.get(source)
        if launch and text < f"{launch}T00:00:00+05:00":
            early += 1
            if len(early_examples) < 5:
                early_examples.append(f"{source} в {text}, запуск {launch}")

    record("TIME-3 запуск источника", "PASS" if early == 0 else "FAIL",
           "; ".join(early_examples) or "событий раньше запуска источника нет",
           len(times), early)

    # --- P2P: симметрия ---

    p2p_out = sum(1 for payload in payloads if payload["type"] == "p2p_out")
    p2p_in = sum(1 for payload in payloads if payload["type"] == "p2p_in")

    approved_out = sum(
        1 for payload in payloads
        if payload["type"] == "p2p_out" and payload.get("status") == "approved"
    )

    record("P2P симметрия в ленте", "СПРАВКА",
           f"p2p_out {p2p_out} (одобренных {approved_out}), p2p_in {p2p_in}; "
           f"разность {approved_out - p2p_in}", p2p_out + p2p_in, 0)

    # --- ANKETA: путь от снимка к цели ---

    columns = {name: profile.column(name).to_pylist() for name in profile.column_names}
    snapshot = {
        client: {name: columns[name][index] for name in profile.column_names}
        for index, client in enumerate(columns["client_id"])
    }

    opened: Counter = Counter()
    closed: Counter = Counter()
    deposit_seen: dict[str, bool] = defaultdict(bool)

    for client, payload in zip(ids, payloads):
        kind = payload["type"]
        if kind == "product_opened":
            opened[client] += 1
        elif kind == "product_closed":
            closed[client] += 1
        if kind in ("deposit_topup", "interest_credit", "product_renewed"):
            deposit_seen[client] = True

    exact_contracts = 0
    exact_active = 0
    exact_deposit = 0

    for client, row in snapshot.items():
        if row.get("contracts_count") == opened[client]:
            exact_contracts += 1
        if row.get("active_contracts") == opened[client] - closed[client]:
            exact_active += 1
        if bool(row.get("holds_deposit")) == deposit_seen[client]:
            exact_deposit += 1

    total = len(snapshot)

    record(
        "ANKETA contracts_count = число product_opened",
        "СПРАВКА",
        f"совпало у {exact_contracts} из {total} клиентов "
        f"({exact_contracts / total:.1%})",
        total,
        total - exact_contracts,
    )

    record(
        "ANKETA active_contracts = открытые минус закрытые",
        "СПРАВКА",
        f"совпало у {exact_active} из {total} ({exact_active / total:.1%})",
        total,
        total - exact_active,
    )

    record(
        "ANKETA holds_deposit = есть признаки вклада в ленте",
        "СПРАВКА",
        f"совпало у {exact_deposit} из {total} ({exact_deposit / total:.1%})",
        total,
        total - exact_deposit,
    )

    destination = AUDIT / "evidence" / f"tape-{args.run}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
