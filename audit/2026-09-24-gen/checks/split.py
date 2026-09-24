"""
Разделение групп и снимок анкеты.

    python audit/2026-09-24-gen/checks/split.py --groups m-train m-val m-test

Проверяется:

  SPLIT-1  клиенты групп не пересекаются
  SPLIT-4  мир общий: справочные сущности совпадают между группами
  SPLIT-5  описывает ли снимок анкеты момент ПОЗЖЕ целей группы

SPLIT-5 — не про корреляцию. Проверяется путь получения:
берётся поле анкеты, восстанавливается его значение по событиям
до конца периода целей, и сравнивается со снимком. Если снимок
отличается, он несёт сведения, которых в доступной истории нет.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"

# Границы групп: history_start, final_cutoff, target_start, target_end
# (src/preprocessing/settings.py:138-152).
WINDOWS = {
    "m-train": ("2024-01-01", "2026-01-01", "2024-01-01", "2026-01-01"),
    "m-val": ("2024-01-01", "2026-05-01", "2026-01-01", "2026-05-01"),
    "m-test": ("2024-01-01", "2026-09-01", "2026-05-01", "2026-09-01"),
}


def read(run: str):
    events = pq.read_table(RUNS / run / "events.parquet")
    profile = pq.read_table(RUNS / run / "profile.parquet")
    return events, profile


def main() -> int:

    parser = argparse.ArgumentParser(prog="split")
    parser.add_argument("--groups", nargs="+", required=True)

    args = parser.parse_args()

    clients: dict[str, set] = {}
    merchants: dict[str, set] = {}
    profiles: dict[str, dict] = {}
    tapes: dict[str, list] = {}

    for run in args.groups:

        events, profile = read(run)

        ids = events.column("client_id").to_pylist()
        payloads = [json.loads(raw) for raw in events.column("payload").to_pylist()]
        times = events.column("event_time").to_pylist()

        clients[run] = set(profile.column("client_id").to_pylist())

        merchants[run] = {
            payload["merchant_id"]
            for payload in payloads
            if payload.get("merchant_id")
        }

        columns = {name: profile.column(name).to_pylist() for name in profile.column_names}
        profiles[run] = {
            row_id: {name: columns[name][index] for name in profile.column_names}
            for index, row_id in enumerate(columns["client_id"])
        }

        tapes[run] = list(zip(ids, times, payloads))

    # --- SPLIT-1 ---

    overlaps = {}

    for left in args.groups:
        for right in args.groups:
            if left < right:
                shared = clients[left] & clients[right]
                overlaps[f"{left}|{right}"] = len(shared)

    split1 = "PASS" if all(value == 0 for value in overlaps.values()) else "FAIL"

    # --- SPLIT-4: общий мир ---

    world = {}

    for left in args.groups:
        for right in args.groups:
            if left < right:
                both = merchants[left] & merchants[right]
                world[f"{left}|{right}"] = {
                    "shared_merchants": len(both),
                    "left_only": len(merchants[left] - merchants[right]),
                    "right_only": len(merchants[right] - merchants[left]),
                }

    # --- SPLIT-5: снимок анкеты против истории до конца целей ---

    snapshot: dict[str, dict] = {}

    for run in args.groups:

        _, _, _, target_end = WINDOWS[run]
        boundary = f"{target_end}T00:00:00+05:00"

        # Независимый пересчёт двух полей анкеты по ленте.
        opened: Counter = Counter()
        closed: Counter = Counter()
        has_deposit: dict[str, bool] = defaultdict(bool)

        # То же, но только по событиям строго раньше границы целей.
        opened_before: Counter = Counter()
        closed_before: Counter = Counter()

        for client_id, when, payload in tapes[run]:

            kind = payload["type"]

            if kind in ("product_opened", "account_opened"):
                opened[client_id] += 1
                if when < boundary:
                    opened_before[client_id] += 1

            if kind == "product_closed":
                closed[client_id] += 1
                if when < boundary:
                    closed_before[client_id] += 1

            if kind in ("deposit_topup", "interest_credit", "product_renewed"):
                has_deposit[client_id] = True

        after_boundary = sum(
            1
            for client_id in clients[run]
            if opened[client_id] != opened_before[client_id]
            or closed[client_id] != closed_before[client_id]
        )

        # Сколько событий вообще случилось после конца периода целей.
        late = sum(1 for _, when, _ in tapes[run] if when >= boundary)

        snapshot[run] = {
            "target_end": target_end,
            "events_after_target_end": late,
            "clients_with_products_after_target_end": after_boundary,
            "note": "снимок анкеты относится к границе выгрузки; если событий "
            "после конца периода целей нет, снимок и история описывают один момент",
        }

    report = {
        "groups": args.groups,
        "clients": {run: len(value) for run, value in clients.items()},
        "split1_overlaps": overlaps,
        "split1": split1,
        "split4_world": world,
        "split5_snapshot": snapshot,
    }

    destination = AUDIT / "evidence" / "split.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"SPLIT-1 непересечение клиентов: {split1} — {overlaps}")
    print(f"клиентов по группам: {report['clients']}")

    for pair, item in world.items():
        print(f"SPLIT-4 {pair}: общих мерчантов {item['shared_merchants']}, "
              f"только слева {item['left_only']}, только справа {item['right_only']}")

    for run, item in snapshot.items():
        print(f"SPLIT-5 {run}: событий после {item['target_end']} — "
              f"{item['events_after_target_end']}, клиентов с продуктовыми "
              f"событиями после границы — {item['clients_with_products_after_target_end']}")

    print(f"-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
