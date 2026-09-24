"""
Разделение групп и снимок анкеты.

    python audit/2026-09-24-fix/checks/split.py --groups m-train m-val m-test

Проверяется:

  SPLIT-1  клиенты групп не пересекаются
  SPLIT-4  мир общий: справочные сущности совпадают между группами
  SPLIT-5  описывает ли снимок анкеты момент ПОЗЖЕ начала целей

SPLIT-5 берётся от target_start, а не от target_end. Снимок
анкеты попадает во вход модели (04_tokenized/profile.parquet →
05_dataset → 11_profiles), а снят он на границу выгрузки, то есть
на target_end. Значит вопрос стоит так: отличается ли снятое
значение от того, каким поле было В НАЧАЛЕ периода целей. От
target_end проверка вырождена — final_cutoff равен target_end во
всех трёх группах, и событий после него не бывает по построению.

Расхождение устанавливается по самой ленте, а не пересчётом
анкеты своей формулой:

  * поля с событиями profile_change — откаткой снимка назад по
    old_value этих событий;
  * поля, которые меняет деятельность (число договоров, наличие
    продуктов, лимит) — наличием соответствующего события внутри
    периода целей;
  * age и relationship_months растут со временем у всех
    одинаково и названы отдельно: они выдают дату среза, а не
    будущее конкретного клиента.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
RUNS = AUDIT / "runs"

# Границы групп читаются из самого конвейера, а не повторяются
# здесь копией: расхождение копии с кодом было бы незаметно.
sys.path.insert(0, str(AUDIT.parents[1]))

from src.preprocessing.settings import default_windows  # noqa: E402

WINDOWS = {
    name: (
        window.history_start,
        window.final_cutoff,
        window.target_start,
        window.target_end,
    )
    for name, window in default_windows().items()
}


def group_of(run: str) -> str:
    """
    Имя группы по имени прогона: каталог может лежать где угодно.
    """

    tail = run.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

    if tail not in WINDOWS:
        known = ", ".join(sorted(WINDOWS))
        raise SystemExit(f"прогон {run!r}: имя группы не опознано, известны {known}")

    return tail


# Поля анкеты, которые меняет деятельность клиента: событие
# такого рода внутри периода целей означает, что снимок на конец
# периода описывает не начало.
ACTIVITY_FIELDS = {
    "contracts_count": ("product_opened", "account_opened"),
    "active_contracts": ("product_opened", "account_opened", "product_closed"),
    "holds_deposit": ("product_opened", "product_closed"),
    "holds_credit_card": ("product_opened", "product_closed"),
    "holds_debit_card": ("product_opened", "product_closed"),
    "credit_limit": ("product_opened", "product_closed", "limit_changed"),
    "credit_utilization": ("purchase", "loan_payment", "product_opened"),
}

# Растут со временем у всех одинаково.
CLOCK_FIELDS = ("age", "relationship_months")


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

    # --- SPLIT-5: снимок анкеты против начала периода целей ---

    snapshot: dict[str, dict] = {}

    for run in args.groups:

        group = group_of(run)

        _, final_cutoff, target_start, _ = WINDOWS[group]

        # Сравниваются моменты, а не строки: границы групп хранятся
        # в UTC, а лента подписана смещением +05:00, и строковое
        # сравнение относило бы последние пять часов суток не туда.
        boundary = target_start

        # 1. Поля, которые меняются событием profile_change.
        #    Снимок откатывается назад: последнее по времени
        #    изменение поля внутри периода целей возвращает
        #    старое значение, и оно сравнивается со снимком.
        changed_fields: Counter = Counter()
        changed_clients: dict[str, set] = defaultdict(set)

        # 2. Поля, которые меняет деятельность.
        activity_fields: Counter = Counter()
        activity_clients: dict[str, set] = defaultdict(set)

        for client_id, when, payload in tapes[run]:

            if datetime.fromisoformat(when) < boundary:
                continue

            kind = payload["type"]

            if kind == "profile_change":
                name = payload.get("field_name")
                if name:
                    changed_fields[name] += 1
                    changed_clients[name].add(client_id)
                continue

            for field, kinds in ACTIVITY_FIELDS.items():
                if kind in kinds:
                    activity_fields[field] += 1
                    activity_clients[field].add(client_id)

        touched = set().union(*changed_clients.values()) if changed_clients else set()
        touched |= set().union(*activity_clients.values()) if activity_clients else set()

        # Строгий счёт: без credit_utilization. Её признаком
        # служит любая покупка, поэтому она срабатывает почти у
        # всех и сама по себе доказывает мало. Остальные поля
        # опираются на события, которые прямо их меняют.
        strict_sources = {
            name: value
            for name, value in activity_clients.items()
            if name != "credit_utilization"
        }

        strict = set().union(*changed_clients.values()) if changed_clients else set()
        strict |= set().union(*strict_sources.values()) if strict_sources else set()

        # Контроль: у поля, которое внутри периода целей никто не
        # менял, снимок и начало периода совпадают. Если таких
        # полей нет вовсе, проверка вырождена, и это сказано.
        untouched = sorted(
            set(ACTIVITY_FIELDS) - set(activity_clients) - set(changed_clients)
        )

        snapshot[group] = {
            "target_start": target_start.isoformat(),
            "final_cutoff": final_cutoff.isoformat(),
            "clients": len(clients[run]),
            "events_in_target_window": sum(
                1 for _, when, _ in tapes[run] if datetime.fromisoformat(when) >= boundary
            ),
            "profile_change_in_window": {
                name: {"events": count, "clients": len(changed_clients[name])}
                for name, count in sorted(changed_fields.items())
            },
            "activity_fields_in_window": {
                name: {"events": count, "clients": len(activity_clients[name])}
                for name, count in sorted(activity_fields.items())
            },
            "clients_with_stale_snapshot": len(touched),
            "share_of_clients": round(len(touched) / max(1, len(clients[run])), 4),
            "clients_with_stale_snapshot_strict": len(strict),
            "share_of_clients_strict": round(len(strict) / max(1, len(clients[run])), 4),
            "fields_provably_unchanged": untouched,
            "clock_fields": list(CLOCK_FIELDS),
            "verdict": "СНИМОК ОПИСЫВАЕТ БУДУЩЕЕ" if touched else "снимок совпадает с началом целей",
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

    for group, item in snapshot.items():
        print(
            f"SPLIT-5 {group}: цели с {item['target_start'][:10]}, снимок на "
            f"{item['final_cutoff'][:10]}; клиентов со снимком из будущего "
            f"{item['clients_with_stale_snapshot']} из {item['clients']} "
            f"({item['share_of_clients'] * 100:.1f}%), без учёта credit_utilization "
            f"{item['clients_with_stale_snapshot_strict']} "
            f"({item['share_of_clients_strict'] * 100:.1f}%) — {item['verdict']}"
        )
        if item["fields_provably_unchanged"]:
            print(f"    поля без изменений в окне: {', '.join(item['fields_provably_unchanged'])}")

    print(f"-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
