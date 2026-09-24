"""
LEAK-1: анкета примера описывает начало периода целей.

    python audit/2026-09-24-fix/checks/profile_border.py --root stages-root

Проверяется готовый набор примеров, а не функция восстановления.
Ожидаемое считается здесь ВТОРЫМ кодом и по ДРУГИМ данным:

  * ожидаемое — по выгрузке 01_raw: снимок анкеты плюс разбор
    payload событий как текста JSON;
  * наблюдаемое — по 05_dataset/<группа>/samples.parquet:
    profile_key_ids и profile_value_ids раскодированы обратно
    словарём в имена ключей и значений.

Совпасть они обязаны по смыслу, а не потому, что посчитаны одной
функцией: конвейер читает очищенную ленту 02_preprocessed и
типизированные колонки, здесь же читается сырой JSON.

Отдельно считается сама LEAK-1: сколько примеров несут в анкете
след события ПОСЛЕ target_start. След — это отличие значения
анкеты от значения на границе.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))

GROUPS = ("train", "val", "test")

# Поля, которые эта проверка умеет восстановить сама. Список
# независим от src: он выписан по смыслу полей, а не взят
# оттуда.
BY_CHANGE = (
    "family_status", "education", "region", "city", "housing_type",
    "income_type", "declared_income", "industry", "income_day", "children",
)

CONSTANT = ("gender",)


def expected_profile(snapshot: dict, payloads: list, border: datetime) -> dict:
    """
    Анкета клиента на border по выгрузке.

    payloads — пары (время, разобранный payload) в порядке ленты.
    """

    later = [(when, item) for when, item in payloads if when >= border]

    values: dict[str, object] = {}

    for name in CONSTANT:
        if snapshot.get(name) is not None:
            values[name] = snapshot[name]

    first: dict[str, dict] = {}

    for _, item in later:
        if item.get("type") != "profile_change":
            continue
        name = item.get("field_name")
        if name in BY_CHANGE and name not in first:
            first[name] = item

    for name in BY_CHANGE:

        item = first.get(name)

        if item is None:
            value = snapshot.get(name)
        else:
            value = item.get("old_value")
            if value is None:
                continue
            # Прежнее значение приходит строкой; числовые поля
            # сравниваются числами.
            if name in ("declared_income", "children", "income_day"):
                value = int(value)

        if value is not None:
            values[name] = value

    opens = sum(1 for _, item in later if item.get("type") == "product_opened")
    closes = sum(1 for _, item in later if item.get("type") == "product_closed")

    if snapshot.get("contracts_count") is not None:
        values["contracts_count"] = int(snapshot["contracts_count"]) - opens

    if snapshot.get("active_contracts") is not None:
        values["active_contracts"] = int(snapshot["active_contracts"]) - opens + closes

    return values


def decode(sample: dict, names: dict[int, str]) -> dict:
    """
    Токены анкеты примера обратно в пары «ключ: значение».

    Значение бывает трёх видов, и все три читаются:

      value:<ключ>=<значение>   категория как есть;
      bucket:<имя>              число попало в диапазон, и
                                сверяется попадание, а не равенство;
      [UNK]                     значения не было на train, и
                                сверяется, что его там и правда нет.

    Составное значение из кусков BPE в модельной анкете не
    встречается; если встретится, попадёт в отчёт как unreadable.
    """

    keys = list(sample["profile_key_ids"])
    values = list(sample["profile_value_ids"])
    positions = list(sample["profile_positions"])

    out: dict[str, object] = {}
    unreadable: list[str] = []

    for index, position in enumerate(positions):

        if position != 0:
            # Продолжение значения: в анкете таких быть не должно.
            unreadable.append(names.get(values[index], f"?{values[index]}"))
            continue

        key = names.get(keys[index], "")
        value = names.get(values[index], "")

        if not key.startswith("key:"):
            # Ведущий токен [USR] и прочее служебное.
            continue

        name = key[len("key:") :]

        if value.startswith("bucket:"):
            out[name] = ("bucket", value[len("bucket:") :])
            continue

        if value == "[UNK]":
            out[name] = ("unknown", None)
            continue

        if not value.startswith("value:"):
            unreadable.append(f"{name}={value}")
            continue

        head, _, text = value[len("value:") :].partition("=")

        if head != name:
            unreadable.append(f"{name}!={head}")
            continue

        out[name] = ("value", text)

    return {"values": out, "unreadable": unreadable}


def inside_bucket(value: object, entry: dict) -> bool:
    """
    Попало ли число в объявленный диапазон.
    """

    number = float(value)

    low = entry.get("min")
    high = entry.get("max")

    if low == 0.0 and high == 0.0:
        return number == 0.0

    if low is not None and number < float(low):
        return False

    if high is not None and number >= float(high):
        return False

    return True


def normalise(name: str, value: object) -> str:
    """
    Одно представление для сравнения: словарь хранит значения
    записью, и число 4 там записано как "4".
    """

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value)


def main() -> int:

    parser = argparse.ArgumentParser(prog="profile_border")
    parser.add_argument("--root", default="stages-root-v2",
                        help="каталог этапов: под runs/ или абсолютный")
    parser.add_argument("--raw", default="regression",
                        help="каталог выгрузок: под runs/ или абсолютный")
    parser.add_argument("--out", default="evidence/leak1-profile-border.json")

    args = parser.parse_args()

    # Путь можно дать и абсолютный: проверять полезно не только
    # аудитную пересборку, но и настоящий data/. Чтение, ничего
    # не пишется.
    root = Path(args.root)
    root = (root if root.is_absolute() else RUNS / root).resolve()

    raw_root = Path(args.raw)
    raw_root = (raw_root if raw_root.is_absolute() else RUNS / raw_root).resolve()

    from src.preprocessing.settings import default_windows

    windows = default_windows()

    vocab = json.loads(
        (root / "03_vocab" / "final_vocab.json").read_text(encoding="utf-8")
    )

    names = {int(number): token for token, number in vocab.items()}

    buckets = json.loads(
        (root / "03_vocab" / "buckets.json").read_text(encoding="utf-8")
    )

    known_values = {
        key: set(items)
        for key, items in json.loads(
            (root / "03_vocab" / "value_vocab.json").read_text(encoding="utf-8")
        ).items()
    }

    report: dict = {"root": str(root), "raw": str(raw_root), "groups": {}}

    failures: list[str] = []

    for group in GROUPS:

        border = windows[group].target_start

        # --- ожидаемое: по выгрузке ---

        raw = raw_root / group

        events = pq.read_table(raw / "events.parquet")

        by_client: dict[str, list] = defaultdict(list)

        for client, when, payload in zip(
            events.column("client_id").to_pylist(),
            events.column("event_time").to_pylist(),
            events.column("payload").to_pylist(),
        ):
            by_client[client].append((datetime.fromisoformat(when), json.loads(payload)))

        snapshots = {
            row["client_id"]: row
            for row in pq.read_table(raw / "profile.parquet").to_pylist()
        }

        # --- наблюдаемое: по набору примеров ---

        samples = pq.read_table(root / "05_dataset" / group / "samples.parquet").to_pylist()

        mismatched: list[dict] = []
        unreadable = 0
        compared = 0
        checked_values = 0
        by_bucket = 0
        by_unknown = 0

        for sample in samples:

            client = sample["client_id"]

            snapshot = snapshots.get(client)

            if snapshot is None:
                continue

            want = expected_profile(snapshot, by_client.get(client, []), border)

            got = decode(sample, names)

            unreadable += len(got["unreadable"])

            compared += 1

            # Сравниваются только те поля, которые конвейер
            # вообще выгружает: состав полей проверяется отдельно.
            for name, (kind, value) in sorted(got["values"].items()):

                field = name[len("profile_") :] if name.startswith("profile_") else name

                if field not in want:
                    mismatched.append(
                        {"client_id": client, "field": field, "expected": None, "got": value}
                    )
                    continue

                checked_values += 1

                expected = normalise(field, want[field])

                if kind == "value":
                    if expected != str(value):
                        mismatched.append(
                            {"client_id": client, "field": field,
                             "expected": expected, "got": value}
                        )
                    continue

                if kind == "bucket":
                    entry = buckets.get(name, {}).get(value)
                    if entry is None or not inside_bucket(want[field], entry):
                        mismatched.append(
                            {"client_id": client, "field": field,
                             "expected": expected, "got": f"bucket {value}"}
                        )
                    else:
                        by_bucket += 1
                    continue

                # [UNK]: значение обязано и правда отсутствовать
                # в словаре, обученном на train.
                if expected in known_values.get(name, ()):
                    mismatched.append(
                        {"client_id": client, "field": field,
                         "expected": expected, "got": "[UNK], хотя значение в словаре есть"}
                    )
                else:
                    by_unknown += 1

        # --- сама LEAK-1 ---
        #
        # След будущего — это анкета, отличная от состояния на
        # границе. Считается по клиентам, а не по значениям.
        leaked = len({item["client_id"] for item in mismatched})

        report["groups"][group] = {
            "target_start": border.isoformat(),
            "samples": len(samples),
            "compared": compared,
            "values_compared": checked_values,
            "checked_as_bucket": by_bucket,
            "checked_as_unknown": by_unknown,
            "mismatched_values": len(mismatched),
            "samples_with_future_trace": leaked,
            "unreadable_tokens": unreadable,
            "examples": mismatched[:5],
            "verdict": "PASS" if not mismatched and not unreadable else "FAIL",
        }

        if mismatched or unreadable:
            failures.append(group)

        print(
            f"{group}: примеров {len(samples)}, сверено значений {checked_values}, "
            f"расхождений {len(mismatched)}, примеров со следом будущего {leaked} — "
            f"{report['groups'][group]['verdict']}"
        )

    report["verdict"] = "PASS" if not failures else "FAIL"

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("ИТОГ:", report["verdict"])

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
