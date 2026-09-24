"""
F-1: продление окна выгрузки сохраняет прежнюю историю.

    python audit/2026-09-24-fix/checks/prefix.py \
        --short runs/w-2025 --long runs/w-2026 --boundary 2025-01-01

Контракт: при одинаковых seed, мире и начальной дате более
длинное окно обязано содержать короткое целиком — те же клиенты,
те же события до прежней границы, в том же порядке, с теми же
суммами и тем же payload. Добавляться имеет право только
продолжение.

Из контракта вычтены последние RECENT_WINDOW суток окна. Решение
о трате смотрит на остаток «с учётом ближайших двух суток»
(finance/ledger.py:69, available_at), а у короткого окна этих
суток нет. Это край обрезанной ленты, а не зависимость планов от
границы: проверка требует точного совпадения до края и отдельно
измеряет, что именно расходится внутри него.

Сравнение независимо от проверяемого кода: читается готовый
parquet, а не внутреннее состояние симуляции, и равенство
считается своим кодом, а не функциями генератора.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

COLUMNS = ("client_id", "event_time", "source", "payload")


def rows(directory: Path) -> list[tuple]:

    table = pq.read_table(directory / "events.parquet")

    missing = [name for name in COLUMNS if name not in table.column_names]

    if missing:
        raise SystemExit(f"{directory}: в events.parquet нет колонок {missing}")

    columns = [table.column(name).to_pylist() for name in COLUMNS]

    return list(zip(*columns))


def moment(text: str) -> datetime:
    return datetime.fromisoformat(text)


def compare(short: list[tuple], long: list[tuple], boundary: datetime, edge: timedelta) -> dict:
    """
    Короткая лента против префикса длинной.
    """

    cut = [item for item in long if moment(item[1]) < boundary]

    # Край: последние edge суток окна. Внутри него расхождение
    # объяснимо заглядыванием вперёд, до него — нет.
    edge_from = boundary - edge

    report = {
        "short_rows": len(short),
        "long_rows": len(long),
        "long_prefix_rows": len(cut),
        "clients_short": len({item[0] for item in short}),
        "clients_long": len({item[0] for item in long}),
    }

    only_short = {item[0] for item in short} - {item[0] for item in long}
    only_long_prefix = {item[0] for item in cut} - {item[0] for item in short}

    report["clients_lost"] = sorted(only_short)[:10]
    report["clients_appeared_in_prefix"] = sorted(only_long_prefix)[:10]

    # 1. Построчное равенство в том же порядке.
    order_diff = None

    for index, (left, right) in enumerate(zip(short, cut)):
        if left != right:
            order_diff = {
                "index": index,
                "short": {name: value for name, value in zip(COLUMNS, left)},
                "long": {name: value for name, value in zip(COLUMNS, right)},
            }
            break

    report["first_difference_in_order"] = order_diff
    report["same_row_count"] = len(short) == len(cut)

    # Главное число: расхождения СТРОГО ДО края.
    before_edge = [
        index
        for index, (left, right) in enumerate(zip(short, cut))
        if left != right and moment(left[1]) < edge_from
    ]

    inside_edge = [
        index
        for index, (left, right) in enumerate(zip(short, cut))
        if left != right and moment(left[1]) >= edge_from
    ]

    report["edge_from"] = edge_from.isoformat()
    report["rows_before_edge"] = sum(1 for item in short if moment(item[1]) < edge_from)
    report["differences_before_edge"] = len(before_edge)
    report["differences_inside_edge"] = len(inside_edge)

    # Что именно расходится на краю: поля, а не только счётчик.
    fields: dict[str, int] = {}

    for index in inside_edge:
        left = json.loads(short[index][3])
        right = json.loads(cut[index][3])
        for name in set(left) | set(right):
            if left.get(name) != right.get(name):
                fields[name] = fields.get(name, 0) + 1

    report["edge_changed_fields"] = dict(sorted(fields.items()))

    # И осталась ли на краю та же последовательность событий:
    # различаться значениям позволено, набору — нет.
    report["edge_identity_preserved"] = all(
        short[index][0] == cut[index][0]
        and short[index][1] == cut[index][1]
        and json.loads(short[index][3])["type"] == json.loads(cut[index][3])["type"]
        for index in inside_edge
    )

    # 2. Равенство как мультимножеств: отдельно от порядка, чтобы
    #    отличить «переставили» от «изменили».
    left_set: dict[tuple, int] = {}
    right_set: dict[tuple, int] = {}

    for item in short:
        left_set[item] = left_set.get(item, 0) + 1
    for item in cut:
        right_set[item] = right_set.get(item, 0) + 1

    missing = [item for item in left_set if right_set.get(item, 0) != left_set[item]]
    extra = [item for item in right_set if left_set.get(item, 0) != right_set[item]]

    report["rows_missing_in_long"] = len(missing)
    report["rows_extra_in_long_prefix"] = len(extra)
    report["missing_examples"] = [
        {name: value for name, value in zip(COLUMNS, item)} for item in missing[:3]
    ]
    report["extra_examples"] = [
        {name: value for name, value in zip(COLUMNS, item)} for item in extra[:3]
    ]

    # 3. Суммы и payload отдельно: равенство строк уже их
    #    покрывает, но при расхождении нужно знать, деньги это
    #    или только структура.
    def money(items: list[tuple]) -> int:
        total = 0
        for item in items:
            payload = json.loads(item[3])
            value = payload.get("amount")
            if isinstance(value, (int, float)):
                total += int(value)
        return total

    report["amount_sum_short"] = money(short)
    report["amount_sum_long_prefix"] = money(cut)
    report["amount_sum_equal"] = report["amount_sum_short"] == report["amount_sum_long_prefix"]

    # 4. Семантическое равенство payload: текст мог совпасть
    #    случайно, порядок ключей — нет.
    semantic = 0

    for left, right in zip(short, cut):
        if json.loads(left[3]) != json.loads(right[3]):
            semantic += 1

    report["payload_semantic_differences"] = semantic

    report["verdict"] = (
        "PASS"
        if (
            report["same_row_count"]
            and not before_edge
            and not only_short
            and not only_long_prefix
        )
        else "FAIL"
    )

    # Отдельный, более сильный вердикт: совпадение вообще всего,
    # включая край. Он не входит в контракт и нужен как мера.
    report["verdict_including_edge"] = (
        "PASS"
        if report["verdict"] == "PASS"
        and order_diff is None
        and not missing
        and not extra
        and semantic == 0
        and report["amount_sum_equal"]
        else "FAIL"
    )

    return report


def profiles(short: Path, long: Path) -> dict:
    """
    Анкета — снимок на границу выгрузки, поэтому меняющиеся поля
    расходиться вправе. Неизменяемые — нет.
    """

    left = pq.read_table(short / "profile.parquet").to_pylist()
    right = pq.read_table(long / "profile.parquet").to_pylist()

    by_id_left = {row["client_id"]: row for row in left}
    by_id_right = {row["client_id"]: row for row in right}

    fixed = ("birth_date", "gender", "city", "client_id")

    broken = []

    # Клиент, пришедший в банк после прежней границы, в коротком
    # прогоне отсутствует по построению: это продолжение, а не
    # потеря. Обратное — потеря клиента — запрещено.
    for key, row in by_id_left.items():
        other = by_id_right.get(key)
        if other is None:
            broken.append({"client_id": key, "reason": "клиента нет в длинном прогоне"})
            continue
        for name in fixed:
            if name in row and row.get(name) != other.get(name):
                broken.append(
                    {"client_id": key, "field": name, "short": row.get(name), "long": other.get(name)}
                )

    return {
        "clients_short": len(by_id_left),
        "clients_long": len(by_id_right),
        "short_is_subset": set(by_id_left) <= set(by_id_right),
        "clients_added": sorted(set(by_id_right) - set(by_id_left)),
        "immutable_field_differences": len(broken),
        "examples": broken[:5],
        "verdict": "PASS" if not broken and set(by_id_left) <= set(by_id_right) else "FAIL",
    }


def main() -> int:

    parser = argparse.ArgumentParser(prog="prefix")
    parser.add_argument("--short", required=True)
    parser.add_argument("--long", required=True)
    parser.add_argument("--boundary", required=True, help="конец короткого окна, YYYY-MM-DD")
    parser.add_argument(
        "--edge-days", type=int, default=2,
        help="ширина края ленты; по умолчанию RECENT_WINDOW генератора",
    )
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    short = (AUDIT / args.short).resolve()
    long = (AUDIT / args.long).resolve()

    boundary = datetime.fromisoformat(args.boundary)

    # Граница в ленте записана со смещением пояса, а в аргументе
    # его нет: берётся смещение из самой ленты, чтобы сравнение
    # не зависело от того, как именно записан аргумент.
    sample = rows(long)[0][1]
    tzinfo = datetime.fromisoformat(sample).tzinfo

    boundary = boundary.replace(tzinfo=tzinfo)

    result = {
        "short": str(short),
        "long": str(long),
        "boundary": boundary.isoformat(),
        "edge_days": args.edge_days,
        "events": compare(rows(short), rows(long), boundary, timedelta(days=args.edge_days)),
        "profile": profiles(short, long),
    }

    result["verdict"] = (
        "PASS"
        if result["events"]["verdict"] == "PASS" and result["profile"]["verdict"] == "PASS"
        else "FAIL"
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)

    print(text)

    if args.out:
        (AUDIT / args.out).write_text(text, encoding="utf-8")

    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
