from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from ..config import RAW_DIR, TIMEZONE


# ============================================================
# ЛЕНТА ОДНОГО КЛИЕНТА
# ============================================================
#
# Одна история подряд, строка за строкой, чтобы глазами
# увидеть, что деньги сходятся, а события следуют друг за
# другом по причине.
#
# Печатается только то, что есть в выгрузке: скрытых состояний
# симуляции она не содержит, и показать их неоткуда.
# ============================================================


MONEY_FIELDS = ("amount", "balance_after")


def _read(path: Path) -> list:
    return pq.read_table(path).to_pylist() if path.exists() else []


def load(raw_dir: Path, client_id: str | None) -> dict:
    """
    Лента и профиль одного клиента.
    """

    if client_id is None:
        events = _read(raw_dir / "events.parquet")
        if not events:
            raise SystemExit("в наборе нет событий")
        client_id = Counter(row["client_id"] for row in events).most_common(1)[0][0]

    events = [row for row in _read(raw_dir / "events.parquet") if row["client_id"] == client_id]

    for row in events:
        # Тип события лежит в payload: отдельной колонки у него
        # нет. Читателю он нужен строкой, поэтому достаётся здесь.
        row["payload"] = json.loads(row["payload"])
        row["type"] = row["payload"].get("type", "—")

        # Время в выгрузке строкой со смещением: отчёт
        # показывает его как есть, в местном времени банка.
        row["event_time"] = datetime.fromisoformat(row["event_time"])

    # Строки уже лежат в порядке ленты; сортировка только по
    # времени события, устойчиво, чтобы порядок не менялся.
    events.sort(key=lambda row: row["event_time"])

    return {
        "client_id": client_id,
        "events": events,
        "profile": [
            row for row in _read(raw_dir / "profile.parquet") if row["client_id"] == client_id
        ],
    }


# ------------------------------------------------------------
# ПЕЧАТЬ
# ------------------------------------------------------------


def _money(value) -> str:
    if value is None:
        return ""
    return f"{int(value):,}".replace(",", " ")


def _describe(row: dict) -> str:
    """
    Короткая суть события: деньги, продукт, контрагент.
    """

    payload = row["payload"]

    parts: list[str] = []

    status = payload.get("status")

    if status and status != "approved":
        parts.append(status.upper())

    if payload.get("amount") is not None:
        sign = "-" if payload.get("direction") == "debit" else "+"
        parts.append(f"{sign}{_money(payload['amount'])} ₸")

    for key in ("merchant_name", "counterparty", "product_id", "template", "topic",
                "firebase_screen", "operation", "decision", "reason"):
        value = payload.get(key)
        if value:
            parts.append(str(value))
            break

    if payload.get("balance_after") is not None:
        parts.append(f"остаток {_money(payload['balance_after'])}")

    if payload.get("days_past_due"):
        parts.append(f"dpd {payload['days_past_due']}")

    if payload.get("decline_reason"):
        parts.append(str(payload["decline_reason"]))

    return " · ".join(parts)


def render(data: dict, limit: int | None, since: datetime | None,
           until: datetime | None, sources: tuple, full_payload: bool) -> str:

    out: list[str] = []

    out.append(f"клиент {data['client_id']}")

    out.append("")

    # --- лента ---

    events = data["events"]

    if sources:
        events = [row for row in events if row["source"] in sources]

    if since is not None:
        events = [row for row in events if row["event_time"] >= since]

    if until is not None:
        events = [row for row in events if row["event_time"] < until]

    total = len(events)

    if limit is not None and total > limit:
        events = events[-limit:]
        out.append(f"ЛЕНТА (последние {limit} из {total})")
    else:
        out.append(f"ЛЕНТА ({total})")

    out.append("")

    month = None

    for row in events:

        current = row["event_time"].strftime("%Y-%m")

        if current != month:
            month = current
            out.append(f"  ── {month} ──")

        out.append(
            f"  {row['event_time'].strftime('%d.%m %H:%M')}  "
            f"{row['type']:<24} {_describe(row)}"
        )

        if full_payload:
            out.append(f"      {json.dumps(row['payload'], ensure_ascii=False)}")

    out.append("")

    # --- профиль ---

    if data["profile"]:
        out.append("ПРОФИЛЬ")
        for row in data["profile"]:
            out.append(
                f"  возраст {row.get('age')}, город {row.get('city')}, "
                f"доход {_money(row.get('declared_income'))}"
            )
        out.append("")

    # --- сводка ---

    types = Counter(row["type"] for row in data["events"])

    out.append("ИТОГО ПО ТИПАМ")

    for name, count in types.most_common():
        out.append(f"  {name:<28} {count}")

    return "\n".join(out)


def _date(value: str | None) -> datetime | None:
    """
    Граница отбора в том же поясе, что и выгрузка:
    иначе наивная дата не сравнится с осознанным временем
    события.
    """

    if not value:
        return None

    moment = datetime.fromisoformat(value)

    return moment.replace(tzinfo=TIMEZONE) if moment.tzinfo is None else moment


def main() -> None:

    parser = argparse.ArgumentParser(description="Лента одного клиента")

    parser.add_argument("--raw", type=Path, default=RAW_DIR / "smoke")
    parser.add_argument("--client", default=None, help="client_id; по умолчанию самый активный")
    parser.add_argument("--limit", type=int, default=200, help="0 — без ограничения")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--payload", action="store_true", help="печатать payload целиком")
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()

    data = load(args.raw, args.client)

    text = render(
        data,
        limit=args.limit or None,
        since=_date(args.since),
        until=_date(args.until),
        sources=tuple(args.source or ()),
        full_payload=args.payload,
    )

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        print(f"записано: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
