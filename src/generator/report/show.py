from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from ..config import RAW_DIR


# ============================================================
# ЛЕНТА ОДНОГО КЛИЕНТА
# ============================================================
#
# Отчёт реализма показывает популяцию. Здесь другое: одна
# история подряд, строка за строкой, чтобы глазами увидеть,
# что деньги сходятся, а события следуют друг за другом по
# причине.
#
# По умолчанию печатается только наблюдаемое. Скрытая истина
# показывается лишь по явному --truth и никогда не смешивается
# с лентой без пометки.
# ============================================================


MONEY_FIELDS = ("amount", "balance_after")


def _read(path: Path) -> list:
    return pq.read_table(path).to_pylist() if path.exists() else []


def load(raw_dir: Path, client_id: str | None, ordinal: int | None) -> dict:
    """
    Лента, профиль, покрытие и скрытая истина одного клиента.
    """

    truth_clients = _read(raw_dir / "truth" / "clients.parquet")

    if client_id is None:

        if ordinal is not None:
            match = [row for row in truth_clients if row.get("client_ordinal") == ordinal]
            if not match:
                raise SystemExit(f"клиента с ординалом {ordinal} нет в наборе")
            client_id = match[0]["client_id"]
        else:
            events = _read(raw_dir / "events.parquet")
            if not events:
                raise SystemExit("в наборе нет событий")
            client_id = Counter(row["client_id"] for row in events).most_common(1)[0][0]

    events = [row for row in _read(raw_dir / "events.parquet") if row["client_id"] == client_id]

    for row in events:
        row["payload"] = json.loads(row["payload"])

    events.sort(key=lambda row: (row["event_time"], row["sequence_number"]))

    return {
        "client_id": client_id,
        "events": events,
        "profile": [
            row for row in _read(raw_dir / "profile.parquet") if row["client_id"] == client_id
        ],
        "coverage": [
            row
            for row in _read(raw_dir / "source_coverage.parquet")
            if row["client_id"] == client_id
        ],
        "truth_client": next(
            (row for row in truth_clients if row["client_id"] == client_id), None
        ),
        "truth_events": [
            row
            for row in _read(raw_dir / "truth" / "events.parquet")
            if row["client_id"] == client_id
        ],
        "relationships": [
            row
            for row in _read(raw_dir / "truth" / "relationships.parquet")
            if row["client_id"] == client_id
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

    for key in ("merchant_name", "counterparty", "product_code", "template", "topic",
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
           until: datetime | None, sources: tuple, truth: bool,
           full_payload: bool) -> str:

    out: list[str] = []

    client_id = data["client_id"]

    out.append(f"клиент {client_id}")

    person = data["truth_client"]

    if person is not None:
        out.append(
            f"ординал {person.get('client_ordinal')}, сообщество {person.get('community_id')}, "
            f"архетип {person.get('archetype')}"
        )

    out.append("")

    # --- покрытие ---

    if data["coverage"]:
        out.append("ИСТОЧНИКИ")
        for row in sorted(data["coverage"], key=lambda item: item["source"]):
            seen = row["first_seen"].strftime("%Y-%m-%d") if row["first_seen"] else "—"
            out.append(
                f"  {row['source']:<16} {row['coverage_status']:<12} первое событие {seen}"
                + (f"  ({row['coverage_reason']})" if row["coverage_reason"] else "")
            )
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

        marks = []

        if row["event_version"] > 1:
            marks.append(f"v{row['event_version']}")

        if row["is_test_account"]:
            marks.append("test")

        if row["time_precision"] != "second":
            marks.append(row["time_precision"])

        delay = (row["record_time"] - row["event_time"]).total_seconds() / 3600.0

        if delay >= 24:
            marks.append(f"+{delay / 24:.0f}д")

        out.append(
            f"  {row['event_time'].strftime('%d.%m %H:%M')}  "
            f"{row['event_type']:<24} {_describe(row)}"
            + (f"   [{' '.join(marks)}]" if marks else "")
        )

        if full_payload:
            out.append(f"      {json.dumps(row['payload'], ensure_ascii=False)}")

    out.append("")

    # --- профиль ---

    if data["profile"]:
        out.append("ПРОФИЛЬ")
        for row in sorted(data["profile"], key=lambda item: item["valid_from"]):
            out.append(
                f"  v{row['profile_version']} с {row['valid_from'].strftime('%Y-%m-%d')} "
                f"({row['change_reason'] or 'начальный'}): "
                f"возраст {row.get('age')}, город {row.get('city')}, "
                f"доход {_money(row.get('declared_income'))}"
            )
        out.append("")

    # --- скрытая истина ---

    if truth:

        out.append("СКРЫТАЯ ИСТИНА (в RAW её нет)")

        if person is not None:
            traits = {
                key[len("trait_"):]: round(value, 2)
                for key, value in person.items()
                if key.startswith("trait_")
                and not key.startswith("trait_final_")
                and isinstance(value, (int, float))
            }
            if traits:
                out.append("  черты: " + ", ".join(f"{k} {v}" for k, v in sorted(traits.items())))

        for row in sorted(data["truth_events"], key=lambda item: item["ts"])[:200]:
            out.append(
                f"  {row['ts'].strftime('%d.%m.%Y')}  {row['kind']:<20} {row['key']}"
            )

        if data["relationships"]:
            out.append("  связи:")
            for row in data["relationships"]:
                out.append(
                    f"    {row['relation_type']:<24} {row['counterpart_id']} "
                    f"(сила {row['strength']:.2f})"
                )

        out.append("")

    # --- сводка ---

    types = Counter(row["event_type"] for row in data["events"])

    out.append("ИТОГО ПО ТИПАМ")

    for name, count in types.most_common():
        out.append(f"  {name:<28} {count}")

    return "\n".join(out)


def _date(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def main() -> None:

    parser = argparse.ArgumentParser(description="Лента одного клиента")

    parser.add_argument("--raw", type=Path, default=RAW_DIR / "smoke")
    parser.add_argument("--client", default=None, help="client_id; по умолчанию самый активный")
    parser.add_argument("--ordinal", type=int, default=None, help="порядковый номер из truth")
    parser.add_argument("--limit", type=int, default=200, help="0 — без ограничения")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--truth", action="store_true", help="показать скрытую истину")
    parser.add_argument("--payload", action="store_true", help="печатать payload целиком")
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()

    data = load(args.raw, args.client, args.ordinal)

    text = render(
        data,
        limit=args.limit or None,
        since=_date(args.since),
        until=_date(args.until),
        sources=tuple(args.source or ()),
        truth=args.truth,
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
