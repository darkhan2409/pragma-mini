from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Маленькая выгрузка в формате генератора: тот же конверт событий,
# та же схема анкеты и manifest. Строится во временном каталоге.

LOCAL = timezone(timedelta(hours=5))
UTC = timezone.utc

PROFILE_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("as_of", pa.timestamp("us", tz="UTC")),
        ("birth_date", pa.date32()),
        ("gender", pa.string()),
        ("family_status", pa.string()),
        ("children", pa.int32()),
        ("education", pa.string()),
        ("region", pa.string()),
        ("city", pa.string()),
        ("housing_type", pa.string()),
        ("income_type", pa.string()),
        ("declared_income", pa.int64()),
        ("industry", pa.string()),
        ("income_day", pa.int32()),
        ("relationship_months", pa.int32()),
        ("contracts_count", pa.int32()),
        ("active_contracts", pa.int32()),
        ("holds_credit_card", pa.bool_()),
        ("holds_debit_card", pa.bool_()),
        ("holds_deposit", pa.bool_()),
        ("credit_limit", pa.float64()),
        ("credit_utilization", pa.float64()),
        (
            "employment",
            pa.list_(pa.struct([("start_date", pa.date32()), ("record_time", pa.timestamp("us", tz="UTC"))])),
        ),
        (
            "lifelong",
            pa.list_(
                pa.struct(
                    [
                        ("type", pa.string()),
                        ("event_time", pa.timestamp("us", tz="UTC")),
                        ("source_id", pa.string()),
                    ]
                )
            ),
        ),
    ]
)


def iso(moment: datetime) -> str:
    """
    Время, как его пишет генератор: местное со смещением, миллисекунды
    только там, где они есть.
    """
    local = moment.astimezone(LOCAL)
    text = local.isoformat(timespec="milliseconds" if local.microsecond else "seconds")
    return text


def event(client: str, moment: datetime, source: str, **payload) -> dict:
    body = {key: value for key, value in payload.items() if value is not None}
    return {"client_id": client, "event_time": iso(moment), "source": source, "payload": json.dumps(body, ensure_ascii=False)}


def purchase(client: str, moment: datetime, amount: int = 1000, **extra) -> dict:
    fields = {"type": "purchase", "channel": "pos", "reason": "purchase", "amount": amount, "direction": "debit", "status": "approved", "mcc": "5411", "is_online": False, "is_subscription": False}
    fields.update(extra)
    return event(client, moment, "transactions", **fields)


def login(client: str, moment: datetime, session: str = "ses_1", status: str = "success") -> dict:
    return event(client, moment, "app_operations", type="app_operation", operation="login", status=status, session_id=session, domain="auth")


def salary(client: str, moment: datetime, amount: int = 300000) -> dict:
    return event(client, moment, "transactions", type="salary_credit", channel="system", reason="salary", amount=amount, direction="credit", status="approved", balance_after=amount)


def push(client: str, moment: datetime, template: str = "DEP_STANDARD") -> dict:
    return event(client, moment, "communications", type="communication_sent", channel="push", template=template, delivered=True)


def profile_change(client: str, moment: datetime, field: str, old: str | None, new: str | None, source: str = "client") -> dict:
    return event(client, moment, "profile", type="profile_change", field_name=field, old_value=old, new_value=new, change_source=source, confirmed=True)


def profile(client: str, as_of: datetime, **overrides) -> dict:
    row = {
        "client_id": client,
        "as_of": as_of.astimezone(UTC),
        "birth_date": date(1990, 6, 15),
        "gender": "F",
        "family_status": "single",
        "children": 0,
        "education": "higher",
        "region": "Almaty",
        "city": "Almaty",
        "housing_type": "rented",
        "income_type": "employed",
        "declared_income": 300000,
        "industry": "it",
        "income_day": 10,
        "relationship_months": 40,
        "contracts_count": 2,
        "active_contracts": 2,
        "holds_credit_card": False,
        "holds_debit_card": True,
        "holds_deposit": False,
        "credit_limit": None,
        "credit_utilization": None,
        "employment": [{"start_date": date(2020, 1, 1), "record_time": datetime(2020, 1, 5, tzinfo=UTC)}],
        "lifelong": [{"type": "bank_registered", "event_time": datetime(2021, 1, 1, tzinfo=UTC), "source_id": None}],
    }
    row.update(overrides)
    return row


def write_group(raw_dir: Path, group: str, period_end: datetime, events: list[dict], profiles: list[dict], row_group_size: int = 4) -> Path:
    """
    Выгрузка одной группы: события по клиенту и времени, как у генератора.
    Малый row_group_size заставляет клиентов пересекать границы групп строк.
    """
    out = raw_dir / group
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(events, key=lambda row: (row["client_id"], datetime.fromisoformat(row["event_time"])))
    table = pa.Table.from_pylist(
        ordered,
        schema=pa.schema([(name, pa.string()) for name in ("client_id", "event_time", "source", "payload")]),
    )
    pq.write_table(table, out / "events.parquet", row_group_size=row_group_size)
    pq.write_table(pa.Table.from_pylist(profiles, schema=PROFILE_SCHEMA), out / "profile.parquet")
    manifest = {
        "period_start": "2024-01-01T00:00:00+05:00",
        "period_end": period_end.astimezone(LOCAL).isoformat(),
        "events_sha256": f"synthetic-{group}",
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out
