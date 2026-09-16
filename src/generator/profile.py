from __future__ import annotations

import pyarrow as pa

from .config import PROFILE_FIELDS


# ============================================================
# ВЕРСИОННЫЙ ПРОФИЛЬ
# ============================================================
#
# Профиль это состояние с историей: строка на версию, с
# границами действия, временем записи, источником значения,
# признаком подтверждённости и причиной изменения.
#
# Месячные снимки генератор не пишет: они строятся отчётом
# калибровки из версий по любой дате.
# ============================================================


PROFILE_FIELD_TYPES: dict[str, pa.DataType] = {
    "age": pa.int32(),
    "gender": pa.string(),
    "family_status": pa.string(),
    "children": pa.int32(),
    "education": pa.string(),
    "region": pa.string(),
    "city": pa.string(),
    "housing_type": pa.string(),
    "pensioner": pa.bool_(),
    "income_type": pa.string(),
    "declared_income": pa.int64(),
    "industry": pa.string(),
    "salary_day": pa.int32(),
    "relationship_months": pa.int32(),
    "contracts_count": pa.int32(),
    "active_contracts": pa.int32(),
    "holds_credit_card": pa.bool_(),
    "holds_debit_card": pa.bool_(),
    "holds_deposit": pa.bool_(),
    "credit_limit": pa.float64(),
    "credit_utilization": pa.float64(),
}


PROFILE_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("profile_version", pa.int32()),
        ("valid_from", pa.timestamp("us")),
        ("valid_to", pa.timestamp("us")),
        ("record_time", pa.timestamp("us")),
        ("change_source", pa.string()),
        ("confirmed", pa.bool_()),
        ("change_reason", pa.string()),
    ]
    + [(name, PROFILE_FIELD_TYPES[name]) for name in PROFILE_FIELDS]
)


def as_of(versions: list, ts, record_time=None) -> dict | None:
    """
    Профиль на дату: последняя версия, которая уже действовала
    к моменту `ts` и была известна банку к моменту `record_time`.

    `valid_to` здесь намеренно НЕ фильтрует. Новая версия
    поступает в витрину позже, чем начинает действовать, и
    закрытие предыдущей по `valid_to` оставляло бы дыру: старая
    версия уже закрыта, новая ещё не известна, и профиля нет
    вовсе. Банк в этот момент знал предыдущую версию, и именно
    её нужно вернуть.

    Смену версии выражает следующая строка: как только её
    `record_time` наступил, она и окажется последней.
    """

    chosen = None

    for row in versions:

        if row["valid_from"] > ts:
            continue

        if record_time is not None and row["record_time"] > record_time:
            continue

        if chosen is None or (
            (row["valid_from"], row["profile_version"])
            > (chosen["valid_from"], chosen["profile_version"])
        ):
            chosen = row

    return chosen


def known_at(versions: list, eval_ts) -> dict | None:
    """
    Профиль, каким его видел бы препроцессинг на `eval_ts`:
    действовал к этому моменту и уже дошёл до витрины.
    """

    return as_of(versions, eval_ts, record_time=eval_ts)


__all__ = ["PROFILE_FIELD_TYPES", "PROFILE_SCHEMA", "as_of", "known_at"]
