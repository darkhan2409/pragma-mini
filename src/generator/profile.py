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
        ("change_source", pa.string()),
        ("confirmed", pa.bool_()),
        ("change_reason", pa.string()),
    ]
    + [(name, PROFILE_FIELD_TYPES[name]) for name in PROFILE_FIELDS]
)


def as_of(versions: list, ts) -> dict | None:
    """
    Профиль на дату: последняя версия, которая уже действовала
    к моменту `ts`.

    `valid_to` здесь намеренно НЕ фильтрует: смену версии
    выражает следующая строка, и как только её `valid_from`
    наступил, она и окажется последней. Закрывать предыдущую по
    `valid_to` незачем, а на границе это оставило бы дыру.
    """

    chosen = None

    for row in versions:

        if row["valid_from"] > ts:
            continue

        if chosen is None or (
            (row["valid_from"], row["profile_version"])
            > (chosen["valid_from"], chosen["profile_version"])
        ):
            chosen = row

    return chosen


def known_at(versions: list, eval_ts) -> dict | None:
    """
    Профиль, каким его видел бы препроцессинг на `eval_ts`.
    Времени поступления у выгрузки нет, поэтому это то же самое,
    что версия, действовавшая на `eval_ts`.
    """

    return as_of(versions, eval_ts)


__all__ = ["PROFILE_FIELD_TYPES", "PROFILE_SCHEMA", "as_of", "known_at"]
