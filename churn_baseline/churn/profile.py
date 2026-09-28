from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd


# ============================================================
# АНКЕТА НА CUTOFF T
# ============================================================
#
# Выгрузка несёт один снимок анкеты на её границу as_of, а она позже T.
# Снимок откатывается назад по ленте: поле, которое меняет событие
# profile_change, на T равно прежнему значению (old_value) первого
# изменения в момент T или позже. Изменений после T нет — снимок и есть
# значение на T.
#
# Остальное считается на T из датированных фактов снимка:
#   age                  полных лет на местную дату T по birth_date;
#   job_tenure_months    полных месяцев на последней работе по найму,
#                        о которой банк узнал строго раньше T;
#   days_since_<веха>    дней от вехи lifelong строго раньше T.
#
# Снимочные поля contracts_count, active_contracts, holds_*, credit_limit,
# credit_utilization и relationship_months описывают клиента на as_of,
# то есть после T. Это будущее, и в признаки они не идут.
# ============================================================


# Поле меняет только событие profile_change, и событие несёт прежнее значение.
CHANGED_BY_EVENT: tuple[str, ...] = (
    "family_status",
    "education",
    "region",
    "city",
    "housing_type",
    "income_type",
    "declared_income",
    "industry",
    "income_day",
    "children",
)

CONSTANT: tuple[str, ...] = ("gender",)

NUMERIC_FIELDS: tuple[str, ...] = ("declared_income", "income_day", "children")

# Снимок на as_of: состояние договоров и стаж на конец выгрузки.
SNAPSHOT_ONLY: tuple[str, ...] = (
    "contracts_count",
    "active_contracts",
    "holds_credit_card",
    "holds_debit_card",
    "holds_deposit",
    "credit_limit",
    "credit_utilization",
    "relationship_months",
)

MILESTONES: tuple[str, ...] = (
    "bank_registered",
    "app_registered",
    "first_card_activated",
    "first_loan_opened",
    "first_deposit_opened",
)

SALARIED: tuple[str, ...] = ("employed", "state_employee")

CATEGORICAL: tuple[str, ...] = tuple(
    name for name in CONSTANT + CHANGED_BY_EVENT if name not in NUMERIC_FIELDS
)

MISSING = "NA"

DESCRIPTIONS: dict[str, str] = {
    "gender": "пол (анкета)",
    "family_status": "семейное положение на T (откат по profile_change)",
    "education": "образование на T (откат по profile_change)",
    "region": "регион на T (откат по profile_change)",
    "city": "город на T (откат по profile_change)",
    "housing_type": "тип жилья на T (откат по profile_change)",
    "income_type": "вид дохода на T (откат по profile_change)",
    "industry": "отрасль работы на T (откат по profile_change)",
    "declared_income": "заявленный доход на T (откат по profile_change)",
    "income_day": "день получения дохода на T (откат по profile_change)",
    "children": "число детей на T (откат по profile_change)",
    "age": "полных лет на местную дату T по birth_date",
    "job_tenure_months": "полных месяцев на последней работе по найму, известной банку раньше T; "
    "пусто — работы по найму нет",
    **{
        f"days_since_{name}": f"дней от вехи {name} до T; пусто — вехи раньше T нет"
        for name in MILESTONES
    },
}


def profile_at(profile: pd.DataFrame, changes: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """
    Анкета каждого клиента на T: одна строка на клиента, индекс client_id.

    changes — события profile_change с колонками client_id, t, raw_row,
    field_name, old_value.
    """
    if (profile["as_of"] < cutoff).any():
        raise ValueError("снимок анкеты раньше T: откатить вперёд нельзя")

    local_day = cutoff.date()
    out = profile.set_index("client_id")[list(CONSTANT + CHANGED_BY_EVENT)].astype(object)

    later = changes[(changes["t"] >= cutoff) & changes["field_name"].isin(CHANGED_BY_EVENT)]
    first = later.sort_values(["t", "raw_row"], kind="stable").drop_duplicates(["client_id", "field_name"])
    for row in first.itertuples(index=False):
        out.at[row.client_id, row.field_name] = None if pd.isna(row.old_value) else row.old_value

    for name in NUMERIC_FIELDS:
        out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
    for name in CATEGORICAL:
        out[name] = out[name].map(lambda value: MISSING if value is None or pd.isna(value) else str(value))

    rows = profile.set_index("client_id")
    out["age"] = [_full_years(born, local_day) for born in rows["birth_date"]]
    out["job_tenure_months"] = [
        _tenure(records, cutoff, local_day) if income in SALARIED else np.nan
        for records, income in zip(rows["employment"], out["income_type"])
    ]
    for name in MILESTONES:
        out[f"days_since_{name}"] = [_since(items, name, cutoff) for items in rows["lifelong"]]
    return out


def _full_years(born: date, day: date) -> int:
    return day.year - born.year - ((day.month, day.day) < (born.month, born.day))


def _full_months(start: date, day: date) -> int:
    return (day.year - start.year) * 12 + day.month - start.month - (day.day < start.day)


def _tenure(records: list[dict], cutoff: datetime, day: date) -> float:
    known = [item for item in records if item["record_time"] < cutoff]
    if not known:
        return np.nan
    last = max(known, key=lambda item: item["record_time"])
    if last["start_date"] is None:
        return np.nan
    return float(max(0, _full_months(last["start_date"], day)))


def _since(items: list[dict], name: str, cutoff: datetime) -> float:
    moments = [item["event_time"] for item in items if item["type"] == name and item["event_time"] < cutoff]
    if not moments:
        return np.nan
    return (cutoff - min(moments)).total_seconds() / 86400.0
