from __future__ import annotations

import pyarrow as pa

from .config import PROFILE_FIELDS


# ============================================================
# ПРОФИЛЬ
# ============================================================
#
# Одна строка на клиента: анкета такой, какой она стала к
# границе выгрузки. Версий, границ действия и признаков записи
# у профиля нет.
#
# История изменений анкеты не пропала — она живёт событиями
# profile_change в ленте, где у каждого изменения есть своё
# точное время, старое и новое значение.
#
# birth_date — календарная дата рождения в поясе банка. Полем
# анкеты для модели она не является и в PROFILE_FIELDS не входит:
# возраст и признак пенсионера меняются со временем без события,
# и только по дате рождения их можно посчитать на любую дату.
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
    "income_day": pa.int32(),
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
    [("client_id", pa.string()), ("birth_date", pa.date32())]
    + [(name, PROFILE_FIELD_TYPES[name]) for name in PROFILE_FIELDS]
)


__all__ = ["PROFILE_FIELD_TYPES", "PROFILE_SCHEMA"]
