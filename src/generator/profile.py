from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa

from . import config
from .config import PROFILE_FIELDS


# ============================================================
# ПРОФИЛЬ
# ============================================================
#
# Одна строка на клиента: анкета такой, какой она стала к
# границе выгрузки. Сама граница записана в строке — as_of:
# снимок описывает клиента непосредственно перед этим моментом.
# Версий, границ действия и признаков записи у профиля нет.
#
# История изменений анкеты не пропала — она живёт событиями
# profile_change в ленте, где у каждого изменения есть своё
# точное время, старое и новое значение.
#
# birth_date — календарная дата рождения в поясе банка. Полем
# анкеты для модели она не является и в PROFILE_FIELDS не входит:
# возраст и признак пенсионера меняются со временем без события,
# и только по дате рождения их можно посчитать на любую дату.
#
# lifelong — датированные вехи отношений клиента с банком. Это
# факты анкеты, а не события ленты: в ленте их нет, и лента их не
# дублирует. Веха может лежать раньше начала выгрузки — клиент,
# пришедший в 2021 году, остаётся клиентом с 2021 года, даже
# если его события видны только с 2024-го.
#
#   relationship_started  начало отношений с банком;
#   kyc_passed            банк идентифицировал клиента. Приход в
#                         генераторе — один акт: клиент принят
#                         сразу после идентификации, поэтому
#                         момент тот же;
#   app_adopted           клиент установил приложение. Вехи нет
#                         у того, кто его не ставил.
#
# Контракт вех тот же полуоткрытый, что у событий: в снимок
# попадает только веха строго раньше as_of.
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


# Типы вех в порядке объявления. Этот же порядок разводит вехи с
# одинаковым временем.
LIFELONG_TYPES: tuple[str, ...] = ("relationship_started", "kyc_passed", "app_adopted")

UTC_MICROS = pa.timestamp("us", tz="UTC")

LIFELONG_ITEM = pa.struct([("type", pa.string()), ("event_time", UTC_MICROS)])


PROFILE_SCHEMA = pa.schema(
    [("client_id", pa.string()), ("as_of", UTC_MICROS), ("birth_date", pa.date32())]
    + [(name, PROFILE_FIELD_TYPES[name]) for name in PROFILE_FIELDS]
    + [("lifelong", pa.list_(LIFELONG_ITEM))]
)


def utc(moment: datetime) -> datetime:
    """
    Местное время генератора как момент в UTC.
    """

    stamped = moment.replace(tzinfo=config.TIMEZONE) if moment.tzinfo is None else moment

    return stamped.astimezone(timezone.utc)


def lifelong(
    relationship_start: datetime,
    app_adopted_at: datetime | None,
    as_of: datetime,
) -> list[dict]:
    """
    Вехи клиента, случившиеся строго раньше as_of, по времени.

    Даты берутся готовыми — те самые, по которым жила симуляция.
    Новых розыгрышей здесь нет, поэтому лента от вех не зависит.
    """

    found = [
        ("relationship_started", relationship_start),
        ("kyc_passed", relationship_start),
        ("app_adopted", app_adopted_at),
    ]

    boundary = utc(as_of)

    items = [
        (utc(moment), LIFELONG_TYPES.index(kind), kind)
        for kind, moment in found
        if moment is not None and utc(moment) < boundary
    ]

    return [{"type": kind, "event_time": moment} for moment, _, kind in sorted(items)]


__all__ = [
    "LIFELONG_ITEM",
    "LIFELONG_TYPES",
    "PROFILE_FIELD_TYPES",
    "PROFILE_SCHEMA",
    "lifelong",
    "utc",
]
