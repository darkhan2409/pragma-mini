from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Mapping

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
# birth_date — календарная дата рождения в поясе банка и
# единственный источник возраста. Полем анкеты для модели она не
# является и в PROFILE_FIELDS не входит. Возраста в снимке нет:
# он меняется со временем без события, и возраст на конец
# выгрузки был бы неверен для любого более раннего cutoff. Его
# считает препроцессинг — полных лет на cutoff примера.
#
# Датированные факты анкеты — у каждого своё время, и потому по
# ним анкета восстанавливается на любую дату раньше as_of:
#
#   employment  записи банка о наёмной работе: дата начала и
#               момент записи (life/income.employment); запись
#               без даты начала — работы больше нет;
#   lifelong    вехи отношений клиента с банком.
#
# Вехи — производные от фактического состояния клиента, без
# собственных розыгрышей. Веха может лежать раньше начала
# выгрузки: клиент, пришедший в 2017 году, остаётся клиентом с
# 2017 года, даже если его события видны только с 2024-го.
#
#   bank_registered       начало отношений с банком;
#   app_registered        клиенту открылось приложение. Это не
#                         первое использование: сессии начинаются
#                         с этого дня, но первая может прийти
#                         через недели. Вехи нет у того, кто
#                         приложения не ставил;
#   first_card_activated  первая активация карты — по картам
#                         договоров, без перевыпусков;
#   first_loan_opened     открытие первого кредита с графиком;
#   first_deposit_opened  открытие первого вклада.
#
# У вех о продуктах есть источник — сама карта или договор:
# source_id называет его идентификатор (card_id первой карты,
# contract_id первого кредита или вклада). Внутри окна акт этого
# источника лежит в ленте: строки типов LIFELONG_SOURCE_EVENTS с
# тем же идентификатором в поле LIFELONG_SOURCE_FIELD. Препроцессинг
# помечает их по этой ссылке, а не по совпадению времени, и целью
# MLM они не становятся (dataset/targets.py). У прихода в банк и
# приложения источника в ленте нет, source_id у них null.
#
# Контракт всех датированных фактов тот же полуоткрытый, что у
# событий: в снимок попадает только то, что случилось строго
# раньше as_of.
# ============================================================


PROFILE_FIELD_TYPES: dict[str, pa.DataType] = {
    "gender": pa.string(),
    "family_status": pa.string(),
    "children": pa.int32(),
    "education": pa.string(),
    "region": pa.string(),
    "city": pa.string(),
    "housing_type": pa.string(),
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
LIFELONG_TYPES: tuple[str, ...] = (
    "bank_registered",
    "app_registered",
    "first_card_activated",
    "first_loan_opened",
    "first_deposit_opened",
)

# Типы строк ленты, которыми записан акт источника вехи. Открытие
# договора со счётом пишет account_opened и product_opened (или
# product_migrated) в один момент — обе строки один акт.
LIFELONG_SOURCE_EVENTS: dict[str, tuple[str, ...]] = {
    "bank_registered": (),
    "app_registered": (),
    "first_card_activated": ("card_activated",),
    "first_loan_opened": ("account_opened", "product_opened", "product_migrated"),
    "first_deposit_opened": ("account_opened", "product_opened", "product_migrated"),
}

# Поле payload этих строк, где лежит source_id вехи.
LIFELONG_SOURCE_FIELD: dict[str, str] = {
    "first_card_activated": "card_id",
    "first_loan_opened": "contract_id",
    "first_deposit_opened": "contract_id",
}

UTC_MICROS = pa.timestamp("us", tz="UTC")

LIFELONG_ITEM = pa.struct(
    [("type", pa.string()), ("event_time", UTC_MICROS), ("source_id", pa.string())]
)

EMPLOYMENT_ITEM = pa.struct([("start_date", pa.date32()), ("record_time", UTC_MICROS)])


PROFILE_SCHEMA = pa.schema(
    [("client_id", pa.string()), ("as_of", UTC_MICROS), ("birth_date", pa.date32())]
    + [(name, PROFILE_FIELD_TYPES[name]) for name in PROFILE_FIELDS]
    + [
        ("employment", pa.list_(EMPLOYMENT_ITEM)),
        ("lifelong", pa.list_(LIFELONG_ITEM)),
    ]
)


def utc(moment: datetime) -> datetime:
    """
    Местное время генератора как момент в UTC.
    """

    stamped = moment.replace(tzinfo=config.TIMEZONE) if moment.tzinfo is None else moment

    return stamped.astimezone(timezone.utc)


def lifelong(
    moments: Mapping[str, tuple[datetime, str | None] | None], as_of: datetime
) -> list[dict]:
    """
    Вехи клиента, случившиеся строго раньше as_of, по времени.

    moments — тип вехи -> (её момент, source_id) или None. Даты
    берутся готовыми — те самые, по которым жила симуляция. Новых
    розыгрышей здесь нет, поэтому лента от вех не зависит.
    """

    unknown = sorted(set(moments) - set(LIFELONG_TYPES))

    if unknown:
        raise ValueError(f"вехи вне контракта: {unknown}")

    boundary = utc(as_of)

    items = [
        (utc(found[0]), LIFELONG_TYPES.index(kind), kind, found[1])
        for kind, found in moments.items()
        if found is not None and utc(found[0]) < boundary
    ]

    return [
        {"type": kind, "event_time": moment, "source_id": source}
        for moment, _, kind, source in sorted(items)
    ]


def employment(records: list[tuple[date | None, datetime]], as_of: datetime) -> list[dict]:
    """
    Записи о работе, появившиеся у банка строго раньше as_of.
    """

    boundary = utc(as_of)

    return [
        {"start_date": start, "record_time": utc(moment)}
        for start, moment in records
        if utc(moment) < boundary
    ]


__all__ = [
    "EMPLOYMENT_ITEM",
    "LIFELONG_ITEM",
    "LIFELONG_SOURCE_EVENTS",
    "LIFELONG_SOURCE_FIELD",
    "LIFELONG_TYPES",
    "PROFILE_FIELD_TYPES",
    "PROFILE_SCHEMA",
    "employment",
    "lifelong",
    "utc",
]
