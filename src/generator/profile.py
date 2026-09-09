from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from .config import PROFILE_FIELDS, SEED
from .persona import Persona, draw_persona
from .products import ProductState
from .rng import NS_PROFILE, keyed_rng
from .trajectory import behavior_state
from .world import INDUSTRY_INCOME_TYPES


# ============================================================
# ИДЕЯ
# ============================================================
#
# Профиль это СОСТОЯНИЕ as-of, а не событие. Он пересчитывается
# на конец каждого месяца, как витрина ABT.
#
# Профиль на cutoff это последний снимок строго раньше cutoff.
#
# Три вида пропусков, все три встречаются в реальной витрине:
#
# 1. Структурные: отрасли нет у пенсионера, лимита нет
#    без кредитной карты. Пропуск здесь несёт смысл.
#
# 2. Сегментные: витрина построена вокруг наличного кредитования,
#    и у части клиентов без кредитных продуктов целые блоки
#    полей пустые во ВСЕХ снимках.
#
# 3. Помесячные: источник блока данных иногда отваливается,
#    и в отдельные месяцы доля пропусков подскакивает.
# ============================================================


FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "demographics": ("gender", "family_status", "children", "education", "housing_type"),
    "employment": ("income_type", "industry", "declared_income", "salary_day"),
    "relationship": ("relationship_months", "contracts_count", "active_contracts"),
    "ownership": ("holds_credit_card", "holds_debit_card", "holds_deposit"),
    "credit": ("credit_limit", "credit_utilization"),
}

# age и region это ключевые поля: они заполнены всегда.
ALWAYS_PRESENT = ("age", "region", "pensioner")

# Стабильный номер блока: hash() строки в Python рандомизирован
# между процессами, а длина имени у блоков совпадает.
GROUP_CODES: dict[str, int] = {
    group: index for index, group in enumerate(FIELD_GROUPS)
}

# Базовая доля пропусков блока в обычный месяц.
BASE_MISSING = {
    "demographics": 0.02,
    "employment": 0.05,
    "relationship": 0.01,
    "ownership": 0.01,
    "credit": 0.03,
}

# Вероятность сбоя источника блока в конкретный месяц.
OUTAGE_PROBABILITY = 0.06

OUTAGE_MISSING = (0.35, 0.85)

# Доля клиентов без кредитных продуктов, у которых витрина
# не заполняет блок employment вообще.
SEGMENT_EMPTY_SHARE = 0.30

# Годовая индексация заявленного дохода.
INCOME_MONTHLY_INDEXATION = 0.006


@dataclass(frozen=True)
class ProfileSnapshot:
    client_id: int
    ts: datetime
    snapshot_month: datetime
    values: dict[str, Any]


# ============================================================
# ПЕРИОДЫ
# ============================================================


def month_start(ts: datetime) -> datetime:
    return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(ts: datetime) -> datetime:
    return (
        ts.replace(year=ts.year + 1, month=1)
        if ts.month == 12
        else ts.replace(month=ts.month + 1)
    )


def month_end(ts: datetime) -> datetime:
    """
    Последняя секунда месяца: момент расчёта витрины.
    """

    return next_month(month_start(ts)) - timedelta(seconds=1)


# ============================================================
# ПРОПУСКИ
# ============================================================


@lru_cache(maxsize=4096)
def month_missing_rate(month: datetime, group: str) -> float:
    """
    Доля пропусков блока в этом месяце. Одинакова для всех клиентов:
    сбой источника задевает всю витрину сразу.
    """

    rng = keyed_rng(SEED, NS_PROFILE, month.toordinal(), GROUP_CODES[group])

    if rng.random() < OUTAGE_PROBABILITY:
        return rng.uniform(*OUTAGE_MISSING)

    return BASE_MISSING[group] * rng.uniform(0.5, 1.8)


@lru_cache(maxsize=131_072)
def segment_empty_groups(client_id: int, has_credit_history: bool) -> frozenset[str]:
    """
    Блоки, пустые у клиента во всех снимках.
    """

    if has_credit_history:
        return frozenset()

    rng = keyed_rng(NS_PROFILE, client_id, 0, 1)

    if rng.random() < SEGMENT_EMPTY_SHARE:
        return frozenset({"employment"})

    return frozenset()


# ============================================================
# ЗНАЧЕНИЯ
# ============================================================


def indexed_income(persona: Persona, ts: datetime) -> int:
    """
    Заявленный доход медленно растёт: клиент переподтверждает
    его при обращениях, и витрина видит новое значение.
    """

    months = max(0, persona.relationship_months_at(ts) - persona.relationship_months_at(persona.relationship_start))

    value = persona.declared_income * (1.0 + INCOME_MONTHLY_INDEXATION) ** min(months, 180)

    return int(round(value / 1_000) * 1_000)


def raw_values(
    persona: Persona,
    state: ProductState,
    ts: datetime,
) -> dict[str, Any]:
    """
    Значения профиля as-of без учёта пропусков.
    """

    owned = state.owned_at(ts)

    credit_limit = state.credit_limit(ts)

    if credit_limit is None:
        utilization = None
    else:
        pressure = behavior_state(persona.client_id, ts).utilization_pressure

        rng = keyed_rng(NS_PROFILE, persona.client_id, ts.toordinal(), 7)

        utilization = round(
            min(1.20, max(0.0, pressure * rng.uniform(0.55, 1.45))), 4
        )

    return {
        "age": persona.age_at(ts),
        "gender": persona.gender,
        "family_status": persona.family_status,
        "children": persona.children,
        "education": persona.education,
        "region": persona.region,
        "housing_type": persona.housing_type,
        "pensioner": persona.is_pensioner_at(ts),
        "income_type": persona.income_type,
        "industry": persona.industry,
        "declared_income": indexed_income(persona, ts),
        "salary_day": persona.salary_day,
        "relationship_months": persona.relationship_months_at(ts),
        "contracts_count": state.contracts_count(ts),
        "active_contracts": state.active_contracts(ts),
        "holds_credit_card": "credit_card" in owned,
        "holds_debit_card": "debit_card" in owned,
        "holds_deposit": "deposit" in owned,
        "credit_limit": credit_limit,
        "credit_utilization": utilization,
    }


def apply_missing(
    values: dict[str, Any],
    client_id: int,
    month: datetime,
    empty_groups: frozenset[str],
) -> dict[str, Any]:
    """
    Накладывает сегментные и помесячные пропуски.
    """

    result = dict(values)

    # Структурный пропуск: отрасли нет у неработающих по найму.
    if values["income_type"] not in INDUSTRY_INCOME_TYPES:
        result["industry"] = None

    for group, fields in FIELD_GROUPS.items():

        if group in empty_groups:
            for field in fields:
                result[field] = None
            continue

        rate = month_missing_rate(month, group)

        if rate <= 0.0:
            continue

        rng = keyed_rng(NS_PROFILE, client_id, month.toordinal(), GROUP_CODES[group])

        if rng.random() < rate:
            for field in fields:
                result[field] = None

    return result


# ============================================================
# СНИМКИ
# ============================================================


def profile_snapshots(
    client_id: int,
    start: datetime,
    end: datetime,
    state: ProductState | None = None,
) -> list[ProfileSnapshot]:
    """
    Помесячные снимки профиля на [start, end).

    Снимок месяца рассчитывается на его последнюю секунду.
    """

    if end <= start:
        raise ValueError("end must be after start")

    persona = draw_persona(client_id)

    if state is None:
        state = ProductState(client_id)

    snapshots: list[ProfileSnapshot] = []

    month = month_start(start)

    while month < end:

        ts = month_end(month)

        if start <= ts < end:

            # Пустые блоки определяются ТОЛЬКО по сведениям,
            # доступным на дату снимка: договор, открытый позже,
            # на этот снимок влиять не может.
            empty_groups = segment_empty_groups(
                client_id,
                state.has_credit_history_at(ts),
            )

            values = raw_values(persona, state, ts)

            snapshots.append(
                ProfileSnapshot(
                    client_id=client_id,
                    ts=ts,
                    snapshot_month=month,
                    values=apply_missing(values, client_id, month, empty_groups),
                )
            )

        month = next_month(month)

    return snapshots


def profile_as_of(
    snapshots: list[ProfileSnapshot],
    cutoff: datetime,
) -> ProfileSnapshot | None:
    """
    Последний снимок строго раньше cutoff.
    """

    visible = [snapshot for snapshot in snapshots if snapshot.ts < cutoff]

    return visible[-1] if visible else None


def snapshot_row(snapshot: ProfileSnapshot) -> dict[str, Any]:
    """
    Плоская строка для parquet.
    """

    row: dict[str, Any] = {
        "client_id": snapshot.client_id,
        "ts": snapshot.ts,
        "snapshot_month": snapshot.snapshot_month,
    }

    for field in PROFILE_FIELDS:
        row[field] = snapshot.values.get(field)

    return row
