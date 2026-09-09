from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from .config import HISTORY_START
from .rng import NS_PERSONA, client_rng
from .world import (
    EDUCATION_TYPES,
    EDUCATION_WEIGHTS,
    FAMILY_STATUSES,
    FAMILY_STATUS_WEIGHTS,
    GENDER_WEIGHTS,
    GENDERS,
    HOUSING_TYPES,
    HOUSING_WEIGHTS,
    INCOME_TYPE_WEIGHTS,
    INCOME_TYPES,
    INDUSTRIES,
    INDUSTRY_INCOME_TYPES,
    INDUSTRY_WEIGHTS,
    REGION_WEIGHTS,
    REGIONS,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Persona это скрытая причина поведения клиента.
#
# Часть её полей НАБЛЮДАЕМА банком и попадает в profile
# (демография, занятость, доход, регион).
#
# Часть остаётся latent и в RAW не пишется никогда:
# activity, digital_affinity, mobility, credit_need,
# risk, volatility, push_reachable.
# ============================================================


PENSION_AGE = 63


@dataclass(frozen=True)
class Persona:
    client_id: int

    # ---- наблюдаемое: демография ----
    birth_date: datetime
    gender: str
    family_status: str
    children: int
    education: str
    region: str
    housing_type: str

    # ---- наблюдаемое: занятость и доход ----
    income_type: str
    industry: str | None
    declared_income: int
    salary_day: int

    # ---- наблюдаемое: отношения с банком ----
    relationship_start: datetime

    # ---- latent: драйверы поведения ----
    activity: float
    digital_affinity: float
    mobility: float
    credit_need: float
    risk: float
    volatility: float
    push_reachable: bool

    # --------------------------------------------------------

    def age_at(self, ts: datetime) -> int:
        """
        Возраст на дату: меняется от снимка к снимку.
        """

        years = ts.year - self.birth_date.year

        if (ts.month, ts.day) < (self.birth_date.month, self.birth_date.day):
            years -= 1

        return years

    @property
    def age(self) -> int:
        """
        Возраст на начало истории.
        """

        return self.age_at(HISTORY_START)

    def is_pensioner_at(self, ts: datetime) -> bool:
        return self.income_type == "pensioner" or self.age_at(ts) >= PENSION_AGE

    def relationship_months_at(self, ts: datetime) -> int:
        """
        Стаж отношений с банком в месяцах на дату.
        """

        months = (ts.year - self.relationship_start.year) * 12 + (
            ts.month - self.relationship_start.month
        )

        return max(0, months)


# ============================================================
# DRAW
# ============================================================


@lru_cache(maxsize=131_072)
def draw_persona(client_id: int) -> Persona:
    """
    Детерминированная персона клиента.

    Порядок розыгрышей фиксирован: перестановка меняет
    всю популяцию, поэтому новые поля добавляются в конец.
    """

    rng = client_rng(client_id, NS_PERSONA)

    # --------------------------------------------------------
    # ВОЗРАСТ И ДАТА РОЖДЕНИЯ
    # --------------------------------------------------------

    age = int(rng.integers(18, 71))

    birth_date = HISTORY_START - timedelta(
        days=int(age * 365.25) + int(rng.integers(0, 365))
    )

    # --------------------------------------------------------
    # ДЕМОГРАФИЯ
    # --------------------------------------------------------

    gender = str(rng.choice(GENDERS, p=GENDER_WEIGHTS))

    region = str(rng.choice(REGIONS, p=REGION_WEIGHTS))

    education = str(rng.choice(EDUCATION_TYPES, p=EDUCATION_WEIGHTS))

    housing_type = str(rng.choice(HOUSING_TYPES, p=HOUSING_WEIGHTS))

    # Семейный статус зависит от возраста: у молодых чаще single.
    family_weights = np.array(FAMILY_STATUS_WEIGHTS, dtype=float)

    if age < 25:
        family_weights = family_weights * np.array([0.4, 3.0, 1.2, 0.3, 0.05, 1.0])
    elif age >= 60:
        family_weights = family_weights * np.array([1.0, 0.6, 0.6, 1.2, 4.0, 1.0])

    family_weights /= family_weights.sum()

    family_status = str(rng.choice(FAMILY_STATUSES, p=family_weights))

    # Дети: зависят от возраста и семейного статуса.
    child_rate = 0.0

    if age >= 22:
        child_rate = 0.9 + 0.03 * (min(age, 45) - 22)

        if family_status in ("married", "civil_marriage"):
            child_rate *= 1.5
        elif family_status == "single":
            child_rate *= 0.3

    children = int(min(6, rng.poisson(child_rate)))

    # --------------------------------------------------------
    # ЗАНЯТОСТЬ И ДОХОД
    # --------------------------------------------------------

    income_weights = np.array(INCOME_TYPE_WEIGHTS, dtype=float)

    if age >= PENSION_AGE:
        income_weights = income_weights * np.array([0.3, 0.4, 0.3, 12.0, 0.3, 0.0, 0.5])
    elif age >= PENSION_AGE - 5:
        income_weights = income_weights * np.array([1.0, 1.0, 1.0, 3.0, 1.0, 0.0, 1.0])
    elif age < 23:
        income_weights = income_weights * np.array([0.8, 1.2, 0.2, 0.02, 0.5, 8.0, 1.5])
    else:
        # До предпенсионного возраста пенсия бывает только по инвалидности.
        income_weights = income_weights * np.array([1.0, 1.0, 1.0, 0.05, 1.0, 1.0, 1.0])

    income_weights /= income_weights.sum()

    income_type = str(rng.choice(INCOME_TYPES, p=income_weights))

    industry = (
        str(rng.choice(INDUSTRIES, p=INDUSTRY_WEIGHTS))
        if income_type in INDUSTRY_INCOME_TYPES
        else None
    )

    declared_income = float(
        np.exp(rng.normal(loc=np.log(350_000), scale=0.55))
    )

    income_scale = {
        "employed": 1.00,
        "state_employee": 0.85,
        "self_employed": 0.95,
        "business_owner": 1.60,
        "pensioner": 0.45,
        "student": 0.35,
        "unemployed": 0.30,
    }[income_type]

    declared_income = int(
        np.clip(declared_income * income_scale, 80_000, 3_000_000)
    )

    salary_day = int(rng.integers(1, 29))

    # --------------------------------------------------------
    # ОТНОШЕНИЯ С БАНКОМ
    # --------------------------------------------------------
    #
    # Клиент пришёл в банк задолго до начала истории:
    # от нескольких месяцев до 12 лет.
    # --------------------------------------------------------

    tenure_months = int(np.clip(rng.gamma(2.0, 24.0), 1, 144))

    max_tenure_months = max(1, (age - 18) * 12)

    tenure_months = min(tenure_months, max_tenure_months)

    relationship_start = HISTORY_START - timedelta(days=int(tenure_months * 30.44))

    # --------------------------------------------------------
    # LATENT
    # --------------------------------------------------------

    activity = float(rng.beta(2.0, 2.0))
    digital_affinity = float(rng.beta(2.5, 1.8))
    mobility = float(rng.beta(1.8, 3.0))
    credit_need = float(rng.beta(1.7, 3.0))
    risk = float(rng.beta(1.5, 5.0))
    volatility = float(rng.beta(1.3, 5.0))

    # Достижимость по push: приложение установлено и уведомления
    # включены. Банк этого признака не видит (пункт П2 отчёта).
    push_reachable = bool(rng.random() < 0.55 + 0.35 * digital_affinity)

    return Persona(
        client_id=client_id,
        birth_date=birth_date,
        gender=gender,
        family_status=family_status,
        children=children,
        education=education,
        region=region,
        housing_type=housing_type,
        income_type=income_type,
        industry=industry,
        declared_income=declared_income,
        salary_day=salary_day,
        relationship_start=relationship_start,
        activity=activity,
        digital_affinity=digital_affinity,
        mobility=mobility,
        credit_need=credit_need,
        risk=risk,
        volatility=volatility,
        push_reachable=push_reachable,
    )
