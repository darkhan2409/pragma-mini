from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from ..config import HISTORY_END, HISTORY_START
from ..rng import NS_PERSONA, numpy_rng, stable_hash, stable_unit, state_cache
from ..world import communities, geography
from .traits import Traits, draw_traits


# ============================================================
# ПЕРСОНА
# ============================================================
#
# Совместные распределения важнее независимого выбора:
#
#   тип поселения -> город и район -> возраст и жизненный этап
#   -> семья -> образование -> занятость и отрасль -> доход
#   -> цифровая зрелость -> роль Home Credit -> режим активности
#
# Наблюдаемая часть уезжает в профиль, скрытая остаётся
# в truth и в RAW не попадает никогда.
# ============================================================


PENSION_AGE = 63


@dataclass(frozen=True)
class Persona:

    client_ordinal: int
    client_id: str
    community_id: int
    is_test_account: bool

    # --- демография ---
    birth_date: datetime
    gender: str
    life_stage: str
    family_status: str
    children: int
    household_size: int
    education: str
    housing_type: str

    # --- география ---
    settlement: str
    region: str
    settlement_type: str
    home_district: str
    work_district: str
    transport: str

    # --- занятость и доход ---
    income_type: str
    industry: str | None
    employer_id: str | None
    true_income: int
    declared_income: int
    salary_day: int
    mandatory_share: float
    rent_share: float

    # --- отношения с банком ---
    relationship_start: datetime
    registered_in_window: bool
    vanished_after_registration: bool
    hcb_role: str
    visible_share: float
    activity_mode: str
    consent_marketing: bool
    night_segment: bool

    # --- скрытый портрет ---
    traits: Traits

    # --------------------------------------------------------

    def age_at(self, ts: datetime) -> int:
        years = ts.year - self.birth_date.year
        if (ts.month, ts.day) < (self.birth_date.month, self.birth_date.day):
            years -= 1
        return years

    @property
    def age(self) -> int:
        return self.age_at(HISTORY_START)

    def is_pensioner_at(self, ts: datetime) -> bool:
        return self.income_type == "pensioner" or self.age_at(ts) >= PENSION_AGE

    def relationship_months_at(self, ts: datetime) -> int:
        months = (ts.year - self.relationship_start.year) * 12 + (
            ts.month - self.relationship_start.month
        )
        return max(0, months)

    def trait(self, name: str, ts: datetime | None = None) -> float:
        return self.traits.value(name, ts)


def _weighted(rng, mapping: dict) -> str:
    keys = list(mapping)
    weights = [max(0.0, float(mapping[key])) for key in keys]
    total = sum(weights) or 1.0
    draw = rng.random() * total
    running = 0.0
    for key, weight in zip(keys, weights):
        running += weight
        if draw <= running:
            return key
    return keys[-1]


@state_cache
def draw_persona(client_ordinal: int) -> Persona:
    """
    Детерминированная персона клиента.
    """

    settings = params_module.active()
    population = settings.population

    rng = numpy_rng(NS_PERSONA, client_ordinal)

    # --- география ---

    places = geography.settlements()
    weights = [item.population_weight for item in places]
    total_weight = sum(weights)

    draw = float(rng.random()) * total_weight
    running = 0.0
    settlement = places[-1]

    for item, weight in zip(places, weights):
        running += weight
        if draw <= running:
            settlement = item
            break

    home_district = settlement.districts[int(rng.integers(0, len(settlement.districts)))]
    work_district = (
        home_district
        if len(settlement.districts) == 1 or rng.random() < 0.28
        else settlement.districts[int(rng.integers(0, len(settlement.districts)))]
    )

    transport = _weighted(rng, settings.geography.transport_mix[settlement.settlement_type])

    # --- возраст и жизненный этап ---

    life_stage = _weighted(rng, population.stage_weights)

    low, high = population.stage_age_range[life_stage]
    age = int(rng.integers(low, high))

    birth_date = HISTORY_START - timedelta(days=int(age * 365.25) + int(rng.integers(0, 365)))

    gender = _weighted(rng, population.gender_weights)

    # --- семья ---

    family_status = _weighted(rng, population.family_status_by_stage[life_stage])

    child_rate = population.children_rate_by_stage[life_stage]

    if family_status in ("married", "civil_marriage"):
        child_rate *= 1.35
    elif family_status == "single":
        child_rate *= 0.35

    children_total = int(min(population.max_children, rng.poisson(child_rate)))

    children = int(round(children_total * population.children_at_home_factor[life_stage]))
    children = max(0, min(children_total, children))

    household_size = 1 + children + (1 if family_status in ("married", "civil_marriage") else 0)

    # --- образование и жильё ---

    education = _weighted(rng, population.education_by_settlement_type[settlement.settlement_type])
    housing_type = _weighted(rng, population.housing_by_settlement_type[settlement.settlement_type])

    # --- занятость ---

    income_weights = dict(population.income_type_by_stage[life_stage])

    for name, factor in (population.income_type_settlement_factor.get(settlement.settlement_type) or {}).items():
        if name in income_weights:
            income_weights[name] *= factor

    income_type = _weighted(rng, income_weights)

    industry = (
        _weighted(rng, population.industry_weights)
        if income_type in population.industry_income_types
        else None
    )

    employer_id = (
        f"emp_{stable_hash(settlement.name, industry, int(rng.integers(0, 4000))) % 10 ** 9:09d}"
        if income_type in ("employed", "state_employee")
        else None
    )

    # --- доход ---

    if industry is not None:
        median = population.income_median_by_industry[industry]
    else:
        median = population.income_median_by_type.get(income_type, 250_000)

    scale = (
        population.income_settlement_factor[settlement.settlement_type]
        * population.income_stage_factor[life_stage]
        * population.income_education_factor[education]
    )

    raw_income = float(rng.lognormal(mean=math.log(median * scale), sigma=population.income_sigma))

    low_bound, high_bound = population.income_bounds

    true_income = int(min(high_bound, max(low_bound, round(raw_income / 1000) * 1000)))

    # Банк знает доход по анкете: он округлён и слегка смещён.
    declared_income = int(round(true_income * float(rng.uniform(0.88, 1.06)) / 5000) * 5000)
    declared_income = int(min(high_bound, max(low_bound, declared_income)))

    if income_type == "pensioner":
        salary_day = int(rng.integers(*settings.income.pension_day_range))
    else:
        salary_day = int(rng.integers(*settings.income.salary_day_range))

    mandatory_low, mandatory_high = population.mandatory_share_by_stage[life_stage]
    mandatory_share = float(rng.uniform(mandatory_low, mandatory_high))

    rent_share = (
        float(rng.uniform(*population.rent_share_of_income))
        if housing_type == "rented"
        else 0.0
    )

    # --- роль банка и режим активности ---

    hcb_role = _weighted(rng, population.hcb_role_weights)

    visible_low, visible_high = population.hcb_visible_share[hcb_role]
    visible_share = float(rng.uniform(visible_low, visible_high))

    mode_weights = dict(population.activity_mode_weights)

    for name, factor in (population.activity_mode_role_factor.get(hcb_role) or {}).items():
        if name in mode_weights:
            mode_weights[name] *= factor

    activity_mode = _weighted(rng, mode_weights)

    # --- отношения с банком ---

    registered_in_window = bool(rng.random() < population.registration_in_window_share)

    if registered_in_window:
        span = (HISTORY_END - HISTORY_START).days - population.registration_margin_days
        offset = int(rng.integers(1, max(2, span)))
        relationship_start = HISTORY_START + timedelta(days=offset)
    else:
        shape, scale_months = population.tenure_months_gamma
        tenure = int(min(population.tenure_months_max, max(1, rng.gamma(shape, scale_months))))
        tenure = min(tenure, max(1, (age - 18) * 12))
        relationship_start = HISTORY_START - timedelta(days=int(tenure * 30.44))

    vanished = bool(
        registered_in_window and rng.random() < population.registered_and_vanished_share
    )

    consent_marketing = bool(rng.random() < 0.92)

    night_segment = bool(rng.random() < settings.activity.night_segment_share)

    traits = draw_traits(
        client_ordinal=client_ordinal,
        life_stage=life_stage,
        hcb_role=hcb_role,
        activity_mode=activity_mode,
        settlement_type=settlement.settlement_type,
    )

    is_test_account = stable_unit("test_account", client_ordinal) < population.test_account_share

    return Persona(
        client_ordinal=client_ordinal,
        client_id=communities.client_id(client_ordinal),
        community_id=communities.community_of(client_ordinal),
        is_test_account=bool(is_test_account),
        birth_date=birth_date,
        gender=gender,
        life_stage=life_stage,
        family_status=family_status,
        children=children,
        household_size=household_size,
        education=education,
        housing_type=housing_type,
        settlement=settlement.name,
        region=settlement.region,
        settlement_type=settlement.settlement_type,
        home_district=home_district,
        work_district=work_district,
        transport=transport,
        income_type=income_type,
        industry=industry,
        employer_id=employer_id,
        true_income=true_income,
        declared_income=declared_income,
        salary_day=salary_day,
        mandatory_share=mandatory_share,
        rent_share=rent_share,
        relationship_start=relationship_start,
        registered_in_window=registered_in_window,
        vanished_after_registration=vanished,
        hcb_role=hcb_role,
        visible_share=visible_share,
        activity_mode=activity_mode,
        consent_marketing=consent_marketing,
        night_segment=night_segment,
        traits=traits,
    )


__all__ = ["PENSION_AGE", "Persona", "draw_persona"]
