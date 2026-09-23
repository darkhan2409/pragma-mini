from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import (
    NS_ONBOARDING,
    NS_PERSONA,
    keyed_rng,
    numpy_rng,
    stable_hash,
    state_cache,
)
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
# внутри симуляции и в выгрузку не попадает никогда.
# ============================================================


PENSION_AGE = 63


@dataclass(frozen=True)
class Persona:

    client_ordinal: int
    client_id: str
    community_id: int

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
    income_day: int
    mandatory_share: float
    rent_share: float

    # --- отношения с банком ---
    relationship_start: datetime
    registered_in_window: bool
    vanished_after_registration: bool
    hcb_role: str
    visible_share: float
    activity_mode: str
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
        return self.age_at(config.HISTORY_START)

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


def employer_payday(employer_id: str) -> int:
    """
    День зарплаты работодателя: один на всех его сотрудников.
    """

    low, high = params_module.active().income.salary_day_range

    return int(low + stable_hash("payday", employer_id) % max(1, high - low))


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

    birth_date = config.HISTORY_START - timedelta(days=int(age * 365.25) + int(rng.integers(0, 365)))

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

    employer_id = None

    if income_type in ("employed", "state_employee"):

        relationships = settings.relationships

        community_id = (client_ordinal - 1) // relationships.community_size

        # Зарплатный проект: часть клиентов сообщества работает
        # у одного работодателя. Общий плательщик, общий день
        # выплаты и общая задержка видны в данных без отдельного
        # признака.
        shared_rng = keyed_rng(NS_PERSONA, client_ordinal, 71)

        if shared_rng.random() < relationships.shared_employer_share:

            slot = int(shared_rng.integers(0, max(1, relationships.shared_employers_per_community)))

            employer_id = f"emp_{stable_hash('shared', community_id, slot) % 10 ** 9:09d}"

        else:

            employer_id = (
                f"emp_{stable_hash(settlement.name, industry, int(rng.integers(0, 4000))) % 10 ** 9:09d}"
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
        income_day = int(rng.integers(*settings.income.pension_day_range))
    else:
        income_day = int(rng.integers(*settings.income.salary_day_range))

    # День зарплаты назначает работодатель, а не сотрудник: у
    # коллег он общий. Личный розыгрыш выше оставлен, чтобы не
    # сдвинуть следующие черты персоны.
    if employer_id is not None:
        income_day = employer_payday(employer_id)

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
        span = (config.HISTORY_END - config.HISTORY_START).days - population.registration_margin_days
        offset = int(rng.integers(1, max(2, span)))
        relationship_start = config.HISTORY_START + timedelta(days=offset)
    else:
        shape, scale_months = population.tenure_months_gamma
        tenure = int(min(population.tenure_months_max, max(1, rng.gamma(shape, scale_months))))
        tenure = min(tenure, max(1, (age - 18) * 12))
        relationship_start = config.HISTORY_START - timedelta(days=int(tenure * 30.44))

    vanished = bool(
        registered_in_window and rng.random() < population.registered_and_vanished_share
    )

    # Здесь разыгрывалось второе, дублирующее согласие на рассылку.
    # Розыгрыш оставлен пустым: без него сдвинулись бы все
    # следующие черты персоны. Согласие решает consent_date.
    rng.random()

    night_segment = bool(rng.random() < settings.activity.night_segment_share)

    traits = draw_traits(
        client_ordinal=client_ordinal,
        life_stage=life_stage,
        hcb_role=hcb_role,
        activity_mode=activity_mode,
        settlement_type=settlement.settlement_type,
    )


    return Persona(
        client_ordinal=client_ordinal,
        client_id=communities.client_id(client_ordinal),
        community_id=communities.community_of(client_ordinal),
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
        income_day=income_day,
        mandatory_share=mandatory_share,
        rent_share=rent_share,
        relationship_start=relationship_start,
        registered_in_window=registered_in_window,
        vanished_after_registration=vanished,
        hcb_role=hcb_role,
        visible_share=visible_share,
        activity_mode=activity_mode,
        night_segment=night_segment,
        traits=traits,
    )


@state_cache
def app_adoption(client_ordinal: int) -> datetime | None:
    """
    Когда клиент установил приложение.

    Приложение — норма, а не исключение: его ставит подавляющее
    большинство. Небольшая доля не ставит никогда, и это не
    дефект данных, а часть жизни: остаются люди, которые ходят
    в отделение.

    Дата установки всегда попадает ВНУТРЬ окна наблюдения.
    """

    settings = params_module.active().defects

    persona = draw_persona(client_ordinal)

    rng = keyed_rng(NS_ONBOARDING, client_ordinal, 1)

    digital = persona.trait("digital_affinity")

    # Цифровая склонность двигает вероятность мягко: разница
    # между самым и наименее цифровым клиентом — проценты, а не
    # разы. Приложением пользуются почти все.
    probability = settings.app_adoption_share * (0.94 + 0.12 * digital)

    if rng.random() >= min(0.995, probability):
        return None

    # Раньше начала наблюдения приложения быть не может, как и
    # раньше того дня, когда человек стал клиентом.
    start = max(config.HISTORY_START, persona.relationship_start)

    # Половина окна на то, чтобы установить: у большинства это
    # случается вскоре после начала отношений с банком.
    span_days = max(1, (config.HISTORY_END - start).days)

    offset = int(rng.integers(0, max(1, span_days // 2)))

    adopted = start + timedelta(days=offset)

    if adopted >= config.HISTORY_END:
        adopted = start

    return adopted.replace(hour=0, minute=0, second=0, microsecond=0)


@state_cache
def consent_date(client_ordinal: int) -> datetime | None:
    """
    Когда клиент дал согласие на коммуникации.
    """

    settings = params_module.active().defects

    persona = draw_persona(client_ordinal)

    rng = keyed_rng(NS_ONBOARDING, client_ordinal, 2)

    if rng.random() >= settings.consent_share:
        return None

    start = persona.relationship_start

    given = start + timedelta(days=int(rng.integers(0, 200)))

    if given >= config.HISTORY_END:
        return None

    return given.replace(hour=0, minute=0, second=0, microsecond=0)


__all__ = [
    "PENSION_AGE",
    "Persona",
    "app_adoption",
    "consent_date",
    "draw_persona",
    "employer_payday",
]
