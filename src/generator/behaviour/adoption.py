from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from ..life.persona import Persona
from ..rng import NS_ADOPTION, keyed_rng, stable_unit
from ..world import products as product_catalog
from ..world.hcb_timeline import Version
from ..world.products import ProductView


# ============================================================
# РАСПРОСТРАНЕНИЕ ПРОДУКТА
# ============================================================
#
#   eligibility -> предложение -> доставка или показ -> заявка
#   -> решение -> активация -> первое использование
#
# Учитываются пилот, сегмент, канал, регион, кампания,
# постепенное принятие и миграция со старого продукта.
# ============================================================


@dataclass(frozen=True)
class Candidate:
    view: ProductView
    version: Version
    weight: float
    reason: str


def _pilot_allows(persona: Persona, view: ProductView, ts: datetime) -> bool:
    """
    В пилоте участвует не каждый: доля определяется устойчивым
    хэшем клиента, а не розыгрышем дня.
    """

    if view.status_at(ts) != product_catalog.STATUS_PILOT:
        return True

    settings = params_module.active().products

    share = float(settings.adoption.get("pilot_share_default", 0.15))

    return stable_unit("pilot", view.code, persona.client_ordinal) < share


def eligible(
    persona: Persona,
    view: ProductView,
    version: Version,
    ts: datetime,
    held_codes: frozenset,
    held_families: dict,
    assets: int,
    has_app: bool,
    has_loan: bool,
    active_contracts: int,
) -> bool:
    """
    Доступен ли продукт клиенту на эту дату.
    """

    settings = params_module.active().products

    rules = settings.bank_rules

    age = persona.age_at(ts)

    if age < rules["min_age"] or age > rules["max_age"]:
        return False

    if active_contracts >= rules["max_active_contracts"]:
        return False

    eligibility = version.eligibility or {}

    if age < int(eligibility.get("min_age", 0)):
        return False

    if eligibility.get("max_age") is not None and age > int(eligibility["max_age"]):
        return False

    if eligibility.get("requires_app") and not has_app:
        return False

    if eligibility.get("requires_pension") and not persona.is_pensioner_at(ts):
        return False

    if eligibility.get("requires_children") and persona.children <= 0:
        return False

    if eligibility.get("requires_loan") and not has_loan:
        return False

    if eligibility.get("min_assets") and assets < int(eligibility["min_assets"]):
        return False

    if eligibility.get("min_existing_loans") and not has_loan:
        return False

    regions = eligibility.get("regions")

    if regions and persona.region not in regions and persona.settlement not in regions:
        return False

    if not _pilot_allows(persona, view, ts):
        return False

    # --- правила владения продукта ---

    row = None

    for item in view.rows:
        if item.valid_from <= ts < item.valid_to:
            row = item
            break

    if row is not None:

        held = held_families.get(view.code, 0)

        if not row.allow_multiple and held >= 1:
            return False

        if held >= row.max_active_holdings:
            return False

        exclusive = tuple(row.compatibility_rules.get("exclusive_with", ()))

        if any(code in held_codes for code in exclusive):
            return False

    # --- общие правила банка ---

    if view.family == "debit_card":
        cards = sum(
            count
            for code, count in held_families.items()
            if product_catalog.catalog().has(code)
            and product_catalog.catalog().view(code).family == "debit_card"
        )
        if cards >= rules["max_debit_cards_total"]:
            return False

    if view.family == "cash_loan":
        loans = sum(
            count
            for code, count in held_families.items()
            if product_catalog.catalog().has(code)
            and product_catalog.catalog().view(code).family == "cash_loan"
        )
        if loans >= rules["max_active_cash_loans"]:
            return False

    return True


def adoption_curve(view: ProductView, ts: datetime) -> float:
    """
    Новый продукт принимают не сразу: доля растёт от запуска
    к насыщению.
    """

    settings = params_module.active().products.adoption

    start = None

    for moment, status in view.status_spans:
        if status in (product_catalog.STATUS_ACTIVE, product_catalog.STATUS_PILOT):
            start = moment
            break

    if start is None:
        return 1.0

    elapsed = (ts - start).days

    if elapsed < 0:
        return 0.0

    ramp = max(1, int(settings.get("ramp_days", 240)))

    base = float(settings.get("ramp_start_share", 0.15))

    return float(min(1.0, base + (1.0 - base) * min(1.0, elapsed / ramp)))


def candidates(
    persona: Persona,
    ts: datetime,
    held_codes: frozenset,
    held_families: dict,
    assets: int,
    has_app: bool,
    has_loan: bool,
    active_contracts: int,
    stress: float = 0.0,
) -> tuple:
    """
    Продукты, которые сейчас можно предложить клиенту.
    """

    settings = params_module.active().products
    adoption = settings.adoption

    catalog = product_catalog.catalog()

    result: list[Candidate] = []

    for view in catalog.views.values():

        if view.family == "service":
            continue

        if not view.sellable_at(ts):
            continue

        version = view.version_at(ts)

        if not eligible(
            persona, view, version, ts, held_codes, held_families,
            assets, has_app, has_loan, active_contracts,
        ):
            continue

        weight = float(adoption["base_rate_per_year"].get(view.family, 0.1))

        if weight <= 0.0:
            continue

        weight *= adoption_curve(view, ts)

        trait_rule = adoption.get("trait_factor", {}).get(view.family)

        if trait_rule:
            name, strength = trait_rule
            value = persona.trait(name, ts)
            weight *= max(0.05, 1.0 + strength * (value - 0.5))

        if view.family in ("cash_loan", "refinance", "credit_card", "installment"):
            weight *= 1.0 + 1.4 * stress

        if view.family in ("deposit", "deposit_certificate", "bonds"):
            weight *= max(0.1, 1.0 - 0.8 * stress)
            weight *= min(2.0, max(0.2, assets / 400_000))

        # Ранние последователи: цифровые клиенты берут новинки
        # раньше остальных.
        row_age = (ts - view.version_start(version)).days

        if row_age < 180:
            weight *= 1.0 + (adoption.get("early_adopter_digital_factor", 2.0) - 1.0) * persona.trait(
                "digital_affinity", ts
            )

        # Миграционное притяжение: держатель прежнего продукта
        # переходит на преемника охотнее.
        reason = "organic"

        predecessor = view.record.predecessor

        if predecessor and predecessor in held_codes:
            weight *= adoption.get("migration_pull", 3.0)
            reason = "successor_offer"

        result.append(Candidate(view=view, version=version, weight=weight, reason=reason))

    result.sort(key=lambda item: item.view.code)

    return tuple(result)


def pick(candidates_pool: tuple, rng) -> Candidate | None:

    if not candidates_pool:
        return None

    weights = [item.weight for item in candidates_pool]

    if sum(weights) <= 0.0:
        return None

    return candidates_pool[int(rng.choice(len(candidates_pool), p=weights))]


def application_probability(
    persona: Persona,
    candidate: Candidate,
    ts: datetime,
    from_offer: bool,
    stress: float,
    total_weight: float = 0.0,
) -> float:
    """
    Вероятность подать заявку по итогам предложения или
    органического интереса.
    """

    products = params_module.active().products

    settings = products.adoption

    stress_params = params_module.active().stress

    # Интенсивность складывается по всем доступным продуктам:
    # вероятность подать хоть какую-то заявку сегодня. Какую
    # именно, решает pick. Потолок нужен только чтобы день не
    # стал заведомым: раньше он был 0.35 и упирался постоянно.
    base = total_weight / 365.0

    if from_offer:
        base *= settings.get("offer_factor", 2.6)
        base *= 1.0 + stress_params.offer_response_boost * stress

    if candidate.view.family == "refinance":
        base *= 1.0 + stress_params.refinance_interest_boost * stress
    elif candidate.view.family in ("cash_loan", "credit_card", "installment"):
        base *= 1.0 + stress_params.loan_interest_boost * stress

    return float(min(products.application_probability_cap, max(0.0, base)))


def migration_targets(ts: datetime, held_codes: frozenset) -> tuple:
    """
    Продукты-преемники для действующих договоров клиента.
    """

    catalog = product_catalog.catalog()

    result = []

    for code in sorted(held_codes):

        if not catalog.has(code):
            continue

        view = catalog.view(code)

        successor = view.record.successor

        if not successor or not catalog.has(successor):
            continue

        policy = view.record.migration_policy

        if policy in ("none", "servicing_only"):
            continue

        target = catalog.view(successor)

        if not target.sellable_at(ts):
            continue

        result.append((view, target, policy))

    return tuple(result)


def first_use_delay(rng) -> timedelta:

    settings = params_module.active().products.adoption

    low, high = settings.get("first_use_delay_days", (0, 21))

    return timedelta(days=int(rng.integers(low, high + 1)))


__all__ = [
    "Candidate",
    "adoption_curve",
    "application_probability",
    "candidates",
    "eligible",
    "first_use_delay",
    "migration_targets",
    "pick",
]
