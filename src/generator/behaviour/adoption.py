from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .. import params as params_module
from ..life.persona import Persona
from ..rng import stable_unit
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


# Ключи eligibility, которые здесь действительно проверяются.
#
# Список нужен не для красоты: незнакомый ключ раньше просто
# игнорировался, и продукт выдавался тому, кому правило его
# запрещало. Молчать об условии хуже, чем упасть на нём.
SUPPORTED_ELIGIBILITY: frozenset[str] = frozenset(
    {
        "min_age",
        "max_age",
        "requires_app",
        "requires_pension",
        "requires_children",
        "requires_loan",
        "min_assets",
        "min_existing_loans",
        "income_months",
        "min_amount",
        "regions",
    }
)


class EligibilityError(Exception):
    """Каталог требует условия, которого генератор не проверяет."""


@dataclass(frozen=True)
class _Requirements:
    min_age: int
    max_age: int | None
    requires_app: bool
    requires_pension: bool
    requires_children: bool
    requires_loan: bool
    min_assets: int | None
    min_existing_loans: int
    income_months: int
    min_amount: int
    regions: object


# Условия версии читаются из её eligibility один раз: eligible
# спрашивают миллионы раз за прогон, а условия версии не меняются.
# Ключ — id версии, рядом сама версия: id переиспользуется только
# после её смерти, а версии живут в каталоге.
_REQUIREMENTS: dict[int, tuple] = {}


def _requirements(view: ProductView, version: Version) -> _Requirements:

    hit = _REQUIREMENTS.get(id(version))

    if hit is not None and hit[0] is version:
        return hit[1]

    eligibility = version.eligibility or {}

    unknown = sorted(set(eligibility) - SUPPORTED_ELIGIBILITY)

    if unknown:
        raise EligibilityError(
            f"{view.code}: условие {unknown} объявлено в каталоге, но не проверяется"
        )

    max_age = eligibility.get("max_age")
    min_assets = eligibility.get("min_assets")

    requirements = _Requirements(
        min_age=int(eligibility.get("min_age", 0)),
        max_age=None if max_age is None else int(max_age),
        requires_app=bool(eligibility.get("requires_app")),
        requires_pension=bool(eligibility.get("requires_pension")),
        requires_children=bool(eligibility.get("requires_children")),
        requires_loan=bool(eligibility.get("requires_loan")),
        min_assets=int(min_assets) if min_assets else None,
        min_existing_loans=int(eligibility.get("min_existing_loans", 0)),
        income_months=int(eligibility.get("income_months", 0)),
        min_amount=int(eligibility.get("min_amount", 0)),
        regions=eligibility.get("regions"),
    )

    _REQUIREMENTS[id(version)] = (version, requirements)

    return requirements


def eligible(
    persona: Persona,
    view: ProductView,
    version: Version,
    ts: datetime,
    held_codes: frozenset,
    held_families: dict,
    assets: int,
    has_app: bool,
    open_loans: int,
    active_contracts: int,
    income_months: int,
    amount: int | None = None,
) -> bool:
    """
    Доступен ли продукт клиенту на эту дату.

    income_months — стаж действующего подтверждаемого дохода в
    месяцах; его считает сам клиент по своим потокам дохода.

    amount — сумма будущего договора, если она уже выбрана. Без
    неё условие min_amount проверить нечем, и оно не проверяется.
    """

    settings = params_module.active().products

    rules = settings.bank_rules

    age = persona.age_at(ts)

    if age < rules["min_age"] or age > rules["max_age"]:
        return False

    if active_contracts >= rules["max_active_contracts"]:
        return False

    need = _requirements(view, version)

    if age < need.min_age:
        return False

    if need.max_age is not None and age > need.max_age:
        return False

    if need.requires_app and not has_app:
        return False

    if need.requires_pension and not persona.is_pensioner_at(ts):
        return False

    if need.requires_children and persona.children <= 0:
        return False

    if need.requires_loan and open_loans <= 0:
        return False

    if need.min_assets is not None and assets < need.min_assets:
        return False

    # Рефинансируют НЕСКОЛЬКО кредитов в один. Раньше здесь
    # стояла та же проверка, что и у requires_loan, и человек с
    # единственным кредитом получал рефинансирование — то есть
    # ровно то, чего условие не разрешает.
    if open_loans < need.min_existing_loans:
        return False

    # Стаж подтверждаемого дохода. Раньше вместо него стоял срок
    # отношений с банком, и это разные вещи: клиент мог держать
    # здесь счёт пять лет, а работу найти вчера.
    if income_months < need.income_months:
        return False

    # Минимальная сумма договора: продукт с порогом не открывают
    # на сумму ниже порога.
    if amount is not None and amount < need.min_amount:
        return False

    regions = need.regions

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
        catalog = product_catalog.catalog()
        cards = sum(
            count
            for code, count in held_families.items()
            if catalog.has(code) and catalog.view(code).family == "debit_card"
        )
        if cards >= rules["max_debit_cards_total"]:
            return False

    if view.family == "cash_loan":
        catalog = product_catalog.catalog()
        loans = sum(
            count
            for code, count in held_families.items()
            if catalog.has(code) and catalog.view(code).family == "cash_loan"
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


# Что о продукте зависит только от момента — продаётся ли он, какая
# версия, базовый вес и кривая принятия, возраст версии, — одинаково у
# всех клиентов, кого спрашивают на этот момент: планы дня всех
# клиентов сообщества спрашивают на полночь. Кэш на один момент;
# сбрасывается со сменой момента, каталога или параметров.
_OFFERABLE: list = [None, None, None, ()]


def _offerable(ts: datetime) -> tuple:

    catalog = product_catalog.catalog()
    settings = params_module.active()

    if _OFFERABLE[0] is catalog and _OFFERABLE[1] is settings and _OFFERABLE[2] == ts:
        return _OFFERABLE[3]

    rates = settings.products.adoption["base_rate_per_year"]

    rows = []

    for view in catalog.views.values():

        if view.family == "service":
            continue

        if not view.sellable_at(ts):
            continue

        version = view.version_at(ts)

        base = float(rates.get(view.family, 0.1))

        rows.append((
            view,
            version,
            base,
            adoption_curve(view, ts) if base > 0.0 else 0.0,
            (ts - view.version_start(version)).days,
        ))

    _OFFERABLE[:] = [catalog, settings, ts, tuple(rows)]

    return _OFFERABLE[3]


def candidates(
    persona: Persona,
    ts: datetime,
    held_codes: frozenset,
    held_families: dict,
    assets: int,
    has_app: bool,
    open_loans: int,
    active_contracts: int,
    income_months: int,
    stress: float = 0.0,
) -> tuple:
    """
    Продукты, которые сейчас можно предложить клиенту.
    """

    adoption = params_module.active().products.adoption

    result: list[Candidate] = []

    for view, version, base, curve, row_age in _offerable(ts):

        if not eligible(
            persona, view, version, ts, held_codes, held_families,
            assets, has_app, open_loans, active_contracts, income_months,
        ):
            continue

        weight = base

        if weight <= 0.0:
            continue

        weight *= curve

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


__all__ = [
    "Candidate",
    "adoption_curve",
    "application_probability",
    "candidates",
    "eligible",
    "migration_targets",
    "pick",
]
