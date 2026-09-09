from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import HISTORY_START, SOURCE_AVAILABILITY
from .persona import Persona, draw_persona
from .rng import NS_PRODUCT, KeyedRandom, client_rng, keyed_rng


# ============================================================
# ИДЕЯ
# ============================================================
#
# PRODUCT_EVENTS это реестр ОТКРЫТИЙ договоров, как DC_LOAN,
# DC_CARD и DC_INSURANCE в банке. Событие одно: продукт открыт.
# Закрытия отдельным событием не приходят, поэтому владение
# выводится из даты открытия и срока договора.
#
# Реестр старше остальных потоков: в нём лежат договоры,
# открытые задолго до окна наблюдения, но не раньше даты
# миграции реестра (SOURCE_AVAILABILITY).
#
# timestamp_quality повторяет реальный дефект: у карт есть
# DTIME_ACTIVATION с временем, у кредитов и страховок только
# DATE_*, то есть полночь.
# ============================================================


# ============================================================
# PRODUCTS
# ============================================================

PRODUCT_TYPES = (
    "debit_card",
    "credit_card",
    "cash_loan",
    "deposit",
    "insurance",
)

CREDIT_PRODUCTS = ("credit_card", "cash_loan")

# Продукты, которые заканчиваются сами: срок договора истёк.
TERM_PRODUCTS = ("cash_loan", "deposit", "insurance")

PRODUCT_SUBTYPES: dict[str, tuple[str, ...]] = {
    "debit_card": ("arna", "salary", "youth", "pension"),
    "credit_card": ("ozen_base", "ozen_gold"),
    "cash_loan": ("standard", "refinance", "topup", "partner_auto"),
    "deposit": ("standard", "max_rate", "child", "pension"),
    "insurance": ("travel", "life", "property", "auto_kasko"),
}

SUBTYPE_WEIGHTS: dict[str, tuple[float, ...]] = {
    "debit_card": (0.60, 0.28, 0.08, 0.04),
    "credit_card": (0.82, 0.18),
    "cash_loan": (0.55, 0.25, 0.15, 0.05),
    "deposit": (0.52, 0.30, 0.10, 0.08),
    "insurance": (0.40, 0.25, 0.20, 0.15),
}

TERM_CHOICES: dict[str, tuple[int, ...]] = {
    "cash_loan": (6, 12, 18, 24, 36, 48, 60),
    "deposit": (3, 6, 12, 24, 36),
    "insurance": (6, 12, 24),
}

TERM_WEIGHTS: dict[str, tuple[float, ...]] = {
    "cash_loan": (0.10, 0.22, 0.16, 0.24, 0.18, 0.06, 0.04),
    "deposit": (0.14, 0.24, 0.40, 0.15, 0.07),
    "insurance": (0.20, 0.70, 0.10),
}


# ============================================================
# TIMESTAMP QUALITY
# ============================================================

QUALITY_EXACT = "exact"
QUALITY_DATE_ONLY = "date_only"

TIMESTAMP_QUALITIES = (QUALITY_EXACT, QUALITY_DATE_ONLY)

# Доля договоров с ТОЧНЫМ временем открытия.
# Карты активируются в системе с временем, кредиты и депозиты
# приходят датой; часть кредитов подтягивается из второй
# системы и время у них есть.
EXACT_TIME_SHARE: dict[str, float] = {
    "debit_card": 1.00,
    "credit_card": 1.00,
    "cash_loan": 0.10,
    "deposit": 0.02,
    "insurance": 0.00,
}


# ============================================================
# EVENT
# ============================================================


@dataclass(frozen=True)
class ProductEvent:
    """
    Открытие договора.

    amount_or_limit:
        лимит кредитной карты, сумма кредита, сумма депозита,
        премия страховки. У дебетовой карты суммы нет.

    term:
        срок в месяцах. У карт срока нет.
    """

    client_id: int
    ts: datetime

    product_type: str
    amount_or_limit: float | None
    term: int | None
    product_subtype: str
    timestamp_quality: str


# ============================================================
# HOLDING
# ============================================================


@dataclass(frozen=True)
class Holding:
    """
    Договор с датой открытия и вычисленной датой окончания.

    closed_at = None: бессрочный договор (карта).
    """

    product_type: str
    subtype: str
    amount_or_limit: float | None
    term: int | None
    opened_at: datetime
    closed_at: datetime | None

    def is_open_at(self, ts: datetime) -> bool:
        if ts < self.opened_at:
            return False

        return self.closed_at is None or ts < self.closed_at


# ============================================================
# HELPERS
# ============================================================


def add_months(anchor: datetime, months: int) -> datetime:
    """
    Дата через months месяцев, день обрезается до 28-го.
    """

    index = anchor.month - 1 + months

    year = anchor.year + index // 12
    month = index % 12 + 1

    return anchor.replace(year=year, month=month, day=min(anchor.day, 28))


def round_to(value: float, step: int) -> float:
    return float(round(value / step) * step)


def apply_timestamp_quality(ts: datetime, quality: str) -> datetime:
    """
    date_only: время договора не сохранилось, остаётся полночь.
    """

    if quality == QUALITY_DATE_ONLY:
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)

    return ts.replace(microsecond=0)


# ============================================================
# CONTRACT PARAMETERS
# ============================================================


def product_rng(client_id: int, product_type: str, opened_at: datetime) -> KeyedRandom:
    """
    RNG параметров договора по его идентичности.
    """

    return keyed_rng(
        NS_PRODUCT,
        client_id,
        PRODUCT_TYPES.index(product_type),
        opened_at.toordinal(),
    )


def draw_amount_or_limit(
    product_type: str,
    persona: Persona,
    rng: KeyedRandom,
) -> float | None:

    income = persona.declared_income

    if product_type == "debit_card":
        return None

    if product_type == "credit_card":
        return round_to(
            min(10_000_000.0, max(100_000.0, rng.lognormal(_log(2.0 * income), 0.50))),
            10_000,
        )

    if product_type == "cash_loan":
        return round_to(
            min(15_000_000.0, max(100_000.0, rng.lognormal(_log(3.0 * income), 0.55))),
            1_000,
        )

    if product_type == "deposit":
        return round_to(
            min(30_000_000.0, max(50_000.0, rng.lognormal(_log(1.5 * income), 0.80))),
            10_000,
        )

    # insurance: годовая премия
    return round_to(
        min(600_000.0, max(8_000.0, rng.lognormal(_log(0.15 * income), 0.60))),
        1_000,
    )


def _log(value: float) -> float:
    import math

    return math.log(max(value, 1.0))


def draw_term(product_type: str, rng: KeyedRandom) -> int | None:

    if product_type not in TERM_CHOICES:
        return None

    return int(rng.choice(TERM_CHOICES[product_type], p=TERM_WEIGHTS[product_type]))


def draw_subtype(product_type: str, persona: Persona, rng: KeyedRandom) -> str:

    subtypes = PRODUCT_SUBTYPES[product_type]
    weights = list(SUBTYPE_WEIGHTS[product_type])

    # Пенсионные и молодёжные варианты идут по возрасту.
    if product_type in ("debit_card", "deposit"):
        age = persona.age

        if age >= 60:
            weights[3] *= 6.0
        elif age <= 23 and product_type == "debit_card":
            weights[2] *= 6.0

    if product_type == "debit_card" and persona.income_type in ("employed", "state_employee"):
        weights[1] *= 2.0

    return str(rng.choice(subtypes, p=weights))


def open_contract(
    client_id: int,
    product_type: str,
    ts: datetime,
) -> tuple[ProductEvent, Holding]:
    """
    Создаёт договор и соответствующее событие открытия.
    """

    persona = draw_persona(client_id)
    rng = product_rng(client_id, product_type, ts)

    subtype = draw_subtype(product_type, persona, rng)
    amount_or_limit = draw_amount_or_limit(product_type, persona, rng)
    term = draw_term(product_type, rng)

    quality = (
        QUALITY_EXACT
        if rng.random() < EXACT_TIME_SHARE[product_type]
        else QUALITY_DATE_ONLY
    )

    event_ts = apply_timestamp_quality(ts, quality)

    closed_at = add_months(event_ts, term) if term is not None else None

    event = ProductEvent(
        client_id=client_id,
        ts=event_ts,
        product_type=product_type,
        amount_or_limit=amount_or_limit,
        term=term,
        product_subtype=subtype,
        timestamp_quality=quality,
    )

    holding = Holding(
        product_type=product_type,
        subtype=subtype,
        amount_or_limit=amount_or_limit,
        term=term,
        opened_at=event_ts,
        closed_at=closed_at,
    )

    return event, holding


# ============================================================
# PRE-HISTORY CONTRACTS
# ============================================================
#
# К началу окна наблюдения у клиента уже есть история договоров:
# первая карта в день прихода в банк и дальше по склонности.
# ============================================================

REGISTRY_START = SOURCE_AVAILABILITY["product_events"]


def initial_open_probabilities(persona: Persona) -> dict[str, float]:
    """
    Вероятность, что продукт был открыт до начала окна наблюдения.
    """

    import math

    log_income = math.log(max(persona.declared_income / 350_000, 0.05))

    credit = persona.credit_need

    return {
        "credit_card": min(0.80, max(0.05, 0.22 + 0.35 * credit + 0.10 * log_income)),
        "cash_loan": min(0.80, max(0.02, 0.13 + 0.40 * credit - 0.05 * log_income)),
        "deposit": min(0.80, max(0.02, 0.05 + 0.25 * (1.0 - credit) + 0.12 * log_income)),
        "insurance": min(0.70, max(0.02, 0.10 + 0.10 * persona.mobility)),
    }


def prehistory_contracts(client_id: int) -> list[tuple[ProductEvent, Holding]]:
    """
    Договоры, открытые до HISTORY_START.

    Первая дебетовая карта появляется в день прихода в банк,
    остальные продукты равномерно в течение стажа отношений.
    """

    persona = draw_persona(client_id)
    rng = client_rng(client_id, NS_PRODUCT)

    start = persona.relationship_start

    contracts: list[tuple[ProductEvent, Holding]] = [
        open_contract(client_id, "debit_card", start.replace(hour=11, minute=30))
    ]

    span_days = max(1, (HISTORY_START - start).days)

    probabilities = initial_open_probabilities(persona)

    for product_type in ("credit_card", "cash_loan", "deposit", "insurance"):

        if rng.random() >= probabilities[product_type]:
            continue

        offset = int(rng.integers(0, span_days))

        ts = start + timedelta(days=offset, hours=int(rng.integers(9, 20)))

        if ts >= HISTORY_START:
            continue

        contracts.append(open_contract(client_id, product_type, ts))

    contracts.sort(key=lambda pair: pair[0].ts)

    return contracts


# ============================================================
# PRODUCT STATE
# ============================================================


class ProductState:
    """
    Владение продуктами во времени.

    Накапливает договоры и отвечает на вопросы as-of:
    открыт ли продукт, сколько активных договоров, какой лимит.
    """

    def __init__(self, client_id: int) -> None:

        self.client_id = client_id

        self.events: list[ProductEvent] = []
        self.holdings: list[Holding] = []

        for event, holding in prehistory_contracts(client_id):
            self._add(event, holding)

    # --------------------------------------------------------

    def _add(self, event: ProductEvent, holding: Holding) -> None:
        self.events.append(event)
        self.holdings.append(holding)

    def open(self, product_type: str, ts: datetime) -> ProductEvent:
        """
        Открывает договор и возвращает событие.
        """

        event, holding = open_contract(self.client_id, product_type, ts)

        self._add(event, holding)

        return event

    # --------------------------------------------------------

    def is_open(self, product_type: str, ts: datetime) -> bool:
        return any(
            holding.product_type == product_type and holding.is_open_at(ts)
            for holding in self.holdings
        )

    def owned_at(self, ts: datetime) -> frozenset[str]:
        return frozenset(
            holding.product_type
            for holding in self.holdings
            if holding.is_open_at(ts)
        )

    def blocked(self, product_type: str, ts: datetime) -> bool:
        """
        Нельзя открыть второй такой же договор, пока действует текущий
        (в том числе если открытие уже запланировано на будущее).
        """

        return any(
            holding.product_type == product_type
            and (holding.closed_at is None or holding.closed_at > ts)
            for holding in self.holdings
        )

    def has_credit_history_at(self, ts: datetime) -> bool:
        """
        Был ли у клиента кредитный продукт К ЭТОМУ МОМЕНТУ.

        Витрина наличного кредитования наполняется с первого
        кредитного договора; закрытие договора клиента из неё
        уже не убирает. Договоры, открытые ПОЗЖЕ ts, здесь
        не учитываются: иначе профиль знал бы будущее.
        """

        return any(
            holding.product_type in CREDIT_PRODUCTS and holding.opened_at <= ts
            for holding in self.holdings
        )

    def contracts_count(self, ts: datetime) -> int:
        """
        Сколько договоров заключено к этому моменту (всего за историю).
        """

        return sum(1 for holding in self.holdings if holding.opened_at <= ts)

    def active_contracts(self, ts: datetime) -> int:
        return sum(1 for holding in self.holdings if holding.is_open_at(ts))

    def credit_limit(self, ts: datetime) -> float | None:
        """
        Суммарный лимит открытых кредитных карт. None: карты нет.
        """

        limits = [
            holding.amount_or_limit or 0.0
            for holding in self.holdings
            if holding.product_type == "credit_card" and holding.is_open_at(ts)
        ]

        return float(sum(limits)) if limits else None

    def visible_events(self) -> list[ProductEvent]:
        """
        События, попавшие в реестр: договоры старше даты миграции
        реестра в нём отсутствуют.
        """

        return sorted(
            (event for event in self.events if event.ts >= REGISTRY_START),
            key=lambda event: event.ts,
        )
