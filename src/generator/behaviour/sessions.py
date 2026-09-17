from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .. import params as params_module
from ..life import calendar as cal
from ..life.persona import Persona
from ..rng import (
    COMPONENT_CONTENT,
    COMPONENT_COUNT,
    COMPONENT_TIME,
    NS_DEVICE,
    NS_SESSION,
    day_rng,
    event_rng,
    stable_hash,
    state_cache,
)
from ..world.dictionaries import (
    BROWSE_SCREENS,
    DEVICE_TYPES,
    DEVICE_WEIGHTS,
    DOMAIN_ADOPTION,
    DOMAIN_FAMILY,
    SCREEN_HOME,
    SCREEN_OFFERS,
)


# ============================================================
# СЕССИИ ПРИЛОЖЕНИЯ
# ============================================================
#
# У сессии есть ЦЕЛЬ, и число экранов зависит от неё, а не
# рисуется независимо: короткая проверка баланса это два экрана,
# оплата счёта это вход, список платежей, форма и подтверждение.
#
# Защищённое действие возможно только после успешного входа либо
# при действующей авторизации.
# ============================================================


GOAL_BALANCE = "balance_check"
GOAL_PAYMENT = "payment"
GOAL_TRANSFER = "transfer"
GOAL_CARDS = "card_management"
GOAL_EXPLORE = "product_explore"
GOAL_SUPPORT = "support"
GOAL_MARKET = "market"
GOAL_PROFILE = "profile_settings"
GOAL_LOAN = "loan_service"
GOAL_DEPOSIT = "deposit_service"

GOALS = (
    GOAL_BALANCE,
    GOAL_PAYMENT,
    GOAL_TRANSFER,
    GOAL_CARDS,
    GOAL_EXPLORE,
    GOAL_SUPPORT,
    GOAL_MARKET,
    GOAL_PROFILE,
    GOAL_LOAN,
    GOAL_DEPOSIT,
)

GOAL_DOMAIN = {
    GOAL_BALANCE: "home",
    GOAL_PAYMENT: "payments",
    GOAL_TRANSFER: "transfers",
    GOAL_CARDS: "cards",
    GOAL_EXPLORE: "loans",
    GOAL_SUPPORT: "support",
    GOAL_MARKET: "market",
    GOAL_PROFILE: "profile",
    GOAL_LOAN: "loans",
    GOAL_DEPOSIT: "deposits",
}

# Цель -> последовательность экранов и итоговая операция.
GOAL_FLOW = {
    GOAL_BALANCE: (("home", "s_000_home"), ("home", "s_010_balance"), ("home", "s_011_history")),
    GOAL_PAYMENT: (("home", "s_000_home"), ("payments", "s_300_payments"), ("payments", None)),
    GOAL_TRANSFER: (("home", "s_000_home"), ("transfers", "s_200_transfers"), ("transfers", "s_203_transfer_confirm")),
    GOAL_CARDS: (("home", "s_000_home"), ("cards", "s_100_cards"), ("cards", "s_101_card_detail")),
    GOAL_EXPLORE: (("home", "s_000_home"), (None, None), (None, None)),
    GOAL_SUPPORT: (("home", "s_000_home"), ("support", "s_900_support"), ("support", "s_901_chat")),
    GOAL_MARKET: (("home", "s_000_home"), ("market", "s_700_market"), ("market", "s_701_market_item")),
    GOAL_PROFILE: (("home", "s_000_home"), ("profile", "s_800_profile"), ("profile", "s_801_settings")),
    GOAL_LOAN: (("home", "s_000_home"), ("loans", "s_400_loans"), ("loans", "s_403_loan_schedule")),
    GOAL_DEPOSIT: (("home", "s_000_home"), ("deposits", "s_500_deposits"), ("deposits", "s_503_deposit_detail")),
}

GOAL_OPERATION = {
    GOAL_BALANCE: None,
    GOAL_PAYMENT: "pay_utility",
    GOAL_TRANSFER: "transfer_phone",
    GOAL_CARDS: "card_view",
    GOAL_EXPLORE: None,
    GOAL_SUPPORT: "chat_open",
    GOAL_MARKET: "market_browse",
    GOAL_PROFILE: "profile_view",
    GOAL_LOAN: "loan_schedule",
    GOAL_DEPOSIT: "deposit_view",
}


@dataclass(frozen=True)
class Step:
    kind: str
    ts: datetime
    domain: str
    screen: str | None = None
    operation: str | None = None
    amount: int | None = None
    target: str | None = None
    tag: str | None = None
    offers: bool = False


@dataclass(frozen=True)
class Session:
    session_id: str
    device_id: str
    device_new: bool
    started_at: datetime
    goal: str
    steps: tuple


@dataclass(frozen=True)
class SessionContext:
    due_bills: tuple = ()
    card_blocked: bool = False
    recent_offer_family: str | None = None
    has_loan: bool = False
    has_deposit: bool = False
    has_card: bool = False
    accounts: int = 0
    dpd: int = 0
    salary_just_arrived: bool = False
    recent_failure: bool = False
    fraud_alert: bool = False


@state_cache
def adopted_domains(client_ordinal: int, digital: float) -> tuple:
    """
    Домены приложения, которыми клиент вообще пользуется.
    Покрытие домена это свойство клиента, а не дня.
    """

    rng = event_rng(NS_SESSION, client_ordinal, 0, 0, COMPONENT_CONTENT)

    domains = ["home", "auth"]

    for domain, share in DOMAIN_ADOPTION.items():

        if domain in domains:
            continue

        probability = share * (0.62 + 0.8 * digital)

        if rng.random() < min(0.98, probability):
            domains.append(domain)

    return tuple(dict.fromkeys(domains))


def device_for(persona: Persona, index: int) -> tuple:
    """
    Устройство клиента. Смена устройства редка и заметна.
    """

    rng = event_rng(NS_DEVICE, persona.client_ordinal, 0, index, COMPONENT_CONTENT)

    kind = str(rng.choice(list(DEVICE_TYPES), p=list(DEVICE_WEIGHTS)))

    device_id = f"dev_{stable_hash('device', persona.client_id, index) % 10 ** 10:010d}"

    return device_id, kind


def stress_session_factor(episodes: tuple, ts) -> float:
    """
    В начале трудного периода в приложение заходят чаще:
    проверяют остаток. Потом заходят реже: смотреть нечего.
    """

    from ..life import stress as stress_module

    settings = params_module.active().stress

    episode = stress_module.active_episode(episodes, ts)

    if episode is None:
        return 1.0

    level = episode.level(ts)

    if level <= 0.0:
        return 1.0

    early = ts <= episode.peak_end

    if early:
        return 1.0 + settings.app_checks_boost_early * level

    return max(0.2, 1.0 - settings.app_checks_drop_late * level)


def daily_session_rate(
    persona: Persona,
    ts: datetime,
    state: str,
    silenced: frozenset,
    app_adopted: bool,
) -> float:

    settings = params_module.active().activity

    if not app_adopted or "sessions" in silenced:
        return 0.0

    rate = settings.sessions_per_day[persona.activity_mode]

    rate *= settings.state_factor.get(state, 1.0)
    rate *= settings.role_factor.get(persona.hcb_role, 1.0)
    rate *= 0.55 + 1.1 * persona.trait("digital_affinity", ts)

    if ts.weekday() >= 5:
        rate *= settings.weekend_factor_sessions

    return float(max(0.0, rate))


def _goal_weights(persona: Persona, ts: datetime, context: SessionContext, adopted: tuple) -> dict:

    digital = persona.trait("digital_affinity", ts)

    weights = {
        GOAL_BALANCE: 3.0,
        GOAL_PAYMENT: 0.25,
        GOAL_TRANSFER: 0.0,
        GOAL_CARDS: 0.0,
        GOAL_EXPLORE: 0.52,
        GOAL_SUPPORT: 0.10,
        GOAL_MARKET: 0.0,
        GOAL_PROFILE: 0.22,
        GOAL_LOAN: 0.0,
        GOAL_DEPOSIT: 0.0,
    }

    if context.due_bills and "payments" in adopted:
        weights[GOAL_PAYMENT] = 0.04 + 0.15 * min(3, len(context.due_bills))

    if "transfers" in adopted:
        weights[GOAL_TRANSFER] = 2.3

    if "cards" in adopted and context.has_card:
        weights[GOAL_CARDS] = 1.7
        if context.card_blocked:
            weights[GOAL_CARDS] *= 4.5

    if "market" in adopted:
        weights[GOAL_MARKET] = 0.8

    if "loans" in adopted and context.has_loan:
        weights[GOAL_LOAN] = 1.5
        if context.dpd > 0:
            weights[GOAL_LOAN] *= 2.8

    if "deposits" in adopted and context.has_deposit:
        weights[GOAL_DEPOSIT] = 0.7

    if context.salary_just_arrived:
        weights[GOAL_BALANCE] *= 1.6
        weights[GOAL_TRANSFER] *= 1.4

    if context.recent_failure:
        weights[GOAL_SUPPORT] *= 4.0

    if context.fraud_alert:
        weights[GOAL_SUPPORT] *= 6.0
        weights[GOAL_CARDS] *= 3.0

    if context.recent_offer_family:
        weights[GOAL_EXPLORE] *= 2.4

    weights[GOAL_EXPLORE] *= 0.5 + 1.5 * persona.trait("credit_appetite", ts)
    weights[GOAL_MARKET] *= 0.4 + 1.4 * digital

    return {name: value for name, value in weights.items() if value > 0.0}


def plan_sessions(
    persona: Persona,
    day: datetime,
    state: str,
    silenced: frozenset,
    app_adopted: bool,
    context: SessionContext,
) -> tuple:
    """
    Сессии приложения за день.
    """

    settings = params_module.active().activity

    rate = daily_session_rate(persona, day, state, silenced, app_adopted)

    if rate <= 0.0:
        return ()

    ordinal = day.toordinal()

    count_rng = day_rng(NS_SESSION, persona.client_ordinal, ordinal, COMPONENT_COUNT)

    count = min(settings.max_sessions_per_day, count_rng.poisson(rate))

    if count <= 0:
        return ()

    digital = persona.trait("digital_affinity", day)

    adopted = adopted_domains(persona.client_ordinal, round(digital, 3))

    weights = _goal_weights(persona, day, context, adopted)

    if not weights:
        return ()

    hours = params_module.active().activity.session_hour_profile

    sessions: list[Session] = []

    for index in range(count):

        time_rng = event_rng(NS_SESSION, persona.client_ordinal, ordinal, index, COMPONENT_TIME)
        content_rng = event_rng(NS_SESSION, persona.client_ordinal, ordinal, index, COMPONENT_CONTENT)

        hour = int(time_rng.choice(24, p=list(hours)))

        started_at = day.replace(
            hour=hour,
            minute=int(time_rng.integers(0, 60)),
            second=int(time_rng.integers(0, 60)),
            microsecond=0,
        )

        goal = content_rng.weighted(weights)

        device_index = 0

        if content_rng.random() < 0.04:
            device_index = int(content_rng.integers(1, 3))

        device_id, _ = device_for(persona, device_index)

        session_id = f"ses_{stable_hash('session', persona.client_id, ordinal, index) % 10 ** 12:012d}"

        steps = _build_steps(
            persona=persona,
            goal=goal,
            started_at=started_at,
            context=context,
            adopted=adopted,
            rng=content_rng,
        )

        sessions.append(
            Session(
                session_id=session_id,
                device_id=device_id,
                device_new=device_index > 0,
                started_at=started_at,
                goal=goal,
                steps=tuple(steps),
            )
        )

    sessions.sort(key=lambda item: item.started_at)

    return tuple(sessions)


def _build_steps(
    persona: Persona,
    goal: str,
    started_at: datetime,
    context: SessionContext,
    adopted: tuple,
    rng,
) -> list:
    """
    Экраны и операции сессии. Длина зависит от цели.
    """

    settings = params_module.active().activity

    steps: list[Step] = []

    moment = started_at

    def advance(low: int = None, high: int = None) -> datetime:
        nonlocal moment
        span_low, span_high = settings.session_step_seconds
        moment = moment + timedelta(
            seconds=int(rng.integers(low or span_low, high or span_high))
        )
        return moment

    # --- вход ---

    biometry = rng.random() < 0.35 + 0.45 * persona.trait("digital_affinity", started_at)

    steps.append(
        Step(
            kind="operation",
            ts=moment,
            domain="auth",
            operation="biometry_login" if biometry else "login",
            tag="auth",
        )
    )

    advance(1, 6)

    # --- домашний экран ---

    steps.append(Step(kind="screen", ts=moment, domain="home", screen=SCREEN_HOME, offers=True))

    # --- витрина предложений ---

    if rng.random() < 0.22 + 0.20 * persona.trait("digital_affinity", started_at):
        advance()
        steps.append(Step(kind="screen", ts=moment, domain="home", screen=SCREEN_OFFERS, offers=True))

    flow = GOAL_FLOW[goal]

    if goal == GOAL_EXPLORE:

        family = context.recent_offer_family or rng.choice(("cash_loan", "deposit", "credit_card", "insurance"))

        domain = next((name for name, value in DOMAIN_FAMILY.items() if value == family), "loans")

        # Изучение продукта тоже идёт в принятом домене: раньше
        # эта ветка возвращалась до общего фильтра и рисовала
        # экраны разделов, которых у клиента нет.
        if domain not in adopted:
            domain = "loans" if "loans" in adopted else next(iter(sorted(adopted)), "home")

        screens = BROWSE_SCREENS.get(domain, BROWSE_SCREENS["loans"])

        depth = 1 + int(rng.integers(0, min(3, len(screens))))

        for position in range(depth):
            advance()
            steps.append(
                Step(
                    kind="screen",
                    ts=moment,
                    domain=domain,
                    screen=screens[position],
                    target=family,
                    tag="explore",
                )
            )

        return steps

    for domain, screen in flow[1:]:

        if domain is None or screen is None:
            continue

        if domain not in adopted and domain != "home":
            continue

        advance()

        steps.append(Step(kind="screen", ts=moment, domain=domain, screen=screen))

    operation = GOAL_OPERATION.get(goal)

    if goal == GOAL_PAYMENT and context.due_bills:

        bill = context.due_bills[int(rng.integers(0, len(context.due_bills)))]

        operation = {
            "utilities": "pay_utility",
            "telecom": "pay_mobile",
            "internet": "pay_internet",
            "kindergarten": "pay_utility",
            "fines": "pay_fine",
            "taxes": "pay_tax",
        }.get(bill, "pay_utility")

        advance()

        steps.append(
            Step(
                kind="screen",
                ts=moment,
                domain="payments",
                screen="s_301_payment_utility",
            )
        )

        advance()

        steps.append(
            Step(
                kind="operation",
                ts=moment,
                domain="payments",
                operation=operation,
                target=bill,
                tag="confirm",
            )
        )

        return steps

    if goal == GOAL_TRANSFER:
        operation = str(
            rng.choice(("transfer_phone", "transfer_card", "transfer_own", "transfer_template"))
        )
        if operation == "transfer_own" and context.accounts < 2:
            operation = "transfer_phone"

    if goal == GOAL_CARDS and context.card_blocked:
        operation = "card_unblock"

    if goal == GOAL_LOAN and context.dpd > 0:
        operation = "loan_repay"

    if goal == GOAL_DEPOSIT and rng.random() < 0.35:
        operation = "deposit_topup"

    if operation is not None:
        advance()
        steps.append(
            Step(
                kind="operation",
                ts=moment,
                domain=GOAL_DOMAIN[goal],
                operation=operation,
                tag="confirm" if goal in (GOAL_PAYMENT, GOAL_TRANSFER) else None,
            )
        )

    return steps


__all__ = [
    "GOALS",
    "GOAL_BALANCE",
    "GOAL_CARDS",
    "GOAL_DEPOSIT",
    "GOAL_DOMAIN",
    "GOAL_EXPLORE",
    "GOAL_LOAN",
    "GOAL_MARKET",
    "GOAL_PAYMENT",
    "GOAL_PROFILE",
    "GOAL_SUPPORT",
    "GOAL_TRANSFER",
    "Session",
    "SessionContext",
    "Step",
    "adopted_domains",
    "daily_session_rate",
    "device_for",
    "plan_sessions",
]
