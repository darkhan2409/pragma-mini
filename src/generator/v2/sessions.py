from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Generator

from ..app import (
    DOMAIN_PRODUCT,
    AppSession,
    adopted_domains,
    daily_session_rate,
    draw_session_start,
    make_session_id,
)
from ..persona import Persona, draw_persona
from ..rng import (
    COMPONENT_CONTENT,
    COMPONENT_COUNT,
    COMPONENT_TIME,
    NS_OPERATION,
    NS_SCREEN,
    NS_V2_NAV,
    NS_V2_SCENARIO,
    KeyedRandom,
    day_rng,
    event_rng,
)
from ..world import SCREEN_OFFERS
from .config import (
    AUTH_MAX_ATTEMPTS,
    AUTH_RECOVERY_SHARE,
    AUTH_RETRY_SECONDS,
    AUTH_SCREEN_GAP_SECONDS,
    CONTINUE_SHARE,
    MAX_INTENTS_PER_SESSION,
    MAX_SCREENS_PER_SESSION,
    MIN_STEP_SECONDS,
)
from .context import ContextView
from .habits import ClientHabits
from .outcomes import CANCELLED, FAILED, SUCCESS, next_step_options
from .scenarios import (
    BILL_KIND_BRANCH,
    BRANCH_BILL_KIND,
    PRODUCT_EXPLORE,
    SCENARIO_ENTRY,
    SCENARIOS,
    SUPPORT_STEPS,
    Step,
    explore_target_weights,
    scenario_weights,
    steps_for,
)
from .state import StateView, operation_offered


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сессия это сопрограмма: она ПРЕДЛАГАЕТ действие и получает
# обратно его исход. Проверку жёстких ограничений, розыгрыш
# исхода и применение последствий делает планировщик в момент
# самого действия, а не после сессии.
#
# Так решается три вещи сразу:
#
#   вход      защищённое действие невозможно, пока авторизация
#             не завершилась успехом; экраны идут после входа
#
#   время     перекрывающиеся сессии исполняются в общем порядке
#             событий, и каждое решение видит состояние на свою
#             секунду, а не на конец дня
#
#   исход     сначала жёсткое ограничение, потом случайность
#
# Всё скрытое (сценарий, намерение, попытка) остаётся здесь
# и в RAW не попадает.
# ============================================================


PURPOSE_BRANCH = 1
PURPOSE_DWELL = 2
PURPOSE_VARIANT = 3
PURPOSE_RECOVER = 4
PURPOSE_BANNER = 5
PURPOSE_CONTINUE = 6
PURPOSE_AUTH = 7

KIND_SCREEN = "screen"
KIND_OPERATION = "operation"

# Метка фазы входа: перепривязка устройства в настройках
# это не вход, хотя домен у неё тот же.
TAG_AUTH = "auth"


@dataclass(frozen=True)
class PaymentIntent:
    """
    Что именно клиент собирается оплатить.

    bill_key = None значит свободный платёж: он ничей счёт
    не гасит, и сумма у него своя.
    """

    kind: str
    bill_key: tuple[int, int] | None = None


@dataclass(frozen=True)
class Proposal:
    """
    Предложенное действие. Исход ещё не разыгран.
    """

    ts: datetime
    kind: str
    screen: str | None = None
    domain: str = "home"
    operation: str | None = None
    intent: PaymentIntent | None = None
    step: int = 0
    attempt: int = 1
    offers: bool = False
    tag: str = ""


@dataclass(frozen=True)
class Resolution:
    status: str | None = None


@dataclass
class ScenarioRun:
    scenario: str = ""
    target: str | None = None
    session_id: str = ""
    started_at: datetime | None = None

    authorized: bool = False
    auth_required: bool = False
    auth_failures: int = 0

    # Моменты, по которым проверяется инвариант входа.
    auth_success_ts: datetime | None = None
    first_screen_ts: datetime | None = None
    first_action_ts: datetime | None = None
    screens: int = 0
    actions: int = 0
    completed: bool = False
    abandoned: bool = False
    support_used: int = 0
    intents: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    explore_depth: dict[str, str] = field(default_factory=dict)
    viewed_products: list[str] = field(default_factory=list)
    reached_confirm: bool = False


DEPTH_ORDER = {"root": 0, "calc": 1, "terms": 2}

Probe = Callable[[datetime], StateView]


# ============================================================
# РАСПИСАНИЕ СЕССИЙ
# ============================================================


def session_starts_for_day(client_id: int, day: datetime) -> list[tuple[int, datetime]]:
    """
    Сессии дня в порядке НАЧАЛА, а не индекса.

    Ключи те же, что в v1: число сессий и время старта не
    меняются, меняется только наполнение.
    """

    persona = draw_persona(client_id)

    count = day_rng(NS_SCREEN, client_id, day.toordinal()).poisson(
        daily_session_rate(persona, day)
    )

    starts = [
        (
            index,
            draw_session_start(
                day,
                event_rng(NS_SCREEN, client_id, day.toordinal(), index, COMPONENT_TIME),
            ),
        )
        for index in range(count)
    ]

    return sorted(starts, key=lambda item: (item[1], item[0]))


def new_session(
    client_id: int, day: datetime, index: int, started_at: datetime
) -> AppSession:
    return AppSession(
        session_id=make_session_id(client_id, day, index), started_at=started_at
    )


# ============================================================
# ВЫБОР СЦЕНАРИЯ
# ============================================================


def choose_scenario(
    habits: ClientHabits,
    view: ContextView,
    owned: frozenset[str],
    adopted: tuple[str, ...],
    rng: KeyedRandom,
) -> tuple[str, str | None]:

    weights = scenario_weights(habits.app, view, owned, adopted)

    names = [name for name in SCENARIOS if weights[name] > 0.0]

    if not names:
        return SCENARIOS[0], None

    scenario = str(rng.choice(names, p=[weights[name] for name in names]))

    target: str | None = None

    if scenario == PRODUCT_EXPLORE:

        targets = explore_target_weights(habits.app, view, owned, adopted)

        if not targets:
            return SCENARIOS[0], None

        keys = sorted(targets)

        target = str(rng.choice(keys, p=[targets[key] for key in keys]))

    return scenario, target


# ============================================================
# СЕССИЯ
# ============================================================


def run_session(
    client_id: int,
    day: datetime,
    index: int,
    started_at: datetime,
    view: ContextView,
    habits: ClientHabits,
    persona: Persona,
    probe: Probe,
    run: ScenarioRun,
) -> Generator[Proposal, Resolution, None]:
    """
    Одна сессия приложения как последовательность предложений.

    Сначала вход, и только после него экраны. Все проверки
    состояния идут через probe на момент самого действия.
    """

    adopted = adopted_domains(client_id)

    ordinal = day.toordinal()

    def nav(step: int, purpose: int) -> KeyedRandom:
        return event_rng(NS_V2_NAV, client_id, ordinal, index * 100 + step, purpose)

    # --------------------------------------------------------
    # ВХОД
    # --------------------------------------------------------

    ts = started_at

    auth_rng = event_rng(NS_OPERATION, client_id, ordinal, index, COMPONENT_COUNT)

    if probe(started_at).authorized:
        run.authorized = True

    else:
        run.auth_required = True

        biometry = auth_rng.random() < habits.app.biometry_share

        for attempt in range(1, AUTH_MAX_ATTEMPTS + 1):

            outcome = yield Proposal(
                ts=ts,
                kind=KIND_OPERATION,
                domain="auth",
                operation="biometry_login" if biometry else "login",
                step=99,
                attempt=attempt,
                tag=TAG_AUTH,
            )

            if outcome.status == SUCCESS:
                run.authorized = True
                break

            run.auth_failures += 1

            if outcome.status == CANCELLED:
                break

            # Не пустило: следующая попытка паролем, а не биометрией.
            biometry = False
            ts = ts + timedelta(seconds=int(auth_rng.integers(*AUTH_RETRY_SECONDS)))

        if (
            not run.authorized
            and run.auth_failures
            and auth_rng.random() < AUTH_RECOVERY_SHARE
        ):

            # Восстановление доступа: перепривязка устройства.
            ts = ts + timedelta(seconds=int(auth_rng.integers(*AUTH_RETRY_SECONDS)))

            outcome = yield Proposal(
                ts=ts,
                kind=KIND_OPERATION,
                domain="auth",
                operation="device_bind",
                step=98,
                attempt=1,
                tag=TAG_AUTH,
            )

            if outcome.status == SUCCESS:
                run.authorized = True

    if not run.authorized:
        # Внутрь приложения клиент так и не попал.
        run.abandoned = True
        return

    # --------------------------------------------------------
    # СЦЕНАРИЙ
    # --------------------------------------------------------

    scenario_rng = event_rng(
        NS_V2_SCENARIO, client_id, ordinal, index, COMPONENT_CONTENT
    )

    scenario, target = choose_scenario(
        habits, view, probe(ts).owned, adopted, scenario_rng
    )

    run.scenario = scenario
    run.target = target
    run.intents = 1

    steps = steps_for(scenario, target)

    current = "start"
    step_index = 0

    ts = ts + timedelta(
        seconds=int(nav(0, PURPOSE_AUTH).integers(*AUTH_SCREEN_GAP_SECONDS))
    )

    while step_index < MAX_SCREENS_PER_SESSION:

        if not current:

            if run.intents >= MAX_INTENTS_PER_SESSION:
                break

            resume = nav(step_index, PURPOSE_CONTINUE)

            if resume.random() >= CONTINUE_SHARE * habits.app.depth_factor:
                break

            scenario, target = choose_scenario(
                habits, view, probe(ts).owned, adopted, resume
            )

            steps = steps_for(scenario, target)
            current = SCENARIO_ENTRY[scenario]
            run.intents += 1

            if current not in steps:
                break

            continue

        step: Step = steps[current]

        dwell = int(nav(step_index, PURPOSE_DWELL).integers(4, 95)) if step_index else 1

        ts = ts + timedelta(seconds=max(MIN_STEP_SECONDS, dwell))

        if (ts - started_at).total_seconds() > 3600:
            run.abandoned = not run.completed
            break

        if step.screen is not None:

            yield Proposal(
                ts=ts,
                kind=KIND_SCREEN,
                screen=step.screen,
                domain=step.domain,
                offers=step.screen == SCREEN_OFFERS,
                step=step_index,
            )

            product = DOMAIN_PRODUCT.get(step.domain)

            if product is not None and product not in run.viewed_products:
                run.viewed_products.append(product)

            if current == "start" and scenario != PRODUCT_EXPLORE:

                # Витрина предложений это часть домашнего экрана:
                # в v1 она показывалась в начале любой сессии,
                # и объём баннеров держится на ней.
                offers_rng = nav(step_index, PURPOSE_BANNER)

                if offers_rng.random() < 0.22 + 0.20 * persona.digital_affinity:

                    ts = ts + timedelta(
                        seconds=max(MIN_STEP_SECONDS, int(offers_rng.integers(4, 40)))
                    )

                    yield Proposal(
                        ts=ts,
                        kind=KIND_SCREEN,
                        screen=SCREEN_OFFERS,
                        domain="home",
                        offers=True,
                        step=step_index,
                    )

        if step.depth is not None and scenario == PRODUCT_EXPLORE and target:

            previous = run.explore_depth.get(target, "root")

            if DEPTH_ORDER[step.depth] >= DEPTH_ORDER[previous]:
                run.explore_depth[target] = step.depth

        if step.tag == "confirm":
            run.reached_confirm = True

        step_index += 1

        # ----------------------------------------------------
        # ОПЕРАЦИЯ
        # ----------------------------------------------------

        if step.operation is not None:

            attempt = run.attempts.get(current, 0) + 1
            run.attempts[current] = attempt

            op_ts = ts + timedelta(
                seconds=int(nav(step_index, PURPOSE_VARIANT).integers(1, 40))
            )

            intent = None

            if step.operation.startswith("pay_") and step.operation != "pay_qr":
                intent = payment_intent(current, probe(op_ts))

            outcome = yield Proposal(
                ts=op_ts,
                kind=KIND_OPERATION,
                domain=step.domain,
                operation=step.operation,
                intent=intent,
                step=step_index,
                attempt=attempt,
                tag=step.tag,
            )

            ts = op_ts

            current, steps = _after_operation(
                status=outcome.status,
                step=step,
                attempt=attempt,
                run=run,
                habits=habits,
                rng=nav(step_index, PURPOSE_RECOVER),
                steps=steps,
            )

            if step.operation == "logout" and outcome.status == SUCCESS:
                break

            step_index += 1

            continue

        # ----------------------------------------------------
        # ОБЫЧНАЯ РАЗВИЛКА
        # ----------------------------------------------------

        current = _branch(
            step=step,
            steps=steps,
            view=probe(ts),
            rng=nav(step_index, PURPOSE_BRANCH),
            abandon=habits.app.abandon,
        )

    if current:
        run.abandoned = not run.completed


def payment_intent(branch: str, view: StateView) -> PaymentIntent:
    """
    Какое платёжное намерение исполняет эта операция.

    Счёт берётся из открытых НА ЭТОТ МОМЕНТ: уже оплаченный
    второй раз не выбирается, поэтому повтор не может списать
    один счёт дважды.
    """

    kind = BRANCH_BILL_KIND.get(branch, "free")

    for bill in view.due_bills:
        if bill.kind == kind or (kind == "tax" and bill.kind == "service"):
            return PaymentIntent(kind=bill.kind, bill_key=bill.key)

    return PaymentIntent(kind=kind, bill_key=None)


def _after_operation(
    status: str | None,
    step: Step,
    attempt: int,
    run: ScenarioRun,
    habits: ClientHabits,
    rng: KeyedRandom,
    steps: dict,
) -> tuple[str, dict]:
    """
    Что клиент делает после исхода.
    """

    if status == SUCCESS:

        if step.tag == "confirm":
            run.completed = True

        return step.on_success or "", steps

    if status == CANCELLED:

        choice = str(rng.choice(("back", "exit"), p=(0.6, 0.4)))

        if choice == "back" and step.on_cancel:
            return step.on_cancel, steps

        run.abandoned = True

        return "", steps

    run.failures.append(step.operation or "")

    options = next_step_options(FAILED, attempt, run.support_used)

    weights: list[float] = []

    for option in options:
        if option == "retry":
            weights.append(2.0 * habits.app.retry)
        elif option == "support":
            weights.append(0.8 * habits.app.support_bias)
        else:
            weights.append(1.0)

    choice = str(rng.choice(options, p=weights))

    if choice == "retry" and step.retry_to:
        return step.retry_to, steps

    if choice == "support":
        run.support_used += 1
        return "root", SUPPORT_STEPS

    run.abandoned = True

    return "", steps


def _branch(
    step: Step,
    steps: dict,
    view: StateView,
    rng: KeyedRandom,
    abandon: float,
) -> str:
    """
    Куда клиент пойдёт дальше.

    Ветка, ведущая к бессмысленному действию, не предлагается:
    разблокировать незаблокированную карту нельзя, а перевод
    между своими счетами требует второго счёта.
    """

    if not step.nexts:
        return ""

    # Клиент пришёл платить конкретный счёт.
    for bill in view.due_bills:

        branch = BILL_KIND_BRANCH.get(bill.kind)

        if branch is not None and branch in steps and branch in dict(step.nexts):
            return branch

    names: list[str] = []
    weights: list[float] = []

    for name, weight in step.nexts:

        if name and name in steps:

            if not operation_offered(steps[name].operation, view):
                continue

        names.append(name)
        weights.append(weight)

    if not names:
        return ""

    if "" not in names:
        names.append("")
        weights.append(sum(weights) * abandon)

    return str(rng.choice(names, p=weights))


__all__ = [
    "KIND_OPERATION",
    "TAG_AUTH",
    "KIND_SCREEN",
    "PaymentIntent",
    "Proposal",
    "Resolution",
    "ScenarioRun",
    "choose_scenario",
    "new_session",
    "payment_intent",
    "run_session",
    "session_starts_for_day",
]
