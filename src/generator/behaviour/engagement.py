from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .. import config
from .. import params as params_module
from ..finance.cards import CLIENT_BLOCK_REASONS
from ..rng import NS_ENGAGEMENT, keyed_rng, stable_hash
from ..truth import transition


# ============================================================
# ОТНОШЕНИЯ КЛИЕНТА С БАНКОМ
# ============================================================
#
# Неактивность не назначается заранее. Она вырастает из
# состояния клиента, которое меняется каждый день и помнит
# прошлое:
#
#   friction F    недовольство банком: отказы, сбои, плохие и
#                 долгие обращения, комиссии, неблагоприятный
#                 repricing, отказ по заявке. Забывается с
#                 полупериодом, хороший исход обращения снимает
#                 часть;
#   x = ln(A/база) доля денежной жизни, которая идёт через этот
#                 банк, относительно обычной для клиента
#                 (persona.visible_share). Медленно дрейфует сама
#                 и уходит вниз во время миграции — латентного
#                 переноса жизни в другой банк;
#   режим          active или lapsed: пользуется ли клиент банком
#                 добровольно. Уход и возвращение — случайные
#                 переходы с опасностью от F, стресса, глубины
#                 миграции и привязок (зарплата здесь, вклад,
#                 кредитная карта);
#   away           внешняя отлучка: отпуск без карты, жизнь на
#                 наличных, поломка телефона. От F, x и стресса
#                 не зависит.
#
# Одно и то же состояние даёт разные исходы: переход — розыгрыш,
# а не правило. Плохой опыт без ухода и уход без заметных
# предвестников — оба нормальные пути.
#
# День D решается на состояние начала D: улики с датой раньше D,
# стресс и привязки на D, розыгрыш дня (клиент, D). Будущее —
# ни конец отлучки, ни будущий опыт — не читается.
#
# Ничего из этого не выходит в RAW: только в truth/ (truth.py).
# ============================================================


ACTIVE = "active"
LAPSED = "lapsed"

# Назначения ключей розыгрыша внутри NS_ENGAGEMENT.
_CLIENT = 1
_DAY = 2
_AWAY = 3
_MIGRATION = 4
_NOTICE = 5
_TIE = 6

# Потоки с персональной чувствительностью к доле банка.
_STREAMS = ("purchases", "transfers", "inbound", "cash", "ties")

# Исход обращения, который снимает часть недовольства.
_GOOD_RESOLUTIONS = frozenset(
    {"card_unblocked", "card_reissued", "refund_issued", "record_corrected", "chargeback_started"}
)

# Условия, рост которых клиенту в убыток, и те, рост которых в
# плюс. Ставка у кредита — цена, у вклада — доход.
_COSTS = ("fee_monthly", "fee_issue", "atm_fee_rate", "atm_fee_min", "transfer_fee", "apr", "cash_apr")
_BENEFITS = ("cashback_base", "cashback_cap", "atm_free_monthly", "transfer_free_monthly",
             "cashback_balance_bonus", "cashback_deposit_bonus")
_SAVINGS = ("deposit", "deposit_certificate", "bonds")

# Внешние события, после которых миграция вероятнее.
_LIFE_CHANGES = ("move", "job_change", "job_loss", "divorce")


@dataclass
class Engagement:
    sensitivity: dict
    frailty: float
    salary_threshold: float
    vanish_at: datetime | None = None

    friction: float = 0.0
    friction_day: int = 0
    x: float = 0.0
    target: float = 0.0
    half_life: float = 0.0
    migrating: bool = False

    regime: str = ACTIVE
    cause: str = ""
    away: str | None = None
    away_until: datetime | None = None
    salary_here: bool = True
    servicing: bool = False
    complaint: bool = False

    winback_at: datetime | None = None
    complaint_at: datetime | None = None
    observed: bool = False

    # Улики по дням: порядковый номер дня -> [(время, приоритет,
    # прибавка, доля снятия)]. В силу вступают со следующего дня.
    pending: dict = field(default_factory=dict)
    cases: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    # Привязки, ушедшие в другой банк (гистерезис in_bank).
    moved_out: set = field(default_factory=set)
    journal: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)


# ============================================================
# НАЧАЛО
# ============================================================


def start(persona) -> Engagement:
    """
    Постоянное у клиента: чувствительность потоков, склонность к
    уходу и порог переноса зарплаты.
    """

    settings = params_module.active().engagement

    rng = keyed_rng(NS_ENGAGEMENT, _CLIENT, persona.client_ordinal)

    sigma, low, high = settings.sensitivity

    sensitivity = {
        name: float(min(high, max(low, rng.lognormal(0.0, sigma)))) for name in _STREAMS
    }
    sensitivity["sessions"] = float(rng.uniform(*settings.session_sensitivity))

    spread = settings.frailty_sigma
    frailty = math.exp(spread * rng.normal() - spread * spread / 2.0)

    salary = float(rng.uniform(*settings.salary_threshold))

    vanish_at = None

    if persona.vanished_after_registration:
        early, late, _ = settings.never_started
        vanish_at = (
            persona.relationship_start + timedelta(days=int(rng.integers(early, late + 1)))
        ).replace(hour=0, minute=0, second=0, microsecond=0)

    return Engagement(
        sensitivity=sensitivity,
        frailty=frailty,
        salary_threshold=salary,
        vanish_at=vanish_at,
        half_life=settings.affinity_half_life_days,
    )


def burn_in(state, ties: float, obligated: bool) -> None:
    """
    Клиент, пришедший до окна, входит в него с прожитым состоянием:
    оно прогоняется до min(стаж, burn_in_days) дней до начала окна в
    типичных условиях (burn_in_drivers), без записи в truth. Иначе
    все начинали бы окно с нуля, и доля ушедших росла бы по окну
    сама собой. Розыгрыши — те же ключи (клиент, день): прогрев от
    конца выгрузки не зависит.
    """

    persona = state.persona
    settings = params_module.active().engagement
    drivers = settings.burn_in_drivers

    first = config.HISTORY_START - timedelta(days=settings.burn_in_days)
    day = max(first, persona.relationship_start).replace(hour=0, minute=0, second=0, microsecond=0)

    engagement = state.engagement

    while day < config.HISTORY_START:
        engagement.friction, engagement.friction_day = drivers["friction"], day.toordinal()
        _step(state, day, drivers["stress"], ties, drivers["life_change"], obligated)
        day += timedelta(days=1)

    engagement.friction_day = config.HISTORY_START.toordinal()


# ============================================================
# УЛИКИ
# ============================================================


def note(state, event) -> None:
    """
    Опыт клиента из события симуляции — до фильтра наблюдения:
    отказ случился, даже если система, которая его записала бы,
    ещё не работает. Улика ложится на дату события и действует со
    следующего дня.
    """

    engagement = state.engagement

    if engagement is None or event.event_time < config.HISTORY_START:
        return

    weights = params_module.active().engagement.evidence
    payload = event.payload
    kind = event.event_type

    add = 0.0
    relief = 0.0

    if payload.get("status") == "declined":
        # Своих денег не хватило — не претензия к банку.
        if payload.get("decline_reason") != "insufficient_funds":
            add = weights["decline_bank"]

    elif kind == "app_operation" and payload.get("status") == "failed":
        add = weights["failed_operation"]

    elif kind == "application_decision" and payload.get("decision") == "rejected":
        add = weights["rejected"]

    # Карту, которую клиент блокирует сам, он опытом с банком не считает.
    elif kind == "card_blocked" and payload.get("reason") not in CLIENT_BLOCK_REASONS:
        add = weights["card_block"]

    elif kind == "fee_charge" and payload.get("reason") == "early_closure":
        add = weights["early_closure"]

    elif kind == "case_opened":
        engagement.cases[payload.get("case_id")] = event.event_time

    elif kind == "case_resolved":
        settings = params_module.active().engagement
        resolution = payload.get("resolution")
        opened = engagement.cases.pop(payload.get("case_id"), None)
        if resolution == "declined":
            add = weights["case_declined"]
        elif resolution == "escalated":
            add = weights["case_escalated"]
        elif resolution in _GOOD_RESOLUTIONS:
            relief = settings.case_relief
        if opened is not None and event.event_time - opened > timedelta(hours=settings.case_slow_hours):
            add += weights["case_slow"]

    if add or relief:
        _pend(engagement, event.event_time, add, relief)


def notice_repricing(state, ts: datetime, family: str, before: dict, after: dict) -> None:
    """
    Новые условия договора. Неблагоприятные клиент замечает не в
    тот же день: задержка до notice_days дней — своя у клиента и
    договора. Иначе repricing продукта, который держит треть базы,
    давал бы всплеск недовольства у всех в один день.
    """

    engagement = state.engagement

    if engagement is None or not _adverse(family, before or {}, after or {}):
        return

    settings = params_module.active().engagement

    delay = keyed_rng(
        NS_ENGAGEMENT, _NOTICE, state.ordinal, ts.toordinal(), stable_hash(family) % (2 ** 31)
    ).integers(0, settings.notice_days + 1)

    weight = settings.evidence["repricing"] * 2.0 * state.persona.trait("price_sensitivity", ts)

    _pend(engagement, ts + timedelta(days=int(delay)), weight, 0.0)


def _adverse(family: str, before: dict, after: dict) -> bool:

    score = 0.0

    costs = _COSTS + (() if family in _SAVINGS else ("rate",))
    benefits = _BENEFITS + (("rate",) if family in _SAVINGS else ())

    for names, sign in ((costs, 1.0), (benefits, -1.0)):
        for name in names:
            old, new = before.get(name), after.get(name)
            if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old != new:
                base = max(abs(float(old)), abs(float(new)), 1e-9)
                score += sign * (float(new) - float(old)) / base

    return score > 0.0


def _pend(engagement: Engagement, moment: datetime, add: float, relief: float) -> None:
    items = engagement.pending.setdefault(moment.toordinal(), [])
    items.append((moment, len(items), add, relief))


def _fold(engagement: Engagement, day: int) -> None:
    """
    Улики с датой раньше day — по порядку дней и времени. Между
    ними недовольство затухает по прошедшим дням: пропущенный
    вызов ничего бы не поменял.
    """

    settings = params_module.active().engagement

    half_life = settings.friction_half_life_days
    cap = settings.friction_cap

    for moment in sorted(item for item in engagement.pending if item < day):

        engagement.friction *= 2.0 ** (-(moment - engagement.friction_day) / half_life)
        engagement.friction_day = moment

        for _, _, add, relief in sorted(engagement.pending.pop(moment), key=lambda item: (item[0], item[1])):
            engagement.friction = min(cap, engagement.friction * (1.0 - relief) + add)

    engagement.friction *= 2.0 ** (-(day - engagement.friction_day) / half_life)
    engagement.friction_day = day


# ============================================================
# ОПАСНОСТИ
# ============================================================
#
# Ставки переходов в сутки и члены их логарифма — по ним же
# называется причина перехода в truth.
# ============================================================


def migration_hazard(engagement: Engagement, persona, stress: float, ties: float,
                     life_change: float) -> tuple[float, dict]:
    """
    life_change — была ли смена жизни в последние дни (1 или 0); при
    прогреве — доля таких дней, и множитель берётся средний.
    """

    settings = params_module.active().engagement
    weights = settings.migration_weights

    terms = {
        "friction": weights["friction"] * engagement.friction,
        "stress": weights["stress"] * stress,
        "life_change": math.log(1.0 + (settings.life_change[0] - 1.0) * float(life_change)),
    }

    mode = settings.mode_factor.get(persona.activity_mode, 1.0)

    return settings.migration_rate * mode * math.exp(sum(terms.values()) - weights["ties"] * ties), terms


def lapse_hazard(engagement: Engagement, persona, day: datetime, stress: float,
                 ties: float) -> tuple[float, dict]:

    settings = params_module.active().engagement
    weights = settings.lapse_weights
    amplitude, scale = settings.onboarding

    tenure = max(0, (day - persona.relationship_start).days)

    terms = {
        "friction": weights["friction"] * engagement.friction,
        "stress": weights["stress"] * stress,
        "migration": weights["deficit"] * max(0.0, -engagement.x),
        "onboarding": math.log(1.0 + amplitude * math.exp(-tenure / scale)),
    }

    mode = settings.mode_factor.get(persona.activity_mode, 1.0)

    rate = (
        settings.lapse_rate * mode * engagement.frailty
        * math.exp(sum(terms.values()) - weights["ties"] * ties)
    )

    return rate, terms


def return_hazard(engagement: Engagement, day: datetime, stress: float) -> float:

    settings = params_module.active().engagement
    weights = settings.return_weights

    rate = settings.return_rate * math.exp(
        -weights["friction"] * engagement.friction
        - weights["stress"] * stress
        - weights["deficit"] * max(0.0, -engagement.x)
    )

    boost, window = settings.winback

    if engagement.winback_at is not None and 0 <= (day - engagement.winback_at).days <= window:
        rate *= boost

    if engagement.cause == "never_started":
        rate *= settings.never_started[2]

    return rate


def away_hazard(persona, day: datetime) -> float:
    """
    Внешняя отлучка: только режим активности и время года. Ни
    недовольство, ни доля банка, ни стресс сюда не входят.
    """

    settings = params_module.active().engagement

    mode = settings.mode_factor.get(persona.activity_mode, 1.0)
    season = 1.0 + settings.away_season * math.cos(2.0 * math.pi * (day.month - 7) / 12.0)

    return settings.away_rate * mode * season


def _chance(rate: float) -> float:
    return 1.0 - math.exp(-rate)


# ============================================================
# ДЕНЬ
# ============================================================


def advance(state, day: datetime, stress: float, ties: float, obligated: bool) -> None:
    """
    Состояние на начало дня. Переходы пишутся в truth, только если
    день внутри окна выгрузки.
    """

    engagement = state.engagement

    if not engagement.observed and day >= config.HISTORY_START:
        engagement.observed = True
        _initial_rows(state, day)

    _step(state, day, stress, ties, _life_change(state, day), obligated)

    if engagement.observed and day.weekday() == 0 and day < config.HISTORY_END:
        engagement.snapshots.append(_snapshot(state, day, stress))


def _step(state, day: datetime, stress: float, ties: float, life_change: float, obligated: bool) -> None:

    engagement = state.engagement
    persona = state.persona
    settings = params_module.active().engagement

    _fold(engagement, day.toordinal())

    friction = engagement.friction

    # Один и тот же набор розыгрышей каждый день, в одном порядке:
    # решения одного дня не сдвигают розыгрыши другого.
    rng = keyed_rng(NS_ENGAGEMENT, _DAY, state.ordinal, day.toordinal())
    pick_regime, pick_migration, pick_away, pick_complaint = (rng.random() for _ in range(4))
    noise = rng.normal()

    # --- доля банка ---

    pace = 1.0 - 2.0 ** (-1.0 / engagement.half_life)
    engagement.x += (engagement.target - engagement.x) * pace + settings.affinity_noise * noise

    # --- миграция ---

    if not engagement.migrating:

        rate, terms = migration_hazard(engagement, persona, stress, ties, life_change)

        if pick_migration < _chance(rate):
            content = keyed_rng(NS_ENGAGEMENT, _MIGRATION, state.ordinal, day.toordinal())
            engagement.target = math.log(content.uniform(*settings.migration_depth))
            engagement.half_life = float(content.uniform(*settings.migration_half_life_days))
            engagement.migrating = True
            _record(state, day, "migration", "start", _cause(terms))

    else:

        base, weight = settings.migration_rollback

        if pick_migration < _chance(base * math.exp(-weight * friction)):
            engagement.target = 0.0
            engagement.half_life = settings.affinity_half_life_days
            engagement.migrating = False
            _record(state, day, "migration", "end", "rollback")

    # --- уход и возвращение ---

    if engagement.regime == ACTIVE:

        if engagement.vanish_at is not None and day >= engagement.vanish_at:
            engagement.vanish_at = None
            _lapse(state, day, "never_started")

        else:

            rate, terms = lapse_hazard(engagement, persona, day, stress, ties)

            if pick_regime < _chance(rate):
                _lapse(state, day, _cause(terms))

    else:

        window = settings.winback[1]
        winback = engagement.winback_at is not None and 0 <= (day - engagement.winback_at).days <= window

        if pick_regime < _chance(return_hazard(engagement, day, stress)):
            engagement.regime = ACTIVE
            engagement.cause = ""
            _record(state, day, "regime", ACTIVE, "winback" if winback else "baseline")

    engagement.servicing = engagement.regime == LAPSED and obligated

    # --- внешняя отлучка ---

    if engagement.away is not None and day >= engagement.away_until:
        _record(state, day, "away", "end", engagement.away)
        engagement.away = None
        engagement.away_until = None

    if engagement.away is None and engagement.regime == ACTIVE:

        if pick_away < _chance(away_hazard(persona, day)):
            content = keyed_rng(NS_ENGAGEMENT, _AWAY, state.ordinal, day.toordinal())
            kinds = settings.away_kinds
            kind = content.weighted({name: value[0] for name, value in kinds.items()})
            _, low, high = kinds[kind]
            length = int(content.uniform(low, high) * content.uniform(*settings.away_jitter))
            engagement.away = kind
            engagement.away_until = day + timedelta(days=max(1, length))
            _record(state, day, "away", kind, "exogenous")

    # --- зарплата ---

    level = math.exp(engagement.sensitivity["ties"] * engagement.x)

    if engagement.salary_here and level < engagement.salary_threshold:
        engagement.salary_here = False
        _record(state, day, "salary", "elsewhere", "migration")
    elif not engagement.salary_here and level > engagement.salary_threshold + settings.tie_hysteresis:
        engagement.salary_here = True
        _record(state, day, "salary", "here", "recovery")

    # --- жалоба ---

    rate, cooldown = settings.complaint
    quiet = engagement.complaint_at is None or (day - engagement.complaint_at).days >= cooldown
    engagement.complaint = quiet and pick_complaint < min(0.05, rate * friction)


def _lapse(state, day: datetime, cause: str) -> None:

    engagement = state.engagement

    engagement.regime = LAPSED
    engagement.cause = cause

    if engagement.away is not None:
        _record(state, day, "away", "end", engagement.away)
        engagement.away = None
        engagement.away_until = None

    _record(state, day, "regime", LAPSED, cause)


def _cause(terms: dict) -> str:

    name, value = max(terms.items(), key=lambda item: item[1])

    threshold = params_module.active().engagement.cause_threshold

    return name if value >= threshold else "baseline"


def _life_change(state, day: datetime) -> bool:

    window = params_module.active().engagement.life_change[1]

    return any(
        event.kind in _LIFE_CHANGES and 0 <= (day - event.ts).days < window
        for event in state.life_events
        if event.ts < day
    )


# ============================================================
# ПОТОКИ
# ============================================================


def silenced(state, ts: datetime) -> frozenset:
    """
    Какие добровольные потоки сейчас молчат. Банк своё продолжает:
    выплаты, начисления, выписки, кредитный учёт, рассылки.
    """

    engagement = state.engagement

    if engagement.regime == LAPSED:
        quiet = {"purchases", "transfers", "cash", "bills"}
        if not engagement.servicing:
            quiet.add("sessions")
        return frozenset(quiet)

    if engagement.away == "offline":
        return frozenset({"purchases", "transfers", "cash", "sessions"})

    if engagement.away == "no_app":
        return frozenset({"sessions"})

    if engagement.away == "no_cards":
        return frozenset({"purchases", "cash"})

    return frozenset()


def factor(state, stream: str) -> float:
    """
    Множитель частоты потока относительно обычной для клиента. При
    x = 0 и F = 0 он равен единице: обычная жизнь не меняется.
    """

    engagement = state.engagement
    settings = params_module.active().engagement

    x = engagement.x
    damp = math.exp(-settings.dampening * engagement.friction)

    if stream in ("purchases", "transfers"):
        return math.exp(engagement.sensitivity[stream] * x) * damp

    if stream in ("inbound", "cash"):
        return math.exp(engagement.sensitivity[stream] * x)

    if stream == "applications":
        return math.exp(x) * damp

    if stream == "sessions":
        value = math.exp(engagement.sensitivity["sessions"] * x) * damp
        return value * settings.servicing if engagement.servicing else value

    if stream == "response":
        power, lapsed = settings.response
        value = math.exp(power * x)
        return value * lapsed if engagement.regime == LAPSED else value

    if stream == "support":
        return 1.0 + settings.support * engagement.friction

    if stream == "deposits":
        return math.exp(x)

    raise KeyError(f"нет множителя для потока {stream!r}")


def depth(state) -> float:
    """
    Во сколько раз глубже обычного клиент смотрит приложение.
    """

    engagement = state.engagement

    return math.exp(0.5 * engagement.sensitivity["sessions"] * engagement.x)


def outflow(state) -> float:
    """
    Переезд жизни в другой банк уносит туда и деньги: переводы на
    свой счёт там учащаются и крупнеют.
    """

    settings = params_module.active().engagement

    return 1.0 + settings.outflow * max(0.0, 1.0 - math.exp(state.engagement.x))


def servicing(state) -> bool:
    return state.engagement.servicing


def in_bank(state, kind: str, key: tuple) -> bool:
    """
    Идёт ли регулярный платёж (счёт, подписка) через этот банк.
    У каждого свой порог: при переезде жизни в другой банк они
    уходят по одному, а не разом, и возвращаются с гистерезисом,
    когда доля банка восстанавливается, — а не мигают от месяца к
    месяцу.
    """

    engagement = state.engagement
    settings = params_module.active().engagement

    name = (kind, *key)

    threshold = engagement.thresholds.get(name)

    if threshold is None:
        low, high = settings.tie_threshold
        draw = keyed_rng(NS_ENGAGEMENT, _TIE, state.ordinal, stable_hash(*name) % (2 ** 31)).random()
        threshold = engagement.thresholds[name] = low + (high - low) * draw

    level = math.exp(engagement.sensitivity["ties"] * engagement.x)

    if name in engagement.moved_out:
        if level > threshold + settings.tie_hysteresis:
            engagement.moved_out.discard(name)
    elif level < threshold:
        engagement.moved_out.add(name)

    return name not in engagement.moved_out


def salary_here(state, ts: datetime) -> bool:
    return state.engagement.salary_here


def complaint(state) -> bool:
    """
    Недовольный клиент жалуется: опасность растёт с F. Жалоба —
    наблюдаемое обращение, но не приговор: уходят и без жалоб, а
    жалуются и те, кто остаётся.
    """

    return state.engagement.complaint


def complained(state, ts: datetime) -> None:
    state.engagement.complaint_at = ts


def clicked_winback(state, ts: datetime) -> None:
    state.engagement.winback_at = ts


def ties(state, day: datetime) -> float:
    """
    Привязки к банку: зарплата приходит сюда, вклад, кредитная
    карта. С ними уходить и мигрировать труднее.
    """

    families = state.owned_families(day)

    salary = state.engagement.salary_here and any(
        stream.kind == "salary" and stream.landing == "hcb_account" and stream.active_at(day)
        for stream in state.income_streams
    )

    return (
        float(salary)
        + float(bool({"deposit", "deposit_certificate"} & families))
        + float("credit_card" in families)
    )


def obligated(state, day: datetime) -> bool:
    """
    Долг в банке, ради которого заходят в приложение и после ухода.
    """

    return state.has_open_loan() or "credit_card" in state.owned_families(day)


# ============================================================
# СТАДИЯ
# ============================================================


def stage(persona, day: datetime, stress: float, dpd: int) -> str:
    """
    Стадия клиента на день — только то, что задано не лентой:
    срок с прихода в банк, стресс и просрочка. От неё зависит
    множитель частот (activity.state_factor).

    Считается каждый день, а не в конце месяца: пришедший в банк в
    середине месяца раньше жил в prospect с нулевым множителем до
    конца месяца, а смена стадии случалась ступенькой на границе
    месяца у всех сразу.
    """

    settings = params_module.active().lifecycle

    if day < persona.relationship_start:
        return "prospect"

    if dpd >= settings.delinquent_dpd:
        return "delinquent"

    months = persona.relationship_months_at(day)

    if months < settings.onboarding_months:
        return "onboarding"

    if months < settings.new_client_months:
        return "new_client"

    if stress >= settings.stress_stage_level:
        return "financial_stress"

    return "active"


# ============================================================
# ПРАВДА
# ============================================================


def _record(state, day: datetime, component: str, value: str, cause: str) -> None:

    engagement = state.engagement

    if engagement.observed and day < config.HISTORY_END:
        engagement.journal.append(transition(state.client_id, day, component, value, cause))


def _initial_rows(state, day: datetime) -> None:
    """
    С чем клиент вошёл в окно: прогрев шёл без записи.
    """

    engagement = state.engagement

    if engagement.regime == LAPSED:
        _record(state, day, "regime", LAPSED, "before_window")

    if engagement.migrating:
        _record(state, day, "migration", "start", "before_window")

    if engagement.away is not None:
        _record(state, day, "away", engagement.away, "before_window")

    if not engagement.salary_here:
        _record(state, day, "salary", "elsewhere", "before_window")


def _snapshot(state, day: datetime, stress: float) -> dict:

    from ..profile import utc

    engagement = state.engagement

    return {
        "client_id": state.client_id,
        "time": utc(day),
        "friction": round(engagement.friction, 6),
        "affinity": round(engagement.x, 6),
        "target": round(engagement.target, 6),
        "regime": engagement.regime,
        "away": engagement.away,
        "migrating": engagement.migrating,
        "salary_here": engagement.salary_here,
        "stress": round(stress, 6),
    }


__all__ = [
    "ACTIVE",
    "LAPSED",
    "Engagement",
    "advance",
    "away_hazard",
    "burn_in",
    "clicked_winback",
    "complained",
    "complaint",
    "depth",
    "factor",
    "in_bank",
    "lapse_hazard",
    "migration_hazard",
    "note",
    "notice_repricing",
    "obligated",
    "outflow",
    "return_hazard",
    "salary_here",
    "servicing",
    "silenced",
    "stage",
    "start",
    "ties",
]
