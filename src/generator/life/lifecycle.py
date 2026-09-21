from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_LIFECYCLE, keyed_rng
from .persona import Persona


# ============================================================
# ЖИЗНЕННЫЙ ЦИКЛ И ПАУЗЫ
# ============================================================
#
# Неактивность это ПОЛНОЦЕННОЕ СОСТОЯНИЕ, а не случайный пропуск
# событий. У паузы есть вид, причина, плановая и фактическая
# длительность, и всё это лежит в скрытой истине.
#
# В наблюдаемой ленте видно только: последнее действие до паузы,
# события банка во время паузы, полное отсутствие клиентских
# событий в календарном интервале и фактическое возвращение.
# Дата окончания паузы и её причина заранее не пишутся никогда.
# ============================================================


STATE_PROSPECT = "prospect"
STATE_ONBOARDING = "onboarding"
STATE_NEW = "new_client"
STATE_ACTIVE = "active"
STATE_GROWING = "growing"
STATE_STABLE = "stable"
STATE_STRESS = "financial_stress"
STATE_DELINQUENT = "delinquent"
STATE_DORMANT = "dormant"
STATE_CHURN_RISK = "churn_risk"
STATE_CHURNED = "churned"
STATE_RETURNED = "returned"
STATE_CLOSED = "closed_relationship"


@dataclass(frozen=True)
class Pause:
    kind: str
    reason: str
    start: datetime
    planned_end: datetime
    actual_end: datetime
    return_trigger: str
    reason_known_to_bank: bool

    def covers(self, ts: datetime) -> bool:
        return self.start <= ts < self.actual_end


@dataclass(frozen=True)
class StateChange:
    ts: datetime
    state: str
    cause: str


def plan_pauses(persona: Persona, events: tuple) -> tuple:
    """
    Паузы клиента на горизонте.
    """

    settings = params_module.active().lifecycle

    # Пауза это целые сутки, а не отрезок с произвольного часа:
    # план дня спрашивает о паузе один раз на день, и граница
    # посреди дня оставила бы в тишине половину суток.
    start = max(config.HISTORY_START, persona.relationship_start).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    span_days = (config.HISTORY_END - start).days

    if span_days <= 40:
        return ()

    years = span_days / 365.25

    rate = settings.pause_probability_per_year.get(persona.activity_mode, 0.6)

    rng = keyed_rng(NS_LIFECYCLE, persona.client_ordinal, 1)

    count = rng.poisson(rate * years)

    if persona.vanished_after_registration:
        # Клиент зарегистрировался и не начал пользоваться:
        # пауза начинается сразу и длится до конца окна.
        begin = (
            persona.relationship_start + timedelta(days=int(rng.integers(1, 20)))
        ).replace(hour=0, minute=0, second=0, microsecond=0)
        return (
            Pause(
                kind="full",
                reason="lost_interest",
                start=begin,
                planned_end=config.HISTORY_END,
                actual_end=config.HISTORY_END,
                return_trigger="none",
                reason_known_to_bank=False,
            ),
        )

    pauses: list[Pause] = []

    occupied: list[tuple[datetime, datetime]] = []

    for index in range(min(count, 4)):

        item_rng = keyed_rng(NS_LIFECYCLE, persona.client_ordinal, 2, index)

        kind = item_rng.weighted(settings.pause_kind_weights)

        low, high = settings.pause_length_days[kind]
        planned_days = int(item_rng.integers(low, high + 1))

        offset = int(item_rng.integers(20, max(21, span_days - 20)))
        begin = start + timedelta(days=offset)

        # Фактическая длительность отличается от плановой.
        actual_days = max(7, int(planned_days * item_rng.uniform(0.55, 1.45)))

        finish = begin + timedelta(days=actual_days)

        if finish > config.HISTORY_END:
            finish = config.HISTORY_END

        if any(not (finish <= left or begin >= right) for left, right in occupied):
            continue

        reasons = settings.pause_reasons[kind]

        trigger = item_rng.weighted(settings.return_triggers)

        pauses.append(
            Pause(
                kind=kind,
                reason=str(item_rng.choice(reasons)),
                start=begin,
                planned_end=begin + timedelta(days=planned_days),
                actual_end=finish,
                return_trigger=trigger if finish < config.HISTORY_END else "none",
                reason_known_to_bank=bool(
                    item_rng.random() < settings.pause_reason_becomes_known_share
                ),
            )
        )

        occupied.append((begin, finish))

    pauses.sort(key=lambda item: item.start)

    return tuple(pauses)


def pause_at(pauses: tuple, ts: datetime) -> Pause | None:

    for pause in pauses:
        if pause.covers(ts):
            return pause

    return None


def silenced_streams(pauses: tuple, ts: datetime) -> frozenset:
    """
    Что именно замолкает в паузе.
    """

    pause = pause_at(pauses, ts)

    if pause is None:
        return frozenset()

    settings = params_module.active().activity

    return frozenset(settings.pause_silences.get(pause.kind, ()))


# ============================================================
# СОСТОЯНИЕ ЖИЗНЕННОГО ЦИКЛА
# ============================================================


def initial_state(persona: Persona, ts: datetime) -> str:

    if ts < persona.relationship_start:
        return STATE_PROSPECT

    settings = params_module.active().lifecycle

    months = persona.relationship_months_at(ts)

    if months < settings.onboarding_months:
        return STATE_ONBOARDING

    if months < settings.new_client_months:
        return STATE_NEW

    return STATE_ACTIVE


def month_state(
    persona: Persona,
    ts: datetime,
    previous: str,
    days_since_client_event: int,
    activity_ratio: float,
    stress_level: float,
    worst_dpd: int,
    has_open_contract: bool,
    returned_recently: bool,
) -> tuple[str, str]:
    """
    Состояние на конец месяца и причина перехода.

    Переход зависит от накопленной истории, а не выбирается
    независимо каждый месяц.
    """

    settings = params_module.active().lifecycle

    if ts < persona.relationship_start:
        return STATE_PROSPECT, "not_a_client_yet"

    months = persona.relationship_months_at(ts)

    if worst_dpd >= 30:
        return STATE_DELINQUENT, "dpd30"

    if returned_recently:
        return STATE_RETURNED, "first_action_after_pause"

    if days_since_client_event >= settings.churned_after_days_without_client_events:

        if not has_open_contract:

            # Молчание без продуктов рано или поздно означает,
            # что отношения закончились. Раньше это состояние
            # было объявлено и недостижимо.
            if days_since_client_event >= settings.closed_relationship_after_days:
                return STATE_CLOSED, "relationship_closed"

            return STATE_CHURNED, "no_client_events_and_no_products"

        return STATE_DORMANT, "long_silence_with_open_contract"

    if days_since_client_event >= settings.dormant_after_days_without_client_events:
        return STATE_DORMANT, "no_client_events"

    if previous in (STATE_CHURNED, STATE_CLOSED) and days_since_client_event < 30:
        return STATE_RETURNED, "activity_after_churn"

    if months < settings.onboarding_months:
        return STATE_ONBOARDING, "just_registered"

    if months < settings.new_client_months:
        return STATE_NEW, "recent_registration"

    if stress_level >= 0.45:
        return STATE_STRESS, "stress_episode"

    if activity_ratio <= settings.churn_risk_drop_ratio:
        return STATE_CHURN_RISK, "activity_dropped"

    if activity_ratio >= settings.growing_ratio:
        return STATE_GROWING, "activity_grew"

    low, high = settings.stable_band

    if low <= activity_ratio <= high:
        return STATE_STABLE, "activity_stable"

    return STATE_ACTIVE, "default"


__all__ = [
    "Pause",
    "STATE_ACTIVE",
    "STATE_CHURNED",
    "STATE_CHURN_RISK",
    "STATE_CLOSED",
    "STATE_DELINQUENT",
    "STATE_DORMANT",
    "STATE_GROWING",
    "STATE_NEW",
    "STATE_ONBOARDING",
    "STATE_PROSPECT",
    "STATE_RETURNED",
    "STATE_STABLE",
    "STATE_STRESS",
    "StateChange",
    "initial_state",
    "month_state",
    "pause_at",
    "plan_pauses",
    "silenced_streams",
]
