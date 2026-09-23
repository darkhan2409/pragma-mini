from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_LIFE, keyed_rng
from ..world import geography
from .persona import Persona


# ============================================================
# ЖИЗНЕННЫЕ СОБЫТИЯ
# ============================================================
#
# Событие меняет НЕСКОЛЬКО потоков сразу. Потеря работы это не
# одна строка в ленте: исчезает зачисление зарплаты, падает
# остаток, меняется структура трат, растёт вероятность пропуска
# платежа, меняется использование приложения и реакция на
# предложения банка.
#
# Банк узнаёт о событии позже, чем оно произошло, и далеко не
# о каждом.
# ============================================================


EVENT_KINDS = (
    "job_change",
    "job_loss",
    "income_up",
    "income_down",
    "child_birth",
    "move",
    "big_purchase",
    "illness",
    "vacation",
    "renovation",
    "education",
    "wedding",
    "divorce",
)

# Событие -> какие поля профиля оно меняет.
PROFILE_EFFECT = {
    "move": ("region", "city"),
    # Новая работа возвращает занятость: без income_type анкета
    # навсегда оставалась «безработный», хотя зарплата уже идёт.
    "job_change": ("income_type", "industry", "declared_income", "income_day"),
    # Потеря работы уносит с собой и отрасль, и день выплаты:
    # оставленные от прошлого места, они сообщали о клиенте то,
    # чего уже нет.
    "job_loss": ("income_type", "industry", "declared_income", "income_day"),
    "income_up": ("declared_income",),
    "income_down": ("declared_income",),
    "child_birth": ("children",),
    "wedding": ("family_status",),
    "divorce": ("family_status",),
}


@dataclass(frozen=True)
class LifeEvent:
    kind: str
    ts: datetime
    payload: dict
    known_to_bank_at: datetime | None
    confirmed: bool

    @property
    def changes_profile(self) -> tuple:
        return PROFILE_EFFECT.get(self.kind, ())


def _rate(persona: Persona, kind: str) -> float:

    settings = params_module.active().lifecycle

    rate = settings.life_event_rate_per_year.get(kind, 0.0)

    factor = settings.life_event_stage_factor.get(persona.life_stage, {}).get(kind, 1.0)

    return rate * factor


def _move_target(persona: Persona, rng) -> dict:

    settings = params_module.active().lifecycle

    home = geography.by_name(persona.settlement)

    if rng.random() < settings.move_to_other_settlement_share:

        candidates = [
            item
            for item in geography.settlements()
            if item.name != home.name
        ]

        weights = []

        for item in candidates:
            weight = item.population_weight
            if item.region == home.region:
                weight *= 6.0
            weights.append(weight)


        index = int(rng.choice(len(candidates), p=weights))
        target = candidates[index]

        return {
            "settlement": target.name,
            "region": target.region,
            "settlement_type": target.settlement_type,
            "district": target.districts[rng.integers(0, len(target.districts))],
            "same_settlement": False,
        }

    district = home.districts[rng.integers(0, len(home.districts))]

    return {
        "settlement": home.name,
        "region": home.region,
        "settlement_type": home.settlement_type,
        "district": district,
        "same_settlement": True,
    }


def plan_events(persona: Persona) -> tuple:
    """
    Расписание жизненных событий на горизонте.
    """

    settings = params_module.active().lifecycle

    span_days = (config.HISTORY_END - config.HISTORY_START).days
    years = span_days / 365.25

    events: list[LifeEvent] = []


    for kind in EVENT_KINDS:

        rate = _rate(persona, kind)

        if rate <= 0.0:
            continue

        rng = keyed_rng(NS_LIFE, persona.client_ordinal, EVENT_KINDS.index(kind))

        count = rng.poisson(rate * years)

        limit = settings.life_event_max.get(kind)

        if limit is not None:
            count = min(count, limit)

        for index in range(count):

            item_rng = keyed_rng(
                NS_LIFE, persona.client_ordinal, EVENT_KINDS.index(kind), 100 + index
            )

            offset = item_rng.integers(0, span_days)
            ts = config.HISTORY_START + timedelta(days=int(offset), hours=int(item_rng.integers(8, 20)))

            payload: dict = {}

            if kind == "move":
                payload = _move_target(persona, item_rng)
            elif kind == "job_change":
                payload = {
                    "industry": persona.industry,
                    "income_factor": float(item_rng.uniform(0.85, 1.45)),
                    "gap_days": int(item_rng.integers(0, 40)),
                }
            elif kind == "job_loss":
                payload = {"recovery_days": int(item_rng.integers(30, 300))}
            elif kind in ("income_up", "income_down"):
                payload = {
                    "factor": float(
                        item_rng.uniform(1.06, 1.30) if kind == "income_up" else item_rng.uniform(0.70, 0.94)
                    )
                }
            elif kind == "big_purchase":
                payload = {
                    "category": str(
                        item_rng.choice(("furniture", "electronics", "appliances", "car_service", "travel"))
                    ),
                    "amount_factor": float(item_rng.uniform(1.5, 6.0)),
                }
            elif kind == "illness":
                payload = {"days": int(item_rng.integers(5, 60))}
            elif kind == "vacation":
                payload = {
                    "days": int(item_rng.integers(5, 21)),
                    "abroad": bool(item_rng.random() < 0.35 + 0.4 * persona.trait("mobility")),
                }
            elif kind == "renovation":
                payload = {"months": int(item_rng.integers(1, 6))}
            elif kind == "education":
                payload = {"months": int(item_rng.integers(3, 12))}

            known_share = settings.profile_change_known_share.get(kind, 0.0)

            if known_share > 0.0 and item_rng.random() < known_share:
                delay = item_rng.integers(*settings.profile_change_delay_days)
                known_at = ts + timedelta(days=int(delay))
                if known_at >= config.HISTORY_END:
                    known_at = None
            else:
                known_at = None

            events.append(
                LifeEvent(
                    kind=kind,
                    ts=ts,
                    payload=payload,
                    known_to_bank_at=known_at,
                    confirmed=bool(item_rng.random() < settings.profile_change_confirmed_share),
                )
            )

    events.sort(key=lambda item: (item.ts, item.kind))

    return tuple(_in_time_order(persona, events))


def _in_time_order(persona: Persona, events: list) -> list:
    """
    Семейное положение меняется по порядку ВРЕМЕНИ, а не
    розыгрыша: даты разыгрываются независимо, и развод иначе мог
    оказаться раньше свадьбы. Событие, которое в этот момент
    невозможно, выпадает. Рождению значение не нужно: анкета
    добавляет по ребёнку на каждое сообщение банку.
    """

    settings = params_module.active().lifecycle

    family_status = persona.family_status

    kept = []

    for event in events:

        if event.kind in ("wedding", "divorce"):
            if family_status not in settings.life_event_requires.get(event.kind, ()):
                continue
            family_status = "married" if event.kind == "wedding" else "divorced"
            event = replace(event, payload={"family_status_after": family_status})

        kept.append(event)

    return kept


def active_vacation(events: tuple, ts: datetime) -> LifeEvent | None:

    for item in events:
        if item.kind != "vacation":
            continue
        if item.ts <= ts < item.ts + timedelta(days=int(item.payload.get("days", 0))):
            return item

    return None


__all__ = [
    "EVENT_KINDS",
    "PROFILE_EFFECT",
    "LifeEvent",
    "active_vacation",
    "plan_events",
]
