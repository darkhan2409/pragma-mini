from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_LIFE, NS_TRIP, keyed_rng
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
#
# События каждого вида идут вперёд по времени: следующее — через
# случайный интервал после предыдущего. Поэтому горизонт
# планирования только обрезает будущее: поднятый PLANNING_END
# дописывает события после прежнего и не трогает ни одного
# раньше него. Раньше число событий бралось Poisson от длины
# горизонта, а даты раскладывались по нему же.
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

# На сколько дней дальше горизонта планирования планируются события.
# Часть последствий наступает РАНЬШЕ самого события: билет за 7–60
# дней до поездки, декрет за 70 дней до родов, покупки к свадьбе и
# рождению (event_purchases). Событие сразу за горизонтом даёт строки
# до него, и без запаса поднятый PLANNING_END переписывал бы
# последние недели прежнего окна. Запас длиннее любого такого сдвига.
LOOKAHEAD = timedelta(days=180)

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
    # Номер события среди событий своего вида: ключ его
    # розыгрышей в других модулях. Позиция в общем списке ключом
    # быть не может — событие другого вида сдвигало бы её.
    serial: int = 0

    @property
    def changes_profile(self) -> tuple:
        return PROFILE_EFFECT.get(self.kind, ())


def _rate(persona: Persona, kind: str) -> float:

    settings = params_module.active().lifecycle

    # Рождения — кандидаты, которые принимаются по возрасту матери
    # на свою дату (_birth_chance).
    if kind == "child_birth":
        return settings.birth_candidate_rate

    rate = settings.life_event_rate_per_year.get(kind, 0.0)

    factor = settings.life_event_stage_factor.get(persona.life_stage, {}).get(kind, 1.0)

    return rate * factor


def _move_target(settlement: str, rng) -> dict:
    """
    Куда переезжает человек, который сейчас живёт в settlement.
    """

    settings = params_module.active().lifecycle

    home = geography.by_name(settlement)

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

    events: list[LifeEvent] = []

    for kind in EVENT_KINDS:

        rate = _rate(persona, kind)

        if rate <= 0.0:
            continue

        # Предел числа рождений считается по принятым кандидатам
        # (_in_time_order), а не по всем.
        limit = None if kind == "child_birth" else settings.life_event_max.get(kind)

        # Интервалы до следующего события — экспоненциальные со
        # ставкой вида: тот же процесс Пуассона, что и прежде, но
        # шагами вперёд, без длины горизонта.
        clock = 0.0

        index = 0

        while limit is None or index < limit:

            item_rng = keyed_rng(
                NS_LIFE, persona.client_ordinal, EVENT_KINDS.index(kind), 100 + index
            )

            clock += -math.log(1.0 - item_rng.random()) / rate * 365.25

            ts = config.HISTORY_START + timedelta(days=int(clock), hours=int(item_rng.integers(8, 20)))

            if ts >= config.PLANNING_END + LOOKAHEAD:
                break

            # Цель переезда разыгрывается позже, по порядку времени
            # (_in_time_order): второй переезд идёт от того места, где
            # человек живёт после первого, а не от исходного.
            payload: dict = {}

            if kind == "job_change":
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
                abroad = bool(item_rng.random() < 0.35 + 0.4 * persona.trait("mobility"))
                # Куда и чем едут — одна поездка, один раз. Прежде
                # страна разыгрывалась на каждую покупку отпуска, и
                # поездка шла по 5–13 странам (аудит 2026-11-05, C1).
                # Свой поток: розыгрыши события выше не сдвигаются.
                geo = params_module.active().geography
                trip_rng = keyed_rng(NS_TRIP, persona.client_ordinal, index)
                payload = {
                    "days": int(item_rng.integers(5, 21)),
                    "abroad": abroad,
                    "country": str(trip_rng.weighted(geo.foreign_countries)),
                    "destination": str(trip_rng.weighted(geo.domestic_destinations)),
                    "lead_days": int(trip_rng.integers(*geo.trip_ticket_lead_days)),
                    "flight": bool(abroad or trip_rng.random() < geo.domestic_flight_share),
                }
            elif kind == "child_birth":
                low, high, _ = settings.return_to_work_days[
                    int(item_rng.choice(len(settings.return_to_work_days),
                                        p=[item[2] for item in settings.return_to_work_days]))
                ]
                payload = {
                    "accept": float(item_rng.random()),
                    "return_days": int(item_rng.integers(low, high)),
                }
            elif kind == "renovation":
                payload = {"months": int(item_rng.integers(1, 6))}
            elif kind == "education":
                payload = {"months": int(item_rng.integers(3, 12))}

            known_share = settings.profile_change_known_share.get(kind, 0.0)

            # Дата, когда банк узнал, может лежать за горизонтом:
            # это значит «ещё не узнал», и в выгрузку она не попадёт.
            if known_share > 0.0 and item_rng.random() < known_share:
                delay = item_rng.integers(*settings.profile_change_delay_days)
                known_at = ts + timedelta(days=int(delay))
            else:
                known_at = None

            events.append(
                LifeEvent(
                    kind=kind,
                    ts=ts,
                    payload=payload,
                    known_to_bank_at=known_at,
                    confirmed=bool(item_rng.random() < settings.profile_change_confirmed_share),
                    serial=index,
                )
            )

            index += 1

    events.sort(key=lambda item: (item.ts, item.kind))

    # Новый отпуск не начинается, пока не кончился прежний: иначе
    # человек был бы в двух поездках сразу (аудит 2026-11-05, C1).
    kept: list[LifeEvent] = []
    away_until = None
    for item in events:
        if item.kind == "vacation":
            if away_until is not None and item.ts < away_until:
                continue
            away_until = item.ts + timedelta(days=int(item.payload.get("days", 0)) + 7)
        kept.append(item)

    return tuple(_in_time_order(persona, kept))


def _in_time_order(persona: Persona, events: list) -> list:
    """
    События, возможные в свой момент, по порядку ВРЕМЕНИ, а не
    розыгрыша: даты разыгрываются независимо по видам.

      семья — развод только после свадьбы, свадьба только вне брака;
      работа — сменить или потерять работу может только тот, кто
               сейчас получает зарплату: не пенсионер, не
               предприниматель и не тот, кто ещё ищет новое место
               после прошлой потери (тот же порядок, что у потоков
               дохода, income.build_streams). Раньше такие события
               оставались в плане без денег и меняли только анкету
               и стресс;
      переезд — цель от того места, где человек живёт сейчас.

      рождение — кандидат принимается с долей по возрасту матери и
               браку на свою дату (_birth_chance), не раньше чем через
               birth_min_interval_days после прошлого и не больше
               life_event_max за горизонт. Работающая мать уходит в
               декрет: работу на это время она не меняет и не теряет.

    Событие, которое в этот момент невозможно, выпадает.
    """

    settings = params_module.active().lifecycle

    primary = params_module.active().income.primary_kind_by_income_type

    family_status = persona.family_status

    salaried = primary.get(persona.income_type) == "salary"
    working_from = None

    settlement = persona.settlement

    births: list[datetime] = []

    kept = []

    for event in events:

        if event.kind == "child_birth":
            if len(births) >= settings.life_event_max.get("child_birth", 99):
                continue
            if births and (event.ts - births[-1]).days < settings.birth_min_interval_days:
                continue
            if event.payload["accept"] >= _birth_chance(persona, event.ts, family_status) / settings.birth_candidate_rate:
                continue
            births.append(event.ts)
            event = replace(event, payload={**event.payload, "order": persona.children + len(births)})
            working = salaried and (working_from is None or event.ts >= working_from)
            if persona.gender == "F" and working:
                working_from = event.ts + timedelta(days=int(event.payload["return_days"]))

        if event.kind in ("wedding", "divorce"):
            if family_status not in settings.life_event_requires.get(event.kind, ()):
                continue
            family_status = "married" if event.kind == "wedding" else "divorced"
            event = replace(event, payload={"family_status_after": family_status})

        if event.kind in ("job_change", "job_loss"):
            if not salaried or (working_from is not None and event.ts < working_from):
                continue
            pause = event.payload.get("recovery_days" if event.kind == "job_loss" else "gap_days", 0)
            working_from = event.ts + timedelta(days=int(pause))

        if event.kind == "move":
            rng = keyed_rng(NS_LIFE, persona.client_ordinal, EVENT_KINDS.index("move"), 500 + event.serial)
            target = _move_target(settlement, rng)
            settlement = target["settlement"]
            event = replace(event, payload=target)

        kept.append(event)

    return kept


def _birth_chance(persona: Persona, ts: datetime, family_status: str) -> float:
    """
    Рождений в год у клиента на дату: по возрасту матери (у мужчины —
    партнёрши) и браку.
    """

    settings = params_module.active().lifecycle

    age = persona.age_at(ts) - (settings.partner_age_gap if persona.gender == "M" else 0)

    rate = 0.0
    for since, value in settings.birth_rate_by_mother_age:
        if age >= since:
            rate = value

    if age < settings.birth_rate_by_mother_age[0][0]:
        rate = 0.0

    return rate * settings.birth_family_factor.get(family_status, 0.25)


def persona_changes(persona: Persona, events: tuple, streams: tuple) -> tuple:
    """
    Как жизнь меняет персону: (порядковый день, поле, операция,
    значение) по порядку дней. Изменение действует со следующего
    дня после события — поведение дня решается на его начало.

    Меняются только те поля, которые поведение читает и после
    подготовки клиента: дети и размер семьи, место жизни,
    занятость и настоящий доход. Анкета банка (profile_values)
    узнаёт об этом своим путём — событиями profile_change, позже и
    не всегда.

    streams — потоки дохода (income.build_streams): новая работа
    начинается тогда, когда начинается её зарплата.
    """

    changes: list[tuple] = []

    for event in events:

        day = event.ts.toordinal() + 1

        if event.kind == "child_birth":
            changes += [(day, "children", "+", 1), (day, "household_size", "+", 1)]
        elif event.kind == "wedding":
            changes.append((day, "household_size", "+", 1))
        elif event.kind == "divorce":
            changes.append((day, "household_size", "-", 1))
        elif event.kind == "move":
            changes += [
                (day, name, "=", event.payload[name])
                for name in ("settlement", "region", "settlement_type")
            ]
        elif event.kind == "job_loss":
            changes.append((day, "income_type", "=", "unemployed"))
        elif event.kind in ("income_up", "income_down"):
            changes.append((day, "true_income", "*", float(event.payload.get("factor") or 1.0)))

    for stream in streams:
        if "_job_" in stream.stream_id:
            day = stream.valid_from.toordinal() + 1
            changes += [
                (day, "income_type", "=", persona.income_type),
                (day, "true_income", "=", int(stream.base_amount)),
            ]

    return tuple(sorted(changes, key=lambda item: item[0]))


def apply_changes(persona: Persona, changes: tuple) -> Persona:
    """
    Персона после изменений одного дня.
    """

    values: dict = {}

    for _, name, operation, value in changes:

        current = values.get(name, getattr(persona, name))

        if operation == "+":
            current = current + value
        elif operation == "-":
            current = max(1 if name == "household_size" else 0, current - value)
        elif operation == "*":
            current = int(current * value)
        else:
            current = value

        values[name] = current

    return replace(persona, **values)


def event_purchases(persona: Persona, events: tuple) -> dict:
    """
    Покупки жизненных событий по порядковому дню: день -> ((категория,
    во сколько раз чек больше обычного, номер покупки, час), ...).
    Свадьба одевает и кормит гостей, болезнь ведёт в аптеку и клинику,
    ремонт — в магазин товаров для дома, учёба — к оплате курса раз в
    месяц, рождение — к детским товарам и врачу. Прежде у этих событий
    трат не было вовсе (аудит 2026-11-05, M15).
    """

    settings = params_module.active().lifecycle

    plan: dict[int, list] = {}

    for event in events:

        basket = settings.event_baskets.get(event.kind)

        if not basket:
            continue

        kind = EVENT_KINDS.index(event.kind)

        rng = keyed_rng(NS_LIFE, persona.client_ordinal, kind, 700 + event.serial)

        span = _span_days(event)

        number = 0

        for category, low, high, first, last, scale_low, scale_high in basket:

            last = span if last is None else last

            if low is None:
                days = list(range(first, max(first, last) + 1, 30))
            else:
                days = [int(rng.integers(first, max(first, last) + 1)) for _ in range(int(rng.integers(low, high + 1)))]

            for offset in days:
                day = (event.ts + timedelta(days=offset)).toordinal()
                plan.setdefault(day, []).append(
                    (category, float(rng.uniform(scale_low, scale_high)), number, int(rng.integers(10, 21)))
                )
                number += 1

    return {day: tuple(items) for day, items in plan.items()}


def _span_days(event: LifeEvent) -> int:
    """
    Сколько дней длится событие: болезнь — дни, ремонт и учёба — месяцы.
    """

    if "days" in event.payload:
        return int(event.payload["days"])

    if "months" in event.payload:
        return 30 * int(event.payload["months"])

    return 0


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
    "apply_changes",
    "event_purchases",
    "persona_changes",
    "plan_events",
]
