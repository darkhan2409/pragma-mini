from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_INCOME, keyed_rng, stable_hash
from . import calendar as cal
from .persona import Persona
from .stress import level_at


# ============================================================
# ДОХОДЫ
# ============================================================
#
# Зарплата не приходит идеальным таймером. Выплата бывает
# обычной, ранней, поздней, частичной и пропущенной, переносится
# с выходного, меняется в сумме, дополняется премией и
# отпускными, прекращается при увольнении и возобновляется у
# нового работодателя.
#
# Часть дохода не видна этому банку: она приземляется на счёт
# другого банка или приходит наличными.
# ============================================================


# Доход, который банк согласится подтвердить справкой или
# оборотом. Пособие, выходное пособие и помощь семьи стажем не
# считаются: они временные по своей природе.
CONFIRMABLE_KINDS: frozenset[str] = frozenset(
    {"salary", "pension", "business", "freelance", "rent"}
)


@dataclass(frozen=True)
class IncomeStream:
    stream_id: str
    kind: str
    payer: str
    schedule: str
    payday: int
    landing: str
    base_amount: int
    valid_from: datetime
    valid_to: datetime | None
    # Объявленные жизненные события, меняющие сумму потока.
    shifts: tuple = ()

    def active_at(self, ts: datetime) -> bool:
        if ts < self.valid_from:
            return False
        return self.valid_to is None or ts < self.valid_to


@dataclass(frozen=True)
class Payout:
    ts: datetime
    planned_ts: datetime
    stream_id: str
    kind: str
    amount: int
    landing: str
    outcome: str
    payer: str


def _landing(persona: Persona, kind: str, rng) -> str:
    """
    Куда приходит доход. Роль банка сдвигает вероятность: у
    клиента, для которого этот банк основной, зарплатный проект
    здесь.
    """

    settings = params_module.active().income

    weights = dict(settings.landing_weights.get(kind, {"hcb_account": 0.5, "other_bank": 0.4, "cash": 0.1}))

    weights["hcb_account"] *= 0.4 + 1.8 * persona.visible_share
    weights["other_bank"] *= 1.6 - 1.2 * persona.visible_share

    return rng.weighted(weights)


def build_streams(persona: Persona, events: tuple) -> tuple:
    """
    Потоки дохода клиента с их сроками действия.
    """

    settings = params_module.active().income

    rng = keyed_rng(NS_INCOME, persona.client_ordinal, 1)

    primary_kind = settings.primary_kind_by_income_type.get(persona.income_type, "salary")

    schedule = rng.weighted(settings.schedule_weights.get(primary_kind, {"monthly": 1.0}))

    payer = persona.employer_id or f"payer_{stable_hash('payer', persona.client_ordinal) % 10 ** 8:08d}"

    streams: list[IncomeStream] = [
        IncomeStream(
            stream_id=f"inc_{persona.client_ordinal}_1",
            kind=primary_kind,
            payer=payer,
            schedule=schedule,
            payday=persona.salary_day,
            landing=_landing(persona, primary_kind, rng),
            base_amount=persona.true_income,
            valid_from=min(config.HISTORY_START, persona.relationship_start),
            valid_to=None,
        )
    ]

    # Второй поток: подработка, аренда, помощь семьи.
    if rng.random() < settings.second_stream_share:

        second_kind = rng.weighted(settings.second_kind_weights)

        share = rng.uniform(*settings.second_stream_amount_share)

        streams.append(
            IncomeStream(
                stream_id=f"inc_{persona.client_ordinal}_2",
                kind=second_kind,
                payer=f"payer_{stable_hash('second', persona.client_ordinal) % 10 ** 8:08d}",
                schedule=rng.weighted(settings.schedule_weights.get(second_kind, {"monthly": 1.0})),
                payday=int(rng.integers(1, 29)),
                landing=_landing(persona, second_kind, rng),
                base_amount=int(persona.true_income * share),
                valid_from=config.HISTORY_START,
                valid_to=None,
            )
        )

    # Жизненные события закрывают и открывают потоки.
    index = len(streams)

    # Указатель на действующий основной поток: список при этом
    # хранит и прежние места работы, иначе история дохода
    # оборвалась бы задним числом.
    primary_position = 0

    # Изменение дохода это изменение ДЕНЕГ, а не только записи
    # в анкете. Раньше событие меняло заявленный доход, а поток
    # выплат жил по своим розыгрышам.
    income_shifts = tuple(
        (event.ts, float(event.payload.get("factor") or 1.0))
        for event in events
        if event.kind in ("income_up", "income_down")
    )

    for event in events:

        if event.kind not in ("job_loss", "job_change"):
            continue

        primary = streams[primary_position]

        if primary.kind not in ("salary",):
            continue

        if primary.valid_to is not None and primary.valid_to <= event.ts:
            continue

        streams[primary_position] = replace(primary, valid_to=event.ts)

        item_rng = keyed_rng(NS_INCOME, persona.client_ordinal, 2, int(event.ts.toordinal()))

        if event.kind == "job_loss":

            # Выходное пособие и пособие по безработице.
            severance_months = item_rng.integers(*settings.severance_months)

            if severance_months > 0:
                index += 1
                streams.append(
                    IncomeStream(
                        stream_id=f"inc_{persona.client_ordinal}_{index}",
                        kind="severance",
                        payer=primary.payer,
                        schedule="irregular",
                        payday=primary.payday,
                        landing=primary.landing,
                        base_amount=int(primary.base_amount * severance_months),
                        valid_from=event.ts,
                        valid_to=event.ts + timedelta(days=30),
                    )
                )

            if item_rng.random() < settings.unemployment_benefit_share:
                index += 1
                streams.append(
                    IncomeStream(
                        stream_id=f"inc_{persona.client_ordinal}_{index}",
                        kind="social_benefit",
                        payer="state_benefit",
                        schedule="monthly",
                        payday=int(item_rng.integers(5, 25)),
                        landing="hcb_account" if item_rng.random() < 0.55 else "other_bank",
                        base_amount=int(
                            primary.base_amount * item_rng.uniform(*settings.unemployment_benefit_of_income)
                        ),
                        valid_from=event.ts + timedelta(days=int(item_rng.integers(10, 45))),
                        valid_to=event.ts + timedelta(days=int(event.payload.get("recovery_days", 120))),
                    )
                )

            restart = event.ts + timedelta(days=int(event.payload.get("recovery_days", 120)))

        else:
            restart = event.ts + timedelta(days=int(event.payload.get("gap_days", 0)))

        if restart >= config.HISTORY_END:
            continue

        factor = float(event.payload.get("income_factor", item_rng.uniform(0.85, 1.35)))

        index += 1

        streams.append(
            IncomeStream(
                stream_id=f"inc_{persona.client_ordinal}_{index}",
                kind="salary",
                payer=f"emp_{stable_hash('employer', persona.client_ordinal, int(event.ts.toordinal())) % 10 ** 9:09d}",
                schedule=item_rng.weighted(settings.schedule_weights["salary"]),
                payday=int(item_rng.integers(1, 29)),
                landing=_landing(persona, "salary", item_rng),
                base_amount=int(max(60_000, primary.base_amount * factor)),
                valid_from=restart,
                valid_to=None,
            )
        )

        primary_position = len(streams) - 1

    # Изменение дохода касается каждого потока клиента.
    if income_shifts:
        streams = [replace(item, shifts=income_shifts) for item in streams]

    return tuple(streams)


def _amount_at(
    stream: IncomeStream,
    ts: datetime,
    rng_seed: int,
    settings,
    income_shifts: tuple = (),
) -> int:
    """
    Сумма потока на дату: индексация, редкие повышения и
    снижения, а также объявленные жизненные события.
    """

    # Индексация считается от начала окна: доход до наблюдения
    # уже учтён в базовой сумме, и накручивать его нельзя.
    anchor = max(stream.valid_from, config.HISTORY_START)

    months = max(0, cal.month_index(ts) - cal.month_index(anchor))

    rng = keyed_rng(NS_INCOME, rng_seed, 7, stable_hash(stream.stream_id) % (2 ** 31))

    indexation = rng.uniform(*settings.annual_indexation)

    amount = stream.base_amount * ((1.0 + indexation) ** (months / 12.0))

    # Повышение и понижение дохода из жизненного события.
    for moment, factor in income_shifts:
        if moment <= ts and moment >= stream.valid_from:
            amount *= factor

    years = months // 12

    for year in range(years + 1):

        year_rng = keyed_rng(NS_INCOME, rng_seed, 8, stable_hash(stream.stream_id, year) % (2 ** 31))

        if year_rng.random() < settings.raise_share_per_year:
            amount *= year_rng.uniform(*settings.raise_factor)
        elif year_rng.random() < settings.cut_share_per_year:
            amount *= year_rng.uniform(*settings.cut_factor)

    return int(round(amount / 100) * 100)


def _shift_for_calendar(planned: datetime) -> datetime:

    settings = params_module.active().income

    if cal.is_business_day(planned):
        return planned

    if settings.weekend_shift == "earlier":
        return cal.previous_business_day(planned)

    return cal.next_business_day(planned)


def payouts(persona: Persona, streams: tuple, stress_episodes: tuple) -> tuple:
    """
    Все выплаты клиента на горизонте.
    """

    settings = params_module.active().income

    discipline_of_payer = 0.5 + 0.5 * persona.trait("financial_discipline")

    result: list[Payout] = []

    for stream in streams:

        start = max(stream.valid_from, config.HISTORY_START)
        stop = min(stream.valid_to or config.HISTORY_END, config.HISTORY_END)

        if stop <= start:
            continue

        if stream.schedule == "irregular":

            month = cal.month_start(start)

            while month < stop:

                rng = keyed_rng(
                    NS_INCOME, persona.client_ordinal, 3,
                    stable_hash(stream.stream_id, cal.month_index(month)) % (2 ** 31),
                )

                low, high = settings.irregular_events_per_month
                count = rng.poisson(rng.uniform(low, high))

                base = _amount_at(stream, month, persona.client_ordinal, settings, stream.shifts)

                for index in range(count):

                    item_rng = keyed_rng(
                        NS_INCOME, persona.client_ordinal, 4,
                        stable_hash(stream.stream_id, cal.month_index(month), index) % (2 ** 31),
                    )

                    day = item_rng.integers(1, 28)
                    ts = cal.day_in_month(month, day).replace(
                        hour=int(item_rng.integers(*settings.payout_hour_range)),
                        minute=int(item_rng.integers(0, 60)),
                    )

                    if not (start <= ts < stop):
                        continue

                    amount = int(
                        base * item_rng.lognormal(0.0, settings.irregular_amount_sigma) / max(1, count)
                    )

                    if amount < 1000:
                        continue

                    result.append(
                        Payout(
                            ts=ts,
                            planned_ts=ts,
                            stream_id=stream.stream_id,
                            kind=stream.kind,
                            amount=amount,
                            landing=stream.landing,
                            outcome="irregular",
                            payer=stream.payer,
                        )
                    )

                month = cal.next_month(month)

            continue

        # --- регулярные выплаты ---

        paydays = [stream.payday]

        if stream.schedule == "twice_monthly":
            paydays.append(((stream.payday + settings.second_payday_offset - 1) % 28) + 1)

        month = cal.month_start(start)

        while month < stop:

            base = _amount_at(stream, month, persona.client_ordinal, settings, stream.shifts)

            per_payday = base // len(paydays)

            for slot, day in enumerate(paydays):

                planned = cal.day_in_month(month, day)

                # У общего работодателя день выплаты и её
                # задержка общие для всех своих сотрудников.
                payer_seed = (
                    stable_hash(stream.payer) % (2 ** 31)
                    if stream.kind == "salary" and str(stream.payer or "").startswith("emp_")
                    else persona.client_ordinal
                )

                rng = keyed_rng(
                    NS_INCOME, payer_seed, 5,
                    stable_hash(stream.stream_id, cal.month_index(month), slot) % (2 ** 31),
                )

                stress = level_at(stress_episodes, planned)

                weights = dict(settings.payout_outcome)
                weights["late"] += settings.stress_late_boost * stress
                weights["skipped"] += settings.stress_skip_boost * stress * stress
                weights["partial"] += settings.stress_partial_boost * stress
                weights["on_time"] *= discipline_of_payer

                outcome = rng.weighted(weights)

                if outcome == "skipped":
                    continue

                moment = planned

                if outcome == "early":
                    moment -= timedelta(days=int(rng.integers(*settings.early_days)))
                elif outcome == "late":
                    moment += timedelta(days=int(rng.integers(*settings.late_days)))

                if rng.random() < settings.weekend_shift_share:
                    moment = _shift_for_calendar(moment)

                moment = moment.replace(
                    hour=int(rng.integers(*settings.payout_hour_range)),
                    minute=int(rng.integers(0, 60)),
                    second=int(rng.integers(0, 60)),
                    microsecond=0,
                )

                amount = per_payday

                if outcome == "partial":
                    share = rng.uniform(*settings.partial_share)
                    first = int(amount * share)

                    if start <= moment < stop and first > 0:
                        result.append(
                            Payout(
                                ts=moment,
                                planned_ts=planned,
                                stream_id=stream.stream_id,
                                kind=stream.kind,
                                amount=first,
                                landing=stream.landing,
                                outcome="partial",
                                payer=stream.payer,
                            )
                        )

                    topup = moment + timedelta(days=int(rng.integers(*settings.partial_topup_days)))

                    if start <= topup < stop and amount - first > 0:
                        result.append(
                            Payout(
                                ts=topup.replace(hour=int(rng.integers(*settings.payout_hour_range))),
                                planned_ts=planned,
                                stream_id=stream.stream_id,
                                kind=stream.kind,
                                amount=amount - first,
                                landing=stream.landing,
                                outcome="partial_topup",
                                payer=stream.payer,
                            )
                        )

                    continue

                if not (start <= moment < stop) or amount <= 0:
                    continue

                result.append(
                    Payout(
                        ts=moment,
                        planned_ts=planned,
                        stream_id=stream.stream_id,
                        kind=stream.kind,
                        amount=amount,
                        landing=stream.landing,
                        outcome=outcome,
                        payer=stream.payer,
                    )
                )

            # --- премии ---

            if stream.kind == "salary":

                bonus_rng = keyed_rng(
                    NS_INCOME, persona.client_ordinal, 6,
                    stable_hash(stream.stream_id, cal.month_index(month)) % (2 ** 31),
                )

                quarterly = month.month in (3, 6, 9, 12)

                if quarterly and bonus_rng.random() < settings.quarterly_bonus_share:
                    amount = int(base * bonus_rng.uniform(*settings.bonus_share_of_income) * 0.5)
                    ts = cal.day_in_month(month, min(28, stream.payday + 2)).replace(hour=12)
                    if start <= ts < stop:
                        result.append(
                            Payout(ts, ts, stream.stream_id, stream.kind, amount,
                                   stream.landing, "bonus", stream.payer)
                        )

                if month.month == settings.annual_bonus_month and bonus_rng.random() < settings.annual_bonus_share:
                    amount = int(base * bonus_rng.uniform(*settings.bonus_share_of_income))
                    ts = cal.day_in_month(month, 26).replace(hour=11)
                    if start <= ts < stop:
                        result.append(
                            Payout(ts, ts, stream.stream_id, stream.kind, amount,
                                   stream.landing, "annual_bonus", stream.payer)
                        )

            month = cal.next_month(month)

    result.sort(key=lambda item: (item.ts, item.stream_id))

    return tuple(result)


def vacation_payouts(persona: Persona, streams: tuple, events: tuple) -> tuple:
    """
    Отпускные перед отпуском.
    """

    settings = params_module.active().income

    salary = next((item for item in streams if item.kind == "salary"), None)

    if salary is None:
        return ()

    result: list[Payout] = []

    for index, event in enumerate(events):

        if event.kind != "vacation":
            continue

        rng = keyed_rng(NS_INCOME, persona.client_ordinal, 9, index)

        if rng.random() >= settings.vacation_pay_share:
            continue

        ts = event.ts - timedelta(days=int(rng.integers(1, 5)))

        if not (config.HISTORY_START <= ts < config.HISTORY_END) or not salary.active_at(ts):
            continue

        amount = int(salary.base_amount * rng.uniform(*settings.vacation_pay_of_income))

        result.append(
            Payout(ts.replace(hour=10), ts, salary.stream_id, salary.kind, amount,
                   salary.landing, "vacation_pay", salary.payer)
        )

    return tuple(result)


def monthly_income(streams: tuple, ts: datetime) -> int:
    """
    Ожидаемый месячный доход домохозяйства на дату.
    """

    settings = params_module.active().income

    total = 0

    for stream in streams:
        if stream.active_at(ts):
            total += stream.base_amount

    return int(total)


__all__ = [
    "IncomeStream",
    "Payout",
    "build_streams",
    "monthly_income",
    "payouts",
    "vacation_payouts",
]
