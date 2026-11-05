from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_EMPLOYMENT, NS_INCOME, keyed_rng, stable_hash
from . import calendar as cal
from .persona import PENSION_AGE, Persona, employer_payday, pension_payday
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


# Государственные плательщики: пенсия (ГК «Правительство для
# граждан»), пособия ГФСС и бюджета. Платят по закону — в срок,
# целиком, общим реестром, с индексацией одной для всех.
STATE_PAYERS: frozenset[str] = frozenset({"state_pension", "state_benefit", "state_budget"})


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


def employer_schedule(employer_id: str) -> str:
    """
    График зарплаты работодателя: один на всех его сотрудников.
    """

    weights = params_module.active().income.schedule_weights["salary"]

    return str(keyed_rng(NS_INCOME, stable_hash("schedule", employer_id) % (2 ** 31), 11).weighted(weights))


def build_streams(persona: Persona, events: tuple) -> tuple:
    """
    Потоки дохода клиента с их сроками действия.
    """

    settings = params_module.active().income

    rng = keyed_rng(NS_INCOME, persona.client_ordinal, 1)

    primary_kind = settings.primary_kind_by_income_type.get(persona.income_type, "salary")

    schedule = rng.weighted(settings.schedule_weights.get(primary_kind, {"monthly": 1.0}))

    # Зарплату по графику платит работодатель: у коллег он общий.
    # Личный розыгрыш выше оставлен, чтобы не сдвинуть следующие.
    if primary_kind == "salary" and persona.employer_id is not None:
        schedule = employer_schedule(persona.employer_id)

    payer = persona.employer_id or f"payer_{stable_hash('payer', persona.client_ordinal) % 10 ** 8:08d}"

    # Пенсию платит государство, а не личный «плательщик»: прежде она
    # опаздывала у 12% и приходила частями (аудит 2026-11-05, M9).
    if primary_kind == "pension":
        payer = "state_pension"

    streams: list[IncomeStream] = [
        IncomeStream(
            stream_id=f"inc_{persona.client_ordinal}_1",
            kind=primary_kind,
            payer=payer,
            schedule=schedule,
            payday=persona.income_day,
            landing=_landing(persona, primary_kind, rng),
            base_amount=persona.true_income,
            valid_from=min(config.HISTORY_START, persona.relationship_start),
            valid_to=None,
        )
    ]

    # Пенсия по возрасту тому, кто пенсионером по типу дохода не
    # объявлен. Профиль называет его пенсионером с того дня, как
    # ему исполнилось PENSION_AGE, и без этого потока запись была
    # бы ярлыком без денег: продукты с условием requires_pension
    # открывались человеку, у которого пенсии нет вовсе.
    #
    # Она НЕ заменяет заработок: работающий пенсионер получает и
    # зарплату, и выплату. Поэтому это отдельный поток, а не
    # смена основного.
    if persona.income_type != "pensioner":

        # 29 февраля переносится на 28-е: replace на невисокосный
        # год иначе падает.
        born = persona.birth_date

        turns = born.replace(
            year=born.year + persona.pension_age,
            day=28 if (born.month, born.day) == (2, 29) else born.day,
        )

        if turns < config.PLANNING_END:

            pension_rng = keyed_rng(NS_INCOME, persona.client_ordinal, 5)

            amount = max(
                settings.state_pension_floor,
                int(persona.true_income * pension_rng.uniform(*settings.state_pension_of_income)),
            )

            # Розыгрыш дня оставлен, чтобы не сдвинуть следующие: день
            # выплаты — по графику даты рождения.
            pension_rng.integers(3, 11)

            streams.append(
                IncomeStream(
                    stream_id=f"inc_{persona.client_ordinal}_pension",
                    kind="pension",
                    payer="state_pension",
                    schedule="monthly",
                    payday=pension_payday(born),
                    landing=_landing(persona, "pension", pension_rng),
                    base_amount=amount,
                    valid_from=max(turns, min(config.HISTORY_START, persona.relationship_start)),
                    valid_to=None,
                )
            )

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

    # Жизненные события закрывают и открывают потоки. Номер потока
    # берётся от события — его дня, — а не от длины списка: иначе
    # пенсия, появившаяся при более далёком горизонте планирования,
    # сдвигала бы номера всех потоков после неё, а с ними и
    # розыгрыши сумм (_amount_at).

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

        if event.kind not in ("job_loss", "job_change", "child_birth"):
            continue

        primary = streams[primary_position]

        if event.kind == "child_birth":

            if persona.gender != "F":
                continue

            leave, benefits, back = _maternity(persona, event, primary, settings, income_shifts)

            streams += benefits

            if leave is not None:
                streams[primary_position] = replace(primary, valid_to=leave)
                if back is not None:
                    streams.append(back)
                    primary_position = len(streams) - 1

            continue

        if primary.kind not in ("salary",):
            continue

        if primary.valid_to is not None and primary.valid_to <= event.ts:
            continue

        item_rng = keyed_rng(NS_INCOME, persona.client_ordinal, 2, int(event.ts.toordinal()))

        tag = int(event.ts.toordinal())

        # Смена работы закрывает прежний поток в день события, даже
        # если новое место начнётся за горизонтом планирования:
        # между работами дохода нет, и это правда. Раньше смену у
        # края горизонта пропускали, и продление горизонта меняло
        # прошлое — поток закрывался задним числом.
        streams[primary_position] = replace(primary, valid_to=event.ts)

        if event.kind == "job_loss":

            # Выходное пособие и пособие по безработице.
            severance_months = item_rng.integers(*settings.severance_months)

            if severance_months > 0:
                streams.append(
                    IncomeStream(
                        stream_id=f"inc_{persona.client_ordinal}_severance_{tag}",
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

            # Пособие ГФСС по потере работы: 45% дохода, 1–6 месяцев.
            # Прежде — 15–40% до нового места, сколько бы ни искал.
            if item_rng.random() < settings.unemployment_benefit_share:
                payday = int(item_rng.integers(5, 25))
                landing = "hcb_account" if item_rng.random() < 0.55 else "other_bank"
                months = int(item_rng.integers(*settings.unemployment_benefit_months))
                begin = event.ts + timedelta(days=int(item_rng.integers(10, 45)))
                streams.append(
                    IncomeStream(
                        stream_id=f"inc_{persona.client_ordinal}_benefit_{tag}",
                        kind="social_benefit",
                        payer="state_benefit",
                        schedule="monthly",
                        payday=payday,
                        landing=landing,
                        base_amount=int(
                            min(primary.base_amount, settings.social_income_cap_mzp * settings.mzp)
                            * settings.unemployment_benefit_of_income
                        ),
                        valid_from=begin,
                        valid_to=min(
                            begin + timedelta(days=30 * months),
                            event.ts + timedelta(days=int(event.payload.get("recovery_days", 120))),
                        ),
                    )
                )

            restart = event.ts + timedelta(days=int(event.payload.get("recovery_days", 120)))

        else:
            restart = event.ts + timedelta(days=int(event.payload.get("gap_days", 0)))

        if restart >= config.PLANNING_END:
            continue

        factor = float(event.payload.get("income_factor", item_rng.uniform(0.85, 1.35)))

        employer = f"emp_{stable_hash('employer', persona.client_ordinal, int(event.ts.toordinal())) % 10 ** 9:09d}"

        # График и день выплаты назначает новый работодатель.
        # Прежние личные розыгрыши оставлены пустыми, чтобы не
        # сдвинуть следующие.
        item_rng.weighted(settings.schedule_weights["salary"])
        item_rng.integers(1, 29)

        schedule = employer_schedule(employer)

        streams.append(
            IncomeStream(
                stream_id=f"inc_{persona.client_ordinal}_job_{tag}",
                kind="salary",
                payer=employer,
                schedule=schedule,
                payday=employer_payday(employer),
                landing=_landing(persona, "salary", item_rng),
                base_amount=int(max(60_000, primary.base_amount * factor)),
                valid_from=restart,
                valid_to=None,
            )
        )

        primary_position = len(streams) - 1

    # Изменение дохода касается каждого потока клиента, кроме
    # государственных выплат: их сумму задаёт закон.
    if income_shifts:
        streams = [item if item.payer in STATE_PAYERS else replace(item, shifts=income_shifts) for item in streams]

    return tuple(streams)


def _maternity(persona: Persona, event, primary: IncomeStream, settings, income_shifts: tuple) -> tuple:
    """
    Деньги рождения у матери: (начало декрета или None, потоки пособий,
    поток зарплаты после выхода или None).

    Пособие при рождении — каждой. Работающая по найму уходит в декрет
    за maternity_days_before до родов: ГФСС платит 126 дней разом, затем
    40% дохода по уходу до полутора лет или до выхода на работу;
    зарплата на это время останавливается и возобновляется у того же
    работодателя. Неработающей по уходу платит бюджет. Прежде рождение
    денег не меняло вовсе (аудит 2026-11-05, M15).
    """

    amounts = params_module.active().amounts

    tag = int(event.ts.toordinal())

    rng = keyed_rng(NS_INCOME, persona.client_ordinal, 13, tag)

    landing = _landing(persona, "social_benefit", rng)

    def delay() -> timedelta:
        return timedelta(days=int(rng.integers(*settings.benefit_delay_days)))

    def mrp(ts: datetime) -> int:
        return amounts.mrp_by_year.get(ts.year, max(amounts.mrp_by_year.values()))

    def stream(name: str, payer: str, schedule: str, amount: int, start: datetime, end: datetime) -> IncomeStream:
        return IncomeStream(
            stream_id=f"inc_{persona.client_ordinal}_{name}_{tag}",
            kind="social_benefit",
            payer=payer,
            schedule=schedule,
            payday=min(28, start.day),
            landing=landing,
            base_amount=int(amount),
            valid_from=start,
            valid_to=end,
        )

    order = int(event.payload.get("order", 1))

    paid = event.ts + delay()

    benefits = [
        stream("birth", "state_budget", "once",
               settings.birth_benefit_mrp[0 if order <= 3 else 1] * mrp(event.ts), paid, paid + timedelta(days=1)),
    ]

    leave = event.ts - timedelta(days=settings.maternity_days_before)

    back_at = event.ts + timedelta(days=int(event.payload.get("return_days", settings.childcare_days)))

    care_from = leave + timedelta(days=settings.maternity_days)

    care_to = min(back_at, event.ts + timedelta(days=settings.childcare_days))

    working = primary.kind == "salary" and primary.valid_from <= leave and primary.active_at(leave)

    if not working:
        if primary.kind == "salary" and primary.active_at(event.ts):
            return None, benefits, None
        share = settings.childcare_budget_mrp[min(order, len(settings.childcare_budget_mrp)) - 1]
        start = event.ts + delay()
        benefits.append(stream("childcare", "state_budget", "monthly", share * mrp(event.ts), start,
                               event.ts + timedelta(days=settings.childcare_days)))
        return None, benefits, None

    income = min(
        settings.social_income_cap_mzp * settings.mzp,
        _amount_at(primary, leave, persona.client_ordinal, settings, income_shifts),
    )

    start = leave + delay()

    benefits.append(stream("maternity", "state_benefit", "once", income * settings.maternity_days / 30.4,
                           start, start + timedelta(days=1)))

    if care_to > care_from:
        start = care_from + delay()
        benefits.append(stream("childcare", "state_benefit", "monthly", income * settings.childcare_share,
                               start, max(start + timedelta(days=1), care_to)))

    back = None

    if back_at < config.PLANNING_END:
        back = replace(
            primary,
            stream_id=f"inc_{persona.client_ordinal}_return_{tag}",
            base_amount=_amount_at(primary, back_at, persona.client_ordinal, settings, income_shifts),
            valid_from=back_at,
            valid_to=None,
        )

    return leave, tuple(benefits), back


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

    # Государство: пенсия индексируется с 1 января одной ставкой для
    # всех, пособие назначено суммой и не меняется. Без личных
    # повышений и понижений.
    if stream.payer in STATE_PAYERS:
        amount = float(stream.base_amount)
        if stream.kind == "pension":
            for year, rate in sorted(settings.pension_indexation.items()):
                if anchor < datetime(year, 1, 1) <= ts:
                    amount *= 1.0 + rate
        return int(round(amount))

    months = max(0, cal.month_index(ts) - cal.month_index(anchor))

    rng = keyed_rng(NS_INCOME, rng_seed, 7, stable_hash(stream.stream_id) % (2 ** 31))

    indexation = rng.uniform(*settings.annual_indexation)

    # Индексация — ступенька раз в год, в месяц индексации, а не
    # непрерывный рост: иначе выплата росла на доли процента каждый
    # месяц, чего в жизни не бывает.
    month = _indexation_month(stream)

    steps = sum(
        1
        for index in range(cal.month_index(anchor) + 1, cal.month_index(ts) + 1)
        if (index - 1) % 12 + 1 == month
    )

    amount = stream.base_amount * ((1.0 + indexation) ** steps)

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


def _indexation_month(stream: IncomeStream) -> int:
    """
    Месяц годовой индексации выплаты. Зарплату индексирует
    работодатель — месяц общий у всех его сотрудников, чаще январь;
    пенсии, пособия и прочие выплаты — с января.
    """

    if stream.kind == "salary" and str(stream.payer or "").startswith("emp_"):
        rng = keyed_rng(NS_INCOME, stable_hash(stream.payer) % (2 ** 31), 11)
        return 1 if rng.random() < 0.5 else int(rng.integers(1, 13))

    return 1


def _bonus_moment(stream, day: datetime, rng) -> datetime:
    """
    Премия — в рабочий день, во время реестра работодателя. Прежде
    100% премий приходили ровно в 12:00:00 или 11:00:00, а годовую
    все получали 26 декабря (аудит 2026-11-05, M8).
    """

    settings = params_module.active().income

    day = _shift_for_calendar(day)

    payer = keyed_rng(NS_INCOME, stable_hash(stream.payer, "bonus", cal.month_index(day)) % (2 ** 31), 12)

    batch = day.replace(hour=int(payer.integers(*settings.payout_hour_range)), minute=int(payer.integers(0, 60)),
                        second=0, microsecond=0)

    return batch + timedelta(seconds=int(rng.integers(0, 240)))


def _state_day(planned: datetime) -> datetime:
    """
    День выплаты государства: с выходного — на рабочий день накануне,
    но не в прошлый месяц. Выплата за месяц приходится на свой месяц,
    иначе в одном месяце было бы две пенсии, а в следующем ни одной.
    """

    if cal.is_business_day(planned):
        return planned

    earlier = cal.previous_business_day(planned)

    return earlier if earlier.month == planned.month else cal.next_business_day(planned)


def _state_moment(stream: IncomeStream, day: datetime, rng) -> datetime:
    """
    Выплата государства — реестром дня: у всех получателей одного
    плательщика и одного дня графика в одно время, строки реестра
    расходятся на секунды–минуты.
    """

    settings = params_module.active().income

    batch_rng = keyed_rng(NS_INCOME, stable_hash(stream.payer, day.toordinal()) % (2 ** 31), 15)

    batch = day.replace(hour=int(batch_rng.integers(*settings.payout_hour_range)),
                        minute=int(batch_rng.integers(0, 60)), second=0, microsecond=0)

    return batch + timedelta(seconds=int(rng.integers(0, 240)))


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
        # Горизонт планирования, а не конец выгрузки: следующая
        # зарплата читается решением о трате (engine.py:191-203),
        # и обрыв списка на границе окна менял бы поведение в
        # последние недели.
        stop = min(stream.valid_to or config.PLANNING_END, config.PLANNING_END)

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

        # --- разовая выплата: пособие при рождении, декретные ---

        if stream.schedule == "once":

            rng = keyed_rng(NS_INCOME, persona.client_ordinal, 14, stable_hash(stream.stream_id) % (2 ** 31))

            day = stream.valid_from if cal.is_business_day(stream.valid_from) else cal.next_business_day(stream.valid_from)

            moment = _state_moment(stream, day, rng)

            if config.HISTORY_START <= moment < config.PLANNING_END:
                result.append(
                    Payout(moment, moment, stream.stream_id, stream.kind, stream.base_amount,
                           stream.landing, "on_time", stream.payer)
                )

            continue

        # --- регулярные выплаты ---

        paydays = [stream.payday]

        if stream.schedule == "twice_monthly":
            paydays.append(((stream.payday + settings.second_payday_offset - 1) % 28) + 1)

        # Выплата следующего за горизонтом месяца, пришедшая раньше
        # срока или перенесённая с выходного, ложится ДО горизонта.
        # Поэтому месяцы перебираются на месяц дальше него, а в ленту
        # идёт только то, что раньше stop: иначе поднятый PLANNING_END
        # дописывал бы выплаты в прежнее окно.
        horizon = stream.valid_to is None or stream.valid_to >= config.PLANNING_END
        last = config.PLANNING_END + timedelta(days=31) if horizon else stop

        month = cal.month_start(start)

        while month < last:

            base = _amount_at(stream, month, persona.client_ordinal, settings, stream.shifts)

            per_payday = base // len(paydays)

            for slot, day in enumerate(paydays):

                planned = cal.day_in_month(month, day)

                # Исход выплаты (ранняя, поздняя, частичная) и её час
                # личные: они зависят от стресса и дисциплины самого
                # клиента.
                rng = keyed_rng(
                    NS_INCOME, persona.client_ordinal, 5,
                    stable_hash(stream.stream_id, cal.month_index(month), slot) % (2 ** 31),
                )

                # Перенос с выходного у зарплаты работодателя общий
                # для всех его сотрудников: день выплаты назначает он.
                employer = stream.kind == "salary" and str(stream.payer or "").startswith("emp_")

                state = stream.payer in STATE_PAYERS

                calendar_rng = (
                    keyed_rng(
                        NS_INCOME, stable_hash(stream.payer) % (2 ** 31), 10,
                        cal.month_index(month), slot,
                    )
                    if employer
                    else rng
                )

                stress = level_at(stress_episodes, planned)

                weights = dict(settings.payout_outcome)

                if state:
                    # Государство платит в срок и целиком.
                    outcome = "on_time"
                elif employer:
                    # Реестр зарплаты общий у работодателя: срок,
                    # задержка и частичная выплата — его, а не стресс и
                    # дисциплина сотрудника. Прежде у коллег в один
                    # день выплаты разошлись на часы, а опаздывали 27%
                    # выплат (аудит 2026-11-05, M8). Работодатель не
                    # «пропускает» месяц: долг он выплачивает задержкой.
                    weights["late"] += weights.pop("skipped")
                    outcome = calendar_rng.weighted(weights)
                else:
                    weights["late"] += settings.stress_late_boost * stress
                    weights["skipped"] += settings.stress_skip_boost * stress * stress
                    weights["partial"] += settings.stress_partial_boost * stress
                    weights["on_time"] *= discipline_of_payer
                    outcome = rng.weighted(weights)

                if outcome == "skipped":
                    continue

                moment = planned

                timing_rng = calendar_rng if employer else rng

                if outcome == "early":
                    moment -= timedelta(days=int(timing_rng.integers(*settings.early_days)))
                elif outcome == "late":
                    moment += timedelta(days=int(timing_rng.integers(*settings.late_days)))

                # Работодатель платит в рабочий день (ТК РК ст. 113: при
                # выходном — накануне) всегда; прочие плательщики — не
                # всегда.
                if state:
                    moment = _state_moment(stream, _state_day(moment), rng)
                elif employer or calendar_rng.random() < settings.weekend_shift_share:
                    moment = _shift_for_calendar(moment)

                if employer:
                    # Пачка реестра — в один момент у всех сотрудников,
                    # строки одного реестра расходятся на секунды–минуты.
                    batch = moment.replace(
                        hour=int(calendar_rng.integers(*settings.payout_hour_range)),
                        minute=int(calendar_rng.integers(0, 60)),
                        second=0,
                        microsecond=0,
                    )
                    moment = batch + timedelta(seconds=int(rng.integers(0, 240)))
                elif not state:
                    moment = moment.replace(
                        hour=int(rng.integers(*settings.payout_hour_range)),
                        minute=int(rng.integers(0, 60)),
                        second=int(rng.integers(0, 60)),
                        microsecond=0,
                    )

                amount = per_payday

                if outcome == "partial":
                    share = (calendar_rng if employer else rng).uniform(*settings.partial_share)
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

                    topup = _shift_for_calendar(
                        moment + timedelta(days=int((calendar_rng if employer else rng).integers(*settings.partial_topup_days)))
                    )

                    if start <= topup < stop and amount - first > 0:
                        result.append(
                            Payout(
                                # Доплата — своим моментом, а не с минутой
                                # и секундой первой части (аудит
                                # 2026-11-05, income: 1809 пар).
                                ts=topup.replace(hour=int(rng.integers(*settings.payout_hour_range)),
                                                 minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60))),
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
                    ts = _bonus_moment(stream, cal.day_in_month(month, min(28, stream.payday + 2)), bonus_rng)
                    if start <= ts < stop:
                        result.append(
                            Payout(ts, ts, stream.stream_id, stream.kind, amount,
                                   stream.landing, "bonus", stream.payer)
                        )

                if month.month == settings.annual_bonus_month and bonus_rng.random() < settings.annual_bonus_share:
                    amount = int(base * bonus_rng.uniform(*settings.bonus_share_of_income))
                    ts = _bonus_moment(stream, cal.day_in_month(month, 15 + stable_hash("annual", stream.payer) % 13), bonus_rng)
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

    result: list[Payout] = []

    for event in events:

        if event.kind != "vacation":
            continue

        # Ключ — номер отпуска среди отпусков, а не место в общем
        # списке событий.
        rng = keyed_rng(NS_INCOME, persona.client_ordinal, 9, event.serial)

        if rng.random() >= settings.vacation_pay_share:
            continue

        ts = event.ts - timedelta(days=int(rng.integers(1, 5)))

        # Отпускные платит работодатель, у которого человек работает
        # в день выплаты, а не первый в списке: после смены работы
        # прежнего уже нет.
        salary = next(
            (item for item in streams if item.kind == "salary" and item.active_at(ts)), None
        )

        if salary is None or not (config.HISTORY_START <= ts < config.PLANNING_END):
            continue

        amount = int(salary.base_amount * rng.uniform(*settings.vacation_pay_of_income))

        # Отпускные — в рабочий день, не позже чем за 3 рабочих дня до
        # отпуска (ТК РК ст. 92), во время реестра работодателя.
        # Прежде они приходили ровно в 10:00:00, в том числе в
        # воскресенье (аудит 2026-11-05, M8).
        day = _shift_for_calendar(min(ts, event.ts - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0))
        moment = _bonus_moment(salary, day, rng)

        result.append(
            Payout(moment, ts, salary.stream_id, salary.kind, amount,
                   salary.landing, "vacation_pay", salary.payer)
        )

    return tuple(result)


def expected_payday(streams: tuple, day: datetime) -> datetime | None:
    """
    Ближайший день зарплаты или пенсии, начиная с day, по графику
    потоков, которые действуют сегодня.

    Человек ждёт зарплату по графику своей работы и не знает, что
    через неделю её потеряет. Поэтому здесь читается только
    сегодняшнее состояние потоков, а не реализованные будущие
    выплаты: в них будущая потеря работы уже стёрла следующую
    зарплату, и траты сегодня от неё зависели.
    """

    settings = params_module.active().income

    month = cal.month_start(day)

    best = None

    for stream in streams:

        if stream.kind not in ("salary", "pension") or stream.schedule == "irregular":
            continue

        if not stream.active_at(day):
            continue

        paydays = [stream.payday]

        if stream.schedule == "twice_monthly":
            paydays.append(((stream.payday + settings.second_payday_offset - 1) % 28) + 1)

        for current in (month, cal.next_month(month)):
            for payday in paydays:
                planned = cal.day_in_month(current, payday)
                if planned >= day and (best is None or planned < best):
                    best = planned

    return best


def monthly_income(streams: tuple, ts: datetime) -> int:
    """
    Ожидаемый месячный доход домохозяйства на дату — с повышениями
    и снижениями из жизненных событий, как и сами выплаты
    (_amount_at). Раньше бюджет их не видел: деньги менялись, а
    траты жили по прежнему доходу.
    """

    total = 0.0

    for stream in streams:
        # Разовая выплата — не месячный доход.
        if stream.active_at(ts) and stream.schedule != "once":
            amount = float(stream.base_amount)
            for moment, factor in stream.shifts:
                if stream.valid_from <= moment <= ts:
                    amount *= factor
            total += amount

    return int(total)


# ============================================================
# НАЁМНАЯ РАБОТА ГЛАЗАМИ БАНКА
# ============================================================
#
# Банк знает о работе клиента то, что ему сообщили: дату начала
# работы и момент, когда запись о ней появилась. Запись без даты
# начала — сообщение, что наёмной работы больше нет.
#
#   первая работа   на ней клиент был к началу потока зарплаты;
#                   банк знает её с начала отношений. Даты начала
#                   генератор прежде не знал: она разыгрывается
#                   здесь, в своём пространстве NS_EMPLOYMENT, и
#                   чужих розыгрышей не сдвигает;
#   смена работы    новое место с даты выхода; банк узнаёт о нём
#                   тогда же, когда об изменении анкеты, но не
#                   раньше выхода на работу;
#   потеря работы   запись без даты начала — только если потеря
#                   оборвала зарплату, как и изменение анкеты.
#
# Событие, о котором банк не узнал, записи не даёт: банк о нём и
# не знает. Работа не по найму (бизнес, фриланс, пенсия, учёба)
# стажа на месте работы не имеет, и записей у неё нет.
# ============================================================


def job_start(persona: Persona, stream_start: datetime) -> date:
    """
    Начало работы, на которой клиент был к началу потока
    зарплаты: не раньше совершеннолетия и не глубже
    job_tenure_months_max месяцев.
    """

    settings = params_module.active().population

    adult = persona.birth_date + timedelta(days=int(18 * 365.25))

    available = max(0, int((stream_start - adult).days / 30.44))

    months = keyed_rng(NS_EMPLOYMENT, persona.client_ordinal, 1).integers(
        0, min(settings.job_tenure_months_max, available) + 1
    )

    return (stream_start - timedelta(days=int(months * 30.44))).date()


def employment(persona: Persona, events: tuple, streams: tuple) -> list[tuple[date | None, datetime]]:
    """
    Записи банка о наёмной работе: (дата начала или None, момент
    записи), по моменту записи.
    """

    settings = params_module.active().income

    if settings.primary_kind_by_income_type.get(persona.income_type, "salary") != "salary":
        return []

    records: list[tuple[date | None, datetime]] = [
        (job_start(persona, streams[0].valid_from), persona.relationship_start)
    ]

    for event in events:

        if event.kind not in ("job_change", "job_loss") or event.known_to_bank_at is None:
            continue

        if event.kind == "job_loss":

            if any(item.kind == "salary" and item.valid_to == event.ts for item in streams):
                records.append((None, event.known_to_bank_at))

            continue

        start = event.ts + timedelta(days=int(event.payload.get("gap_days", 0)))

        if any(item.kind == "salary" and item.valid_from == start for item in streams):
            records.append((start.date(), max(event.known_to_bank_at, start)))

    return sorted(records, key=lambda item: item[1])


__all__ = [
    "STATE_PAYERS",
    "IncomeStream",
    "Payout",
    "build_streams",
    "employer_schedule",
    "employment",
    "job_start",
    "monthly_income",
    "payouts",
    "vacation_payouts",
]
