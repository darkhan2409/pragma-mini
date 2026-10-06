from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .activity import is_client_action, is_target_action, is_visit
from .config import LOCAL_OFFSET
from .products import DEBIT_CARD, PREMIUM, SERVICE, families


# ============================================================
# CUSTOMER LIFECYCLE МОБИЛЬНОГО ПРИЛОЖЕНИЯ (CAPP)
# ============================================================
#
# Стадия клиента на момент t по его ленте СТРОГО до t: бизнес-разметка
# наблюдаемого поведения, а не скрытое состояние генератора. В признаки
# моделей и во вход PRAGMA стадия не идёт; она нужна для диагностики,
# стратифицированной оценки, определения At Risk и будущего NBA/NBO.
#
# Источник — Excel мобильной команды «CAPP Customer lifecycle stages»
# (ячейки ниже — его) и решения владельца 2026-10-06:
#
#   New        регистрация в приложении (веха app_registered), B2.
#   Activated  первое целевое действие после регистрации: действие
#              клиента или продукт по его заявке (activity), B3.
#   Growing    визит в календарном месяце, следующем за месяцем
#              активации (B4: «in MAU of next month from activation»).
#              Ветка «w/ DC» отдельного порога не имеет — то же правило.
#   CORE       4+ визитов в неделю: скользящее среднее по 4 неделям ≥ 4
#              четыре недели подряд (SQL H7), и 2+ продукта, среди них
#              дебетовая карта (B5).
#   Loyal      CORE и премиальная карта (Тау, Алем; I7). «Close to VIP»
#              (B6.2) не формализован — не реализован.
#   At Risk    по прежней стадии (B7):
#                New        7 дней от регистрации без целевого действия;
#                Activated  14 дней от регистрации без действия после
#                           активации (если активация была в эти 14 дней);
#                Growing    закончился календарный месяц без визита;
#                CORE/Loyal спад H7: среднее по 2 неделям ≤ 2 две недели
#                           подряд после начала последней high-серии;
#                Loyal      закрыта премиальная карта и другой нет (I7).
#              Порог депозита «< 15 mln etc.» (B7.5) не подтверждён и не
#              реализован.
#   Churn      60+ дней без действия клиента (B8; решение владельца:
#              по действиям, а не только по визитам).
#
# Выход из At Risk (решение владельца: стадия пересчитывается заново):
# условие входа перестало выполняться — целевое действие, действие,
# визит, неделя без спада, снова премиальная карта. Новая стадия
# считается по свежим фактам — с начала At Risk: Growing — визит после
# начала у клиента, который уже был Growing; CORE — high-серия, начавшаяся
# после спада. Кто возвращается в Activated, отсчитывает Growing от
# месяца выхода. Выход из Churn — любое действие, стадия пересчитывается
# так же.
#
# Приоритет: Churn > At Risk > Loyal > CORE > Growing > Activated > New.
# Путь активных стадий — только вверх по цепочке; вниз — через At Risk.
#
# Время. Стадия считается раз в сутки, на конец местного дня D: в ней
# все события дня D и раньше, то есть строго до местной полуночи D+1, —
# этот момент и есть started_at перехода. Недельные правила — по
# завершённым неделям ISO (понедельник — воскресенье), месячные — по
# завершённым календарным месяцам.
#
# Владение продуктами — только по событиям ленты: открытие, миграция,
# закрытие, а договор, открытый до окна выгрузки, — с первого события о
# нём (перевыпуск, пролонгация, тарифы). Договоры до окна без таких
# событий не видны: CORE и Loyal для давних клиентов недосчитаны
# (решение владельца: пока так, с пометкой).
# ============================================================


NEW = "new"
ACTIVATED = "activated"
GROWING = "growing"
CORE = "core"
LOYAL = "loyal"
AT_RISK = "at_risk"
CHURN = "churn"

STAGES: tuple[str, ...] = (NEW, ACTIVATED, GROWING, CORE, LOYAL, AT_RISK, CHURN)

# Порядок активных стадий: продвижение — только вверх.
ACTIVE: tuple[str, ...] = (NEW, ACTIVATED, GROWING, CORE, LOYAL)

COLUMNS: tuple[str, ...] = ("client_id", "started_at", "stage", "previous_stage", "reason")

# Продуктовые события: открытие, первое появление и закрытие договора.
OPENED = ("product_opened", "product_migrated", "account_opened")
SIGHTED = ("card_activated", "card_reissued", "product_renewed", "contract_terms_changed", "product_repriced")
CLOSED = ("product_closed",)

APP_REGISTERED = "app_registered"



@dataclass(frozen=True)
class Rules:
    new_days: int = 7            # B7.1, E7
    activated_days: int = 14     # B7.2
    churn_days: int = 60         # B8, E8
    high_visits: float = 4.0     # B5, H7: avg4 ≥ 4
    high_window: int = 4         # H7: 4 недели
    high_run: int = 4            # H7: серия ≥ 4 недель
    low_visits: float = 2.0      # B7.4, H7: avg2 ≤ 2
    low_window: int = 2          # H7: 2 недели
    low_run: int = 2             # H7: серия ≥ 2 недель
    core_products: int = 2       # B5


RULES = Rules()


def registrations(profile: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """
    Момент регистрации в приложении по вехе app_registered анкеты.
    Веха датирована, поэтому её знание на t причинно: она раньше t или
    нет. Клиент без вехи приложением не пользуется, стадии у него нет.
    """
    out: dict[str, pd.Timestamp] = {}
    for client, items in zip(profile["client_id"], profile["lifelong"]):
        moments = [item["event_time"] for item in items if item["type"] == APP_REGISTERED]
        if moments:
            out[client] = pd.Timestamp(min(moments)).tz_convert("UTC")
    return out


def history(
    events: pd.DataFrame,
    registered: dict[str, pd.Timestamp],
    until: datetime,
    rules: Rules = RULES,
) -> pd.DataFrame:
    """
    Переходы стадий клиентов registered до момента until: одна строка на
    переход, первая — New в момент регистрации. Переход с started_at s
    посчитан только по событиям строго раньше s.

    Клиенты — ключи registered, а не те, у кого в events есть события:
    иначе клиент, все события которого позже until, пропал бы целиком, и
    его прошлое зависело бы от будущего. events — события этих клиентов
    (блок выгрузки); у клиента без событий стадии идут только от
    регистрации.
    """
    until = pd.Timestamp(until).tz_convert("UTC")
    catalog = families()
    action = is_client_action(events)
    frame = events.assign(
        _action=action,
        _visit=is_visit(events),
        _target=is_target_action(events, action),
    )
    groups = dict(tuple(frame.groupby("client_id", sort=False)))
    rows: list[tuple] = []
    for client in sorted(registered):
        moment = registered[client]
        if moment >= until:
            continue
        rows.extend(_client(client, groups.get(client, frame.iloc[0:0]), moment, until, rules, catalog))
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    # Тип времени — всегда дата UTC, и у пустой таблицы тоже: иначе склейка
    # блоков превращала бы его в object.
    out["started_at"] = pd.to_datetime(out["started_at"], utc=True)
    return out


def stage_at(transitions: pd.DataFrame, moment: datetime) -> pd.DataFrame:
    """
    Стадия каждого клиента на момент moment: последний переход с
    started_at ≤ moment. Клиента, не зарегистрированного к moment, нет.
    """
    moment = pd.Timestamp(moment).tz_convert("UTC")
    known = transitions[transitions["started_at"] <= moment]
    last = known.sort_values(["client_id", "started_at"], kind="stable").groupby("client_id").tail(1)
    return last.set_index("client_id")[["stage", "previous_stage", "reason", "started_at"]].rename(
        columns={"started_at": "stage_since"}
    )


# Время внутри автомата — целые наносекунды UTC: так цикл по дням
# клиента в десятки раз быстрее, чем на объектах Timestamp.
NS_DAY = 86_400 * 10**9
NS_OFFSET = int(LOCAL_OFFSET.total_seconds()) * 10**9
EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def _day(ns: int) -> int:
    """
    Номер местного дня от 1970-01-01.
    """
    return (ns + NS_OFFSET) // NS_DAY


def _end_of_day(day: int) -> int:
    """
    Местная полночь после дня day.
    """
    return (day + 1) * NS_DAY - NS_OFFSET


def _date(day: int) -> date:
    return date.fromordinal(EPOCH_ORDINAL + day)


def _monday(day: int) -> int:
    """
    Понедельник недели ISO, в которой лежит день day.
    """
    return day - _date(day).weekday()


def _month(ns: int) -> tuple[int, int]:
    local = _date(_day(ns))
    return local.year, local.month


def _next_month(month: tuple[int, int]) -> tuple[int, int]:
    year, number = month
    return (year + 1, 1) if number == 12 else (year, number + 1)


def _client(
    client: str,
    events: pd.DataFrame,
    registered: pd.Timestamp,
    until: pd.Timestamp,
    rules: Rules,
    catalog: dict[str, str],
) -> list[tuple]:

    times = events["t"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64").tolist()
    action = events["_action"].tolist()
    visit = events["_visit"].tolist()
    target = events["_target"].tolist()
    kinds = events["type"].tolist()
    products = events["product_id"].tolist()
    contracts = events["contract_id"].tolist()

    reg = registered.value
    first = _day(reg)
    last = _day(until.value) - 1

    new_after = reg + rules.new_days * NS_DAY
    activated_by = reg + rules.activated_days * NS_DAY
    silence = rules.churn_days * NS_DAY

    held: dict[str, str] = {}

    def own(index: int) -> bool:
        """
        Продуктовое событие меняет набор договоров. Истина — закрыта
        премиальная карта.
        """
        contract = contracts[index]
        if not isinstance(contract, str):
            return False
        if kinds[index] in OPENED or kinds[index] in SIGHTED:
            if catalog.get(products[index]) not in (None, SERVICE):
                held[contract] = products[index]
        elif kinds[index] in CLOSED:
            return held.pop(contract, None) in PREMIUM
        return False

    # До регистрации в приложении: договоры и действия клиента банка.
    # Тишина Churn считается от последнего действия или от регистрации,
    # что позже.
    last_action = reg
    by_day: dict[int, list[int]] = {}
    for index, ns in enumerate(times):
        if ns < reg:
            own(index)
            if action[index]:
                last_action = max(last_action, ns)
        elif _day(ns) <= last:
            by_day.setdefault(_day(ns), []).append(index)

    stage, reason = NEW, "registered"
    out = [(client, registered, NEW, None, "registered")]

    activated: int | None = None
    acted_after_activation = False
    anchor: tuple[int, int] | None = None      # месяц отсчёта Growing
    growing_ever = False
    last_visit = -1
    visited: set[tuple[int, int]] = set()
    risk_since = reg

    weeks: list[int] = []
    week_visits = 0
    high_run = run_start = 0
    high_start = -1        # начало последней high-серии длиной ≥ high_run
    low_run = low_start = 0
    first_monday = _monday(first)

    for day in range(first, last + 1):

        moment = _end_of_day(day)
        today = _date(day)
        premium_closed = False

        for index in by_day.get(day, ()):
            ns = times[index]
            premium_closed |= own(index)
            if action[index]:
                last_action = max(last_action, ns)
                if activated is not None and ns > activated:
                    acted_after_activation = True
            if target[index] and activated is None:
                activated = ns
                anchor = _month(ns)
            if visit[index]:
                last_visit = max(last_visit, ns)
                visited.add((today.year, today.month))
                week_visits += 1

        # --- завершённая неделя ISO: серии H7 ---

        declined = calm = False
        if today.weekday() == 6:
            weeks.append(week_visits)
            week_visits = 0
            number = len(weeks) - 1
            high = len(weeks) >= rules.high_window and sum(weeks[-rules.high_window:]) >= rules.high_visits * rules.high_window
            low = len(weeks) >= rules.low_window and sum(weeks[-rules.low_window:]) <= rules.low_visits * rules.low_window
            high_run = high_run + 1 if high else 0
            if high_run == 1:
                run_start = number
            if high_run == rules.high_run:
                high_start = run_start
            low_run = low_run + 1 if low else 0
            if low_run == 1:
                low_start = number
            declined = low_run == rules.low_run and high_start >= 0 and low_start > high_start
            calm = not low

        # --- завершённый месяц: был ли визит ---

        month_without_visit = (today + timedelta(days=1)).day == 1 and (today.year, today.month) not in visited

        premium = any(product in PREMIUM for product in held.values())
        debit = any(catalog.get(product) == DEBIT_CARD for product in held.values())
        core_ok = high_run >= rules.high_run and len(held) >= rules.core_products and debit

        def fresh(since: int) -> str:
            """
            Активная стадия заново, по фактам с момента since: Growing —
            визит после since у клиента, который уже был Growing; CORE —
            high-серия, начавшаяся не раньше недели since.
            """
            nonlocal anchor
            if activated is None:
                return NEW
            if not (growing_ever and last_visit >= since):
                # Месяц отсчёта Growing — месяц выхода, но не раньше
                # месяца активации.
                anchor = max(anchor or _month(since), _month(since))
                return ACTIVATED
            if core_ok and run_start >= (_monday(_day(since)) - first_monday) // 7:
                return LOYAL if premium else CORE
            return GROWING

        new_stage, new_reason = stage, reason

        if moment - last_action >= silence:
            if stage != CHURN:
                new_stage, new_reason = CHURN, "no_action_60d"

        elif stage == CHURN:
            new_stage, new_reason = fresh(last_action), "reactivated"

        elif stage == AT_RISK:
            if reason == "new_no_target_action_7d":
                back = activated is not None
            elif reason == "activated_no_action_14d":
                back = last_action >= risk_since
            elif reason == "growing_not_in_mau":
                back = last_visit >= risk_since
            elif reason == "core_wau_decline":
                back = calm
            else:
                back = premium
            if back:
                new_stage, new_reason = fresh(risk_since), "recovered"

        elif stage == NEW and activated is None and moment >= new_after:
            new_stage, new_reason = AT_RISK, "new_no_target_action_7d"

        elif stage == ACTIVATED and not acted_after_activation and activated < activated_by <= moment:
            new_stage, new_reason = AT_RISK, "activated_no_action_14d"

        elif stage == LOYAL and premium_closed and not premium:
            new_stage, new_reason = AT_RISK, "loyal_premium_closed"

        elif stage in (CORE, LOYAL) and declined:
            new_stage, new_reason = AT_RISK, "core_wau_decline"

        elif stage == GROWING and month_without_visit:
            new_stage, new_reason = AT_RISK, "growing_not_in_mau"

        else:
            # Продвижение по цепочке, сколько позволяют факты дня.
            level = stage
            if level == NEW and activated is not None:
                level = ACTIVATED
            if level == ACTIVATED and anchor is not None and _next_month(anchor) in visited:
                level = GROWING
            if level == GROWING and core_ok:
                level = CORE
            if level == CORE and premium:
                level = LOYAL
            if level != stage:
                new_stage, new_reason = level, "promoted"

        if new_stage == GROWING:
            growing_ever = True
        if new_stage == AT_RISK and stage != AT_RISK:
            risk_since = moment

        if new_stage != stage:
            out.append((client, pd.Timestamp(moment, tz="UTC"), new_stage, stage, new_reason))
            stage, reason = new_stage, new_reason

    return out
