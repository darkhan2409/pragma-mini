from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..history import SourceState


# ============================================================
# ИДЕЯ
# ============================================================
#
# Активность по месяцам различает четыре случая, и разница между
# ними принципиальна:
#
#   has_client_action        клиент что-то сделал сам;
#   records_without_action   записи есть, но все от банка,
#                            системы или внешнего источника;
#   no_records               записей нет, а покрытие было —
#                            это доказанное молчание;
#   insufficient_coverage    записей нет, но и источников не
#                            было: сказать нечего.
#
# Пустой месяц САМ ПО СЕБЕ не означает неактивность. Вывод
# «действий не было» требует, чтобы применимые источники в этом
# месяце наблюдались; одного профиля для этого мало. Месяц, в
# котором у источника был день сбоя, покрытым не считается:
# молчание витрины это не молчание человека.
#
# Считается только до cutoff. Будущая дата возвращения и итоговая
# длительность паузы не рассчитываются: их знание лежит за
# границей среза.
# ============================================================


HAS_CLIENT_ACTION = "has_client_action"
RECORDS_WITHOUT_ACTION = "records_without_client_action"
NO_RECORDS = "no_records"
INSUFFICIENT_COVERAGE = "insufficient_coverage"

MONTH_STATES: tuple[str, ...] = (
    HAS_CLIENT_ACTION,
    RECORDS_WITHOUT_ACTION,
    NO_RECORDS,
    INSUFFICIENT_COVERAGE,
)

from ..projection import CLIENT_ACTION_EVENT_TYPES

# Источники, которые вообще способны записать действие клиента.
# Доказать молчание можно только по ним: витрина договоров
# молчит и о клиенте, который каждый день платит картой.
ACTION_SOURCES: tuple[str, ...] = ("transactions", "app_operations", "applications")

# Причины, по которым источник к этому клиенту НЕПРИМЕНИМ вовсе.
# Требовать его для доказательства молчания нельзя: у клиента без
# приложения не бывает месяца, в котором приложение записало бы
# действие, и все его месяцы оказывались бы «неизвестными».
NOT_APPLICABLE_REASONS: frozenset[str] = frozenset({"client_not_onboarded", "no_consent"})


@dataclass(frozen=True)
class ActivityMonth:
    month: str
    state: str
    events: int
    client_actions: int
    partial: bool

    def as_dict(self) -> dict:
        return {
            "month": self.month,
            "state": self.state,
            "events": self.events,
            "client_actions": self.client_actions,
            "partial": self.partial,
        }


def _month_key(moment: datetime) -> str:
    return f"{moment.year:04d}-{moment.month:02d}"


def _months_before(start: datetime, cutoff: datetime) -> list[tuple[int, int]]:
    """
    Месяцы, у которых есть хоть один прожитый до cutoff день.

    Граница исключительная и здесь: месяц, начинающийся ровно в
    cutoff, не прожит ни на день и месяцем наблюдения не
    считается.
    """

    out: list[tuple[int, int]] = []

    year, month = start.year, start.month

    while datetime(year, month, 1) < cutoff:
        out.append((year, month))
        month += 1
        if month > 12:
            year, month = year + 1, 1

    return out


def _applicable(item: SourceState | None) -> bool:
    """
    Источник вообще применим к этому клиенту.

    «Клиент не подключён» и «нет согласия» это свойство клиента, а
    не месяца: требовать такой источник значит объявить всю его
    историю ненаблюдаемой.
    """

    if item is None:
        return False

    if item.first_seen is not None:
        return True

    return item.reason not in NOT_APPLICABLE_REASONS


def _next_month(moment: datetime) -> datetime:
    return (
        datetime(moment.year + 1, 1, 1)
        if moment.month == 12
        else datetime(moment.year, moment.month + 1, 1)
    )


def _covered(coverage: list[SourceState], base_sources: tuple[str, ...], year: int, month: int) -> bool:
    """
    Месяц наблюдался достаточно, чтобы говорить о молчании.

    Считается ПО ДАТИРОВАННЫМ полям этого месяца, а не по
    состоянию источника на cutoff: закрывшийся в прошлом году
    источник ничего не отнимает у позапрошлого, а работающий
    сегодня ничего не добавляет к нему.

    Требуются все ПРИМЕНИМЫЕ источники, записывающие действия
    клиента. Если применимых нет вовсе, доказать молчание нечем.

    День сбоя внутри месяца снимает покрытие целиком. Иначе
    молчание клиента и молчание витрины стали бы неотличимы: в
    месяце со сбоем отсутствие событий доказывает не бездействие
    человека, а потерю наблюдения.
    """

    start = datetime(year, month, 1)

    prefix = f"{year:04d}-{month:02d}"

    states = {item.source: item for item in coverage}

    wanted = [
        name
        for name in ACTION_SOURCES
        if name in base_sources and _applicable(states.get(name))
    ]

    if not wanted:
        return False

    for name in wanted:

        item = states[name]

        if item.first_available_at is None or item.first_available_at > start:
            return False

        if item.first_seen is None or item.first_seen > start:
            return False

        # Конец покрытия это последний ПОКРЫТЫЙ день включительно:
        # закрытие отношений регистрируется в конце этого дня, после
        # его событий. Месяц покрыт, только если покрыт его последний
        # день; источник, кончившийся пятнадцатого, молчания второй
        # половины месяца не доказывает.
        last_day = _next_month(start) - timedelta(days=1)

        if item.last_available_at is not None and item.last_available_at < last_day:
            return False

        if any(day.startswith(prefix) for day in item.outage_days):
            return False

    return True


def activity_months(
    rows: list[dict],
    coverage: list[SourceState],
    observed_start: datetime | None,
    cutoff: datetime,
    base_sources: tuple[str, ...],
) -> list[ActivityMonth]:
    """
    Месяцы от наблюдаемого начала до cutoff.

    Неполным месяц называется только тогда, когда граница
    наблюдения действительно прошла внутри него: первый — если
    наблюдение началось не первого числа, последний — если cutoff
    не пришёлся ровно на 00:00 первого числа. Месяц, совпавший с
    границей ровно, полон, и объявлять его неполным значит терять
    настоящий месяц наблюдения.
    """

    if observed_start is None:
        return []

    events: dict[str, int] = {}
    actions: dict[str, int] = {}

    for row in rows:
        key = _month_key(row["event_time"])
        events[key] = events.get(key, 0) + 1
        if row["event_type"] in CLIENT_ACTION_EVENT_TYPES:
            actions[key] = actions.get(key, 0) + 1

    months = _months_before(observed_start, cutoff)

    out: list[ActivityMonth] = []

    for index, (year, month) in enumerate(months):

        key = f"{year:04d}-{month:02d}"

        count = events.get(key, 0)
        client = actions.get(key, 0)

        covered = _covered(coverage, base_sources, year, month)

        if client:
            state = HAS_CLIENT_ACTION
        elif count and covered:
            state = RECORDS_WITHOUT_ACTION
        elif count:
            # Записи банка есть, а покрытия действий нет: сказать
            # «клиент молчал» по ним нельзя. Число записей
            # остаётся, состояние честное.
            state = INSUFFICIENT_COVERAGE
        elif covered:
            state = NO_RECORDS
        else:
            state = INSUFFICIENT_COVERAGE

        first_day = datetime(year, month, 1)

        # Неполон месяц только тогда, когда граница наблюдения
        # действительно прошла внутри него.
        partial = (index == 0 and observed_start > first_day) or (
            index == len(months) - 1 and cutoff < _next_month(first_day)
        )

        out.append(ActivityMonth(key, state, count, client, partial))

    return out


def activity_summary(months: list[ActivityMonth]) -> dict:
    """
    Доли считаются на явном знаменателе: месяцы без покрытия в
    него не входят и показываются отдельно.
    """

    known = [item for item in months if item.state != INSUFFICIENT_COVERAGE]

    counts = {state: sum(1 for item in months if item.state == state) for state in MONTH_STATES}

    silent = sum(1 for item in known if item.state != HAS_CLIENT_ACTION)

    return {
        "months": len(months),
        "known_months": len(known),
        "unknown_months": counts[INSUFFICIENT_COVERAGE],
        "by_state": counts,
        "share_without_client_action": (silent / len(known)) if known else None,
        "denominator": "месяцы с достаточным покрытием; месяцы без покрытия считаются отдельно",
        "current_pause_months": _current_pause(months),
    }


def _current_pause(months: list[ActivityMonth]) -> int | None:
    """
    Доказанная пауза к cutoff: сколько месяцев подряд не было
    действий клиента. Неизвестный месяц паузу прерывает, потому
    что доказать её он не может.
    """

    pause = 0

    for item in reversed(months):

        if item.state == INSUFFICIENT_COVERAGE:
            return pause if pause else None

        if item.state == HAS_CLIENT_ACTION:
            return pause

        pause += 1

    return pause


__all__ = [
    "ACTION_SOURCES",
    "HAS_CLIENT_ACTION",
    "INSUFFICIENT_COVERAGE",
    "MONTH_STATES",
    "NO_RECORDS",
    "RECORDS_WITHOUT_ACTION",
    "ActivityMonth",
    "activity_months",
    "activity_summary",
]
