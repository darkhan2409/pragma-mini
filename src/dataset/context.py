from __future__ import annotations

from dataclasses import dataclass, field

from .settings import POLICY_ALL, POLICY_RECENT_PLUS_MILESTONES, ContextPolicy


# ============================================================
# ИДЕЯ
# ============================================================
#
# Какие события длинной истории попадут в один пример.
#
# Правила отбора:
#
#   - отбираются только ЦЕЛЫЕ события. Текстовое значение и его
#     границы не режутся никогда: половина названия магазина это
#     не «меньше контекста», это другое значение;
#   - бюджеты по событиям и по токенам действуют ОДНОВРЕМЕННО, и
#     ни один из них не выражается через другой;
#   - после отбора восстанавливается исходный хронологический
#     порядок: отбор меняет состав, а не ход времени;
#   - каждое исключённое событие получает причину, и потерянные
#     возможные цели считаются отдельно. Молчаливая потеря
#     контекста выглядит как более простые данные.
#
# Здесь нет ни хэшей, ни случайности, ни обращения к данным:
# результат зависит только от упорядоченного входа и решения
# человека в конфигурации.
# ============================================================


# Почему событие попало в пример.
REASON_ALL = "all"
REASON_RECENT = "recent"
REASON_MILESTONE = "milestone"
REASON_RECENT_EXTENDED = "recent_extended"

# Почему событие в пример не попало.
EXCLUDED_EVENTS_BUDGET = "events_budget"
EXCLUDED_TOKENS_BUDGET = "tokens_budget"
EXCLUDED_MILESTONE_EVENTS_BUDGET = "milestone_events_budget"
EXCLUDED_MILESTONE_TOKENS_BUDGET = "milestone_tokens_budget"


class ContextError(ValueError):
    """
    Историю нельзя уместить в объявленный бюджет.
    """


@dataclass(frozen=True)
class EventStub:
    """
    Всё, что нужно знать об одном событии, чтобы решить его
    судьбу. Ни значений, ни токенов: отбор смотрит на размер,
    время и тип.
    """

    index: int
    event_type: str | None
    n_tokens: int
    eligible: bool


@dataclass
class Selection:
    """
    Что вошло в пример и что осталось за его границей.
    """

    kept: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    excluded: list[int] = field(default_factory=list)
    excluded_reasons: list[str] = field(default_factory=list)

    kept_tokens: int = 0
    excluded_tokens: int = 0
    excluded_eligible: int = 0
    excluded_milestones: int = 0
    excluded_by_type: dict[str, int] = field(default_factory=dict)

    truncated: bool = False
    budget_binding: str | None = None

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)

    def as_dict(self) -> dict:
        return {
            "kept_events": self.n_kept,
            "kept_tokens": self.kept_tokens,
            "excluded_events": self.n_excluded,
            "excluded_tokens": self.excluded_tokens,
            "excluded_eligible": self.excluded_eligible,
            "excluded_milestones": self.excluded_milestones,
            "excluded_by_type": dict(sorted(self.excluded_by_type.items())),
            "truncated": self.truncated,
            "budget_binding": self.budget_binding,
        }


def select(events: list[EventStub], policy: ContextPolicy) -> Selection:
    """
    Отбор по объявленной политике.
    """

    if policy.policy == POLICY_ALL:
        return select_all(events, policy)

    if policy.policy == POLICY_RECENT_PLUS_MILESTONES:
        return select_recent_plus_milestones(events, policy)

    raise ContextError(f"неизвестная политика контекста {policy.policy!r}")


def select_all(events: list[EventStub], policy: ContextPolicy) -> Selection:
    """
    Вся видимая история.

    Бюджеты примера здесь не применяются, но объявленные лимиты
    всё же проверяются: если они заданы и нарушены, молчать об
    этом нельзя — политика all выбрана осознанно, а не ради
    обхода предела.
    """

    _check_limits(events, policy)

    total = sum(item.n_tokens for item in events)

    _check_declared_budgets(events, policy, total)

    return Selection(
        kept=[item.index for item in events],
        reasons=[REASON_ALL] * len(events),
        kept_tokens=total,
    )


def select_recent_plus_milestones(events: list[EventStub], policy: ContextPolicy) -> Selection:
    """
    Последние события плюс выделенный бюджет важных старых.

    Ход отбора:

      1. если история помещается целиком, берётся целиком;
      2. недавние: с конца назад, пока помещаются в свою долю
         обоих бюджетов. Первое неуместившееся останавливает
         проход, поэтому недавние это непрерывный хвост, а не
         решето с дырами;
      3. вехи: среди более старых событий по приоритету типа, а
         внутри типа от новых к старым. Здесь пропуск разрешён:
         не поместившаяся веха не должна закрывать дорогу
         следующей, более дешёвой;
      4. неизрасходованный резерв возвращается недавним, если
         это разрешено.
    """

    _check_limits(events, policy)

    max_events = policy.max_events if policy.max_events is not None else len(events)
    max_tokens = policy.max_tokens if policy.max_tokens is not None else _total(events)

    milestone_events = int(max_events * policy.milestone_share)
    milestone_tokens = int(max_tokens * policy.milestone_share)

    recent_events = max_events - milestone_events
    recent_tokens = max_tokens - milestone_tokens

    # --- шаг 1: всё помещается ---
    #
    # Проверяется до предела недавних: история, влезающая целиком,
    # берётся целиком, и делить её на доли незачем.

    if len(events) <= max_events and _total(events) <= max_tokens:
        return select_all(events, ContextPolicy(policy=POLICY_ALL,
                                                max_event_tokens=policy.max_event_tokens,
                                                max_profile_tokens=policy.max_profile_tokens))

    _check_fits_budget(events, max_tokens)

    reason_of: dict[int, str] = {}
    exclusion_of: dict[int, str] = {}

    # --- шаг 2: недавние ---

    kept_tokens = 0
    first_recent = len(events)
    recent_stop: str | None = None

    for position in range(len(events) - 1, -1, -1):

        item = events[position]

        if len(reason_of) + 1 > recent_events:
            recent_stop = EXCLUDED_EVENTS_BUDGET
            break

        if kept_tokens + item.n_tokens > recent_tokens:
            recent_stop = EXCLUDED_TOKENS_BUDGET
            break

        reason_of[item.index] = REASON_RECENT
        kept_tokens += item.n_tokens
        first_recent = position

    # --- шаг 3: вехи среди более старых ---

    priority = {name: number for number, name in enumerate(policy.milestone_event_types)}

    older = events[:first_recent]

    candidates = sorted(
        (item for item in older if item.event_type in priority),
        key=lambda item: (priority[item.event_type], -item.index),
    )

    milestone_kept = 0
    milestone_used = 0

    for item in candidates:

        if milestone_kept + 1 > milestone_events:
            exclusion_of[item.index] = EXCLUDED_MILESTONE_EVENTS_BUDGET
            continue

        if milestone_used + item.n_tokens > milestone_tokens:
            exclusion_of[item.index] = EXCLUDED_MILESTONE_TOKENS_BUDGET
            continue

        reason_of[item.index] = REASON_MILESTONE
        milestone_kept += 1
        milestone_used += item.n_tokens

    kept_tokens += milestone_used

    # --- шаг 4: возврат неизрасходованного резерва ---

    extension_stop: str | None = None

    if policy.return_unused_budget:

        for position in range(first_recent - 1, -1, -1):

            item = events[position]

            if item.index in reason_of:
                # Веха уже взята: её размер учтён, и она не
                # обрывает проход.
                continue

            if len(reason_of) + 1 > max_events:
                extension_stop = EXCLUDED_EVENTS_BUDGET
                break

            if kept_tokens + item.n_tokens > max_tokens:
                extension_stop = EXCLUDED_TOKENS_BUDGET
                break

            reason_of[item.index] = REASON_RECENT_EXTENDED
            kept_tokens += item.n_tokens

    # --- итог ---

    selection = Selection(truncated=True)

    fallback = extension_stop or recent_stop or EXCLUDED_EVENTS_BUDGET

    selection.budget_binding = fallback

    for item in events:

        reason = reason_of.get(item.index)

        if reason is not None:
            selection.kept.append(item.index)
            selection.reasons.append(reason)
            continue

        selection.excluded.append(item.index)
        selection.excluded_reasons.append(exclusion_of.get(item.index, fallback))
        selection.excluded_tokens += item.n_tokens

        if item.eligible:
            selection.excluded_eligible += 1

        if item.event_type in priority:
            selection.excluded_milestones += 1

        name = item.event_type or "—"
        selection.excluded_by_type[name] = selection.excluded_by_type.get(name, 0) + 1

    selection.kept_tokens = kept_tokens

    return selection


def _total(events: list[EventStub]) -> int:
    return sum(item.n_tokens for item in events)


def _check_limits(events: list[EventStub], policy: ContextPolicy) -> None:
    """
    Одна запись не бывает длиннее объявленного предела.

    Это ошибка настройки, а не повод обрезать значение: предел
    существует, чтобы о таком событии узнали, а не чтобы оно
    молча потеряло половину полей.
    """

    for item in events:
        if item.n_tokens > policy.max_event_tokens:
            raise ContextError(
                f"событие {item.index} занимает {item.n_tokens} токенов при пределе "
                f"{policy.max_event_tokens}: ничего не обрезается, поднимите предел осознанно"
            )


def _check_declared_budgets(events: list[EventStub], policy: ContextPolicy, total: int) -> None:

    if policy.max_events is not None and len(events) > policy.max_events:
        raise ContextError(
            f"история из {len(events)} событий при политике all и пределе {policy.max_events}: "
            "выберите политику отбора либо снимите предел"
        )

    if policy.max_tokens is not None and total > policy.max_tokens:
        raise ContextError(
            f"история из {total} токенов при политике all и пределе {policy.max_tokens}: "
            "выберите политику отбора либо снимите предел"
        )


def _check_fits_budget(events: list[EventStub], max_tokens: int) -> None:
    """
    Событие, которое не помещается в ПОЛНЫЙ бюджет примера, не
    попадёт в него никогда.

    Считается именно полный бюджет, а не доля недавних: событие
    крупнее своей доли законно помещается при возврате
    неизрасходованного резерва, и отказывать ему рано.

    А вот событие крупнее всего бюджета исчезало бы из каждого
    примера каждого клиента, и заметить это можно было бы только
    по счётчику потерь.
    """

    for item in events:
        if item.n_tokens > max_tokens:
            raise ContextError(
                f"событие {item.index} занимает {item.n_tokens} токенов при бюджете примера "
                f"{max_tokens}: оно не попадёт в пример ни при каком отборе. "
                "Поднимите max_tokens"
            )


__all__ = [
    "EXCLUDED_EVENTS_BUDGET",
    "EXCLUDED_MILESTONE_EVENTS_BUDGET",
    "EXCLUDED_MILESTONE_TOKENS_BUDGET",
    "EXCLUDED_TOKENS_BUDGET",
    "REASON_ALL",
    "REASON_MILESTONE",
    "REASON_RECENT",
    "REASON_RECENT_EXTENDED",
    "ContextError",
    "EventStub",
    "Selection",
    "select",
    "select_all",
    "select_recent_plus_milestones",
]
