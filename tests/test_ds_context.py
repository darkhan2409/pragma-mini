from __future__ import annotations

from dataclasses import replace

import pytest

from src.dataset.context import (
    EXCLUDED_EVENTS_BUDGET,
    EXCLUDED_TOKENS_BUDGET,
    REASON_MILESTONE,
    REASON_RECENT,
    ContextError,
    EventStub,
    select,
)
from src.dataset.settings import POLICY_ALL, POLICY_RECENT_PLUS_MILESTONES, ContextPolicy


# ============================================================
# ИДЕЯ
# ============================================================
#
# Отбор контекста это чистая функция, и проверяется она без
# данных вовсе: на списке заглушек видно то, что на живой
# истории пришлось бы выуживать.
# ============================================================


def _events(count: int, tokens: int = 10, milestones: dict[int, str] | None = None,
            eligible_from: int | None = None) -> list[EventStub]:

    milestones = milestones or {}

    return [
        EventStub(
            index=number,
            event_type=milestones.get(number, "purchase"),
            n_tokens=tokens,
            eligible=eligible_from is not None and number >= eligible_from,
        )
        for number in range(count)
    ]


def _policy(**kwargs) -> ContextPolicy:
    return ContextPolicy(policy=POLICY_RECENT_PLUS_MILESTONES, **kwargs)


# ============================================================
# БЮДЖЕТЫ
# ============================================================


def test_both_budgets_hold_at_once():
    """
    Предел событий и предел токенов действуют одновременно, и ни
    один не выражается через другой.
    """

    events = _events(20)

    by_events = select(events, _policy(max_events=6, max_tokens=10_000, milestone_share=0.0))

    assert by_events.n_kept == 6
    assert by_events.budget_binding == EXCLUDED_EVENTS_BUDGET

    by_tokens = select(events, _policy(max_events=100, max_tokens=45, milestone_share=0.0))

    assert by_tokens.kept_tokens <= 45
    assert by_tokens.n_kept == 4
    assert by_tokens.budget_binding == EXCLUDED_TOKENS_BUDGET


def test_selection_keeps_whole_events_in_chronological_order():
    """
    Отбор меняет состав, а не ход времени: отобранное лежит по
    возрастанию, и ни одно событие не разрезано.
    """

    events = _events(20, tokens=7)

    selection = select(events, _policy(max_events=5, max_tokens=10_000, milestone_share=0.0))

    assert selection.kept == sorted(selection.kept)
    assert selection.kept_tokens == 5 * 7
    assert selection.n_kept + selection.n_excluded == 20


def test_recent_events_are_a_tail_without_holes():

    events = _events(12)

    selection = select(events, _policy(max_events=4, max_tokens=10_000, milestone_share=0.0))

    assert selection.kept == [8, 9, 10, 11]
    assert set(selection.reasons) == {REASON_RECENT}


# ============================================================
# ВЕХИ
# ============================================================


def test_milestones_are_taken_by_priority_then_recency():
    """
    Порядок списка вех это и есть приоритет, а внутри одного
    типа берётся более свежее.
    """

    events = _events(
        12,
        milestones={
            0: "product_opened",
            1: "delinquency_registered",
            2: "product_opened",
            3: "card_blocked",
        },
    )

    selection = select(events, _policy(max_events=6, max_tokens=10_000, milestone_share=0.5))

    kept = dict(zip(selection.kept, selection.reasons))

    milestones = sorted(key for key, reason in kept.items() if reason == REASON_MILESTONE)

    # Резерв на три события: просрочка (приоритет выше всех),
    # блокировка карты, затем более свежее открытие продукта.
    assert milestones == [1, 2, 3]
    assert 0 not in kept


def test_selection_is_the_same_for_the_same_input():
    """
    Ни хэшей, ни случайности: два одинаковых входа дают один и
    тот же отбор.
    """

    events = _events(30, milestones={i: "product_opened" for i in range(0, 10)})

    policy = _policy(max_events=9, max_tokens=10_000, milestone_share=0.34)

    first = select(events, policy)
    second = select(list(reversed(list(reversed(events)))), policy)

    assert first.kept == second.kept
    assert first.reasons == second.reasons


def test_unused_reserve_returns_only_when_allowed():

    events = _events(12, milestones={0: "product_opened"})

    policy = _policy(max_events=6, max_tokens=10_000, milestone_share=0.5)

    kept = select(events, replace(policy, return_unused_budget=False))
    extended = select(events, replace(policy, return_unused_budget=True))

    # Веха всего одна, а резерв на три события: два места
    # возвращаются недавним только при разрешении.
    assert kept.n_kept == 4
    assert extended.n_kept == 6


# ============================================================
# ПОТЕРИ
# ============================================================


def test_lost_targets_are_counted():
    """
    Исключённое событие периода целей считается отдельно: без
    этого потеря контекста выглядела бы как более простые данные.
    """

    events = _events(10, eligible_from=2)

    selection = select(events, _policy(max_events=3, max_tokens=10_000, milestone_share=0.0))

    assert selection.excluded_eligible == 5
    assert sum(selection.excluded_by_type.values()) == selection.n_excluded


def test_everything_fits_keeps_everything():

    events = _events(5)

    selection = select(events, _policy(max_events=50, max_tokens=10_000))

    assert selection.kept == [0, 1, 2, 3, 4]
    assert not selection.truncated


# ============================================================
# ОШИБКИ
# ============================================================


def test_event_larger_than_the_whole_budget_is_an_error():
    """
    Событие, которое не помещается в бюджет примера, исчезало бы
    из каждого примера каждого клиента.
    """

    events = _events(5, tokens=100)

    with pytest.raises(ContextError, match="не попадёт в пример"):
        select(events, _policy(max_tokens=60, milestone_share=0.5))


def test_event_larger_than_its_share_but_fitting_the_budget_is_allowed():
    """
    Доля недавних поводом для отказа не является: при возврате
    резерва такое событие законно помещается.
    """

    events = _events(1, tokens=80)

    selection = select(events, _policy(max_events=4, max_tokens=100, milestone_share=0.5))

    assert selection.kept == [0]


def test_oversized_single_event_is_refused_even_with_all_history():

    events = _events(3, tokens=50)

    with pytest.raises(ContextError, match="ничего не обрезается"):
        select(events, ContextPolicy(policy=POLICY_ALL, max_event_tokens=40))
