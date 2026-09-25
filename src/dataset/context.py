from __future__ import annotations

from dataclasses import dataclass, field

from .settings import POLICY_ALL, POLICY_RECENT, ContextPolicy


# ============================================================
# ИДЕЯ
# ============================================================
#
# Какие события длинной истории попадут в один пример.
#
#   all     вся история; объявленный max_events — граница, за
#           которой сборка останавливается;
#   recent  последние max_events событий. Короче — берётся целиком.
#
# Правила отбора:
#
#   - предел считается по событиям, а не по токенам;
#   - отбираются только ЦЕЛЫЕ события. Текстовое значение и его
#     границы не режутся никогда: половина названия магазина это
#     не «меньше контекста», это другое значение;
#   - оставшиеся события — непрерывный хвост истории в исходном
#     хронологическом порядке. Старые события отдельно не
#     сохраняются: долгосрочные факты о клиенте лежат в Lifelong
#     анкеты;
#   - потерянные события считаются, и отдельно — те, что лежали в
#     периоде целей. Молчаливая потеря контекста выглядит как
#     более простые данные.
#
# Здесь нет ни хэшей, ни случайности, ни обращения к данным:
# результат зависит только от упорядоченного входа и решения
# человека в конфигурации.
# ============================================================


class ContextError(ValueError):
    """
    Историю нельзя уместить в объявленный предел.
    """


@dataclass(frozen=True)
class EventStub:
    """
    Всё, что нужно знать об одном событии, чтобы решить его
    судьбу: место, размер и лежит ли оно в периоде целей.
    """

    index: int
    n_tokens: int
    eligible: bool


@dataclass
class Selection:
    """
    Что вошло в пример и что осталось за его границей.
    """

    kept: list[int] = field(default_factory=list)
    excluded: list[int] = field(default_factory=list)

    kept_tokens: int = 0
    excluded_tokens: int = 0
    excluded_eligible: int = 0

    truncated: bool = False

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)


def select(events: list[EventStub], policy: ContextPolicy) -> Selection:
    """
    Отбор по объявленной политике.
    """

    _check_limits(events, policy)

    if policy.policy == POLICY_ALL:

        if policy.max_events is not None and len(events) > policy.max_events:
            raise ContextError(
                f"история из {len(events)} событий при политике all и пределе "
                f"{policy.max_events}: выберите политику recent либо снимите предел"
            )

        return _split(events, len(events))

    if policy.policy == POLICY_RECENT:
        return _split(events, min(len(events), policy.max_events))

    raise ContextError(f"неизвестная политика контекста {policy.policy!r}")


def _split(events: list[EventStub], keep: int) -> Selection:
    """
    Последние keep событий остаются, более старые — за границей.
    """

    border = len(events) - keep

    selection = Selection(
        kept=[item.index for item in events[border:]],
        excluded=[item.index for item in events[:border]],
        kept_tokens=sum(item.n_tokens for item in events[border:]),
        excluded_tokens=sum(item.n_tokens for item in events[:border]),
        excluded_eligible=sum(1 for item in events[:border] if item.eligible),
        truncated=border > 0,
    )

    return selection


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


__all__ = [
    "ContextError",
    "EventStub",
    "Selection",
    "select",
]
