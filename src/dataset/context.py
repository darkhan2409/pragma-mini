from __future__ import annotations

from .settings import POLICY_ALL, POLICY_RECENT, ContextPolicy


# ============================================================
# ИДЕЯ
# ============================================================
#
# Какие события длинной истории попадут в один пример.
#
#   all     вся история; объявленные max_events и max_tokens —
#           границы, за которыми сборка останавливается;
#   recent  самый длинный свежий хвост, в котором одновременно не
#           больше max_events событий и не больше max_tokens
#           токенов событий. Укладывается в оба — берётся целиком.
#
# Правила отбора:
#
#   - пределов два: число событий и сумма их токенов. Память шага
#     растёт с токенами, а токенов на событие у клиентов от 5 до
#     13, поэтому одного предела по событиям мало. Токены анкеты
#     не считаются: анкета не режется;
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


def border(sizes: list[int], policy: ContextPolicy) -> int:
    """
    Сколько самых старых событий остаётся за границей примера; sizes —
    токены событий по порядку истории. Оставшиеся — непрерывный хвост
    sizes[border:], поэтому списков номеров для него не нужно.
    """

    _check_limits(sizes, policy)

    if policy.policy == POLICY_ALL:

        if policy.max_events is not None and len(sizes) > policy.max_events:
            raise ContextError(
                f"история из {len(sizes)} событий при политике all и пределе "
                f"{policy.max_events}: выберите политику recent либо снимите предел"
            )

        tokens = sum(sizes)

        if policy.max_tokens is not None and tokens > policy.max_tokens:
            raise ContextError(
                f"история из {tokens} токенов при политике all и пределе "
                f"{policy.max_tokens}: выберите политику recent либо снимите предел"
            )

        return 0

    if policy.policy == POLICY_RECENT:
        return len(sizes) - _tail(sizes, policy)

    raise ContextError(f"неизвестная политика контекста {policy.policy!r}")


def _tail(sizes: list[int], policy: ContextPolicy) -> int:
    """
    Сколько последних событий помещается в оба предела: хвост растёт
    от самого свежего события, пока следующее не нарушило бы любой.
    """

    keep = 0
    tokens = 0

    for size in reversed(sizes):

        if policy.max_events is not None and keep == policy.max_events:
            break

        if policy.max_tokens is not None and tokens + size > policy.max_tokens:
            break

        keep += 1
        tokens += size

    return keep


def _check_limits(sizes: list[int], policy: ContextPolicy) -> None:
    """
    Одна запись не бывает длиннее объявленного предела.

    Это ошибка настройки, а не повод обрезать значение: предел
    существует, чтобы о таком событии узнали, а не чтобы оно
    молча потеряло половину полей. Номер события — его место в
    sizes, то есть в истории.
    """

    if not sizes or max(sizes) <= policy.max_event_tokens:
        return

    for number, size in enumerate(sizes):
        if size > policy.max_event_tokens:
            raise ContextError(
                f"событие {number} занимает {size} токенов при пределе "
                f"{policy.max_event_tokens}: ничего не обрезается, поднимите предел осознанно"
            )


__all__ = [
    "ContextError",
    "border",
]
