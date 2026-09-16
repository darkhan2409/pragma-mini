from __future__ import annotations

from .. import params as params_module
from ..rng import current_seed, stable_hash


# ============================================================
# СООБЩЕСТВА
# ============================================================
#
# Единица симуляции это СООБЩЕСТВО, а не runtime-чанк.
#
# Связанные клиенты обязаны жить в одной очереди событий: иначе
# внутрибанковский перевод не смог бы повлиять на решения
# получателя. Состав сообщества определяется только порядковым
# номером клиента и размером сообщества из конфигурации, а
# распределение сообществ по воркерам на данные не влияет.
#
# client_id это непрозрачная строка, выведенная из seed и
# порядкового номера: соседство номеров не должно читаться
# из наблюдаемых данных.
# ============================================================


def community_size() -> int:
    return int(params_module.active().relationships.community_size)


def community_of(client_ordinal: int) -> int:
    """
    Номер сообщества клиента. Порядковый номер начинается с 1.
    """

    if client_ordinal < 1:
        raise ValueError("client_ordinal начинается с 1")

    return (client_ordinal - 1) // community_size()


def members(community_id: int, total_clients: int) -> tuple:
    """
    Порядковые номера клиентов сообщества, не выходящие за
    пределы популяции.
    """

    size = community_size()

    first = community_id * size + 1
    last = min(first + size, total_clients + 1)

    return tuple(range(first, last))


def community_count(total_clients: int) -> int:

    size = community_size()

    return (total_clients + size - 1) // size


def client_id(client_ordinal: int) -> str:
    """
    Непрозрачный идентификатор клиента.
    """

    return f"c{stable_hash('client', current_seed(), client_ordinal) % 10 ** 12:012d}"


def client_ordinals(total_clients: int) -> tuple:
    return tuple(range(1, total_clients + 1))


__all__ = [
    "client_id",
    "client_ordinals",
    "community_count",
    "community_of",
    "community_size",
    "members",
]
