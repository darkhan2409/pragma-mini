from __future__ import annotations

from dataclasses import replace
from typing import Any

from .rng import NS_NOISE, KeyedRandom, keyed_rng, second_of_day


# ============================================================
# ИДЕЯ
# ============================================================
#
# Наблюдательный шум это то, что теряется по дороге события
# в хранилище: не разобранный адрес точки, служебное значение
# GA4 вместо имени экрана, не доехавший статус операции.
#
# Шум применяется к СОБЫТИЮ, а не к строке таблицы, и результат
# один на все представления: и типизированная таблица, и единая
# лента строятся из одного и того же наблюдаемого события.
# Иначе одно и то же событие выглядело бы в них по-разному.
#
# RNG привязан к идентичности события (клиент, источник, день,
# секунда дня), а не к номеру строки: добавление события выше
# не меняет шум событий ниже.
# ============================================================


SOURCE_CODES = {
    "profile": 1,
    "transactions": 2,
    "product_events": 3,
    "communications": 4,
    "app_screens": 5,
    "app_operations": 6,
    "banners": 7,
}

# Источники, в которых наблюдение что-то теряет.
NOISY_SOURCES = frozenset({"transactions", "app_screens", "app_operations"})

# Адрес торговой точки не разобрался.
MISSING_CITY = 0.012

# GA4 отдаёт служебное значение вместо имени экрана.
GA4_NOT_SET = "(not set)"
NOT_SET_SHARE = 0.006

# Статус операции не доехал из мобильного бэкенда.
MISSING_STATUS = 0.004


def noise_rng(client_id: int, source: str, ts) -> KeyedRandom:

    return keyed_rng(
        NS_NOISE,
        client_id,
        SOURCE_CODES[source],
        ts.toordinal(),
        second_of_day(ts),
    )


def noise_overrides(source: str, event: Any, client_id: int) -> dict[str, Any]:
    """
    Какие поля события теряются при наблюдении.
    Пустой словарь: событие дошло без потерь.
    """

    if source not in NOISY_SOURCES:
        return {}

    draw = noise_rng(client_id, source, event.ts).random()

    if source == "transactions":

        if event.merchant_city is not None and draw < MISSING_CITY:
            return {"merchant_city": None}

    elif source == "app_screens":

        # Воронку не портим: без имени экрана стадия теряет смысл.
        if event.funnel_stage is None and draw < NOT_SET_SHARE:
            return {"firebase_screen": GA4_NOT_SET}

    elif source == "app_operations":

        if draw < MISSING_STATUS:
            return {"status": None}

    return {}


def apply_noise(source: str, event: Any, client_id: int) -> Any:
    """
    Событие, каким его видит хранилище.
    """

    changes = noise_overrides(source, event, client_id)

    return replace(event, **changes) if changes else event
