from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from .config import HISTORY_START, LABEL_END, SEED, SOURCE_AVAILABILITY, SOURCES
from .persona import draw_persona
from .rng import NS_COVERAGE


# ============================================================
# ИДЕЯ
# ============================================================
#
# Два независимых дефекта покрытия.
#
# 1. availability_start источника: до этой даты в хранилище
#    нет НИЧЕГО, даже если клиент активен.
#
# 2. client_first_seen_in_source: конкретный клиент появляется
#    в источнике позже. Приложение он установил через год,
#    согласие на маркетинг дал не сразу, а часть клиентов
#    не появляется в источнике никогда.
#
# Поток фильтруется по обоим правилам, а сами даты попадают
# в таблицу source_coverage: preprocessing обязан отличать
# "события не было" от "источник ещё не наблюдал клиента".
# ============================================================


# Доля клиентов, которые вообще пользуются приложением
# (отчёт: домен auth покрывает 84.9 процента когорты).
APP_ADOPTION_SHARE = 0.85

# Доля клиентов с действующим согласием на коммуникации.
CONSENT_SHARE = 0.92


def _rng(client_id: int, slot: int) -> np.random.Generator:
    """
    Отдельный поток на каждый вид покрытия: добавление одного
    не сдвигает розыгрыши другого.
    """

    return np.random.default_rng([SEED, NS_COVERAGE, client_id, slot])


@lru_cache(maxsize=131_072)
def app_adoption(client_id: int) -> datetime | None:
    """
    Дата установки приложения. None: клиент им не пользуется.
    """

    persona = draw_persona(client_id)

    rng = _rng(client_id, 1)

    share = APP_ADOPTION_SHARE * (0.75 + 0.45 * persona.digital_affinity)

    if rng.random() >= min(0.99, share):
        return None

    # Часть клиентов пришла в приложение задолго до окна наблюдения,
    # часть уже внутри него.
    horizon_days = (LABEL_END - HISTORY_START).days

    offset = int(rng.gamma(1.4, 260.0)) - 540

    adopted = HISTORY_START + timedelta(days=offset)

    earliest = persona.relationship_start

    if adopted < earliest:
        adopted = earliest

    if adopted > HISTORY_START + timedelta(days=horizon_days):
        return None

    return adopted.replace(hour=0, minute=0, second=0, microsecond=0)


@lru_cache(maxsize=131_072)
def consent_date(client_id: int) -> datetime | None:
    """
    Дата согласия на маркетинговые коммуникации.
    None: клиент в рассылки не попадает.
    """

    persona = draw_persona(client_id)

    rng = _rng(client_id, 2)

    if rng.random() >= CONSENT_SHARE:
        return None

    # Согласие обычно даётся при оформлении первого продукта.
    offset = int(rng.gamma(1.2, 120.0)) - 240

    granted = persona.relationship_start + timedelta(days=max(0, offset))

    if granted > LABEL_END:
        return None

    return granted.replace(hour=0, minute=0, second=0, microsecond=0)


# ============================================================
# FIRST SEEN
# ============================================================


@lru_cache(maxsize=1_048_576)
def first_seen(client_id: int, source: str) -> datetime | None:
    """
    Момент, начиная с которого клиент виден в источнике.

    None: клиент в этом источнике не появляется никогда.
    """

    if source not in SOURCE_AVAILABILITY:
        raise ValueError(f"unknown source: {source}")

    persona = draw_persona(client_id)

    availability = SOURCE_AVAILABILITY[source]

    if source in ("profile", "product_events"):
        client_start = persona.relationship_start

    elif source == "transactions":
        # Транзакции идут с первой карты, то есть со дня прихода в банк.
        client_start = persona.relationship_start

    elif source == "communications":
        client_start = consent_date(client_id)

    elif source in ("app_screens", "app_operations", "banners"):
        client_start = app_adoption(client_id)

    else:
        client_start = HISTORY_START

    if client_start is None:
        return None

    return max(availability, client_start)


def is_visible(client_id: int, source: str, ts: datetime) -> bool:
    """
    Попадает ли событие источника в хранилище.
    """

    start = first_seen(client_id, source)

    return start is not None and ts >= start


def coverage_rows(client_id: int) -> list[dict]:
    """
    Строки таблицы source_coverage для клиента.
    """

    return [
        {
            "client_id": client_id,
            "source": source,
            "availability_start": SOURCE_AVAILABILITY[source],
            "first_seen": first_seen(client_id, source),
        }
        for source in SOURCES
    ]
