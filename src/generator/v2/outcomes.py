from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from ..rng import COMPONENT_CONTENT, NS_V2_OUTAGE, KeyedRandom, event_rng
from ..world import OPERATION_STATUSES
from .config import (
    MAX_ATTEMPTS_PER_INTENT,
    MAX_SUPPORT_HOPS,
    OUTAGE_DAY_PROBABILITY,
    OUTAGE_DOMAINS,
    OUTAGE_FAILURE_BOOST,
    OUTAGE_HOURS,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# В v1 статус операции разыгрывался по константам 0.93 / 0.045
# / 0.025 одинаково для входа в приложение и для перевода за
# границу, и ни на что не влиял.
#
# В v2 исход зависит только от того, что к операции ОТНОСИТСЯ:
# её тип, номер попытки, сбой сервиса, кредитный стресс для
# денежных операций, утилизация только для смены лимита,
# заблокированная карта для списаний.
#
# Стресс не трогает просмотр баланса, а утилизация не трогает
# вход в приложение: это проверяется тестом.
# ============================================================

SUCCESS, FAILED, CANCELLED = OPERATION_STATUSES

FLOOR = 0.02


# ------------------------------------------------------------
# СЕМЕЙСТВА ОПЕРАЦИЙ
# ------------------------------------------------------------

READ_ONLY = frozenset(
    {
        "card_view",
        "loan_view",
        "loan_schedule",
        "loan_statement",
        "loan_calc",
        "deposit_view",
        "market_browse",
    }
)

AUTH = frozenset({"login", "biometry_login", "logout"})

MONEY = frozenset(
    {
        "transfer_phone",
        "transfer_card",
        "transfer_own",
        "transfer_template",
        "transfer_abroad",
        "pay_utility",
        "pay_mobile",
        "pay_internet",
        "pay_fine",
        "pay_tax",
        "pay_qr",
        "deposit_topup",
        "loan_early_repay",
        "market_order",
    }
)

# Списания, которым мешает заблокированная карта.
CARD_DEPENDENT = frozenset(
    {
        "pay_utility",
        "pay_mobile",
        "pay_internet",
        "pay_fine",
        "pay_tax",
        "pay_qr",
        "transfer_card",
        "market_order",
    }
)

# Многошаговые формы: их чаще бросают на полпути.
MULTI_STEP = frozenset(
    {
        "transfer_abroad",
        "card_order",
        "limit_change",
        "deposit_open",
        "deposit_close",
        "loan_early_repay",
        "pay_tax",
        "device_bind",
    }
)

BASE_RATES: dict[str, tuple[float, float, float]] = {
    "read_only": (0.985, 0.010, 0.005),
    "auth": (0.970, 0.020, 0.010),
    "money": (0.900, 0.045, 0.055),
    "abroad": (0.800, 0.120, 0.080),
    "order": (0.850, 0.050, 0.100),
    "action": (0.930, 0.040, 0.030),
}


def family(operation: str) -> str:

    if operation == "transfer_abroad":
        return "abroad"

    if operation == "card_order":
        return "order"

    if operation in READ_ONLY:
        return "read_only"

    if operation in AUTH:
        return "auth"

    if operation in MONEY:
        return "money"

    return "action"


@dataclass(frozen=True)
class OperationContext:
    operation: str
    domain: str
    attempt: int = 1
    outage: bool = False
    stress: float = 0.0
    utilization: float = 0.0
    digital: float = 0.5
    card_blocked: bool = False

    # Жёсткое ограничение: действие в принципе не может пройти.
    # Проверяется ДО розыгрыша, и пол вероятности его не смягчает.
    feasible: bool = True


def outcome_probabilities(ctx: OperationContext) -> tuple[float, float, float]:
    """
    Вероятности success / failed / cancelled.

    Если действие невозможно, успех невозможен тоже: пол
    вероятности применяется только к оставшимся исходам.
    """

    success, failed, cancelled = BASE_RATES[family(ctx.operation)]

    if not ctx.feasible:

        # Невозможное действие заканчивается отказом системы
        # или отменой клиентом, но никогда успехом.
        share = 0.5 * (1.0 + 0.6 * (1.0 - ctx.digital))

        failed = max(failed, 1.0 - share)
        cancelled = max(cancelled, share)

        total = failed + cancelled
        scale = 1.0 - 2.0 * FLOOR

        return (
            0.0,
            FLOOR + scale * failed / total,
            FLOOR + scale * cancelled / total,
        )

    # Повтор чаще проходит, но третья попытка значит,
    # что проблема не в случайности.
    if ctx.attempt == 2:
        failed *= 0.6
    elif ctx.attempt >= 3:
        failed *= 1.3

    if ctx.outage:
        failed += OUTAGE_FAILURE_BOOST

    if ctx.operation in MONEY:
        failed += 0.10 * ctx.stress

    if ctx.operation == "limit_change":
        failed += 0.25 * ctx.utilization

    if ctx.card_blocked and ctx.operation in CARD_DEPENDENT:
        failed += 0.50

    if ctx.operation in MULTI_STEP:
        cancelled += 0.15 * (1.0 - ctx.digital)

    success = max(0.0, 1.0 - failed - cancelled) if failed + cancelled < 1.0 else 0.0

    total = success + failed + cancelled

    if total <= 0.0:
        success, failed, cancelled = 1.0, 1.0, 1.0
        total = 3.0

    # Пол вероятности через смесь с равномерным: никакой исход
    # не становится невозможным и никакой не становится точным.
    scale = 1.0 - 3.0 * FLOOR

    return (
        FLOOR + scale * success / total,
        FLOOR + scale * failed / total,
        FLOOR + scale * cancelled / total,
    )


def draw_status(ctx: OperationContext, rng: KeyedRandom) -> str:

    success, failed, _ = outcome_probabilities(ctx)

    value = rng.random()

    if value < success:
        return SUCCESS

    if value < success + failed:
        return FAILED

    return CANCELLED


# ------------------------------------------------------------
# ЧТО КЛИЕНТ ДЕЛАЕТ ПОСЛЕ ИСХОДА
# ------------------------------------------------------------


def next_step_options(status: str, attempt: int, support_used: int) -> tuple[str, ...]:
    """
    Допустимые продолжения. Цепочки ограничены: попыток на одно
    намерение не больше MAX_ATTEMPTS_PER_INTENT, поход в поддержку
    не больше MAX_SUPPORT_HOPS за сессию.
    """

    if status == SUCCESS:
        return ("result",)

    if status == CANCELLED:
        return ("back", "exit")

    options: list[str] = []

    if attempt < MAX_ATTEMPTS_PER_INTENT:
        options.append("retry")

    if support_used < MAX_SUPPORT_HOPS:
        options.append("support")

    options.append("abandon")

    return tuple(options)


# ------------------------------------------------------------
# СБОЙ СЕРВИСА
# ------------------------------------------------------------
#
# Общий для всех клиентов календарь: инцидент это свойство
# банка, а не клиента. В RAW не пишется, виден только как
# кучность сбоев в один день у разных людей.
# ------------------------------------------------------------


@lru_cache(maxsize=4096)
def outage_windows(day: int) -> tuple[tuple[str, int, int], ...]:

    rng = event_rng(NS_V2_OUTAGE, 0, day, 0, COMPONENT_CONTENT)

    if rng.random() >= OUTAGE_DAY_PROBABILITY:
        return ()

    domain = str(rng.choice(OUTAGE_DOMAINS))

    start = int(rng.integers(0, 24))
    length = int(rng.integers(OUTAGE_HOURS[0], OUTAGE_HOURS[1] + 1))

    return ((domain, start, min(24, start + length)),)


def in_outage(domain: str, ts: datetime) -> bool:

    for name, start, end in outage_windows(ts.toordinal()):
        if name == domain and start <= ts.hour < end:
            return True

    return False


__all__ = [
    "CANCELLED",
    "FAILED",
    "SUCCESS",
    "OperationContext",
    "draw_status",
    "family",
    "in_outage",
    "next_step_options",
    "outage_windows",
    "outcome_probabilities",
]
