from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .. import params as params_module
from ..rng import NS_OUTAGE, KeyedRandom, day_rng, state_cache
from ..world.dictionaries import OPERATION_STATUSES


# ============================================================
# ИСХОД ОПЕРАЦИИ
# ============================================================
#
# Порядок обязателен: сначала ЖЁСТКОЕ ограничение, потом
# случайный исход. Невозможность выражается флагом, а не малым
# весом, иначе пол вероятности разрешил бы невозможный успех.
# ============================================================


SUCCESS, FAILED, CANCELLED = OPERATION_STATUSES

FLOOR = 0.02

READ_ONLY = frozenset(
    {
        "card_view", "loan_view", "loan_schedule", "loan_statement", "loan_calc",
        "deposit_view", "market_browse", "profile_view", "statement_order",
    }
)

AUTH_OPERATIONS = frozenset({"login", "biometry_login", "logout", "pin_change", "device_bind"})

MONEY_OPERATIONS = frozenset(
    {
        "transfer_phone", "transfer_card", "transfer_own", "transfer_template", "transfer_abroad",
        "pay_utility", "pay_mobile", "pay_internet", "pay_fine", "pay_tax", "pay_qr",
        "loan_repay", "loan_early_repay", "deposit_topup", "deposit_withdraw",
        "market_order", "bond_buy", "certificate_open",
    }
)

MULTI_STEP = frozenset(
    {"transfer_abroad", "card_order", "card_reissue", "deposit_open", "certificate_open", "bond_buy"}
)

BASE_RATES = {
    "read_only": (0.985, 0.010, 0.005),
    "auth": (0.965, 0.022, 0.013),
    "money": (0.900, 0.045, 0.055),
    "abroad": (0.790, 0.125, 0.085),
    "order": (0.845, 0.050, 0.105),
    "action": (0.930, 0.040, 0.030),
}


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
    insufficient: bool = False
    feasible: bool = True


def family(operation: str) -> str:

    if operation == "transfer_abroad":
        return "abroad"

    if operation in ("card_order", "card_reissue"):
        return "order"

    if operation in READ_ONLY:
        return "read_only"

    if operation in AUTH_OPERATIONS:
        return "auth"

    if operation in MONEY_OPERATIONS:
        return "money"

    return "action"


def outcome_probabilities(context: OperationContext) -> tuple:
    """
    Вероятности success, failed, cancelled.
    """

    settings = params_module.active().activity

    if not context.feasible:
        # Успех НЕВОЗМОЖЕН. Пол применяется только к остальным
        # исходам, иначе он разрешил бы невозможное.
        cancel_share = 0.5 * (1.0 + 0.6 * (1.0 - context.digital))
        failed = max(FLOOR, 1.0 - cancel_share)
        cancelled = max(FLOOR, cancel_share)
        total = failed + cancelled
        return (0.0, failed / total, cancelled / total)

    success, failed, cancelled = BASE_RATES[family(context.operation)]

    if context.attempt == 2:
        failed *= 0.6
    elif context.attempt >= 3:
        failed *= 1.3

    if context.outage:
        failed += settings.outage_failure_boost

    if family(context.operation) == "money":
        failed += 0.10 * context.stress

    if context.operation == "limit_change":
        failed += 0.25 * context.utilization

    if context.insufficient:
        failed += 0.55

    if context.operation in MULTI_STEP:
        cancelled += 0.15 * (1.0 - context.digital)

    success = max(FLOOR, success)
    failed = max(FLOOR, failed)
    cancelled = max(FLOOR, cancelled)

    total = success + failed + cancelled

    return (success / total, failed / total, cancelled / total)


def draw_status(context: OperationContext, rng: KeyedRandom) -> str:

    success, failed, _ = outcome_probabilities(context)

    value = rng.random()

    if value < success:
        return SUCCESS

    if value < success + failed:
        return FAILED

    return CANCELLED


# ============================================================
# СБОЙ СЕРВИСА
# ============================================================
#
# Сбой общий для всех клиентов и в RAW не пишется: он виден
# только по всплеску неуспешных операций.
# ============================================================


@state_cache
def outage_windows(day: int) -> tuple:

    settings = params_module.active().activity

    rng = day_rng(NS_OUTAGE, 0, day)

    if rng.random() >= settings.outage_day_probability:
        return ()

    domain = str(rng.choice(settings.outage_domains))

    start_hour = rng.integers(0, 22)
    length = rng.integers(*settings.outage_hours)

    return ((domain, start_hour, min(24, start_hour + length)),)


def in_outage(domain: str, ts: datetime) -> bool:

    for name, start, stop in outage_windows(ts.toordinal()):
        if name == domain and start <= ts.hour < stop:
            return True

    return False


__all__ = [
    "AUTH_OPERATIONS",
    "CANCELLED",
    "FAILED",
    "FLOOR",
    "MONEY_OPERATIONS",
    "MULTI_STEP",
    "READ_ONLY",
    "SUCCESS",
    "OperationContext",
    "draw_status",
    "family",
    "in_outage",
    "outage_windows",
    "outcome_probabilities",
]
