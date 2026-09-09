from __future__ import annotations

from datetime import datetime, timedelta

from ..app import AppScreenEvent, AppSession
from ..persona import Persona
from ..rng import NS_FUNNEL, KeyedRandom, keyed_rng
from ..world import FUNNEL_SCREENS


# ============================================================
# ИДЕЯ
# ============================================================
#
# В v1 reject_reason разыгрывался по постоянным весам и никак
# не был связан с тем, почему заявку отклонили: `age_limit`
# мог достаться клиенту тридцати лет, а `existing_debt` тому,
# у кого кредитов нет.
#
# Здесь причина следует из состояния, которое и привело
# к отказу. Разнообразие при этом не растёт: набор причин
# тот же самый, меняется только их согласованность.
# ============================================================


MIN_AGE = 21
MAX_AGE = 70

LOW_INCOME = 90_000

NO_INCOME_TYPES = frozenset({"unemployed", "student"})

# Причины, между которыми уже нет наблюдаемого различия.
GENERIC_REASONS = ("scoring_declined", "documents_invalid", "manual_review_timeout")
GENERIC_WEIGHTS = (0.62, 0.26, 0.12)


def reject_reason_for(
    product: str,
    persona: Persona,
    stress: float,
    owned: frozenset[str],
    rng: KeyedRandom,
) -> str:
    """
    Почему банк отказал.

    Порядок проверок это порядок отсечения в скоринге: сначала
    формальные ограничения, затем подтверждение дохода, затем
    долговая нагрузка, затем сам скоринг.
    """

    if persona.age < MIN_AGE or persona.age > MAX_AGE:
        return "age_limit"

    if persona.income_type in NO_INCOME_TYPES or persona.declared_income < LOW_INCOME:
        return "income_not_confirmed"

    # Долговая нагрузка мешает любому кредитному продукту,
    # а не только второму кредиту наличными: заявку на такой же
    # продукт банк вообще не принимает, пока действует прежний.
    if product in ("cash_loan", "credit_card") and "cash_loan" in owned:
        return "existing_debt"

    if stress > 0.55:
        return "scoring_declined"

    # Чёрный список это редкое и не выводимое из наблюдаемого
    # состояние: оно остаётся случайным по построению.
    if rng.random() < 0.04:
        return "blacklist"

    return str(rng.choice(GENERIC_REASONS, p=GENERIC_WEIGHTS))


def application_session_v2(
    client_id: int,
    product: str,
    started_at: datetime,
    approved: bool,
    persona: Persona,
    stress: float,
    owned: frozenset[str],
) -> AppSession:
    """
    Воронка заявки: view -> application -> kyc -> approved | rejected.

    Форма и тайминги те же, что в v1; отличается только причина
    отказа, которая теперь соответствует самому отказу.
    """

    rng = keyed_rng(NS_FUNNEL, client_id, started_at.toordinal(), started_at.hour)

    session_id = str(rng.integers(1_000_000_000, 9_999_999_999))

    session = AppSession(session_id=session_id, started_at=started_at)

    stages = ["view", "application", "kyc", "approved" if approved else "rejected"]

    gaps = [0, rng.integers(30, 200), rng.integers(60, 600), rng.integers(20, 900)]

    ts = started_at

    screens: list[AppScreenEvent] = []

    for stage, gap in zip(stages, gaps):

        ts = ts + timedelta(seconds=gap)

        screens.append(
            AppScreenEvent(
                client_id=client_id,
                ts=ts,
                session_id=session_id,
                firebase_screen=FUNNEL_SCREENS[stage],
                product=product,
                funnel_stage=stage,
                reject_reason=(
                    reject_reason_for(product, persona, stress, owned, rng)
                    if stage == "rejected"
                    else None
                ),
            )
        )

    session.screens = screens
    session.browsed_products = (product,)

    return session


__all__ = ["application_session_v2", "reject_reason_for"]
