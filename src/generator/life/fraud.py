from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import params as params_module
from .. import config
from ..rng import NS_FRAUD, keyed_rng
from .events import active_vacation
from .persona import Persona


# ============================================================
# МОШЕННИЧЕСТВО КАК ЭПИЗОД
# ============================================================
#
# Клиент не бывает «мошенническим типом»: у него бывает эпизод.
# Уязвимость и экспозиция повышают вероятность, но в наблюдаемых
# данных нет ни флага, ни персоны — только цепочка операций,
# срабатывание антифрода, блокировка, обращение и восстановление.
# ============================================================


@dataclass(frozen=True)
class FraudStep:
    ts: datetime
    kind: str
    amount_hint: float
    foreign: bool
    online: bool


@dataclass(frozen=True)
class FraudEpisode:
    kind: str
    start: datetime
    steps: tuple
    detected: bool
    detect_delay_minutes: int
    decision: str
    client_response: str
    opens_case: bool
    chargeback: bool
    reissue: bool


def plan_episodes(persona: Persona, events: tuple) -> tuple:
    """
    Мошеннические эпизоды клиента на горизонте.
    """

    settings = params_module.active().fraud

    vulnerability = persona.trait("fraud_vulnerability")
    digital = persona.trait("digital_affinity")
    mobility = persona.trait("mobility")

    rate = settings.base_rate_per_year
    rate *= 1.0 + settings.vulnerability_factor * vulnerability
    rate *= 1.0 + (settings.online_exposure_factor - 1.0) * digital
    rate *= 1.0 + (settings.travel_exposure_factor - 1.0) * mobility

    start = max(config.HISTORY_START, persona.relationship_start)
    span_days = (config.PLANNING_END - start).days

    if span_days <= 30:
        return ()

    rng = keyed_rng(NS_FRAUD, persona.client_ordinal, 1)

    count = rng.poisson(rate * span_days / 365.25)

    episodes: list[FraudEpisode] = []

    for index in range(min(count, 3)):

        item_rng = keyed_rng(NS_FRAUD, persona.client_ordinal, 2, index)

        kind = item_rng.weighted(settings.kind_weights)

        offset = int(item_rng.integers(5, span_days - 5))
        begin = start + timedelta(days=offset, hours=int(item_rng.integers(0, 24)))

        steps: list[FraudStep] = []

        moment = begin

        probe_low, probe_high = settings.probe_purchases.get(kind, (0, 0))

        for _ in range(item_rng.integers(probe_low, probe_high + 1)):
            moment += timedelta(minutes=int(item_rng.integers(*settings.step_gap_minutes)))
            steps.append(
                FraudStep(
                    ts=moment,
                    kind="probe",
                    # Проба соразмерна доходу жертвы, а не
                    # фиксированной вилке в тенге.
                    amount_hint=float(
                        max(
                            200,
                            persona.true_income
                            * item_rng.uniform(*settings.probe_amount_share_of_income),
                        )
                    ),
                    foreign=bool(item_rng.random() < 0.45),
                    online=True,
                )
            )

        strike_low, strike_high = settings.strike_count.get(kind, (1, 1))

        for _ in range(max(1, item_rng.integers(strike_low, strike_high + 1))):
            moment += timedelta(minutes=int(item_rng.integers(*settings.step_gap_minutes)))
            steps.append(
                FraudStep(
                    ts=moment,
                    kind="strike",
                    amount_hint=float(item_rng.uniform(*settings.strike_amount_share_of_limit)),
                    foreign=bool(item_rng.random() < 0.55),
                    online=bool(item_rng.random() < 0.72),
                )
            )

        if kind == "false_positive":
            # Ложное срабатывание: обычная операция клиента,
            # просто необычная для его профиля.
            vacation = active_vacation(events, begin)
            steps = [
                FraudStep(
                    ts=begin,
                    kind="legitimate",
                    amount_hint=float(item_rng.uniform(0.15, 0.60)),
                    foreign=bool(vacation is not None and vacation.payload.get("abroad")),
                    online=bool(item_rng.random() < 0.5),
                )
            ]

        detected = bool(item_rng.random() < settings.detection_probability.get(kind, 0.6))

        decision_weights = dict(settings.decision_weights)

        if kind in ("card_compromise", "account_takeover"):
            decision_weights["block"] *= settings.block_decision_boost_high_band

        decision = item_rng.weighted(decision_weights)

        if kind == "false_positive":
            response = "confirmed_by_client" if item_rng.random() < settings.false_positive_confirm_share else "disputed"
        elif item_rng.random() < settings.client_disputes_share:
            response = "disputed"
        elif item_rng.random() < settings.client_confirms_share:
            response = "confirmed_by_client"
        else:
            response = "no_response"

        opens_case = bool(
            response == "disputed" and item_rng.random() < settings.dispute_opens_case_share
        )

        chargeback = bool(
            opens_case
            and kind != "false_positive"
            and item_rng.random() < settings.chargeback_share_of_disputes
        )

        reissue = bool(
            decision == "block"
            and kind != "false_positive"
            and item_rng.random() < settings.reissue_share_after_block
        )

        episodes.append(
            FraudEpisode(
                kind=kind,
                start=begin,
                steps=tuple(steps),
                detected=detected,
                detect_delay_minutes=int(item_rng.integers(*settings.detection_delay_minutes)),
                decision=decision,
                client_response=response,
                opens_case=opens_case,
                chargeback=chargeback,
                reissue=reissue,
            )
        )

    episodes.sort(key=lambda item: item.start)

    return tuple(episodes)


__all__ = ["FraudEpisode", "FraudStep", "plan_episodes"]
