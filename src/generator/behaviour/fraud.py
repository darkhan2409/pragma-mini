from __future__ import annotations

from .. import params as params_module


# ============================================================
# АНТИФРОД
# ============================================================
#
# Банк видит не мошенника, а СОБЫТИЯ: необычную операцию,
# срабатывание правила, решение, блокировку, обращение клиента,
# возврат по оспариванию и перевыпуск карты.
#
# В RAW нет ни флага мошенничества, ни персоны: всё это живёт
# только внутри симуляции.
# ============================================================


RULE_CODES = {
    "card_compromise": "R_CARD_VELOCITY",
    "unusual_purchase": "R_AMOUNT_ANOMALY",
    "suspicious_transfer": "R_TRANSFER_PATTERN",
    "social_engineering": "R_SOCIAL_PATTERN",
    "account_takeover": "R_NEW_DEVICE",
    "false_positive": "R_GEO_ANOMALY",
}


def score_band(kind: str, step_kind: str, foreign: bool, amount_share: float) -> str:
    """
    Полоса риска операции.
    """

    settings = params_module.active().fraud

    low, high = settings.score_band_thresholds

    score = 0.25

    if step_kind == "strike":
        score += 0.35

    if foreign:
        score += 0.15

    score += 0.30 * min(1.0, amount_share)

    if kind in ("card_compromise", "account_takeover"):
        score += 0.15

    if kind == "false_positive":
        score -= 0.10

    if score >= high:
        return "high"

    if score >= low:
        return "medium"

    return "low"


def rule_code(kind: str) -> str:
    return RULE_CODES.get(kind, "R_GENERIC")


def blocks_card(decision: str) -> bool:
    return decision == "block"


def requires_confirmation(decision: str) -> bool:
    return decision == "confirm_request"


__all__ = ["RULE_CODES", "blocks_card", "requires_confirmation", "rule_code", "score_band"]
