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


def score_band(foreign: bool, amount_share: float, rng) -> str:
    """
    Полоса риска операции.

    Скоринг видит только операцию: её долю от известного банку
    дохода и страну. Вид эпизода и шаг сценария ему неизвестны,
    иначе полоса сама называла бы, настоящая это тревога или
    ложная. Разброс — неточность модели скоринга.
    """

    settings = params_module.active().fraud

    low, high = settings.score_band_thresholds

    score = 0.35 + 0.50 * min(1.0, amount_share) + rng.uniform(-0.10, 0.10)

    if foreign:
        score += 0.20

    if score >= high:
        return "high"

    if score >= low:
        return "medium"

    return "low"


def rule_code(subject: str, foreign: bool, rng) -> str:
    """
    Правило, по которому сработал антифрод.

    Как и полоса риска, правило видит только операцию: на чём
    сработало и из какой страны. Вид эпизода ему неизвестен —
    иначе код правила сам называл бы, настоящая это тревога или
    ложная, и модель читала бы ответ вместо того, чтобы его
    предсказывать.
    """

    settings = params_module.active().fraud

    weights = dict(settings.rule_weights.get(subject, settings.rule_weights["card"]))

    if foreign and "R_GEO_ANOMALY" in weights:
        weights["R_GEO_ANOMALY"] *= settings.rule_geo_boost

    return str(rng.weighted(weights))


__all__ = ["rule_code", "score_band"]
