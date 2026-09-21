from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# СКРЫТЫЕ ХАРАКТЕРИСТИКИ
# ============================================================
#
# Архетип задаёт РАСПРЕДЕЛЕНИЕ, а не поведение. Два клиента
# одного архетипа обязаны заметно различаться, поэтому средние
# сдвигаются умеренно, а разброс остаётся большим.
#
# Характеристики коррелированы: дисциплинированный клиент реже
# импульсивен, мобильный чаще цифровой. Розыгрыш идёт гауссовой
# копулой по общей корреляционной матрице.
# ============================================================


TRAIT_NAMES = (
    "financial_discipline",
    "risk_tolerance",
    "spending_impulsivity",
    "digital_affinity",
    "price_sensitivity",
    "merchant_loyalty",
    "mobility",
    "sociality",
    "fraud_vulnerability",
    "credit_appetite",
    "savings_propensity",
)

# Предпочтение каналов это отдельный вектор, а не одно число.
CHANNEL_NAMES = ("app", "branch", "call_center", "qr", "cash", "pos", "ecom")


# Пары с ненулевой корреляцией. Матрица достраивается
# симметрично, диагональ единичная, затем чинится до
# положительно определённой.
TRAIT_CORRELATIONS: dict = {
    ("financial_discipline", "spending_impulsivity"): -0.55,
    ("financial_discipline", "savings_propensity"): 0.48,
    ("financial_discipline", "credit_appetite"): -0.30,
    ("financial_discipline", "risk_tolerance"): -0.25,
    ("financial_discipline", "fraud_vulnerability"): -0.18,
    ("risk_tolerance", "credit_appetite"): 0.42,
    ("risk_tolerance", "spending_impulsivity"): 0.34,
    ("risk_tolerance", "savings_propensity"): -0.28,
    ("spending_impulsivity", "savings_propensity"): -0.40,
    ("spending_impulsivity", "price_sensitivity"): -0.32,
    ("spending_impulsivity", "merchant_loyalty"): -0.22,
    ("digital_affinity", "mobility"): 0.30,
    ("digital_affinity", "sociality"): 0.22,
    ("digital_affinity", "fraud_vulnerability"): -0.20,
    ("price_sensitivity", "savings_propensity"): 0.30,
    ("price_sensitivity", "merchant_loyalty"): 0.18,
    ("mobility", "merchant_loyalty"): -0.26,
    ("sociality", "mobility"): 0.20,
    ("credit_appetite", "savings_propensity"): -0.34,
}


@dataclass(frozen=True)
class TraitParams:

    names: tuple = TRAIT_NAMES

    channels: tuple = CHANNEL_NAMES

    correlations: dict = field(default_factory=lambda: dict(TRAIT_CORRELATIONS))

    # Базовое среднее каждой характеристики в шкале [0, 1].
    base_mean: dict = field(
        default_factory=lambda: {
            "financial_discipline": 0.55,
            "risk_tolerance": 0.40,
            "spending_impulsivity": 0.42,
            "digital_affinity": 0.52,
            "price_sensitivity": 0.55,
            "merchant_loyalty": 0.50,
            "mobility": 0.40,
            "sociality": 0.48,
            "fraud_vulnerability": 0.30,
            "credit_appetite": 0.38,
            "savings_propensity": 0.42,
        }
    )

    # Сдвиг среднего архетипом. Ключ это часть архетипа:
    # жизненный этап, роль банка или режим активности.
    stage_shift: dict = field(
        default_factory=lambda: {
            "young_adult": {"digital_affinity": 0.18, "spending_impulsivity": 0.14, "financial_discipline": -0.12, "savings_propensity": -0.10, "mobility": 0.12, "fraud_vulnerability": 0.08},
            "early_career": {"digital_affinity": 0.12, "credit_appetite": 0.08, "mobility": 0.08},
            "family": {"financial_discipline": 0.06, "price_sensitivity": 0.08, "credit_appetite": 0.06, "spending_impulsivity": -0.04},
            "mature": {"financial_discipline": 0.10, "savings_propensity": 0.10, "digital_affinity": -0.10, "spending_impulsivity": -0.08, "mobility": -0.06},
            "retired": {"financial_discipline": 0.14, "savings_propensity": 0.12, "digital_affinity": -0.26, "price_sensitivity": 0.16, "merchant_loyalty": 0.16, "mobility": -0.18, "fraud_vulnerability": 0.18, "credit_appetite": -0.12},
        }
    )

    role_shift: dict = field(
        default_factory=lambda: {
            "primary": {"digital_affinity": 0.12, "merchant_loyalty": 0.08, "sociality": 0.06},
            "secondary": {},
            "credit_only": {"credit_appetite": 0.22, "savings_propensity": -0.14, "financial_discipline": -0.06},
            "deposit_only": {"savings_propensity": 0.26, "credit_appetite": -0.20, "financial_discipline": 0.10, "spending_impulsivity": -0.10},
            "episodic": {"digital_affinity": -0.08, "merchant_loyalty": -0.06, "sociality": -0.06},
        }
    )

    mode_shift: dict = field(
        default_factory=lambda: {
            "silent": {"digital_affinity": -0.22, "sociality": -0.14},
            "rare": {"digital_affinity": -0.14, "sociality": -0.08},
            "regular": {"digital_affinity": 0.06},
            "high": {"digital_affinity": 0.14, "spending_impulsivity": 0.08, "sociality": 0.08},
            "extreme": {"digital_affinity": 0.22, "spending_impulsivity": 0.16, "mobility": 0.12, "sociality": 0.12},
        }
    )

    settlement_shift: dict = field(
        default_factory=lambda: {
            "metropolis": {"digital_affinity": 0.10, "mobility": 0.10, "merchant_loyalty": -0.06},
            "major_city": {"digital_affinity": 0.05},
            "regional_centre": {},
            "industrial_town": {"price_sensitivity": 0.05},
            "small_town": {"digital_affinity": -0.08, "merchant_loyalty": 0.08, "price_sensitivity": 0.08},
            "district_centre": {"digital_affinity": -0.12, "merchant_loyalty": 0.12, "price_sensitivity": 0.10, "mobility": -0.06},
            "rural": {"digital_affinity": -0.18, "merchant_loyalty": 0.16, "price_sensitivity": 0.12, "mobility": -0.10},
        }
    )

    # Разброс внутри архетипа. Чем больше, тем сильнее два
    # клиента одного архетипа отличаются друг от друга.
    spread: float = 1.0

    # Логистический перевод из нормальной шкалы в [0, 1].
    logistic_scale: float = 1.55

    # Дрейф характеристик после жизненных событий.
    drift_per_event: dict = field(
        default_factory=lambda: {
            "job_loss": {"price_sensitivity": 0.10, "spending_impulsivity": -0.06, "savings_propensity": -0.05, "credit_appetite": 0.06},
            "job_change": {"credit_appetite": 0.03},
            "income_up": {"spending_impulsivity": 0.05, "price_sensitivity": -0.06, "savings_propensity": 0.04},
            "income_down": {"price_sensitivity": 0.08, "spending_impulsivity": -0.05},
            "child_birth": {"financial_discipline": 0.07, "savings_propensity": 0.05, "spending_impulsivity": -0.05, "mobility": -0.06},
            "move": {"merchant_loyalty": -0.18, "mobility": 0.05},
            "wedding": {"financial_discipline": 0.04, "savings_propensity": 0.04},
            "divorce": {"financial_discipline": -0.05, "savings_propensity": -0.07, "sociality": -0.05},
            "illness": {"price_sensitivity": 0.05, "mobility": -0.08},
            "fraud_incident": {"digital_affinity": -0.08, "financial_discipline": 0.06, "fraud_vulnerability": -0.12},
            "big_purchase": {"savings_propensity": -0.06, "credit_appetite": 0.05},
            "recovery": {"financial_discipline": 0.05, "savings_propensity": 0.05},
        }
    )

    # Сколько дней занимает переход к новому значению.
    drift_blend_days: int = 45

    # Предпочтение каналов: база и сдвиг цифровой зрелостью.
    channel_base: dict = field(
        default_factory=lambda: {
            "app": 0.34, "pos": 0.26, "ecom": 0.13, "qr": 0.09,
            "cash": 0.10, "branch": 0.05, "call_center": 0.03,
        }
    )

    # Насколько черта меняет поведение, которым она управляет.
    # Без этих множителей черта остаётся украшением: корреляция
    # с собственным поведением была около нуля.
    impulsivity_rate_factor: float = 3.4
    sociality_transfer_factor: float = 1.0
    favourite_outlet_bonus: float = 2.2

    channel_digital_factor: dict = field(
        default_factory=lambda: {
            "app": 2.1, "ecom": 2.3, "qr": 1.9, "pos": 1.0,
            "cash": 0.35, "branch": 0.30, "call_center": 0.45,
        }
    )
