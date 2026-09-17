from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================
# ПРОДУКТЫ
# ============================================================
#
# Тарифов и дат реальных продуктов здесь НЕТ: они приходят из
# подтверждённой хронологии. Здесь живут правила банка, модель
# распространения, одобрение, вехи просрочки и вымышленные
# продукты с кодами SYNTH_*.
# ============================================================


MIGRATION_REASONS = ("successor_offer", "auto_renewal", "forced_migration", "client_request")

DATE_POLICIES = ("period_start", "period_middle", "period_end")


@dataclass(frozen=True)
class ProductParams:

    # Какой день периода берёт симуляция, когда известны только
    # месяц или год. Сам период при этом не переписывается.
    effective_date_policy: str = "period_start"

    # Чем считается дата, о которой известно только, что она
    # уже наступила к моменту свидетельства. registry_start
    # держит порядок дат согласованным: закрытие продаж не
    # может оказаться раньше начала продаж.
    unknown_start_policy: str = "registry_start"

    # Сколько дней после закрытия продаж продукт числится
    # closed_to_new_clients, прежде чем стать
    # servicing_existing_contracts. Правило, а не факт.
    closed_to_servicing_days: int = 365

    # Общие правила банка поверх правил продукта.
    bank_rules: dict = field(
        default_factory=lambda: {
            "max_debit_cards_total": 4,
            "max_active_cash_loans": 3,
            "max_active_contracts": 9,
            "max_debt_service_ratio": 0.50,
            "min_age": 18,
            "max_age": 79,
        }
    )

    # Одобрение по семейству продукта.
    approval_base: dict = field(
        default_factory=lambda: {
            "debit_card": 0.98,
            "credit_card": 0.52,
            "cash_loan": 0.44,
            "refinance": 0.40,
            "installment": 0.58,
            "deposit": 0.99,
            "deposit_certificate": 0.99,
            "bonds": 0.99,
            "insurance": 0.96,
            "service": 1.0,
        }
    )

    approval_income_factor: float = 0.20
    approval_discipline_factor: float = 0.18
    approval_stress_penalty: float = 0.38
    approval_dpd_penalty: float = 0.55
    approval_existing_loan_penalty: float = 0.22
    approval_bounds: tuple = (0.02, 0.99)

    reject_reasons: tuple = (
        "scoring_declined",
        "income_not_confirmed",
        "documents_invalid",
        "existing_debt",
        "age_limit",
        "blacklist",
        "manual_review_timeout",
        "debt_service_ratio",
    )

    reject_reason_weights: dict = field(
        default_factory=lambda: {
            "scoring_declined": 0.40,
            "documents_invalid": 0.20,
            "manual_review_timeout": 0.10,
            "income_not_confirmed": 0.16,
            "existing_debt": 0.08,
            "blacklist": 0.03,
            "debt_service_ratio": 0.03,
        }
    )

    blacklist_share: float = 0.03
    low_income_threshold: int = 95_000

    decision_delay_seconds: dict = field(
        default_factory=lambda: {
            "app": (30, 900),
            "web": (60, 1_800),
            "branch": (600, 7_200),
            "partner_pos": (120, 1_800),
            "call_center": (300, 5_400),
            "qazpost": (3_600, 172_800),
        }
    )

    disbursement_delay_seconds: tuple = (60, 21_600)

    application_cooldown_days: int = 20

    # Потолок дневной вероятности заявки. Раньше вероятность
    # считалась от суммы весов ВСЕХ кандидатов и почти каждый
    # день упиралась в потолок: отсюда семь договоров на
    # клиента вместо двух-трёх.
    application_probability_cap: float = 0.08

    # Вехи просрочки.
    dpd_milestones: tuple = (1, 30, 60, 90)

    # Сумма кредита как кратность месячному доходу. Банк не
    # выдаёт восемь зарплат наличными: это и было главной
    # причиной нереального уровня просрочки.
    loan_amount_income_multiple: dict = field(
        default_factory=lambda: {
            "cash_loan": (1.0, 4.0),
            "refinance": (1.0, 4.0),
            "installment": (0.15, 1.2),
            "credit_card": (0.8, 2.5),
        }
    )

    loan_term_options: tuple = (6, 12, 18, 24, 36, 48, 60)
    loan_term_weights: tuple = (0.08, 0.20, 0.14, 0.24, 0.20, 0.08, 0.06)

    # Доля лимита карты, которую банк считает месячным
    # обязательством при расчёте долговой нагрузки.
    credit_card_payment_share_of_limit: float = 0.10

    # Рефинансирование добирает немного наличных сверх
    # погашаемых долгов.
    refinance_cash_topup_share: tuple = (0.0, 0.3)

    # Обслуживание кредита.
    autopay_share: float = 0.58
    grace_days_before_missed: int = 3

    # Вероятность заплатить вовремя по полосам дисциплины.
    on_time_payment_probability: dict = field(
        default_factory=lambda: {
            "low": 0.935,
            "mid": 0.985,
            "high": 0.998,
        }
    )
    discipline_bands: tuple = (0.33, 0.66)

    # Частичный платёж имеет смысл, только если покрывает
    # заметную часть взноса.
    partial_payment_min_share: float = 0.20

    # Клиент подтягивает деньги из другого банка к сроку.
    loan_topup_from_other_bank_share: float = 0.86

    # Автоплатёж повторяет попытку внутри льготных дней.
    autopay_retry_days: int = 3

    # Кредитной картой кредит не гасят.
    loan_payment_from_credit_card: bool = False

    cure_probability_per_day: dict = field(
        default_factory=lambda: {
            "low": 0.025,
            "mid": 0.075,
            "high": 0.17,
        }
    )
    early_repayment_share_per_year: float = 0.11
    restructure_share_at_dpd60: float = 0.14

    # Проникновение продуктов до окна наблюдения: база и
    # наклон по соответствующей черте.
    prehistory_penetration: dict = field(
        default_factory=lambda: {
            "credit_card": (0.08, 0.30, "credit_appetite"),
            "cash_loan": (0.07, 0.28, "credit_appetite"),
            "deposit": (0.04, 0.24, "savings_propensity"),
            "installment": (0.09, 0.22, "credit_appetite"),
        }
    )

    prehistory_max_probability: float = 0.55

    # Депозиты.
    deposit_open_share_of_free_cash: tuple = (0.35, 0.95)
    deposit_early_close_share_per_year: float = 0.12
    deposit_rollover_share: float = 0.62

    # Карты.
    card_activation_delay_days: tuple = (0, 9)
    card_block_max_days: int = 21

    # Клиент и сам блокирует карту: странное списание в
    # выписке, поездка, карта не нашлась в кармане. Чаще
    # это временная заморозка, которую он же и снимает;
    # реже карта потеряна или скомпрометирована, и тогда
    # размораживать нечего, нужен перевыпуск.
    card_block_client_share_per_year: float = 0.06
    card_block_lost_share: float = 0.30
    card_freeze_days: tuple = (2, 14)
    card_freeze_self_unblock_share: float = 0.75
    card_lost_reissue_delay_days: tuple = (1, 9)
    card_expiry_years: int = 4

    # Карта рассрочки: минимальный платёж по наличному долгу
    # и запасной срок рассрочки, если тариф его не называет.
    card_cash_min_share: float = 0.10
    card_installment_months_default: int = 6

    # Распространение продукта.
    adoption: dict = field(
        default_factory=lambda: {
            "base_rate_per_year": {
                "debit_card": 0.10,
                "credit_card": 0.13,
                "cash_loan": 0.19,
                "refinance": 0.05,
                "installment": 0.26,
                "deposit": 0.12,
                "deposit_certificate": 0.02,
                "bonds": 0.01,
                "insurance": 0.07,
                "service": 0.0,
            },
            "ramp_days": 240,
            "ramp_start_share": 0.15,
            "early_adopter_digital_factor": 2.2,
            "offer_factor": 2.6,
            "organic_share": 0.34,
            "migration_pull": 3.1,
            "pilot_share_default": 0.15,
            "trait_factor": {
                "credit_card": ("credit_appetite", 2.2),
                "cash_loan": ("credit_appetite", 2.6),
                "refinance": ("credit_appetite", 2.0),
                "installment": ("credit_appetite", 1.7),
                "deposit": ("savings_propensity", 2.4),
                "deposit_certificate": ("savings_propensity", 2.8),
                "bonds": ("savings_propensity", 3.0),
                "debit_card": ("digital_affinity", 1.5),
                "insurance": ("risk_tolerance", -0.9),
            },
            "first_use_delay_days": (0, 21),
            "notice_days_default": 30,
        }
    )

    # Вымышленные продукты. Только коды SYNTH_*: реальные
    # продукты синтетических дат и тарифов не получают.
    synthetic_products: tuple = (
        {
            "product_code": "SYNTH_DEPOSIT_KIDS",
            "family": "deposit",
            "name": "Синтетический детский депозит",
            "launch": {
                "announced_at": "2026-01-10",
                "pilot": {"from": "2026-02-01", "days": 45, "share": 0.20,
                          "regions": ("Almaty", "Astana"), "channels": ("app",)},
                "sales_start_at": "2026-03-18",
            },
            "versions": (
                {
                    "product_version": 1,
                    "tariff_version": 1,
                    "eligibility": {"min_age": 21, "requires_children": True},
                    "channels": ("app", "branch"),
                    "terms": {"rate": 0.155, "term_months": 12, "min_amount": 50_000, "topup": True},
                },
            ),
            "events": (),
            "migration_policy": "none",
        },
        {
            "product_code": "SYNTH_CARD_YOUTH",
            "family": "debit_card",
            "name": "Синтетическая молодёжная карта",
            "launch": {"announced_at": None, "pilot": None, "sales_start_at": "2026-03-15"},
            "versions": (
                {
                    "product_version": 1,
                    "tariff_version": 1,
                    "eligibility": {"min_age": 18, "max_age": 27, "requires_app": True},
                    "channels": ("app",),
                    "terms": {"fee_monthly": 0, "cashback_base": 0.01, "cashback_cap": 8_000},
                },
            ),
            "events": (
                {"kind": "tariff_change", "at": "2026-06-01", "applies_to": "existing_from_date",
                 "notice_days": 30, "terms": {"fee_monthly": 0, "cashback_base": 0.005, "cashback_cap": 5_000}},
                {"kind": "suspension", "from": "2026-07-10", "to": "2026-07-24"},
            ),
            "migration_policy": "none",
        },
        {
            "product_code": "SYNTH_LOAN_GREEN",
            "family": "cash_loan",
            "name": "Синтетический зелёный кредит",
            "launch": {"announced_at": None, "pilot": None, "sales_start_at": "2025-11-01"},
            "versions": (
                {
                    "product_version": 1,
                    "tariff_version": 1,
                    "eligibility": {"min_age": 23, "max_age": 70},
                    "channels": ("app", "web", "branch"),
                    "terms": {"rate": 0.24, "term_min": 6, "term_max": 36,
                              "amount_min": 100_000, "amount_max": 4_000_000},
                },
            ),
            "events": (
                {"kind": "sales_close", "at": "2026-05-01", "successor": "SYNTH_LOAN_GREEN_2"},
            ),
            "migration_policy": "voluntary",
        },
        {
            "product_code": "SYNTH_LOAN_GREEN_2",
            "family": "cash_loan",
            "name": "Синтетический зелёный кредит 2.0",
            "launch": {"announced_at": None, "pilot": None, "sales_start_at": "2026-05-01"},
            "versions": (
                {
                    "product_version": 1,
                    "tariff_version": 1,
                    "eligibility": {"min_age": 23, "max_age": 70},
                    "channels": ("app", "web", "branch"),
                    "terms": {"rate": 0.22, "term_min": 6, "term_max": 48,
                              "amount_min": 100_000, "amount_max": 6_000_000},
                },
            ),
            "events": (),
            "predecessor": "SYNTH_LOAN_GREEN",
            "migration_policy": "voluntary",
        },
    )
