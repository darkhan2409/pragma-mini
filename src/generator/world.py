from __future__ import annotations


# ============================================================
# ИДЕЯ
# ============================================================
#
# Справочники наблюдаемого мира: география, демография,
# занятость, каналы и шаблоны коммуникаций, экраны и операции
# приложения, слоты и офферы баннеров.
#
# У большинства справочников намеренно ДЛИННЫЙ ХВОСТ:
# несколько частых значений и десятки редких. Реальные
# категориальные поля выглядят именно так, и preprocessing
# обязан уметь сворачивать хвост.
# ============================================================


# ============================================================
# GEOGRAPHY
# ============================================================

REGIONS = (
    "Almaty",
    "Astana",
    "Shymkent",
    "Karaganda",
    "Aktobe",
    "Taraz",
    "Pavlodar",
    "Oskemen",
    "Kyzylorda",
    "Kostanay",
    "Atyrau",
    "Oral",
    "Semey",
    "Taldykorgan",
    "Petropavl",
)

# Веса расселения: Алматы и Астана дают больше трети когорты.
REGION_WEIGHTS = (
    0.210, 0.170, 0.095, 0.070, 0.055,
    0.050, 0.045, 0.045, 0.040, 0.040,
    0.040, 0.035, 0.035, 0.035, 0.035,
)

# Города региона: главный город и редкий хвост малых городов.
CITY_TAIL: dict[str, tuple[str, ...]] = {
    "Almaty": ("Kaskelen", "Talgar", "Esik", "Konaev"),
    "Astana": ("Kosshy", "Akkol", "Stepnogorsk"),
    "Shymkent": ("Turkistan", "Kentau", "Saryagash", "Lenger"),
    "Karaganda": ("Temirtau", "Balkhash", "Saran", "Zhezkazgan"),
    "Aktobe": ("Kandyagash", "Khromtau", "Alga"),
    "Taraz": ("Shu", "Karatau", "Zhanatas"),
    "Pavlodar": ("Ekibastuz", "Aksu"),
    "Oskemen": ("Ridder", "Zyryanovsk", "Altai"),
    "Kyzylorda": ("Baikonur", "Aral", "Zhosaly"),
    "Kostanay": ("Rudny", "Lisakovsk", "Arkalyk"),
    "Atyrau": ("Kulsary", "Makat"),
    "Oral": ("Aksai", "Zhanibek"),
    "Semey": ("Ayagoz", "Kurchatov"),
    "Taldykorgan": ("Tekeli", "Sarkand"),
    "Petropavl": ("Bulaevo", "Mamlyutka"),
}

MAIN_CITY_SHARE = 0.85

HOME_COUNTRY = "KZ"

# Зарубежные страны: покупки в поездках и на иностранных сайтах.
FOREIGN_COUNTRIES = (
    "RU", "TR", "AE", "CN", "GE", "UZ", "KG", "DE",
    "US", "TH", "EG", "IT", "ES", "GB", "AZ", "PL",
)

FOREIGN_COUNTRY_WEIGHTS = (
    0.20, 0.16, 0.13, 0.09, 0.07, 0.06, 0.05, 0.05,
    0.04, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01,
)


# ============================================================
# DEMOGRAPHY
# ============================================================

GENDERS = ("M", "F")
GENDER_WEIGHTS = (0.47, 0.53)

FAMILY_STATUSES = (
    "married",
    "single",
    "civil_marriage",
    "divorced",
    "widow",
    "unknown",
)
FAMILY_STATUS_WEIGHTS = (0.52, 0.24, 0.10, 0.09, 0.04, 0.01)

EDUCATION_TYPES = (
    "secondary",
    "higher",
    "incomplete_higher",
    "vocational",
    "academic_degree",
)
EDUCATION_WEIGHTS = (0.38, 0.33, 0.12, 0.16, 0.01)

HOUSING_TYPES = (
    "own_apartment",
    "rented",
    "with_parents",
    "own_house",
    "municipal",
    "office_housing",
)
HOUSING_WEIGHTS = (0.44, 0.22, 0.16, 0.14, 0.03, 0.01)

INCOME_TYPES = (
    "employed",
    "self_employed",
    "business_owner",
    "pensioner",
    "state_employee",
    "student",
    "unemployed",
)
INCOME_TYPE_WEIGHTS = (0.58, 0.13, 0.06, 0.11, 0.08, 0.03, 0.01)

INDUSTRIES = (
    "trade",
    "construction",
    "transport",
    "education",
    "healthcare",
    "government",
    "manufacturing",
    "oil_and_gas",
    "agriculture",
    "it",
    "finance",
    "hospitality",
    "mining",
    "utilities",
    "telecom",
    "security",
    "logistics",
    "real_estate",
    "media",
    "science",
)

INDUSTRY_WEIGHTS = (
    0.150, 0.110, 0.095, 0.085, 0.080,
    0.075, 0.070, 0.055, 0.050, 0.045,
    0.040, 0.035, 0.030, 0.025, 0.020,
    0.012, 0.010, 0.006, 0.004, 0.003,
)

# Отрасль наблюдается только у работающих по найму и госслужащих.
INDUSTRY_INCOME_TYPES = frozenset({"employed", "state_employee"})


# ============================================================
# COMMUNICATIONS
# ============================================================
#
# Три исходящих канала, как в реальном контуре.
# Доли доставки из отчёта: call 6.2, sms 81.3, push 42.6 процента.
# ============================================================

COMM_CHANNELS = ("call", "sms", "push")

DELIVERY_RATE = {
    "call": 0.062,
    "sms": 0.813,
    "push": 0.700,
}

# Кампания -> продукт, который она предлагает (None: сервисная).
CAMPAIGN_PRODUCT: dict[str, str | None] = {
    "cash_loan_offer": "cash_loan",
    "credit_card_offer": "credit_card",
    "deposit_offer": "deposit",
    "insurance_offer": "insurance",
    "debit_card_offer": "debit_card",
    "cashback": None,
    "payment_reminder": None,
    "security": None,
    "service": None,
    "nps_survey": None,
}

CAMPAIGN_CHANNELS: dict[str, tuple[str, ...]] = {
    "cash_loan_offer": ("sms", "push", "call"),
    "credit_card_offer": ("push", "sms", "call"),
    "deposit_offer": ("push", "sms"),
    "insurance_offer": ("push", "call"),
    "debit_card_offer": ("push", "sms"),
    "cashback": ("push",),
    "payment_reminder": ("sms", "push", "call"),
    "security": ("sms", "push"),
    "service": ("push", "sms"),
    "nps_survey": ("push",),
}

# Шаблоны кампании: первые частотные, дальше редкий хвост.
CAMPAIGN_TEMPLATES: dict[str, tuple[str, ...]] = {
    "cash_loan_offer": (
        "CL_BASE_2024", "CL_PRIME_RATE", "CL_TOPUP", "CL_REFIN",
        "CL_SUMMER_PROMO", "CL_WINTER_PROMO", "CL_PARTNER_AUTO",
    ),
    "credit_card_offer": (
        "CC_OZEN_BASE", "CC_OZEN_GOLD", "CC_LIMIT_UP",
        "CC_TRAVEL", "CC_PARTNER_RETAIL",
    ),
    "deposit_offer": (
        "DEP_STANDARD", "DEP_MAX_RATE", "DEP_CHILD", "DEP_PENSION",
    ),
    "insurance_offer": (
        "INS_TRAVEL", "INS_LIFE", "INS_PROPERTY", "INS_AUTO_KASKO",
    ),
    "debit_card_offer": (
        "DC_ARNA", "DC_SALARY", "DC_YOUTH",
    ),
    "cashback": (
        "CB_MONTHLY", "CB_CATEGORY_FOOD", "CB_CATEGORY_FUEL",
        "CB_PARTNER_MARKET",
    ),
    "payment_reminder": (
        "PR_DUE_SOON", "PR_DUE_TODAY", "PR_OVERDUE_SOFT",
    ),
    "security": (
        "SEC_LOGIN_ALERT", "SEC_CARD_BLOCK", "SEC_FRAUD_WARN",
    ),
    "service": (
        "SRV_APP_UPDATE", "SRV_BRANCH_INFO", "SRV_TARIFF_CHANGE",
        "SRV_DOC_EXPIRY",
    ),
    "nps_survey": (
        "NPS_APP", "NPS_BRANCH",
    ),
}


# ============================================================
# APP SCREENS
# ============================================================
#
# firebase_screen как в GA4-выгрузке: часть экранов общая,
# часть привязана к домену, отдельная ветка это воронка заявки.
# ============================================================

SCREEN_HOME = "s_000_home"
SCREEN_OFFERS = "s_001_for_me"

BROWSE_SCREENS: dict[str, tuple[str, ...]] = {
    "home": ("s_000_home", "s_010_balance", "s_011_history"),
    "cards": ("s_100_cards", "s_101_card_detail", "s_102_card_limits", "s_103_card_pin"),
    "transfers": ("s_200_transfers", "s_201_transfer_phone", "s_202_transfer_card", "s_203_transfer_confirm"),
    "payments": ("s_300_payments", "s_301_payment_utility", "s_302_payment_mobile", "s_303_payment_fine"),
    "loans": ("s_400_loans", "s_401_loan_calc", "s_402_loan_terms"),
    "deposits": ("s_500_deposits", "s_501_deposit_calc", "s_502_deposit_terms"),
    "insurance": ("s_600_insurance", "s_601_insurance_terms"),
    "market": ("s_700_market", "s_701_market_item", "s_702_market_cart"),
    "profile": ("s_800_profile", "s_801_settings", "s_802_documents"),
    "support": ("s_900_support", "s_901_chat", "s_902_faq"),
}

BROWSE_DOMAINS = tuple(BROWSE_SCREENS)

# Воронка заявки: стадии идут строго в этом порядке.
FUNNEL_STAGES = ("view", "application", "kyc", "approved", "rejected")

FUNNEL_SCREENS: dict[str, str] = {
    "view": "s_a00_offer_view",
    "application": "s_a01_application_form",
    "kyc": "s_a02_kyc_check",
    "approved": "s_a03_success",
    "rejected": "s_a04_reject",
}

REJECT_REASONS = (
    "scoring_declined",
    "income_not_confirmed",
    "documents_invalid",
    "existing_debt",
    "age_limit",
    "blacklist",
    "manual_review_timeout",
)

REJECT_REASON_WEIGHTS = (0.38, 0.22, 0.14, 0.12, 0.06, 0.05, 0.03)


# ============================================================
# APP OPERATIONS
# ============================================================
#
# bdp_capp_* по доменам: операция и её статус.
# ============================================================

DOMAIN_OPERATIONS: dict[str, tuple[str, ...]] = {
    "auth": ("login", "logout", "biometry_login", "pin_change", "device_bind"),
    "cards": ("card_view", "card_block", "card_unblock", "limit_change", "pin_reset", "card_order"),
    "transfers": ("transfer_phone", "transfer_card", "transfer_own", "transfer_template", "transfer_abroad"),
    "payments": ("pay_utility", "pay_mobile", "pay_internet", "pay_fine", "pay_tax", "pay_qr"),
    "loans": ("loan_view", "loan_schedule", "loan_calc", "loan_early_repay", "loan_statement"),
    "deposits": ("deposit_view", "deposit_open", "deposit_topup", "deposit_close"),
    "market": ("market_browse", "market_order", "market_return"),
    "support": ("chat_open", "callback_request", "complaint"),
}

APP_DOMAINS = tuple(DOMAIN_OPERATIONS)

# Доля когорты, у которой домен вообще появляется
# (отчёт: auth 84.9, cards 60.1, transfers 50.9,
#  loans 44.2, payments 29.6 процента).
DOMAIN_ADOPTION = {
    "auth": 0.85,
    "cards": 0.60,
    "transfers": 0.51,
    "loans": 0.44,
    "payments": 0.30,
    "deposits": 0.18,
    "market": 0.14,
    "support": 0.12,
}

OPERATION_STATUSES = ("success", "failed", "cancelled")
OPERATION_STATUS_WEIGHTS = (0.93, 0.045, 0.025)


# ============================================================
# BANNERS
# ============================================================
#
# Слот в приложении и оффер в нём. Одна строка на действие:
# action = shown | clicked (двух флагов нет намеренно).
# ============================================================

BANNER_SLOTS = (
    "main_top",
    "for_me_1",
    "for_me_2",
    "for_me_3",
    "cards_bottom",
    "payments_bottom",
    "story_1",
)

BANNER_SLOT_WEIGHTS = (0.28, 0.20, 0.15, 0.10, 0.10, 0.10, 0.07)

BANNER_OFFERS = (
    "cash_loan",
    "credit_card",
    "deposit",
    "insurance",
    "debit_card",
    "cashback",
    "market_promo",
    "referral",
)

BANNER_OFFER_WEIGHTS = (0.22, 0.18, 0.13, 0.10, 0.10, 0.12, 0.10, 0.05)

# Оффер -> продукт (None: не продуктовый баннер).
BANNER_OFFER_PRODUCT: dict[str, str | None] = {
    "cash_loan": "cash_loan",
    "credit_card": "credit_card",
    "deposit": "deposit",
    "insurance": "insurance",
    "debit_card": "debit_card",
    "cashback": None,
    "market_promo": None,
    "referral": None,
}

ACTION_SHOWN = "shown"
ACTION_CLICKED = "clicked"
