from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# СПРАВОЧНИКИ НАБЛЮДАЕМОГО МИРА
# ============================================================
#
# Категории расходов, MCC, экраны и операции приложения,
# кампании и шаблоны, устройства, темы обращений и коды причин.
# У большинства справочников намеренно длинный хвост.
# ============================================================


# ------------------------------------------------------------
# ПОТРЕБНОСТИ И КАТЕГОРИИ
# ------------------------------------------------------------
#
# Потребность порождает категорию, категория задаёт MCC и
# доступные каналы. Сумма приходит отдельно, из корзины и
# ценового уровня точки.
# ------------------------------------------------------------


@dataclass(frozen=True)
class Category:
    name: str
    sector: str
    subcategories: tuple
    mccs: tuple
    mcc_weights: tuple
    online_share: float
    essential: bool
    need: str
    rural_available: bool = True
    hours: tuple = (8, 22)


CATEGORIES: tuple[Category, ...] = (
    # --- еда ---
    Category("grocery", "retail_food", ("supermarket", "hypermarket", "discounter"),
             ("5411", "5499"), (0.88, 0.12), 0.06, True, "food"),
    Category("market", "retail_food", ("bazaar", "farm_stall"),
             ("5499", "5411"), (0.70, 0.30), 0.01, True, "food"),
    Category("convenience", "retail_food", ("corner_shop", "kiosk"),
             ("5499",), (1.0,), 0.02, True, "food"),
    Category("fastfood", "food_service", ("burger", "shawarma", "pizza", "canteen"),
             ("5814",), (1.0,), 0.22, False, "food_out"),
    Category("restaurant", "food_service", ("national", "european", "asian", "steak"),
             ("5812",), (1.0,), 0.10, False, "food_out", rural_available=False),
    Category("coffee", "food_service", ("coffee_shop", "bakery"),
             ("5814", "5462"), (0.80, 0.20), 0.12, False, "food_out"),
    Category("delivery", "food_service", ("aggregator", "restaurant_delivery"),
             ("5812", "5814"), (0.55, 0.45), 1.0, False, "food_out", rural_available=False),

    # --- здоровье ---
    Category("pharmacy", "health", ("chain_pharmacy", "hospital_pharmacy"),
             ("5912",), (1.0,), 0.10, True, "health", hours=(0, 24)),
    Category("medical", "health", ("clinic", "dentist", "private_doctor"),
             ("8011", "8021", "8062"), (0.55, 0.25, 0.20), 0.06, True, "health", rural_available=False),
    Category("lab", "health", ("laboratory", "diagnostics"),
             ("8071",), (1.0,), 0.30, True, "health", rural_available=False),

    # --- транспорт ---
    Category("fuel", "transport", ("station", "gas"),
             ("5541", "5542"), (0.45, 0.55), 0.03, True, "commute", hours=(0, 24)),
    Category("car_service", "transport", ("repair", "tyres", "car_wash"),
             ("7538", "7534", "7542"), (0.50, 0.20, 0.30), 0.02, False, "car"),
    Category("parking", "transport", ("street", "mall", "airport"),
             ("7523",), (1.0,), 0.55, False, "commute", rural_available=False),
    Category("taxi", "transport", ("aggregator", "local"),
             ("4121",), (1.0,), 0.92, False, "commute"),
    Category("transit", "transport", ("bus", "metro", "tram"),
             ("4111", "4131"), (0.55, 0.45), 0.45, False, "commute", rural_available=False),
    Category("car_rental", "transport", ("rental",),
             ("7512",), (1.0,), 0.80, False, "travel", rural_available=False),
    Category("micromobility", "transport", ("scooter", "bike"),
             ("4121",), (1.0,), 1.0, False, "commute", rural_available=False),

    # --- поездки ---
    Category("airline", "travel", ("carrier", "agency"),
             ("4511", "4722"), (0.70, 0.30), 0.92, False, "travel"),
    Category("railway", "travel", ("rail",),
             ("4112",), (1.0,), 0.75, False, "travel"),
    Category("hotel", "travel", ("hotel", "hostel", "apartment"),
             ("7011",), (1.0,), 0.60, False, "travel"),
    Category("travel", "travel", ("tour_operator", "agency"),
             ("4722",), (1.0,), 0.65, False, "travel"),

    # --- покупки ---
    Category("clothing", "retail_goods", ("fashion", "sportswear", "kidswear"),
             ("5651", "5691", "5641"), (0.45, 0.35, 0.20), 0.30, False, "clothing"),
    Category("shoes", "retail_goods", ("shoes",),
             ("5661",), (1.0,), 0.28, False, "clothing"),
    Category("home_goods", "retail_goods", ("household", "tools", "decor"),
             ("5200", "5251", "5719"), (0.40, 0.30, 0.30), 0.25, False, "home"),
    Category("furniture", "retail_goods", ("furniture",),
             ("5712",), (1.0,), 0.22, False, "home", rural_available=False),
    Category("electronics", "retail_goods", ("phones", "computers", "gadgets"),
             ("5732", "5734"), (0.65, 0.35), 0.45, False, "electronics"),
    Category("appliances", "retail_goods", ("large_appliances",),
             ("5722",), (1.0,), 0.38, False, "home", rural_available=False),

    # --- регулярные платежи ---
    Category("telecom", "services", ("mobile",),
             ("4814",), (1.0,), 0.95, True, "bills", hours=(0, 24)),
    Category("internet", "services", ("home_internet", "tv"),
             ("4899",), (1.0,), 0.95, True, "bills", hours=(0, 24)),
    Category("subscription", "services", ("streaming", "music", "cloud", "software"),
             ("5815", "5817", "5818"), (0.45, 0.30, 0.25), 1.0, False, "bills", hours=(0, 24)),
    Category("utilities", "services", ("electricity", "water", "heating", "gas_supply"),
             ("4900",), (1.0,), 0.80, True, "bills", hours=(0, 24)),

    # --- дети и образование ---
    Category("education", "education", ("courses", "university", "school"),
             ("8220", "8299"), (0.40, 0.60), 0.55, False, "education"),
    Category("kids", "education", ("toys", "kindergarten", "kids_club"),
             ("5945", "8351"), (0.60, 0.40), 0.30, False, "kids"),
    Category("books", "education", ("bookstore", "stationery"),
             ("5942", "5943"), (0.55, 0.45), 0.40, False, "education"),

    # --- досуг и красота ---
    Category("entertainment", "leisure", ("park", "club", "event"),
             ("7996", "7999"), (0.45, 0.55), 0.35, False, "leisure"),
    Category("cinema", "leisure", ("cinema",),
             ("7832",), (1.0,), 0.55, False, "leisure", rural_available=False),
    Category("sports", "leisure", ("gym", "pool", "sport_shop"),
             ("7997", "5941"), (0.65, 0.35), 0.25, False, "leisure"),
    Category("beauty", "leisure", ("salon", "barber", "spa"),
             ("7230", "7298"), (0.70, 0.30), 0.05, False, "beauty"),
    Category("cosmetics", "leisure", ("cosmetics_shop", "perfume"),
             ("5977",), (1.0,), 0.40, False, "beauty"),

    # --- маркетплейсы ---
    Category("marketplace", "ecommerce", ("marketplace",),
             ("5399", "5999"), (0.55, 0.45), 1.0, False, "shopping"),
    Category("ecom", "ecommerce", ("online_shop",),
             ("5999",), (1.0,), 1.0, False, "shopping"),

    # --- государство и финансы ---
    Category("government", "government", ("service_centre", "notary"),
             ("9399",), (1.0,), 0.70, True, "government", hours=(9, 18)),
    Category("fines", "government", ("traffic_fine", "other_fine"),
             ("9222",), (1.0,), 0.85, True, "government", hours=(0, 24)),
    Category("taxes", "government", ("tax",),
             ("9311",), (1.0,), 0.85, True, "government", hours=(0, 24)),
    Category("financial", "financial", ("insurance_agent", "money_service"),
             ("6300", "6051"), (0.55, 0.45), 0.60, False, "financial"),
    Category("charity", "financial", ("charity",),
             ("8398",), (1.0,), 0.85, False, "charity"),

    # --- прочее ---
    Category("pets", "retail_goods", ("pet_shop", "vet"),
             ("5995", "0742"), (0.70, 0.30), 0.25, False, "pets"),
    Category("tobacco", "retail_goods", ("tobacco",),
             ("5993",), (1.0,), 0.02, False, "tobacco"),
    Category("gambling", "leisure", ("betting",),
             ("7995",), (1.0,), 0.95, False, "gambling", rural_available=False),
)

CATEGORY_BY_NAME = {item.name: item for item in CATEGORIES}

CATEGORY_NAMES = tuple(item.name for item in CATEGORIES)

SECTORS = tuple(sorted({item.sector for item in CATEGORIES}))

NEEDS = tuple(sorted({item.need for item in CATEGORIES}))

ALL_MCCS = tuple(sorted({mcc for item in CATEGORIES for mcc in item.mccs}))

# MCC служебных зачислений и снятий.
MCC_SALARY = "6012"
MCC_TRANSFER = "4829"
MCC_CASH = "6011"
MCC_LOAN = "6012"
MCC_DEPOSIT = "6012"
MCC_FEE = "6012"

SERVICE_MCCS = (MCC_SALARY, MCC_TRANSFER, MCC_CASH)


# ------------------------------------------------------------
# КАНАЛЫ ОПЕРАЦИЙ
# ------------------------------------------------------------

OPERATION_CHANNELS = ("pos", "ecom", "atm", "app", "branch", "qr", "system", "partner_pos")

DECLINE_REASONS = (
    "insufficient_funds",
    "card_blocked",
    "limit_exceeded",
    "antifraud_hold",
    "technical_error",
    "wrong_details",
    "expired_card",
)

ERROR_CODES = ("E100", "E205", "E301", "E402", "E403", "E500", "E503", "E777")


# ------------------------------------------------------------
# КОММУНИКАЦИИ
# ------------------------------------------------------------

COMM_CHANNELS = ("call", "sms", "push", "email")

DELIVERY_RATE = {"call": 0.062, "sms": 0.813, "push": 0.426, "email": 0.55}

PURPOSES = ("offer", "service", "collection", "security", "survey", "winback")


@dataclass(frozen=True)
class Campaign:
    code: str
    purpose: str
    family: str | None
    channels: tuple
    templates: tuple


CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign("CL_OFFER", "offer", "cash_loan", ("sms", "push", "call"),
             ("CL_BASE", "CL_PRIME_RATE", "CL_TOPUP", "CL_SUMMER", "CL_WINTER", "CL_PARTNER")),
    Campaign("CC_OFFER", "offer", "credit_card", ("push", "sms", "call"),
             ("CC_OZEN_BASE", "CC_LIMIT_UP", "CC_TRAVEL", "CC_PARTNER_RETAIL")),
    Campaign("INST_OFFER", "offer", "installment", ("push", "sms"),
             ("INST_PARTNER", "INST_0024", "INST_TOUR")),
    Campaign("DEP_OFFER", "offer", "deposit", ("push", "sms", "email"),
             ("DEP_STANDARD", "DEP_MAX_RATE", "DEP_PROMO", "DEP_PENSION")),
    Campaign("CERT_OFFER", "offer", "deposit_certificate", ("push", "email"),
             ("CERT_STANDARD", "CERT_FLEX", "CERT_PRIME")),
    Campaign("BOND_OFFER", "offer", "bonds", ("push", "email"),
             ("BOND_USD", "BOND_DISCOUNT")),
    Campaign("DC_OFFER", "offer", "debit_card", ("push", "sms"),
             ("DC_ARNA", "DC_HOME", "DC_ASPAN", "DC_ALEM", "DC_TAN")),
    Campaign("INS_OFFER", "offer", "insurance", ("push", "call"),
             ("INS_TRAVEL", "INS_FAMILY", "INS_PROPERTY", "INS_SICK_LEAVE")),
    Campaign("REFIN_OFFER", "offer", "refinance", ("sms", "push", "call"),
             ("REFIN_BASE", "REFIN_LOWER_PAYMENT")),
    Campaign("CASHBACK", "offer", None, ("push",),
             ("CB_MONTHLY", "CB_CATEGORY_FOOD", "CB_CATEGORY_FUEL", "CB_PARTNER")),
    Campaign("PAYMENT_REMINDER", "collection", None, ("sms", "push", "call"),
             ("PR_DUE_SOON", "PR_DUE_TODAY", "PR_OVERDUE_SOFT", "PR_OVERDUE_HARD", "PR_RESTRUCTURE")),
    Campaign("SECURITY", "security", None, ("sms", "push", "call"),
             ("SEC_LOGIN_ALERT", "SEC_CARD_BLOCK", "SEC_FRAUD_WARN", "SEC_CONFIRM_REQUEST")),
    Campaign("SERVICE", "service", None, ("push", "sms", "email"),
             ("SRV_APP_UPDATE", "SRV_BRANCH_INFO", "SRV_TARIFF_CHANGE", "SRV_DOC_EXPIRY",
              "SRV_OWNER_CHANGE", "SRV_MIGRATION_NOTICE")),
    Campaign("NPS", "survey", None, ("push", "email"),
             ("NPS_APP", "NPS_BRANCH", "NPS_SUPPORT")),
    Campaign("WINBACK", "winback", None, ("sms", "push", "call"),
             ("WB_MISS_YOU", "WB_BONUS_BACK", "WB_NEW_APP")),
)

CAMPAIGN_BY_CODE = {item.code: item for item in CAMPAIGNS}

CAMPAIGNS_BY_FAMILY: dict[str, tuple] = {}
for _campaign in CAMPAIGNS:
    if _campaign.family:
        CAMPAIGNS_BY_FAMILY.setdefault(_campaign.family, ())
        CAMPAIGNS_BY_FAMILY[_campaign.family] += (_campaign,)


# ------------------------------------------------------------
# ПРИЛОЖЕНИЕ
# ------------------------------------------------------------

SCREEN_HOME = "s_000_home"
SCREEN_OFFERS = "s_001_for_me"

BROWSE_SCREENS: dict[str, tuple] = {
    "home": ("s_000_home", "s_010_balance", "s_011_history", "s_012_analytics"),
    "cards": ("s_100_cards", "s_101_card_detail", "s_102_card_limits", "s_103_card_pin", "s_104_card_order"),
    "transfers": ("s_200_transfers", "s_201_transfer_phone", "s_202_transfer_card",
                  "s_203_transfer_confirm", "s_204_transfer_own", "s_205_transfer_abroad"),
    "payments": ("s_300_payments", "s_301_payment_utility", "s_302_payment_mobile",
                 "s_303_payment_fine", "s_304_payment_tax", "s_305_payment_qr"),
    "loans": ("s_400_loans", "s_401_loan_calc", "s_402_loan_terms", "s_403_loan_schedule",
              "s_404_loan_repay"),
    "deposits": ("s_500_deposits", "s_501_deposit_calc", "s_502_deposit_terms", "s_503_deposit_detail"),
    "insurance": ("s_600_insurance", "s_601_insurance_terms"),
    "market": ("s_700_market", "s_701_market_item", "s_702_market_cart"),
    "profile": ("s_800_profile", "s_801_settings", "s_802_documents", "s_803_consents"),
    "support": ("s_900_support", "s_901_chat", "s_902_faq", "s_903_dispute"),
    "invest": ("s_a10_invest", "s_a11_bond_detail", "s_a12_certificate"),
}

FUNNEL_STAGES = ("view", "application", "kyc", "approved", "rejected")

FUNNEL_SCREENS = {
    "view": "s_b00_offer_view",
    "application": "s_b01_application_form",
    "kyc": "s_b02_kyc_check",
    "approved": "s_b03_success",
    "rejected": "s_b04_reject",
}

DOMAIN_OPERATIONS: dict[str, tuple] = {
    "auth": ("login", "logout", "biometry_login", "pin_change", "device_bind"),
    "cards": ("card_view", "card_block", "card_unblock", "limit_change", "pin_reset", "card_order", "card_reissue"),
    "transfers": ("transfer_phone", "transfer_card", "transfer_own", "transfer_template", "transfer_abroad"),
    "payments": ("pay_utility", "pay_mobile", "pay_internet", "pay_fine", "pay_tax", "pay_qr"),
    "loans": ("loan_view", "loan_schedule", "loan_calc", "loan_repay", "loan_early_repay", "loan_statement"),
    "deposits": ("deposit_view", "deposit_open", "deposit_topup", "deposit_withdraw", "deposit_close"),
    "market": ("market_browse", "market_order", "market_return"),
    "profile": ("profile_view", "profile_edit", "consent_change", "statement_order"),
    "support": ("chat_open", "callback_request", "complaint", "dispute_open"),
    "invest": ("bond_buy", "certificate_open", "certificate_close"),
}

APP_DOMAINS = tuple(DOMAIN_OPERATIONS)

DOMAIN_ADOPTION = {
    "auth": 0.849,
    "cards": 0.77,
    "transfers": 0.87,
    "loans": 0.72,
    "payments": 0.27,
    "deposits": 0.185,
    "market": 0.140,
    "profile": 0.62,
    "support": 0.120,
    "invest": 0.035,
    # Раздел страхования существовал в экранах и семействах,
    # но принять его было нельзя: клиент попадал туда мимо
    # правил.
    "insurance": 0.155,
}

DOMAIN_FAMILY = {
    "loans": "cash_loan",
    "deposits": "deposit",
    "insurance": "insurance",
    "cards": "credit_card",
    "invest": "deposit_certificate",
}

OPERATION_STATUSES = ("success", "failed", "cancelled")

DEVICE_TYPES = ("android_phone", "ios_phone", "android_tablet", "ios_tablet", "web")

DEVICE_WEIGHTS = (0.56, 0.33, 0.04, 0.03, 0.04)


# ------------------------------------------------------------
# БАННЕРЫ
# ------------------------------------------------------------

BANNER_SLOTS = ("main_top", "for_me_1", "for_me_2", "for_me_3",
                "cards_bottom", "payments_bottom", "story_1", "story_2")

BANNER_SLOT_WEIGHTS = (0.26, 0.18, 0.14, 0.09, 0.09, 0.09, 0.09, 0.06)

BANNER_OFFERS = ("cash_loan", "credit_card", "installment", "deposit", "deposit_certificate",
                 "debit_card", "insurance", "cashback", "market_promo", "referral", "bonds")

BANNER_OFFER_FAMILY = {
    "cash_loan": "cash_loan",
    "credit_card": "credit_card",
    "installment": "installment",
    "deposit": "deposit",
    "deposit_certificate": "deposit_certificate",
    "debit_card": "debit_card",
    "insurance": "insurance",
    "bonds": "bonds",
    "cashback": None,
    "market_promo": None,
    "referral": None,
}


# ------------------------------------------------------------
# ПОДДЕРЖКА
# ------------------------------------------------------------

SUPPORT_CHANNELS = ("chat", "call_center", "branch", "email")

SUPPORT_TOPICS = (
    "operation_question",
    "card_block",
    "dispute",
    "fraud_report",
    "app_error",
    "loan_payment",
    "loan_restructure",
    "data_change",
    "complaint",
    "statement_request",
    "product_question",
    "deposit_question",
)

SUPPORT_RESOLUTIONS = (
    "explained",
    "card_unblocked",
    "card_reissued",
    "chargeback_started",
    "refund_issued",
    "record_corrected",
    "escalated",
    "declined",
    "callback_scheduled",
    "document_sent",
)

CASE_STATUSES = ("open", "in_progress", "waiting_client", "resolved", "closed")


# ------------------------------------------------------------
# ПРОФИЛЬ
# ------------------------------------------------------------

PROFILE_CHANGE_SOURCES = ("client", "branch", "application", "external_registry", "call_center")

PROFILE_TRACKED_FIELDS = (
    "family_status",
    "children",
    "education",
    "region",
    "city",
    "housing_type",
    "income_type",
    "declared_income",
    "industry",
    "income_day",
    "consent_marketing",
)


__all__ = [
    "ALL_MCCS",
    "APP_DOMAINS",
    "BANNER_OFFERS",
    "BANNER_OFFER_FAMILY",
    "BANNER_SLOTS",
    "BANNER_SLOT_WEIGHTS",
    "BROWSE_SCREENS",
    "CAMPAIGNS",
    "CAMPAIGNS_BY_FAMILY",
    "CAMPAIGN_BY_CODE",
    "CASE_STATUSES",
    "CATEGORIES",
    "CATEGORY_BY_NAME",
    "CATEGORY_NAMES",
    "COMM_CHANNELS",
    "Campaign",
    "Category",
    "DECLINE_REASONS",
    "DELIVERY_RATE",
    "DEVICE_TYPES",
    "DEVICE_WEIGHTS",
    "DOMAIN_ADOPTION",
    "DOMAIN_FAMILY",
    "DOMAIN_OPERATIONS",
    "ERROR_CODES",
    "FUNNEL_SCREENS",
    "FUNNEL_STAGES",
    "MCC_CASH",
    "MCC_SALARY",
    "MCC_TRANSFER",
    "NEEDS",
    "OPERATION_CHANNELS",
    "OPERATION_STATUSES",
    "PROFILE_CHANGE_SOURCES",
    "PROFILE_TRACKED_FIELDS",
    "PURPOSES",
    "SCREEN_HOME",
    "SCREEN_OFFERS",
    "SECTORS",
    "SERVICE_MCCS",
    "SUPPORT_CHANNELS",
    "SUPPORT_RESOLUTIONS",
    "SUPPORT_TOPICS",
]
