from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ============================================================
# ПРОЕКТ
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"

PRODUCT_TIMELINE_PATH = REFERENCE_DIR / "home_product_timeline.yaml"

GENERATOR_VERSION = "3.0"
SCHEMA_VERSION = 3

SEED = 42


# ============================================================
# ГОРИЗОНТ
# ============================================================
#
# Лента хранит фактические события всего окна. Ни cutoff,
# ни меток здесь нет: выбор среза принадлежит препроцессингу.
# ============================================================

HISTORY_START = datetime(2024, 6, 1)

HISTORY_END = datetime(2026, 9, 1)

# Реестр договоров старше окна наблюдения.
REGISTRY_START = datetime(2018, 1, 1)


# ============================================================
# ИСТОЧНИКИ
# ============================================================
#
# Источник это система банка, которая записала факт.
# До availability источник не существует, и событий в нём нет.
# ============================================================

SOURCE_AVAILABILITY: dict[str, datetime] = {
    "profile": HISTORY_START,
    "applications": HISTORY_START,
    "product_events": REGISTRY_START,
    "loans": HISTORY_START,
    "transactions": HISTORY_START,
    "antifraud": datetime(2025, 1, 15),
    "communications": datetime(2024, 12, 22),
    "banners": datetime(2024, 8, 1),
    "app_screens": datetime(2024, 12, 16),
    "app_operations": HISTORY_START,
    "support": datetime(2025, 3, 1),
}

SOURCES = tuple(SOURCE_AVAILABILITY)


# ============================================================
# КОНВЕРТ
# ============================================================

PRECISION_SECOND = "second"
PRECISION_MINUTE = "minute"
PRECISION_DAY = "day"

# Точность month не встречается ни в одном источнике.
TIME_PRECISIONS = (PRECISION_SECOND, PRECISION_MINUTE, PRECISION_DAY)

# Точность времени источника: у витрины кредитного обслуживания
# времени нет вовсе, у коммуникаций оно округлено до минуты.
SOURCE_PRECISION: dict[str, str] = {
    "profile": PRECISION_MINUTE,
    "applications": PRECISION_MINUTE,
    "product_events": PRECISION_SECOND,
    "loans": PRECISION_DAY,
    "transactions": PRECISION_SECOND,
    "antifraud": PRECISION_SECOND,
    "communications": PRECISION_MINUTE,
    "banners": PRECISION_SECOND,
    "app_screens": PRECISION_SECOND,
    "app_operations": PRECISION_SECOND,
    "support": PRECISION_MINUTE,
}

INITIATOR_CLIENT = "client"
INITIATOR_BANK = "bank_employee"
INITIATOR_SYSTEM = "system"
INITIATOR_EXTERNAL = "external_source"

CHANGE_INITIATORS = (
    INITIATOR_CLIENT,
    INITIATOR_BANK,
    INITIATOR_SYSTEM,
    INITIATOR_EXTERNAL,
)

LINK_TYPES = (
    "offer",
    "application",
    "contract",
    "schedule",
    "session",
    "transfer",
    "case",
    "fraud_episode",
    "reversal",
    "refund",
    "chargeback",
    "correction",
    "duplicate",
)


# ============================================================
# ТИПЫ СОБЫТИЙ
# ============================================================
#
# Порядок задаёт приоритет детерминированного tie-break при
# одинаковом event_time и отражает причинность:
#
#   состояние -> заявка -> решение -> договор -> график ->
#   деньги -> антифрод -> контакт -> баннер -> экран ->
#   операция -> обращение
# ============================================================

EVENT_TYPES = (
    "profile_change",
    "application_submitted",
    "application_decision",
    "account_opened",
    "product_opened",
    "contract_terms_changed",
    "product_repriced",
    "product_renewed",
    "product_migrated",
    "product_closed",
    "card_activated",
    "card_blocked",
    "card_unblocked",
    "card_reissued",
    "schedule_created",
    "installment_due",
    "installment_paid",
    "installment_missed",
    "delinquency_registered",
    "arrears_cleared",
    "loan_restructured",
    "early_repayment",
    "loan_closed",
    "loan_disbursement",
    "salary_credit",
    "pension_credit",
    "other_income_credit",
    "transfer_in",
    "p2p_in",
    "cash_deposit",
    "interest_credit",
    "cashback_credit",
    "purchase",
    "bill_payment",
    "cash_withdrawal",
    "transfer_out",
    "p2p_out",
    "loan_payment",
    "deposit_topup",
    "deposit_withdrawal",
    "fee_charge",
    "refund",
    "reversal",
    "chargeback",
    "balance_snapshot",
    "fraud_alert",
    "fraud_decision",
    "communication_sent",
    "banner_shown",
    "banner_clicked",
    "app_screen",
    "app_operation",
    "case_opened",
    "case_updated",
    "case_resolved",
)

EVENT_TYPE_PRIORITY: dict[str, int] = {
    name: index for index, name in enumerate(EVENT_TYPES)
}


# ============================================================
# КАТАЛОГ КЛЮЧЕЙ PAYLOAD
# ============================================================
#
# Контракт данных: для каждого типа события перечислены поля,
# их тип, допустимость null, уровень и смысл. Каталог уезжает
# в манифест и заменяет собой отдельные схемы таблиц.
# ============================================================

LEVEL_CLIENT = "client"
LEVEL_PRODUCT = "product"
LEVEL_CONTRACT = "contract"
LEVEL_OPERATION = "operation"
LEVEL_SESSION = "session"
LEVEL_COMMUNICATION = "communication"
LEVEL_CASE = "case"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    nullable: bool
    level: str
    description: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "level": self.level,
            "description": self.description,
        }


def _f(name: str, dtype: str, nullable: bool, level: str, description: str) -> FieldSpec:
    return FieldSpec(name, dtype, nullable, level, description)


# --- группы полей, общие для семейств событий ---

MONEY_FIELDS: tuple[FieldSpec, ...] = (
    _f("amount", "int", False, LEVEL_OPERATION, "сумма операции в тенге"),
    _f("currency", "str", True, LEVEL_OPERATION, "валюта счёта: всегда KZT"),
    _f("original_amount", "int", True, LEVEL_OPERATION, "сумма в валюте страны покупки"),
    _f("original_currency", "str", True, LEVEL_OPERATION, "валюта страны покупки"),
    _f("direction", "str", False, LEVEL_OPERATION, "debit или credit для счёта клиента"),
    _f("status", "str", False, LEVEL_OPERATION, "approved, declined или reversed"),
    _f("decline_reason", "str", True, LEVEL_OPERATION, "причина отказа, если операция не прошла"),
    _f("channel", "str", True, LEVEL_OPERATION, "канал операции: pos, ecom, atm, app, branch, qr, system"),
    _f("account_id", "str", True, LEVEL_CONTRACT, "счёт клиента, по которому прошли деньги"),
    _f("card_id", "str", True, LEVEL_CONTRACT, "карта, если операция картой"),
    _f("contract_id", "str", True, LEVEL_CONTRACT, "договор, если операция относится к договору"),
    _f("cause_event_id", "str", True, LEVEL_OPERATION, "event_id события-причины, либо null"),
    _f("accrual_period", "str", True, LEVEL_CONTRACT, "период начисления YYYY-MM для периодических сумм"),
    _f("reason", "str", True, LEVEL_OPERATION, "основание операции"),
    _f("merchant_id", "str", True, LEVEL_OPERATION, "сеть или поставщик услуг"),
    _f("outlet_id", "str", True, LEVEL_OPERATION, "торговая точка или онлайн-витрина"),
    _f("merchant_name", "str", True, LEVEL_OPERATION, "имя в терминальной строке"),
    _f("mcc", "str", True, LEVEL_OPERATION, "код категории точки"),
    _f("merchant_city", "str", True, LEVEL_OPERATION, "город точки"),
    _f("merchant_country", "str", True, LEVEL_OPERATION, "страна точки"),
    _f("is_online", "bool", True, LEVEL_OPERATION, "операция без присутствия карты"),
    _f("is_subscription", "bool", True, LEVEL_OPERATION, "регулярное списание подписки"),
    _f("counterparty", "str", True, LEVEL_OPERATION, "устойчивое маскированное имя контрагента"),
    _f("balance_after", "int", True, LEVEL_CONTRACT, "остаток счёта после проводки"),
)

PRODUCT_FIELDS: tuple[FieldSpec, ...] = (
    _f("product_id", "str", False, LEVEL_PRODUCT, "продукт каталога"),
    _f("product_code", "str", False, LEVEL_PRODUCT, "код продукта"),
    _f("product_version", "int", False, LEVEL_PRODUCT, "версия условий на момент договора"),
    _f("tariff_version", "int", False, LEVEL_PRODUCT, "версия тарифа на момент договора"),
    _f("product_family", "str", False, LEVEL_PRODUCT, "семейство продукта"),
    _f("contract_id", "str", True, LEVEL_CONTRACT, "договор"),
    _f("account_id", "str", True, LEVEL_CONTRACT, "счёт договора"),
    _f("card_id", "str", True, LEVEL_CONTRACT, "карта договора"),
    _f("offer_id", "str", True, LEVEL_COMMUNICATION, "предложение, из которого вырос договор"),
    _f("previous_product_id", "str", True, LEVEL_PRODUCT, "продукт, с которого перешёл клиент"),
    _f("migration_reason", "str", True, LEVEL_PRODUCT, "причина перехода"),
    _f("amount_or_limit", "int", True, LEVEL_CONTRACT, "сумма договора или лимит"),
    _f("term", "int", True, LEVEL_CONTRACT, "срок договора в месяцах"),
    _f("rate", "float", True, LEVEL_CONTRACT, "номинальная ставка"),
    _f("reason", "str", True, LEVEL_CONTRACT, "основание события"),
    _f("timestamp_quality", "str", False, LEVEL_CONTRACT, "exact или date_only"),
)

LOAN_FIELDS: tuple[FieldSpec, ...] = (
    _f("contract_id", "str", False, LEVEL_CONTRACT, "кредитный договор"),
    _f("installment_no", "int", True, LEVEL_CONTRACT, "номер платежа в графике"),
    _f("amount_due", "int", True, LEVEL_CONTRACT, "сумма планового платежа"),
    _f("amount_paid", "int", True, LEVEL_CONTRACT, "фактически уплачено"),
    _f("principal_outstanding", "int", True, LEVEL_CONTRACT, "остаток основного долга"),
    _f("days_past_due", "int", True, LEVEL_CONTRACT, "дней просрочки"),
    _f("due_date", "str", True, LEVEL_CONTRACT, "плановая дата платежа"),
    _f("cause_event_id", "str", True, LEVEL_CONTRACT, "event_id события-причины, либо null"),
    _f("reason", "str", True, LEVEL_CONTRACT, "основание события"),
)

APPLICATION_FIELDS: tuple[FieldSpec, ...] = (
    _f("application_id", "str", False, LEVEL_OPERATION, "заявка"),
    _f("product_id", "str", False, LEVEL_PRODUCT, "запрошенный продукт"),
    _f("product_code", "str", False, LEVEL_PRODUCT, "код продукта"),
    _f("product_version", "int", False, LEVEL_PRODUCT, "версия условий на момент заявки"),
    _f("offer_id", "str", True, LEVEL_COMMUNICATION, "предложение, из которого выросла заявка"),
    _f("channel", "str", False, LEVEL_OPERATION, "канал подачи"),
    _f("requested_amount", "int", True, LEVEL_OPERATION, "запрошенная сумма"),
    _f("requested_term", "int", True, LEVEL_OPERATION, "запрошенный срок"),
    _f("decision", "str", True, LEVEL_OPERATION, "approved или rejected"),
    _f("reject_reason", "str", True, LEVEL_OPERATION, "причина отказа"),
    _f("approved_amount", "int", True, LEVEL_OPERATION, "одобренная сумма"),
    _f("approved_term", "int", True, LEVEL_OPERATION, "одобренный срок"),
)

EVENT_SPECS: dict[str, dict] = {}


def _spec(event_type: str, source: str, fields: tuple[FieldSpec, ...], description: str) -> None:
    EVENT_SPECS[event_type] = {
        "source": source,
        "description": description,
        "fields": fields,
    }


_spec(
    "profile_change",
    "profile",
    (
        _f("field_name", "str", False, LEVEL_CLIENT, "изменившееся поле профиля"),
        _f("old_value", "str", True, LEVEL_CLIENT, "прежнее значение"),
        _f("new_value", "str", True, LEVEL_CLIENT, "новое значение"),
        _f("change_source", "str", False, LEVEL_CLIENT, "откуда банк узнал: client, branch, application, external_registry"),
        _f("confirmed", "bool", False, LEVEL_CLIENT, "значение подтверждено документом"),
    ),
    "банк узнал об изменении атрибута клиента",
)

_spec("application_submitted", "applications", APPLICATION_FIELDS, "клиент подал заявку")
_spec("application_decision", "applications", APPLICATION_FIELDS, "банк принял решение по заявке")

for _product_event, _text in (
    ("account_opened", "открыт счёт"),
    ("product_opened", "открыт договор по продукту"),
    ("contract_terms_changed", "изменены условия действующего договора"),
    ("product_repriced", "изменён тариф действующего договора"),
    ("product_renewed", "договор пролонгирован на новых условиях"),
    ("product_migrated", "клиент переведён на другой продукт"),
    ("product_closed", "договор закрыт"),
    ("card_activated", "карта активирована"),
    ("card_blocked", "карта заблокирована"),
    ("card_unblocked", "карта разблокирована"),
    ("card_reissued", "карта перевыпущена"),
):
    _spec(_product_event, "product_events", PRODUCT_FIELDS, _text)

for _loan_event, _text in (
    ("schedule_created", "сформирован график платежей"),
    ("installment_due", "наступил срок планового платежа"),
    ("installment_paid", "плановый платёж исполнен"),
    ("installment_missed", "плановый платёж пропущен"),
    ("delinquency_registered", "зарегистрирована просрочка на вехе DPD"),
    ("arrears_cleared", "просрочка погашена"),
    ("loan_restructured", "договор реструктурирован"),
    ("early_repayment", "досрочное погашение"),
    ("loan_closed", "кредитный договор закрыт"),
):
    _spec(_loan_event, "loans", LOAN_FIELDS, _text)

for _money_event, _text in (
    ("purchase", "покупка"),
    ("refund", "возврат покупки"),
    ("reversal", "отмена операции"),
    ("chargeback", "возврат по оспариванию"),
    ("cash_withdrawal", "снятие наличных"),
    ("cash_deposit", "внесение наличных"),
    ("transfer_in", "входящий внешний перевод"),
    ("transfer_out", "исходящий внешний перевод"),
    ("p2p_in", "входящий внутрибанковский перевод"),
    ("p2p_out", "исходящий внутрибанковский перевод"),
    ("salary_credit", "зачисление зарплаты"),
    ("pension_credit", "зачисление пенсии"),
    ("other_income_credit", "зачисление прочего дохода"),
    ("bill_payment", "оплата счёта"),
    ("loan_disbursement", "выдача кредита"),
    ("loan_payment", "платёж по кредиту"),
    ("deposit_topup", "пополнение депозита"),
    ("deposit_withdrawal", "снятие с депозита"),
    ("interest_credit", "начисление процентов"),
    ("fee_charge", "комиссия"),
    ("cashback_credit", "начисление кешбэка"),
    ("balance_snapshot", "остаток счёта на конец месяца"),
):
    _spec(_money_event, "transactions", MONEY_FIELDS, _text)

_spec(
    "fraud_alert",
    "antifraud",
    (
        _f("subject", "str", False, LEVEL_OPERATION, "объект проверки: card, transfer, login"),
        _f("card_id", "str", True, LEVEL_CONTRACT, "карта"),
        _f("account_id", "str", True, LEVEL_CONTRACT, "счёт"),
        _f("score_band", "str", False, LEVEL_OPERATION, "полоса риска: low, medium, high"),
        _f("rule_code", "str", True, LEVEL_OPERATION, "сработавшее правило"),
        _f("cause_event_id", "str", True, LEVEL_OPERATION, "event_id операции, вызвавшей проверку"),
    ),
    "антифрод зафиксировал подозрение",
)

_spec(
    "fraud_decision",
    "antifraud",
    (
        _f("subject", "str", False, LEVEL_OPERATION, "объект проверки"),
        _f("card_id", "str", True, LEVEL_CONTRACT, "карта"),
        _f("account_id", "str", True, LEVEL_CONTRACT, "счёт"),
        _f("decision", "str", False, LEVEL_OPERATION, "monitor, confirm_request или block"),
        _f("resolution", "str", True, LEVEL_OPERATION, "confirmed_by_client, disputed, false_positive"),
        _f("cause_event_id", "str", True, LEVEL_OPERATION, "event_id проверки"),
    ),
    "решение антифрода",
)

_spec(
    "communication_sent",
    "communications",
    (
        _f("channel", "str", False, LEVEL_COMMUNICATION, "call, sms, push или email"),
        _f("template", "str", False, LEVEL_COMMUNICATION, "шаблон сообщения"),
        _f("campaign_code", "str", False, LEVEL_COMMUNICATION, "код кампании"),
        _f("offer_id", "str", True, LEVEL_COMMUNICATION, "предложение"),
        _f("product_id", "str", True, LEVEL_PRODUCT, "продукт предложения"),
        _f("purpose", "str", False, LEVEL_COMMUNICATION, "offer, service, collection, security, survey, winback"),
        _f("delivered", "bool", False, LEVEL_COMMUNICATION, "доставлено ли сообщение"),
        _f("day_of_week", "int", False, LEVEL_COMMUNICATION, "день недели отправки"),
        _f("hour", "int", False, LEVEL_COMMUNICATION, "час отправки"),
    ),
    "банк отправил сообщение",
)

for _banner_event, _text in (("banner_shown", "показ баннера"), ("banner_clicked", "клик по баннеру")):
    _spec(
        _banner_event,
        "banners",
        (
            _f("slot", "str", False, LEVEL_SESSION, "слот в приложении"),
            _f("offer", "str", False, LEVEL_SESSION, "оффер баннера"),
            _f("offer_id", "str", True, LEVEL_COMMUNICATION, "предложение"),
            _f("product_id", "str", True, LEVEL_PRODUCT, "продукт оффера"),
            _f("campaign_code", "str", True, LEVEL_COMMUNICATION, "код кампании"),
        ),
        _text,
    )

_spec(
    "app_screen",
    "app_screens",
    (
        _f("firebase_screen", "str", False, LEVEL_SESSION, "экран приложения"),
        _f("domain", "str", True, LEVEL_SESSION, "раздел приложения"),
        _f("product_id", "str", True, LEVEL_PRODUCT, "продукт раздела"),
        _f("funnel_stage", "str", True, LEVEL_SESSION, "стадия воронки заявки"),
        _f("reject_reason", "str", True, LEVEL_SESSION, "причина отказа на экране отказа"),
    ),
    "экран приложения",
)

_spec(
    "app_operation",
    "app_operations",
    (
        _f("domain", "str", False, LEVEL_SESSION, "домен приложения"),
        _f("operation", "str", False, LEVEL_SESSION, "операция"),
        _f("status", "str", True, LEVEL_SESSION, "success, failed или cancelled"),
        _f("amount", "int", True, LEVEL_SESSION, "сумма операции, если она денежная"),
        _f("error_code", "str", True, LEVEL_SESSION, "код ошибки"),
        _f("device_new", "bool", True, LEVEL_SESSION, "операция с нового устройства"),
    ),
    "операция в приложении",
)

for _case_event, _text in (
    ("case_opened", "обращение открыто"),
    ("case_updated", "обращение обновлено"),
    ("case_resolved", "обращение закрыто"),
):
    _spec(
        _case_event,
        "support",
        (
            _f("case_id", "str", False, LEVEL_CASE, "обращение"),
            _f("channel", "str", False, LEVEL_CASE, "канал обращения"),
            _f("topic", "str", False, LEVEL_CASE, "тема обращения"),
            _f("status", "str", False, LEVEL_CASE, "статус обращения"),
            _f("resolution", "str", True, LEVEL_CASE, "исход обращения"),
            _f("cause_event_id", "str", True, LEVEL_CASE, "event_id события-причины"),
        ),
        _text,
    )


EVENT_TYPE_SOURCE: dict[str, str] = {
    event_type: spec["source"] for event_type, spec in EVENT_SPECS.items()
}

PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    event_type: tuple(field.name for field in spec["fields"])
    for event_type, spec in EVENT_SPECS.items()
}


def key_catalogue() -> dict:
    """
    Каталог ключей payload для манифеста.
    """

    return {
        event_type: {
            "source": spec["source"],
            "description": spec["description"],
            "fields": [field.as_dict() for field in spec["fields"]],
        }
        for event_type, spec in EVENT_SPECS.items()
    }


assert set(EVENT_SPECS) == set(EVENT_TYPES), "каталог ключей и EVENT_TYPES разошлись"
assert set(EVENT_TYPE_SOURCE.values()) <= set(SOURCES), "источник события вне SOURCES"


# ============================================================
# ЗАПРЕЩЁННЫЕ ПОЛЯ RAW
# ============================================================
#
# Готовые ответы и скрытые причины не попадают в наблюдаемые
# данные ни колонкой, ни ключом payload. Проверяется тестом
# и proxy-leak audit.
# ============================================================

FORBIDDEN_RAW_FIELDS: frozenset[str] = frozenset(
    {
        "future_churn",
        "will_default",
        "will_churn",
        "fraud_persona",
        "is_fraudster",
        "scenario",
        "archetype",
        "household_id",
        "relation_type",
        "employer_id",
        "community_id",
        "client_ordinal",
        "stress_episode",
        "stress_level",
        "financial_discipline",
        "risk_tolerance",
        "spending_impulsivity",
        "digital_affinity",
        "price_sensitivity",
        "merchant_loyalty",
        "channel_preferences",
        "mobility",
        "sociality",
        "fraud_vulnerability",
        "credit_appetite",
        "savings_propensity",
        "true_income",
        "hcb_role",
        "activity_mode",
        "pause_reason",
        "clicked",
        "campaign_clicked",
        "life_event",
        "trait_shift",
        "lifecycle_state",
    }
)


# ============================================================
# ПРОФИЛЬ
# ============================================================

PROFILE_FIELDS = (
    "age",
    "gender",
    "family_status",
    "children",
    "education",
    "region",
    "city",
    "housing_type",
    "pensioner",
    "income_type",
    "declared_income",
    "industry",
    "salary_day",
    "relationship_months",
    "contracts_count",
    "active_contracts",
    "holds_credit_card",
    "holds_debit_card",
    "holds_deposit",
    "credit_limit",
    "credit_utilization",
)


# ============================================================
# ПРЕСЕТЫ
# ============================================================

PRESETS = {
    "smoke": 100,
    "check": 300,
    "dev": 10_000,
    "eval": 50_000,
}
