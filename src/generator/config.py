from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ============================================================
# ПРОЕКТ
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "01_raw"

# Справочник фактов о банке лежит в репозитории, а не в data/:
# это рукотворный ВХОД генератора, а не его результат, и чистка
# данных не должна его уносить.
REFERENCE_DIR = BASE_DIR / "reference"

PRODUCT_TIMELINE_PATH = REFERENCE_DIR / "home_product_timeline.json"

# Справочник реальных мерчантов (2ГИС и OpenStreetMap). Тоже
# ВХОД генератора: названия точек берутся отсюда, а не
# выдумываются по слогам. MCC в справочнике нет намеренно —
# его даёт внутренняя категория генератора.
MERCHANT_REFERENCE_PATH = REFERENCE_DIR / "merchants.json"

# Контракт v13: конверт из ЧЕТЫРЁХ колонок — client_id,
# event_time, source, payload, — а тип события лежит внутри
# payload под ключом type. Идентификатора записи и причинных
# ссылок нет. Выгрузка это две таблицы, events.parquet и
# profile.parquet; справочники остались входом генератора, а в
# событие попадают только поля выбранного объекта.
#
# Отличие v13 от v12 одно: event_time выгружается СТРОКОЙ
# ISO 8601 со смещением, а не готовым timestamp. Генератор
# пишет читаемое время так, как его отдала бы банковская
# система, а приводит его к UTC препроцессинг.
GENERATOR_VERSION = "12.0"
SCHEMA_VERSION = 14

SEED = 42


# ============================================================
# ВРЕМЯ СОБЫТИЯ
# ============================================================
#
# У генератора один явно объявленный часовой пояс: время
# Казахстана, UTC+05:00. Пояс задан смещением, а не именем
# зоны, и намеренно: смещение видно в самой строке выгрузки,
# одинаково читается на любой машине и не зависит ни от
# базы часовых поясов системы, ни от перевода стрелок.
#
# Локальное время БЕЗ смещения в выгрузку не попадает никогда:
# по нему нельзя однозначно восстановить момент.
# ============================================================

TIMEZONE = timezone(timedelta(hours=5), "Asia/Almaty")

TIMEZONE_NAME = "Asia/Almaty"


def event_time_text(moment: datetime) -> str:
    """
    Время события строкой ISO 8601 со смещением.

    Миллисекунды пишутся только тогда, когда они у
    события есть: нули в долях секунды говорили бы о точности,
    которой у записи нет.
    """

    stamped = moment.replace(tzinfo=TIMEZONE) if moment.tzinfo is None else moment.astimezone(TIMEZONE)

    if stamped.microsecond:
        return stamped.isoformat(timespec="milliseconds")

    return stamped.isoformat(timespec="seconds")


# ============================================================
# ГОРИЗОНТ
# ============================================================
#
# Лента хранит фактические события всего окна. Ни cutoff,
# ни меток здесь нет: выбор среза принадлежит препроцессингу.
#
# Окно у каждой группы своё и объявлено в DATASETS ниже.
# Перед генерацией группы оно ставится через activate_horizon,
# поэтому модули обязаны читать config.HISTORY_START, а не
# импортировать это имя значением: импортированное значение
# переключения не увидит.
# ============================================================

HISTORY_START = datetime(2024, 1, 1)

HISTORY_END = datetime(2026, 9, 1)

# Реестр договоров старше окна наблюдения.
REGISTRY_START = datetime(2018, 1, 1)

# ГОРИЗОНТ ПЛАНИРОВАНИЯ — до какой даты симуляция строит планы
# клиента: жизненные события, паузы, стресс, мошенничество,
# потоки дохода, подписки, дату прихода в банк и установки
# приложения.
#
# Он НЕ равен концу выгрузки и от него не зависит. Иначе продление
# окна переписывало бы уже случившееся: число планируемых событий
# считалось от длины окна, а их даты раскладывались по ней же, и
# удлинение окна раздвигало даты задним числом.
#
# Теперь окно выгрузки только ОБРЕЗАЕТ ленту. Клиент живёт по
# одному плану, а сколько от этой жизни попадёт в выгрузку —
# решает граница.
#
# Значение выбрано по самому дальнему объявленному окну
# (DATASETS ниже): так плотность событий остаётся той же, что
# была, а не размазывается по лишним годам.
PLANNING_END = datetime(2026, 9, 1)


# ============================================================
# ИСТОЧНИКИ
# ============================================================
#
# Источник это система банка, которая записала факт.
# До availability источник не существует, и событий в нём нет.
#
# None означает «существует с начала наблюдения» и следует за
# горизонтом группы. Остальные даты — настоящие даты запуска
# систем, они от горизонта не зависят.
# ============================================================

SOURCE_LAUNCH: dict[str, datetime | None] = {
    "profile": None,
    "applications": None,
    # Договоры старше окна в выгрузку не попадают, поэтому
    # витрина договоров наблюдается с начала окна, как и
    # остальные.
    "product_events": None,
    "loans": None,
    "transactions": None,
    "antifraud": datetime(2025, 1, 15),
    "communications": datetime(2024, 12, 22),
    "banners": datetime(2024, 8, 1),
    "app_screens": datetime(2024, 12, 16),
    "app_operations": None,
    "support": datetime(2025, 3, 1),
}

# Словарь МЕНЯЕТСЯ НА МЕСТЕ при смене горизонта: так его видят
# и те модули, которые импортировали его именем.
SOURCE_AVAILABILITY: dict[str, datetime] = {}


def _apply_horizon() -> None:
    for source, launch in SOURCE_LAUNCH.items():
        SOURCE_AVAILABILITY[source] = HISTORY_START if launch is None else launch


_apply_horizon()

SOURCES = tuple(SOURCE_LAUNCH)


def activate_horizon(start: datetime, end: datetime) -> None:
    """
    Ставит окно наблюдения перед генерацией группы.
    """

    global HISTORY_START, HISTORY_END

    if start >= end:
        raise ValueError(f"горизонт пуст: начало {start} не раньше конца {end}")

    if start < REGISTRY_START:
        raise ValueError(f"история начинается раньше реестра договоров: {start} < {REGISTRY_START}")

    # Планы строятся до PLANNING_END; окно, выходящее за него,
    # получило бы обрезанную жизнь клиента вместо продолжения.
    if end > PLANNING_END:
        raise ValueError(
            f"конец окна {end} дальше горизонта планирования {PLANNING_END}: "
            "поднимите PLANNING_END, иначе планы кончатся раньше выгрузки"
        )

    HISTORY_START = start
    HISTORY_END = end

    _apply_horizon()


# ============================================================
# КОНВЕРТ
# ============================================================
#
# Конверт состоит из четырёх колонок: client_id, event_time,
# source, payload. Что именно произошло, говорит ключ type
# внутри payload: отдельной колонки у типа события нет.
# Идентификатора записи в конверте нет, как нет ни точности
# времени, ни версии, ни метки связи.
#
# event_time — точное время события строкой ISO 8601 со
# смещением часового пояса. Деловая связь живёт
# деловыми ключами payload: contract_id, account_id, card_id,
# application_id, case_id, offer_id, session_id, transfer_id и
# merchant_id называют сущность, частью которой запись
# является. Ссылки на событие-причину нет вовсе.


# ============================================================
# ДЕЙСТВИЕ КЛИЕНТА
# ============================================================
#
# Клиент действовал в этом месяце или молчал — вопрос о типе
# события, а не о метке в конверте. Метки инициатора больше
# нет: покупку делает клиент, начисление процентов — банк, и
# различить их можно по самому типу.
#
# Сюда входит только то, что клиент делает САМ. Зачисление
# зарплаты, списание по подписке, плановый платёж по графику и
# любые решения банка — не действия клиента, даже когда они
# касаются его денег.

CLIENT_ACTION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # деньги, которые клиент двигает сам
        "purchase",
        "cash_withdrawal",
        "cash_deposit",
        "transfer_out",
        "p2p_out",
        "deposit_topup",
        "deposit_withdrawal",
        "bill_payment",
        "early_repayment",
        # приложение
        "app_screen",
        "app_operation",
        "banner_clicked",
        # обращения и заявки
        "application_submitted",
        "case_opened",
    }
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
# их тип, допустимость null, уровень и смысл. Каталог статичен
# и живёт здесь: и генератор, и препроцессинг читают его из
# кода, а не из копии в манифесте каждой выгрузки.
# ============================================================

LEVEL_EVENT = "event"
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
    _f("transfer_id", "str", True, LEVEL_OPERATION, "перевод, если у операции есть вторая нога"),
    _f("session_id", "str", True, LEVEL_SESSION, "сессия приложения, если операция сделана в нём"),
    _f("accrual_period", "str", True, LEVEL_CONTRACT, "период начисления YYYY-MM для периодических сумм"),
    _f("reason", "str", True, LEVEL_OPERATION, "основание операции"),
    # Точка описывается шестью полями и только ими: справочник
    # мерчантов в событие не копируется. Сектор, подкатегория,
    # район, часы и ценовой сегмент остаются внутри генератора.
    _f("merchant_id", "str", True, LEVEL_OPERATION, "сеть или поставщик услуг"),
    _f("merchant_name", "str", True, LEVEL_OPERATION, "имя в терминальной строке"),
    _f("merchant_category", "str", True, LEVEL_OPERATION, "категория точки"),
    _f("merchant_city", "str", True, LEVEL_OPERATION, "город операции"),
    _f("merchant_country", "str", True, LEVEL_OPERATION, "страна точки"),
    _f("mcc", "str", True, LEVEL_OPERATION, "код категории точки"),
    _f("is_online", "bool", True, LEVEL_OPERATION, "операция без присутствия карты"),
    _f("is_subscription", "bool", True, LEVEL_OPERATION, "регулярное списание подписки"),
    _f("counterparty", "str", True, LEVEL_OPERATION, "устойчивое маскированное имя контрагента"),
    _f("balance_after", "int", True, LEVEL_CONTRACT, "остаток счёта после проводки"),
)

# Продукт назван идентификатором каталога и договором, и только
# ими. Код, версия, тариф, семейство и название — это строка
# справочника продуктов: копировать её в каждое событие незачем,
# а product_name и product_type вдобавок повторяют product_id.
PRODUCT_FIELDS: tuple[FieldSpec, ...] = (
    _f("product_id", "str", False, LEVEL_PRODUCT, "продукт каталога"),
    _f("contract_id", "str", True, LEVEL_CONTRACT, "договор"),
    _f("account_id", "str", True, LEVEL_CONTRACT, "счёт договора"),
    _f("card_id", "str", True, LEVEL_CONTRACT, "карта договора"),
    _f("offer_id", "str", True, LEVEL_COMMUNICATION, "предложение, из которого вырос договор"),
    _f("application_id", "str", True, LEVEL_OPERATION, "заявка, по которой открыт договор"),
    _f("previous_product_id", "str", True, LEVEL_PRODUCT, "продукт, с которого перешёл клиент"),
    _f("migration_reason", "str", True, LEVEL_PRODUCT, "причина перехода"),
    _f("reason", "str", True, LEVEL_CONTRACT, "основание события"),
)

# Условия договора записываются только там, где банк их назначил
# или изменил: при открытии и при смене условий. В остальных
# продуктовых событиях сумма, срок и ставка не повторяются —
# действующие условия задаёт последнее такое событие.
PRODUCT_TERMS_FIELDS: tuple[FieldSpec, ...] = PRODUCT_FIELDS + (
    _f("amount_or_limit", "int", True, LEVEL_CONTRACT, "сумма договора или лимит"),
    _f("term", "int", True, LEVEL_CONTRACT, "срок договора в месяцах"),
    _f("rate", "float", True, LEVEL_CONTRACT, "номинальная ставка"),
)

LOAN_FIELDS: tuple[FieldSpec, ...] = (
    _f("contract_id", "str", False, LEVEL_CONTRACT, "кредитный договор"),
    _f("installment_no", "int", True, LEVEL_CONTRACT, "номер платежа в графике"),
    _f("amount_due", "int", True, LEVEL_CONTRACT, "сумма планового платежа"),
    _f("amount_paid", "int", True, LEVEL_CONTRACT, "фактически уплачено"),
    _f("principal_outstanding", "int", True, LEVEL_CONTRACT, "остаток основного долга"),
    _f("days_past_due", "int", True, LEVEL_CONTRACT, "дней просрочки"),
    _f("due_date", "str", True, LEVEL_CONTRACT, "плановая дата платежа"),
    _f("reason", "str", True, LEVEL_CONTRACT, "основание события"),
)

# Ключи условий договора: чем набор с условиями отличается от
# набора без них. Нужны тому, кто шлёт один payload нескольким
# событиям: там, где условий не назначают, их надо снять.
CONTRACT_TERMS_KEYS: frozenset[str] = frozenset(
    item.name for item in PRODUCT_TERMS_FIELDS
) - frozenset(item.name for item in PRODUCT_FIELDS)


APPLICATION_FIELDS: tuple[FieldSpec, ...] = (
    _f("application_id", "str", False, LEVEL_OPERATION, "заявка"),
    _f("product_id", "str", False, LEVEL_PRODUCT, "запрошенный продукт"),
    _f("offer_id", "str", True, LEVEL_COMMUNICATION, "предложение, из которого выросла заявка"),
    _f("channel", "str", False, LEVEL_OPERATION, "канал подачи"),
    _f("requested_amount", "int", True, LEVEL_OPERATION, "запрошенная сумма"),
    _f("requested_term", "int", True, LEVEL_OPERATION, "запрошенный срок"),
    _f("decision", "str", True, LEVEL_OPERATION, "approved или rejected"),
    _f("reject_reason", "str", True, LEVEL_OPERATION, "причина отказа"),
    _f("approved_amount", "int", True, LEVEL_OPERATION, "одобренная сумма"),
    _f("approved_term", "int", True, LEVEL_OPERATION, "одобренный срок"),
)

# Снимок остатка — НЕ операция. У него нет суммы, направления и
# статуса: банк ничего не проводил, он сообщил, сколько лежит на
# счёте на конец периода. Раньше снимок шёл общим денежным
# набором и выглядел зачислением: amount равнялся остатку,
# direction был credit, status — approved, а currency при этом
# оставалась пустой. Читатель ленты видел несуществующий приход.
SNAPSHOT_FIELDS: tuple[FieldSpec, ...] = (
    _f("account_id", "str", False, LEVEL_CONTRACT, "счёт, остаток которого зафиксирован"),
    _f("balance_after", "int", False, LEVEL_CONTRACT, "остаток счёта на конец периода"),
    _f("currency", "str", False, LEVEL_OPERATION, "валюта счёта: всегда KZT"),
    _f("accrual_period", "str", False, LEVEL_CONTRACT, "период YYYY-MM, на конец которого снят остаток"),
)

EVENT_SPECS: dict[str, dict] = {}

# Тип события это ПЕРВЫЙ ключ payload каждого события: колонки
# event_type в конверте нет, и что произошло, говорит сама
# запись. Ключ обязателен везде и добавляется каталогу здесь, а
# не переписывается в каждом объявлении.
TYPE_FIELD: FieldSpec = _f("type", "str", False, LEVEL_EVENT, "что произошло")


def _spec(event_type: str, source: str, fields: tuple[FieldSpec, ...], description: str) -> None:
    EVENT_SPECS[event_type] = {
        "source": source,
        "description": description,
        "fields": (TYPE_FIELD,) + fields,
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

# Открытие и смена условий: сумма, срок и ставка записываются.
for _terms_event, _text in (
    ("account_opened", "открыт счёт"),
    ("product_opened", "открыт договор по продукту"),
    ("contract_terms_changed", "изменены условия действующего договора"),
    ("product_repriced", "изменён тариф действующего договора"),
    ("product_renewed", "договор пролонгирован на новых условиях"),
    ("product_migrated", "клиент переведён на другой продукт"),
):
    _spec(_terms_event, "product_events", PRODUCT_TERMS_FIELDS, _text)

# Остальные продуктовые события условий не назначают: они
# называют договор и продукт.
for _product_event, _text in (
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
):
    _spec(_money_event, "transactions", MONEY_FIELDS, _text)

_spec("balance_snapshot", "transactions", SNAPSHOT_FIELDS, "остаток счёта на конец месяца")

_spec(
    "fraud_alert",
    "antifraud",
    (
        _f("subject", "str", False, LEVEL_OPERATION, "объект проверки: card, transfer"),
        _f("card_id", "str", True, LEVEL_CONTRACT, "карта"),
        _f("account_id", "str", True, LEVEL_CONTRACT, "счёт"),
        _f("score_band", "str", False, LEVEL_OPERATION, "полоса риска: low, medium, high"),
        _f("rule_code", "str", True, LEVEL_OPERATION, "сработавшее правило"),
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
        _f("delivered", "bool", True, LEVEL_COMMUNICATION,
           "true — доставка подтверждена, false — недоставка подтверждена, null — результат неизвестен"),
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
            _f("session_id", "str", True, LEVEL_SESSION, "сессия приложения, в которой показан баннер"),
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
        _f("session_id", "str", True, LEVEL_SESSION, "сессия приложения, если экран её часть"),
        _f("application_id", "str", True, LEVEL_OPERATION, "заявка, если экран её воронка"),
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
        _f("session_id", "str", True, LEVEL_SESSION, "сессия приложения"),
        _f("contract_id", "str", True, LEVEL_CONTRACT, "договор, если операция относится к договору"),
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
        ),
        _text,
    )


EVENT_TYPE_SOURCE: dict[str, str] = {
    event_type: spec["source"] for event_type, spec in EVENT_SPECS.items()
}

PAYLOAD_FIELDS: dict[str, frozenset[str]] = {
    event_type: frozenset(field.name for field in spec["fields"])
    for event_type, spec in EVENT_SPECS.items()
}

# Ключи, без которых события не бывает. Каталог уже объявляет
# это полем nullable; карта нужна, чтобы проверка на выходе
# генератора стоила один поиск в множестве, а не обход схемы.
PAYLOAD_REQUIRED: dict[str, frozenset[str]] = {
    event_type: frozenset(field.name for field in spec["fields"] if not field.nullable)
    for event_type, spec in EVENT_SPECS.items()
}


def key_catalogue() -> dict:
    """
    Каталог ключей payload: тип события -> источник, описание
    и поля.
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
# СМЕНА СХЕМЫ В СЕРЕДИНЕ ИСТОРИИ
# ============================================================
#
# Поле начинает собираться с определённой даты: до неё оно
# пусто по известной причине, а не по неизвестной. Таблица
# статична, как и каталог ключей: генератор её применяет,
# препроцессинг по ней объясняет пустые ячейки.
# ============================================================

SCHEMA_CHANGES: tuple[dict, ...] = (
    {"source": "app_screens", "field": "product_id", "from": "2025-06-01",
     "reason": "not_collected"},
    {"source": "app_operations", "field": "device_new", "from": "2025-03-01",
     "reason": "not_collected"},
    {"source": "banners", "field": "campaign_code", "from": "2025-01-15",
     "reason": "not_collected"},
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
    "income_day",
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
# ГРУППЫ ДАТАСЕТА
# ============================================================
#
# Один запуск генератора рождает три группы подряд, каждую в
# свой каталог data/01_raw/<группа>. Всё, что их различает, живёт
# здесь и правится руками: ни пресетов, ни ключей командной
# строки нет, чтобы состав выгрузки нельзя было сменить
# случайно, опечаткой в команде.
#
# Мир у групп ОБЩИЙ: один world_seed даёт одну географию, одни
# бренды и одни торговые точки. Клиентов и их поведение делает
# seed группы, поэтому популяции не пересекаются.
#
# Горизонт у каждой группы свой. Начало обычно общее: слой
# разделения требует истории с required_history_start и ругается
# на группу, которая началась позже. Конец задаёт границу
# выгрузки, и он должен быть не раньше конечного среза группы
# в preprocessing.settings.default_windows.
#
# Большие выгрузки делаются только по отдельному решению.

WORLD_SEED = 42


@dataclass(frozen=True)
class DatasetGroup:
    """
    Группа датасета: сколько клиентов, какое окно, какой seed.
    """

    clients: int
    history_start: datetime
    history_end: datetime
    seed: int


DATASETS: dict[str, DatasetGroup] = {
    "train": DatasetGroup(
        clients=200,
        history_start=datetime(2024, 1, 1),
        history_end=datetime(2026, 1, 1),
        seed=100,
    ),
    "val": DatasetGroup(
        clients=50,
        history_start=datetime(2024, 1, 1),
        history_end=datetime(2026, 5, 1),
        seed=200,
    ),
    "test": DatasetGroup(
        clients=50,
        history_start=datetime(2024, 1, 1),
        history_end=datetime(2026, 9, 1),
        seed=300,
    ),
}
