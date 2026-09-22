from __future__ import annotations

from dataclasses import dataclass

from .projection import ENTITY_REFS, EVENT_TYPE_FIELD, SEMANTIC_PAYLOAD_FIELDS, validate_projection


# ============================================================
# ИДЕЯ
# ============================================================
#
# Таблица «физическое поле → semantic key». Это данные, а не
# алгоритм: каждое решение видно глазами и объяснено.
#
# Главное правило: одинаковое имя ничего не доказывает. `channel`
# у операции это pos, ecom, atm, а у коммуникации call, sms,
# push — разные множества значений, значит разные ключи. И
# наоборот: `amount` у операции и у операции в приложении это
# одни и те же тенге, значит ключ один, и это сказано явно.
#
# Разделяет источник события. Он объявлен в каталоге ключей
# выгрузки, а не угадан: 46 имён встречаются ровно в одном
# источнике, 11 — в нескольких, и для этих одиннадцати решение
# принято поимённо. Ещё два имени, old_value и new_value, смысла
# сами по себе не имеют: его задаёт изменившееся поле профиля.
#
# Ключи объявляются не только для полей события: анкета,
# её изменения и локальные ссылки объявляются здесь же и теми
# же правилами. Расчётных признаков среди ключей нет: их
# никто не считает.
#
# Здесь нет словарей, бакетов, частот и порогов: ключ обозначает
# смысл поля, а не идентификатор токена.
# ============================================================


KEYS_VERSION = "9.0.0"


# ------------------------------------------------------------
# ЕДИНИЦЫ ИЗМЕРЕНИЯ
# ------------------------------------------------------------
#
# Только то, что прямо сказано в описании поля каталогом ключей.
# Там, где единицы нет (код, идентификатор, категория), стоит
# None.
# ------------------------------------------------------------

UNITS: dict[str, str] = {
    # деньги: генератор объявляет тенге и хранит целые
    "amount": "KZT",
    "amount_or_limit": "KZT",
    "amount_due": "KZT",
    "amount_paid": "KZT",
    "principal_outstanding": "KZT",
    "balance_after": "KZT",
    "requested_amount": "KZT",
    "approved_amount": "KZT",
    "declared_income": "KZT",
    "credit_limit": "KZT",
    # сумма в валюте страны покупки: единица лежит в original_currency
    "original_amount": "original_currency",
    # сроки
    "term": "months",
    "requested_term": "months",
    "approved_term": "months",
    "relationship_months": "months",
    "days_past_due": "days",
    # доли
    "rate": "fraction_per_year",
    "credit_utilization": "fraction",
    # счётчики
    "children": "count",
    "contracts_count": "count",
    "active_contracts": "count",
    "installment_no": "count",
    "age": "years",
    # код календаря: число это код, а не величина. Час и день
    # недели события сюда не входят: они не поля выгрузки, а
    # отдельный временной канал, считаемый из event_time.
    "income_day": "day_of_month_code",
}

# Вид значения. Их ровно три. Служебные поля сюда не попадают
# вовсе: их отсеяла модельная проекция.
#
# Отдельного вида «дата» нет намеренно: календарь дат словарём не
# кодируется. Плановая дата платежа остаётся в canonical и в модель
# не выходит: числа дней до неё больше никто не считает.
NUMERIC = "numeric"
CATEGORICAL = "categorical"
TEXT = "text"

VALUE_KINDS: tuple[str, ...] = (NUMERIC, CATEGORICAL, TEXT)

# Локальная ссылка это не четвёртый вид значения, а отдельная
# роль: ACCOUNT_1 ничего не означает сам по себе и связывает
# события одного клиента. Значением модели он не становится.
REFERENCE = "reference"

# Поля, смысл которых зависит не от источника, а от значения
# соседнего поля. Прямого ключа у них нет: он выдаётся по
# field_name в profile_change_keys.
DYNAMIC_FIELDS: frozenset[str] = frozenset({"old_value", "new_value"})


class KeysError(ValueError):
    """
    Поле не сопоставлено смыслу или сопоставлено дважды.
    """


@dataclass(frozen=True)
class SemanticKey:
    key: str
    kind: str
    description: str
    unit: str | None = None
    temporal: str | None = None
    derived_from: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "value_kind": self.kind,
            "unit": self.unit,
            "temporal": self.temporal,
            "derived_from": list(self.derived_from),
            "description": self.description,
        }


def _k(key: str, kind: str, description: str, temporal: str | None = None) -> SemanticKey:
    return SemanticKey(key=key, kind=kind, description=description, unit=UNITS.get(key), temporal=temporal)


# ------------------------------------------------------------
# ИМЕНА, ОДНОЗНАЧНЫЕ ВО ВСЕЙ ВЫГРУЗКЕ
# ------------------------------------------------------------

DIRECT_KEYS: dict[str, SemanticKey] = {
    # --- что произошло ---
    #
    # Физическое поле называется type, смысл — event_type:
    # отдельной колонки конверта у типа события больше нет.
    EVENT_TYPE_FIELD: _k("event_type", CATEGORICAL, "что произошло"),
    # --- деньги операции ---
    "currency": _k("currency", CATEGORICAL, "валюта счёта операции"),
    "original_amount": SemanticKey("original_amount", NUMERIC, "сумма в валюте страны покупки",
                                   unit="original_currency"),
    "original_currency": _k("original_currency", CATEGORICAL, "валюта страны покупки"),
    "balance_after": SemanticKey("balance_after", NUMERIC, "остаток счёта после проводки", unit="KZT"),
    "direction": _k("direction", CATEGORICAL, "направление по счёту клиента: списание или зачисление"),
    "decline_reason": _k("decline_reason", CATEGORICAL, "причина отказа по операции"),
    "accrual_period": _k("accrual_period", CATEGORICAL, "период начисления периодической суммы",
                         temporal="месяц начисления"),
    # --- торговая точка ---
    "merchant_name": _k("merchant_name", TEXT, "название точки в терминальной строке"),
    "merchant_category": _k("merchant_category", CATEGORICAL, "категория точки"),
    "mcc": _k("mcc", CATEGORICAL, "категория точки кодом MCC: цифры это код, а не величина"),
    "merchant_city": _k("merchant_city", CATEGORICAL, "город операции"),
    "merchant_country": _k("merchant_country", CATEGORICAL, "страна точки"),
    "is_online": _k("is_online", CATEGORICAL, "операция без присутствия карты"),
    "is_subscription": _k("is_subscription", CATEGORICAL, "регулярное списание подписки"),
    "counterparty": _k("counterparty", TEXT, "устойчивое маскированное имя контрагента"),
    # --- договор и продукт ---
    "migration_reason": _k("migration_reason", CATEGORICAL, "причина перехода между продуктами"),
    "amount_or_limit": SemanticKey("amount_or_limit", NUMERIC, "сумма договора или кредитный лимит", unit="KZT"),
    "term": SemanticKey("term", NUMERIC, "срок договора", unit="months"),
    "rate": SemanticKey("rate", NUMERIC, "номинальная ставка договора", unit="fraction_per_year"),
    # --- заявка ---
    "requested_amount": SemanticKey("requested_amount", NUMERIC, "запрошенная клиентом сумма", unit="KZT"),
    "requested_term": SemanticKey("requested_term", NUMERIC, "запрошенный срок", unit="months"),
    "approved_amount": SemanticKey("approved_amount", NUMERIC, "одобренная банком сумма", unit="KZT"),
    "approved_term": SemanticKey("approved_term", NUMERIC, "одобренный срок", unit="months"),
    # --- график и просрочка ---
    "installment_no": SemanticKey("installment_no", NUMERIC, "номер платежа в графике", unit="count"),
    "amount_due": SemanticKey("amount_due", NUMERIC, "сумма планового платежа", unit="KZT"),
    "amount_paid": SemanticKey("amount_paid", NUMERIC, "фактически уплаченная сумма", unit="KZT"),
    "principal_outstanding": SemanticKey("principal_outstanding", NUMERIC, "остаток основного долга", unit="KZT"),
    "days_past_due": SemanticKey("days_past_due", NUMERIC, "дней просрочки", unit="days"),
    # --- мошенничество ---
    "subject": _k("subject", CATEGORICAL, "объект проверки: карта, перевод, вход"),
    "score_band": _k("score_band", CATEGORICAL, "полоса риска"),
    "rule_code": _k("rule_code", CATEGORICAL, "сработавшее правило антифрода"),
    # --- коммуникации и баннеры ---
    "template": _k("template", CATEGORICAL, "шаблон сообщения"),
    "purpose": _k("purpose", CATEGORICAL, "назначение коммуникации"),
    "delivered": _k("delivered", CATEGORICAL, "результат доставки: подтверждена, не доставлено или неизвестно"),
    "slot": _k("slot", CATEGORICAL, "место показа баннера в приложении"),
    # Закрытый перечень кодов предложения (cash_loan, credit_card,
    # cashback и далее), а не свободный текст: разбивать его на
    # куски нечего. Идентификатором предложения он при этом не
    # является — тот остаётся ссылкой offer_ref.
    "offer": _k("offer", CATEGORICAL, "код предложения баннера: не идентификатор предложения"),
    # --- приложение ---
    "firebase_screen": _k("firebase_screen", CATEGORICAL, "экран приложения"),
    "funnel_stage": _k("funnel_stage", CATEGORICAL, "стадия воронки заявки"),
    "operation": _k("operation", CATEGORICAL, "операция в приложении"),
    "error_code": _k("error_code", CATEGORICAL, "код ошибки в приложении"),
    "device_new": _k("device_new", CATEGORICAL, "действие с нового устройства"),
    # --- обращения ---
    "topic": _k("topic", CATEGORICAL, "тема обращения"),
    # --- профиль как событие ---
    "field_name": _k("field_name", CATEGORICAL, "какое поле профиля изменилось"),
    "change_source": _k("change_source", CATEGORICAL, "откуда банк узнал об изменении профиля"),
    "confirmed": _k("confirmed", CATEGORICAL, "значение профиля подтверждено документом"),
}


# ------------------------------------------------------------
# ИМЕНА, ЗНАЧЕНИЕ КОТОРЫХ ЗАВИСИТ ОТ ИСТОЧНИКА
# ------------------------------------------------------------
#
# Источник → ключ. Причина расхождения записана рядом: это те
# самые случаи, где одинаковое имя скрывает разные множества
# значений.
# ------------------------------------------------------------

BY_SOURCE_KEYS: dict[str, dict[str, SemanticKey]] = {
    "amount": {
        "transactions": SemanticKey("transaction_amount", NUMERIC, "сумма денежной операции клиента", unit="KZT"),
        "app_operations": SemanticKey("transaction_amount", NUMERIC, "сумма денежной операции клиента", unit="KZT"),
    },
    "channel": {
        "transactions": _k("operation_channel", CATEGORICAL, "канал операции: pos, ecom, atm, app, branch, qr"),
        "applications": _k("application_channel", CATEGORICAL, "канал подачи заявки"),
        "communications": _k("communication_channel", CATEGORICAL, "канал сообщения: call, sms, push, email"),
        "support": _k("case_channel", CATEGORICAL, "канал обращения в поддержку"),
    },
    "status": {
        "transactions": _k("operation_status", CATEGORICAL, "исход операции: approved, declined, reversed"),
        "app_operations": _k("app_operation_status", CATEGORICAL, "исход операции в приложении"),
        "support": _k("case_status", CATEGORICAL, "статус обращения"),
    },
    "reason": {
        "transactions": _k("operation_reason", CATEGORICAL, "основание операции"),
        "product_events": _k("product_event_reason", CATEGORICAL, "основание продуктового события"),
        "loans": _k("loan_event_reason", CATEGORICAL, "основание кредитного события"),
    },
    "decision": {
        "applications": _k("application_decision", CATEGORICAL, "решение по заявке: approved или rejected"),
        "antifraud": _k("fraud_decision", CATEGORICAL, "решение антифрода: monitor, confirm_request, block"),
    },
    "resolution": {
        "antifraud": _k("fraud_resolution", CATEGORICAL, "исход проверки мошенничества"),
        "support": _k("case_resolution", CATEGORICAL, "исход обращения"),
    },
    "reject_reason": {
        "applications": _k("application_reject_reason", CATEGORICAL, "причина отказа по заявке"),
        "app_screens": _k("screen_reject_reason", CATEGORICAL, "причина отказа, показанная на экране"),
    },
    "campaign_code": {
        "communications": _k("campaign_code", CATEGORICAL, "код маркетинговой кампании"),
        "banners": _k("campaign_code", CATEGORICAL, "код маркетинговой кампании"),
    },
    "domain": {
        "app_screens": _k("app_domain", CATEGORICAL, "раздел приложения"),
        "app_operations": _k("app_domain", CATEGORICAL, "раздел приложения"),
    },
}


# Локальные ссылки: смысла значения не несут, но связывают события
# одного клиента. Роль отдельная, значением модели они не
# становятся автоматически.
REFERENCE_KEYS: dict[str, SemanticKey] = {
    name: SemanticKey(name, REFERENCE, f"локальная ссылка на наблюдаемую сущность клиента ({column})")
    for column, (name, _prefix) in ENTITY_REFS.items()
}


# ------------------------------------------------------------
# ПРОФИЛЬ КЛИЕНТА
# ------------------------------------------------------------
#
# Версия профиля, действующая на cutoff. Ключи начинаются с
# profile_, потому что город жизни клиента и город покупки это
# разные смыслы, а имя у них похожее.
# ------------------------------------------------------------

PROFILE_NUMERIC: dict[str, tuple[str, str | None]] = {
    "age": ("возраст клиента", "years"),
    "children": ("число детей", "count"),
    "declared_income": ("заявленный доход", "KZT"),
    "relationship_months": ("месяцев отношений с банком по данным профиля", "months"),
    "contracts_count": ("договоров всего по данным профиля", "count"),
    "active_contracts": ("действующих договоров по данным профиля", "count"),
    "credit_limit": ("кредитный лимит клиента", "KZT"),
    "credit_utilization": ("доля использования лимита", "fraction"),
}

PROFILE_CATEGORICAL: dict[str, str] = {
    "gender": "пол",
    "family_status": "семейное положение",
    "education": "образование",
    "region": "регион проживания",
    "city": "город проживания",
    "housing_type": "тип жилья",
    "pensioner": "пенсионер",
    "income_type": "вид дохода",
    "industry": "отрасль занятости",
    "income_day": "день выплаты основного дохода: число это код дня месяца",
    "holds_credit_card": "держит кредитную карту",
    "holds_debit_card": "держит дебетовую карту",
    "holds_deposit": "держит вклад",
}

PROFILE_KEYS: dict[str, SemanticKey] = {
    **{
        name: SemanticKey(f"profile_{name}", NUMERIC, description, unit=unit)
        for name, (description, unit) in PROFILE_NUMERIC.items()
    },
    **{
        name: SemanticKey(f"profile_{name}", CATEGORICAL, description)
        for name, description in PROFILE_CATEGORICAL.items()
    },
}


# ------------------------------------------------------------
# ИЗМЕНЕНИЕ ПРОФИЛЯ
# ------------------------------------------------------------
#
# У события profile_change смысл значения задаёт не имя колонки,
# а поле field_name. Доход, город и число детей лежат в одних и
# тех же old_value и new_value, но это три разных смысла: если
# оставить их текстом, BPE будет резать доход на куски, как имя
# города.
#
# Поэтому пара old/new превращается в пару ключей того поля,
# которое изменилось, и наследует его вид значения и единицу.
# ------------------------------------------------------------

# Согласие на маркетинг живёт только в событии изменения: в
# версии профиля такой колонки нет.
PROFILE_CHANGE_EXTRA: dict[str, SemanticKey] = {
    "consent_marketing": SemanticKey(
        "profile_consent_marketing", CATEGORICAL, "согласие на маркетинговые коммуникации"
    ),
}

# Поля профиля, которые меняются за жизнь клиента. Список
# позитивный: незнакомое field_name останавливает слой, а не
# молча превращается в текст.
CHANGEABLE_PROFILE_FIELDS: tuple[str, ...] = (
    "region",
    "city",
    "children",
    "family_status",
    "education",
    "housing_type",
    "declared_income",
    "income_type",
    "industry",
    "income_day",
    "pensioner",
    "consent_marketing",
)


def _change_pair(base: SemanticKey) -> tuple[SemanticKey, SemanticKey]:
    return (
        SemanticKey(
            f"{base.key}_old", base.kind, f"прежнее значение: {base.description}",
            unit=base.unit, derived_from=("field_name", "old_value"),
        ),
        SemanticKey(
            f"{base.key}_new", base.kind, f"новое значение: {base.description}",
            unit=base.unit, derived_from=("field_name", "new_value"),
        ),
    )


PROFILE_CHANGE_KEYS: dict[str, tuple[SemanticKey, SemanticKey]] = {
    name: _change_pair(PROFILE_KEYS[name] if name in PROFILE_KEYS else PROFILE_CHANGE_EXTRA[name])
    for name in CHANGEABLE_PROFILE_FIELDS
}


# ------------------------------------------------------------
# ЧЕГО ЗДЕСЬ НЕТ
# ------------------------------------------------------------
#
# Расчётных и временных признаков больше нет ни одного:
# ни интервалов между событиями, ни отношений суммы к доходу
# или к лимиту, ни возраста истории. Их никто не считает, и
# объявленный ключ без источника был бы обещанием поля,
# которого модель никогда не увидит.
#
# Календарь сюда не относится: час суток и день недели это
# отдельный числовой канал, он словарём не кодируется.
# ------------------------------------------------------------


# ------------------------------------------------------------
# СПОРНЫЕ ОБЪЕДИНЕНИЯ
# ------------------------------------------------------------
#
# Пары, которые выглядят одинаково, но склеивать их нельзя.
# Список ведётся руками: он объясняет решения таблицы.
# ------------------------------------------------------------

AMBIGUOUS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("transaction_amount", "amount_or_limit", "approved_amount", "requested_amount", "amount_due", "amount_paid"),
     "все в тенге, но это разные показатели: проводка, лимит договора, решение банка, запрос клиента и график"),
    (("operation_channel", "application_channel", "communication_channel", "case_channel"),
     "канал у операции, заявки, сообщения и обращения берётся из разных множеств значений"),
    (("operation_status", "app_operation_status", "case_status"),
     "исход операции, операции в приложении и обращения описываются разными словарями"),
    (("operation_reason", "product_event_reason", "loan_event_reason"),
     "основание операции, продуктового события и кредитного события это три разных перечня"),
    (("application_decision", "fraud_decision"),
     "решение по заявке и решение антифрода не сравнимы"),
    (("fraud_resolution", "case_resolution"),
     "исход проверки мошенничества и исход обращения в поддержку это разные вещи"),
    (("merchant_city", "profile_city", "profile_region"),
     "город операции это место покупки, а город и регион профиля это место жизни клиента"),
    (("merchant_category", "mcc"),
     "категория точки и код MCC описывают одно и то же разными перечнями: "
     "внутренняя категория подробнее кода"),
    (("is_online", "is_subscription", "delivered", "confirmed", "device_new"),
     "булевы значения разных фактов: истина у одного ничего не говорит об истине у другого"),
    (("amount_to_declared_income", "amount_to_limit", "amount_to_balance_after", "amount_to_client_average"),
     "все безразмерные отношения, но знаменатели разные: доход, лимит договора, остаток счёта и прошлое клиента"),
    (("profile_declared_income_new", "profile_city_new", "profile_children_new"),
     "новое значение профиля наследует смысл изменённого поля: деньги, город и количество детей "
     "не один текстовый ключ"),
)

# Явно разрешённые объединения одного смысла из разных источников.
ALLOWED_SHARING: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("transaction_amount", ("transactions", "app_operations"),
     "сумма денежной операции в тенге, записанная витриной операций или приложением"),
    ("campaign_code", ("communications", "banners"),
     "код одной и той же маркетинговой кампании"),
    ("app_domain", ("app_screens", "app_operations"),
     "раздел приложения, названный экраном или операцией"),
)


# ============================================================
# РАЗРЕШЕНИЕ КЛЮЧА
# ============================================================


def key_for(name: str, source: str) -> SemanticKey:
    """
    Смысл физического поля по его имени и источнику события.
    """

    if name in DYNAMIC_FIELDS:
        raise KeysError(
            f"поле {name!r} не имеет постоянного смысла: его задаёт field_name, "
            "ключ выдаёт profile_change_keys"
        )

    by_source = BY_SOURCE_KEYS.get(name)

    if by_source is not None:

        key = by_source.get(source)

        if key is None:
            raise KeysError(
                f"поле {name!r} источника {source!r} не сопоставлено смыслу: "
                "имя встречается в нескольких источниках, решение принимается поимённо"
            )

        return key

    key = DIRECT_KEYS.get(name)

    if key is None:
        raise KeysError(f"поле {name!r} не сопоставлено смыслу")

    return key


def profile_change_keys(field_name: str | None) -> tuple[SemanticKey, SemanticKey]:
    """
    Пара ключей «прежнее и новое значение» того поля профиля,
    которое изменилось.
    """

    pair = PROFILE_CHANGE_KEYS.get(field_name or "")

    if pair is None:
        raise KeysError(
            f"поле профиля {field_name!r} не объявлено изменяемым: его old_value и new_value "
            "остались бы текстом без смысла"
        )

    return pair


def validate_keys(catalogue: dict) -> None:
    """
    Каждое поле, прошедшее модельную проекцию, имеет ровно один
    смысл; у ключа один вид значения во всех своих полях.

    catalogue: event_type -> {"source": ..., "fields": [...]} из
    каталога ключей выгрузки.

    Сначала проверяется сама граница модели: новое поле
    выгрузки обязано быть названо либо смысловым, либо
    ссылкой, либо внутренним, иначе оно молча пропало бы по
    дороге к модели.
    """

    kinds: dict[str, set[str]] = {}
    missing: list[str] = []

    validate_projection(
        item["name"] if isinstance(item, dict) else item.name
        for info in catalogue.values()
        for item in (info["fields"] if isinstance(info, dict) else info.fields)
    )

    for event_type, info in sorted(catalogue.items()):

        source = info["source"] if isinstance(info, dict) else info.source
        fields = info["fields"] if isinstance(info, dict) else info.fields

        for item in fields:

            name = item["name"] if isinstance(item, dict) else item.name

            if name not in SEMANTIC_PAYLOAD_FIELDS or name in DYNAMIC_FIELDS:
                continue

            try:
                key = key_for(name, source)
            except KeysError as error:
                missing.append(str(error))
                continue

            kinds.setdefault(key.key, set()).add(key.kind)

    # Ключи, которых нет в payload: конверт, профиль, справочники,
    # связи, время и расчёты. Они тоже обязаны иметь один вид
    # значения, иначе следующий этап закодирует их двумя способами.
    for key in _all_declared_keys():
        kinds.setdefault(key.key, set()).add(key.kind)

    if missing:
        raise KeysError("; ".join(sorted(set(missing))))

    split = {key: sorted(values) for key, values in kinds.items() if len(values) > 1}

    if split:
        raise KeysError(f"у ключа больше одного вида значения: {split}")


def _all_declared_keys() -> list[SemanticKey]:
    """
    Все ключи, объявленные помимо разбора payload.
    """

    out: list[SemanticKey] = [
        *REFERENCE_KEYS.values(),
        *PROFILE_KEYS.values(),
    ]

    for old, new in PROFILE_CHANGE_KEYS.values():
        out.extend((old, new))

    return out


def keys_registry(catalogue: dict) -> dict:
    """
    Реестр смыслов: что во что превращается и почему.
    """

    validate_keys(catalogue)

    rows: dict[str, dict] = {}

    for event_type, info in sorted(catalogue.items()):

        source = info["source"] if isinstance(info, dict) else info.source
        fields = info["fields"] if isinstance(info, dict) else info.fields

        for item in fields:

            name = item["name"] if isinstance(item, dict) else item.name

            if name not in SEMANTIC_PAYLOAD_FIELDS or name in DYNAMIC_FIELDS:
                continue

            key = key_for(name, source)

            row = rows.setdefault(key.key, {**key.as_dict(), "physical_fields": []})
            row["physical_fields"].append(f"{event_type}.{name}")

    for key in REFERENCE_KEYS.values():
        rows.setdefault(key.key, {**key.as_dict(), "physical_fields": ["derived:local_ref"]})

    for name, key in PROFILE_KEYS.items():
        rows.setdefault(key.key, {**key.as_dict(), "physical_fields": [f"profile.{name}"]})

    for field_name, (old, new) in PROFILE_CHANGE_KEYS.items():
        rows.setdefault(old.key, {**old.as_dict(), "physical_fields": [f"profile_change[{field_name}].old_value"]})
        rows.setdefault(new.key, {**new.as_dict(), "physical_fields": [f"profile_change[{field_name}].new_value"]})

    for row in rows.values():
        row["physical_fields"] = sorted(row["physical_fields"])

    by_kind: dict[str, int] = {}
    for row in rows.values():
        by_kind[row["value_kind"]] = by_kind.get(row["value_kind"], 0) + 1

    return {
        "keys_version": KEYS_VERSION,
        "rules": {
            "identity": "смысл задаётся парой «имя поля, источник события»: одинаковое имя ничего не доказывает",
            "value_kind": "ровно три вида значения: numeric, categorical, text; цифровые коды это "
                          "категории, а не величины, а календарных дат словарём не кодируют",
            "references": "reference это не четвёртый вид значения, а отдельная роль: "
                          "локальная ссылка связывает события клиента и значением модели не становится",
            "derived": "расчётных признаков среди ключей нет: модель получает фактические "
                       "поля события и анкеты, а не выведенные из них отношения",
            "profile_change": "old_value и new_value наследуют смысл изменённого поля профиля: "
                              "доход остаётся числом, город категорией",
            "no_fit": "здесь нет словарей, бакетов, частот и порогов: ключ обозначает смысл, а не токен",
        },
        "counts": {"keys": len(rows), "by_value_kind": dict(sorted(by_kind.items()))},
        "keys": dict(sorted(rows.items())),
        "ambiguous": [{"keys": list(group), "reason": reason} for group, reason in AMBIGUOUS],
        "allowed_sharing": [
            {"key": key, "sources": list(sources), "reason": reason}
            for key, sources, reason in ALLOWED_SHARING
        ],
    }


__all__ = [
    "ALLOWED_SHARING",
    "AMBIGUOUS",
    "BY_SOURCE_KEYS",
    "CATEGORICAL",
    "CHANGEABLE_PROFILE_FIELDS",
    "DIRECT_KEYS",
    "DYNAMIC_FIELDS",
    "KEYS_VERSION",
    "NUMERIC",
    "PROFILE_CATEGORICAL",
    "PROFILE_CHANGE_KEYS",
    "PROFILE_KEYS",
    "PROFILE_NUMERIC",
    "REFERENCE",
    "REFERENCE_KEYS",
    "TEXT",
    "VALUE_KINDS",
    "KeysError",
    "SemanticKey",
    "key_for",
    "keys_registry",
    "profile_change_keys",
    "validate_keys",
]
