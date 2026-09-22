from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterable

from .settings import CALENDAR_ENCODING

if TYPE_CHECKING:  # pragma: no cover - только для подсказок типов
    from .history import ClientHistory


# ============================================================
# ИДЕЯ
# ============================================================
#
# Граница между данными и моделью. Всё, что выше, служит очистке:
# версии, дубли, конфликты, порядок, связи, трассировка к RAW.
# Ниже этой границы модель видит только банковский смысл.
#
# Модельное событие:
#
#   client_id    чей это факт
#   event_time   когда произошло
#   source       какая система записала
#   fields       смысловые банковские значения
#
# event_type лежит ВНУТРИ fields как смысловое поле и в проекции
# больше нигде не дублируется.
#
# Список разрешённого позитивный. «Всё, кроме нескольких
# исключений» здесь запрещено: новое поле в выгрузке не должно
# попасть в модель само собой, оно обязано быть названо.
#
# Сырые идентификаторы наружу не выходят. Сущности клиента
# получают локальные ссылки (ACCOUNT_1, CARD_1, CONTRACT_1),
# пронумерованные по первому появлению в истории этого клиента.
# Между клиентами такие ссылки не значат ничего и ничего не
# идентифицируют.
#
# Здесь нет словарей, бакетов, BPE, Masker и токенизации. Этап
# только называет границу, внутри которой им позже разрешено
# учиться.
# ============================================================


PROJECTION_VERSION = "5.0.0"

# Тип события. В выгрузке это ключ payload под именем type, в
# canonical — колонка event_type; здесь названа колонка, потому
# что проекция читает уже очищенную строку.
EVENT_TYPE_FIELD = "event_type"


class ProjectionError(ValueError):
    """
    Поле выгрузки не классифицировано: его нельзя ни пропустить
    в модель, ни молча выбросить.
    """


# ------------------------------------------------------------
# РАЗРЕШЁННЫЕ ПОЛЯ PAYLOAD
# ------------------------------------------------------------
#
# Имя поля -> почему оно смысловое. Причина короткая и по делу:
# она объясняет, что именно узнаёт модель.
# ------------------------------------------------------------

SEMANTIC_PAYLOAD_FIELDS: dict[str, str] = {
    # --- что произошло ---
    #
    # В выгрузке это ключ payload под именем type, в canonical —
    # колонка event_type. Поле смысловое и идёт в fields первым.
    EVENT_TYPE_FIELD: "тип события",
    # --- деньги ---
    "amount": "сумма операции",
    "original_amount": "сумма в валюте страны покупки",
    "original_currency": "валюта страны покупки",
    "currency": "валюта счёта",
    "balance_after": "остаток счёта после проводки",
    "amount_or_limit": "сумма договора или лимит",
    "amount_due": "плановый платёж",
    "amount_paid": "фактически уплачено",
    "principal_outstanding": "остаток основного долга",
    "requested_amount": "запрошенная сумма",
    "approved_amount": "одобренная сумма",
    "rate": "ставка договора",
    # --- сроки и графики ---
    "term": "срок договора",
    "requested_term": "запрошенный срок",
    "approved_term": "одобренный срок",
    "days_past_due": "дней просрочки",
    "installment_no": "номер платежа в графике",
    "accrual_period": "период начисления",
    # --- операция ---
    "direction": "направление по счёту клиента",
    "channel": "канал: pos, ecom, atm, app, branch, qr",
    "status": "исход операции или обращения",
    "reason": "основание операции или события",
    "decline_reason": "причина отказа по операции",
    "is_online": "операция без присутствия карты",
    "is_subscription": "регулярное списание подписки",
    # --- торговая точка ---
    "merchant_name": "название в терминальной строке",
    "merchant_category": "категория точки",
    "merchant_city": "город операции",
    "merchant_country": "страна точки",
    "mcc": "категория точки кодом",
    "counterparty": "устойчивое маскированное имя контрагента",
    # --- продукт ---
    "migration_reason": "причина перехода между продуктами",
    # --- заявка и решение ---
    "decision": "решение по заявке или по мошенничеству",
    "reject_reason": "причина отказа по заявке",
    "resolution": "исход обращения или проверки",
    # --- обращения ---
    "topic": "тема обращения",
    # --- коммуникации и маркетинг ---
    "campaign_code": "код кампании",
    "template": "шаблон сообщения",
    "purpose": "назначение коммуникации",
    "delivered": "сообщение доставлено",
    "offer": "код предложения баннера",
    "slot": "место показа в приложении",
    # --- приложение ---
    "domain": "раздел приложения",
    "firebase_screen": "экран приложения",
    "funnel_stage": "стадия воронки заявки",
    "operation": "операция в приложении",
    "error_code": "код ошибки в приложении",
    "device_new": "действие с нового устройства",
    # --- мошенничество ---
    "subject": "объект проверки: карта, перевод, вход",
    "score_band": "полоса риска",
    "rule_code": "сработавшее правило",
    # --- профиль ---
    "field_name": "какое поле профиля изменилось",
    "old_value": "прежнее значение поля профиля",
    "new_value": "новое значение поля профиля",
    "change_source": "откуда банк узнал об изменении",
    "confirmed": "значение подтверждено документом",
}


# ------------------------------------------------------------
# ЛОКАЛЬНЫЕ ССЫЛКИ
# ------------------------------------------------------------
#
# Сырое имя поля -> (имя в fields, префикс значения).
#
# Номер выдаётся по первому появлению сущности в истории
# клиента. Один и тот же идентификатор всегда получает одну
# ссылку, разные клиенты нумеруются независимо.
# ------------------------------------------------------------

ENTITY_REFS: dict[str, tuple[str, str]] = {
    "account_id": ("account_ref", "ACCOUNT"),
    "card_id": ("card_ref", "CARD"),
    "contract_id": ("contract_ref", "CONTRACT"),
    "application_id": ("application_ref", "APPLICATION"),
    "case_id": ("case_ref", "CASE"),
    "offer_id": ("offer_ref", "OFFER"),
    "merchant_id": ("merchant_ref", "MERCHANT"),
    # Сессия приложения и перевод — такие же наблюдаемые сущности
    # клиента: они различают шаги одной сессии и две ноги одного
    # перевода. Сырой ключ наружу не выходит, выходит ссылка.
    "session_id": ("session_ref", "SESSION"),
    "transfer_id": ("transfer_ref", "TRANSFER"),
}


# ------------------------------------------------------------
# ВНУТРЕННИЕ ПОЛЯ
# ------------------------------------------------------------
#
# Имя -> почему поле остаётся внутри. Список служит отчёту и
# документации; фильтром он НЕ работает, фильтр только
# позитивный.
# ------------------------------------------------------------

INTERNAL_FIELDS: dict[str, str] = {
    # конверт
    # payload
    "product_id": "идентификатор продукта в каталоге банка: связывает события одного продукта",
    "previous_product_id": "идентификатор прежнего продукта при переходе",
    "due_date": "плановая дата платежа: модель получает days_to_due, календарных дат в словарях нет",
    # производные canonical
    "client_idx": "внутренний индекс клиента в группе",
    "stable_event_index": "внутренний номер логического события",
    "before_window": "строка старше окна наблюдения",
    "at_or_after_extract": "строка на границе выгрузки или позже",
    "ambiguous_local_time": "признак качества времени",
    "balance_chain_gap": "между наблюдаемыми строками потеряно движение денег, признак качества",
    "payload_status": "результат разбора payload",
    "payload_violations": "нарушения контракта payload",
    "known_missing": "причина пропуска по датированному правилу схемы",
    "merchant_name_norm": "нормализованная копия текста",
    "counterparty_norm": "нормализованная копия текста",
    "raw_file": "трассировка к RAW",
    "raw_row_group": "трассировка к RAW",
    "raw_row": "трассировка к RAW",
}


# ------------------------------------------------------------
# ИНИЦИАТОР
# ------------------------------------------------------------
# ДЕЙСТВИЕ КЛИЕНТА
# ------------------------------------------------------------
#
# Метки инициатора в конверте больше нет. Кто совершил событие,
# видно по его типу: покупку и перевод делает клиент, начисление
# процентов и блокировку по подозрению — банк.
#
# Список объявлен здесь, а не выводится из источника: источник
# говорит, какая система записала факт, а не кто действовал.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ЧТО ДОБАВЛЯЕТ ЭТАП 5
# ------------------------------------------------------------
#
# Проекция пропускает только то, что лежит в самой строке.
# Значения, которые нужно вывести из связи, из времени или из
# расчёта, добавляет смысловой слой — и объявляет их такими же
# ключами, как физические поля. Справочников среди источников
# больше нет: точку и продукт описывает само событие.
#
# Список нужен здесь, чтобы граница читалась целиком: в модель
# приходит не только payload.
# ------------------------------------------------------------

SEMANTIC_LAYER_FIELDS: dict[str, str] = {
    "days_to_due": "дней до планового платежа вместо самой даты",
    "since_previous_hours": "часов с прошлого видимого события",
    "since_same_type_hours": "часов с прошлого события того же типа",
    "since_last_income_hours": "часов с последнего видимого дохода",
    "age_of_history_days": "дней от начала наблюдаемой истории",
    "amount_to_declared_income": "сумма операции к заявленному доходу",
    "amount_to_limit": "сумма операции к лимиту договора",
    "amount_to_balance_after": "сумма операции к остатку счёта",
    "amount_to_client_average": "сумма операции к среднему по прошлым операциям того же типа",
    "profile_<поле>_old": "прежнее значение изменившегося поля профиля в его собственном смысле",
    "profile_<поле>_new": "новое значение изменившегося поля профиля в его собственном смысле",
}


# ============================================================
# МОДЕЛЬНОЕ СОБЫТИЕ
# ============================================================


@dataclass(frozen=True)
class ModelEvent:
    client_id: str
    event_time: datetime
    source: str
    fields: dict

    def as_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "event_time": self.event_time,
            "source": self.source,
            "fields": dict(self.fields),
        }


class LocalRefs:
    """
    Локальные ссылки одного клиента: сущность получает номер по
    первому появлению в его истории и держит его дальше.
    """

    def __init__(self) -> None:
        self._numbers: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = {}

    def ref(self, column: str, value: str) -> str:

        name, prefix = ENTITY_REFS[column]

        key = (prefix, str(value))

        known = self._numbers.get(key)

        if known is None:
            self._counts[prefix] = self._counts.get(prefix, 0) + 1
            known = f"{prefix}_{self._counts[prefix]}"
            self._numbers[key] = known

        return known

    @property
    def size(self) -> int:
        return len(self._numbers)


def model_event(row: dict, refs: LocalRefs) -> ModelEvent:
    """
    Строит модельное событие из строки очищенной истории.

    Проходит только то, что названо разрешённым: смысловые поля
    payload, тип события, локальные ссылки и, где это осмысленно,
    инициатор действия.
    """

    fields: dict = {EVENT_TYPE_FIELD: row[EVENT_TYPE_FIELD]}

    for name in SEMANTIC_PAYLOAD_FIELDS:
        value = row.get(name)
        if value is not None:
            fields[name] = value

    for column in ENTITY_REFS:
        value = row.get(column)
        if value is not None:
            name, _ = ENTITY_REFS[column]
            fields[name] = refs.ref(column, value)

    return ModelEvent(
        client_id=row["client_id"],
        event_time=row["event_time"],
        source=row["source"],
        fields=fields,
    )


def model_history(history: "ClientHistory") -> list[ModelEvent]:
    """
    Финальная очищенная история клиента глазами модели.

    Вход — результат history_as_of: по одному ряду на событие, в
    действующей версии, в бизнес-порядке. Второй реализации
    видимости здесь нет.
    """

    refs = LocalRefs()

    return [model_event(row, refs) for row in history.events.to_pylist()]


# ============================================================
# РЕЕСТР И ПРОВЕРКА
# ============================================================


def model_role(name: str) -> str:
    """
    Судьба поля: смысловое значение, локальная ссылка или
    внутреннее поле слоя.
    """

    if name in SEMANTIC_PAYLOAD_FIELDS:
        return "semantic_field"

    if name in ENTITY_REFS:
        return "local_ref"

    return "internal"


def validate_projection(payload_names: Iterable[str]) -> None:
    """
    Каждое имя payload выгрузки названо ровно один раз: либо
    разрешено, либо превращается в ссылку, либо объявлено
    внутренним. Незнакомое имя останавливает работу.
    """

    unknown: list[str] = []
    twice: list[str] = []

    for name in sorted(set(payload_names)):

        places = sum(
            1
            for table in (SEMANTIC_PAYLOAD_FIELDS, ENTITY_REFS, INTERNAL_FIELDS)
            if name in table
        )

        if places == 0:
            unknown.append(name)
        elif places > 1:
            twice.append(name)

    if unknown:
        raise ProjectionError(
            "поля выгрузки не классифицированы модельной проекцией: "
            + ", ".join(unknown)
            + ". Назовите каждое либо смысловым, либо ссылкой, либо внутренним"
        )

    if twice:
        raise ProjectionError("поля названы дважды: " + ", ".join(twice))


def projection_registry(payload_names: Iterable[str], timezone: str | None = None) -> dict:
    """
    Что именно пересекает границу модели. Пишется в
    field_registry.json рядом с физическими полями.
    """

    names = sorted(set(payload_names))

    validate_projection(names)

    allowed = [name for name in names if name in SEMANTIC_PAYLOAD_FIELDS]
    refs = [name for name in names if name in ENTITY_REFS]
    internal = [name for name in names if name in INTERNAL_FIELDS]

    return {
        "projection_version": PROJECTION_VERSION,
        "schema": {
            "client_id": "чей это факт",
            "event_time": "когда произошло",
            "source": "какая система записала",
            "fields": "смысловые банковские значения, включая event_type",
        },
        "rules": {
            "allowlist": "в модель проходит только явно названное поле; «всё, кроме исключений» запрещено",
            "identifiers": "сырые идентификаторы не выходят наружу: сущности клиента получают локальные ссылки",
            "refs": "номер ссылки выдаётся по первому появлению сущности в истории клиента и между клиентами ничего не значит",
            "internal": "технические поля не становятся semantic keys, не попадают в словари, маски и модель",
            "client_action": "кто действовал, видно по типу события; метки инициатора в конверте нет",
        },
        "calendar": {
            **{key: value for key, value in CALENDAR_ENCODING.items() if key != "features"},
            "features": list(CALENDAR_ENCODING["features"]),
            "excluded_from": list(CALENDAR_ENCODING["excluded_from"]),
            "timezone": timezone,
            "note": "считается из event_time при подготовке входа модели; в fields не входит",
        },
        "counts": {
            "semantic_payload_fields": len(allowed),
            "local_refs": len(refs),
            "internal_payload_fields": len(internal),
            "client_action_event_types": len(CLIENT_ACTION_EVENT_TYPES),
            "added_by_semantic_stage": len(SEMANTIC_LAYER_FIELDS),
        },
        "semantic_fields": {name: SEMANTIC_PAYLOAD_FIELDS[name] for name in allowed},
        "local_refs": {name: ENTITY_REFS[name][0] for name in refs},
        "internal_fields": dict(sorted(INTERNAL_FIELDS.items())),
        "client_action_event_types": sorted(CLIENT_ACTION_EVENT_TYPES),
        "added_by_semantic_stage": dict(sorted(SEMANTIC_LAYER_FIELDS.items())),
    }


__all__ = [
    "ENTITY_REFS",
    "EVENT_TYPE_FIELD",
    "CLIENT_ACTION_EVENT_TYPES",
    "INTERNAL_FIELDS",
    "SEMANTIC_LAYER_FIELDS",
    "PROJECTION_VERSION",
    "SEMANTIC_PAYLOAD_FIELDS",
    "LocalRefs",
    "ModelEvent",
    "ProjectionError",
    "model_event",
    "model_history",
    "model_role",
    "projection_registry",
    "validate_projection",
]
