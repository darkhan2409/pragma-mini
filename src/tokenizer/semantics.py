from __future__ import annotations

import json
from typing import Iterable

from src.preprocessing.artifacts import sha256_bytes
from src.preprocessing.config import (
    KIND_BOOLEAN,
    KIND_CATEGORICAL,
    KIND_NUMERIC,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственное место, где сказано, какие РАЗНЫЕ физические поля
# несут один и тот же смысл, и какие значения одинаковы не по
# написанию, а по сути.
#
# Здесь живут только правила склейки. Само пространство ID
# строится в vocab.py, и там же держится главное различие:
#
#   field_id   идентичность физического поля; кандидаты, маски,
#              predictable, корзины, unigram — всё по нему
#   token_id   идентификатор токена в словаре режима; только
#              embedding lookup
#
# Склейка меняет token_id и НИКОГДА не меняет field_id.
#
# Namespace не снимается автоматически: список объединяемых
# групп задан явно и по реальной схеме проекта. Рядом лежит
# такой же явный список НЕобъединяемых пар с причиной, чтобы
# «не склеили» было решением, а не забывчивостью.
# ============================================================


KEY_MODE_NAMESPACED = "namespaced"
KEY_MODE_SEMANTIC = "semantic"

KEY_MODES: tuple[str, ...] = (KEY_MODE_NAMESPACED, KEY_MODE_SEMANTIC)

VALUE_MODE_FIELD_SPECIFIC = "field_specific"
VALUE_MODE_SHARED = "shared"

VALUE_MODES: tuple[str, ...] = (VALUE_MODE_FIELD_SPECIFIC, VALUE_MODE_SHARED)

# Умолчания это baseline: старый словарь и старые checkpoint.
DEFAULT_KEY_MODE = KEY_MODE_NAMESPACED
DEFAULT_VALUE_MODE = VALUE_MODE_FIELD_SPECIFIC


# ------------------------------------------------------------
# СЕМАНТИЧЕСКИЕ ГРУППЫ КЛЮЧЕЙ
# ------------------------------------------------------------
#
# Девять полей профиля это буквально одно и то же измерение,
# снятое в двух местах: as-of контекст (namespace profile) и
# событие ленты (namespace profile_snapshot). Десятая группа —
# продукт: экран приложения и открытый договор называют один и
# тот же каталог, и множества их значений пересекаются.
#
# Внимание: общий key token НЕ делает поля одним полем. У
# profile__declared_income predictable=False, у
# profile_snapshot__declared_income True, и корзины у них свои —
# всё это живёт на field_id и склейкой не затрагивается.
# ------------------------------------------------------------

SEMANTIC_KEY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("declared_income", ("profile__declared_income", "profile_snapshot__declared_income")),
    ("relationship_months", ("profile__relationship_months", "profile_snapshot__relationship_months")),
    ("contracts_count", ("profile__contracts_count", "profile_snapshot__contracts_count")),
    ("active_contracts", ("profile__active_contracts", "profile_snapshot__active_contracts")),
    ("holds_credit_card", ("profile__holds_credit_card", "profile_snapshot__holds_credit_card")),
    ("holds_debit_card", ("profile__holds_debit_card", "profile_snapshot__holds_debit_card")),
    ("holds_deposit", ("profile__holds_deposit", "profile_snapshot__holds_deposit")),
    ("credit_limit", ("profile__credit_limit", "profile_snapshot__credit_limit")),
    ("credit_utilization", ("profile__credit_utilization", "profile_snapshot__credit_utilization")),
    ("product", ("product_event__product_type", "app_screen__product")),
)


# ------------------------------------------------------------
# ЧТО СОЗНАТЕЛЬНО НЕ ОБЪЕДИНЯЕТСЯ
# ------------------------------------------------------------
#
# Список нужен не для кода, а для проверки: он попадает в
# artifacts и в тест. Совпадение типа или похожее имя не
# является совпадением смысла.
# ------------------------------------------------------------

AMBIGUOUS_KEYS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("transaction__amount", "product_event__amount_or_limit", "profile__credit_limit"),
        "деньги разного смысла: сумма покупки, размер договора и кредитный лимит",
    ),
    (
        ("banner__offer", "product_event__product_type"),
        "оффер кампании это предложение, а не открытый договор; часть офферов продуктом не является",
    ),
    (
        ("transaction__merchant_city", "profile__region"),
        "место мерчанта это не место клиента, даже когда строки совпадают",
    ),
    (
        ("transaction__is_online", "transaction__is_subscription", "communication__delivered"),
        "общий тип bool при полностью разном смысле",
    ),
    (
        ("app_operation__domain", "app_screen__funnel_stage"),
        "домен операции и стадия воронки описывают разные разрезы",
    ),
)


# ------------------------------------------------------------
# ОБЩИЕ ЗНАЧЕНИЯ
# ------------------------------------------------------------
#
# Делятся только строки и булевы. Целое число это КОД, чей смысл
# задаёт поле: '1' у communication__day_of_week это понедельник,
# '1' у profile__children это один ребёнок, '1' у
# profile__salary_day это первое число. Склеить их значило бы
# ровно то же, что склеить age_B7 и amount_B7.
#
# Numeric не делится никогда: его значение это номер корзины, а
# границы корзин у каждого поля свои.
# ------------------------------------------------------------

SHARED_VALUE_ARROW_TYPES: frozenset[str] = frozenset({"string", "bool"})

SHARED_VALUE_KINDS: frozenset[str] = frozenset({KIND_CATEGORICAL, KIND_BOOLEAN})

SHARING_RULES = {
    "keys": "объединяются только пары из SEMANTIC_KEY_GROUPS; namespace автоматически не снимается",
    "values": "делятся значения kind categorical|boolean с arrow_type string|bool, по паре (arrow_type, value)",
    "numeric": "корзины никогда не делятся: значение это номер корзины, границы у каждого поля свои",
    "int_categorical": "целочисленные категории (day_of_week, hour, children, salary_day) не делятся: число это код поля",
    "specials": "[PAD] [UNK] [MASK] [EVT] [USR] [MISSING] не затрагиваются ни в одном режиме",
    "field_id": "склейка меняет только token_id; field_id, кандидаты, predictable и корзины остаются своими",
}


class SemanticRegistryError(ValueError):
    """
    Реестр склейки противоречит схеме проекта.
    """


# ------------------------------------------------------------
# ПРАВИЛА
# ------------------------------------------------------------


def check_mode(key_mode: str, value_mode: str) -> None:

    if key_mode not in KEY_MODES:
        raise SemanticRegistryError(f"key_mode должен быть одним из {KEY_MODES}, получено {key_mode!r}")

    if value_mode not in VALUE_MODES:
        raise SemanticRegistryError(
            f"categorical_value_mode должен быть одним из {VALUE_MODES}, получено {value_mode!r}"
        )


def is_baseline(key_mode: str, value_mode: str) -> bool:
    return key_mode == DEFAULT_KEY_MODE and value_mode == DEFAULT_VALUE_MODE


def read_modes(config: dict) -> tuple[str, str]:
    """
    Режимы словаря из tokenizer_config.
    """

    modes = config["modes"]

    return (modes["key_mode"], modes["categorical_value_mode"])


def group_of_key() -> dict[str, str]:
    """
    Физический ключ -> имя семантической группы.
    """

    mapping: dict[str, str] = {}

    for name, members in SEMANTIC_KEY_GROUPS:
        for key in members:
            mapping[key] = name

    return mapping


def semantic_key_name(key: str, key_mode: str) -> str:
    """
    Имя key token для физического ключа.

    В namespaced это сам ключ, в semantic — имя группы, если
    ключ в неё входит.
    """

    if key_mode == KEY_MODE_NAMESPACED:
        return key

    return group_of_key().get(key, key)


def shares_values(kind: str, arrow_type: str, value_mode: str) -> bool:
    """
    Делит ли поле свои значения с другими полями.
    """

    if value_mode == VALUE_MODE_FIELD_SPECIFIC:
        return False

    if kind == KIND_NUMERIC:
        return False

    return kind in SHARED_VALUE_KINDS and arrow_type in SHARED_VALUE_ARROW_TYPES


def value_class(field_id: int, kind: str, arrow_type: str, value: str, value_mode: str):
    """
    Класс склейки значения.

    Один класс это один token_id. Для делимого значения класс не
    содержит поля, поэтому одинаковая строка одного типа из
    разных полей попадает в один токен.
    """

    if shares_values(kind, arrow_type, value_mode):
        return ("shared", arrow_type, value)

    return ("field", int(field_id), value)


# ------------------------------------------------------------
# ПРОВЕРКА РЕЕСТРА
# ------------------------------------------------------------


def validate_registry(known_keys: Iterable[str]) -> None:
    """
    Реестр обязан говорить о существующих полях.

    Опечатка в имени ключа иначе тихо означала бы «ничего не
    склеиваем» — ровно та же ловушка, что у пустого паттерна в
    resolve_excluded.
    """

    known = set(known_keys)

    seen: dict[str, str] = {}

    for name, members in SEMANTIC_KEY_GROUPS:

        if len(members) < 2:
            raise SemanticRegistryError(f"семантическая группа {name} содержит меньше двух ключей")

        for key in members:

            if key not in known:
                raise SemanticRegistryError(
                    f"семантическая группа {name} называет ключ {key}, которого нет в реестре полей"
                )

            if key in seen:
                raise SemanticRegistryError(
                    f"ключ {key} входит сразу в две семантические группы: {seen[key]} и {name}"
                )

            seen[key] = name

    for members, reason in AMBIGUOUS_KEYS:

        for key in members:

            if key not in known:
                raise SemanticRegistryError(
                    f"список неоднозначных ключей называет {key}, которого нет в реестре полей"
                )

        pair = set(members)

        for name, group in SEMANTIC_KEY_GROUPS:

            shared = pair & set(group)

            if len(shared) > 1:
                raise SemanticRegistryError(
                    f"ключи {sorted(shared)} объявлены и объединяемыми (группа {name}), "
                    f"и неоднозначными ({reason})"
                )


def registry_as_dict() -> dict:
    """
    Реестр в том виде, в каком он едет в artifacts и в отпечаток.
    """

    return {
        "key_groups": {name: list(members) for name, members in SEMANTIC_KEY_GROUPS},
        "ambiguous_keys": [
            {"keys": list(members), "reason": reason} for members, reason in AMBIGUOUS_KEYS
        ],
        "shared_value_arrow_types": sorted(SHARED_VALUE_ARROW_TYPES),
        "shared_value_kinds": sorted(SHARED_VALUE_KINDS),
        "rules": SHARING_RULES,
    }


def registry_digest() -> str:
    """
    Отпечаток реестра: смена правил склейки обязана делать
    прежние checkpoint несовместимыми, а не менять смысл молча.
    """

    payload = json.dumps(registry_as_dict(), sort_keys=True, ensure_ascii=False)

    return sha256_bytes(payload.encode("utf-8"))
