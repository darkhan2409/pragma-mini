from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# ИДЕЯ
# ============================================================
#
# Отношения и отклонения считаются из наблюдаемых значений и
# возвращают ИСХОДНЫЕ числа: ни квантилей, ни нормализации по
# датасету, ни корзин здесь нет — это работа следующего этапа и
# только на train.
#
# Неизвестный или нулевой знаменатель это не ноль и не
# бесконечность, а причина. Значение остаётся пустым, а рядом
# лежит объяснение: дохода не знаем, лимита не знаем, прошлого
# для сравнения ещё нет.
#
# У каждого расчёта есть происхождение: из какого события, какой
# версии профиля и какой строки справочника он получен. Это
# нужно будущему Masker, чтобы не маскировать значение и его
# производную по отдельности.
# ============================================================


FORMULAS_VERSION = "1.2.0"

REASON_NO_INCOME = "income_unknown"
REASON_NO_LIMIT = "limit_unknown"
# Лимит у сущности известен, но это не кредитный лимит: сумма
# вклада или тело кредита. Делить операцию на них бессмысленно.
REASON_NO_CREDIT_LIMIT = "no_credit_limit"
REASON_NO_BALANCE = "balance_unknown"
REASON_NO_HISTORY = "no_visible_past"
REASON_ZERO = "zero_denominator"

# Семейство продуктов, у которого amount_or_limit это КРЕДИТНЫЙ
# ЛИМИТ. У вклада то же поле означает сумму размещения.
CREDIT_LIMIT_FAMILY = "credit_card"


@dataclass(frozen=True)
class Derived:
    """
    Расчётное значение или причина, по которой его нет.
    """

    key: str
    value: float | None
    reason: str | None
    derived_from: tuple[dict, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "reason": self.reason,
            "derived_from": [dict(item) for item in self.derived_from],
        }


def _event_ref(row: dict, key: str) -> dict:
    return {"kind": "event", "stable_event_index": row["stable_event_index"], "key": key}


def _profile_ref(profile: dict, key: str) -> dict:
    return {
        "kind": "profile",
        "client_id": profile["client_id"],
        "key": key,
    }


def _ratio(key: str, numerator: float, denominator, sources: tuple[dict, ...], missing: str) -> Derived:

    if denominator is None:
        return Derived(key, None, missing, sources)

    if float(denominator) == 0.0:
        return Derived(key, None, REASON_ZERO, sources)

    return Derived(key, float(numerator) / float(denominator), None, sources)


def amount_ratios(row: dict, profile: dict | None, limits: dict[str, dict]) -> list[Derived]:
    """
    Отношения суммы операции к доходу, лимиту и остатку счёта.

    Доход берётся из версии профиля, действовавшей В МОМЕНТ
    ОПЕРАЦИИ, а не на cutoff: иначе покупка двухлетней давности
    делилась бы на сегодняшний доход, и в признак попадало бы
    знание из будущего этой строки.

    Лимит — кредитный лимит договора той же сущности, видимого к
    этому моменту. Сумма вклада и тело кредита лимитом не
    являются: делить на них операцию бессмысленно.

    Остаток — из самой строки операции.
    """

    amount = row.get("amount")

    if amount is None:
        return []

    out: list[Derived] = []

    income = None if profile is None else profile.get("declared_income")

    sources = (_event_ref(row, "transaction_amount"),)

    out.append(
        _ratio(
            "amount_to_declared_income",
            amount,
            income,
            sources if profile is None else sources + (_profile_ref(profile, "declared_income"),),
            REASON_NO_INCOME,
        )
    )

    ref = row.get("contract_ref") or row.get("card_ref") or row.get("account_ref")

    known = limits.get(ref) if ref else None

    entity_sources = (
        sources
        if ref is None
        else sources + ({"kind": "entity", "ref": ref, "key": "amount_or_limit"},)
    )

    if known is None:
        out.append(Derived("amount_to_limit", None, REASON_NO_LIMIT, entity_sources))
    elif known.get("product_family") != CREDIT_LIMIT_FAMILY:
        out.append(Derived("amount_to_limit", None, REASON_NO_CREDIT_LIMIT, entity_sources))
    else:
        out.append(
            _ratio("amount_to_limit", amount, known["value"], entity_sources, REASON_NO_LIMIT)
        )

    balance = row.get("balance_after")

    out.append(
        _ratio(
            "amount_to_balance_after",
            amount,
            balance,
            sources + (_event_ref(row, "balance_after"),) if balance is not None else sources,
            REASON_NO_BALANCE,
        )
    )

    return out


def deviation_from_past(row: dict, past: list[float]) -> Derived:
    """
    Во сколько раз сумма отличается от среднего по прошлым
    видимым операциям того же типа.

    Прошлого нет — есть причина, а не ноль и не единица.
    """

    amount = row.get("amount")

    sources = (_event_ref(row, "transaction_amount"),)

    if amount is None:
        return Derived("amount_to_client_average", None, REASON_NO_HISTORY, sources)

    if not past:
        return Derived("amount_to_client_average", None, REASON_NO_HISTORY, sources)

    average = sum(past) / len(past)

    if average == 0.0:
        return Derived("amount_to_client_average", None, REASON_ZERO, sources)

    return Derived(
        "amount_to_client_average",
        float(amount) / average,
        None,
        sources + ({"kind": "client_past", "events": len(past), "key": "transaction_amount"},),
    )


def note_limit(limits: dict[str, dict], row: dict) -> None:
    """
    Запоминает видимый лимит или сумму договора по сущностям
    строки ВМЕСТЕ С СЕМЕЙСТВОМ продукта.

    Одно поле amount_or_limit означает разное: у кредитной карты
    это лимит, у вклада сумма размещения, у кредита тело долга.
    Без семейства отношение «сумма к лимиту» считалось бы и по
    вкладу тоже.

    Вызывается по ходу истории: лимит, объявленный позже, к
    прошлой операции не применяется, и знание из будущего в
    отношение не попадает. Сущность названа локальной ссылкой,
    поэтому сырой идентификатор в расчёт не входит.
    """

    value = row.get("amount_or_limit")

    if value is None:
        return

    family = row.get("product_family")

    for column in ("contract_ref", "card_ref", "account_ref"):

        ref = row.get(column)

        if ref is not None:
            limits[ref] = {"value": float(value), "product_family": family}


def merchant_provenance(ref: str | None) -> tuple[dict, ...]:
    """
    Происхождение расшифровки точки: справочник и локальная
    ссылка. Сырой идентификатор наружу не выходит.
    """

    if ref is None:
        return ()

    return ({"kind": "catalog", "table": "merchants", "via": "outlet_ref", "ref": ref},)


def income_moments(rows: list[dict], income_types: frozenset[str]) -> list[datetime]:
    return [row["event_time"] for row in rows if row["event_type"] in income_types]


__all__ = [
    "FORMULAS_VERSION",
    "REASON_NO_BALANCE",
    "REASON_NO_HISTORY",
    "REASON_NO_INCOME",
    "REASON_NO_LIMIT",
    "REASON_ZERO",
    "Derived",
    "amount_ratios",
    "deviation_from_past",
    "income_moments",
    "merchant_provenance",
    "note_limit",
]
