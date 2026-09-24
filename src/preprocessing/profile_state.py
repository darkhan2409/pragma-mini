from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping

import pyarrow as pa

from src.generator.profile import PROFILE_FIELD_TYPES

from .keys import CATEGORICAL, COUNT, NUMERIC, PROFILE_KEYS, SemanticKey


# ============================================================
# АНКЕТА НА ЗАДАННУЮ ДАТУ
# ============================================================
#
# В выгрузке анкета одна: строка на клиента на границу выгрузки
# (src/generator/profile.py). Модели же нужна анкета на начало
# периода целей: иначе в её вход попадает состояние, сложившееся
# ПОСЛЕ событий, которые она же и должна восстановить.
#
# Версий анкеты здесь не заводится. Берётся конечный снимок и
# откатывается назад по самой ленте — ровно на те изменения,
# которые лента показывает.
#
# Поле попадает в модельную анкету, только если его значение на
# нужную дату СЛЕДУЕТ из данных. Три способа:
#
#   постоянное поле        значение не меняется за жизнь клиента;
#   откат по изменению     событие profile_change несёт прежнее
#                          значение, и первое изменение на
#                          границе или позже возвращает то, что
#                          было на границе;
#
# Остальные поля НЕ ВОССТАНАВЛИВАЮТСЯ и в модельную анкету не
# идут: подставить конечное значение нельзя — это и есть утечка,
# а выдумать начальное нельзя тем более. Причина у каждого
# названа в UNPROVABLE_FIELDS и уезжает в отчёт этапа.
#
# Состав полей ОДИНАКОВ для всех клиентов группы и не зависит от
# того, что случилось с конкретным клиентом после границы. Иначе
# само отсутствие поля стало бы сообщением о будущем: ровно так
# прежде работало отсутствие resolution у решения антифрода.
#
# ЧЕГО ЗДЕСЬ НЕ ПРОВЕРЯЕТСЯ. Клиент мог прийти в банк уже после
# границы; тогда анкеты на неё у банка не было вовсе. Отличить
# такого клиента лента не позволяет: отдельного события о приходе
# в банк в ней нет — приход виден только открытием первого
# договора, как и любое другое открытие. Откат в этом случае
# даёт значения, с которыми клиент пришёл, а счётчики договоров —
# ноль. Это не состояние из будущего периода целей, но и не
# состояние на границу; ограничение названо в отчёте.
# ============================================================


class ProfileStateError(ValueError):
    """
    Анкету на дату восстановить нельзя: лента противоречит
    снимку.
    """


# Поле не меняется за всю жизнь клиента: ни события изменения,
# ни пересчёта по времени у него нет.
CONSTANT_FIELDS: tuple[str, ...] = ("gender",)


# Поле меняется ТОЛЬКО событием profile_change, и событие несёт
# прежнее значение. Пересчёта по времени у этих полей нет, а
# значит между изменениями значение стоит на месте.
CHANGED_BY_EVENT: tuple[str, ...] = (
    "family_status",
    "education",
    "region",
    "city",
    "housing_type",
    "income_type",
    "declared_income",
    "industry",
    "income_day",
    "children",
)


# Чего в модельной анкете нет и почему. Текст уходит в отчёт
# этапа: исключение обязано быть названным, а не молчаливым.
UNPROVABLE_FIELDS: dict[str, str] = {
    "age": (
        "возраст пересчитывается со временем, события об этом нет, "
        "а даты рождения в данных нет вовсе"
    ),
    "pensioner": (
        "признак пересчитывается при достижении пенсионного возраста "
        "молча, без события profile_change"
    ),
    "relationship_months": (
        "стаж пересчитывается со временем без события; вычесть месяцы "
        "из снимка нельзя — снимок относится к последнему пересчёту "
        "анкеты, а не к самой границе выгрузки"
    ),
    "holds_credit_card": (
        "в очищенной ленте у открытия и закрытия договора нет ни "
        "семейства продукта, ни идентификатора договора, поэтому "
        "число открытых договоров этого вида на границу неизвестно"
    ),
    "holds_debit_card": (
        "в очищенной ленте у открытия и закрытия договора нет ни "
        "семейства продукта, ни идентификатора договора, поэтому "
        "число открытых договоров этого вида на границу неизвестно"
    ),
    "holds_deposit": (
        "в очищенной ленте у открытия и закрытия договора нет ни "
        "семейства продукта, ни идентификатора договора, поэтому "
        "число открытых договоров этого вида на границу неизвестно"
    ),
    "credit_limit": (
        "лимит это сумма по открытым кредитным картам; какой из "
        "договоров кредитная карта, очищенная лента не говорит"
    ),
    "credit_utilization": (
        "доля использования считается от остатка кредитного счёта на "
        "момент; ни счёта, ни лимита на границу восстановить нельзя"
    ),
    "contracts_count": (
        "счётчик пересчитывается раз в месяц, а не при каждом открытии "
        "договора: снимок описывает последний пересчёт анкеты, а не конец "
        "выгрузки, и откатить его на заданную дату нельзя"
    ),
    "active_contracts": (
        "счётчик пересчитывается раз в месяц, а не при открытии и закрытии "
        "договора: снимок описывает последний пересчёт анкеты, а не конец "
        "выгрузки, и откатить его на заданную дату нельзя"
    ),
}


# Поля, которые модельная анкета несёт. Порядок как в
# PROFILE_KEYS: он же задаёт порядок пар в закодированном
# профиле.
INCLUDED_FIELDS: tuple[str, ...] = tuple(
    name for name in PROFILE_KEYS if name not in UNPROVABLE_FIELDS
)


def _check_coverage() -> None:
    """
    Каждое поле анкеты названо ровно один раз.

    Новое поле в PROFILE_KEYS без решения о нём — ошибка сборки,
    а не молчаливое исключение и не молчаливое включение.
    """

    known = set(CONSTANT_FIELDS) | set(CHANGED_BY_EVENT)

    missing = sorted(set(PROFILE_KEYS) - known - set(UNPROVABLE_FIELDS))

    if missing:
        raise ProfileStateError(
            "поля анкеты без решения о восстановлении: " + ", ".join(missing)
        )

    both = sorted(known & set(UNPROVABLE_FIELDS))

    if both:
        raise ProfileStateError(
            "поля анкеты и восстанавливаются, и объявлены невосстановимыми: "
            + ", ".join(both)
        )


_check_coverage()


def typed_profile_value(
    key: SemanticKey, raw: object, field_name: str
) -> tuple[object, str | None]:
    """
    Значение профиля в виде своего ключа. Неразобранное число не
    подменяется текстом и признаком не становится.

    Счётчик объявлен категорией, но числом быть не перестал.
    Изменение профиля приходит строкой, и без привода «2» у
    прежнего значения и 2 у самого поля стали бы разными записями
    одного факта: словарь хранит запись значения вместе с его
    типом. Дробный счётчик это ошибка, а не повод молча стать
    числом с точкой.
    """

    counter = key.kind == CATEGORICAL and key.unit == COUNT

    if key.kind != NUMERIC and not counter:
        return raw, None

    text = str(raw)

    try:
        return int(text), None
    except ValueError:
        pass

    if counter:
        return None, f"значение профиля {field_name} не разобрано как целый счётчик: {text!r}"

    try:
        return float(text), None
    except ValueError:
        return None, f"значение профиля {field_name} не разобрано как число: {text!r}"


def as_declared(name: str, value: object) -> object:
    """
    Значение в том же физическом типе, в каком поле лежит в
    выгрузке.

    Прежнее значение приходит из события строкой всегда, а в
    анкете день выплаты это число. Без привода у одного ключа
    оказалось бы два типа значения, и словарь на этом
    останавливается: true и "true" — разные значения, и
    объединять их молча нельзя.
    """

    if value is None or not isinstance(value, str):
        return value

    declared = PROFILE_FIELD_TYPES[name]

    text = value.strip()

    if declared.equals(pa.bool_()):
        return text.lower() in ("true", "1", "yes")

    try:
        if pa.types.is_integer(declared):
            return int(text)
        if pa.types.is_floating(declared):
            return float(text)
    except ValueError:
        # Неразобранное число не подменяется текстом: поле
        # просто не попадает в анкету.
        return None

    return text


@dataclass(frozen=True)
class ProfileAt:
    """
    Анкета клиента на дату: поля и то, чего не хватило.
    """

    values: dict[str, object]
    notes: list[str] = field(default_factory=list)

    # Поле менялось после даты, и прежнее значение в событии не
    # названо: на дату значения не было. Считается отдельно,
    # чтобы «не было» не путалось с «не смогли».
    rolled_back_to_absent: tuple[str, ...] = ()


def profile_at(
    snapshot: Mapping[str, object] | None,
    rows: Iterable[Mapping[str, object]],
    moment: datetime,
) -> ProfileAt:
    """
    Анкета клиента на moment по конечному снимку и его ленте.

    rows — строки клиента из очищенной ленты, в их обычном
    порядке (по времени). Берутся ВСЕ строки клиента, а не
    отобранные срезом событий: откат смотрит именно на то, что
    случилось после moment.
    """

    if snapshot is None:
        return ProfileAt({})

    later = [row for row in rows if row["event_time"] >= moment]

    values: dict[str, object] = {}
    notes: list[str] = []
    absent: list[str] = []

    for name in CONSTANT_FIELDS:
        if snapshot.get(name) is not None:
            values[name] = snapshot[name]

    # --- откат по изменениям профиля ---

    first_change: dict[str, Mapping[str, object]] = {}

    for row in later:

        if row["type"] != "profile_change":
            continue

        name = row.get("field_name")

        if name in CHANGED_BY_EVENT and name not in first_change:
            first_change[name] = row

    for name in CHANGED_BY_EVENT:

        row = first_change.get(name)

        if row is None:
            value = snapshot.get(name)
        else:
            raw = row.get("old_value")

            if raw is None:
                # Прежнего значения у изменения нет: значит на
                # moment поля не было заполнено.
                absent.append(name)
                continue

            # Тип поля задаёт выгрузка, а не то, чем значение
            # приехало в событии: из события всё приходит
            # строкой.
            value = as_declared(name, raw)

            if value is None:
                notes.append(
                    f"прежнее значение поля {name} не приводится к его типу: {raw!r}"
                )
                continue

        if value is not None:
            values[name] = value

    return ProfileAt(values, notes, tuple(sorted(absent)))


__all__ = [
    "CHANGED_BY_EVENT",
    "CONSTANT_FIELDS",
    "INCLUDED_FIELDS",
    "as_declared",
    "UNPROVABLE_FIELDS",
    "ProfileAt",
    "ProfileStateError",
    "profile_at",
    "typed_profile_value",
]
