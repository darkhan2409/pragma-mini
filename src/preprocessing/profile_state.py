from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, tzinfo
from typing import Iterable, Mapping

import pyarrow as pa

from src.generator.profile import LIFELONG_TYPES as RAW_LIFELONG_TYPES
from src.generator.profile import PROFILE_FIELD_TYPES

from .keys import CATEGORICAL, COUNT, NUMERIC, PROFILE_KEYS, SemanticKey


# ============================================================
# АНКЕТА НА CUTOFF СОБЫТИЙ
# ============================================================
#
# Пример для модели это события строго раньше cutoff T и анкета
# клиента на тот же T (PROFILE_SEMANTICS). Анкета из двух частей:
#
#   Attributes @ T  недатированное состояние клиента на T —
#                   значения полей анкеты;
#   Lifelong < T    датированные вехи клиента строго раньше T.
#
# Данных позже T нет ни в событиях, ни в анкете.
#
# В выгрузке анкета одна: снимок на её границу as_of
# (src/generator/profile.py), а она может лежать позже T. Версий
# анкеты здесь не заводится: снимок откатывается назад по самой
# ленте — только на изменения с временем >= T. Вперёд снимок не
# восстанавливается: T позже as_of это ошибка, а не повод
# выдумать состояние.
#
# Вехи отката не требуют: это факты с собственным временем, и
# анкета на T — ровно те из них, что случились строго раньше T.
# Контракт тот же полуоткрытый, что у событий. Вехи лежат в
# снимке, а не в ленте, поэтому переживают окно выгрузки: клиент,
# пришедший в 2021 году, остаётся клиентом с 2021 года, хотя его
# события видны только с 2024-го. Со значениями полей вехи не
# смешиваются.
#
# Изменения анкеты (profile_change) остаются событиями ленты.
# Анкета на T — лишь их итог к этому моменту. Целями MLM они не
# становятся (dataset/targets.py): иначе итог в анкете был бы
# готовым ответом на закрытое новое значение.
#
# Поле попадает в анкету, только если оно описывает клиента, а не
# пересказывает его события, И его значение на T следует из
# данных. Способы получить значение:
#
#   постоянное поле        значение не меняется за жизнь клиента;
#   откат по изменению     событие profile_change несёт прежнее
#                          значение, и первое изменение в момент T
#                          или позже возвращает то, что было на T;
#   счёт от даты рождения  возраст меняется со временем без
#                          события, но выгрузка несёт постоянную
#                          birth_date, и по ней возраст считается
#                          на любую дату. Самой даты рождения среди
#                          полей анкеты нет: модель видит возраст
#                          на cutoff, а не дату;
#   датированные факты     у записей о работе в снимке своё
#                          время, и на T берутся те, что были
#                          известны банку строго раньше T.
#
# Остальные поля в анкету не идут (EXCLUDED_FIELDS), у каждого
# названа причина:
#
#   SHORTCUT_FIELDS    выводятся из событий истории — договоров,
#                      карт, остатков. Даже точные на T, они были
#                      бы готовым ответом на закрытое событие:
#                      закрыт product_opened кредитной карты, а
#                      holds_credit_card = True его выдаёт;
#   FROM_LIFELONG      выводятся из вехи. Модель получает саму
#                      веху с её временем, а не посчитанный из
#                      неё признак.
#
# Состав полей ОДИНАКОВ для всех клиентов группы. Иначе само
# отсутствие поля стало бы сообщением о клиенте.
#
# Клиент мог прийти в банк уже после T. Событий до T у него нет,
# вехи bank_registered до T нет тоже, а откат полей даёт
# значения, с которыми клиент пришёл.
# ============================================================


# Смысл анкеты в артефактах этапов 04 и 05: Attributes на тот же
# cutoff, до которого клиенту доступны события, и Lifelong строго
# раньше него.
PROFILE_SEMANTICS = "attributes_and_lifelong_at_event_cutoff"

# Типы вех Lifelong. Источник истины — контракт выгрузки.
LIFELONG_TYPES: tuple[str, ...] = RAW_LIFELONG_TYPES

# Шаг меток стажа, в месяцах: 0-5, 6-11, 12-17, … Граница задана
# шагом, а не данными train.
JOB_TENURE_STEP = 6


class ProfileStateError(ValueError):
    """
    Состав полей анкеты объявлен противоречиво.
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


# Поле считается на дату из birth_date выгрузки:
#
#   age  полных лет на местную дату момента — с учётом дня и
#        месяца рождения, а не разностью годов.
#
# Признака пенсионера среди полей нет: он был производным от
# возраста и начального вида дохода, отдельного банковского
# статуса за ним не стоит, а обе его части модель уже видит —
# возраст полем, вид дохода полем и изменениями анкеты в ленте.
FROM_BIRTH_DATE: tuple[str, ...] = ("age",)


# Поле считается на дату по датированным фактам снимка:
#
#   job_tenure_months  стаж на последней работе по найму, о которой
#                      банк узнал строго раньше T: полных месяцев от
#                      её начала до T, меткой полугодия. Нет такой
#                      работы, последняя запись — «работы нет» или
#                      вид дохода на T не наёмный — поля нет.
FROM_DATED_FACTS: tuple[str, ...] = ("job_tenure_months",)

# Виды дохода работы по найму: стаж бывает только при них. Записи
# о работе и вид дохода — разные факты анкеты, и расходятся они,
# например, у безработного, чья новая работа началась позже, чем
# банк о ней узнал: вид дохода на T ещё unemployed.
SALARIED_INCOME_TYPES: tuple[str, ...] = ("employed", "state_employee")


_FROM_CONTRACTS = (
    "производное состояние договоров клиента: его меняют открытия и "
    "закрытия продуктов из истории, и на закрытом событии поле стало бы "
    "готовым ответом"
)

# Выводятся из событий истории. В анкету не идут, даже если их
# можно было бы посчитать на T.
SHORTCUT_FIELDS: dict[str, str] = {
    "contracts_count": _FROM_CONTRACTS,
    "active_contracts": _FROM_CONTRACTS,
    "holds_credit_card": _FROM_CONTRACTS,
    "holds_debit_card": _FROM_CONTRACTS,
    "holds_deposit": _FROM_CONTRACTS,
    "credit_limit": (
        "сумма лимитов открытых кредитных карт: производное от открытий и "
        "закрытий продуктов в истории"
    ),
    "credit_utilization": (
        "доля использования лимита: производное от остатков и операций "
        "по кредитному счёту в истории"
    ),
}


# Выводятся из вехи Lifelong.
FROM_LIFELONG: dict[str, str] = {
    "relationship_months": (
        "стаж — производное от вехи bank_registered: модель получает "
        "саму дату в Lifelong, а давность до cutoff видит временным каналом"
    ),
}


# Чего в анкете нет и почему. Текст уходит в метаданные этапа:
# исключение обязано быть названным, а не молчаливым.
EXCLUDED_FIELDS: dict[str, str] = {**SHORTCUT_FIELDS, **FROM_LIFELONG}


# Поля, которые модельная анкета несёт. Порядок как в
# PROFILE_KEYS: он же задаёт порядок пар в закодированном
# профиле.
INCLUDED_FIELDS: tuple[str, ...] = tuple(
    name for name in PROFILE_KEYS if name not in EXCLUDED_FIELDS
)


def _check_coverage() -> None:
    """
    Каждое поле анкеты названо ровно один раз.

    Новое поле в PROFILE_KEYS без решения о нём — ошибка сборки,
    а не молчаливое исключение и не молчаливое включение.
    """

    groups = (
        CONSTANT_FIELDS, CHANGED_BY_EVENT, FROM_BIRTH_DATE, FROM_DATED_FACTS,
        tuple(SHORTCUT_FIELDS), tuple(FROM_LIFELONG),
    )

    named = [name for group in groups for name in group]

    twice = sorted({name for name in named if named.count(name) > 1})

    if twice:
        raise ProfileStateError("поля анкеты названы больше одного раза: " + ", ".join(twice))

    missing = sorted(set(PROFILE_KEYS) - set(named))

    if missing:
        raise ProfileStateError("поля анкеты без решения: " + ", ".join(missing))


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
    Анкета клиента на дату: поля (Attributes), вехи (Lifelong) и
    то, чего не хватило.
    """

    values: dict[str, object]
    notes: list[str] = field(default_factory=list)

    # Поле менялось после даты, и прежнее значение в событии не
    # названо: на дату значения не было. Считается отдельно,
    # чтобы «не было» не путалось с «не смогли».
    rolled_back_to_absent: tuple[str, ...] = ()

    # Вехи строго раньше даты: (тип, время) в порядке снимка.
    lifelong: tuple[tuple[str, datetime], ...] = ()


def profile_at(
    snapshot: Mapping[str, object] | None,
    rows: Iterable[Mapping[str, object]],
    moment: datetime,
    timezone: tzinfo,
) -> ProfileAt:
    """
    Анкета клиента на moment по конечному снимку и его ленте.

    rows — строки клиента из очищенной ленты, в их обычном
    порядке (по времени). Берутся ВСЕ строки клиента, а не
    отобранные срезом событий: откат смотрит именно на то, что
    случилось после moment.

    timezone — пояс банка. Дата рождения это календарная дата
    банка, и возраст считается на МЕСТНУЮ дату moment: по UTC
    день рождения у самой границы сдвинул бы возраст на год.
    """

    if snapshot is None:
        return ProfileAt({})

    rows = list(rows)

    # Вехи: только строго раньше moment. Порядок снимка — по
    # времени — сохраняется.
    lifelong = tuple(
        (item["type"], item["event_time"])
        for item in snapshot.get("lifelong") or ()
        if item["event_time"] < moment
    )

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

    # --- счёт от даты рождения ---

    born = snapshot.get("birth_date")

    # Возраст — только от даты рождения и только на moment: ни
    # снимок, ни конец выгрузки в него не входят. Даты рождения нет
    # — нет и возраста; дата постоянна, поэтому её отсутствие о
    # будущем клиента ничего не говорит.
    if born is not None:
        values["age"] = _full_years(born, moment.astimezone(timezone).date())

    # --- датированные факты ---

    known = [item for item in snapshot.get("employment") or () if item["record_time"] < moment]

    if known and values.get("income_type") in SALARIED_INCOME_TYPES:

        start = max(known, key=lambda item: item["record_time"])["start_date"]

        if start is not None:
            values["job_tenure_months"] = tenure_label(
                _full_months(start, moment.astimezone(timezone).date())
            )

    return ProfileAt(values, notes, tuple(sorted(absent)), lifelong)


def tenure_label(months: int) -> str:
    """
    Метка полугодия стажа: 0-5, 6-11, 12-17, … Стаж 41 — «36-41»,
    стаж 42 — «42-47».
    """

    low = (int(months) // JOB_TENURE_STEP) * JOB_TENURE_STEP

    return f"{low}-{low + JOB_TENURE_STEP - 1}"


def _full_months(start: date, day: date) -> int:
    """
    Полных месяцев от start до day: месяц засчитан, когда число
    месяца дошло до числа начала.
    """

    return (day.year - start.year) * 12 + (day.month - start.month) - (day.day < start.day)


def _full_years(born: date, day: date) -> int:
    """
    Полных лет на day — тем же правилом, что Persona.age_at.
    """

    return day.year - born.year - ((day.month, day.day) < (born.month, born.day))


__all__ = [
    "CHANGED_BY_EVENT",
    "CONSTANT_FIELDS",
    "EXCLUDED_FIELDS",
    "FROM_BIRTH_DATE",
    "FROM_DATED_FACTS",
    "FROM_LIFELONG",
    "INCLUDED_FIELDS",
    "JOB_TENURE_STEP",
    "LIFELONG_TYPES",
    "PROFILE_SEMANTICS",
    "SALARIED_INCOME_TYPES",
    "SHORTCUT_FIELDS",
    "as_declared",
    "ProfileAt",
    "ProfileStateError",
    "profile_at",
    "tenure_label",
    "typed_profile_value",
]
