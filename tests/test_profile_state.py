from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator.config import EVENT_TYPE_SOURCE, GENERATOR_VERSION, SCHEMA_VERSION
from src.generator.emit import EVENTS_SCHEMA
from src.generator.profile import PROFILE_SCHEMA
from src.preprocessing.profile_state import (
    EXCLUDED_FIELDS,
    FROM_BIRTH_DATE,
    INCLUDED_FIELDS,
    SHORTCUT_FIELDS,
    profile_at,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Пример модели это события строго раньше cutoff T и анкета —
# Attributes на тот же T. Снимок анкеты в выгрузке лежит на её
# границу as_of, возможно позже T, и откатывается по ленте. Вехи
# Lifelong проверяются отдельно, в test_lifelong.py.
#
# Здесь проверяется не то, что функция согласна сама с собой, а
# независимые утверждения:
#
#   изменение ДО T входит в анкету;
#
#   любые события В МОМЕНТ T и позже не меняют ни анкеты, ни
#   токенов, ни событий примера;
#
#   поля, выводимые из событий истории (договоры, карты,
#   остатки), в анкету не попадают вовсе;
#
#   изменение анкеты остаётся событием истории.
#
# Ожидаемые значения написаны здесь руками, а не получены тем же
# восстановлением.
# ============================================================


UTC = timezone.utc

# Пояс банка: граница 2026-01-01 00:00 UTC это уже 05:00 местного
# 1 января.
BANK = timezone(timedelta(hours=5))


def when(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


CUTOFF = when("2026-01-01T00:00:00")


def change(moment: str, field: str, old: str | None, new: str | None) -> dict:
    return {
        "event_time": when(moment),
        "type": "profile_change",
        "field_name": field,
        "old_value": old,
        "new_value": new,
    }


def product(moment: str, kind: str) -> dict:
    return {"event_time": when(moment), "type": kind, "field_name": None, "old_value": None}


SNAPSHOT = {
    "birth_date": date(1984, 6, 15),
    "gender": "F",
    "city": "Shymkent",
    "region": "Turkestan",
    "children": 2,
    "declared_income": 700_000,
    "family_status": "married",
    "education": "higher",
    "housing_type": "own_flat",
    "income_type": "employed",
    "industry": "trade",
    "income_day": 10,
    "contracts_count": 5,
    "active_contracts": 3,
    # Возраста и пенсионера в выгрузке больше нет. Здесь они
    # оставлены устаревшими нарочно: анкета не имеет права их
    # читать — возраст считается от birth_date на cutoff, а
    # пенсионера среди полей нет вовсе.
    "age": 42,
    "pensioner": True,
    # Невосстановимые поля в снимке есть и обязаны остаться за
    # бортом.
    "relationship_months": 90,
    "holds_credit_card": True,
    "holds_debit_card": True,
    "holds_deposit": False,
    "credit_limit": 300_000.0,
    "credit_utilization": 0.4,
}


# ============================================================
# СОСТАВ ПОЛЕЙ
# ============================================================


def test_excluded_fields_never_reach_the_profile():
    """
    Исключённое поле не попадает в анкету даже тогда, когда в
    снимке оно заполнено.
    """

    state = profile_at(SNAPSHOT, [], CUTOFF, BANK)

    leaked = sorted(set(state.values) & set(EXCLUDED_FIELDS))

    assert not leaked, f"в анкету прошли исключённые поля: {leaked}"


def test_field_set_does_not_depend_on_the_client():
    """
    Состав полей один и тот же у клиента с изменениями и без.

    Иначе само отсутствие поля сообщало бы, что с клиентом
    что-то случилось после cutoff.
    """

    quiet = profile_at(SNAPSHOT, [], CUTOFF, BANK)

    busy = profile_at(
        SNAPSHOT,
        [
            change("2026-02-01T10:00:00", "city", "Astana", "Shymkent"),
            product("2026-03-01T10:00:00", "product_opened"),
            product("2026-04-01T10:00:00", "product_closed"),
        ],
        CUTOFF, BANK,
    )

    assert set(quiet.values) == set(busy.values)


# ============================================================
# ГЛАВНОЕ: БУДУЩЕЕ НЕ МЕНЯЕТ АНКЕТУ
# ============================================================


def test_events_after_the_cutoff_do_not_change_any_value():
    """
    Одна и та же история до cutoff, разное после — анкета одна.
    """

    before = [change("2025-06-01T10:00:00", "city", "Almaty", "Astana")]

    first = profile_at(dict(SNAPSHOT, city="Astana"), before, CUTOFF, BANK)

    second = profile_at(
        SNAPSHOT,
        before
        + [
            change("2026-02-01T10:00:00", "city", "Astana", "Shymkent"),
            product("2026-02-05T10:00:00", "product_opened"),
            product("2026-03-05T10:00:00", "product_closed"),
            product("2026-04-05T10:00:00", "product_opened"),
        ],
        CUTOFF, BANK,
    )

    assert first.values == second.values


def test_reconstructed_values_are_the_ones_written_by_hand():
    """
    Ожидаемое написано руками, а не получено той же функцией.
    """

    rows = [
        change("2025-06-01T10:00:00", "city", "Almaty", "Astana"),
        change("2026-02-01T10:00:00", "city", "Astana", "Shymkent"),
        change("2026-02-02T10:00:00", "declared_income", "500000", "700000"),
        product("2026-02-05T10:00:00", "product_opened"),
        product("2026-03-05T10:00:00", "product_closed"),
    ]

    state = profile_at(SNAPSHOT, rows, CUTOFF, BANK)

    assert state.values == {
        "gender": "F",
        # Переезд после cutoff откатан, переезд до него сохранён.
        "city": "Astana",
        "region": "Turkestan",
        "children": 2,
        # Прежнее значение изменения — число, а не строка.
        "declared_income": 500_000,
        "family_status": "married",
        "education": "higher",
        "housing_type": "own_flat",
        "income_type": "employed",
        "industry": "trade",
        "income_day": 10,
        # Родилась 15.06.1984: на 1 января 2026 ей 41, а не 42 из
        # снимка. Пенсионера среди полей нет, что бы ни стояло в
        # снимке.
        "age": 41,
    }


# ============================================================
# ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ
# ============================================================


def test_change_before_the_cutoff_does_change_the_field():
    """
    Без этого первое утверждение выполнял бы и пустой словарь.
    """

    quiet = profile_at(SNAPSHOT, [], CUTOFF, BANK)

    moved = profile_at(
        dict(SNAPSHOT, city="Astana"),
        [change("2025-06-01T10:00:00", "city", "Almaty", "Astana")],
        CUTOFF, BANK,
    )

    assert quiet.values["city"] == "Shymkent"
    assert moved.values["city"] == "Astana"

    # И расходятся они ровно в одном поле.
    differing = {
        name
        for name in set(quiet.values) | set(moved.values)
        if quiet.values.get(name) != moved.values.get(name)
    }

    assert differing == {"city"}


def test_product_events_do_not_reach_the_profile_at_all():
    """
    Счётчики договоров из анкеты исключены: они пересчитываются
    раз в месяц, и снимок относится к последнему пересчёту, а не
    к концу выгрузки. Значит открытие и закрытие договора — с
    любой стороны границы — анкету не трогают.
    """

    quiet = profile_at(SNAPSHOT, [], CUTOFF, BANK)

    busy = profile_at(
        SNAPSHOT,
        [
            product("2025-06-01T10:00:00", "product_opened"),
            product("2026-06-01T10:00:00", "product_opened"),
            product("2026-07-01T10:00:00", "product_closed"),
        ],
        CUTOFF, BANK,
    )

    assert quiet.values == busy.values

    assert "contracts_count" not in quiet.values
    assert "active_contracts" not in quiet.values


# ============================================================
# ГРАНИЦЫ И КРАЙНИЕ СЛУЧАИ
# ============================================================


def test_event_exactly_at_the_cutoff_belongs_to_the_future():
    """
    Период целей это [target_start, target_end): событие ровно на
    границе уже цель, и анкета обязана его откатить.
    """

    state = profile_at(
        SNAPSHOT,
        [change("2026-01-01T00:00:00", "city", "Karaganda", "Shymkent")],
        CUTOFF, BANK,
    )

    assert state.values["city"] == "Karaganda"


def test_event_a_microsecond_before_the_cutoff_stays_in_the_past():

    state = profile_at(
        SNAPSHOT,
        [change("2025-12-31T23:59:59.999999", "city", "Karaganda", "Shymkent")],
        CUTOFF, BANK,
    )

    assert state.values["city"] == "Shymkent"


def test_several_changes_of_one_field_roll_back_to_the_earliest():
    """
    Откат идёт к ПЕРВОМУ изменению после cutoff, а не к
    последнему: между ними значение уже менялось.
    """

    rows = [
        change("2026-02-01T10:00:00", "declared_income", "400000", "550000"),
        change("2026-04-01T10:00:00", "declared_income", "550000", "700000"),
    ]

    assert profile_at(SNAPSHOT, rows, CUTOFF, BANK).values["declared_income"] == 400_000


def test_missing_old_value_means_the_field_was_empty():
    """
    Изменение без прежнего значения говорит, что на cutoff поля
    не было заполнено. Подставлять конечное нельзя.
    """

    state = profile_at(
        SNAPSHOT, [change("2026-02-01T10:00:00", "industry", None, "trade")], CUTOFF, BANK
    )

    assert "industry" not in state.values
    assert state.rolled_back_to_absent == ("industry",)


def test_client_without_a_questionnaire_gets_an_empty_profile():

    assert profile_at(None, [product("2026-02-01T10:00:00", "product_opened")], CUTOFF, BANK).values == {}


def test_unknown_change_field_is_ignored_not_guessed():
    """
    Изменение поля, которого в анкете нет, ничего не меняет.
    """

    state = profile_at(
        SNAPSHOT,
        [change("2026-02-01T10:00:00", "consent_marketing", "true", "false")],
        CUTOFF, BANK,
    )

    assert state.values == profile_at(SNAPSHOT, [], CUTOFF, BANK).values


# ============================================================
# СОСТОЯНИЕ НА CUTOFF: ДВА ПРОСТЫХ СЛУЧАЯ
# ============================================================


def test_change_before_cutoff_is_the_state_at_cutoff():
    """
    2024-01 город Almaty, 2024-06 переезд в Astana, cutoff 2024-12:
    на cutoff клиент живёт в Astana.
    """

    state = profile_at(
        dict(SNAPSHOT, city="Astana"),
        [change("2024-06-01T10:00:00", "city", "Almaty", "Astana")],
        when("2024-12-01T00:00:00"), BANK,
    )

    assert state.values["city"] == "Astana"


def test_change_after_cutoff_is_rolled_back():
    """
    2024-01 город Almaty, 2025-02 переезд в Astana, cutoff 2024-12:
    на cutoff клиент ещё в Almaty, хотя снимок уже говорит Astana.
    """

    state = profile_at(
        dict(SNAPSHOT, city="Astana"),
        [change("2025-02-01T10:00:00", "city", "Almaty", "Astana")],
        when("2024-12-01T00:00:00"), BANK,
    )

    assert state.values["city"] == "Almaty"


def test_shortcut_fields_are_the_event_derived_ones():
    """
    Поля, которые выводятся из событий истории, объявлены
    shortcut-полями и в анкету не входят.
    """

    expected = {
        "contracts_count", "active_contracts", "holds_credit_card", "holds_debit_card",
        "holds_deposit", "credit_limit", "credit_utilization",
    }

    assert set(SHORTCUT_FIELDS) == expected
    assert not expected & set(INCLUDED_FIELDS)


# ============================================================
# ВОЗРАСТ ОТ ДАТЫ РОЖДЕНИЯ
# ============================================================
#
# Возраст меняется со временем без события, поэтому снимок для
# него бесполезен: он описывает конец выгрузки. Возраст — полных
# лет на местную дату cutoff по постоянной birth_date, с учётом
# дня и месяца. Признака пенсионера среди полей нет: он был
# производным от возраста и начального вида дохода.
# ============================================================


def born(day: date, **fields) -> dict:
    return dict(SNAPSHOT, birth_date=day, **fields)


def local(text: str) -> datetime:
    """
    Момент по местному времени банка.
    """

    return datetime.fromisoformat(text).replace(tzinfo=BANK)


def age_at(day: date, moment: datetime) -> int:
    return profile_at(born(day), [], moment, BANK).values["age"]


def test_only_age_is_counted_from_the_birth_date():

    assert FROM_BIRTH_DATE == ("age",)
    assert set(FROM_BIRTH_DATE) <= set(INCLUDED_FIELDS)
    assert not set(FROM_BIRTH_DATE) & set(EXCLUDED_FIELDS)


def test_age_after_this_years_birthday():

    assert age_at(date(1990, 3, 10), local("2026-05-01T00:00:00")) == 36


def test_age_before_this_years_birthday_is_one_less():

    assert age_at(date(1990, 6, 10), local("2026-05-01T00:00:00")) == 35


def test_age_on_the_birthday_itself():
    """
    Cutoff ровно в день рождения: год уже исполнился. Мгновением
    раньше — ещё нет.
    """

    assert age_at(date(1990, 5, 1), local("2026-05-01T00:00:00")) == 36
    assert age_at(date(1990, 5, 1), local("2026-04-30T23:59:59")) == 35


def test_one_birth_date_gives_the_age_of_each_cutoff():

    day = date(1990, 5, 1)

    ages = [
        age_at(day, local(moment))
        for moment in ("2025-05-01T00:00:00", "2026-01-01T00:00:00",
                       "2026-05-01T00:00:00", "2026-09-01T00:00:00")
    ]

    assert ages == [35, 35, 36, 36]


def test_age_is_counted_on_the_local_date_of_the_cutoff():
    """
    19:00 UTC 31 декабря — это уже 1 января у банка. День рождения
    1 января наступил, хотя по UTC дата ещё прежняя.
    """

    assert age_at(date(1990, 1, 1), when("2025-12-31T19:00:00")) == 36


def test_age_of_the_snapshot_is_never_read():
    """
    В снимке устаревший возраст 42. Анкета берёт возраст только от
    даты рождения, а без неё возраста нет вовсе — ни из снимка, ни
    подставленного.
    """

    assert SNAPSHOT["age"] == 42

    assert profile_at(born(date(1990, 1, 1)), [], CUTOFF, BANK).values["age"] == 36

    snapshot = dict(SNAPSHOT)
    del snapshot["birth_date"]

    assert "age" not in profile_at(snapshot, [], CUTOFF, BANK).values


def test_events_after_the_cutoff_do_not_move_the_age():

    later = [change("2026-03-01T10:00:00", "income_type", "employed", "pensioner")]

    assert profile_at(born(date(1990, 1, 1)), later, CUTOFF, BANK).values["age"] == 36


def test_pensioner_is_not_an_attribute():
    """
    Ни полем анкеты, ни исключённым полем, ни ключом словаря, ни
    изменением анкеты пенсионер больше не бывает — даже если в
    снимке он заполнен, а клиенту за 63.
    """

    from src.preprocessing.keys import CHANGEABLE_PROFILE_FIELDS, PROFILE_KEYS
    from src.tokenization.schema import SemanticSchema

    for names in (INCLUDED_FIELDS, EXCLUDED_FIELDS, PROFILE_KEYS, CHANGEABLE_PROFILE_FIELDS):
        assert "pensioner" not in names

    state = profile_at(
        born(date(1950, 1, 1), income_type="pensioner", pensioner=True), [], CUTOFF, BANK
    )

    assert state.values["age"] == 76
    assert "pensioner" not in state.values

    assert not [key for key in SemanticSchema().keys if "pensioner" in key]


def test_attributes_are_exactly_the_thirteen():
    """
    Прежние поля без пенсионера и стаж на месте работы — ровно 13.
    Порядок в кортеже — порядок объявления ключей, смысла он не
    несёт: токены анкеты идут по номеру ключа.
    """

    assert set(INCLUDED_FIELDS) == {
        "age", "gender", "family_status", "education", "region", "city", "housing_type",
        "income_type", "declared_income", "industry", "income_day", "children",
        "job_tenure_months",
    }
    assert len(INCLUDED_FIELDS) == 13


def test_raw_profile_keeps_the_birth_date_not_the_age():

    assert "birth_date" in PROFILE_SCHEMA.names

    for name in ("age", "pensioner"):
        assert name not in PROFILE_SCHEMA.names


# ============================================================
# ТО ЖЕ, НО ЧЕРЕЗ ВЕСЬ ПУТЬ ДАННЫХ
# ============================================================
#
# Выше проверено восстановление. Ниже — что оно доезжает до
# модели: выгрузка, препроцессинг и токены профиля.
# ============================================================


RAW_CLIENT = "c000000000001"

# Граница выгрузки write_raw по умолчанию: 1 июля 2026 у банка.
# Снимок анкеты описывает клиента перед ней.
END = "2026-07-01T00:00:00+05:00"

AS_OF = datetime.fromisoformat(END).astimezone(UTC)


def raw_event(client: str, moment: str, payload: dict) -> dict:
    return {
        "client_id": client,
        "event_time": when(moment).astimezone(UTC).isoformat().replace("+00:00", "+00:00"),
        "source": EVENT_TYPE_SOURCE[payload["type"]],
        "payload": json.dumps(payload, ensure_ascii=False),
    }


def write_raw(directory, events: list[dict], snapshot: dict, end: str = END) -> None:
    """
    Выгрузка группы по контракту RAW: две таблицы и манифест.
    end — граница выгрузки, она же as_of снимка.
    """

    directory.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(events, schema=EVENTS_SCHEMA)

    pq.write_table(table, directory / "events.parquet", compression="zstd")

    row = {name: None for name in PROFILE_SCHEMA.names}
    row["as_of"] = datetime.fromisoformat(end).astimezone(UTC)
    row["employment"] = []
    row["lifelong"] = []
    row.update(snapshot)
    row["client_id"] = RAW_CLIENT

    profile = pa.Table.from_pylist([row], schema=PROFILE_SCHEMA)

    pq.write_table(profile, directory / "profile.parquet", compression="zstd")

    def digest(name: str) -> str:
        return hashlib.sha256((directory / name).read_bytes()).hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "timezone": "Asia/Almaty",
        "period_start": "2024-01-01T00:00:00+05:00",
        # Выгрузка идёт дальше cutoff val (1 мая): события после T
        # в ней есть, и тесты проверяют, что они никуда не доходят.
        "period_end": end,
        "seed": 1,
        "world_seed": 1,
        "total_clients": 1,
        "community_size": 1,
        "chunk_clients": 1,
        "generation_config_sha256": "0" * 64,
        "reference_sha256": {"merchants": "0" * 64, "product_timeline": "0" * 64},
        "events_rows": table.num_rows,
        "profile_rows": profile.num_rows,
        "events_sha256": digest("events.parquet"),
        "profile_sha256": digest("profile.parquet"),
    }

    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare(stage, events: list[dict], snapshot: dict, group: str = "val", end: str = END):
    """
    Выгрузка → препроцессинг → история клиента на cutoff группы.
    """

    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.read import Group
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir

    write_raw(raw_group_dir(group), events, snapshot, end)

    settings = PreprocessingConfig.load(None)

    build_group(raw_group_dir(group), group_dir(group), settings, group)

    return Group(group).history(RAW_CLIENT, settings.windows[group].final_cutoff)


def val_cutoff() -> datetime:

    from src.preprocessing.settings import PreprocessingConfig

    return PreprocessingConfig.load(None).windows["val"].final_cutoff


# cutoff val — полночь 1 мая 2026 у банка (19:00 UTC 30 апреля).

EARLY = [
    raw_event(RAW_CLIENT, "2024-03-01T09:00:00", {
        "type": "purchase", "amount": 5000, "direction": "debit", "status": "approved"}),
    raw_event(RAW_CLIENT, "2025-06-01T10:00:00", {
        "type": "profile_change", "field_name": "city", "old_value": "Almaty",
        "new_value": "Astana", "change_source": "client", "confirmed": True}),
    # Период целей, но всё ещё раньше cutoff: в анкету входит.
    raw_event(RAW_CLIENT, "2026-02-01T10:00:00", {
        "type": "profile_change", "field_name": "city", "old_value": "Astana",
        "new_value": "Shymkent", "change_source": "client", "confirmed": True}),
    raw_event(RAW_CLIENT, "2026-03-01T10:00:00", {
        "type": "product_opened", "product_id": "prd_test", "reason": "application_approved"}),
]

# Всё позже cutoff: ни в события, ни в анкету.
AFTER = [
    raw_event(RAW_CLIENT, "2026-05-15T10:00:00", {
        "type": "product_opened", "product_id": "prd_late", "reason": "application_approved"}),
    raw_event(RAW_CLIENT, "2026-06-01T10:00:00", {
        "type": "profile_change", "field_name": "city", "old_value": "Shymkent",
        "new_value": "Astana", "change_source": "client", "confirmed": True}),
]

# Снимки на границу выгрузки (1 июля): у второго она уже знает
# о событиях после cutoff.
QUIET_SNAPSHOT = {"gender": "F", "city": "Shymkent", "children": 1,
                  "contracts_count": 4, "active_contracts": 2, "age": 40,
                  "holds_debit_card": True, "holds_credit_card": True,
                  "relationship_months": 40}

BUSY_SNAPSHOT = dict(QUIET_SNAPSHOT, city="Astana", contracts_count=5, active_contracts=3,
                     age=41, relationship_months=44)


def test_pipeline_profile_is_the_state_at_the_cutoff(stage):
    """
    Переезд в периоде целей, но раньше cutoff, в анкете виден.
    """

    profile = prepare(stage, EARLY, QUIET_SNAPSHOT).profile

    assert profile == {
        "profile_gender": "F",
        "profile_city": "Shymkent",
        "profile_children": 1,
    }


def test_pipeline_ignores_everything_after_the_cutoff(stage):
    """
    Две выгрузки с общей историей до cutoff дают одну и ту же
    историю: анкету и события.
    """

    quiet = prepare(stage, EARLY, QUIET_SNAPSHOT)
    busy = prepare(stage, EARLY + AFTER, BUSY_SNAPSHOT)

    assert quiet.profile == busy.profile

    assert [(event.event_time, event.values) for event in quiet.events] == [
        (event.event_time, event.values) for event in busy.events
    ]

    assert all(event.event_time < val_cutoff() for event in busy.events)


def test_pipeline_change_after_the_cutoff_is_rolled_back(stage):
    """
    Снимок уже говорит Astana, но переезд случился после cutoff.
    """

    assert prepare(stage, EARLY + AFTER, BUSY_SNAPSHOT).profile["profile_city"] == "Shymkent"


def test_profile_change_stays_an_event_of_the_history(stage):
    """
    Анкета — итог изменений, но сами изменения из истории не
    пропадают.
    """

    history = prepare(stage, EARLY, QUIET_SNAPSHOT)

    moves = [
        event.values for event in history.events if "profile_city_new" in event.values
    ]

    assert [(item["profile_city_old"], item["profile_city_new"]) for item in moves] == [
        ("Almaty", "Astana"),
        ("Astana", "Shymkent"),
    ]


LIFELONG_VALUES = (
    "bank_registered", "app_registered", "first_card_activated",
    "first_loan_opened", "first_deposit_opened",
)

# Возрасты, которые «видел train» крошечного словаря: каждый — своё
# значение. Остальные возрасты словарю неизвестны.
AGE_VALUES = ("34", "35", "36", "39", "62", "63")

# Метки стажа, которые «видел train» крошечного словаря.
TENURE_VALUES = ("0-5", "6-11", "54-59")


def write_profile_vocab(
    root,
    lifelong: tuple[str, ...] | None = LIFELONG_VALUES,
    age_scale: bool = False,
    event_types: tuple[str, ...] = (),
) -> None:
    """
    Крошечный словарь, знающий ключи анкеты.

    Общий словарь тестов знает только key_a/b/c, и на нём любое
    значение профиля стало бы [UNK]: проверка совпадения токенов
    выполнялась бы сама собой.

    Ключи — все поля анкеты (INCLUDED_FIELDS) и вехи: словарь без
    ключа поля кодирование отвергает. Значения известны у немногих
    ключей, остальным хватает пустого домена.

    lifelong — известные словарю типы вех; None — словарь прежнего
    кода, без ключа вех вовсе. age_scale — словарь прежнего кода,
    где возраст кодировался диапазонами. event_types — известные
    словарю типы событий: без них событие это один маркер [EVT].
    """

    import json as _json

    from tokenizers import Tokenizer, models

    from src.preprocessing.keys import NUMERIC, PROFILE_KEYS
    from src.tokenization.finalvocab import build_final_vocab
    from src.tokenization.settings import (
        BPE_FILE,
        BUCKETS_FILE,
        FINAL_VOCAB_FILE,
        KEY_VOCAB_FILE,
        VALUE_VOCAB_FILE,
        SPECIAL_TOKENS_FILE,
    )
    from src.tokenization.specials import build_special_tokens
    from src.tokenization.text import BpeModel

    directory = root / "03_vocab"
    directory.mkdir(parents=True, exist_ok=True)

    specials = build_special_tokens()

    domains = {
        "profile_gender": ("F", "M"),
        "profile_city": ("Almaty", "Astana", "Shymkent"),
        "profile_children": ("0", "1", "2"),
        "profile_age": AGE_VALUES,
        "profile_income_type": ("employed", "unemployed"),
        "profile_job_tenure_months": TENURE_VALUES,
    }

    catalogue: dict[str, tuple[str, ...]] = {}
    scales: list[str] = []

    for name in INCLUDED_FIELDS:

        key = PROFILE_KEYS[name]

        if key.kind == NUMERIC or (age_scale and key.key == "profile_age"):
            scales.append(key.key)
        else:
            catalogue[key.key] = domains.get(key.key, ())

    if lifelong is not None:
        catalogue["profile_lifelong"] = lifelong

    if event_types:
        catalogue["event_type"] = event_types

    names = [*catalogue, *scales]

    keys = {name: len(specials) + number for number, name in enumerate(names)}

    number = len(specials) + len(keys)

    values: dict[str, dict[str, int]] = {}

    for key, items in catalogue.items():
        values[key] = {}
        for item in items:
            values[key][item] = number
            number += 1

    # Два диапазона на число: граница — середина шкалы ключа.
    middle = {"profile_age": 40.0, "profile_declared_income": 500_000.0}

    buckets: dict[str, dict[str, dict]] = {}

    for key in scales:
        buckets[key] = {
            f"{key}_bucket_1": {"id": number, "min": None, "max": middle[key]},
            f"{key}_bucket_2": {"id": number + 1, "min": middle[key], "max": None},
        }
        number += 2

    bpe = BpeModel(tokenizer=Tokenizer(models.BPE(vocab={"ab": 0}, merges=[])))

    vocab = build_final_vocab(specials, keys, values, buckets, bpe)

    for name, data in (
        (FINAL_VOCAB_FILE, vocab),
        (SPECIAL_TOKENS_FILE, specials),
        (KEY_VOCAB_FILE, keys),
        (VALUE_VOCAB_FILE, values),
        (BUCKETS_FILE, buckets),
    ):
        (directory / name).write_text(
            _json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    bpe.save(directory / BPE_FILE)


def test_profile_change_is_context_not_a_target(stage):
    """
    Изменение анкеты в периоде целей остаётся событием примера, но
    целью MLM не становится: анкета на cutoff уже содержит его
    итог. Покупка рядом — обычная цель.
    """

    from src.dataset.sample import build_sample
    from src.dataset.settings import ContextPolicy
    from src.dataset.tokenized import TokenizedClient, TokenizedEvent
    from src.preprocessing.settings import PreprocessingConfig
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.specials import EVT, USR

    write_profile_vocab(stage)

    window = PreprocessingConfig.load(None).windows["val"]

    artifacts = FrozenArtifacts.load()

    evt, usr = artifacts.special(EVT), artifacts.special(USR)

    # Маркер и одно значение: содержание событий здесь не важно,
    # правило цели смотрит только на тип.
    def event(moment: str, kind: str) -> TokenizedEvent:
        return TokenizedEvent(
            event_time=when(moment), event_type=kind,
            key_ids=[evt, evt + 1], value_ids=[evt, evt + 1], positions=[0, 0],
            calendar=[0.0] * 6,
        )

    client = TokenizedClient(
        client_id=RAW_CLIENT,
        events=[
            event("2025-06-01T10:00:00", "profile_change"),
            event("2026-02-01T10:00:00", "profile_change"),
            event("2026-02-02T10:00:00", "purchase"),
        ],
        profile_key_ids=[usr], profile_value_ids=[usr], profile_positions=[0],
        profile_time=[None],
    )

    sample = build_sample(artifacts, client, window, ContextPolicy())

    assert list(sample.target_event_mask) == [False, False, True]
    assert len(sample.event_starts) == 3


def test_pipeline_profile_tokens_are_identical(stage):
    """
    Совпадают не только значения, но и токены во входе модели.
    """

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()

    quiet, quiet_times = encode_profile(artifacts, prepare(stage, EARLY, QUIET_SNAPSHOT), 4)
    busy, busy_times = encode_profile(
        artifacts, prepare(stage, EARLY + AFTER, BUSY_SNAPSHOT), 4
    )

    assert list(quiet.key_ids) == list(busy.key_ids)
    assert list(quiet.value_ids) == list(busy.value_ids)
    assert list(quiet.positions) == list(busy.positions)
    assert quiet_times == busy_times

    # Проверка не вырождена: значения словарю известны, и
    # неизвестных ключей среди них нет.
    assert len(quiet.key_ids) > 1
    assert not quiet.unknown_keys


def test_pipeline_profile_carries_no_shortcut_key(stage):
    """
    В снимке holds_credit_card и счётчики договоров заполнены, а
    в истории есть открытие продукта — в анкету они не идут.
    """

    profile = prepare(stage, EARLY + AFTER, BUSY_SNAPSHOT).profile

    forbidden = {f"profile_{name}" for name in EXCLUDED_FIELDS}

    assert not set(profile) & forbidden

    for name in ("holds_credit_card", "contracts_count", "active_contracts",
                 "holds_deposit", "credit_utilization"):
        assert f"profile_{name}" in forbidden

    assert set(profile) <= {f"profile_{name}" for name in INCLUDED_FIELDS}


def test_pipeline_age_on_the_local_cutoff(stage):
    """
    cutoff val — полночь 1 мая у банка, то есть 19:00 UTC 30
    апреля. Клиентке, родившейся 1 мая 1963-го, в этот момент уже
    63: по UTC вышло бы на год меньше. События после cutoff на
    возраст не влияют, а пенсионера в анкете нет вовсе.
    """

    day = date(1963, 5, 1)

    quiet = prepare(stage, EARLY, dict(QUIET_SNAPSHOT, birth_date=day)).profile
    busy = prepare(stage, EARLY + AFTER, dict(BUSY_SNAPSHOT, birth_date=day)).profile

    assert quiet["profile_age"] == 63
    assert "profile_pensioner" not in quiet
    assert quiet == busy


def test_end_of_the_export_does_not_change_the_age(stage):
    """
    Одна и та же клиентка в выгрузках, кончающихся 1 июля и 1
    сентября: на cutoff val возраст и вся анкета одни и те же.
    Прежде возраст брался бы из снимка на конец выгрузки и
    разошёлся бы.
    """

    snapshot = dict(QUIET_SNAPSHOT, birth_date=date(1963, 7, 15),
                    lifelong=[{"type": "bank_registered", "event_time": when("2021-05-17T00:00:00")}])

    short = prepare(stage, EARLY, snapshot, end="2026-07-01T00:00:00+05:00")
    long = prepare(stage, EARLY, snapshot, end="2026-09-01T00:00:00+05:00")

    assert short.profile["profile_age"] == 62
    assert short.profile == long.profile
    assert short.lifelong == long.lifelong == [("bank_registered", when("2021-05-17T00:00:00"))]


def test_age_is_a_profile_token_and_birth_date_is_not(stage):
    """
    Модель видит возраст на cutoff токеном анкеты. Даты рождения
    нет ни среди значений анкеты, ни среди ключей словаря, ни в
    токенах.
    """

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.schema import SemanticSchema

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()

    history = prepare(stage, EARLY, dict(QUIET_SNAPSHOT, birth_date=date(1990, 1, 1)))

    record, _ = encode_profile(artifacts, history, 4)

    names = [artifacts.describe(token) for token in record.key_ids]

    assert history.profile["profile_age"] == 36
    assert artifacts.key_id("profile_age") in record.key_ids
    assert not record.unknown_keys

    assert not [key for key in history.profile if "birth" in key]
    assert not [name for name in names if "birth" in name or "pensioner" in name]
    assert not [key for key in SemanticSchema().keys if "birth" in key]


def test_vocab_fit_and_encoding_read_the_same_cutoff(stage, monkeypatch):
    """
    Словарь учится и группа кодируется на один и тот же cutoff:
    иначе словарь видел бы анкету одной даты, а модель — другой.
    """

    from src.preprocessing.read import Group
    from src.preprocessing.settings import PreprocessingConfig
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.fit import read_train
    from src.tokenization.schema import SemanticSchema
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    prepare(stage, EARLY, QUIET_SNAPSHOT, group="train")

    seen: dict[str, set] = {"fit": set(), "encode": set()}
    stage_name = {"now": "fit"}

    original = Group.history

    def spy(self, client_id, cutoff):
        seen[stage_name["now"]].add(cutoff)
        return original(self, client_id, cutoff)

    monkeypatch.setattr(Group, "history", spy)

    config = TokenizerConfig.load(None)

    read_train(config, SemanticSchema.open())

    stage_name["now"] = "encode"

    write_profile_vocab(stage)

    encode_group(FrozenArtifacts.load(), config.fit_group, config)

    cutoff = PreprocessingConfig.load(None).windows[config.fit_group].final_cutoff

    assert seen == {"fit": {cutoff}, "encode": {cutoff}}


# ============================================================
# СТАРЫЙ АРТЕФАКТ НЕ СМЕШИВАЕТСЯ С НОВЫМ
# ============================================================
#
# Схема таблиц от переноса анкеты не изменилась, поэтому одной
# сверки схемы мало: каталог прежней сборки выглядит исправным и
# молча вернул бы в примеры состояние на конец выгрузки.
# ============================================================


def test_tokenized_without_meta_is_refused(stage):

    import pyarrow as pa

    from src.dataset.tokenized import TokenizedError, TokenizedGroup
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import tokenized_dir
    from src.tokenization.transform import EVENTS_SCHEMA, PROFILE_SCHEMA

    write_profile_vocab(stage)

    directory = tokenized_dir("val")
    directory.mkdir(parents=True, exist_ok=True)

    pq.write_table(EVENTS_SCHEMA.empty_table(), directory / "events.parquet")
    pq.write_table(PROFILE_SCHEMA.empty_table(), directory / "profile.parquet")

    with pytest.raises(TokenizedError, match="meta.json"):
        TokenizedGroup("val", FrozenArtifacts.load())

    # И с чужой версией — тоже.
    (directory / "meta.json").write_text(
        json.dumps({"format": 1, "profile_moment": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    with pytest.raises(TokenizedError, match="прежним кодом"):
        TokenizedGroup("val", FrozenArtifacts.load())

    del pa


def test_samples_without_meta_are_refused(stage):

    from src.dataset.build import SAMPLES_SCHEMA
    from src.dataset.settings import dataset_dir
    from src.temporal.samples import SamplesError, SamplesGroup

    directory = dataset_dir("val")
    directory.mkdir(parents=True, exist_ok=True)

    pq.write_table(SAMPLES_SCHEMA.empty_table(), directory / "samples.parquet")

    with pytest.raises(SamplesError, match="прежним кодом"):
        SamplesGroup("val")

    (directory / "meta.json").write_text(json.dumps({"format": 1}), encoding="utf-8")

    with pytest.raises(SamplesError, match="формат"):
        SamplesGroup("val")


def test_encoded_group_records_the_profile_semantics(stage):
    """
    meta.json называет cutoff событий и смысл анкеты: Attributes на
    тот же cutoff и вехи Lifelong раньше него, с их набором и
    шкалой времени.
    """

    from src.preprocessing.profile_state import LIFELONG_TYPES, PROFILE_SEMANTICS
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig, tokenized_dir
    from src.tokenization.transform import TOKENIZED_FORMAT, encode_group

    write_profile_vocab(stage)

    prepare(stage, EARLY + AFTER, BUSY_SNAPSHOT)

    encode_group(FrozenArtifacts.load(), "val", TokenizerConfig.load(None))

    meta = json.loads((tokenized_dir("val") / "meta.json").read_text(encoding="utf-8"))

    assert meta["format"] == TOKENIZED_FORMAT
    assert meta["events_cutoff"] == val_cutoff().isoformat()
    assert meta["profile_semantics"] == PROFILE_SEMANTICS == "attributes_and_lifelong_at_event_cutoff"
    assert "profile_moment" not in meta
    assert meta["profile_fields"] == list(INCLUDED_FIELDS)
    assert meta["profile_lifelong_types"] == list(LIFELONG_TYPES) == [
        "bank_registered", "app_registered", "first_card_activated",
        "first_loan_opened", "first_deposit_opened",
    ]
    assert meta["profile_lifelong_time"] == {"anchor": "cutoff", "transform": "8*log1p(seconds/8)"}

    # Исключённые поля названы вместе с причиной.
    assert set(meta["profile_fields_excluded"]) == set(EXCLUDED_FIELDS)


def test_tokenized_of_the_previous_semantics_is_refused(stage):
    """
    Каталог формата 2 — анкета на начало периода целей — читать
    нельзя, как и формата 3 — анкета без вех, — и текущий формат
    без смысла анкеты или с другим набором вех.
    """

    from src.preprocessing.profile_state import LIFELONG_TYPES, PROFILE_SEMANTICS

    from src.dataset.tokenized import TokenizedError, TokenizedGroup
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import tokenized_dir
    from src.tokenization.transform import EVENTS_SCHEMA, PROFILE_SCHEMA, TOKENIZED_FORMAT

    write_profile_vocab(stage)

    directory = tokenized_dir("val")
    directory.mkdir(parents=True, exist_ok=True)

    pq.write_table(EVENTS_SCHEMA.empty_table(), directory / "events.parquet")
    pq.write_table(PROFILE_SCHEMA.empty_table(), directory / "profile.parquet")

    for meta in (
        {"format": 2, "profile_moment": "2026-01-01T00:00:00+00:00"},
        {"format": 3, "profile_semantics": "state_at_event_cutoff"},
        # Формат 4 — анкета ещё с признаком пенсионера, формат 5 —
        # прежние вехи и без стажа.
        {"format": 4, "profile_semantics": PROFILE_SEMANTICS,
         "profile_lifelong_types": list(LIFELONG_TYPES)},
        {"format": 5, "profile_semantics": PROFILE_SEMANTICS,
         "profile_lifelong_types": ["relationship_started", "kyc_passed", "app_adopted"]},
        {"format": TOKENIZED_FORMAT, "profile_moment": "2026-01-01T00:00:00+00:00"},
        {"format": TOKENIZED_FORMAT, "profile_semantics": PROFILE_SEMANTICS,
         "profile_lifelong_types": list(LIFELONG_TYPES)[:2]},
    ):
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(TokenizedError, match="прежним кодом"):
            TokenizedGroup("val", FrozenArtifacts.load())


def test_stage_06_to_08_without_lineage_are_refused(stage):
    """
    Схемы 06-08 от смысла анкеты не зависят: каталог без отметки
    происхождения отвергается, а не читается молча.
    """

    from src.batching.temporal import TemporalError, TemporalGroup
    from src.batching.settings import BATCHES_FILE, batches_dir
    from src.dataset.lineage import LINEAGE_FILE
    from src.masking.batches import BatchesError, BatchesGroup
    from src.mlm.inputs import InputError, Source
    from src.temporal.build import TEMPORAL_SCHEMA
    from src.temporal.settings import TEMPORAL_FILE, temporal_dir

    temporal = temporal_dir("val")
    temporal.mkdir(parents=True, exist_ok=True)
    pq.write_table(TEMPORAL_SCHEMA.empty_table(), temporal / TEMPORAL_FILE)

    with pytest.raises(TemporalError, match="прежним кодом"):
        TemporalGroup("val")

    from tests import world

    # Батчи мира тестов пишутся с отметкой, как у настоящего
    # этапа: снимаем её.
    world.write_batches(batches_dir("train") / BATCHES_FILE, [world.population()])

    (batches_dir("train") / LINEAGE_FILE).unlink()

    with pytest.raises(BatchesError, match="прежним кодом"):
        BatchesGroup("train")

    with pytest.raises(InputError, match="прежним кодом"):
        Source("train")



def test_rolled_back_value_keeps_the_type_of_the_field():
    """
    Прежнее значение приезжает из события строкой всегда, а день
    выплаты в анкете это число. Без привода у одного ключа стало
    бы два физических типа значения, и словарь на этом
    останавливается.
    """

    state = profile_at(
        SNAPSHOT, [change("2026-02-01T10:00:00", "income_day", "27", "10")], CUTOFF, BANK
    )

    assert state.values["income_day"] == 27
    assert isinstance(state.values["income_day"], int)

    # И у остальных полей тип тот же, что в снимке.
    for name, value in state.values.items():
        if SNAPSHOT.get(name) is not None and name != "income_day":
            assert type(value) is type(SNAPSHOT[name]), name


def test_unparsable_old_value_drops_the_field_not_the_type():
    """
    Неразобранное число не подменяется текстом: поле просто не
    попадает в анкету, и об этом остаётся запись.
    """

    state = profile_at(
        SNAPSHOT, [change("2026-02-01T10:00:00", "income_day", "среда", "10")], CUTOFF, BANK
    )

    assert "income_day" not in state.values
    assert any("income_day" in note for note in state.notes)
