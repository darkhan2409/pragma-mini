from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator.config import GENERATOR_VERSION, PENSION_AGE, SCHEMA_VERSION
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
# состояние клиента на тот же T. Снимок анкеты в выгрузке лежит
# на её границу, возможно позже T, и откатывается по ленте.
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
    # Снимок возраста и пенсионера описывает конец выгрузки: в
    # анкету они идут посчитанными от birth_date, а не отсюда.
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
        # снимка. Пришла работающей и моложе PENSION_AGE — не
        # пенсионер, что бы ни стояло в снимке.
        "age": 41,
        "pensioner": False,
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
# ВОЗРАСТ И ПЕНСИОНЕР ОТ ДАТЫ РОЖДЕНИЯ
# ============================================================
#
# Оба поля меняются со временем без события, поэтому снимок для
# них бесполезен: он описывает конец выгрузки. Считаются они от
# постоянной birth_date — тем же правилом, что в генераторе.
# ============================================================


def born(day: date, **fields) -> dict:
    return dict(SNAPSHOT, birth_date=day, **fields)


def test_birth_date_fields_reach_the_profile():

    assert set(FROM_BIRTH_DATE) == {"age", "pensioner"}
    assert set(FROM_BIRTH_DATE) <= set(INCLUDED_FIELDS)
    assert not set(FROM_BIRTH_DATE) & set(EXCLUDED_FIELDS)


def test_age_counts_only_birthdays_before_the_cutoff():

    before = profile_at(born(date(1990, 1, 1)), [], CUTOFF, BANK)
    after = profile_at(born(date(1990, 1, 2)), [], CUTOFF, BANK)

    assert before.values["age"] == 36
    assert after.values["age"] == 35


def test_age_is_counted_on_the_local_date_of_the_cutoff():
    """
    19:00 UTC 31 декабря — это уже 1 января у банка. День рождения
    1 января наступил, хотя по UTC дата ещё прежняя.
    """

    moment = when("2025-12-31T19:00:00")

    state = profile_at(born(date(1990, 1, 1)), [], moment, BANK)

    assert state.values["age"] == 36


def test_pensioner_by_age_is_not_taken_from_the_future():
    """
    Пенсионный возраст наступил ПОСЛЕ границы: в снимке признак
    уже стоит, а на cutoff его ещё не было.
    """

    turning = date(2026 - PENSION_AGE, 3, 1)

    at_cutoff = profile_at(born(turning, pensioner=True), [], CUTOFF, BANK)
    later = profile_at(born(turning, pensioner=True), [], when("2026-03-01T00:00:00"), BANK)

    assert at_cutoff.values["age"] == PENSION_AGE - 1
    assert at_cutoff.values["pensioner"] is False

    assert later.values["age"] == PENSION_AGE
    assert later.values["pensioner"] is True


def test_pensioner_by_initial_income_type_at_any_age():

    state = profile_at(
        born(date(1970, 5, 5), income_type="pensioner", pensioner=False), [], CUTOFF, BANK
    )

    assert state.values["age"] == 55
    assert state.values["pensioner"] is True


def test_income_type_change_does_not_move_pensioner():
    """
    Признак ставится по виду дохода, с которым клиент пришёл, а не
    по текущему: так его считает генератор.
    """

    became = profile_at(
        born(date(1970, 5, 5), income_type="pensioner"),
        [change("2025-03-01T10:00:00", "income_type", "employed", "pensioner")],
        CUTOFF, BANK,
    )

    left = profile_at(
        born(date(1970, 5, 5), income_type="employed"),
        [change("2025-03-01T10:00:00", "income_type", "pensioner", "employed")],
        CUTOFF, BANK,
    )

    assert became.values["pensioner"] is False
    assert left.values["pensioner"] is True


def test_no_birth_date_gives_neither_field():

    snapshot = dict(SNAPSHOT)
    del snapshot["birth_date"]

    state = profile_at(snapshot, [], CUTOFF, BANK)

    assert "age" not in state.values
    assert "pensioner" not in state.values


def test_unknown_initial_income_type_leaves_pensioner_out_only_for_the_young():
    """
    Первое изменение вида дохода без прежнего значения: с чем
    клиент пришёл, неизвестно. Молодому признак не подставляется,
    а достигшему PENSION_AGE он следует из одного возраста.
    """

    rows = [change("2025-03-01T10:00:00", "income_type", None, "employed")]

    young = profile_at(born(date(1985, 5, 5)), rows, CUTOFF, BANK)
    old = profile_at(born(date(1950, 5, 5)), rows, CUTOFF, BANK)

    assert young.values["age"] == 40
    assert "pensioner" not in young.values

    assert old.values["pensioner"] is True


# ============================================================
# ТО ЖЕ, НО ЧЕРЕЗ ВЕСЬ ПУТЬ ДАННЫХ
# ============================================================
#
# Выше проверено восстановление. Ниже — что оно доезжает до
# модели: выгрузка, препроцессинг и токены профиля.
# ============================================================


RAW_CLIENT = "c000000000001"


def raw_event(client: str, moment: str, payload: dict) -> dict:
    return {
        "client_id": client,
        "event_time": when(moment).astimezone(UTC).isoformat().replace("+00:00", "+00:00"),
        "source": {
            "profile_change": "profile",
            "product_opened": "product_events",
            "product_closed": "product_events",
            "purchase": "transactions",
        }[payload["type"]],
        "payload": json.dumps(payload, ensure_ascii=False),
    }


def write_raw(directory, events: list[dict], snapshot: dict) -> None:
    """
    Выгрузка группы по контракту RAW: две таблицы и манифест.
    """

    directory.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(events, schema=EVENTS_SCHEMA)

    pq.write_table(table, directory / "events.parquet", compression="zstd")

    row = {name: None for name in PROFILE_SCHEMA.names}
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
        "period_end": "2026-07-01T00:00:00+05:00",
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


def prepare(stage, events: list[dict], snapshot: dict, group: str = "val"):
    """
    Выгрузка → препроцессинг → история клиента на cutoff группы.
    """

    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.read import Group
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir

    write_raw(raw_group_dir(group), events, snapshot)

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


def write_profile_vocab(root) -> None:
    """
    Крошечный словарь, знающий ключи анкеты.

    Общий словарь тестов знает только key_a/b/c, и на нём любое
    значение профиля стало бы [UNK]: проверка совпадения токенов
    выполнялась бы сама собой.
    """

    import json as _json

    from tokenizers import Tokenizer, models

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

    names = ("profile_gender", "profile_city", "profile_children")

    keys = {name: len(specials) + number for number, name in enumerate(names)}

    catalogue = {
        "profile_gender": ("F", "M"),
        "profile_city": ("Almaty", "Astana", "Shymkent"),
        "profile_children": ("0", "1", "2"),
    }

    number = len(specials) + len(keys)

    values: dict[str, dict[str, int]] = {}

    for key, items in catalogue.items():
        values[key] = {}
        for item in items:
            values[key][item] = number
            number += 1

    bpe = BpeModel(tokenizer=Tokenizer(models.BPE(vocab={"ab": 0}, merges=[])))

    vocab = build_final_vocab(specials, keys, values, {}, bpe)

    for name, data in (
        (FINAL_VOCAB_FILE, vocab),
        (SPECIAL_TOKENS_FILE, specials),
        (KEY_VOCAB_FILE, keys),
        (VALUE_VOCAB_FILE, values),
        (BUCKETS_FILE, {}),
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

    quiet = encode_profile(artifacts, prepare(stage, EARLY, QUIET_SNAPSHOT), 4)
    busy = encode_profile(artifacts, prepare(stage, EARLY + AFTER, BUSY_SNAPSHOT), 4)

    assert list(quiet.key_ids) == list(busy.key_ids)
    assert list(quiet.value_ids) == list(busy.value_ids)
    assert list(quiet.positions) == list(busy.positions)

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


def test_pipeline_age_and_pensioner_on_the_local_cutoff(stage):
    """
    cutoff val — полночь 1 мая у банка, то есть 19:00 UTC 30
    апреля. Клиентке, родившейся 1 мая, в этот момент уже
    PENSION_AGE: по UTC вышло бы на год меньше и не пенсионер.
    События после cutoff на оба поля не влияют.
    """

    day = date(2026 - PENSION_AGE, 5, 1)

    quiet = prepare(stage, EARLY, dict(QUIET_SNAPSHOT, birth_date=day)).profile
    busy = prepare(stage, EARLY + AFTER, dict(BUSY_SNAPSHOT, birth_date=day)).profile

    assert quiet["profile_age"] == PENSION_AGE
    assert quiet["profile_pensioner"] is True
    assert quiet == busy


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
    meta.json называет cutoff событий и смысл анкеты: состояние на
    тот же cutoff.
    """

    from src.preprocessing.profile_state import PROFILE_SEMANTICS
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig, tokenized_dir
    from src.tokenization.transform import TOKENIZED_FORMAT, encode_group

    write_profile_vocab(stage)

    prepare(stage, EARLY + AFTER, BUSY_SNAPSHOT)

    encode_group(FrozenArtifacts.load(), "val", TokenizerConfig.load(None))

    meta = json.loads((tokenized_dir("val") / "meta.json").read_text(encoding="utf-8"))

    assert meta["format"] == TOKENIZED_FORMAT
    assert meta["events_cutoff"] == val_cutoff().isoformat()
    assert meta["profile_semantics"] == PROFILE_SEMANTICS == "state_at_event_cutoff"
    assert "profile_moment" not in meta
    assert meta["profile_fields"] == list(INCLUDED_FIELDS)

    # Исключённые поля названы вместе с причиной.
    assert set(meta["profile_fields_excluded"]) == set(EXCLUDED_FIELDS)


def test_tokenized_of_the_previous_semantics_is_refused(stage):
    """
    Каталог формата 2 — анкета на начало периода целей — читать
    нельзя, как и текущий формат без смысла анкеты.
    """

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
        {"format": TOKENIZED_FORMAT, "profile_moment": "2026-01-01T00:00:00+00:00"},
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
