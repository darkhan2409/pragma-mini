from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator.config import GENERATOR_VERSION, SCHEMA_VERSION
from src.generator.emit import EVENTS_SCHEMA
from src.generator.profile import PROFILE_SCHEMA
from src.preprocessing.profile_state import (
    INCLUDED_FIELDS,
    UNPROVABLE_FIELDS,
    profile_at,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Анкета во входе модели обязана описывать клиента на НАЧАЛО
# периода целей. Прежде туда шёл конечный снимок, и он
# пересказывал события, которые модель должна восстанавливать:
# открытый в периоде целей договор был виден в числе договоров,
# а переезд — в городе.
#
# Здесь проверяется не то, что функция согласна сама с собой, а
# два независимых утверждения:
#
#   при одинаковой истории ДО границы любые события ПОСЛЕ неё не
#   меняют ни одного значения анкеты;
#
#   изменение ДО границы меняет ровно своё поле — без этого
#   первое утверждение выполнял бы и пустой словарь.
#
# Ожидаемые значения написаны здесь руками, а не получены тем же
# восстановлением.
# ============================================================


UTC = timezone.utc


def when(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


BORDER = when("2026-01-01T00:00:00")


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
    # Невосстановимые поля в снимке есть и обязаны остаться за
    # бортом.
    "age": 41,
    "pensioner": False,
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


def test_unprovable_fields_never_reach_the_profile():
    """
    Невосстановимое поле не попадает в анкету даже тогда, когда
    в снимке оно заполнено.
    """

    state = profile_at(SNAPSHOT, [], BORDER)

    leaked = sorted(set(state.values) & set(UNPROVABLE_FIELDS))

    assert not leaked, f"в анкету прошли невосстановимые поля: {leaked}"


def test_field_set_does_not_depend_on_the_client():
    """
    Состав полей один и тот же у клиента с изменениями и без.

    Иначе само отсутствие поля сообщало бы, что с клиентом
    что-то случилось после границы.
    """

    quiet = profile_at(SNAPSHOT, [], BORDER)

    busy = profile_at(
        SNAPSHOT,
        [
            change("2026-02-01T10:00:00", "city", "Astana", "Shymkent"),
            product("2026-03-01T10:00:00", "product_opened"),
            product("2026-04-01T10:00:00", "product_closed"),
        ],
        BORDER,
    )

    assert set(quiet.values) == set(busy.values)


# ============================================================
# ГЛАВНОЕ: БУДУЩЕЕ НЕ МЕНЯЕТ АНКЕТУ
# ============================================================


def test_events_after_the_border_do_not_change_any_value():
    """
    Одна и та же история до границы, разное после — анкета одна.
    """

    before = [change("2025-06-01T10:00:00", "city", "Almaty", "Astana")]

    first = profile_at(dict(SNAPSHOT, city="Astana"), before, BORDER)

    second = profile_at(
        SNAPSHOT,
        before
        + [
            change("2026-02-01T10:00:00", "city", "Astana", "Shymkent"),
            product("2026-02-05T10:00:00", "product_opened"),
            product("2026-03-05T10:00:00", "product_closed"),
            product("2026-04-05T10:00:00", "product_opened"),
        ],
        BORDER,
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

    state = profile_at(SNAPSHOT, rows, BORDER)

    assert state.values == {
        "gender": "F",
        # Переезд после границы откатан, переезд до неё сохранён.
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
    }


# ============================================================
# ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ
# ============================================================


def test_change_before_the_border_does_change_the_field():
    """
    Без этого первое утверждение выполнял бы и пустой словарь.
    """

    quiet = profile_at(SNAPSHOT, [], BORDER)

    moved = profile_at(
        dict(SNAPSHOT, city="Astana"),
        [change("2025-06-01T10:00:00", "city", "Almaty", "Astana")],
        BORDER,
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

    quiet = profile_at(SNAPSHOT, [], BORDER)

    busy = profile_at(
        SNAPSHOT,
        [
            product("2025-06-01T10:00:00", "product_opened"),
            product("2026-06-01T10:00:00", "product_opened"),
            product("2026-07-01T10:00:00", "product_closed"),
        ],
        BORDER,
    )

    assert quiet.values == busy.values

    assert "contracts_count" not in quiet.values
    assert "active_contracts" not in quiet.values


# ============================================================
# ГРАНИЦЫ И КРАЙНИЕ СЛУЧАИ
# ============================================================


def test_event_exactly_at_the_border_belongs_to_the_future():
    """
    Период целей это [target_start, target_end): событие ровно на
    границе уже цель, и анкета обязана его откатить.
    """

    state = profile_at(
        SNAPSHOT,
        [change("2026-01-01T00:00:00", "city", "Karaganda", "Shymkent")],
        BORDER,
    )

    assert state.values["city"] == "Karaganda"


def test_event_a_microsecond_before_the_border_stays_in_the_past():

    state = profile_at(
        SNAPSHOT,
        [change("2025-12-31T23:59:59.999999", "city", "Karaganda", "Shymkent")],
        BORDER,
    )

    assert state.values["city"] == "Shymkent"


def test_several_changes_of_one_field_roll_back_to_the_earliest():
    """
    Откат идёт к ПЕРВОМУ изменению после границы, а не к
    последнему: между ними значение уже менялось.
    """

    rows = [
        change("2026-02-01T10:00:00", "declared_income", "400000", "550000"),
        change("2026-04-01T10:00:00", "declared_income", "550000", "700000"),
    ]

    assert profile_at(SNAPSHOT, rows, BORDER).values["declared_income"] == 400_000


def test_missing_old_value_means_the_field_was_empty():
    """
    Изменение без прежнего значения говорит, что на границе поля
    не было заполнено. Подставлять конечное нельзя.
    """

    state = profile_at(
        SNAPSHOT, [change("2026-02-01T10:00:00", "industry", None, "trade")], BORDER
    )

    assert "industry" not in state.values
    assert state.rolled_back_to_absent == ("industry",)


def test_client_without_a_questionnaire_gets_an_empty_profile():

    assert profile_at(None, [product("2026-02-01T10:00:00", "product_opened")], BORDER).values == {}


def test_unknown_change_field_is_ignored_not_guessed():
    """
    Изменение поля, которого в анкете нет, ничего не меняет.
    """

    state = profile_at(
        SNAPSHOT,
        [change("2026-02-01T10:00:00", "consent_marketing", "true", "false")],
        BORDER,
    )

    assert state.values == profile_at(SNAPSHOT, [], BORDER).values


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
        "period_end": "2026-05-01T00:00:00+05:00",
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


def prepare(stage, events: list[dict], snapshot: dict):
    """
    Выгрузка → препроцессинг → анкета клиента на начало целей.
    """

    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.read import Group
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir

    write_raw(raw_group_dir("val"), events, snapshot)

    settings = PreprocessingConfig.load(None)

    out = group_dir("val")

    build_group(raw_group_dir("val"), out, settings, "val")

    window = settings.windows["val"]

    return Group("val").history(RAW_CLIENT, window.final_cutoff, window.target_start)


EARLY = [
    raw_event(RAW_CLIENT, "2024-03-01T09:00:00", {
        "type": "purchase", "amount": 5000, "direction": "debit", "status": "approved"}),
    raw_event(RAW_CLIENT, "2025-06-01T10:00:00", {
        "type": "profile_change", "field_name": "city", "old_value": "Almaty",
        "new_value": "Astana", "change_source": "client", "confirmed": True}),
]

LATE = [
    raw_event(RAW_CLIENT, "2026-02-01T10:00:00", {
        "type": "profile_change", "field_name": "city", "old_value": "Astana",
        "new_value": "Shymkent", "change_source": "client", "confirmed": True}),
    raw_event(RAW_CLIENT, "2026-03-01T10:00:00", {
        "type": "product_opened", "product_id": "prd_test", "reason": "application_approved"}),
]

QUIET_SNAPSHOT = {"gender": "F", "city": "Astana", "children": 1,
                  "contracts_count": 4, "active_contracts": 2, "age": 40,
                  "holds_debit_card": True, "relationship_months": 40}

BUSY_SNAPSHOT = dict(QUIET_SNAPSHOT, city="Shymkent", contracts_count=5, active_contracts=3,
                     age=41, relationship_months=44)


def test_pipeline_profile_ignores_everything_after_the_border(stage):
    """
    Две выгрузки с общей историей до границы дают одну анкету.
    """

    quiet = prepare(stage, EARLY, QUIET_SNAPSHOT).profile
    busy = prepare(stage, EARLY + LATE, BUSY_SNAPSHOT).profile

    assert quiet == busy

    # И это именно состояние на границу, написанное руками.
    assert quiet == {
        "profile_gender": "F",
        "profile_city": "Astana",
        "profile_children": 1,
    }


def test_pipeline_profile_follows_a_change_before_the_border(stage):
    """
    Отрицательный контроль на том же пути.
    """

    moved = prepare(stage, EARLY, QUIET_SNAPSHOT).profile

    stayed = prepare(
        stage,
        [EARLY[0]],
        dict(QUIET_SNAPSHOT, city="Almaty"),
    ).profile

    assert moved["profile_city"] == "Astana"
    assert stayed["profile_city"] == "Almaty"


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


def test_pipeline_profile_tokens_are_identical(stage):
    """
    Совпадают не только значения, но и токены во входе модели.
    """

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()

    quiet = encode_profile(artifacts, prepare(stage, EARLY, QUIET_SNAPSHOT), 4)
    busy = encode_profile(artifacts, prepare(stage, EARLY + LATE, BUSY_SNAPSHOT), 4)

    assert list(quiet.key_ids) == list(busy.key_ids)
    assert list(quiet.value_ids) == list(busy.value_ids)
    assert list(quiet.positions) == list(busy.positions)

    # Проверка не вырождена: значения словарю известны, и
    # неизвестных ключей среди них нет.
    assert len(quiet.key_ids) > 1
    assert not quiet.unknown_keys


def test_pipeline_profile_carries_no_unprovable_key(stage):

    profile = prepare(stage, EARLY + LATE, BUSY_SNAPSHOT).profile

    forbidden = {f"profile_{name}" for name in UNPROVABLE_FIELDS}

    assert not set(profile) & forbidden

    assert set(profile) <= {f"profile_{name}" for name in INCLUDED_FIELDS}


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


def test_encoded_group_records_both_moments(stage):
    """
    meta.json называет оба среза: их два, и перепутать их нельзя.
    """

    from src.preprocessing.settings import PreprocessingConfig
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig, tokenized_dir
    from src.tokenization.transform import encode_group

    write_profile_vocab(stage)

    prepare(stage, EARLY + LATE, BUSY_SNAPSHOT)

    report = encode_group(FrozenArtifacts.load(), "val", TokenizerConfig.load(None))

    meta = json.loads((tokenized_dir("val") / "meta.json").read_text(encoding="utf-8"))

    window = PreprocessingConfig.load(None).windows["val"]

    assert meta["events_cutoff"] == window.final_cutoff.isoformat()
    assert meta["profile_moment"] == window.target_start.isoformat()
    assert meta["events_cutoff"] != meta["profile_moment"]

    assert report["profile_moment"] == meta["profile_moment"]

    # Исключённые поля названы вместе с причиной.
    assert set(meta["profile_fields_excluded"]) == set(UNPROVABLE_FIELDS)


def test_rolled_back_value_keeps_the_type_of_the_field():
    """
    Прежнее значение приезжает из события строкой всегда, а день
    выплаты в анкете это число. Без привода у одного ключа стало
    бы два физических типа значения, и словарь на этом
    останавливается.
    """

    state = profile_at(
        SNAPSHOT, [change("2026-02-01T10:00:00", "income_day", "27", "10")], BORDER
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
        SNAPSHOT, [change("2026-02-01T10:00:00", "income_day", "среда", "10")], BORDER
    )

    assert "income_day" not in state.values
    assert any("income_day" in note for note in state.notes)
