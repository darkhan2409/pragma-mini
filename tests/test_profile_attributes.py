from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.preprocessing.keys import PROFILE_KEYS
from src.preprocessing.profile_state import (
    EXCLUDED_FIELDS,
    INCLUDED_FIELDS,
    JOB_TENURE_STEP,
    profile_at,
    tenure_label,
)

from tests.test_profile_state import (
    BANK,
    EARLY,
    QUIET_SNAPSHOT,
    SNAPSHOT,
    TENURE_VALUES,
    prepare,
    when,
    write_profile_vocab,
    write_raw,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Анкета — 13 полей Attributes и вехи Lifelong. Здесь проверяется
# то, что появилось вместе с этим составом:
#
#   стаж               метка полугодия от начала последней работы
#                      по найму, о которой банк узнал строго раньше
#                      cutoff; границы задаёт шаг, а не train;
#   событие-источник   проверяется отдельно, по ссылке вехи, в
#                      test_lifelong_source.py;
#   словарь            новые значения учатся только на train,
#                      невиданное — [UNK]; словарь прежнего кода
#                      кодирование отвергает.
#
# Ожидаемые значения написаны руками.
# ============================================================


def local(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=BANK)


def job(start: date | None, recorded: str) -> dict:
    return {"start_date": start, "record_time": when(recorded)}


def tenure_at(employment: list[dict], moment: datetime) -> str | None:
    return profile_at(dict(SNAPSHOT, employment=employment), [], moment, BANK).values.get(
        "job_tenure_months"
    )


# ============================================================
# СОСТАВ
# ============================================================


def test_dropped_candidates_are_not_attributes():
    """
    Иждивенцев и источников дохода в анкете нет: независимого
    источника у первых генератор не знает, вторые убраны решением.
    """

    for name in ("num_dependents", "income_sources", "pensioner"):
        assert name not in INCLUDED_FIELDS
        assert name not in PROFILE_KEYS

    assert "job_tenure_months" in INCLUDED_FIELDS
    assert "income_type" in INCLUDED_FIELDS
    assert not set(INCLUDED_FIELDS) & set(EXCLUDED_FIELDS)


def test_fields_left_out_of_the_profile_have_no_tokens():
    """
    Ключ анкеты в реестре есть ровно у полей, которые модельная
    анкета несёт. У исключённых (лимит, договоры, стаж отношений)
    нет ни ключа, ни шкалы: иначе в словаре лежали бы токены поля,
    которого модель никогда не видит, а у числа — шкала,
    «выученная» на нуле наблюдений.
    """

    from src.preprocessing.keys import PROFILE_LIFELONG_KEY
    from src.tokenization.keyvocab import build_key_vocab
    from src.tokenization.schema import ORIGIN_PROFILE, SemanticSchema
    from src.tokenization.settings import default_numeric_encoders
    from src.tokenization.specials import build_special_tokens

    schema = SemanticSchema.open()

    profile = {info.key for info in schema.keys.values() if info.origin == ORIGIN_PROFILE}

    assert profile == {PROFILE_KEYS[name].key for name in INCLUDED_FIELDS} | {PROFILE_LIFELONG_KEY.key}

    dead = {PROFILE_KEYS[name].key for name in EXCLUDED_FIELDS}

    assert not dead & set(build_key_vocab(build_special_tokens(), schema))
    assert not dead & set(default_numeric_encoders())


# ============================================================
# СТАЖ ПОЛУГОДИЯМИ
# ============================================================


def test_tenure_label_is_a_fixed_six_month_step():

    assert JOB_TENURE_STEP == 6

    assert tenure_label(0) == tenure_label(5) == "0-5"
    assert tenure_label(6) == tenure_label(11) == "6-11"
    assert tenure_label(12) == "12-17"
    assert tenure_label(41) == "36-41"
    assert tenure_label(42) == "42-47"


def test_tenure_counts_full_months_to_the_local_cutoff():

    records = [job(date(2020, 1, 31), "2020-02-01T00:00:00")]

    assert tenure_at(records, local("2020-07-30T00:00:00")) == "0-5"
    assert tenure_at(records, local("2020-07-31T00:00:00")) == "6-11"


def test_tenure_follows_what_the_bank_knew_at_the_cutoff():
    """
    Работа с 10.03.2019, потеря работы записана 1 февраля 2025, новая
    работа с 15 июня записана 1 августа.
    """

    records = [
        job(date(2019, 3, 10), "2020-01-01T00:00:00"),
        job(None, "2025-02-01T00:00:00"),
        job(date(2025, 6, 15), "2025-08-01T00:00:00"),
    ]

    # 57 полных месяцев с 10.03.2019 до 1.01.2024.
    assert tenure_at(records, local("2024-01-01T00:00:00")) == "54-59"

    # Банк знает о потере работы.
    assert tenure_at(records, local("2025-03-01T00:00:00")) is None

    # Новая работа уже идёт, но банк ещё не знает — поля нет.
    assert tenure_at(records, local("2025-07-01T00:00:00")) is None

    # Запись ровно в момент cutoff — уже будущее.
    assert tenure_at(records, when("2025-08-01T00:00:00")) is None

    # Два полных месяца с 15 июня до 1 сентября.
    assert tenure_at(records, local("2025-09-01T00:00:00")) == "0-5"


def test_tenure_needs_salaried_income_at_the_cutoff():
    """
    Работа с 1 июня записана банком, но вид дохода на T не наёмный:
    стажа нет. Решает вид дохода именно на T — изменение после T
    откатывается.
    """

    from tests.test_profile_state import change

    records = [job(date(2025, 6, 1), "2025-06-01T00:00:00")]
    moment = local("2025-09-01T00:00:00")

    def tenure(income: str, rows: list = ()) -> str | None:
        snapshot = dict(SNAPSHOT, income_type=income, employment=records)
        return profile_at(snapshot, list(rows), moment, BANK).values.get("job_tenure_months")

    for income in ("unemployed", "self_employed", "business_owner", "pensioner", "student"):
        assert tenure(income) is None, income

    assert tenure("employed") == tenure("state_employee") == "0-5"

    # В снимке уже employed, но до 1 октября клиент числился безработным.
    later = change("2025-10-01T00:00:00", "income_type", "unemployed", "employed")

    assert tenure("employed", [later]) is None


def test_salaried_income_types_are_the_generators_salary_ones():

    from src.generator import params as params_module
    from src.preprocessing.profile_state import SALARIED_INCOME_TYPES

    kinds = params_module.active().income.primary_kind_by_income_type

    assert set(SALARIED_INCOME_TYPES) == {name for name, kind in kinds.items() if kind == "salary"}


def test_no_employment_record_means_no_tenure():

    assert tenure_at([], local("2024-01-01T00:00:00")) is None

    snapshot = dict(SNAPSHOT)

    assert "job_tenure_months" not in profile_at(snapshot, [], local("2024-01-01T00:00:00"), BANK).values


def test_tenure_reaches_the_tokens_and_unseen_labels_are_unk(stage):
    """
    Метка из словаря — своё значение ключа profile_job_tenure_months,
    метка, которой train не видел, — [UNK].
    """

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.specials import UNK

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()

    key = artifacts.key_id("profile_job_tenure_months")

    def token(start: date) -> tuple[str, int]:
        history = prepare(stage, EARLY, dict(
            QUIET_SNAPSHOT, income_type="employed",
            employment=[{"start_date": start, "record_time": when("2021-12-01T00:00:00")}],
        ))
        record, _ = encode_profile(artifacts, history, 4)
        return history.profile["profile_job_tenure_months"], record.value_ids[record.key_ids.index(key)]

    # С 1.11.2021 до cutoff val (1 мая 2026 у банка) — 54 месяца.
    label, value = token(date(2021, 11, 1))

    assert label == "54-59" and label in TENURE_VALUES
    assert artifacts.describe(value) == "value:profile_job_tenure_months=54-59"

    # С 1.12.2021 — 53 месяца, метка 48-53: словарю она неизвестна.
    label, value = token(date(2021, 12, 1))

    assert label == "48-53" and label not in TENURE_VALUES
    assert value == artifacts.special(UNK)


# ============================================================
# СЛОВАРЬ
# ============================================================


def test_new_values_are_learned_on_train_only(stage):
    """
    В train у клиентки стаж 48-53 и веха bank_registered, в val —
    стаж 12-17 и ещё app_registered. Словарь, собранный функциями
    fit, знает только значения train.
    """

    from src.tokenization.categorical import build_value_vocab
    from src.tokenization.fit import read_train
    from src.tokenization.keyvocab import build_key_vocab
    from src.tokenization.schema import SemanticSchema
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.specials import build_special_tokens

    old = when("2021-05-17T00:00:00")

    prepare(stage, EARLY, dict(
        QUIET_SNAPSHOT, income_type="employed",
        employment=[{"start_date": date(2021, 11, 1), "record_time": when("2022-01-01T00:00:00")}],
        lifelong=[{"type": "bank_registered", "event_time": old}],
    ), group="train")

    prepare(stage, EARLY, dict(
        QUIET_SNAPSHOT, income_type="employed",
        employment=[{"start_date": date(2025, 3, 1), "record_time": when("2025-04-01T00:00:00")}],
        lifelong=[
            {"type": "bank_registered", "event_time": old},
            {"type": "app_registered", "event_time": old + timedelta(days=1)},
        ],
    ))

    config = TokenizerConfig.load(None)
    schema = SemanticSchema.open()

    train = read_train(config, schema)

    values = build_value_vocab(train, build_key_vocab(build_special_tokens(), schema), config, schema)

    # С 1.11.2021 до cutoff train (1 января 2026 у банка) — 50 месяцев.
    assert list(values["profile_job_tenure_months"]) == ["48-53"]
    assert list(values["profile_lifelong"]) == ["bank_registered"]


def stale(artifacts, **changes):
    """
    Тот же словарь, но с изменёнными частями: так выглядит словарь
    прежнего кода.
    """

    keys = changes.get("keys", artifacts.keys)

    return SimpleNamespace(
        keys=keys,
        values=changes.get("values", artifacts.values),
        buckets=changes.get("buckets", artifacts.buckets),
        key_id=keys.get,
    )


def test_current_vocab_passes_and_old_ones_are_refused(stage):

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.transform import TransformError, check_vocab

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()

    check_vocab(artifacts)

    without_tenure = {key: value for key, value in artifacts.keys.items()
                      if key != "profile_job_tenure_months"}

    old_lifelong = {**artifacts.values, "profile_lifelong": {"relationship_started": 1, "kyc_passed": 2}}

    for broken, reason in (
        (stale(artifacts, keys={**artifacts.keys, "profile_pensioner": 999}), "больше нет"),
        (stale(artifacts, keys=without_tenure), "нет ключей анкеты"),
        (stale(artifacts, values=old_lifelong), "прежнего набора"),
        (stale(artifacts, buckets={**artifacts.buckets, "profile_age": ()}), "числовая шкала"),
    ):
        with pytest.raises(TransformError, match=reason):
            check_vocab(broken)


# ============================================================
# СТАРЫЕ АРТЕФАКТЫ
# ============================================================
