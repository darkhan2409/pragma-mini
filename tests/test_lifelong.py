from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src.generator.profile import LIFELONG_SOURCE_FIELD
from src.mlm.model import pack
from src.preprocessing.profile_state import (
    EXCLUDED_FIELDS,
    FROM_LIFELONG,
    INCLUDED_FIELDS,
    LIFELONG_TYPES,
)

from tests import world
from tests.test_profile_state import (
    AFTER,
    AS_OF,
    BANK,
    BUSY_SNAPSHOT,
    EARLY,
    QUIET_SNAPSHOT,
    RAW_CLIENT,
    UTC,
    prepare,
    raw_event,
    val_cutoff,
    write_profile_vocab,
    write_raw,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Анкета на cutoff T — Attributes @ T и вехи Lifelong строго
# раньше T. Вехи это факты снимка со своим временем, а не события
# ленты, поэтому проверяется путь целиком:
#
#   выгрузка → препроцессинг  граница T полуоткрытая, данные
#                             после T ничего не меняют, веха
#                             старше ленты не теряется;
#   токены                    вехи идут после Attributes, по
#                             времени, у каждого токена своё
#                             время; словарь учится только на
#                             train;
#   05 → 06 → 07              время доезжает как давность до T
#                             той же шкалой, что у событий;
#   энкодер                   время действительно влияет, чужие
#                             вехи — нет;
#   старые артефакты          отвергаются с командой пересборки.
#
# Ожидаемые значения написаны руками.
# ============================================================


CPU = torch.device("cpu")

# Клиент пришёл задолго до начала ленты (2024).
OLD = datetime(2021, 5, 17, tzinfo=BANK).astimezone(UTC)

MICROSECOND = timedelta(microseconds=1)


def life(*items: tuple[str, datetime]) -> list[dict]:
    """
    Вехи снимка. У вех о продуктах источник назван: ссылка на
    карту или договор (source_id).
    """

    return [
        {"type": kind, "event_time": moment,
         "source_id": f"src_{kind}" if kind in LIFELONG_SOURCE_FIELD else None}
        for kind, moment in items
    ]


def with_life(snapshot: dict, *items: tuple[str, datetime]) -> dict:
    return dict(snapshot, lifelong=life(*items))


# ============================================================
# ГРАНИЦА T
# ============================================================


def test_milestone_just_before_the_cutoff_is_seen_and_at_it_is_not(stage):

    cutoff = val_cutoff()

    # Первая карта активирована за микросекунду до T: внутри окна у
    # вехи есть событие-источник в ленте.
    activated = raw_event(RAW_CLIENT, (cutoff - MICROSECOND).replace(tzinfo=None).isoformat(), {
        "type": "card_activated", "product_id": "prd_card", "card_id": "src_first_card_activated"})

    history = prepare(stage, EARLY + [activated], with_life(
        QUIET_SNAPSHOT,
        ("bank_registered", OLD),
        ("first_card_activated", cutoff - MICROSECOND),
        ("app_registered", cutoff),
    ))

    assert history.lifelong == [
        ("bank_registered", OLD),
        ("first_card_activated", cutoff - MICROSECOND),
    ]


def test_milestone_after_the_cutoff_is_not_seen(stage):

    history = prepare(stage, EARLY, with_life(
        QUIET_SNAPSHOT,
        ("bank_registered", OLD),
        ("app_registered", val_cutoff() + timedelta(days=20)),
    ))

    assert history.lifelong == [("bank_registered", OLD)]


def test_milestone_older_than_the_tape_survives(stage):
    """
    Самое раннее событие ленты — 2024 год, а веха прихода — 2021-й.
    Lifelong берётся из снимка, а не из видимой истории.
    """

    history = prepare(stage, EARLY, with_life(QUIET_SNAPSHOT, ("bank_registered", OLD)))

    assert min(event.event_time for event in history.events).year == 2024
    assert history.lifelong == [("bank_registered", OLD)]


def test_future_raw_data_does_not_change_the_profile(stage):
    """
    Две выгрузки с общим прошлым: у второй после T есть события,
    переезд и установка приложения. Анкета на T одна и та же — и
    значениями, и токенами, и временем токенов.
    """

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()

    before = (("bank_registered", OLD), ("first_card_activated", OLD))

    quiet = prepare(stage, EARLY, with_life(QUIET_SNAPSHOT, *before))
    quiet_tokens, quiet_times = encode_profile(artifacts, quiet, 4)

    busy = prepare(stage, EARLY + AFTER, with_life(
        BUSY_SNAPSHOT, *before, ("app_registered", val_cutoff() + timedelta(days=20))
    ))
    busy_tokens, busy_times = encode_profile(artifacts, busy, 4)

    assert quiet.profile == busy.profile
    assert quiet.lifelong == busy.lifelong == list(before)

    assert list(quiet_tokens.key_ids) == list(busy_tokens.key_ids)
    assert list(quiet_tokens.value_ids) == list(busy_tokens.value_ids)
    assert quiet_times == busy_times


def test_cutoff_after_the_snapshot_is_refused(stage):
    """
    Снимок описывает клиента перед as_of. Группа test режет на 1
    сентября, а выгрузка кончается 1 июля: состояние позже снимка
    из данных не следует, и выдумывать его нельзя.
    """

    from src.preprocessing.read import ReadError

    with pytest.raises(ReadError, match="позже снимка"):
        prepare(stage, EARLY, QUIET_SNAPSHOT, group="test")


def test_attributes_and_lifelong_stay_apart(stage):
    """
    Вехи не смешиваются со словарём значений, а стаж, выводимый из
    вехи, в Attributes не возвращается, хотя в снимке он есть.
    """

    history = prepare(stage, EARLY, with_life(QUIET_SNAPSHOT, ("bank_registered", OLD)))

    assert QUIET_SNAPSHOT["relationship_months"] == 40

    assert "profile_lifelong" not in history.profile
    assert "profile_relationship_months" not in history.profile

    assert "relationship_months" in FROM_LIFELONG
    assert "relationship_months" in EXCLUDED_FIELDS
    assert "relationship_months" not in INCLUDED_FIELDS


# ============================================================
# ТОКЕНЫ
# ============================================================


def test_milestones_follow_the_attributes_with_their_time(stage):
    """
    [USR], три поля Attributes без времени, затем три вехи под
    одним ключом, по времени, каждая своим значением с нуля.
    """

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.specials import USR

    write_profile_vocab(stage)

    artifacts = FrozenArtifacts.load()

    adopted = datetime(2025, 2, 10, tzinfo=BANK).astimezone(UTC)

    history = prepare(stage, EARLY, with_life(
        QUIET_SNAPSHOT,
        ("bank_registered", OLD),
        ("first_card_activated", OLD),
        ("app_registered", adopted),
    ))

    record, times = encode_profile(artifacts, history, 4)

    key = artifacts.key_id("profile_lifelong")

    assert record.key_ids[0] == artifacts.special(USR)
    assert record.key_ids[1:4] == sorted(
        artifacts.key_id(name) for name in ("profile_gender", "profile_city", "profile_children")
    )
    assert record.key_ids[4:] == [key, key, key]
    assert record.value_ids[4:] == [
        artifacts.categorical_id("profile_lifelong", name)
        for name in ("bank_registered", "first_card_activated", "app_registered")
    ]
    assert record.positions[4:] == [0, 0, 0]

    assert times == [None, None, None, None, OLD, OLD, adopted]


def test_milestone_unknown_to_the_vocab_becomes_unk(stage):

    from src.tokenization.encode import encode_profile
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.specials import UNK

    write_profile_vocab(stage, lifelong=("bank_registered", "first_card_activated"))

    artifacts = FrozenArtifacts.load()

    history = prepare(stage, EARLY, with_life(
        QUIET_SNAPSHOT, ("bank_registered", OLD), ("app_registered", OLD + timedelta(days=1))
    ))

    record, _ = encode_profile(artifacts, history, 4)

    assert record.value_ids[-2:] == [
        artifacts.categorical_id("profile_lifelong", "bank_registered"),
        artifacts.special(UNK),
    ]


def test_vocab_learns_milestones_from_train_only(stage):
    """
    В train у клиента только приход, в val — ещё и приложение.
    Словарь видит вехи одного train.
    """

    from src.tokenization.fit import read_train
    from src.tokenization.schema import SemanticSchema
    from src.tokenization.settings import TokenizerConfig

    prepare(stage, EARLY, with_life(QUIET_SNAPSHOT, ("bank_registered", OLD)), group="train")
    prepare(stage, EARLY, with_life(
        QUIET_SNAPSHOT, ("bank_registered", OLD), ("app_registered", OLD + timedelta(days=1))
    ))

    corpus = read_train(TokenizerConfig.load(None), SemanticSchema.open())

    seen = {text for key, _, text in corpus.statistics.categorical if key == "profile_lifelong"}

    assert corpus.group == "train"
    assert seen == {"bank_registered"}


def test_vocab_without_the_milestone_key_is_refused(stage):
    """
    Словарь прежнего кода ключа вех не знает. Кодировать им нельзя:
    вехи ушли бы в неизвестные ключи молча.
    """

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import TransformError, encode_group

    write_profile_vocab(stage, lifelong=None)

    prepare(stage, EARLY, with_life(QUIET_SNAPSHOT, ("bank_registered", OLD)))

    with pytest.raises(TransformError, match="fit"):
        encode_group(FrozenArtifacts.load(), "val", TokenizerConfig.load(None))


# ============================================================
# ПРИМЕР (05)
# ============================================================


def lifelong_client(times: list):
    """
    Анкета [USR], пол и две вехи — время задаёт тест.
    """

    from src.dataset.tokenized import TokenizedClient
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.specials import USR

    artifacts = FrozenArtifacts.load()

    usr = artifacts.special(USR)
    key = artifacts.key_id("profile_lifelong")

    return artifacts, TokenizedClient(
        client_id=RAW_CLIENT,
        events=[],
        profile_key_ids=[usr, artifacts.key_id("profile_gender"), key, key],
        profile_value_ids=[
            usr,
            artifacts.categorical_id("profile_gender", "F"),
            artifacts.categorical_id("profile_lifelong", "bank_registered"),
            artifacts.categorical_id("profile_lifelong", "app_registered"),
        ],
        profile_positions=[0, 0, 0, 0],
        profile_time=times,
    )


def test_sample_carries_the_time_of_milestones_only(stage):

    from src.dataset.sample import SampleError, build_sample
    from src.dataset.settings import ContextPolicy
    from src.preprocessing.settings import PreprocessingConfig

    write_profile_vocab(stage)

    window = PreprocessingConfig.load(None).windows["val"]

    cutoff = window.final_cutoff
    recent = cutoff - timedelta(days=1)

    artifacts, client = lifelong_client([None, None, OLD, recent])

    sample = build_sample(artifacts, client, window, ContextPolicy())

    assert sample.profile_time.tolist() == [
        None, None, OLD.replace(tzinfo=None), recent.replace(tzinfo=None)
    ]

    for times, reason in (
        ([None, OLD, OLD, recent], "не ровно у вех"),
        ([None, None, None, recent], "не ровно у вех"),
        ([None, None, OLD, cutoff], "не раньше cutoff"),
        ([None, None, recent, OLD], "не по времени"),
        ([None, None, OLD], "не по одному на токен"),
    ):
        artifacts, client = lifelong_client(times)

        with pytest.raises(SampleError, match=reason):
            build_sample(artifacts, client, window, ContextPolicy())


# ============================================================
# ВРЕМЯ (06)
# ============================================================


def test_profile_time_is_the_age_before_the_cutoff():
    """
    Ноль у недатированного, 8·log1p(секунды/8) до cutoff у вехи —
    одно и то же у всех кусков значения. Более старая веха дальше.
    """

    from src.temporal.position import profile_time_log

    cutoff = datetime(2026, 1, 1, tzinfo=UTC)

    years = cutoff - timedelta(days=3 * 365)
    month = cutoff - timedelta(days=30)

    found = profile_time_log("c", [None, None, years, month, month], cutoff)

    def age(moment: datetime) -> float:
        seconds = (cutoff - moment).total_seconds()
        return float(np.float32(8.0 * np.log1p(seconds / 8.0)))

    assert found == [0.0, 0.0, age(years), age(month), age(month)]
    assert found[2] > found[3] > 0.0

    # Порядок величины написан руками: три года — около 130.
    assert math.isclose(found[2], 8.0 * math.log1p(3 * 365 * 86_400 / 8.0), rel_tol=1e-6)


def test_milestone_at_or_after_the_cutoff_has_no_time():

    from src.temporal.position import TemporalError, check_profile, profile_time_log

    cutoff = datetime(2026, 1, 1, tzinfo=UTC)

    for moment in (cutoff, cutoff + timedelta(days=1)):
        with pytest.raises(TemporalError, match="не раньше cutoff"):
            profile_time_log("c", [None, moment], cutoff)

    with pytest.raises(TemporalError, match="недатированного"):
        check_profile("c", [0.0, 1.0], [None, None])

    with pytest.raises(TemporalError, match="положительная"):
        check_profile("c", [0.0, 0.0], [None, cutoff - timedelta(days=1)])


def test_time_reaches_the_batches_anchored_at_the_cutoff(stage):
    """
    Выгрузка → 04 → 05 → 06 → 07. Давность вехи считается до T
    группы, а не до последнего события клиента.
    """

    from src.batching.batch import BatchError, check, widths
    from src.batching.build import build_group as build_batches
    from src.batching.settings import BATCHES_FILE, BatchingConfig, batches_dir
    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig
    from src.temporal.build import build_group as build_temporal
    from src.temporal.settings import TEMPORAL_FILE, temporal_dir
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.specials import PAD
    from src.tokenization.transform import encode_group

    write_profile_vocab(stage)

    adopted = datetime(2025, 2, 10, tzinfo=BANK).astimezone(UTC)

    prepare(stage, EARLY, with_life(
        QUIET_SNAPSHOT, ("bank_registered", OLD), ("app_registered", adopted)
    ))

    artifacts = FrozenArtifacts.load()

    encode_group(artifacts, "val", TokenizerConfig.load(None))
    build_dataset(artifacts, "val", DatasetConfig.load(None))
    build_temporal("val")

    row = pq.read_table(temporal_dir("val") / TEMPORAL_FILE).to_pylist()[0]

    cutoff = val_cutoff()

    def age(moment: datetime) -> float:
        seconds = (cutoff - moment).total_seconds()
        return float(np.float32(8.0 * np.log1p(seconds / 8.0)))

    # [USR] и три поля Attributes — ноль, вехи — давность до T.
    assert row["profile_time"] == [None, None, None, None, OLD, adopted]
    assert row["profile_time_log"] == [0.0, 0.0, 0.0, 0.0, age(OLD), age(adopted)]

    # Последнее событие примера раньше T: будь отсчёт от него,
    # давность приложения вышла бы другой.
    assert max(row["event_time"]) < cutoff - timedelta(days=30)

    build_batches("val", BatchingConfig.load(None))

    batch = pq.read_table(batches_dir("val") / BATCHES_FILE).to_pylist()[0]

    assert batch["profile_time_log"] == row["profile_time_log"]

    # Маркер анкеты — якорь: ненулевое время у него отвергается.
    broken = dict(batch, profile_time_log=[1.0] + batch["profile_time_log"][1:])

    with pytest.raises(BatchError, match="маркера анкеты"):
        check(broken, widths([row]), artifacts.special(PAD))


# ============================================================
# ЭНКОДЕР АНКЕТЫ
# ============================================================


def profile_of(model, *made) -> torch.Tensor:

    with torch.no_grad():
        return model._profiles(pack([item.client for item in made], CPU))


def client(name: str, profile: list[tuple]) -> world.Made:
    return world.make(name, [[(world.KEY_A, [10], False)]], profile)


def test_time_of_a_milestone_changes_the_profile_vector(model):

    near = client("c", [(world.KEY_A, [20]), (world.KEY_B, [21], 5.0)])
    far = client("c", [(world.KEY_A, [20]), (world.KEY_B, [21], 50.0)])

    assert not torch.equal(profile_of(model, near), profile_of(model, far))


def test_zero_time_is_the_same_as_no_time(model):
    """
    Поворот на нулевой угол ничего не делает: поле Attributes со
    временем 0 и то же поле без времени дают один вектор.
    """

    dated = client("c", [(world.KEY_A, [20]), (world.KEY_B, [21], 0.0)])
    plain = client("c", [(world.KEY_A, [20]), (world.KEY_B, [21])])

    assert torch.equal(profile_of(model, dated), profile_of(model, plain))

    x = torch.randn(3, world.HEADS, world.DIM // world.HEADS)

    cos, sin = model.profile.rope.angles(torch.zeros(3))

    assert torch.equal(model.profile.rope.rotate(x, cos[:, None], sin[:, None]), x)


def test_attention_is_bidirectional(model):
    """
    Вектор берётся из колонки [USR], и она видит токены после себя:
    причинной маски нет.
    """

    torch.manual_seed(5)

    tokens = torch.randn(1, 3, world.DIM)
    times = torch.tensor([[0.0, 0.0, 12.0]])

    other = tokens.clone()
    other[0, 2] += 1.0

    with torch.no_grad():
        first = model.profile(tokens, times, None)
        second = model.profile(other, times, None)

    assert not torch.equal(first, second)


def test_milestones_of_another_client_do_not_reach_mine(model):
    """
    Соседи одной корзины различаются только временем вех. Мой
    вектор обязан совпасть бит в бит, соседский — измениться.
    """

    mine = client("mine", [(world.KEY_A, [20]), (world.KEY_B, [21], 7.0)])

    first = client("other", [(world.KEY_C, [22]), (world.KEY_B, [23], 3.0)])
    second = client("other", [(world.KEY_C, [22]), (world.KEY_B, [23], 90.0)])

    one = profile_of(model, mine, first)
    two = profile_of(model, mine, second)

    assert torch.equal(one[0], two[0])
    assert not torch.equal(one[1], two[1])


# ============================================================
# СТАРЫЕ АРТЕФАКТЫ
# ============================================================


def test_raw_of_the_previous_contract_is_refused(stage):

    from src.preprocessing.rawdata import RawContractError, check_raw
    from src.preprocessing.settings import raw_group_dir

    directory = raw_group_dir("val")

    cases = (
        ({"as_of": None}, "нет as_of"),
        ({"as_of": AS_OF - timedelta(days=1)}, "выгрузка кончается"),
        ({"lifelong": life(("bank_registered", AS_OF))}, "не раньше as_of"),
        ({"lifelong": life(("account_opened", OLD))}, "вне контракта"),
        # Прежние вехи — тоже вне контракта.
        ({"lifelong": life(("relationship_started", OLD))}, "вне контракта"),
        ({"lifelong": life(("app_adopted", OLD))}, "вне контракта"),
        ({"employment": [{"start_date": date(2020, 1, 1), "record_time": AS_OF}]}, "не раньше as_of"),
        ({"employment": [{"start_date": date(2025, 1, 1), "record_time": OLD}]}, "записана раньше"),
        ({"employment": [{"start_date": None, "record_time": OLD + MICROSECOND},
                         {"start_date": date(2019, 1, 1), "record_time": OLD}]}, "не по времени"),
        ({"lifelong": life(("first_card_activated", OLD), ("first_card_activated", OLD))}, "повторяется"),
        ({"lifelong": life(("app_registered", OLD + MICROSECOND), ("first_card_activated", OLD))}, "не по времени"),
        ({"lifelong": life(("first_card_activated", OLD), ("bank_registered", OLD))}, "не по времени"),
        # Веха о продукте без ссылки на источник и ссылка у вехи,
        # у которой источника в ленте не бывает.
        ({"lifelong": [{"type": "first_loan_opened", "event_time": OLD, "source_id": None}]},
         "нет source_id"),
        ({"lifelong": [{"type": "bank_registered", "event_time": OLD, "source_id": "ctr_1"}]},
         "не бывает"),
    )

    for snapshot, reason in cases:

        write_raw(directory, [], dict(QUIET_SNAPSHOT, **snapshot))

        with pytest.raises(RawContractError, match=reason):
            check_raw(directory)

    write_raw(directory, [], QUIET_SNAPSHOT)

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = 18
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RawContractError, match="schema_version"):
        check_raw(directory)


def test_samples_of_the_previous_format_are_refused(stage):

    from src.dataset.build import SAMPLES_SCHEMA
    from src.dataset.settings import DATASET_FORMAT, dataset_dir
    from src.preprocessing.profile_state import PROFILE_SEMANTICS
    from src.temporal.samples import SamplesError, SamplesGroup

    directory = dataset_dir("val")
    directory.mkdir(parents=True, exist_ok=True)

    pq.write_table(SAMPLES_SCHEMA.empty_table(), directory / "samples.parquet")

    for meta in (
        {"format": 3, "profile_semantics": "state_at_event_cutoff"},
        # Формат 4 — анкета ещё с признаком пенсионера.
        {"format": 4, "profile_semantics": PROFILE_SEMANTICS,
         "profile_lifelong_types": list(LIFELONG_TYPES)},
        {"format": DATASET_FORMAT, "profile_semantics": PROFILE_SEMANTICS},
        {"format": DATASET_FORMAT, "profile_semantics": PROFILE_SEMANTICS,
         "profile_lifelong_types": ["relationship_started"]},
    ):
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(SamplesError, match="прежним кодом"):
            SamplesGroup("val")


def test_stage_06_stamped_by_the_previous_code_is_refused(stage):

    from src.batching.temporal import TemporalError, TemporalGroup
    from src.dataset.lineage import LINEAGE_FILE
    from src.temporal.build import TEMPORAL_SCHEMA
    from src.temporal.settings import TEMPORAL_FILE, temporal_dir

    directory = temporal_dir("val")
    directory.mkdir(parents=True, exist_ok=True)

    pq.write_table(TEMPORAL_SCHEMA.empty_table(), directory / TEMPORAL_FILE)

    (directory / LINEAGE_FILE).write_text(
        json.dumps({"dataset_format": 3, "profile_semantics": "state_at_event_cutoff"}),
        encoding="utf-8",
    )

    with pytest.raises(TemporalError, match="собран из"):
        TemporalGroup("val")


@pytest.mark.parametrize("stamped", ["09_embeddings", "11_profiles"])
def test_weights_without_a_stamp_are_refused(stage, stamped: str):
    """
    Веса 09 собраны под словарь, веса 11 — под анкету. Без отметки
    происхождения load_model их не берёт.
    """

    from src.dataset.lineage import LINEAGE_FILE
    from src.mlm.inputs import InputError
    from src.mlm.model import load_model

    world.install(stage, {"train": [world.population()]})

    load_model("train", 1, 512, 0.1, CPU, "sdpa")

    (stage / stamped / "train" / LINEAGE_FILE).unlink()

    with pytest.raises(InputError, match="прежним кодом"):
        load_model("train", 1, 512, 0.1, CPU, "sdpa")


def test_profile_stage_refuses_unstamped_embeddings(stage):

    from src.dataset.lineage import LINEAGE_FILE
    from src.profile.build import ProfileError, build_group
    from src.profile.settings import ProfileConfig

    world.install(stage, {"train": [world.population()]})

    (stage / "09_embeddings" / "train" / LINEAGE_FILE).unlink()

    with pytest.raises(ProfileError, match="прежним кодом"):
        build_group("train", ProfileConfig(heads=world.HEADS))


def test_history_refuses_profiles_of_the_previous_encoder(stage):

    from src.batching.settings import BATCHES_FILE, batches_dir
    from src.event.build import EVENTS_SCHEMA
    from src.event.settings import EVENTS_FILE, events_dir
    from src.history.inputs import InputError, Source
    from src.profile.build import PROFILES_SCHEMA
    from src.profile.settings import PROFILES_FILE, profiles_dir

    world.write_batches(batches_dir("train") / BATCHES_FILE, [world.population()])

    for directory, name, schema in (
        (events_dir("train"), EVENTS_FILE, EVENTS_SCHEMA),
        (profiles_dir("train"), PROFILES_FILE, PROFILES_SCHEMA),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(schema.empty_table(), directory / name)

    with pytest.raises(InputError, match="прежним кодом"):
        Source("train")


def test_declared_milestones_are_the_raw_contract():

    from src.generator.profile import LIFELONG_TYPES as RAW

    assert LIFELONG_TYPES == RAW == (
        "bank_registered", "app_registered", "first_card_activated",
        "first_loan_opened", "first_deposit_opened",
    )

    for old in ("relationship_started", "kyc_passed", "app_adopted"):
        assert old not in LIFELONG_TYPES
