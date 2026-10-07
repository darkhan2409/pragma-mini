from __future__ import annotations

import copy
import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from importlib import import_module
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.generator import emit
from src.preprocessing.canonical.build import build_group
from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сборка примера обязана давать то же, что давал прежний код, как бы
# ни была устроена внутри. Прежний код здесь — эталон, перенесённый
# как был: отбор по EventStub, сборка по событию, проверка обходом.
#
#   - пример каждого клиента тот же: массивы, усечение, потерянные
#     цели, ошибки — при обычной, тесной и полной политике контекста,
#     у молчащего клиента и у анкеты из одного [USR];
#   - таблица файла та же, что from_pylist по спискам примеров;
#   - граница отбора та же, что у прежнего select, на любых размерах
#     и пределах, включая ровно предел и предел плюс один;
#   - проверка примера на любой порче говорит то же, что прежний
#     обход: то же нарушение тем же текстом, и на целом примере молчит;
#   - время и номера переводятся так же, тип события из готовой карты
#     тот же, что из имени токена;
#   - клиент на границе группы строк закодированной ленты читается
#     целиком, равное время сохраняет порядок файла, а сборка не
#     меняет историю клиента.
# ============================================================


PLACES = (
    ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
    ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
    ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ("src.dataset.settings", "DATASET_DIR", "05_dataset"),
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """
    Мир генератора до конца окна train: 02, словарь fit, 04 train.
    """

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.run import run_fit
    from src.tokenization.settings import TokenizerConfig, tokenized_dir
    from src.tokenization.transform import encode_group

    base = tmp_path_factory.mktemp("dataset")

    with pytest.MonkeyPatch.context() as patch:

        for name, attribute, folder in PLACES:
            patch.setattr(import_module(name), attribute, base / "data" / folder)

        emit.generate_dataset(
            total_clients=12, out_dir=raw_group_dir("train"), seed=77, world_seed=42,
            history_start=datetime(2025, 9, 1), history_end=datetime(2026, 1, 1),
            workers=1, community_size=4, quiet=True,
        )

        build_group(raw_group_dir("train"), group_dir("train"), PreprocessingConfig.load(None), "train")

        assert run_fit(SimpleNamespace(config=None)) == 0

        artifacts = FrozenArtifacts.load()

        encode_group(artifacts, "train", TokenizerConfig.load(None))

        tokens = tokenized_dir("train") / "events.parquet"

        yield SimpleNamespace(
            artifacts=artifacts,
            window=PreprocessingConfig.load(None).windows["train"],
            tokens=tokens,
            tape=pq.read_table(tokens),
        )


def clients(world) -> list:

    from src.dataset.tokenized import TokenizedGroup

    return list(TokenizedGroup("train", world.artifacts).clients())


def extra_clients(world, real) -> list:
    """
    Молчащий клиент и анкета из одного [USR] — то, чего в мире может
    не оказаться.
    """

    from src.dataset.tokenized import TokenizedClient
    from src.tokenization.specials import USR

    usr = world.artifacts.special(USR)

    return [
        replace(real[0], client_id="~silent", events=[]),
        TokenizedClient(client_id="~usr", events=real[1].events, profile_key_ids=[usr], profile_value_ids=[usr],
                        profile_positions=[0], profile_time=[None]),
    ]


# ------------------------------------------------------------
# прежний код — эталон
# ------------------------------------------------------------


def old_select(events, policy):

    from src.dataset.context import ContextError, Selection
    from src.dataset.settings import POLICY_ALL, POLICY_RECENT

    for item in events:
        if item.n_tokens > policy.max_event_tokens:
            raise ContextError(
                f"событие {item.index} занимает {item.n_tokens} токенов при пределе "
                f"{policy.max_event_tokens}: ничего не обрезается, поднимите предел осознанно"
            )

    if policy.policy == POLICY_ALL:
        if policy.max_events is not None and len(events) > policy.max_events:
            raise ContextError(
                f"история из {len(events)} событий при политике all и пределе "
                f"{policy.max_events}: выберите политику recent либо снимите предел"
            )
        tokens = sum(item.n_tokens for item in events)
        if policy.max_tokens is not None and tokens > policy.max_tokens:
            raise ContextError(
                f"история из {tokens} токенов при политике all и пределе "
                f"{policy.max_tokens}: выберите политику recent либо снимите предел"
            )
        keep = len(events)
    elif policy.policy == POLICY_RECENT:
        keep = tokens = 0
        for item in reversed(events):
            if policy.max_events is not None and keep == policy.max_events:
                break
            if policy.max_tokens is not None and tokens + item.n_tokens > policy.max_tokens:
                break
            keep += 1
            tokens += item.n_tokens
    else:
        raise ContextError(f"неизвестная политика контекста {policy.policy!r}")

    border = len(events) - keep

    return Selection(
        kept=[item.index for item in events[border:]],
        excluded=[item.index for item in events[:border]],
        kept_tokens=sum(item.n_tokens for item in events[border:]),
        excluded_tokens=sum(item.n_tokens for item in events[:border]),
        excluded_eligible=sum(1 for item in events[:border] if item.eligible),
        truncated=border > 0,
    )


def old_moments(moments) -> np.ndarray:
    return np.asarray(
        [None if moment is None else moment.replace(tzinfo=None) for moment in moments],
        dtype="datetime64[us]",
    )


def old_ints(values) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


def old_build_sample(artifacts, client, window, policy):

    from src.dataset.context import EventStub
    from src.dataset.sample import Sample, SampleError
    from src.dataset.targets import can_be_target, eligible

    flags = [eligible(item.event_time, window) for item in client.events]

    stubs = [EventStub(index=number, n_tokens=item.n_tokens, eligible=flags[number])
             for number, item in enumerate(client.events)]

    selection = old_select(stubs, policy)

    if client.profile_tokens > policy.max_profile_tokens:
        raise SampleError(
            f"клиент {client.client_id}: представление профиля занимает "
            f"{client.profile_tokens} токенов при пределе {policy.max_profile_tokens}"
        )

    key_ids, value_ids, positions, starts, lengths, calendar, moments, mask = [], [], [], [], [], [], [], []

    for position in selection.kept:

        item = client.events[position]

        starts.append(len(key_ids))
        lengths.append(item.n_tokens)

        key_ids.extend(item.key_ids)
        value_ids.extend(item.value_ids)
        positions.extend(item.positions)

        if len(item.calendar) != 6:
            raise SampleError(
                f"клиент {client.client_id}, событие {item.event_time.isoformat()}: "
                f"календарь из {len(item.calendar)} чисел вместо шести"
            )

        calendar.extend(item.calendar)
        moments.append(item.event_time)
        mask.append(flags[position] and can_be_target(item.event_type) and item.lifelong_source is None)

    sample = Sample(
        client_id=client.client_id,
        key_ids=old_ints(key_ids), value_ids=old_ints(value_ids), positions=old_ints(positions),
        event_starts=old_ints(starts), event_lengths=old_ints(lengths), event_time=old_moments(moments),
        calendar=np.asarray(calendar, dtype=np.float32), target_event_mask=np.asarray(mask, dtype=bool),
        profile_key_ids=old_ints(client.profile_key_ids), profile_value_ids=old_ints(client.profile_value_ids),
        profile_positions=old_ints(client.profile_positions), profile_time=old_moments(client.profile_time),
        truncated=selection.truncated, excluded_events=selection.n_excluded,
        excluded_eligible=selection.excluded_eligible,
    )

    old_check(sample, artifacts, window.final_cutoff)

    return sample


def old_check(sample, artifacts, cutoff) -> None:

    from src.dataset.sample import SampleError, _check_positions, _check_profile_time, _check_record
    from src.tokenization.specials import PAD

    s = sample

    if not (s.key_ids.size == s.value_ids.size == s.positions.size):
        raise SampleError(f"{s.client_id}: массивы токенов разной длины")
    if s.event_starts.size != s.event_lengths.size:
        raise SampleError(f"{s.client_id}: границы событий разной длины")
    if s.calendar.size != s.n_events * 6:
        raise SampleError(f"{s.client_id}: календарь не по шесть чисел на событие")
    for name in ("event_time", "target_event_mask"):
        if getattr(s, name).size != s.n_events:
            raise SampleError(f"{s.client_id}: канал {name} не по одному значению на событие")

    covered = 0

    for start, length in zip(s.event_starts.tolist(), s.event_lengths.tolist()):
        if start != covered:
            raise SampleError(f"{s.client_id}: событие начинается в {start}, а покрыто {covered}")
        if length < 1:
            raise SampleError(f"{s.client_id}: событие без единого токена")
        if s.key_ids[start] != s.value_ids[start] or s.positions[start] != 0:
            raise SampleError(f"{s.client_id}: событие начинается не с маркера")
        _check_positions(s.client_id, "событие", s.positions, start + 1, start + length)
        covered += length

    if covered != s.n_tokens:
        raise SampleError(f"{s.client_id}: границы покрывают {covered} токенов из {s.n_tokens}")

    _check_record(s.client_id, "профиль", s.profile_key_ids, s.profile_value_ids, s.profile_positions)
    _check_positions(s.client_id, "профиль", s.profile_positions, 1, s.profile_tokens)
    _check_profile_time(s, artifacts, cutoff)

    pad, size = artifacts.special(PAD), artifacts.size

    for name in ("key_ids", "value_ids", "profile_key_ids", "profile_value_ids"):
        values = getattr(s, name)
        if values.size and (values.min() < 0 or values.max() >= size):
            raise SampleError(f"{s.client_id}: {name} выходит за пространство ID {size}")
        if values.size and bool((values == pad).any()):
            raise SampleError(
                f"{s.client_id}: в {name} встретился [PAD]. Он существует только для "
                "выравнивания batch и в сохранённом примере невозможен"
            )


def old_event_type_of(artifacts, row, event_type_key):

    from src.dataset.tokenized import EVENT_TYPE_KEY
    from src.tokenization.finalvocab import VALUE_PREFIX

    if event_type_key is None:
        return None

    positions = row["positions"]

    for index in range(1, len(positions)):
        if positions[index] != 0:
            continue
        if row["key_ids"][index] != event_type_key:
            continue
        name = artifacts.describe(row["value_ids"][index])
        prefix = f"{VALUE_PREFIX}{EVENT_TYPE_KEY}="
        return name[len(prefix):] if name.startswith(prefix) else None

    return None


def old_row(sample) -> dict:
    from src.dataset.build import SAMPLES_SCHEMA
    return {name: sample.client_id if name == "client_id" else getattr(sample, name).tolist()
            for name in SAMPLES_SCHEMA.names}


def outcome(function, *items):
    try:
        return "ok", function(*items)
    except (ValueError, IndexError) as error:
        return type(error).__name__, str(error)


ARRAYS = ("key_ids", "value_ids", "positions", "event_starts", "event_lengths", "event_time", "calendar",
          "target_event_mask", "profile_key_ids", "profile_value_ids", "profile_positions", "profile_time")


def same_sample(one, two) -> bool:
    return all(
        getattr(one, name).dtype == getattr(two, name).dtype
        and np.array_equal(getattr(one, name).view(np.uint8) if getattr(one, name).dtype.kind == "M"
                           else getattr(one, name),
                           getattr(two, name).view(np.uint8) if getattr(two, name).dtype.kind == "M"
                           else getattr(two, name))
        for name in ARRAYS
    ) and (one.client_id, one.truncated, one.excluded_events, one.excluded_eligible) == (
        two.client_id, two.truncated, two.excluded_events, two.excluded_eligible)


# ------------------------------------------------------------
# пример и файл
# ------------------------------------------------------------


POLICIES = {
    "default": {},
    "tight": {"max_events": 40, "max_tokens": 300, "max_event_tokens": 64},
    "all": {"policy": "all", "max_events": None, "max_tokens": None},
    "all, tight": {"policy": "all", "max_events": 5},
}


@pytest.mark.parametrize("name", POLICIES)
def test_samples_are_the_samples_of_the_old_code(world, name):

    from src.dataset.build import SAMPLES_SCHEMA, _table
    from src.dataset.sample import build_sample
    from src.dataset.settings import ContextPolicy

    policy = ContextPolicy(**POLICIES[name])

    real = clients(world)

    news, olds = [], []

    for client in real + extra_clients(world, real):

        new = outcome(build_sample, world.artifacts, client, world.window, policy)
        old = outcome(old_build_sample, world.artifacts, client, world.window, policy)

        if old[0] != "ok":
            assert new == old, client.client_id
            continue

        assert new[0] == "ok", (client.client_id, new)
        assert same_sample(new[1], old[1]), client.client_id

        news.append(new[1])
        olds.append(old[1])

    if name == "all, tight":
        # Длиннее пяти событий истории нет только у молчащего.
        assert [sample.client_id for sample in news] == ["~silent"]
        return

    assert len(news) == 14
    assert any(sample.n_events == 0 for sample in news)

    if name == "tight":
        assert any(sample.truncated for sample in news)
        assert any(sample.excluded_eligible for sample in news)

    expected = pa.Table.from_pylist([old_row(sample) for sample in olds], schema=SAMPLES_SCHEMA)

    assert _table(news).equals(expected)
    assert _table([]).equals(pa.Table.from_pylist([], schema=SAMPLES_SCHEMA))


def test_a_batch_too_long_for_int32_offsets_takes_the_old_path(world, monkeypatch):

    from src.dataset import build
    from src.dataset.sample import build_sample
    from src.dataset.settings import ContextPolicy

    samples = [build_sample(world.artifacts, client, world.window, ContextPolicy()) for client in clients(world)]

    expected = pa.Table.from_pylist([old_row(sample) for sample in samples], schema=build.SAMPLES_SCHEMA)

    monkeypatch.setattr(build, "OFFSET_LIMIT", 100)

    assert build._table(samples).equals(expected)


def test_the_border_is_the_old_selection():

    from src.dataset.context import EventStub, border, select
    from src.dataset.settings import ContextPolicy

    rng = random.Random(3)

    checked = 0

    for _ in range(600):

        count = rng.randrange(0, 40)
        sizes = [rng.randrange(1, 30) for _ in range(count)]
        total = sum(sizes)
        stubs = [EventStub(index=number, n_tokens=size, eligible=rng.random() < 0.5)
                 for number, size in enumerate(sizes)]

        limit = rng.choice([max(sizes, default=1), 29, 31])
        events = rng.choice([1, max(count, 1), count + 1, rng.randrange(1, 50)])
        tokens = rng.choice([None, max(total, limit), max(total - 1, limit), limit, limit + rng.randrange(0, 200)])

        for policy in (
            ContextPolicy(max_events=events, max_tokens=tokens, max_event_tokens=limit),
            ContextPolicy(policy="all", max_events=rng.choice([None, count, max(count - 1, 1)]),
                          max_tokens=tokens, max_event_tokens=limit),
        ):
            expected = outcome(old_select, stubs, policy)

            assert outcome(select, stubs, policy) == expected

            if expected[0] == "ok":
                assert border(sizes, policy) == expected[1].n_excluded
            else:
                assert outcome(border, sizes, policy) == expected

            checked += 1

    assert checked == 1200


def test_check_says_what_the_old_check_said(world):

    from src.dataset.sample import build_sample
    from src.dataset.settings import ContextPolicy

    cutoff = world.window.final_cutoff

    samples = [build_sample(world.artifacts, client, world.window, ContextPolicy(max_events=30, max_tokens=4096))
               for client in clients(world)[:4]]

    rng = random.Random(9)

    verdicts = {"ok": 0, "error": 0}

    for number in range(800):

        sample = copy.deepcopy(samples[number % len(samples)])

        tokens, events = sample.n_tokens, sample.n_events
        kind = number % 8

        if kind == 0:
            sample.positions[rng.randrange(tokens)] = rng.randrange(0, 4)
        elif kind == 1:
            sample.event_lengths[rng.randrange(events)] += rng.choice([-1, 1])
        elif kind == 2:
            sample.event_starts[rng.randrange(events)] += rng.choice([-1, 1])
        elif kind == 3:
            start = sample.event_starts[rng.randrange(events)]
            sample.key_ids[start] += 1
        elif kind == 4:
            sample.positions[sample.event_starts[rng.randrange(events)]] = 1
        elif kind == 5:
            one, two = rng.randrange(events), rng.randrange(events)
            lengths = sample.event_lengths
            lengths[one], lengths[two] = lengths[two], lengths[one]
        elif kind == 6:
            sample.event_lengths[rng.randrange(events)] = 0
        # kind 7 — пример как есть

        new = outcome(sample.check, world.artifacts, cutoff)

        assert new == outcome(old_check, sample, world.artifacts, cutoff), (number, kind)

        verdicts["ok" if new[0] == "ok" else "error"] += 1

    assert verdicts["ok"] >= 100 and verdicts["error"] >= 400


def test_corrupted_positions_are_still_refused(world):

    from src.dataset.sample import SampleError, build_sample
    from src.dataset.settings import ContextPolicy

    sample = build_sample(world.artifacts, clients(world)[0], world.window, ContextPolicy())

    # Кусок значения без начала: позиция 2 сразу после маркера.
    sample.positions[sample.event_starts[3] + 1] = 2

    with pytest.raises(SampleError, match="куски значения идут подряд от нуля"):
        sample.check(world.artifacts, world.window.final_cutoff)


# ------------------------------------------------------------
# время, номера, тип события
# ------------------------------------------------------------


def test_moments_and_numbers_are_the_old_ones():

    from src.dataset.sample import _ints, _utc_moments

    base = datetime(2025, 3, 1, 4, 5, 6, 789, tzinfo=timezone.utc)
    almaty = ZoneInfo("Asia/Almaty")

    cases = [
        [],
        [None, None],
        [base, None, base.replace(tzinfo=ZoneInfo("UTC")) + timedelta(microseconds=1)],
        [base.replace(tzinfo=None), base],
        [base.astimezone(almaty), base],
        [datetime(1970, 1, 1, tzinfo=timezone.utc), datetime(2262, 4, 1, tzinfo=timezone.utc)],
    ]

    for moments in cases:
        new, old = _utc_moments(moments), old_moments(moments)
        assert new.dtype == old.dtype
        assert np.array_equal(new.view(np.int64), old.view(np.int64)), moments

    source = np.array([1, 2, 3], dtype=np.int32)

    for make in (lambda: [1, 2, 3], lambda: (1, 2, 3), lambda: source, lambda: iter([1, 2, 3]),
                 lambda: np.array([1, 2, 3], dtype=np.int64), lambda: []):
        new = _ints(make())
        assert new.dtype == np.int32 and new.tolist() == old_ints(make()).tolist()

    assert not np.shares_memory(_ints(source), source)


def test_event_types_come_from_the_map(world):

    from src.dataset.tokenized import EVENT_TYPE_KEY, _event_type, event_type_names, event_type_of
    from src.tokenization.finalvocab import VocabError

    artifacts = world.artifacts
    key = artifacts.key_id(EVENT_TYPE_KEY)
    names = event_type_names(artifacts)

    rows = world.tape.select(["key_ids", "value_ids", "positions"]).to_pylist()

    found = [_event_type(artifacts, row["key_ids"], row["value_ids"], row["positions"], key, names) for row in rows]

    assert found == [old_event_type_of(artifacts, row, key) for row in rows]
    assert found == [event_type_of(artifacts, row, key) for row in rows]
    assert None not in found and "profile_change" in found

    # Номер вне словаря и номер не типа события — как через имя.
    strange = [{"key_ids": [3, key], "value_ids": [3, artifacts.size + 5], "positions": [0, 0]},
               {"key_ids": [3, key], "value_ids": [3, artifacts.key_id("currency")], "positions": [0, 0]},
               {"key_ids": [3, key, key], "value_ids": [3, 7, names and next(iter(names))], "positions": [0, 1, 0]}]

    for row in strange:
        expected = outcome(old_event_type_of, artifacts, row, key)
        assert outcome(event_type_of, artifacts, row, key, names) == expected
        assert outcome(event_type_of, artifacts, row, key) == expected

    assert outcome(event_type_of, artifacts, strange[2], key, names) == ("ok", next(iter(names.values())))

    with pytest.raises(VocabError):
        event_type_of(artifacts, strange[0], key, names)


# ------------------------------------------------------------
# чтение закодированной группы
# ------------------------------------------------------------


@pytest.fixture
def tokens(world):
    """Закодированную ленту можно переписать в тесте: после него она прежняя."""

    yield world.tape

    pq.write_table(world.tape, world.tokens)


def test_a_client_cut_by_a_row_group_is_read_whole(world, tokens):

    expected = clients(world)

    pq.write_table(tokens, world.tokens, row_group_size=50)

    small = pq.ParquetFile(world.tokens)
    edges = [small.read_row_group(index, columns=["client_id"]).column("client_id")
             for index in range(small.num_row_groups)]

    assert any(edges[index][-1] == edges[index + 1][0] for index in range(len(edges) - 1))

    assert clients(world) == expected


def test_equal_times_keep_the_order_of_the_file(world, tokens):

    first = tokens.column("client_id")[0].as_py()
    own = pc.equal(tokens.column("client_id"), first).to_numpy(zero_copy_only=False)
    rows = np.flatnonzero(own)

    # Строки клиента задом наперёд, у половины одно и то же время.
    times = tokens.column("event_time").to_pylist()
    moment = times[rows[len(rows) // 2]]

    order = list(rows[::-1])
    changed = [moment if number % 2 else times[index] for number, index in enumerate(order)]

    rest = [index for index in range(tokens.num_rows) if not own[index]]
    table = tokens.take(pa.array(order + rest))
    table = table.set_column(table.schema.get_field_index("event_time"), table.schema.field("event_time"),
                             pa.array(changed + [times[index] for index in rest], table.schema.field("event_time").type))

    pq.write_table(table, world.tokens)

    events = clients(world)[0].events

    # Устойчивая сортировка по времени: равные — в порядке файла.
    written = list(zip(changed, (table.column("value_ids")[number].as_py() for number in range(len(order)))))
    expected = sorted(written, key=lambda item: item[0])

    assert [(event.event_time, event.value_ids) for event in events] == expected
    assert sum(1 for event in events if event.event_time == moment) > 2


def test_building_does_not_change_the_client(world):

    from src.dataset.sample import build_sample
    from src.dataset.settings import ContextPolicy

    for client in clients(world):

        before = copy.deepcopy(client)

        build_sample(world.artifacts, client, world.window, ContextPolicy(max_events=30, max_tokens=4096))

        assert client == before
