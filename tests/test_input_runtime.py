from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from src.generator.rng import KeyedRandom, stable_hash


# ============================================================
# ИДЕЯ
# ============================================================
#
# Подготовка входа модели обязана давать тот же вход, что и прежний
# код, — ни одного другого числа. Прежний код здесь эталон:
# розыгрыш по одному числу, разбор значений обходом, время по
# клиенту, корзины при сборке раскладки.
#
#   - пачка розыгрышей — те же числа, что розыгрыши по одному;
#   - значения разбираются один раз и в том же порядке;
#   - вероятности value массивом те же, что по одному;
#   - выбор (choices) и испорченный контекст (corrupted) те же при
#     любых вероятностях, с весами и без;
#   - маска массивами та же, что apply по списку, и ошибка наложения
#     та же;
#   - время событий и анкеты всей группы строк то же до бита, и
#     ошибки времени называет прежний обход;
#   - клиенты Source те же при любом числе процессов подготовки и
#     обратном порядке групп строк;
#   - корзины раскладки строятся только по требованию и те же, что
#     раньше.
# ============================================================


# ------------------------------------------------------------
# прежний код — эталон
# ------------------------------------------------------------


def old_values(row: dict, targets_only: bool = True) -> list:

    from src.masking.choose import Value

    key_ids, positions = row["key_ids"], row["positions"]
    found = []

    for event, start in enumerate(row["event_starts"]):
        if targets_only and not row["target_event_mask"][event]:
            continue
        end = start + row["event_lengths"][event]
        opened = -1
        for index in range(start + 1, end):
            if positions[index] != 0:
                continue
            if opened >= 0:
                found.append(Value(event, key_ids[opened], opened, index - opened))
            opened = index
        if opened >= 0:
            found.append(Value(event, key_ids[opened], opened, end - opened))

    return found


def old_choose(group: str, row: dict, config, weights=None) -> tuple:

    from src.masking.choose import EVENT, KEY, VALUE, Choice, value_chance

    values = old_values(row)
    eligible = [event for event in range(len(row["event_starts"])) if row["target_event_mask"][event]]

    if not values:
        return len(eligible), values, [], ()

    client = stable_hash(group, row["client_id"]) % (2 ** 31)

    def stream(number: int) -> KeyedRandom:
        return KeyedRandom((config.seed, number, client))

    events = stream(1)
    chosen_events = {event: events.chance(config.event_probability) for event in eligible}
    keys = stream(2)
    chosen_keys = {key_id: keys.chance(config.key_probability) for key_id in sorted({v.key_id for v in values})}
    singles = stream(3)
    chosen_values = [singles.chance(value_chance(value, row, config, weights)) for value in values]
    unknown = stream(4)

    picked = []
    for index, value in enumerate(values):
        if chosen_events[value.event]:
            reason = EVENT
        elif chosen_keys[value.key_id]:
            reason = KEY
        elif chosen_values[index]:
            reason = VALUE
        else:
            continue
        picked.append(Choice(value, reason, unknown.chance(config.unknown_probability)))

    targets = set(eligible)
    streams: dict = {}
    corrupted = []
    for value in old_values(row, targets_only=False):
        if value.event in targets or not chosen_keys.get(value.key_id, False):
            continue
        if value.key_id not in streams:
            streams[value.key_id] = KeyedRandom((config.seed, 5, client, value.key_id))
        if streams[value.key_id].chance(config.key_context_corruption_probability):
            corrupted.append(value)

    return len(eligible), values, picked, tuple(corrupted)


def random_row(rng: random.Random, number: int) -> dict:
    """
    Строка примера: события подряд, у каждого маркер, значения из
    одного и нескольких кусков, ключи повторяются, часть событий —
    цели.
    """

    key_ids, value_ids, positions, starts, lengths, mask = [], [], [], [], [], []

    for _ in range(rng.randrange(0, 12)):

        starts.append(len(key_ids))
        key_ids.append(3)
        value_ids.append(3)
        positions.append(0)

        for _ in range(rng.randrange(0, 5)):
            key = rng.choice([7, 9, 11, 13, 17])
            for piece in range(rng.choice([1, 1, 1, 2, 3])):
                key_ids.append(key)
                value_ids.append(rng.randrange(20, 40))
                positions.append(piece)

        lengths.append(len(key_ids) - starts[-1])
        mask.append(rng.random() < 0.6)

    return {"client_id": f"c{number}", "key_ids": key_ids, "value_ids": value_ids, "positions": positions,
            "event_starts": starts, "event_lengths": lengths, "target_event_mask": mask}


def random_weights(rng: random.Random):

    from src.masking.weights import ValueWeights

    keys = {}
    for key in (7, 9, 13):
        table = {(token,): rng.uniform(0.2, 3.0) for token in range(20, 40) if rng.random() < 0.6}
        table.update({(rng.randrange(20, 40), rng.randrange(20, 40)): rng.uniform(0.2, 3.0) for _ in range(30)})
        keys[key] = (rng.uniform(0.1, 1.0), table)

    return ValueWeights(scale=rng.uniform(0.5, 2.0), keys=keys)


def configs(rng: random.Random):

    from src.masking.settings import MaskingConfig

    for number in range(6):
        yield MaskingConfig(
            seed=rng.randrange(1000),
            value_probability=rng.choice([0.0, 0.15, 0.5, 1.0]),
            event_probability=rng.choice([0.0, 0.1, 0.5, 1.0]),
            key_probability=rng.choice([0.0, 0.1, 0.5, 1.0]),
            unknown_probability=rng.choice([0.0, 0.1, 1.0]),
            key_context_corruption_probability=rng.choice([0.0, 0.5, 1.0]),
            informativeness_weighted_masking=number % 2 == 0,
        )


# ------------------------------------------------------------
# розыгрыш, разбор, вероятности, выбор, маска
# ------------------------------------------------------------


def test_a_batch_of_draws_is_the_draws_one_by_one():

    rng = random.Random(1)

    for _ in range(200):
        key = tuple(rng.randrange(-2 ** 40, 2 ** 40) for _ in range(rng.randrange(1, 5)))
        one, two = KeyedRandom(key), KeyedRandom(key)
        for _ in range(rng.randrange(0, 4)):
            assert one.random() == two.random()
        count = rng.randrange(0, 300)
        assert two.randoms(count).tolist() == [one.random() for _ in range(count)]
        assert one.random() == two.random()


def test_values_are_parsed_once_in_the_old_order():

    from src.masking.choose import _parse, values_of

    rng = random.Random(2)

    for number in range(400):

        row = random_row(rng, number)

        assert _parse(row).as_list() == old_values(row, targets_only=False)
        assert values_of(row) == old_values(row)
        assert values_of(row, targets_only=False) == old_values(row, targets_only=False)


def test_value_chances_are_the_chances_one_by_one():

    from src.masking.choose import _parse, value_chance, value_chances

    rng = random.Random(3)

    for number in range(200):

        row = random_row(rng, number)
        weights = random_weights(rng)

        for config in configs(rng):
            found = _parse(row)
            expected = [value_chance(value, row, config, weights) for value in found.as_list()]
            assert value_chances(found, row, config, weights).tolist() == expected


def test_choices_and_corruption_are_the_old_ones():

    from src.masking.choose import choose

    rng = random.Random(4)

    picked = spoiled = 0

    for number in range(300):

        row = random_row(rng, number)
        weights = random_weights(rng)

        for config in configs(rng):

            events, values, choices, corrupted = old_choose("train", row, config, weights)

            selection = choose("train", row, config, weights)

            assert (selection.events, selection.values, selection.choices, selection.corrupted) == (
                events, values, choices, corrupted)

            picked += len(choices)
            spoiled += len(corrupted)

    assert picked > 1000 and spoiled > 100


def test_the_mask_in_arrays_is_the_mask_of_apply():

    from src.masking.apply import MaskError, apply, apply_selection
    from src.masking.choose import choose

    rng = random.Random(5)

    for number in range(300):

        row = random_row(rng, number)

        for config in configs(rng):

            selection = choose("val", row, config, random_weights(rng))

            closed = apply(row["client_id"], row, selection.choices, 2, 1, selection.corrupted)
            masked = apply_selection(row["client_id"], row, selection, 2, 1)

            assert masked["value_ids"].dtype == np.int64 and masked["labels"].dtype == np.int64
            assert masked["value_ids"].tolist() == closed["value_ids"]
            assert masked["labels"].tolist() == closed["labels"]
            assert masked["reason"] == closed["reason"]

    # Наложение называется той же ошибкой, что у apply.
    from src.masking.settings import MaskingConfig

    config = MaskingConfig(value_probability=1.0, informativeness_weighted_masking=False)
    row = random_row(rng, 0)

    while not choose("val", row, config).choices:
        row = random_row(rng, rng.randrange(10 ** 6))

    selection = choose("val", row, config)

    first = selection.picked[:1]
    twice = SimpleNamespace(
        found=selection.found, spoiled=selection.spoiled,
        picked=np.concatenate([selection.picked, first]),
        reasons=np.concatenate([selection.reasons, selection.reasons[:1]]),
        unknown=np.concatenate([selection.unknown, selection.unknown[:1]]),
        choices=selection.choices + selection.choices[:1], corrupted=selection.corrupted,
    )

    with pytest.raises(MaskError) as expected:
        apply(row["client_id"], row, twice.choices, 2, 1, twice.corrupted)

    with pytest.raises(MaskError, match="выбрана дважды") as found:
        apply_selection(row["client_id"], row, twice, 2, 1)

    assert str(found.value) == str(expected.value)


# ------------------------------------------------------------
# время
# ------------------------------------------------------------


def old_times(column: list[list], cutoff) -> list:

    from src.temporal.position import time_log

    return [time_log(f"c{number}", moments, cutoff) for number, moments in enumerate(column)]


def lists(column: list[list]) -> pa.ChunkedArray:
    return pa.chunked_array([pa.array(column, type=pa.list_(pa.timestamp("us", tz="UTC")))])


def test_times_of_a_row_group_are_the_times_of_each_client():

    from src.temporal.position import TemporalError, checks_hold, profile_time_log, profile_time_logs, time_logs
    from src.temporal.samples import _flat_moments

    rng = random.Random(8)
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for _ in range(100):

        clients = []
        profiles = []

        for _ in range(rng.randrange(0, 8)):
            count = rng.choice([0, 1, 2, rng.randrange(3, 40)])
            moments = sorted(cutoff - timedelta(seconds=rng.randrange(1, 10 ** 8), microseconds=rng.randrange(10 ** 6))
                             for _ in range(count))
            if count > 2 and rng.random() < 0.3:
                moments[1] = moments[0]
            clients.append(moments)
            profiles.append([None if rng.random() < 0.5 else cutoff - timedelta(seconds=rng.randrange(1, 10 ** 8))
                             for _ in range(rng.randrange(1, 6))])

        for anchor, at in (("cutoff", cutoff), ("last_event", None)):
            starts, moments, _ = _flat_moments(lists(clients))
            found = time_logs(starts, moments, at)
            assert checks_hold(starts, found, anchor)
            expected = old_times(clients, at)
            assert [found[first:last].tolist() for first, last in zip(starts[:-1], starts[1:])] == expected

        starts, moments, dated = _flat_moments(lists(profiles))
        found = profile_time_logs(starts, moments, dated, cutoff)
        assert [found[first:last].tolist() for first, last in zip(starts[:-1], starts[1:])] == [
            profile_time_log("c", times, cutoff) for times in profiles]

    # Ошибки: путь массивами отказывается, прежний обход называет.
    late = [[cutoff - timedelta(days=2), cutoff + timedelta(seconds=1)]]
    unordered = [[cutoff - timedelta(days=1), cutoff - timedelta(days=2)]]

    for column in (late, unordered):
        starts, moments, _ = _flat_moments(lists(column))
        assert time_logs(starts, moments, cutoff) is None
        with pytest.raises(TemporalError):
            old_times(column, cutoff)

    starts, moments, dated = _flat_moments(lists([[None, cutoff]]))
    assert profile_time_logs(starts, moments, dated, cutoff) is None


# ------------------------------------------------------------
# Source на мире генератора
# ------------------------------------------------------------


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
    Мир генератора до конца окна train, собранный до 05 с мелкими
    группами строк.
    """

    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig
    from src.generator import emit
    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.run import run_fit
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    base = tmp_path_factory.mktemp("runtime")

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
        build_dataset(artifacts, "train", DatasetConfig(row_group_samples=3))

        yield base


def described(clients) -> list:
    return [
        (client.batch_index, client.client_id, client.reason, client.event_time,
         *[(getattr(client, name).dtype.str, getattr(client, name).tolist()) for name in (
             "key_ids", "value_ids", "positions", "labels", "event_starts", "event_lengths", "event_time_log",
             "calendar", "profile_key_ids", "profile_value_ids", "profile_positions", "profile_time_log")])
        for client in clients
    ]


def old_clients(source) -> list:
    """
    Клиенты прежним путём: время по клиенту, строки to_pylist, выбор
    эталоном, маска apply по спискам.
    """

    from src.masking.apply import apply
    from src.mlm.inputs import Client, _bools, _check, _ints
    from src.temporal.position import check, check_profile, profile_time_log, time_log

    group = source._samples
    out = []

    for index in range(source.count):

        table = group._file.read_row_group(index)
        rows = table.to_pylist()

        for row in rows:

            cutoff = group.cutoff if group.anchor == "cutoff" else None
            row["event_time_log"] = time_log(row["client_id"], row["event_time"], cutoff)
            check(row["client_id"], row["event_time_log"], len(row["event_time"]), group.anchor)
            row["profile_time_log"] = profile_time_log(row["client_id"], row["profile_time"], group.cutoff)
            check_profile(row["client_id"], row["profile_time_log"], row["profile_time"])

            _, _, choices, corrupted = old_choose(source.group, row, source.masking, source.weights)
            masked = apply(row["client_id"], row, choices, source.mask_id, source.unknown_id, corrupted)

            client = Client(
                batch_index=index, client_id=row["client_id"], key_ids=_ints(row["key_ids"]),
                value_ids=_ints(masked["value_ids"]), positions=_ints(row["positions"]),
                labels=_ints(masked["labels"]), reason=list(masked["reason"]),
                event_starts=_ints(row["event_starts"]), event_lengths=_ints(row["event_lengths"]),
                event_time_log=np.asarray(row["event_time_log"], dtype=np.float32),
                calendar=np.asarray(row["calendar"], dtype=np.float32).reshape(-1, 6),
                event_time=list(row["event_time"]), profile_key_ids=_ints(row["profile_key_ids"]),
                profile_value_ids=_ints(row["profile_value_ids"]), profile_positions=_ints(row["profile_positions"]),
                profile_time_log=np.asarray(row["profile_time_log"], dtype=np.float32),
            )
            _check(client, _bools(row["target_event_mask"]), source.mask_id)
            out.append(client)

    return out


def test_source_clients_are_the_old_clients(world):

    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Source

    for group_config in (MaskingConfig(), MaskingConfig(informativeness_weighted_masking=False, seed=5)):

        source = Source("train", masking=group_config)

        new = list(source.clients())

        assert len(new) == 12 and source.count == 4

        # Массивы клиента свои и записываемые, как у прежнего пути.
        assert all(getattr(client, name).flags.writeable for client in new for name in (
            "key_ids", "value_ids", "positions", "labels", "event_starts", "event_lengths", "event_time_log",
            "calendar", "profile_key_ids", "profile_value_ids", "profile_positions", "profile_time_log"))
        assert sum(int((client.labels != -100).sum()) for client in new) > 0
        assert described(new) == described(old_clients(source))


def test_workers_and_order_do_not_change_the_clients(world):

    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Prefetch, Source

    source = Source("train", masking=MaskingConfig(seed=11))

    alone = described(Prefetch(source, 0).clients())

    assert described(Prefetch(source, 2).clients()) == alone

    # Обратный порядок групп строк: те же клиенты в обратном порядке
    # групп — и с процессами, и без.
    source.order = list(reversed(source.order))

    reverse = described(Prefetch(source, 0).clients())

    assert described(Prefetch(source, 3).clients()) == reverse

    by_group = {}
    for item in alone:
        by_group.setdefault(item[0], []).append(item)

    assert reverse == [item for index in reversed(range(4)) for item in by_group[index]]


# ------------------------------------------------------------
# раскладка
# ------------------------------------------------------------


def old_buckets(lengths: np.ndarray) -> list:

    cu = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=cu[1:])
    keys = np.ceil(np.log2(lengths)).astype(np.int64) if lengths.size else lengths
    bucket_of = np.zeros(lengths.size, dtype=np.int64)
    row_of = np.zeros(lengths.size, dtype=np.int64)
    out = []

    for number, key in enumerate(np.unique(keys)):
        segments = np.nonzero(keys == key)[0]
        width = int(lengths[segments].max())
        steps = np.arange(width, dtype=np.int64)[None, :]
        mask = steps < lengths[segments][:, None]
        starts = cu[segments][:, None]
        bucket_of[segments] = number
        row_of[segments] = np.arange(segments.size)
        out.append((segments.tolist(), np.where(mask, starts + steps, starts).tolist(), mask.tolist()))

    return [out, bucket_of.tolist(), row_of.tolist()]


def test_buckets_are_built_only_on_demand_and_are_the_old_ones():

    import torch

    from src.mlm.varlen import VarlenLayout

    rng = np.random.default_rng(9)

    for size in (0, 1, 5, 40, 300):

        lengths = rng.integers(1, 70, size=size).astype(np.int64)

        layout = VarlenLayout.build(lengths, torch.device("cpu"), "сегменты")

        # Путь flash берёт только группы: корзин ещё нет.
        assert "buckets" not in vars(layout)

        expected = old_buckets(lengths)

        assert [layout.bucket_of.tolist(), layout.row_of.tolist()] == expected[1:]
        assert "buckets" not in vars(layout)

        assert [(bucket.segments.tolist(), bucket.index.tolist(), bucket.mask.tolist())
                for bucket in layout.buckets] == expected[0]
