from __future__ import annotations

import copy
import random
from datetime import datetime
from importlib import import_module
from types import SimpleNamespace

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
# Кодирование группы обязано давать одно и то же содержимое, как
# бы ни была разложена лента и как бы ни писался результат:
#
#   - группа строк закодированной ленты — ровно один клиент:
#     датасет читает группу строк целиком и держит в памяти одного
#     клиента;
#   - клиент, разрезанный границей группы строк ленты, кодируется
#     целиком, а лента без порядка клиентов идёт прежним путём —
#     и выходит то же;
#   - клиент ленты без анкеты в группу не попадает, как и прежде;
#   - кодирование только читает значения событий;
#   - события, собранные по колонкам, это события построчного
#     разбора: те же значения в том же порядке ключей, те же
#     заметки, та же ошибка данных;
#   - двоичный поиск по шкале находит тот же диапазон, что перебор
#     locate, на любой шкале;
#   - запас разобранных текстов отвечает тем же, что разбор
#     заново, а предел кусков проверяется на каждом значении.
# ============================================================


# Каталоги этапов — в свой временный data/ (как stage в conftest),
# но на весь модуль: мир строится один раз.
PLACES = (
    ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
    ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
    ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """
    Мир генератора до конца окна train, препроцессированный, со
    словарём fit и эталонным кодированием исходной ленты.
    """

    from src.tokenization.run import run_fit

    base = tmp_path_factory.mktemp("encode")

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

        tape = pq.read_table(group_dir("train") / "events.parquet")

        reference = encode(base / "reference")

        yield SimpleNamespace(base=base, tape=tape, reference=reference)


def encode(directory):

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    report = encode_group(FrozenArtifacts.load(), "train", TokenizerConfig.load(None), directory=directory)

    return SimpleNamespace(directory=directory, report=report)


def same_group(left, right) -> None:
    """
    Одинаковое содержимое двух кодирований: таблицы, meta, отчёт и
    клиенты так, как их читает датасет.
    """

    from src.dataset.tokenized import TokenizedGroup
    from src.tokenization.finalvocab import FrozenArtifacts

    for name in ("events.parquet", "profile.parquet", "meta.json"):
        if name.endswith(".parquet"):
            one, two = pq.read_table(left.directory / name), pq.read_table(right.directory / name)
            assert one.schema.equals(two.schema, check_metadata=True), name
            assert one.equals(two), name
        else:
            assert (left.directory / name).read_bytes() == (right.directory / name).read_bytes()

    assert left.report == right.report

    artifacts = FrozenArtifacts.load()

    clients = [list(TokenizedGroup("train", artifacts, directory=item.directory).clients()) for item in (left, right)]

    assert clients[0] == clients[1]
    assert sum(client.n_events for client in clients[0]) > 0


def rewrite_tape(world, table: pa.Table, **options) -> None:
    pq.write_table(table, group_dir("train") / "events.parquet", **options)


@pytest.fixture
def tape(world):
    """Ленту можно переписать в тесте: после него она прежняя."""

    yield world.tape

    rewrite_tape(world, world.tape)


# ------------------------------------------------------------
# раскладка и чтение
# ------------------------------------------------------------


def test_a_row_group_holds_one_client(world):

    for name, clients in (("events.parquet", 12 - world.reference.report["counts"]["silent_clients"]),
                          ("profile.parquet", 12)):

        parquet = pq.ParquetFile(world.reference.directory / name)

        assert parquet.num_row_groups == clients, name

        for index in range(parquet.num_row_groups):
            ids = parquet.read_row_group(index, columns=["client_id"]).column("client_id")
            assert len(set(ids.to_pylist())) == 1, (name, index)


def test_a_client_cut_by_a_row_group_is_encoded_whole(world, tape):

    rewrite_tape(world, tape, row_group_size=97)

    small = pq.ParquetFile(group_dir("train") / "events.parquet")
    edges = [small.read_row_group(index, columns=["client_id"]).column("client_id")
             for index in range(small.num_row_groups)]

    # Проверка не вырождена: клиент лежит на двух группах строк.
    assert any(edges[index][-1] == edges[index + 1][0] for index in range(len(edges) - 1))

    same_group(world.reference, encode(world.base / "cut"))


def test_a_tape_without_client_order_takes_the_old_path(world, tape):

    from src.preprocessing.read import Group

    ids = tape.column("client_id").to_pylist()
    order = sorted(set(ids), reverse=True)

    rewrite_tape(world, tape.take([index for client in order for index, value in enumerate(ids) if value == client]))

    group = Group("train")
    group._addresses()

    assert not group._sorted_runs

    same_group(world.reference, encode(world.base / "shuffled"))


def test_a_client_without_a_profile_stays_out(world, tape):

    from src.preprocessing.read import Group

    first = min(tape.column("client_id").to_pylist())

    # Чужие клиенты между клиентами анкеты и после них, ленту
    # по-прежнему можно обойти потоком.
    strangers = [
        tape.slice(0, 5).set_column(0, "client_id", pa.array([name] * 5))
        for name in (f"{first}~", "~stranger")
    ]

    rewrite_tape(world, pa.concat_tables([tape, *strangers]).sort_by("client_id"))

    group = Group("train")
    group._addresses()

    assert group._sorted_runs

    same_group(world.reference, encode(world.base / "stranger"))


def test_encoding_only_reads_the_values_of_events(world):

    from src.preprocessing.read import Group
    from src.tokenization.encode import encode_event
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.transform import group_window

    artifacts = FrozenArtifacts.load()

    events = [event for history in Group("train").histories(group_window("train").final_cutoff)
              for event in history.events]

    before = copy.deepcopy([list(event.values.items()) for event in events])

    for event in events:
        encode_event(artifacts, event, 256)

    assert [list(event.values.items()) for event in events] == before


# ------------------------------------------------------------
# события по колонкам
# ------------------------------------------------------------


def events_of(table: pa.Table, how) -> tuple:

    from src.preprocessing.read import client_events, client_events_by_rows

    timezone = PreprocessingConfig.load(None).bank_timezone()

    try:
        if how == "columns":
            events, notes = client_events(table, table.column("event_time").to_pylist(), timezone)
        else:
            events, notes = client_events_by_rows(table.to_pylist(), timezone)
    except ValueError as error:
        return type(error), str(error)

    return [(event.client_id, event.event_time, event.source, list(event.values.items()), event.calendar,
             event.lifelong_source) for event in events], notes


def test_columns_give_the_events_of_rows(world):

    tape = world.tape

    for client in sorted(set(tape.column("client_id").to_pylist())):

        table = tape.filter(pc.equal(tape.column("client_id"), client))

        assert events_of(table, "columns") == events_of(table, "rows")

    # Изменения профиля с неразобранным прежним значением дают
    # заметки — в том же порядке.
    changes = tape.filter(pc.equal(tape.column("type"), "profile_change"))

    assert changes.num_rows

    broken = changes.set_column(
        changes.schema.get_field_index("old_value"), "old_value",
        pa.array(["не число"] * changes.num_rows, changes.schema.field("old_value").type),
    )

    columns, rows = events_of(broken, "columns"), events_of(broken, "rows")

    assert columns == rows
    assert any(columns[1])


def test_a_data_error_is_the_error_of_rows(world):

    tape = world.tape

    # Поле без смысла у своего источника: ошибку называет построчный
    # разбор — та же строка и то же поле.
    sources = tape.column("source").to_pylist()

    broken = tape.set_column(
        tape.schema.get_field_index("source"), "source",
        pa.array([f"unknown_{index % 3}" if index % 7 == 3 else value for index, value in enumerate(sources)]),
    )

    from src.preprocessing.keys import KeysError

    columns, rows = events_of(broken, "columns"), events_of(broken, "rows")

    assert columns[0] is KeysError
    assert columns == rows


# ------------------------------------------------------------
# шкалы и тексты
# ------------------------------------------------------------


def scales(seed: int):
    """
    Шкалы любого вида: квантильные с повторами границ, с нулевым
    диапазоном и без, вырожденные, и несплошные — для перебора.
    """

    from src.tokenization.numeric import Bucket, build_bucket_list

    rng = random.Random(seed)

    for number in range(400):

        count = rng.randrange(0, 9)
        edges = sorted(rng.choice([-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 2.0, 1e9, rng.uniform(-10, 10)])
                       for _ in range(count))

        buckets = build_bucket_list("k", tuple(edges), rng.choice(["separate", "none"]))

        if number % 5 == 0 and len(buckets) > 2:
            # Несплошная шкала: перестановка или дыра.
            buckets = list(buckets)
            if rng.random() < 0.5:
                rng.shuffle(buckets)
            else:
                gap = buckets[1]
                buckets[1] = Bucket(gap.name, gap.minimum, (gap.maximum or 0.0) - 0.25 if gap.maximum else None)

        yield tuple(buckets)


def test_the_scale_finds_what_locate_finds():

    from src.tokenization.numeric import BucketsError, Scale, locate

    values = [0, 0.0, -0.0, 1, 1.0, 2, 2.0, 0.5, -1.0, -5.0, -5.0000001, 1e9, 1e9 + 1, -1e12, 3, True, False,
              "1.5", "x", None, float("nan"), float("inf"), -float("inf")]

    rng = random.Random(5)

    values += [rng.uniform(-20, 20) for _ in range(40)]

    def answer(function, value):
        try:
            return "ok", function(value)
        except (BucketsError, TypeError, ValueError) as error:
            return type(error).__name__, str(error)

    checked = 0

    for buckets in scales(11):

        scale = Scale(buckets)

        for value in values:

            expected = answer(lambda item: locate(buckets, item)[1], value)

            assert answer(scale.locate, value) == expected, (buckets, value)

            checked += 1

    assert checked > 10_000


def test_bucket_id_is_locate_on_the_scale_of_its_condition(world):

    from src.preprocessing.read import Group
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.numeric import FOUND_BUCKET, BucketsError, locate
    from src.tokenization.scan import value_text
    from src.tokenization.transform import group_window

    artifacts = FrozenArtifacts.load()

    def by_filter(key, value, record):
        """Прежний bucket_id: отбор диапазонов условия на каждый вызов."""
        buckets = artifacts.buckets.get(key, ())
        split = buckets[0].split_by if buckets else None
        if split is not None:
            if record is None:
                raise BucketsError(f"ключ {key}: шкала делится по {split}, а записи нет")
            condition = record.get(split)
            condition = None if condition is None else value_text(condition)
            buckets = tuple(bucket for bucket in buckets if bucket.when == condition)
        found, bucket = locate(buckets, value)
        return bucket.token_id if found == FOUND_BUCKET and bucket is not None else None

    def answer(function, *items):
        try:
            return "ok", function(*items)
        except BucketsError as error:
            return "error", str(error)

    conditions = {items[0].split_by for items in artifacts.buckets.values() if items and items[0].split_by}

    assert conditions

    checked = 0

    for history in Group("train").histories(group_window("train").final_cutoff):
        for event in history.events:
            unseen = {**event.values, **{condition: "unseen" for condition in conditions}}
            for key, value in event.values.items():
                if value is None or not isinstance(value, (int, float)):
                    continue
                for name in (key, "no_such_key"):
                    for record in (event.values, unseen, {}, None):
                        assert answer(artifacts.bucket_id, name, value, record) == answer(by_filter, name, value, record)
                        checked += 1

    assert checked > 1000


def test_the_text_store_answers_as_parsing_anew(world, monkeypatch):

    from src.preprocessing.canonical.events import normalize_text
    from src.preprocessing.read import Group
    from src.tokenization import encode as encode_module
    from src.tokenization.encode import EncodeError, _text_value_ids
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.transform import group_window

    reference = FrozenArtifacts.load()

    def anew(key, value, limit):
        """Прежний разбор: нормализация, BPE, предел, номера кусков."""
        normalized = normalize_text(value)
        if normalized is None:
            return None
        pieces = reference.bpe.pieces(normalized)
        if len(pieces) > limit:
            return (f"ключ {key}: значение разбилось на {len(pieces)} кусков при пределе {limit}. "
                    "Текст не обрезается: поднимите предел осознанно")
        return [reference.piece_id(piece) for piece in pieces]

    texts = [(key, value) for history in Group("train").histories(group_window("train").final_cutoff)
             for event in history.events for key, value in event.values.items()
             if reference.kind(key) == "text" and isinstance(value, str)]

    texts += [("merchant_name", "   "), ("merchant_name", "МАГНУМ  Cash&Carry"), ("merchant_name", "\ufb01 x"),
              ("merchant_name", "ҚАЗАҚ\u00a0Банкі\tАлматы\n"), ("merchant_name", "")]

    assert len({value for _, value in texts}) > 10

    # Запас один на все значения и такой маленький, что
    # переполняется на ходу; предел то широкий, то в один кусок —
    # и на значении из запаса тоже.
    monkeypatch.setattr(encode_module, "TEXT_CACHE_SIZE", 7)

    shared = FrozenArtifacts.load()

    split = 0

    for key, value in texts:
        for limit in (256, 1, 256):

            try:
                found = _text_value_ids(shared, key, value, limit)
            except EncodeError as error:
                found = str(error)
                split += 1

            assert found == anew(key, value, limit), (key, value, limit)

    assert split
    assert 0 < len(shared.texts) <= 7
