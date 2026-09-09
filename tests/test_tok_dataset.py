"""
Контракт чтения: пример это профиль плюс упорядоченная история.

Проверяется и то, что данные сохранены полностью (время, seq,
тип события), и то, что маски в них нет.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.preprocessing.config import CLIENT_GROUPS, DATASET_NAMES
from src.tokenizer.build import iter_client_blocks, tokenized_examples_schema
from src.tokenizer.config import EVT_ID, MASK_ID, MISSING_ID, USR_ID, IncompatibleArtifactsError
from src.tokenizer.dataset import TokenizedDataset, collate
from src.tokenizer.encode import PROFILE_WIDTH, tokenized_events_schema, tokenized_profile_schema
from src.tokenizer.vocab import Vocab


@pytest.fixture(scope="module")
def vocab(tok_run) -> Vocab:
    return Vocab.load(tok_run["vocab"])


def dataset(tok_run, name: str) -> TokenizedDataset:
    return TokenizedDataset(tok_run["tokenized"], name, vocab_dir=tok_run["vocab"])


# ============================================================
# СОСТАВ
# ============================================================


def test_all_outputs_exist(tok_run):
    root = tok_run["tokenized"]

    assert (root / "tokenized_manifest.json").exists()

    for group in CLIENT_GROUPS:
        assert (root / "clients" / f"{group}_clients" / "events.parquet").exists()
        assert (root / "clients" / f"{group}_clients" / "profile.parquet").exists()

    for name in DATASET_NAMES:
        assert (root / name / "examples.parquet").exists()

    for name in (
        "special_tokens.json",
        "key_vocab.json",
        "value_vocab.json",
        "field_value_ids.json",
        "tokenizer_config.json",
        "tokenizer_stats.json",
        "golden_examples.json",
    ):
        assert (tok_run["vocab"] / name).exists(), name


def test_schemas_match_declaration(tok_run):
    for group in CLIENT_GROUPS:
        events = pq.read_schema(tok_run["tokenized"] / "clients" / f"{group}_clients" / "events.parquet")
        profile = pq.read_schema(tok_run["tokenized"] / "clients" / f"{group}_clients" / "profile.parquet")

        assert events.remove_metadata().equals(tokenized_events_schema())
        assert profile.remove_metadata().equals(tokenized_profile_schema())

    for name in DATASET_NAMES:
        schema = pq.read_schema(tok_run["tokenized"] / name / "examples.parquet")
        assert schema.remove_metadata().equals(tokenized_examples_schema())


def test_examples_match_preprocessing_one_to_one(tok_run):
    for name in DATASET_NAMES:

        before = pq.read_table(tok_run["processed"] / name / "examples.parquet")
        after = pq.read_table(tok_run["tokenized"] / name / "examples.parquet")

        assert after.num_rows == before.num_rows, name

        for column in ("client_id", "cutoff", "seq_end", "snapshot_ts", "n_events", "client_group"):
            assert after.column(column).to_pylist() == before.column(column).to_pylist(), (name, column)


def test_every_event_of_processed_is_tokenized(tok_run):
    for group in CLIENT_GROUPS:

        before = pq.read_table(tok_run["processed"] / "clients" / f"{group}_clients" / "events.parquet",
                               columns=["client_id", "seq", "ts", "event_type"])
        after = pq.read_table(tok_run["tokenized"] / "clients" / f"{group}_clients" / "events.parquet",
                              columns=["client_id", "seq", "ts", "event_type"])

        assert after.num_rows == before.num_rows, group

        for column in ("client_id", "seq", "ts", "event_type"):
            assert after.column(column).to_pylist() == before.column(column).to_pylist(), (group, column)


# ============================================================
# ЗАПИСЬ БЕЗ МАСОК
# ============================================================


def test_saved_datasets_hold_no_mask(tok_run):
    checked = 0

    for path in sorted(tok_run["tokenized"].rglob("*.parquet")):

        if "value_ids" not in pq.read_schema(path).names:
            continue

        flat = (
            pq.read_table(path, columns=["value_ids"])
            .column("value_ids")
            .combine_chunks()
            .flatten()
            .to_numpy(zero_copy_only=False)
        )

        assert not (flat == MASK_ID).any(), path

        checked += 1

    assert checked >= 6


# ============================================================
# ПРИМЕР
# ============================================================


def test_profile_starts_with_usr_and_has_fixed_width(tok_run):
    data = dataset(tok_run, "train")

    for index in (0, len(data) // 2, len(data) - 1):

        example = data.load(index)

        assert example.profile.n_tokens == PROFILE_WIDTH
        assert example.profile.key_ids[0] == USR_ID
        assert example.profile.value_ids[0] == USR_ID
        assert list(example.profile.positions) == list(range(PROFILE_WIDTH))


def test_every_event_starts_with_evt(tok_run):
    example = dataset(tok_run, "train").load(0)

    events = example.events

    for index in range(events.n_events):
        record = events.event(index)
        assert record.key_ids[0] == EVT_ID
        assert record.value_ids[0] == EVT_ID
        assert record.positions[0] == 0


def test_event_type_is_the_first_field(tok_run, vocab):
    example = dataset(tok_run, "train").load(0)

    events = example.events

    for index in range(min(50, events.n_events)):

        record = events.event(index)

        assert vocab.decode(int(record.key_ids[1])) == "timeline__event_type"
        assert vocab.decode(int(record.value_ids[1])) == str(events.event_type[index])


def test_time_is_kept_next_to_tokens(tok_run):
    data = dataset(tok_run, "train")

    example = data.load(0)

    processed = pq.read_table(
        tok_run["processed"] / "clients" / "train_clients" / "events.parquet",
        columns=["client_id", "seq", "ts"],
    )

    rows = processed.filter(pc.equal(processed.column("client_id"), example.client_id))

    assert list(example.events.seq) == rows.column("seq").to_pylist()[: example.seq_end]
    assert list(example.events.ts) == list(
        np.asarray(rows.column("ts").to_pylist()[: example.seq_end], dtype="datetime64[us]")
    )


def test_history_is_ordered_and_allows_equal_timestamps(tok_run):
    data = dataset(tok_run, "train")

    equal_seen = 0

    for index in range(0, len(data), max(1, len(data) // 40)):

        example = data.load(index)

        ts = example.events.ts
        seq = example.events.seq

        assert (np.diff(ts.astype("int64")) >= 0).all()
        assert (np.diff(seq) == 1).all()

        equal_seen += int((np.diff(ts.astype("int64")) == 0).sum())

    assert equal_seen > 0, "равные timestamps должны встречаться: это дефект, который мы сохраняем"


def test_no_future_data_in_any_example(tok_run):
    for name in DATASET_NAMES:

        data = dataset(tok_run, name)

        for index in range(0, len(data), max(1, len(data) // 25)):

            example = data.load(index)

            boundary = np.datetime64(example.cutoff, "us")

            assert (example.events.ts < boundary).all(), (name, example.client_id)
            assert example.snapshot_ts < example.cutoff
            assert example.events.n_events == example.seq_end


def test_missing_becomes_missing_token(tok_run, vocab):
    """
    Пропуск processed виден в токенах как [MISSING].
    """

    processed = pq.read_table(
        tok_run["processed"] / "clients" / "train_clients" / "events.parquet"
    ).slice(0, 40000)

    tokens = pq.read_table(
        tok_run["tokenized"] / "clients" / "train_clients" / "events.parquet"
    ).slice(0, 40000)

    screens = pc.equal(processed.column("event_type"), "app_screen")

    left = processed.filter(screens)
    right = tokens.filter(screens)

    assert left.num_rows > 0

    reject = left.column("app_screen__reject_reason").to_pylist()

    key_id = vocab.key_id("app_screen", "reject_reason")

    hits = 0

    for value, row in zip(reject, right.to_pylist()):

        position = row["key_ids"].index(key_id)

        if value is None:
            assert row["value_ids"][position] == MISSING_ID
            hits += 1
        else:
            assert row["value_ids"][position] != MISSING_ID

    assert hits > 0


def test_profile_block_missingness_is_visible(tok_run, vocab):
    profile = pq.read_table(tok_run["processed"] / "clients" / "train_clients" / "profile.parquet")
    tokens = pq.read_table(tok_run["tokenized"] / "clients" / "train_clients" / "profile.parquet")

    key_id = vocab.key_id("profile", "income_type")

    values = profile.column("income_type").to_pylist()

    missing = 0

    for value, row in zip(values, tokens.to_pylist()):

        position = row["key_ids"].index(key_id)

        if value is None:
            assert row["value_ids"][position] == MISSING_ID
            missing += 1

    assert missing > 0


# ============================================================
# ОБХОД
# ============================================================


def test_iter_examples_matches_load(tok_run):
    data = dataset(tok_run, "val_client")

    by_key = {}

    for example in data.iter_examples():
        by_key[(example.client_id, example.cutoff)] = example

    assert len(by_key) == len(data)

    for index in range(0, len(data), max(1, len(data) // 20)):

        one = data.load(index)

        other = by_key[(one.client_id, one.cutoff)]

        assert np.array_equal(one.events.key_ids, other.events.key_ids)
        assert np.array_equal(one.events.value_ids, other.events.value_ids)
        assert np.array_equal(one.events.offsets, other.events.offsets)
        assert np.array_equal(one.profile.value_ids, other.profile.value_ids)


def test_collate_keeps_event_and_example_membership(tok_run):
    data = dataset(tok_run, "test_time")

    examples = [data.load(index) for index in range(min(5, len(data)))]

    batch = collate(examples)

    assert batch.n_examples == len(examples)
    assert batch.n_tokens == sum(example.events.n_tokens for example in examples)
    assert batch.n_events == sum(example.events.n_events for example in examples)

    assert batch.event_offsets[-1] == batch.n_tokens

    # Токен принадлежит тому же примеру, что и его событие.
    assert np.array_equal(batch.example_ids, batch.example_of_event[batch.event_ids])

    widths = np.diff(batch.event_offsets)

    assert np.array_equal(
        batch.event_ids,
        np.repeat(np.arange(batch.n_events), widths),
    )

    assert batch.profile_key_ids.size == len(examples) * PROFILE_WIDTH


# ============================================================
# БЛОКИ КЛИЕНТОВ
# ============================================================


def test_client_blocks_do_not_split_a_client(tok_run):
    path = tok_run["tokenized"] / "clients" / "train_clients" / "events.parquet"

    seen: set[int] = set()

    total = 0

    for block in iter_client_blocks(path):

        ids = set(block.column("client_id").to_pylist())

        assert not (ids & seen), "клиент разрезан между блоками"

        seen |= ids
        total += block.num_rows

    assert total == pq.read_metadata(path).num_rows


def test_client_blocks_survive_a_foreign_row_group_layout(tok_run, tmp_path):
    """
    Раскладка row group у реальных данных другая: клиент может
    начинаться в одной группе и заканчиваться в другой.
    """

    path = tok_run["tokenized"] / "clients" / "val_clients" / "events.parquet"

    table = pq.read_table(path)

    rewritten = tmp_path / "tiny.parquet"

    pq.write_table(table, rewritten, row_group_size=7, compression="zstd")

    assert pq.ParquetFile(rewritten).num_row_groups > pq.ParquetFile(path).num_row_groups

    expected = [block.column("client_id").to_pylist() for block in iter_client_blocks(path)]
    actual = [block.column("client_id").to_pylist() for block in iter_client_blocks(rewritten)]

    assert [item for block in actual for item in block] == [item for block in expected for item in block]

    for block in iter_client_blocks(rewritten):
        ids = block.column("client_id").to_numpy()
        assert np.array_equal(ids, np.sort(ids))


# ============================================================
# СТАРЫЙ ДАТАСЕТ НЕ ОТКРЫВАЕТСЯ МОЛЧА
# ============================================================


def test_dataset_rejects_a_foreign_vocab(tok_run, tmp_path):
    import json
    import shutil

    other = tmp_path / "vocab"

    shutil.copytree(tok_run["vocab"], other)

    path = other / "tokenizer_config.json"

    config = json.loads(path.read_text(encoding="utf-8"))
    config["note"] = "другой словарь"

    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(IncompatibleArtifactsError):
        TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=other)
