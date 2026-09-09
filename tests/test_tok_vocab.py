"""
Словарь: пространство ID, порядок, кандидаты и совместимость.

Главное свойство — evaluation-данные не влияют на словарь.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.preprocessing.config import REGISTRY, KIND_METADATA, feature_specs, numeric_specs
from src.tokenizer.artifacts import Tokenizer
from src.tokenizer.config import (
    CONFIG_FILE,
    KEY_VOCAB_FILE,
    N_SPECIAL,
    SPECIAL_IDS,
    SPECIAL_TOKENS,
    VALUE_VOCAB_FILE,
    IncompatibleArtifactsError,
)
from src.tokenizer.vocab import Vocab, fit_vocab


# ============================================================
# ПРОСТРАНСТВО ID
# ============================================================


@pytest.fixture(scope="module")
def vocab(tok_run) -> Vocab:
    return Vocab.load(tok_run["vocab"])


def test_special_ids_are_fixed():
    assert SPECIAL_IDS == {
        "[PAD]": 0,
        "[UNK]": 1,
        "[MASK]": 2,
        "[EVT]": 3,
        "[USR]": 4,
        "[MISSING]": 5,
    }

    assert SPECIAL_TOKENS[0] == "[PAD]"
    assert N_SPECIAL == 6


def test_id_space_is_continuous(vocab):
    assert vocab.first_value_id == N_SPECIAL + vocab.n_keys
    assert vocab.size == N_SPECIAL + vocab.n_keys + vocab.n_values

    assert [entry.id for entry in vocab.keys] == list(range(N_SPECIAL, vocab.first_value_id))
    assert [entry.id for entry in vocab.values] == list(range(vocab.first_value_id, vocab.size))


def test_value_ids_of_a_key_are_one_range(vocab):
    for entry in vocab.keys:
        ids = [value.id for value in vocab.values if value.key_id == entry.id]
        assert ids == list(range(entry.value_start, entry.value_end)), entry.key


def test_keys_cover_every_feature_field(vocab):
    expected = {spec.column for spec in feature_specs()}

    assert {entry.key for entry in vocab.keys} == expected


def test_metadata_is_never_a_key(vocab):
    names = {entry.key for entry in vocab.keys}

    for spec in REGISTRY.values():
        if spec.kind == KIND_METADATA:
            assert spec.column not in names, spec.key

    for forbidden in ("app_screen__session_id", "timeline__seq", "timeline__client_id", "timeline__payload"):
        assert forbidden not in names


def test_profile_and_snapshot_are_different_namespaces(vocab):
    left = vocab.key_entry("profile__declared_income")
    right = vocab.key_entry("profile_snapshot__declared_income")

    assert left is not None and right is not None
    assert left.id != right.id
    assert left.predictable is False
    assert right.predictable is True


def test_same_string_in_two_fields_gets_two_ids(vocab):
    online = vocab.key_id("transaction", "is_online")
    delivered = vocab.key_id("communication", "delivered")

    assert vocab.value_id(online, "true") != vocab.value_id(delivered, "true")


def test_value_belongs_to_its_key(vocab):
    for value in vocab.values:
        entry = vocab.key_entry_by_id(value.key_id)
        assert entry is not None
        assert entry.value_start <= value.id < entry.value_end


def test_numeric_values_are_the_train_buckets(tok_run, vocab):
    edges = json.loads((tok_run["artifacts"] / "bucket_edges.json").read_text(encoding="utf-8"))

    for spec in numeric_specs():

        entry = vocab.key_entry(spec.column)

        declared = edges["fields"][spec.namespace][spec.field]

        expected = 0 if declared["status"] == "no_fit_data" else declared["actual_bucket_count"]

        assert entry.n_values == expected, spec.key

        values = [value.value for value in vocab.values if value.key_id == entry.id]

        assert values == [str(index) for index in range(expected)]


def test_categorical_values_are_sorted_by_typed_value(vocab):
    entry = vocab.key_entry("communication__day_of_week")

    values = [value.value for value in vocab.values if value.key_id == entry.id]

    assert values == sorted(values, key=int)
    assert values[0] == "0"


def test_boolean_values_are_false_then_true(vocab):
    entry = vocab.key_entry("transaction__is_online")

    values = [value.value for value in vocab.values if value.key_id == entry.id]

    assert values == ["false", "true"]


# ============================================================
# ЧАСТОТА СЧИТАЕТСЯ ОДИН РАЗ
# ============================================================


def test_frequency_counts_each_record_once(tok_run, vocab):
    """
    Запись входит в десятки cutoff-примеров, но в частоту
    словаря попадает один раз.
    """

    stats = json.loads((tok_run["artifacts"] / "field_stats.json").read_text(encoding="utf-8"))

    entry = vocab.key_entry("timeline__event_type")

    total = sum(value.count for value in vocab.values if value.key_id == entry.id)

    assert total == stats["fields"]["timeline"]["event_type"]["n_total"]

    examples = pq.read_table(tok_run["processed"] / "train" / "examples.parquet")

    # Сумма по примерам кратно больше: счёт идёт не по ним.
    assert int(np.sum(examples.column("n_events").to_numpy())) > 5 * total


def test_profile_counts_use_selected_snapshots(tok_run, vocab):
    stats = json.loads((tok_run["artifacts"] / "field_stats.json").read_text(encoding="utf-8"))

    entry = vocab.key_entry("profile__age")

    total = sum(value.count for value in vocab.values if value.key_id == entry.id)

    assert total == stats["fields"]["profile"]["age"]["n_total"]


# ============================================================
# КАНДИДАТЫ
# ============================================================


def test_field_value_ids_match_vocab(tok_run, vocab):
    stored = json.loads((tok_run["vocab"] / "field_value_ids.json").read_text(encoding="utf-8"))["fields"]

    assert set(stored) == {entry.key for entry in vocab.keys}

    for entry in vocab.keys:

        ids = stored[entry.key]["value_ids"]

        assert ids == sorted(ids)
        assert ids == list(range(entry.value_start, entry.value_end))

        for token_id in ids:
            assert token_id >= vocab.first_value_id
            assert vocab.values[token_id - vocab.first_value_id].key_id == entry.id


def test_candidates_exclude_special_tokens(tok_run):
    stored = json.loads((tok_run["vocab"] / "field_value_ids.json").read_text(encoding="utf-8"))["fields"]

    for entry in stored.values():
        assert not (set(entry["value_ids"]) & set(SPECIAL_IDS.values()))


def test_local_and_global_ids_round_trip(vocab):
    index = vocab.candidates()

    key_ids = np.array([value.key_id for value in vocab.values], dtype=np.int64)
    value_ids = np.array([value.id for value in vocab.values], dtype=np.int64)

    local = index.to_local(key_ids, value_ids)

    assert (local >= 0).all()
    assert (local < index.size_of(key_ids)).all()

    assert np.array_equal(index.to_global(key_ids, local), value_ids)


def test_local_index_starts_at_zero_for_every_key(vocab):
    index = vocab.candidates()

    for entry in vocab.keys:
        if entry.n_values:
            assert index.to_local(np.array([entry.id]), np.array([entry.value_start]))[0] == 0


# ============================================================
# EVALUATION НЕ ВЛИЯЕТ НА СЛОВАРЬ
# ============================================================


@pytest.fixture(scope="module")
def poisoned(prep_run, tmp_path_factory) -> Path:
    """
    Копия processed, в которой evaluation-данные испорчены
    значениями, никогда не встречавшимися на train: клиенты val
    и test целиком и хвост train-клиентов после fit-cutoff.

    Словарь обязан остаться прежним.
    """

    root = tmp_path_factory.mktemp("poisoned")

    processed = root / "processed"

    shutil.copytree(prep_run["processed"], processed)

    stats = json.loads((prep_run["artifacts"] / "field_stats.json").read_text(encoding="utf-8"))

    limit = np.datetime64(datetime.fromisoformat(stats["fit"]["fit_cutoff_max"]), "us")

    for group in ("train", "val", "test"):

        path = processed / "clients" / f"{group}_clients" / "events.parquet"

        table = pq.read_table(path)

        if table.num_rows == 0:
            continue

        touched = (
            table.column("ts").to_numpy() >= limit
            if group == "train"
            else np.ones(table.num_rows, dtype=bool)
        )

        columns = {name: table.column(name) for name in table.schema.names}

        for name, replacement in (("banner__offer", "offer_from_the_future"), ("app_operation__status", "status_x")):

            values = table.column(name).to_pylist()

            values = [
                replacement if (flag and value is not None) else value
                for flag, value in zip(touched, values)
            ]

            columns[name] = pa.array(values, pa.string())

        # Сдвигаем корзину: значение вне обученного диапазона.
        buckets = table.column("transaction__amount__bucket").to_pylist()

        columns["transaction__amount__bucket"] = pa.array(
            [
                (value + 500) if (flag and value is not None) else value
                for flag, value in zip(touched, buckets)
            ],
            pa.int16(),
        )

        pq.write_table(
            pa.table(columns, schema=table.schema), path, compression="zstd"
        )

    return processed


def test_poisoned_evaluation_data_does_not_change_vocab(prep_run, poisoned, tmp_path):
    clean, _ = fit_vocab(prep_run["processed"], prep_run["artifacts"])
    dirty, _ = fit_vocab(poisoned, prep_run["artifacts"])

    left = tmp_path / "clean"
    right = tmp_path / "dirty"

    clean.save(left)
    dirty.save(right)

    for name in ("key_vocab.json", "value_vocab.json", "field_value_ids.json", "special_tokens.json"):
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


def test_poisoned_evaluation_data_becomes_unk(prep_run, poisoned, tmp_path):
    """
    Frozen словарь не расширяется: новое значение это [UNK].
    """

    from src.tokenizer.run import run

    out = tmp_path / "tokenized"
    vocab_out = tmp_path / "vocab"

    result = run(poisoned, prep_run["artifacts"], out, vocab_out, quiet=True)

    stats = result["stats"]["examples"]

    assert stats["train"]["n_unknown"] == 0
    assert stats["val_client"]["n_unknown"] > 0
    assert stats["test_client"]["n_unknown"] > 0

    unknown = result["stats"]["records"]["val_clients"]["events"]["unknown_by_key"]

    assert unknown.get("banner__offer", 0) > 0
    assert unknown.get("transaction__amount", 0) > 0


# ============================================================
# СОВМЕСТИМОСТЬ ARTIFACTS
# ============================================================


def test_tokenizer_loads_with_matching_artifacts(tok_run):
    tokenizer = Tokenizer.load(tok_run["vocab"], tok_run["artifacts"])

    assert tokenizer.size == tokenizer.vocab.size
    assert tokenizer.config["format_version"] == 1


def copy_vocab(tok_run, target: Path) -> Path:
    shutil.copytree(tok_run["vocab"], target)
    return target


def test_edited_value_vocab_is_caught(tok_run, tmp_path):
    directory = copy_vocab(tok_run, tmp_path / "edited")

    data = json.loads((directory / VALUE_VOCAB_FILE).read_text(encoding="utf-8"))
    data["values"][0]["count"] += 1

    (directory / VALUE_VOCAB_FILE).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(IncompatibleArtifactsError):
        Tokenizer.load(directory)


def test_foreign_format_version_is_caught(tok_run, tmp_path):
    directory = copy_vocab(tok_run, tmp_path / "version")

    config = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    config["format_version"] = 99

    (directory / CONFIG_FILE).write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(IncompatibleArtifactsError):
        Tokenizer.load(directory)


def test_changed_bucket_edges_are_caught(tok_run, tmp_path):
    artifacts = tmp_path / "artifacts"

    shutil.copytree(tok_run["artifacts"], artifacts)

    path = artifacts / "bucket_edges.json"

    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(IncompatibleArtifactsError):
        Tokenizer.load(tok_run["vocab"], artifacts)


def test_changed_split_manifest_is_caught_at_fit(prep_run, tmp_path):
    artifacts = tmp_path / "split"

    shutil.copytree(prep_run["artifacts"], artifacts)

    path = artifacts / "split_manifest.json"

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["clients"]["sha256"]["train"] = "0" * 64

    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(IncompatibleArtifactsError):
        fit_vocab(prep_run["processed"], artifacts)


def test_key_vocab_edit_is_caught(tok_run, tmp_path):
    directory = copy_vocab(tok_run, tmp_path / "keys")

    path = directory / KEY_VOCAB_FILE

    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(IncompatibleArtifactsError):
        Tokenizer.load(directory)


# ============================================================
# CONFIG
# ============================================================


def test_config_describes_the_contract(tok_run):
    config = json.loads((tok_run["vocab"] / CONFIG_FILE).read_text(encoding="utf-8"))

    assert config["special_tokens"]["ids"] == SPECIAL_IDS
    assert config["event_format"]["event_type_position"] == 1
    assert config["event_format"]["lead_position"] == 0
    assert config["profile_format"]["lead"] == "[USR]"
    assert config["order_rules"]["keys"]
    assert config["value_rules"]["missing"]
    assert set(config["preprocessing"]["sha256"]) == {
        "bucket_edges.json",
        "field_stats.json",
        "split_manifest.json",
    }
    assert set(config["vocab"]["sha256"]) == {
        "special_tokens.json",
        "key_vocab.json",
        "value_vocab.json",
        "field_value_ids.json",
    }
    assert config["limits"]["max_tokens_per_event"] > 0
    assert config["limits"]["max_events_per_history"] > 0


def test_vocab_artifacts_hold_no_paths_or_run_time(tok_run):
    suspicious = re.compile(r"[A-Za-z]:\\\\|/tmp/|pytest-of-")

    for path in sorted(tok_run["vocab"].glob("*.json")):

        text = path.read_text(encoding="utf-8")

        assert not suspicious.search(text), path

        for token in ("Users", "AppData", "elapsed", "generated_at"):
            assert token not in text, (path, token)

        assert path.read_bytes().endswith(b"\n")
        assert b"\r\n" not in path.read_bytes()
