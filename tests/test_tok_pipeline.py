"""
Конвейер целиком: воспроизводимость, статистика и лимиты.

Лимиты только считаются. Ничего не обрезается, поэтому байты
сохранённых записей от лимитов не зависят.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.preprocessing.artifacts import tree_digests
from src.preprocessing.config import CLIENT_GROUPS, DATASET_NAMES
from src.tokenizer.config import IncompatibleArtifactsError
from src.tokenizer.dataset import TokenizedDataset
from src.tokenizer.encode import PROFILE_WIDTH
from src.tokenizer.run import check_processed, run


ROOT = Path(__file__).resolve().parents[1]


def stats(tok_run) -> dict:
    return json.loads((tok_run["vocab"] / "tokenizer_stats.json").read_text(encoding="utf-8"))


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================


def test_second_run_is_byte_identical(tok_run, tmp_path):
    again = tmp_path / "again"

    run(
        processed_in=tok_run["processed"],
        artifacts_in=tok_run["artifacts"],
        out_dir=again / "tokenized",
        vocab_out=again / "vocab",
        quiet=True,
    )

    assert tree_digests(again / "tokenized") == tree_digests(tok_run["tokenized"])
    assert tree_digests(again / "vocab") == tree_digests(tok_run["vocab"])


def test_run_in_subprocess_is_byte_identical(tok_run, tmp_path):
    """
    Другой PYTHONHASHSEED не должен ничего менять.
    """

    other = tmp_path / "subprocess"

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "31337"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tokenizer.run",
            "--processed",
            str(tok_run["processed"]),
            "--artifacts",
            str(tok_run["artifacts"]),
            "--out",
            str(other / "tokenized"),
            "--vocab-out",
            str(other / "vocab"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-2000:]

    assert tree_digests(other / "tokenized") == tree_digests(tok_run["tokenized"])
    assert tree_digests(other / "vocab") == tree_digests(tok_run["vocab"])


def test_at_least_a_hundred_clients_pass_the_pipeline(tok_run, prep_clients):
    ids: set[int] = set()

    for group in CLIENT_GROUPS:
        table = pq.read_table(
            tok_run["tokenized"] / "clients" / f"{group}_clients" / "events.parquet",
            columns=["client_id"],
        )
        ids |= set(table.column("client_id").to_pylist())

    assert len(ids) == prep_clients >= 100


# ============================================================
# СТАТИСТИКА
# ============================================================


def test_train_has_no_unknown(tok_run):
    assert stats(tok_run)["examples"]["train"]["n_unknown"] == 0


def test_records_are_counted_once(tok_run):
    """
    records считает записи, examples считает то, что видит
    модель. Второе кратно больше первого.
    """

    report = stats(tok_run)

    records = report["records"]["train_clients"]["events"]["n_records"]

    stored = pq.read_metadata(
        tok_run["tokenized"] / "clients" / "train_clients" / "events.parquet"
    ).num_rows

    assert records == stored

    seen_by_model = sum(
        pq.read_table(tok_run["tokenized"] / name / "examples.parquet", columns=["n_events"])
        .column("n_events")
        .to_numpy()
        .sum()
        for name in ("train", "val_time", "test_time")
    )

    assert seen_by_model > 5 * records


def test_stats_cover_every_dataset_and_group(tok_run):
    report = stats(tok_run)

    assert set(report["examples"]) == set(DATASET_NAMES)
    assert set(report["records"]) == {f"{group}_clients" for group in CLIENT_GROUPS}

    for name in DATASET_NAMES:
        summary = report["examples"][name]
        assert summary["n_examples"] > 0
        assert summary["history_length"]["max"] > 0
        assert summary["tokens_per_example"]["min"] >= PROFILE_WIDTH


def test_vocab_sizes_are_reported(tok_run):
    report = stats(tok_run)

    assert report["vocab"]["size"] == report["vocab"]["n_keys"] + report["vocab"]["n_values"] + 6

    assert set(report["vocab"]["values_by_key"]) >= {"timeline__event_type", "profile__age"}


def test_manifest_points_at_the_vocab(tok_run):
    from src.preprocessing.artifacts import sha256_file

    manifest = json.loads(
        (tok_run["tokenized"] / "tokenized_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["tokenizer_config_sha256"] == sha256_file(tok_run["vocab"] / "tokenizer_config.json")
    assert manifest["rows"]["train/examples"] > 0


# ============================================================
# ЛИМИТЫ НЕЗАВИСИМЫ И НИЧЕГО НЕ РЕЖУТ
# ============================================================


@pytest.fixture(scope="module")
def tight(tok_run, tmp_path_factory) -> dict:
    """
    Тот же вход с искусственно маленькими лимитами.
    """

    from src.tokenizer.config import TokenizerSettings

    root = tmp_path_factory.mktemp("tight")

    result = run(
        processed_in=tok_run["processed"],
        artifacts_in=tok_run["artifacts"],
        out_dir=root / "tokenized",
        vocab_out=root / "vocab",
        settings=TokenizerSettings(max_tokens_per_event=5, max_events_per_history=3),
        quiet=True,
    )

    return {"tokenized": root / "tokenized", "vocab": root / "vocab", **result}


def test_default_limits_come_from_the_manifest(tok_run):
    report = stats(tok_run)

    raw = json.loads((tok_run["artifacts"] / "split_manifest.json").read_text(encoding="utf-8"))["raw"]

    assert report["limits"]["max_tokens_per_event"] == raw["max_tokens_per_event"]
    assert report["limits"]["max_events_per_history"] == raw["max_events_per_history"]
    assert report["limits"]["source"] == "split_manifest.raw"


def test_nothing_exceeds_the_real_limits(tok_run):
    report = stats(tok_run)

    for name in DATASET_NAMES:
        assert report["examples"][name]["events_over_token_limit"] == 0, name
        assert report["examples"][name]["histories_over_limit"] == 0, name


def test_tight_limits_count_events_and_histories_separately(tight):
    report = tight["stats"]

    train = report["examples"]["train"]

    assert train["events_over_token_limit"] > 0
    assert train["histories_over_limit"] > 0

    # Два разных параметра: счётчики не обязаны совпадать.
    assert train["events_over_token_limit"] != train["histories_over_limit"]

    assert report["limits"]["source"] == "cli"


def test_tight_limits_do_not_truncate_anything(tok_run, tight):
    """
    Лимит это отчёт, а не нож: записи те же байты.
    """

    for group in CLIENT_GROUPS:
        relative = Path("clients") / f"{group}_clients" / "events.parquet"

        assert (tight["tokenized"] / relative).read_bytes() == (tok_run["tokenized"] / relative).read_bytes()


def test_history_keeps_every_event_of_the_prefix(tok_run):
    data = TokenizedDataset(tok_run["tokenized"], "train", vocab_dir=tok_run["vocab"])

    examples = pq.read_table(tok_run["tokenized"] / "train" / "examples.parquet")

    checked = 0

    for index in range(0, len(data), max(1, len(data) // 30)):

        example = data.load(index)

        row = examples.slice(index, 1).to_pylist()[0]

        assert example.events.n_events == row["n_events"] == row["seq_end"]
        assert example.events.n_tokens == row["n_event_tokens"]
        assert example.profile.n_tokens == row["n_profile_tokens"] == PROFILE_WIDTH
        assert example.n_tokens == row["n_tokens"]

        checked += 1

    assert checked > 10


def test_longest_history_is_stored_whole(tok_run):
    examples = pq.read_table(tok_run["tokenized"] / "test_time" / "examples.parquet")

    n_events = examples.column("n_events").to_numpy()

    position = int(np.argmax(n_events))

    data = TokenizedDataset(tok_run["tokenized"], "test_time", vocab_dir=tok_run["vocab"])

    example = data.load(position)

    assert example.events.n_events == int(n_events[position])
    assert example.events.n_events > 1000


# ============================================================
# ВХОД ПРОВЕРЯЕТСЯ
# ============================================================


def test_foreign_processed_is_rejected(tmp_path):
    with pytest.raises(IncompatibleArtifactsError):
        check_processed(tmp_path)


def test_processed_with_a_wrong_schema_is_rejected(tok_run, tmp_path):
    import shutil

    processed = tmp_path / "processed"

    shutil.copytree(tok_run["processed"], processed)

    path = processed / "clients" / "train_clients" / "events.parquet"

    table = pq.read_table(path)

    pq.write_table(table.drop_columns(["banner__slot"]), path, compression="zstd")

    with pytest.raises(IncompatibleArtifactsError):
        check_processed(processed)
