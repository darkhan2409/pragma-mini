"""
Конвейер целиком: воспроизводимость, отсутствие утечки,
независимость обучаемых artifacts от evaluation-данных.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.generator.config import PROFILE_FIELDS
from src.preprocessing.artifacts import tree_digests
from src.preprocessing.build import events_schema, profile_schema
from src.tokenizer.artifacts import processed_revision
from src.preprocessing.config import (
    CLIENT_GROUPS,
    DATASET_NAMES,
    REGISTRY,
    numeric_specs,
    predictable_specs,
)
from src.preprocessing.cutoffs import CUTOFF_INDEX_SCHEMA, EXAMPLES_SCHEMA
from src.preprocessing.run import run


ROOT = Path(__file__).resolve().parents[1]

FIT_ARTIFACTS = ("bucket_edges.json", "field_stats.json", "unigram_baselines.json")


# ============================================================
# ХЕЛПЕРЫ
# ============================================================


def artifact(prep_run, name: str) -> dict:
    return json.loads((prep_run["artifacts"] / name).read_text(encoding="utf-8"))


def table(prep_run, relative: str) -> pa.Table:
    return pq.read_table(prep_run["processed"] / relative)


def group_of(prep_run) -> dict[int, str]:
    index = table(prep_run, "cutoff_index.parquet")
    return dict(zip(index.column("client_id").to_pylist(), index.column("client_group").to_pylist()))


# ============================================================
# СОСТАВ И СХЕМЫ
# ============================================================


def test_all_outputs_exist(prep_run):
    processed = prep_run["processed"]
    artifacts = prep_run["artifacts"]

    assert (processed / "cutoff_index.parquet").exists()

    for group in CLIENT_GROUPS:
        assert (processed / "clients" / f"{group}_clients" / "events.parquet").exists()
        assert (processed / "clients" / f"{group}_clients" / "profile.parquet").exists()

    for dataset in DATASET_NAMES:
        assert (processed / dataset / "examples.parquet").exists()

    for name in FIT_ARTIFACTS + ("split_manifest.json", "validation_report.json", "validation_report.md"):
        assert (artifacts / name).exists(), name

    assert (artifacts / "value_counts").is_dir()


def test_processed_schemas_match_declaration(prep_run):
    assert table(prep_run, "cutoff_index.parquet").schema.equals(CUTOFF_INDEX_SCHEMA)

    for dataset in DATASET_NAMES:
        assert table(prep_run, f"{dataset}/examples.parquet").schema.equals(EXAMPLES_SCHEMA)

    # Схема событий зависит от ревизии схемы RAW: фикстура это
    # V1, то есть ревизия 1, и колонок session_id у операций и
    # баннеров у неё быть не должно.
    revision = processed_revision(prep_run["artifacts"])

    assert revision == 1

    for group in CLIENT_GROUPS:
        events = table(prep_run, f"clients/{group}_clients/events.parquet")
        assert events.schema.equals(events_schema(revision))
        assert "app_operation__session_id" not in events.schema.names
        assert table(prep_run, f"clients/{group}_clients/profile.parquet").schema.equals(profile_schema())


def test_events_carry_typed_columns_and_buckets(prep_run):
    events = table(prep_run, "clients/train_clients/events.parquet")

    assert "transaction__amount" in events.schema.names
    assert "transaction__amount__bucket" in events.schema.names
    assert "banner__slot" in events.schema.names
    assert "payload" not in events.schema.names

    assert events.schema.field("transaction__amount").type == pa.int64()
    assert events.schema.field("transaction__amount__bucket").type == pa.int16()


def test_row_fills_only_its_own_type(prep_run):
    events = table(prep_run, "clients/train_clients/events.parquet").slice(0, 20000)

    banners = events.filter(pc.equal(events.column("event_type"), "banner"))

    assert banners.num_rows > 0
    assert banners.column("banner__slot").null_count == 0
    assert banners.column("transaction__amount").null_count == banners.num_rows


def test_events_are_sorted_by_client_and_seq(prep_run):
    for group in CLIENT_GROUPS:

        events = table(prep_run, f"clients/{group}_clients/events.parquet")

        client_id = events.column("client_id").to_numpy()
        seq = events.column("seq").to_numpy()

        order = np.lexsort((seq, client_id))

        assert (order == np.arange(len(order))).all(), group


def test_processed_holds_no_labels(prep_run):
    forbidden = {"product_open_90d", "label_start", "label_end"}

    for path in sorted(prep_run["processed"].rglob("*.parquet")):
        assert not (set(pq.read_schema(path).names) & forbidden), path


# ============================================================
# ГРУППЫ КЛИЕНТОВ
# ============================================================


def test_client_groups_are_disjoint_and_complete(prep_run, prep_clients):
    ids = {}

    for group in CLIENT_GROUPS:
        events = table(prep_run, f"clients/{group}_clients/events.parquet")
        ids[group] = set(events.column("client_id").to_pylist())

    for left in CLIENT_GROUPS:
        for right in CLIENT_GROUPS:
            if left != right:
                assert not (ids[left] & ids[right]), (left, right)

    assert len(set().union(*ids.values())) == prep_clients


def test_split_manifest_matches_reality(prep_run, prep_clients):
    manifest = artifact(prep_run, "split_manifest.json")

    assert manifest["clients"]["total"] == prep_clients
    assert sum(manifest["clients"]["counts"].values()) == prep_clients
    assert manifest["clients"]["shares"]["train"] > 0.6

    for dataset in DATASET_NAMES:
        assert manifest["datasets"][dataset]["examples"] == table(prep_run, f"{dataset}/examples.parquet").num_rows


# ============================================================
# ДАТАСЕТЫ
# ============================================================


def test_dataset_definitions(prep_run):
    groups = group_of(prep_run)

    manifest = artifact(prep_run, "split_manifest.json")

    val_cutoff = datetime.fromisoformat(manifest["months"]["by_role"]["val_month"][0])
    test_cutoff = datetime.fromisoformat(manifest["months"]["by_role"]["test_month"][0])

    expected = {
        "train": ("train", None),
        "val_client": ("val", None),
        "test_client": ("test", None),
        "val_time": ("train", val_cutoff),
        "test_time": ("train", test_cutoff),
    }

    for dataset, (group, cutoff) in expected.items():

        rows = table(prep_run, f"{dataset}/examples.parquet")

        assert rows.num_rows > 0, dataset
        assert {groups[cid] for cid in rows.column("client_id").to_pylist()} == {group}

        cutoffs = set(rows.column("cutoff").to_pylist())

        if cutoff is None:
            assert val_cutoff not in cutoffs and test_cutoff not in cutoffs, dataset
        else:
            assert cutoffs == {cutoff}, dataset


def test_one_example_per_client_and_cutoff(prep_run):
    for dataset in DATASET_NAMES:

        rows = table(prep_run, f"{dataset}/examples.parquet")

        keys = list(zip(rows.column("client_id").to_pylist(), rows.column("cutoff").to_pylist()))

        assert len(keys) == len(set(keys)), dataset


# ============================================================
# УТЕЧКА БУДУЩЕГО
# ============================================================


def test_no_future_data_in_any_example(prep_run):
    """
    История примера это префикс seq < seq_end. Внутри него не
    должно быть ни одного события на cutoff или позже, а первое
    отброшенное событие обязано быть не раньше cutoff.
    """

    groups = group_of(prep_run)

    events = {
        group: table(prep_run, f"clients/{group}_clients/events.parquet").select(["client_id", "seq", "ts"])
        for group in CLIENT_GROUPS
    }

    frames = {}

    for group, rows in events.items():
        frames[group] = (
            rows.column("client_id").to_numpy(),
            rows.column("seq").to_numpy(),
            rows.column("ts").to_numpy(),
        )

    checked = 0

    for dataset in DATASET_NAMES:

        rows = table(prep_run, f"{dataset}/examples.parquet")

        for client_id, cutoff, seq_end, snapshot in zip(
            rows.column("client_id").to_pylist(),
            rows.column("cutoff").to_pylist(),
            rows.column("seq_end").to_pylist(),
            rows.column("snapshot_ts").to_pylist(),
        ):
            client_ids, seq, ts = frames[groups[client_id]]

            own = client_ids == client_id

            history = ts[own & (seq < seq_end)]
            future = ts[own & (seq >= seq_end)]

            boundary = np.datetime64(cutoff, "us")

            assert history.size == seq_end
            assert (history < boundary).all(), (dataset, client_id, cutoff)

            if future.size:
                assert future.min() >= boundary, (dataset, client_id, cutoff)

            assert snapshot < cutoff

            checked += 1

    assert checked > 1000


def test_profile_snapshot_of_example_exists(prep_run):
    groups = group_of(prep_run)

    profiles = {
        group: table(prep_run, f"clients/{group}_clients/profile.parquet").select(["client_id", "ts"])
        for group in CLIENT_GROUPS
    }

    known = {
        group: set(zip(rows.column("client_id").to_pylist(), rows.column("ts").to_pylist()))
        for group, rows in profiles.items()
    }

    rows = table(prep_run, "train/examples.parquet").slice(0, 500)

    for client_id, snapshot in zip(rows.column("client_id").to_pylist(), rows.column("snapshot_ts").to_pylist()):
        assert (client_id, snapshot) in known[groups[client_id]]


# ============================================================
# FIT СЧИТАЕТСЯ РОВНО ОДИН РАЗ
# ============================================================


def test_fit_counts_events_once_not_per_example(prep_run, prep_raw_dir):
    """
    Событие входит во множество cutoff-примеров, но в
    статистики попадает один раз.
    """

    stats = artifact(prep_run, "field_stats.json")
    manifest = artifact(prep_run, "split_manifest.json")

    groups = group_of(prep_run)

    fit_cutoff = np.datetime64(datetime.fromisoformat(stats["fit"]["fit_cutoff_max"]), "us")

    timeline = pq.read_table(prep_raw_dir / "timeline.parquet", columns=["client_id", "ts", "event_type"])

    client_id = timeline.column("client_id").to_numpy()
    ts = timeline.column("ts").to_numpy()

    train = np.array([groups.get(int(cid)) == "train" for cid in client_id])

    mask = train & (ts < fit_cutoff)

    assert stats["fields"]["timeline"]["event_type"]["n_total"] == int(mask.sum())

    # Примеров кратно больше, чем событий: счёт не по примерам.
    assert manifest["datasets"]["train"]["examples"] > 1000

    types = np.asarray(timeline.column("event_type").to_pylist(), dtype=object)[mask]

    assert stats["fields"]["transaction"]["amount"]["n_total"] == int((types == "transaction").sum())


def test_fit_profile_uses_selected_snapshots_only(prep_run):
    stats = artifact(prep_run, "field_stats.json")

    examples = table(prep_run, "train/examples.parquet")

    selected = set(zip(examples.column("client_id").to_pylist(), examples.column("snapshot_ts").to_pylist()))

    assert stats["fields"]["profile"]["age"]["n_total"] == len(selected)
    assert stats["fit"]["n_fit_snapshots"] == len(selected)


def test_fit_scope_is_train_only(prep_run):
    stats = artifact(prep_run, "field_stats.json")
    manifest = artifact(prep_run, "split_manifest.json")

    assert stats["fit"]["dataset"] == "train"
    assert stats["fit"]["n_fit_clients"] == manifest["clients"]["counts"]["train"]


# ============================================================
# СТАТИСТИКИ И ГРАНИЦЫ
# ============================================================


def test_field_stats_cover_every_registry_field(prep_run):
    stats = artifact(prep_run, "field_stats.json")

    for spec in REGISTRY.values():
        entry = stats["fields"].get(spec.namespace, {}).get(spec.field)
        assert entry is not None, spec.key

        if not spec.is_feature:
            assert entry["excluded"] is True, spec.key
            assert entry["role"] == spec.role


def test_metadata_entries_say_why_excluded(prep_run):
    stats = artifact(prep_run, "field_stats.json")

    for key in (("app_screen", "session_id"), ("timeline", "seq"), ("labels", "product_open_90d")):
        entry = stats["fields"][key[0]][key[1]]
        assert entry["excluded"] is True
        assert entry["note"]


def test_numeric_fields_have_edges_and_sidecar(prep_run):
    edges = artifact(prep_run, "bucket_edges.json")

    for spec in numeric_specs():

        entry = edges["fields"][spec.namespace][spec.field]

        assert entry["status"] in {"fitted", "constant", "no_fit_data"}

        if entry["status"] == "no_fit_data":
            assert entry["edges"] is None
        else:
            assert entry["actual_bucket_count"] >= 1

        sidecar = prep_run["artifacts"] / "value_counts" / f"{spec.namespace}__{spec.field}.parquet"

        assert sidecar.exists(), spec.key


def test_no_empty_buckets_on_train(prep_run):
    stats = artifact(prep_run, "field_stats.json")

    for spec in numeric_specs():

        buckets = stats["fields"][spec.namespace][spec.field]["buckets"]

        if buckets["status"] != "fitted":
            continue

        assert buckets["empty_buckets"] == 0, spec.key
        assert sum(row[1] for row in buckets["distribution"]) == stats["fields"][spec.namespace][spec.field]["n_valid"]


def test_unigram_covers_predictable_fields(prep_run):
    unigram = artifact(prep_run, "unigram_baselines.json")

    for spec in predictable_specs():
        entry = unigram["fields"].get(spec.namespace, {}).get(spec.field)
        assert entry is not None, spec.key

        if entry["n_valid"]:
            assert 0.0 < entry["mode_probability"] <= 1.0
            assert entry["distribution"]


def test_unigram_excludes_non_predictable(prep_run):
    unigram = artifact(prep_run, "unigram_baselines.json")

    assert "profile" not in unigram["fields"]
    assert "timestamp_quality" not in unigram["fields"].get("product_event", {})


def test_histories_are_not_truncated(prep_run, prep_raw_dir):
    """
    Ограничения по числу событий пока не применяются: длина
    истории только сообщается.
    """

    report = artifact(prep_run, "validation_report.json")

    limit = report["summary"]["history_length"]["max_events_per_history"]

    assert report["summary"]["history_length"]["max"] <= limit

    timeline = pq.read_table(prep_raw_dir / "timeline.parquet", columns=["client_id"])

    stored = sum(
        table(prep_run, f"clients/{group}_clients/events.parquet").num_rows for group in CLIENT_GROUPS
    )

    assert stored == timeline.num_rows


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================


def digests(root: Path) -> dict[str, str]:
    return tree_digests(root)


def test_second_run_is_byte_identical(prep_run, prep_raw_dir, tmp_path):
    again = tmp_path / "again"

    run(prep_raw_dir, "test", again / "processed", again / "artifacts", quiet=True)

    assert digests(again / "processed") == digests(prep_run["processed"])
    assert digests(again / "artifacts") == digests(prep_run["artifacts"])


def test_run_in_subprocess_is_byte_identical(prep_run, prep_raw_dir, tmp_path):
    """
    Другой PYTHONHASHSEED не должен ничего менять.
    """

    import os

    other = tmp_path / "subprocess"

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "12345"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.preprocessing.run",
            "--raw",
            str(prep_raw_dir),
            "--out",
            str(other / "processed"),
            "--artifacts",
            str(other / "artifacts"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-2000:]

    assert digests(other / "processed") == digests(prep_run["processed"])
    assert digests(other / "artifacts") == digests(prep_run["artifacts"])


def test_artifacts_hold_no_paths_or_run_time(prep_run):
    suspicious = re.compile(r"[A-Za-z]:\\\\|/tmp/|pytest-of-|\\bT\\d{2}:\\d{2}:\\d{2}\\.\\d+")

    for path in sorted(prep_run["artifacts"].rglob("*.json")):

        text = path.read_text(encoding="utf-8")

        assert not suspicious.search(text), path

        for token in ("Users", "AppData", "processed_dir", "elapsed", "generated_at"):
            assert token not in text, (path, token)


def test_json_artifacts_end_with_newline(prep_run):
    for path in sorted(prep_run["artifacts"].glob("*.json")):
        assert path.read_bytes().endswith(b"\n"), path
        assert b"\r\n" not in path.read_bytes(), path


# ============================================================
# EVALUATION НЕ ВЛИЯЕТ НА FIT
# ============================================================


@pytest.fixture(scope="module")
def perturbed_run(prep_raw_dir, tmp_path_factory):
    """
    Копия RAW, из которой удалены все evaluation-данные:
    клиенты val и test целиком и хвост истории от val-месяца.
    Обучаемые artifacts обязаны остаться прежними.
    """

    from src.preprocessing.config import DEFAULT_SETTINGS
    from src.preprocessing.splits import assign_groups

    root = tmp_path_factory.mktemp("perturbed")

    raw_dir = root / "raw"

    shutil.copytree(prep_raw_dir, raw_dir)

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))

    coverage = pq.read_table(raw_dir / "source_coverage.parquet")

    groups = assign_groups(
        sorted(set(coverage.column("client_id").to_pylist())),
        DEFAULT_SETTINGS.split_seed,
        DEFAULT_SETTINGS.split_shares,
    )

    keep = {client_id for client_id, group in groups.items() if group == "train"}

    limit = np.datetime64(datetime(2026, 4, 1), "us")

    for name in ("profile", "transactions", "product_events", "communications",
                 "app_screens", "app_operations", "banners", "timeline",
                 "source_coverage", "labels"):

        path = raw_dir / f"{name}.parquet"

        rows = pq.read_table(path)

        mask = np.array([int(cid) in keep for cid in rows.column("client_id").to_numpy()])

        if "ts" in rows.schema.names:
            mask &= rows.column("ts").to_numpy() < limit

        rows = rows.filter(pa.array(mask))

        pq.write_table(rows, path, compression="zstd")

        manifest["rows"][name] = rows.num_rows

    manifest["total_clients"] = len(keep)

    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out = root / "out"

    run(raw_dir, "perturbed", out / "processed", out / "artifacts", quiet=True)

    return out / "artifacts"


def test_dropping_evaluation_data_keeps_fit_artifacts(prep_run, perturbed_run):
    for name in FIT_ARTIFACTS:
        assert (perturbed_run / name).read_bytes() == (prep_run["artifacts"] / name).read_bytes(), name


def test_dropping_evaluation_data_keeps_value_counts(prep_run, perturbed_run):
    original = digests(prep_run["artifacts"] / "value_counts")
    changed = digests(perturbed_run / "value_counts")

    assert changed == original
