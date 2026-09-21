from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.dataset.build import BuildError, build_dataset
from src.dataset.reader import Dataset, DatasetError
from src.dataset.settings import ContextPolicy, DatasetConfig
from src.dataset.storage import (
    BUILDING_SUFFIX,
    INDEX_FILE,
    MANIFEST_FILE,
    SHARDS_DIR,
    StorageError,
    prepare_build,
)
from src.preprocessing.artifacts import sha256_file, write_json

from tests.ds_fixtures import config_all, inputs_of, ready


# ============================================================
# ОБЩЕЕ
# ============================================================


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path, Path]:
    return ready(tmp_path_factory.mktemp("ds_storage"))


@pytest.fixture(scope="module")
def built(dataset, tmp_path_factory) -> tuple[Path, str, Path, Path, Path]:
    """
    Собранный набор на фикстуре.
    """

    root, out, target = dataset

    home = tmp_path_factory.mktemp("datasets")

    config = config_all()

    inputs = inputs_of(root, out, target, config)

    result = build_dataset(inputs, config, home)

    return home, result.dataset_id, root, out, target


def _digests(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


# ============================================================
# ЗАПИСЬ И ЧТЕНИЕ
# ============================================================


def test_sample_survives_the_round_trip(built):
    """
    Пример после чтения совпадает с записанным массив за
    массивом.
    """

    home, dataset_id, root, out, target = built

    config = config_all()

    inputs = inputs_of(root, out, target, config)

    stored = Dataset.open(home / dataset_id, artifacts=inputs.artifacts)

    from src.dataset.encoding import causes_as_of, encode_history
    from src.dataset.sample import build_sample

    for group in stored.groups:

        entry = inputs.groups[group]

        for sample in stored.iter_samples(group):

            encoded = encode_history(
                inputs.artifacts,
                entry.history(sample.client_id, sample.cutoff),
                inputs.tokenizer_config.max_pieces_per_value,
                cause_of=causes_as_of(entry.corpus.store, sample.client_id, sample.cutoff),
            )

            fresh = build_sample(
                artifacts=inputs.artifacts,
                encoded=encoded,
                group=group,
                window=entry.window,
                weight=entry.weight,
                sources=inputs.sources(),
                policy=config.context,
            )

            for name in ("key_ids", "value_ids", "positions", "event_starts", "event_lengths",
                         "calendar", "hour_known", "hours_to_cutoff", "event_eligible",
                         "value_event", "value_start", "value_length", "value_key_id",
                         "profile_key_ids", "profile_value_ids", "profile_positions",
                         "coverage_at_cutoff"):
                assert np.array_equal(getattr(fresh, name), getattr(sample, name)), name

            assert fresh.sample_seed == sample.sample_seed
            assert fresh.weight == sample.weight


def test_empty_history_survives_storage(built):
    """
    Клиент без событий остаётся примером: ни одной выдуманной
    покупки и ни одной потерянной строки.
    """

    home, dataset_id, *_ = built

    stored = Dataset.open(home / dataset_id, verify_files=False)

    empty = [
        sample for sample in stored.iter_samples("train") if sample.n_events == 0
    ]

    assert len(empty) == 1

    sample = empty[0]

    assert sample.key_ids.size == 0
    assert sample.profile_tokens == 1, "маркер профиля на месте"
    assert not sample.has_targets


def test_index_and_shards_agree(built):

    home, dataset_id, *_ = built

    stored = Dataset.open(home / dataset_id, verify_files=False)

    index = stored.index.to_pylist()

    assert len(index) == stored.manifest["counts"]["samples"]

    by_id = {row["sample_id"]: row for row in index}

    for group in stored.groups:
        for sample in stored.iter_samples(group):

            row = by_id[sample.sample_id]

            assert row["n_events"] == sample.n_events
            assert row["n_tokens"] == sample.n_tokens
            assert row["n_values"] == sample.n_values


# ============================================================
# ПОРЯДОК ЧТЕНИЯ
# ============================================================


def test_fixed_order_equals_the_index(built):

    home, dataset_id, *_ = built

    stored = Dataset.open(home / dataset_id, verify_files=False)

    index = [row["sample_id"] for row in stored.index.to_pylist() if row["group"] == "train"]

    read = [sample.sample_id for sample in stored.iter_samples("train")]

    assert read == index


def test_shuffle_repeats_for_the_same_epoch(built):

    home, dataset_id, *_ = built

    stored = Dataset.open(home / dataset_id, verify_files=False)

    first = [item.sample_id for item in stored.iter_samples("train", order="shuffled", seed=7, epoch=0)]
    again = [item.sample_id for item in stored.iter_samples("train", order="shuffled", seed=7, epoch=0)]
    other = [item.sample_id for item in stored.iter_samples("train", order="shuffled", seed=7, epoch=1)]

    assert first == again
    assert sorted(first) == sorted(other)


def test_batches_never_mix_groups(built):

    home, dataset_id, *_ = built

    stored = Dataset.open(home / dataset_id, verify_files=False)

    for group in stored.groups:
        for batch in stored.iter_batches(group, batch_size=2):
            assert set(batch.groups) == {group}


def test_token_ceiling_closes_a_batch_early(built):

    home, dataset_id, *_ = built

    stored = Dataset.open(home / dataset_id, verify_files=False)

    wide = list(stored.iter_batches("train", batch_size=10))
    narrow = list(stored.iter_batches("train", batch_size=10, max_batch_tokens=100))

    assert len(narrow) >= len(wide)

    # Потолок считается после выравнивания, и он соблюдается
    # везде, кроме batch из одного примера: одиночный пример
    # разрезать нельзя.
    for batch in narrow:
        assert batch.padded_tokens <= 100 or batch.n_samples == 1


# ============================================================
# ЦЕЛОСТНОСТЬ
# ============================================================


def test_tampered_shard_is_refused(built, tmp_path):

    home, dataset_id, *_ = built

    copy = tmp_path / "copy"

    _copy_tree(home / dataset_id, copy / dataset_id)

    shard = next((copy / dataset_id / SHARDS_DIR).glob("*.parquet"))

    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF
    shard.write_bytes(bytes(data))

    with pytest.raises(DatasetError, match="изменился после сборки"):
        Dataset.open(copy / dataset_id)


def test_missing_file_is_refused(built, tmp_path):

    home, dataset_id, *_ = built

    copy = tmp_path / "gone"

    _copy_tree(home / dataset_id, copy / dataset_id)

    (copy / dataset_id / INDEX_FILE).unlink()

    with pytest.raises(DatasetError, match="пропал из набора"):
        Dataset.open(copy / dataset_id)


def test_edited_manifest_is_refused(built, tmp_path):

    home, dataset_id, *_ = built

    copy = tmp_path / "edited"

    _copy_tree(home / dataset_id, copy / dataset_id)

    path = copy / dataset_id / MANIFEST_FILE

    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["identity"]["config_sha256"] = "0" * 64
    write_json(path, manifest)

    with pytest.raises(DatasetError, match="манифест изменён после сборки"):
        Dataset.open(copy / dataset_id)


def test_unfinished_build_never_looks_ready(built, tmp_path):
    """
    Каталог сборки готовым набором не объявляется, а публикация
    без манифеста невозможна.
    """

    home, dataset_id, *_ = built

    building = tmp_path / f"{dataset_id}{BUILDING_SUFFIX}"
    building.mkdir()

    with pytest.raises(DatasetError, match="незавершённой сборки"):
        Dataset.open(building)

    from src.dataset.storage import publish

    with pytest.raises(StorageError, match="нет манифеста"):
        publish(tmp_path, dataset_id)


def test_existing_dataset_needs_force(built, tmp_path):

    home, dataset_id, *_ = built

    with pytest.raises(StorageError, match="уже собран"):
        prepare_build(home, dataset_id, force=False)

    directory = prepare_build(home, dataset_id, force=True)

    assert directory.name.endswith(BUILDING_SUFFIX)
    assert (home / dataset_id / MANIFEST_FILE).exists(), "готовый набор до публикации цел"


def _copy_tree(source: Path, destination: Path) -> None:

    destination.mkdir(parents=True, exist_ok=True)

    for path in source.rglob("*"):

        target = destination / path.relative_to(source)

        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


# ============================================================
# ВОСПРОИЗВОДИМОСТЬ
# ============================================================


def test_rebuild_gives_the_same_bytes(dataset, tmp_path):

    root, out, target = dataset

    config = config_all()

    first = tmp_path / "one"
    second = tmp_path / "two"

    one = build_dataset(inputs_of(root, out, target, config), config, first)
    two = build_dataset(inputs_of(root, out, target, config), config, second)

    assert one.dataset_id == two.dataset_id
    assert _digests(one.directory) == _digests(two.directory)


def test_build_is_byte_identical_in_another_process(dataset, tmp_path):
    """
    Другой процесс с другим PYTHONHASHSEED даёт те же байты.

    Порядок обхода словарей Python зависит от хэшей строк, и
    статистика, собранная в таком словаре, легко начинает
    зависеть от запуска.
    """

    root, out, target = dataset

    config = config_all()

    here = build_dataset(inputs_of(root, out, target, config), config, tmp_path / "here")

    script = (
        "from pathlib import Path;"
        "from src.dataset.build import build_dataset;"
        "from src.dataset.inputs import DatasetInputs;"
        "from src.dataset.settings import DatasetConfig;"
        "config = DatasetConfig();"
        f"inputs = DatasetInputs.open(Path({str(out)!r}), Path({str(root)!r}),"
        f" Path({str(target)!r}), config);"
        f"build_dataset(inputs, config, Path({str(tmp_path / 'there')!r}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONHASHSEED": "31337", "PYTHONPATH": str(Path.cwd())},
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")

    there = tmp_path / "there" / here.dataset_id

    assert _digests(there) == _digests(here.directory)


def test_another_policy_is_another_dataset(dataset, tmp_path):
    """
    Политика контекста входит в имя набора: два решения не
    ложатся в один каталог.
    """

    root, out, target = dataset

    first = config_all()
    second = replace(
        DatasetConfig(),
        context=ContextPolicy(policy="recent_plus_milestones", max_events=4, max_tokens=100_000),
    )

    one = build_dataset(inputs_of(root, out, target, first), first, tmp_path / "home")
    two = build_dataset(inputs_of(root, out, target, second), second, tmp_path / "home")

    assert one.dataset_id != two.dataset_id
    assert one.directory.exists() and two.directory.exists()


# ============================================================
# ПРАВИЛО ОЦЕНОЧНЫХ ГРУПП
# ============================================================


def test_lost_target_in_an_evaluation_group_stops_the_build(dataset, tmp_path):
    """
    Потерять цель в validation или test нельзя: оценка стала бы
    зависеть от политики усечения.
    """

    root, out, target = dataset

    config = replace(
        DatasetConfig(),
        context=ContextPolicy(
            policy="recent_plus_milestones",
            max_events=1,
            max_tokens=100_000,
            milestone_share=0.0,
        ),
    )

    inputs = inputs_of(root, out, target, config)

    with pytest.raises(BuildError, match="период[а]? целей"):
        build_dataset(inputs, config, tmp_path / "strict")


# ============================================================
# ГРАНИЦЫ ПАКЕТА
# ============================================================


def test_dataset_does_not_import_the_old_model():
    """
    Датасет не зависит ни от прежнего слоя модели, ни от torch.
    """

    script = (
        "import sys;"
        "import src.dataset.build, src.dataset.reader, src.dataset.collate;"
        "names = [name for name in sys.modules if name.startswith('src.model') or name == 'torch'];"
        "print(names)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8", "replace").strip() == "[]"
