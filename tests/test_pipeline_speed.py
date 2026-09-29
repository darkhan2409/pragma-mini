from __future__ import annotations

from datetime import datetime

import pyarrow.parquet as pq
import pytest

from src.generator import emit
from src.pipeline import commands


# ============================================================
# ИДЕЯ
# ============================================================
#
# Ускорения подготовки данных не меняют сами данные:
#
#   - генератор пишет строки по сообществу, воркеры живут в
#     ProcessPoolExecutor, — а выгрузка побайтно та же, сколько бы
#     ни было воркеров; другой размер пачки меняет только раскладку
#     файла, а не строки;
#   - число воркеров по умолчанию помещается в свободную память;
#   - конвейер одной командой строит этапы по порядку и не строит
#     того, что обучение не читает.
# ============================================================


START = datetime(2024, 1, 1)
END = datetime(2024, 3, 1)


def generate(out, workers: int, chunk: int) -> dict:

    return emit.generate_dataset(
        total_clients=16,
        out_dir=out,
        seed=100,
        world_seed=42,
        history_start=START,
        history_end=END,
        workers=workers,
        chunk_clients=chunk,
        community_size=4,
        quiet=True,
    )


def test_workers_do_not_change_the_export(tmp_path):
    """
    Две пачки на одном процессе и на двух — побайтно одна выгрузка;
    пачки вдвое крупнее — те же строки в той же последовательности.
    """

    alone = tmp_path / "alone"
    pooled = tmp_path / "pooled"
    wide = tmp_path / "wide"

    generate(alone, workers=1, chunk=8)
    generate(pooled, workers=2, chunk=8)
    generate(wide, workers=1, chunk=16)

    for name in ("events.parquet", "profile.parquet"):
        assert (alone / name).read_bytes() == (pooled / name).read_bytes(), name
        assert pq.read_table(alone / name).equals(pq.read_table(wide / name)), name

    assert pq.ParquetFile(alone / "events.parquet").num_row_groups == 2


def test_default_workers_fit_into_free_memory(monkeypatch):

    monkeypatch.setattr(emit.os, "cpu_count", lambda: 12)

    monkeypatch.setattr(emit, "available_memory", lambda: None)
    assert emit.default_workers() == 11

    monkeypatch.setattr(emit, "available_memory", lambda: emit.MEMORY_RESERVE + 3 * emit.WORKER_MEMORY)
    assert emit.default_workers() == 3

    monkeypatch.setattr(emit, "available_memory", lambda: emit.MEMORY_RESERVE // 2)
    assert emit.default_workers() == 1


def test_report_reads_one_client(tmp_path):
    """
    Лента одного клиента без чтения всей выгрузки; без явного
    клиента — клиент с самой длинной лентой.
    """

    from src.generator.report.show import load

    generate(tmp_path / "raw", workers=1, chunk=16)

    events = pq.read_table(tmp_path / "raw" / "events.parquet", columns=["client_id"])
    counts = events["client_id"].value_counts().to_pylist()
    busiest = max(counts, key=lambda item: item["counts"])

    shown = load(tmp_path / "raw", None)

    assert shown["client_id"] == busiest["values"]
    assert len(shown["events"]) == busiest["counts"]
    assert {row["client_id"] for row in shown["events"]} == {busiest["values"]}
    assert len(shown["profile"]) == 1


# ============================================================
# КОНВЕЙЕР
# ============================================================


def test_pipeline_runs_stages_in_order_and_builds_only_what_is_read():

    planned = commands("preprocess", "backbone", ("train", "val", "test"), {})

    names = [name for name, _ in planned]
    argv = [" ".join(command[2:]) for _, command in planned]

    order = ["preprocess", "fit", "encode", "dataset", "temporal", "batches", "masks", "embeddings",
             "backbone"]

    assert [name for name in order if name in names] == order
    assert names == sorted(names, key=order.index), "этапы идут строго по порядку"

    assert "src.masking.run train" not in argv, "маска train не строится: обучение разыгрывает её само"
    assert "src.masking.run val" in argv and "src.masking.run test" in argv
    assert [line for line in argv if line.startswith("src.embedding.run")] == ["src.embedding.run train"]


def test_pipeline_range_groups_and_extra_arguments():

    planned = commands("dataset", "batches", ("val",), {"dataset": ["--config", "ctx.json"]})

    assert [" ".join(command[2:]) for _, command in planned] == [
        "src.dataset.run val --config ctx.json",
        "src.temporal.run val",
        "src.batching.run val",
    ]

    with pytest.raises(ValueError, match="идёт после"):
        commands("batches", "dataset", ("val",), {})
