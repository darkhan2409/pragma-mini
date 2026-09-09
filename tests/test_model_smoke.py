"""
Benchmark: сто клиентов, по одному примеру, истории целиком.
"""

from __future__ import annotations

import pytest
import torch

from src.preprocessing.config import DATASET_NAMES
from src.tokenizer.dataset import TokenizedDataset
from src.model.smoke import TARGET_CLIENTS, render, run_benchmark, select_examples


# ============================================================
# ВЫБОР ПРИМЕРОВ
# ============================================================


def all_cutoffs(tok_run) -> dict[int, list]:
    """
    Все cutoff каждого клиента по всем датасетам.
    """

    found: dict[int, list] = {}

    for name in DATASET_NAMES:

        data = TokenizedDataset(tok_run["tokenized"], name, vocab_dir=tok_run["vocab"])

        for client_id, cutoff in zip(
            data.examples.column("client_id").to_pylist(),
            data.examples.column("cutoff").to_pylist(),
        ):
            found.setdefault(int(client_id), []).append(cutoff)

    return found


def test_one_example_per_client(tok_run, prep_clients):
    chosen, warning = select_examples(tok_run["tokenized"], tok_run["vocab"])

    assert warning is None
    assert len(chosen) == prep_clients == TARGET_CLIENTS

    ids = [item.client_id for item in chosen]

    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_last_cutoff_is_taken(tok_run):
    chosen, _ = select_examples(tok_run["tokenized"], tok_run["vocab"], cutoff="last")

    expected = all_cutoffs(tok_run)

    for item in chosen:
        assert item.cutoff == max(expected[item.client_id]), item.client_id


def test_first_cutoff_is_taken(tok_run):
    chosen, _ = select_examples(tok_run["tokenized"], tok_run["vocab"], cutoff="first")

    expected = all_cutoffs(tok_run)

    for item in chosen:
        assert item.cutoff == min(expected[item.client_id]), item.client_id


def test_selection_points_at_a_real_row(tok_run):
    chosen, _ = select_examples(tok_run["tokenized"], tok_run["vocab"])

    for item in chosen[:10]:

        data = TokenizedDataset(tok_run["tokenized"], item.dataset, vocab_dir=tok_run["vocab"])

        row = data.examples.slice(item.index, 1).to_pylist()[0]

        assert int(row["client_id"]) == item.client_id
        assert row["cutoff"] == item.cutoff


def test_too_few_clients_is_reported_not_padded(tok_run):
    """
    Меньше ста клиентов это повод сообщить, а не размножить их.
    """

    chosen, warning = select_examples(
        tok_run["tokenized"], tok_run["vocab"], datasets=("val_client",)
    )

    assert warning is not None
    assert str(len(chosen)) in warning

    assert len(chosen) < TARGET_CLIENTS
    assert len({item.client_id for item in chosen}) == len(chosen)


def test_max_clients_trims_the_selection(tok_run):
    chosen, _ = select_examples(tok_run["tokenized"], tok_run["vocab"], max_clients=5)

    assert len(chosen) == 5


def test_unknown_cutoff_mode_is_rejected(tok_run):
    with pytest.raises(ValueError, match="last или first"):
        select_examples(tok_run["tokenized"], tok_run["vocab"], cutoff="middle")


# ============================================================
# ЗАМЕР
# ============================================================


@pytest.fixture(scope="module")
def report(tok_run) -> dict:
    return run_benchmark(
        root=tok_run["tokenized"],
        vocab=tok_run["vocab"],
        artifacts=tok_run["artifacts"],
        device="cpu",
        batch_examples=4,
        microbatch=256,
        max_clients=8,
        warmup=1,
    )


def test_benchmark_reports_shapes(report):
    assert report["clients"] == 8
    assert report["examples"] == 8

    assert report["event_output_shape"] == (report["events"], 64)
    assert report["profile_output_shape"] == (report["profiles"], 64)

    assert report["profiles"] == report["examples"]


def test_benchmark_counts_every_event(tok_run, report):
    chosen, _ = select_examples(tok_run["tokenized"], tok_run["vocab"], max_clients=8)

    expected = 0

    for item in chosen:
        data = TokenizedDataset(tok_run["tokenized"], item.dataset, vocab_dir=tok_run["vocab"])
        expected += int(data.examples.column("seq_end")[item.index].as_py())

    assert report["events"] == expected


def test_benchmark_measures_time_separately(report):
    assert report["load_seconds"] > 0
    assert report["event_seconds"] > 0
    assert report["profile_seconds"] > 0
    assert report["events_per_second"] > 0


def test_benchmark_keeps_the_widths(report):
    assert report["max_event_tokens"] <= 11
    assert report["max_profile_tokens"] == 21


def test_benchmark_reports_its_settings(report):
    assert report["settings"]["microbatch"] == 256
    assert report["settings"]["batch_examples"] == 4
    assert report["config"]["d_model"] == 64
    assert report["peak_cuda_bytes"] is None


def test_report_renders(report):
    text = render(report)

    assert "EVENT И PROFILE ENCODER" in text
    assert "Event Encoder" in text
    assert str(report["events"]) in text.replace(" ", "")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA недоступна")
def test_benchmark_runs_on_cuda(tok_run):
    report = run_benchmark(
        root=tok_run["tokenized"],
        vocab=tok_run["vocab"],
        artifacts=tok_run["artifacts"],
        device="cuda",
        batch_examples=4,
        microbatch=256,
        max_clients=4,
    )

    assert report["peak_cuda_bytes"] > 0
    assert report["device"] == "cuda"
