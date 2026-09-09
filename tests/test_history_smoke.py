"""
Замер при разных лимитах истории.

Лимит это потолок, а не факт: отчёт обязан показывать
фактические длины, иначе «прогнали на 4096» ничего не значит.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.model.history_smoke import DEFAULT_LIMITS, render, run_history_benchmark, summarize_lengths


# ============================================================
# СТАТИСТИКА ДЛИН
# ============================================================


def test_summarize_reports_every_percentile():
    values = np.arange(1, 101)

    summary = summarize_lengths(values)

    assert set(summary) == {"p50", "p90", "p95", "p99", "max", "mean"}
    assert summary["max"] == 100
    assert summary["p50"] <= summary["p90"] <= summary["p95"] <= summary["p99"] <= summary["max"]


def test_summarize_handles_an_empty_input():
    summary = summarize_lengths(np.array([]))

    assert summary["max"] == 0


# ============================================================
# ЗАМЕР
# ============================================================


@pytest.fixture(scope="module")
def report(tok_run) -> dict:
    return run_history_benchmark(
        root=tok_run["tokenized"],
        vocab=tok_run["vocab"],
        artifacts=tok_run["artifacts"],
        device="cpu",
        limits=(8, 16),
        max_clients=6,
        batch_examples=2,
        event_microbatch=256,
        warmup=1,
    )


def test_every_limit_is_measured(report):
    assert [item["limit"] for item in report["results"]] == [8, 16]
    assert report["failure"] is None


def test_used_length_respects_the_limit(report):
    for item in report["results"]:
        assert item["used"]["max"] <= item["limit"]


def test_a_tighter_limit_truncates_more(report):
    tight, loose = report["results"]

    assert tight["truncated_share"] >= loose["truncated_share"]
    assert tight["events"] < loose["events"]


def test_events_processed_match_the_used_lengths(report):
    for item in report["results"]:
        assert item["events"] == item["used"]["mean"] * item["examples"]


def test_original_length_does_not_depend_on_the_limit(report):
    first, second = report["results"]

    assert first["original"] == second["original"]
    assert first["original"]["max"] > second["limit"]


def test_shapes_are_reported(report):
    for item in report["results"]:
        assert item["client_embedding_shape"] == (item["examples"], 64)
        assert item["contextualized_shape"][1] == 1 + item["used"]["max"]


def test_times_are_measured_separately(report):
    assert report["load_seconds"] > 0

    for item in report["results"]:
        assert item["prepare_seconds"] > 0
        assert item["event_seconds"] > 0
        assert item["profile_seconds"] > 0
        assert item["history_seconds"] > 0
        assert item["forward_seconds"] == pytest.approx(
            item["event_seconds"] + item["profile_seconds"] + item["history_seconds"]
        )


def test_settings_are_fixed_across_limits(report):
    assert report["settings"]["batch_examples"] == 2
    assert report["settings"]["event_microbatch"] == 256
    assert report["settings"]["dtype"] == "float32"


def test_reaching_the_limit_is_reported(report):
    for item in report["results"]:
        assert item["reached_limit"] == (item["used"]["max"] >= item["limit"])


def test_report_renders(report):
    text = render(report)

    assert "HISTORY ENCODER" in text
    assert "исходные длины" in text
    assert "подготовка" in text


def test_short_data_is_flagged(tok_run):
    """
    Лимит выше самой длинной истории обязан отмечаться: иначе
    из отчёта нельзя понять, проверена вместимость или нет.
    """

    report = run_history_benchmark(
        root=tok_run["tokenized"],
        vocab=tok_run["vocab"],
        artifacts=tok_run["artifacts"],
        device="cpu",
        limits=(DEFAULT_LIMITS[-1],),
        max_clients=2,
        batch_examples=2,
    )

    item = report["results"][0]

    assert item["reached_limit"] is False
    assert "вместимость не проверена" in render(report)


# ============================================================
# ПОТОКОВЫЙ РЕЖИМ
# ============================================================


def test_streaming_matches_the_plain_run(tok_run, report):
    """
    Потоковый режим меняет только порядок обхода и объём
    памяти, но не то, что измеряется.
    """

    streamed = run_history_benchmark(
        root=tok_run["tokenized"],
        vocab=tok_run["vocab"],
        artifacts=tok_run["artifacts"],
        device="cpu",
        limits=(8, 16),
        max_clients=6,
        batch_examples=2,
        event_microbatch=256,
        warmup=1,
        stream=True,
    )

    assert streamed["settings"]["stream"] is True

    for left, right in zip(report["results"], streamed["results"]):

        assert left["limit"] == right["limit"]
        assert left["examples"] == right["examples"]
        assert left["events"] == right["events"]
        assert left["original"] == right["original"]
        assert left["used"] == right["used"]
        assert left["truncated_share"] == right["truncated_share"]
        assert left["client_embedding_shape"] == right["client_embedding_shape"]
