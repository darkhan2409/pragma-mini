"""
Сравнение двух checkpoint'ов внутри одного режима маскирования.

Главное здесь не числа, а условия их получения: одни примеры,
одни маски, одни цели. Плюс честность вывода: разница без
интервала это не результат.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tokenizer.config import IncompatibleArtifactsError
from src.model.compare import (
    EVAL_MODES,
    evaluation_masking,
    excluded_fields,
    run_comparison,
)
from src.model.history_encoder import ATTENTION_RULES, RULE_FULL
from src.model.metrics import aggregate_fields, bootstrap_combination, bootstrap_contrast
from src.model.trainer import run_training

from tests.test_combined_masking import combined_config
from tests.test_trainer import env, small_config  # noqa: F401


# ============================================================
# АГРЕГАТЫ ПО ПОДМНОЖЕСТВУ
# ============================================================


def field(name: str, ce: float, targets: int = 10, unigram: float = 2.0) -> dict:
    return {
        "field": name,
        "key_id": abs(hash(name)) % 1000,
        "kind": "categorical",
        "n_candidates": 4,
        "status": "ok",
        "n_targets": targets,
        "n_classes_with_support": 2,
        "ce_model": ce,
        "ce_unigram": unigram,
        "nce_gain": (unigram - ce) / unigram,
        "accuracy": 0.5,
        "unigram_accuracy": 0.4,
        "macro_f1": 0.3,
        "top_k_accuracy": None,
    }


def test_subset_changes_the_aggregate():
    fields = [field("timeline__event_type", 0.1), field("transaction__mcc", 3.0)]

    everything = aggregate_fields(fields)
    subset = aggregate_fields(fields, frozenset({"timeline__event_type"}))

    assert everything["field_balanced_ce"] == pytest.approx(1.55)
    assert subset["field_balanced_ce"] == pytest.approx(3.0)

    assert everything["n_fields_with_targets"] == 2
    assert subset["n_fields_with_targets"] == 1

    assert subset["excluded"] == ["timeline__event_type"]
    assert subset["n_fields_excluded"] == 1


def test_subset_ignores_fields_without_targets():
    fields = [field("a", 1.0), field("b", 2.0, targets=0)]

    assert aggregate_fields(fields)["n_fields_with_targets"] == 1


def test_excluded_fields_cover_event_type_and_profile_snapshot(env):
    excluded = excluded_fields(env.table)

    assert "timeline__event_type" in excluded

    assert all(
        name == "timeline__event_type" or name.startswith("profile_snapshot__") for name in excluded
    )

    assert any(name.startswith("profile_snapshot__") for name in excluded)

    # Поля событий остаются.
    assert "transaction__mcc" not in excluded


# ============================================================
# НАБОРЫ МАСОК
# ============================================================


def test_every_evaluation_mode_has_its_own_settings():
    stored = {"balanced_share": 0.15}

    made = {mode: evaluation_masking(mode, stored, 7) for mode in EVAL_MODES}

    assert made["field_balanced"].mode == "field_balanced"
    assert made["field_balanced"].balanced_share == 0.15

    assert made["token"].mode == "token"
    assert made["token"].token_rate == 0.15

    assert made["event"].mode == "event"
    assert made["event"].event_rate == 0.10

    # key это combined с одной включённой стратегией: нужен
    # независимый розыгрыш каждого ключа, а не ровно один ключ.
    assert made["key"].mode == "combined"
    assert made["key"].key_rate == 0.10
    assert made["key"].token_rate == 0.0
    assert made["key"].event_rate == 0.0

    assert {config.seed for config in made.values()} == {7}


def test_unknown_evaluation_mode_is_refused():
    with pytest.raises(ValueError, match="неизвестный режим оценки"):
        evaluation_masking("everything", {}, 7)


# ============================================================
# BOOTSTRAP
# ============================================================


def sample(rows: int = 40, fields_count: int = 3, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)

    counts = rng.integers(1, 5, size=(rows, fields_count)).astype(np.float64)

    total = counts * (1.0 + rng.random((rows, fields_count))) + shift * counts

    return np.arange(rows, dtype=np.int64), total, counts


def test_identical_sides_give_zero():
    left = sample(seed=1)

    result = bootstrap_contrast(left, left, n_boot=200, seed=3)

    assert result["estimate"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert not result["significant"]


def test_a_constant_shift_is_recovered():
    owners, total, counts = sample(seed=2)

    right = (owners, total + 0.5 * counts, counts)

    result = bootstrap_contrast((owners, total, counts), right, n_boot=300, seed=4)

    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)
    assert result["significant"]


def test_noise_alone_is_not_significant():
    left = sample(seed=5)
    right = sample(seed=6)

    result = bootstrap_contrast(left, right, n_boot=500, seed=7)

    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert result["n_units"] == 40


def test_different_units_are_refused():
    left = sample(rows=10, seed=8)
    right = sample(rows=12, seed=9)

    with pytest.raises(ValueError, match="одних и тех же примеров"):
        bootstrap_contrast(left, right, n_boot=10)


def test_different_field_sets_are_refused():
    owners, total, counts = sample(rows=10, fields_count=3, seed=10)

    other = (owners, total[:, :2], counts[:, :2])

    with pytest.raises(ValueError, match="наборы полей"):
        bootstrap_contrast((owners, total, counts), other, n_boot=10)


def test_empty_input_is_refused():
    empty = (np.zeros(0, dtype=np.int64), np.zeros((0, 2)), np.zeros((0, 2)))

    with pytest.raises(ValueError, match="нечего пересэмплировать"):
        bootstrap_contrast(empty, empty, n_boot=10)


def test_difference_of_differences_cancels_a_shared_shift():
    """
    Разность разностей на общей выборке: сдвиг, одинаковый у
    обеих моделей, обязан сократиться.
    """

    owners, total, counts = sample(seed=11)

    shifted = (owners, total + 0.3 * counts, counts)

    result = bootstrap_combination(
        [(-1.0, (owners, total, counts)), (1.0, shifted), (1.0, (owners, total, counts)), (-1.0, shifted)],
        n_boot=200,
        seed=12,
    )

    assert result["estimate"] == pytest.approx(0.0)
    assert not result["significant"]


def test_difference_of_differences_finds_a_real_gap():
    owners, total, counts = sample(seed=13)

    small = (owners, total + 0.1 * counts, counts)
    large = (owners, total + 0.4 * counts, counts)

    result = bootstrap_combination(
        [(-1.0, (owners, total, counts)), (1.0, large), (1.0, (owners, total, counts)), (-1.0, small)],
        n_boot=200,
        seed=14,
    )

    assert result["estimate"] == pytest.approx(0.3)
    assert result["significant"]


def test_combination_needs_terms():
    with pytest.raises(ValueError, match="нечего комбинировать"):
        bootstrap_combination([], n_boot=10)


# ============================================================
# ЦЕЛИКОМ
# ============================================================


@pytest.fixture(scope="module")
def two_runs(env, tmp_path_factory):
    """
    Два коротких run одинаковой конфигурации и разного objective.
    """

    old_dir = tmp_path_factory.mktemp("old")
    new_dir = tmp_path_factory.mktemp("new")

    old = run_training(env, small_config(), old_dir, device="cpu", quiet=True)
    new = run_training(env, combined_config(), new_dir, device="cpu", quiet=True)

    return {"old": old_dir / "best.pt", "new": new_dir / "best.pt", "reports": (old, new)}


@pytest.fixture(scope="module")
def comparison(env, two_runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("comparison")

    report = run_comparison(
        env,
        two_runs["old"],
        two_runs["new"],
        out,
        device="cpu",
        n_boot=200,
        quiet=True,
    )

    return {"report": report, "out": out}


def test_all_evaluation_modes_are_covered(comparison):
    report = comparison["report"]

    assert report["modes"] == list(EVAL_MODES)
    assert set(report["comparisons"]) == set(EVAL_MODES)

    for mode in EVAL_MODES:
        assert set(report["comparisons"][mode]["splits"]) == {"val_client", "val_time"}


def test_both_checkpoints_see_the_same_targets(comparison):
    for section in comparison["report"]["comparisons"].values():
        for name in section["splits"]:
            assert section["old"][name]["n_targets"] == section["new"][name]["n_targets"]
            assert section["old"][name]["n_targets"] > 0


def test_masks_differ_between_modes(comparison):
    digests = {
        mode: section["splits"]["val_time"]["targets_sha256"]
        for mode, section in comparison["report"]["comparisons"].items()
    }

    assert len(set(digests.values())) == len(digests)


def test_field_balanced_masks_match_the_old_checkpoint(comparison, two_runs):
    """
    Старый run обязан оцениваться на своём же наборе масок.
    """

    payload = torch.load(two_runs["old"], map_location="cpu", weights_only=False)

    for name, description in comparison["report"]["comparisons"]["field_balanced"]["splits"].items():
        assert description["targets_sha256"] == payload["splits"][name]["targets_sha256"]


def test_both_aggregates_are_reported(comparison):
    report = comparison["report"]

    assert report["excluded_fields"]

    for section in report["comparisons"].values():
        for name in section["splits"]:
            for label in ("old", "new"):
                subset = section[label][name]["subset"]
                assert subset is not None
                assert subset["n_fields_excluded"] > 0
                assert subset["n_fields_with_targets"] < section[label][name]["n_fields_with_targets"]


def test_contrast_carries_an_interval(comparison):
    for section in comparison["report"]["comparisons"].values():
        for item in section["contrast"].values():
            for scope in ("all_fields", "subset"):
                entry = item[scope]
                assert entry["ci_low"] <= entry["estimate"] <= entry["ci_high"]
                assert entry["n_boot"] == 200
                assert isinstance(entry["significant"], bool)


def test_low_support_fields_are_marked(comparison):
    rows = comparison["report"]["comparisons"]["token"]["fields"]["val_time"]

    assert rows

    for row in rows:
        expected = "low_support" if row["n_targets"] < comparison["report"]["min_targets"] else "ok"
        assert row["support"] == expected


def test_configs_differ_only_in_masking(comparison):
    differences = comparison["report"]["config_differences"]

    assert set(differences) <= {"masking_mode", "token_rate", "event_rate", "key_rate", "balanced_share"}

    assert differences["masking_mode"] == {"old": "field_balanced", "new": "combined"}


def test_ablation_is_repeated_for_both_checkpoints(comparison):
    report = comparison["report"]

    assert set(report["ablations"]) == set(report["ablation_modes"])

    for section in report["ablations"].values():

        for label in ("old", "new"):
            assert set(section[label]["aggregates"]) == {"val_client", "val_time"}
            rules = [row["rule"] for row in section[label]["aggregates"]["val_time"]]
            assert rules == list(ATTENTION_RULES)

        for name, rules in section["contrast"].items():
            assert RULE_FULL not in rules
            for rule, entry in rules.items():
                for scope in ("all_fields", "subset"):
                    assert entry[scope]["old"]["ci_low"] <= entry[scope]["old"]["ci_high"]

                    # Разность разностей это тоже интервал, а не
                    # вычитание двух точечных оценок.
                    difference = entry[scope]["difference"]

                    assert difference["ci_low"] <= difference["estimate"] <= difference["ci_high"]

                    assert difference["estimate"] == pytest.approx(
                        entry[scope]["new"]["estimate"] - entry[scope]["old"]["estimate"]
                    )


def test_answers_are_explicit(comparison):
    answers = comparison["report"]["answers"]

    assert answers

    for item in answers.values():
        assert set(item) == {"history_exchange", "direct_event_to_event", "without_leaky_fields"}
        for entry in item.values():
            assert entry["question"]
            assert entry["answer"]
            assert entry["difference"]["ci_low"] <= entry["difference"]["ci_high"]


def test_growth_verdict_follows_the_difference_not_the_level(comparison):
    """
    «Зависимость выросла» решается интервалом разности, а не
    тем, значима ли потеря у новой модели сама по себе.
    """

    report = comparison["report"]

    for key, answers in report["answers"].items():

        item = answers["history_exchange"]

        difference = item["difference"]

        if not difference["significant"]:
            assert "не подтверждено" in item["answer"], key
        elif difference["estimate"] > 0:
            assert item["answer"] == "да, выросла", key
        else:
            assert item["answer"] == "нет, зависимость снизилась", key


def test_artefacts_are_written(comparison):
    out = comparison["out"]

    for name in ("comparison.json", "comparison.md", "masks.json", "ablation_comparison.md"):
        assert (out / name).exists(), name

    text = (out / "comparison.md").read_text(encoding="utf-8")

    assert "Режим оценки `token`" in text
    assert "bootstrap" in text

    masks = json.loads((out / "masks.json").read_text(encoding="utf-8"))

    assert set(masks["modes"]) == set(EVAL_MODES)

    for splits in masks["modes"].values():
        for description in splits.values():
            assert description["targets_sha256"]

    ablation = (out / "ablation_comparison.md").read_text(encoding="utf-8")

    assert "self_only" in ablation
    assert "Ответы" in ablation


def test_incomparable_checkpoints_are_refused(env, two_runs, tmp_path):
    """
    Другие настройки оценки означают другие маски: сравнение
    было бы бессмысленным и обязано падать.
    """

    payload = torch.load(two_runs["new"], map_location="cpu", weights_only=False)

    payload["train_config"] = {**payload["train_config"], "max_val_clients": 1}

    path = tmp_path / "other.pt"

    torch.save(payload, path)

    with pytest.raises(IncompatibleArtifactsError, match="max_val_clients"):
        run_comparison(env, two_runs["old"], path, tmp_path / "out", device="cpu", n_boot=10, quiet=True)


def test_same_checkpoint_on_both_sides_shows_no_difference(env, two_runs, tmp_path):
    report = run_comparison(
        env,
        two_runs["old"],
        two_runs["old"],
        tmp_path / "self",
        device="cpu",
        modes=("token",),
        ablation_modes=(),
        n_boot=100,
        quiet=True,
    )

    for item in report["comparisons"]["token"]["contrast"].values():
        assert item["all_fields"]["estimate"] == pytest.approx(0.0)
        assert not item["all_fields"]["significant"]

    assert report["config_differences"] == {}
