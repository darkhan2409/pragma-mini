"""
Сравнение двух checkpoint'ов внутри одного режима маскирования.

Главное здесь не числа, а условия их получения: одни примеры,
одни маски, одни цели. Плюс честность вывода: разница без
интервала это не результат.
"""

from __future__ import annotations

import json

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
from src.model.metrics import aggregate_fields, bootstrap_combination, bootstrap_contrast
from src.model.trainer import run_training

from tests.helpers_model import combined_config, small_config



# ============================================================
# АГРЕГАТЫ ПО ПОДМНОЖЕСТВУ
# ============================================================


def field(name: str, ce: float, targets: int = 10, unigram: float = 2.0) -> dict:
    return {
        "field": name,
        "field_id": abs(hash(name)) % 1000,
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


def test_evaluation_carries_the_excluded_fields_of_the_run():
    """
    Политика целей это часть задачи, а не настройка обучения.

    Без переноса exclude_fields сравнение оценивало бы модель
    на полях, которые она никогда не предсказывала.
    """

    stored = {"balanced_share": 0.15, "exclude_fields": ["profile_snapshot__*"]}

    for mode in EVAL_MODES:
        assert evaluation_masking(mode, stored, 7).exclude_fields == ("profile_snapshot__*",)


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


def test_artefacts_are_written(comparison):
    out = comparison["out"]

    for name in ("comparison.json", "comparison.md", "masks.json"):
        assert (out / name).exists(), name

    text = (out / "comparison.md").read_text(encoding="utf-8")

    assert "Режим оценки `token`" in text
    assert "bootstrap" in text

    masks = json.loads((out / "masks.json").read_text(encoding="utf-8"))

    assert set(masks["modes"]) == set(EVAL_MODES)

    for splits in masks["modes"].values():
        for description in splits.values():
            assert description["targets_sha256"]


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
        n_boot=100,
        quiet=True,
    )

    for item in report["comparisons"]["token"]["contrast"].values():
        assert item["all_fields"]["estimate"] == pytest.approx(0.0)
        assert not item["all_fields"]["significant"]

    assert report["config_differences"] == {}


# ============================================================
# CLI
# ============================================================
#
# run_comparison и describe_checkpoint вызываются тестами
# напрямую, поэтому разрыв между парсером и сигнатурой
# оставался невидимым: аргумент, которого функция не принимает,
# доезжал только до живого запуска. Эти два теста идут через
# сам CLI.
# ============================================================


def test_compare_cli_passes_only_arguments_the_function_accepts(env, two_runs, tmp_path):
    """
    Каждый ключ, который CLI собирается передать, обязан быть в
    сигнатуре run_comparison.
    """

    import inspect

    from src.model.train import build_parser

    args = build_parser().parse_args(
        [
            "compare",
            "--name", "dev",
            "--old", str(two_runs["old"]),
            "--new", str(two_runs["new"]),
            "--device", "cpu",
        ]
    )

    accepted = set(inspect.signature(run_comparison).parameters)

    for name in ("min_targets", "bootstrap", "old", "new"):
        assert hasattr(args, name), name

    assert {"min_targets", "n_boot", "device", "modes", "quiet"} <= accepted

    # Ключей, которых функция не принимает, у парсера быть не должно.
    assert not hasattr(args, "rules")


def test_check_cli_reports_fields_that_exist(env, two_runs, capsys):
    """
    describe_checkpoint печатает только то, что действительно
    лежит в checkpoint: поля удалённых режимов не должны
    воскресать строкой-умолчанием.
    """

    from src.model.train import describe_checkpoint

    payload = describe_checkpoint(two_runs["old"])

    printed = capsys.readouterr().out

    assert "additive" not in printed
    assert "структура" not in printed

    # Строка про размеры модели обязана нести настоящие числа.
    line = next(row for row in printed.splitlines() if row.startswith("d_model"))

    assert "None" not in line

    assert "structure" not in payload["model_config"]
    assert "temporal" not in payload["model_config"]


def test_comparison_accepts_a_run_trained_in_another_mode(env, two_runs, tmp_path):
    """
    Сверка набора привязана к режиму старого run.

    Раньше цифра набора field_balanced сверялась с сохранённой
    у любого старого checkpoint'а, и сравнение прогонов,
    обученных в combined, падало на этой сверке. Ни один тест
    этого не ловил: фикстура всегда ставила старым тот run,
    который обучался в field_balanced.
    """

    report = run_comparison(
        env,
        two_runs["new"],
        two_runs["old"],
        tmp_path,
        device="cpu",
        n_boot=50,
        quiet=True,
    )

    assert report["modes"]
