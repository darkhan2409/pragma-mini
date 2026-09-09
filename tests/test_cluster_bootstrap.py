"""
Bootstrap по клиентам вместо месячных примеров.

Проверяется два утверждения. Первое: группировка меняет только
разброс, а точечная оценка остаётся прежней до разряда. Второе:
все сравнения идут по ОДНОЙ выборке клиентов, иначе разность
разностей считалась бы на двух независимых розыгрышах.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.model import diagnostics
from src.model.diagnostics import (
    LEVEL_CLIENT,
    LEVEL_EXAMPLE,
    Levels,
    run_cluster_bootstrap,
)
from src.model.metrics import (
    MetricAccumulator,
    bootstrap_combination,
    bootstrap_contrast,
    cluster_draws,
    group_sample,
)
from src.model.mlm_head import FieldLogits

from tests.test_compare import sample, two_runs  # noqa: F401
from tests.test_metrics import metrics_vocab, table, unigram  # noqa: F401
from tests.test_trainer import env, small_config  # noqa: F401


# ============================================================
# ГРУППИРОВКА
# ============================================================


def hand_sample(rows: int = 4, fields: int = 2):
    owners = np.arange(rows, dtype=np.int64)
    total = np.arange(rows * fields, dtype=np.float64).reshape(rows, fields) + 1.0
    counts = np.ones((rows, fields), dtype=np.float64)
    return owners, total, counts


def test_examples_of_one_client_become_one_row():
    owners, total, counts = hand_sample()

    keys, grouped_total, grouped_counts = group_sample(
        (owners, total, counts), np.array([7, 7, 9, 9])
    )

    assert keys.tolist() == [7, 9]

    assert grouped_total.tolist() == [
        (total[0] + total[1]).tolist(),
        (total[2] + total[3]).tolist(),
    ]

    assert grouped_counts.tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_grouping_preserves_column_sums():
    sample_in = hand_sample(rows=6, fields=3)

    grouped = group_sample(sample_in, np.array([1, 1, 1, 2, 2, 3]))

    assert np.allclose(sample_in[1].sum(axis=0), grouped[1].sum(axis=0))
    assert np.allclose(sample_in[2].sum(axis=0), grouped[2].sum(axis=0))


def test_grouping_sorts_clients():
    grouped = group_sample(hand_sample(), np.array([9, 3, 9, 3]))

    assert grouped[0].tolist() == [3, 9]


def test_one_example_per_client_changes_nothing():
    sample_in = hand_sample()

    grouped = group_sample(sample_in, sample_in[0])

    assert np.array_equal(grouped[1], sample_in[1])
    assert np.array_equal(grouped[2], sample_in[2])


def test_cluster_count_must_match_rows():
    with pytest.raises(ValueError, match="кластеров"):
        group_sample(hand_sample(), np.array([1, 2]))


# ============================================================
# КРАТНОСТИ
# ============================================================


def test_draws_are_deterministic_and_sum_to_the_unit_count():
    left = cluster_draws(5, 4, seed=3)
    right = cluster_draws(5, 4, seed=3)

    assert np.array_equal(left, right)
    assert left.shape == (4, 5)
    assert left.sum(axis=1).tolist() == [5.0] * 4

    assert not np.array_equal(left, cluster_draws(5, 4, seed=4))


def test_multiplicity_is_applied_as_repetition():
    """
    Строка с кратностью два обязана вносить столько же, сколько
    та же строка, повторённая дважды.
    """

    owners, total, counts = hand_sample(rows=3, fields=2)

    draws = np.array([[2.0, 1.0, 0.0]])

    result = bootstrap_combination([(1.0, (owners, total, counts))], draws=draws)

    repeated_total = np.stack([total[0], total[0], total[1]])
    repeated_counts = np.stack([counts[0], counts[0], counts[1]])

    direct = bootstrap_combination(
        [(1.0, (np.arange(3), repeated_total, repeated_counts))],
        draws=np.ones((1, 3)),
    )

    assert result["ci_low"] == pytest.approx(direct["ci_low"])
    assert result["ci_high"] == pytest.approx(direct["ci_high"])


def test_draw_matrix_of_wrong_width_is_refused():
    with pytest.raises(ValueError, match="матрица кратностей"):
        bootstrap_combination([(1.0, hand_sample())], draws=np.ones((3, 9)))


# ============================================================
# ТОЧЕЧНАЯ ОЦЕНКА
# ============================================================


def test_point_estimate_survives_grouping():
    owners, total, counts = sample(rows=40, seed=21)

    right = (owners, total + 0.4 * counts, counts)

    clusters = owners // 4

    by_example = bootstrap_contrast((owners, total, counts), right, n_boot=100, seed=5)

    by_client = bootstrap_contrast(
        group_sample((owners, total, counts), clusters),
        group_sample(right, clusters),
        n_boot=100,
        seed=5,
    )

    assert by_client["estimate"] == pytest.approx(by_example["estimate"])
    assert by_client["n_units"] == 10
    assert by_example["n_units"] == 40


def test_clustering_does_not_narrow_the_interval():
    """
    У зависимых примеров клиентский интервал обязан быть не уже
    примерного: в этом и была ошибка прежней оценки.
    """

    rng = np.random.default_rng(7)

    clients = 12
    per_client = 10

    counts = np.ones((clients * per_client, 2))

    total = rng.normal(size=(clients * per_client, 2)) + 3.0

    owners = np.arange(clients * per_client, dtype=np.int64)
    clusters = owners // per_client

    # Разница моделей задаётся клиентом и внутри клиента
    # повторяется: ровно та зависимость, которую bootstrap по
    # примерам не видит.
    level = rng.normal(scale=0.5, size=(clients, 1))

    right = (owners, total + np.repeat(level, per_client, axis=0) * counts, counts)

    by_example = bootstrap_contrast((owners, total, counts), right, n_boot=500, seed=9)

    by_client = bootstrap_contrast(
        group_sample((owners, total, counts), clusters),
        group_sample(right, clusters),
        n_boot=500,
        seed=9,
    )

    assert by_client["estimate"] == pytest.approx(by_example["estimate"])

    width_example = by_example["ci_high"] - by_example["ci_low"]
    width_client = by_client["ci_high"] - by_client["ci_low"]

    # Внутри клиента разница одинакова, поэтому честный интервал
    # опирается на 12 единиц, а не на 120.
    assert width_client > 2.0 * width_example


# ============================================================
# ПУСТЫЕ ПОЛЯ И ВСЕЛЕННАЯ
# ============================================================


def test_a_field_without_targets_is_dropped_not_zeroed():
    owners = np.arange(2, dtype=np.int64)

    total = np.array([[2.0, 0.0], [4.0, 0.0]])
    counts = np.array([[1.0, 0.0], [1.0, 0.0]])

    result = bootstrap_combination([(1.0, (owners, total, counts))], draws=np.ones((1, 2)))

    # Среднее только по непустому полю: (2+4)/2 = 3, а не 1.5.
    assert result["estimate"] == pytest.approx(3.0)


def test_universe_covers_units_without_targets(table, unigram):
    accumulator = MetricAccumulator(table, unigram, keep_units=True)

    accumulator.update(
        [FieldLogits(6, torch.tensor([0]), torch.zeros(1, 2))],
        torch.tensor([0], dtype=torch.long),
        units=torch.tensor([2]),
    )

    owners, total, counts = accumulator.unit_losses(universe=np.arange(5))

    assert owners.tolist() == [0, 1, 2, 3, 4]

    assert counts[2].sum() == 1.0
    assert counts[0].sum() == 0.0
    assert total[0].sum() == 0.0


def test_clients_with_targets_counts_clients_not_examples(table, unigram):
    accumulator = MetricAccumulator(table, unigram, keep_units=True)

    accumulator.update(
        [FieldLogits(6, torch.arange(3), torch.zeros(3, 2))],
        torch.zeros(3, dtype=torch.long),
        units=torch.tensor([0, 1, 2]),
    )

    # Три примера, но два клиента.
    clients = accumulator.clients_with_targets(np.array([10, 10, 11]))

    assert clients[6] == 2


# ============================================================
# ОБЩИЕ РОЗЫГРЫШИ
# ============================================================


class FakeSplit:
    def __init__(self, clusters):
        self.client_of_example = np.asarray(clusters, dtype=np.int64)
        self.n_examples = len(clusters)


def test_levels_build_one_matrix_per_level():
    levels = Levels(FakeSplit([1, 1, 2, 2, 3, 3]), n_boot=8, seed=2)

    assert levels.n_examples == 6
    assert levels.n_clients == 3

    assert levels.draws[LEVEL_EXAMPLE].shape == (8, 6)
    assert levels.draws[LEVEL_CLIENT].shape == (8, 3)


def test_every_comparison_uses_the_same_draws(monkeypatch):
    levels = Levels(FakeSplit([1, 1, 2, 2]), n_boot=8, seed=2)

    seen: list[tuple[str, int]] = []

    original = diagnostics.bootstrap_combination

    def spy(terms, **kwargs):
        seen.append(id(kwargs["draws"]))
        return original(terms, **kwargs)

    monkeypatch.setattr(diagnostics, "bootstrap_combination", spy)

    left = {
        LEVEL_EXAMPLE: hand_sample(),
        LEVEL_CLIENT: group_sample(hand_sample(), np.array([1, 1, 2, 2])),
    }

    levels.combine([(1.0, left)])
    levels.combine([(-1.0, left), (1.0, left)])

    assert len(seen) == 4

    # По две ссылки на каждую матрицу и ровно две матрицы.
    assert len(set(seen)) == 2
    assert seen[0] == seen[2]
    assert seen[1] == seen[3]


def test_levels_report_when_the_verdict_changes():
    levels = Levels(FakeSplit([1, 1, 2, 2]), n_boot=64, seed=2)

    left = {
        LEVEL_EXAMPLE: hand_sample(),
        LEVEL_CLIENT: group_sample(hand_sample(), np.array([1, 1, 2, 2])),
    }

    result = levels.combine([(-1.0, left), (1.0, left)])

    assert result[LEVEL_EXAMPLE]["estimate"] == pytest.approx(0.0)
    assert isinstance(result["significance_changed"], bool)
    assert result["ci_width_ratio"] is None


# ============================================================
# ЦЕЛИКОМ
# ============================================================


@pytest.fixture(scope="module")
def bootstrap_report(env, two_runs, tmp_path_factory):
    out = tmp_path_factory.mktemp("cluster")

    report = run_cluster_bootstrap(
        env,
        two_runs["old"],
        two_runs["new"],
        out,
        device="cpu",
        modes=("token", "event"),
        ablation_modes=("token",),
        n_boot=200,
        quiet=True,
    )

    return {"report": report, "out": out}


def test_both_levels_are_present(bootstrap_report):
    report = bootstrap_report["report"]

    for mode in report["modes"]:
        for name, item in report["results"][mode]["contrast"].items():
            for scope in ("all_fields", "subset"):
                assert LEVEL_EXAMPLE in item[scope]
                assert LEVEL_CLIENT in item[scope]
                assert isinstance(item[scope]["significance_changed"], bool)


def test_point_estimates_agree_between_levels(bootstrap_report):
    for section in bootstrap_report["report"]["results"].values():
        for item in section["contrast"].values():
            for scope in ("all_fields", "subset"):
                assert item[scope][LEVEL_CLIENT]["estimate"] == pytest.approx(
                    item[scope][LEVEL_EXAMPLE]["estimate"]
                )


def test_val_client_has_more_examples_than_clients(bootstrap_report):
    units = bootstrap_report["report"]["units"]

    assert units["val_client"]["n_examples"] > units["val_client"]["n_clients"]
    assert units["val_client"]["examples_per_client"] > 1


def test_val_time_is_one_example_per_client(bootstrap_report):
    """
    Там, где на клиента один пример, клиентский bootstrap обязан
    совпасть с прежним: это контроль воспроизводимости.
    """

    report = bootstrap_report["report"]

    units = report["units"]["val_time"]

    assert units["n_examples"] == units["n_clients"]

    for section in report["results"].values():

        item = section["contrast"]["val_time"]["all_fields"]

        assert item[LEVEL_CLIENT]["ci_low"] == pytest.approx(item[LEVEL_EXAMPLE]["ci_low"])
        assert item[LEVEL_CLIENT]["ci_high"] == pytest.approx(item[LEVEL_EXAMPLE]["ci_high"])
        assert not item["significance_changed"]


def test_ablation_is_reported_only_for_its_modes(bootstrap_report):
    report = bootstrap_report["report"]

    assert report["results"]["token"]["ablation"]["val_time"]
    assert report["results"]["event"]["ablation"] == {"val_client": {}, "val_time": {}}


def test_field_support_counts_clients(bootstrap_report):
    report = bootstrap_report["report"]

    rows = report["results"]["token"]["fields"]["val_time"]

    assert rows

    clients = report["units"]["val_time"]["n_clients"]

    for row in rows:
        assert 0 < row["n_clients_with_targets"] <= clients
        assert row["n_targets"] >= row["n_clients_with_targets"]

        if row["n_clients_with_targets"] < 2:
            assert row["support"] == "low_support"


def test_summary_lists_every_contrast(bootstrap_report):
    report = bootstrap_report["report"]

    summary = report["summary"]

    assert summary["contrasts_total"] == len(report["modes"]) * 2

    assert not set(summary["significance_changed"]) & set(summary["significance_kept"])


def test_artefacts_are_written(bootstrap_report):
    out = bootstrap_report["out"]

    assert (out / "cluster_bootstrap.json").exists()

    text = (out / "cluster_bootstrap.md").read_text(encoding="utf-8")

    assert "Bootstrap по клиентам" in text
    assert "по клиентам" in text
    assert "примеров на клиента" in text

    saved = json.loads((out / "cluster_bootstrap.json").read_text(encoding="utf-8"))

    assert saved["n_boot"] == 200
