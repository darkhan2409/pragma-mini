"""
Unigram-baseline и метрики masked-field prediction.

Все ожидаемые числа посчитаны руками. Метрики агрегируются по
всему набору, а не усредняются по batch: F1 и NCE нелинейны, и
среднее готовых batch-значений это другая величина.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from src.tokenizer.vocab import KeyEntry, ValueEntry, Vocab
from src.model.metrics import (
    STATUS_DEGENERATE,
    STATUS_NO_TARGETS,
    STATUS_OK,
    STATUS_UNDEFINED,
    MetricAccumulator,
    UnigramTable,
    render_metrics,
)
from src.model.mlm_head import FieldLogits, FieldTable


# ============================================================
# СИНТЕТИЧЕСКИЙ СЛОВАРЬ
# ============================================================


def metrics_vocab() -> Vocab:
    """
    Два предсказуемых поля и одно вырожденное.
    """

    keys = [
        KeyEntry(6, "m__two", "m", "two", "categorical", True, "string", 9, 11),
        KeyEntry(7, "m__eight", "m", "eight", "numeric", True, "int64", 11, 19),
        KeyEntry(8, "m__one", "m", "one", "categorical", True, "string", 19, 20),
    ]

    values = [
        ValueEntry(9, 6, "m__two", "no", 1),
        ValueEntry(10, 6, "m__two", "yes", 1),
        *[ValueEntry(11 + index, 7, "m__eight", str(index), 1) for index in range(8)],
        ValueEntry(19, 8, "m__one", "only", 1),
    ]

    return Vocab(keys, values)


def unigram_artifact() -> dict:
    return {
        "fields": {
            "m": {
                "two": {"encoding": "value", "distribution": [["yes", 0.75], ["no", 0.25]]},
                "eight": {
                    "encoding": "bucket",
                    "distribution": [[index, 0.5 if index == 0 else 0.5 / 7] for index in range(8)],
                },
            }
        }
    }


@pytest.fixture()
def table() -> FieldTable:
    return FieldTable(metrics_vocab())


@pytest.fixture()
def unigram(table) -> UnigramTable:
    return UnigramTable(unigram_artifact(), table.vocab, table)


# ============================================================
# UNIGRAM
# ============================================================


def test_value_encoding_maps_strings_to_local_indices(table, unigram):
    # "no" это первый кандидат поля, "yes" второй.
    probabilities = np.exp(unigram.log_probs[6])

    assert probabilities.tolist() == pytest.approx([0.25, 0.75])
    assert unigram.mode[6] == 1


def test_bucket_encoding_uses_the_index_directly(table, unigram):
    probabilities = np.exp(unigram.log_probs[7])

    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] == pytest.approx(0.5 / 7)
    assert unigram.mode[7] == 0


def test_degenerate_field_gets_no_distribution(table, unigram):
    assert not unigram.has(8)


def test_unmatched_values_are_reported(table):
    artifact = {
        "fields": {
            "m": {
                "two": {"encoding": "value", "distribution": [["yes", 0.5], ["green", 0.5]]},
                "eight": {"encoding": "bucket", "distribution": [[0, 1.0]]},
            }
        }
    }

    unigram = UnigramTable(artifact, table.vocab, table)

    assert unigram.unmatched["m__two"] == ["green"]
    assert unigram.coverage["m__two"] == pytest.approx(0.5)

    # Масса несопоставленного значения не достаётся кандидатам.
    probabilities = np.exp(unigram.log_probs[6])

    assert probabilities[1] > 0.99


def test_out_of_range_bucket_is_unmatched(table):
    artifact = {
        "fields": {
            "m": {
                "two": {"encoding": "value", "distribution": [["yes", 1.0]]},
                "eight": {"encoding": "bucket", "distribution": [[0, 0.5], [99, 0.5]]},
            }
        }
    }

    unigram = UnigramTable(artifact, table.vocab, table)

    assert unigram.unmatched["m__eight"] == ["99"]


def test_epsilon_is_a_floor_and_the_result_is_normalised(table):
    artifact = {"fields": {"m": {"two": {"encoding": "value", "distribution": [["yes", 1.0]]}}}}

    unigram = UnigramTable(artifact, table.vocab, table, epsilon=1e-6)

    probabilities = np.exp(unigram.log_probs[6])

    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx(1e-6, rel=1e-3)
    assert np.isfinite(unigram.log_probs[6]).all()


def test_missing_field_is_listed(table):
    artifact = {"fields": {"m": {"two": {"encoding": "value", "distribution": [["yes", 1.0]]}}}}

    unigram = UnigramTable(artifact, table.vocab, table)

    assert unigram.missing == ["m__eight"]
    assert not unigram.has(7)


def test_the_artifact_is_not_modified(table):
    artifact = unigram_artifact()

    original = copy.deepcopy(artifact)

    UnigramTable(artifact, table.vocab, table)

    assert artifact == original


# ============================================================
# МЕТРИКИ НА РУЧНОМ ПРИМЕРЕ
# ============================================================


def two_field_batch() -> tuple[list[FieldLogits], torch.Tensor]:
    """
    Поле 6: softmax [0.8, 0.2] на четырёх позициях, цели 0,0,0,1.
    """

    logits = torch.tensor([[math.log(0.8), math.log(0.2)]]).repeat(4, 1)

    return (
        [FieldLogits(6, torch.arange(4), logits)],
        torch.tensor([0, 0, 0, 1], dtype=torch.long),
    )


def test_model_cross_entropy_matches_the_hand_calculation(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    field_logits, targets = two_field_batch()

    accumulator.update(field_logits, targets)

    report = accumulator.finalize()

    item = next(entry for entry in report["fields"] if entry["key_id"] == 6)

    expected = (3 * -math.log(0.8) + -math.log(0.2)) / 4

    assert item["ce_model"] == pytest.approx(expected, abs=1e-6)
    assert item["n_targets"] == 4
    assert item["n_classes_with_support"] == 2
    assert item["status"] == STATUS_OK


def test_unigram_cross_entropy_and_gain(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    field_logits, targets = two_field_batch()

    accumulator.update(field_logits, targets)

    item = next(entry for entry in accumulator.finalize()["fields"] if entry["key_id"] == 6)

    # p(no) = 0.25, p(yes) = 0.75
    expected = (3 * -math.log(0.25) + -math.log(0.75)) / 4

    assert item["ce_unigram"] == pytest.approx(expected, abs=1e-6)

    assert item["nce_gain"] == pytest.approx(
        (item["ce_unigram"] - item["ce_model"]) / item["ce_unigram"]
    )

    # Модель здесь лучше unigram: gain положителен.
    assert item["nce_gain"] > 0


def test_accuracy_and_unigram_accuracy(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    field_logits, targets = two_field_batch()

    accumulator.update(field_logits, targets)

    item = next(entry for entry in accumulator.finalize()["fields"] if entry["key_id"] == 6)

    # Модель всегда предсказывает 0, цели 0,0,0,1.
    assert item["accuracy"] == pytest.approx(0.75)

    # Мода unigram это "yes" (индекс 1), совпадает один раз.
    assert item["unigram_accuracy"] == pytest.approx(0.25)


def test_macro_f1_counts_the_whole_candidate_set(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    field_logits, targets = two_field_batch()

    accumulator.update(field_logits, targets)

    item = next(entry for entry in accumulator.finalize()["fields"] if entry["key_id"] == 6)

    # Класс 0: P = 3/4, R = 1 → F1 = 6/7. Класс 1: ни одного
    # предсказания → 0, как при zero_division=0.
    assert item["macro_f1"] == pytest.approx((6 / 7 + 0.0) / 2)


def test_top_k_only_when_there_are_more_candidates(table, unigram):
    accumulator = MetricAccumulator(table, unigram, top_k=3)

    logits = torch.zeros(2, 8)
    logits[0, 7] = 5.0
    logits[1, 0] = 5.0

    accumulator.update(
        [FieldLogits(7, torch.arange(2), logits)], torch.tensor([7, 3], dtype=torch.long)
    )

    report = accumulator.finalize()

    wide = next(entry for entry in report["fields"] if entry["key_id"] == 7)

    assert wide["top_k_accuracy"] == pytest.approx(0.5)

    narrow = next(entry for entry in report["fields"] if entry["key_id"] == 6)

    assert narrow["top_k_accuracy"] is None


def test_undefined_gain_when_the_baseline_is_certain(table):
    artifact = {"fields": {"m": {"two": {"encoding": "value", "distribution": [["no", 1.0]]}}}}

    unigram = UnigramTable(artifact, table.vocab, table, epsilon=1e-12)

    accumulator = MetricAccumulator(table, unigram, epsilon=1e-8)

    accumulator.update(
        [FieldLogits(6, torch.arange(2), torch.zeros(2, 2))],
        torch.tensor([0, 0], dtype=torch.long),
    )

    item = next(entry for entry in accumulator.finalize()["fields"] if entry["key_id"] == 6)

    assert item["status"] == STATUS_UNDEFINED
    assert item["nce_gain"] is None
    assert item["ce_unigram"] is not None


def test_fields_without_targets_and_degenerate_fields(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    field_logits, targets = two_field_batch()

    accumulator.update(field_logits, targets)

    report = accumulator.finalize()

    quiet = next(entry for entry in report["fields"] if entry["key_id"] == 7)
    single = next(entry for entry in report["fields"] if entry["key_id"] == 8)

    assert quiet["status"] == STATUS_NO_TARGETS
    assert single["status"] == STATUS_DEGENERATE

    assert quiet["ce_model"] is None
    assert single["ce_model"] is None


# ============================================================
# АГРЕГАЦИЯ
# ============================================================


def test_aggregates_are_computed_over_the_whole_set(table, unigram):
    """
    Два batch подряд обязаны дать то же, что один объединённый:
    метрики копятся, а не усредняются по batch.
    """

    torch.manual_seed(1)

    left = torch.randn(5, 2)
    right = torch.randn(3, 2)

    left_targets = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    right_targets = torch.tensor([1, 0, 0], dtype=torch.long)

    split = MetricAccumulator(table, unigram)
    split.update([FieldLogits(6, torch.arange(5), left)], left_targets)
    split.update([FieldLogits(6, torch.arange(3), right)], right_targets)

    whole = MetricAccumulator(table, unigram)
    whole.update(
        [FieldLogits(6, torch.arange(8), torch.cat([left, right]))],
        torch.cat([left_targets, right_targets]),
    )

    a = next(entry for entry in split.finalize()["fields"] if entry["key_id"] == 6)
    b = next(entry for entry in whole.finalize()["fields"] if entry["key_id"] == 6)

    for name in ("n_targets", "ce_model", "ce_unigram", "accuracy", "macro_f1", "nce_gain"):
        assert a[name] == pytest.approx(b[name]), name


def test_field_balanced_and_token_weighted_aggregates_differ(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    # Узкое поле: одна позиция. Широкое: четыре.
    accumulator.update(
        [FieldLogits(6, torch.tensor([0]), torch.zeros(1, 2))],
        torch.tensor([0], dtype=torch.long),
    )
    accumulator.update(
        [FieldLogits(7, torch.arange(4), torch.zeros(4, 8))],
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )

    report = accumulator.finalize()

    assert report["n_targets"] == 5
    assert report["n_fields_with_targets"] == 2

    assert report["field_balanced_ce"] == pytest.approx((math.log(2) + math.log(8)) / 2)
    assert report["token_weighted_ce"] == pytest.approx((math.log(2) + 4 * math.log(8)) / 5)


def test_mean_gain_skips_fields_without_a_baseline(table):
    artifact = {"fields": {"m": {"two": {"encoding": "value", "distribution": [["yes", 0.75], ["no", 0.25]]}}}}

    unigram = UnigramTable(artifact, table.vocab, table)

    accumulator = MetricAccumulator(table, unigram)

    accumulator.update(
        [FieldLogits(6, torch.tensor([0]), torch.zeros(1, 2))],
        torch.tensor([0], dtype=torch.long),
    )
    accumulator.update(
        [FieldLogits(7, torch.arange(2), torch.zeros(2, 8))],
        torch.tensor([0, 1], dtype=torch.long),
    )

    report = accumulator.finalize()

    wide = next(entry for entry in report["fields"] if entry["key_id"] == 7)

    assert wide["status"] == "no_unigram"
    assert wide["nce_gain"] is None
    assert wide["ce_model"] is not None

    # В среднем gain участвует только поле с baseline.
    narrow = next(entry for entry in report["fields"] if entry["key_id"] == 6)

    assert report["mean_nce_gain"] == pytest.approx(narrow["nce_gain"])


def test_masked_and_degenerate_counts_are_kept(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    accumulator.note_batch(n_masked=10, n_degenerate=3)
    accumulator.note_batch(n_masked=5, n_degenerate=1)

    report = accumulator.finalize()

    assert report["n_masked_positions"] == 15
    assert report["n_degenerate_skipped"] == 4


def test_empty_report_is_still_a_report(table, unigram):
    report = MetricAccumulator(table, unigram).finalize()

    assert report["n_targets"] == 0
    assert report["field_balanced_ce"] is None
    assert report["mean_nce_gain"] is None
    assert len(report["fields"]) == 3


def test_report_renders(table, unigram):
    accumulator = MetricAccumulator(table, unigram)

    field_logits, targets = two_field_batch()

    accumulator.update(field_logits, targets)

    text = render_metrics(accumulator.finalize())

    assert "m__two" in text
    assert "NCE" in text
    assert "field-balanced CE" in text
