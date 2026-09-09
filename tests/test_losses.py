"""
Два усреднения MLM-loss на ручном примере.

Числа здесь считаются руками, а не сравниваются с другой
реализацией того же: иначе тест проверял бы только то, что код
согласен сам с собой.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.losses import mlm_loss
from src.model.mlm_head import FieldLogits, TargetError


LN2 = math.log(2.0)
LN3 = math.log(3.0)


# ============================================================
# РУЧНОЙ ПРИМЕР
# ============================================================


def hand_made() -> tuple[list[FieldLogits], torch.Tensor]:
    """
    Поле 10: одна позиция и два равновероятных кандидата → CE = ln 2.
    Поле 20: три позиции и три равновероятных кандидата → CE = ln 3.
    """

    narrow = FieldLogits(
        key_id=10,
        index=torch.tensor([0]),
        logits=torch.zeros(1, 2),
    )

    wide = FieldLogits(
        key_id=20,
        index=torch.tensor([1, 2, 3]),
        logits=torch.zeros(3, 3),
    )

    targets = torch.tensor([0, 0, 1, 2], dtype=torch.long)

    return [narrow, wide], targets


def test_field_balanced_averages_over_fields():
    field_logits, targets = hand_made()

    result = mlm_loss(field_logits, targets)

    assert float(result.field_balanced) == pytest.approx((LN2 + LN3) / 2)


def test_token_weighted_averages_over_positions():
    field_logits, targets = hand_made()

    result = mlm_loss(field_logits, targets)

    assert float(result.token_weighted) == pytest.approx((LN2 + 3 * LN3) / 4)


def test_the_two_averages_differ_when_fields_are_unbalanced():
    field_logits, targets = hand_made()

    result = mlm_loss(field_logits, targets)

    assert float(result.field_balanced) < float(result.token_weighted)


def test_per_field_keeps_counts():
    field_logits, targets = hand_made()

    result = mlm_loss(field_logits, targets)

    assert set(result.per_field) == {10, 20}

    assert result.per_field[10][1] == 1
    assert result.per_field[20][1] == 3

    assert float(result.per_field[10][0]) == pytest.approx(LN2)
    assert float(result.per_field[20][0]) == pytest.approx(LN3)

    assert result.n_targets == 4
    assert result.n_fields == 2


def test_token_weighted_equals_cross_entropy_over_all_positions():
    """
    Взвешивание по числу позиций это ровно средняя CE по всем целям.
    """

    torch.manual_seed(0)

    narrow = FieldLogits(10, torch.tensor([0, 1]), torch.randn(2, 4))
    wide = FieldLogits(20, torch.tensor([2, 3, 4]), torch.randn(3, 4))

    targets = torch.tensor([0, 3, 1, 2, 2], dtype=torch.long)

    result = mlm_loss([narrow, wide], targets)

    every = torch.cat([narrow.logits, wide.logits], dim=0)

    direct = torch.nn.functional.cross_entropy(every, targets[torch.tensor([0, 1, 2, 3, 4])])

    assert float(result.token_weighted) == pytest.approx(float(direct), abs=1e-6)


# ============================================================
# ПУСТОЙ СЛУЧАЙ
# ============================================================


def test_empty_batch_reports_a_reason_and_no_tensors():
    result = mlm_loss([], torch.zeros(0, dtype=torch.long))

    assert result.empty
    assert result.field_balanced is None
    assert result.token_weighted is None
    assert result.n_targets == 0
    assert result.reason


def test_fields_without_positions_are_also_empty():
    empty = FieldLogits(10, torch.zeros(0, dtype=torch.long), torch.zeros(0, 2))

    result = mlm_loss([empty], torch.zeros(0, dtype=torch.long))

    assert result.empty
    assert "без целей" in result.reason


def test_empty_result_serialises():
    result = mlm_loss([], torch.zeros(0, dtype=torch.long))

    assert result.as_dict()["field_balanced"] is None
    assert result.as_dict()["reason"]


# ============================================================
# ОШИБКИ
# ============================================================


def test_targets_must_be_long():
    field_logits, targets = hand_made()

    with pytest.raises(TargetError, match="torch.long"):
        mlm_loss(field_logits, targets.to(torch.int32))


def test_target_outside_the_candidates_is_rejected():
    field_logits, targets = hand_made()

    targets = targets.clone()
    targets[0] = 5

    with pytest.raises(TargetError, match="вне"):
        mlm_loss(field_logits, targets)


# ============================================================
# ТОЧНОСТЬ
# ============================================================


def test_cross_entropy_is_computed_in_float32():
    """
    Logits приходят в bf16 из autocast; логарифм софтмакса на
    bf16 терял бы разряды именно на редких значениях.
    """

    field_logits, targets = hand_made()

    half = [
        FieldLogits(item.key_id, item.index, item.logits.to(torch.bfloat16))
        for item in field_logits
    ]

    result = mlm_loss(half, targets)

    assert result.field_balanced.dtype == torch.float32
    assert float(result.field_balanced) == pytest.approx((LN2 + LN3) / 2, abs=1e-5)


def test_loss_carries_gradients():
    logits = torch.zeros(2, 3, requires_grad=True)

    result = mlm_loss(
        [FieldLogits(10, torch.tensor([0, 1]), logits)],
        torch.tensor([0, 1], dtype=torch.long),
    )

    result.field_balanced.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
