from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .mlm_head import FieldLogits, TargetError


# ============================================================
# ИДЕЯ
# ============================================================
#
# Два усреднения отвечают на разные вопросы.
#
#   field_balanced   среднее CE по ПОЛЯМ
#   token_weighted   среднее CE по ПОЗИЦИЯМ
#
# Основной это field_balanced: позиций у transaction__mcc на
# три порядка больше, чем у product_event__term, и взвешивание
# по токенам превратило бы обучение в обучение одному полю.
# Token-weighted считается рядом, чтобы разрыв между ними было
# видно, а не чтобы им оптимизировать.
#
# CE всегда в float32: под autocast логарифм софтмакса на bf16
# теряет значащие разряды именно там, где вероятность мала, то
# есть на самых информативных позициях.
# ============================================================


@dataclass(frozen=True)
class LossResult:
    """
    Пустой результат это не ноль, а причина.
    """

    field_balanced: torch.Tensor | None
    token_weighted: torch.Tensor | None

    per_field: dict[int, tuple[torch.Tensor, int]] = field(default_factory=dict)

    n_targets: int = 0
    n_fields: int = 0

    reason: str | None = None

    @property
    def empty(self) -> bool:
        return self.field_balanced is None

    def as_dict(self) -> dict:
        return {
            "field_balanced": None if self.empty else float(self.field_balanced.detach()),
            "token_weighted": None if self.empty else float(self.token_weighted.detach()),
            "n_targets": self.n_targets,
            "n_fields": self.n_fields,
            "reason": self.reason,
            "per_field": {
                int(key): {"ce": float(value.detach()), "n": int(count)}
                for key, (value, count) in sorted(self.per_field.items())
            },
        }


def mlm_loss(field_logits: list[FieldLogits], local_targets: torch.Tensor) -> LossResult:
    """
    Cross-entropy по кандидатам каждого поля.
    """

    if local_targets.dtype != torch.long:
        raise TargetError(f"цели обязаны быть torch.long, получено {local_targets.dtype}")

    if not field_logits:
        return LossResult(
            field_balanced=None,
            token_weighted=None,
            reason="в batch нет ни одной замаскированной позиции предсказуемого поля",
        )

    per_field: dict[int, tuple[torch.Tensor, int]] = {}

    losses: list[torch.Tensor] = []
    weights: list[int] = []

    for item in field_logits:

        targets = local_targets[item.index]

        if targets.numel() == 0:
            continue

        if int(targets.max()) >= item.n_candidates or int(targets.min()) < 0:
            raise TargetError(
                f"цель вне {item.n_candidates} кандидатов поля {item.key_id}: "
                f"диапазон {int(targets.min())}..{int(targets.max())}"
            )

        value = F.cross_entropy(item.logits.float(), targets)

        per_field[item.key_id] = (value, int(targets.numel()))

        losses.append(value)
        weights.append(int(targets.numel()))

    if not losses:
        return LossResult(
            field_balanced=None,
            token_weighted=None,
            reason="все поля batch остались без целей",
        )

    stacked = torch.stack(losses)

    counts = torch.tensor(weights, dtype=torch.float32, device=stacked.device)

    return LossResult(
        field_balanced=stacked.mean(),
        token_weighted=(stacked * counts).sum() / counts.sum(),
        per_field=per_field,
        n_targets=int(sum(weights)),
        n_fields=len(losses),
    )
