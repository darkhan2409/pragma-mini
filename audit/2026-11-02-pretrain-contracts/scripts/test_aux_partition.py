"""
Scratch-проверка аудита: зависит ли градиент шага от разбиения окна
на micro-batch'и, когда включена цель [USR] (usr_aux_weight > 0).
Повторяет test_one_big_batch_and_several_micro_batches_train_the_same,
но с aux и с перехватом ВСЕХ параметров (включая recent.proj).
"""
import pytest
import torch

from src.mlm.train import train
from tests.test_recent_types import head
from tests.test_training_math import every_value, settle, tiny, uneven, windows


def grads(monkeypatch, config, masking):
    taken = []
    original = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        taken.append([p.grad.clone() if p.grad is not None else None for p in parameters])
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    train(config, epochs=1, max_steps=None, masking=masking)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", original)
    assert len(taken) == 1
    return taken[0]


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_partition(stage, monkeypatch, weight):
    import src.mlm.model as model_module

    monkeypatch.setattr(model_module, "recent_types", lambda artifacts, dim, seed: head(dim))

    people = uneven(stage)
    masking = every_value()

    settle(stage, train_people=people)
    whole = tiny(token_budget=10_000, grad_accum_steps=1, usr_aux_weight=weight)
    one = grads(monkeypatch, whole, masking)

    settle(stage, train_people=people)
    split = tiny(token_budget=14, grad_accum_steps=len(windows(whole, masking)) + 5, usr_aux_weight=weight)
    assert len(windows(split, masking)) > 1
    many = grads(monkeypatch, split, masking)

    assert len(one) == len(many)
    rel = []
    for a, b in zip(one, many):
        if a is None:
            continue
        rel.append(float((a - b).abs().max() / (a.abs().max() + 1e-12)))
    print(f"\nweight={weight}: params={len(rel)} max_rel_diff_all={max(rel):.3e} "
          f"recent_head_rel_diff={rel[-2] if weight else float('nan'):.3e} "
          f"n_microbatches_split={len(windows(split, masking))}")
