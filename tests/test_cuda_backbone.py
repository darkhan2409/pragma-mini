from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from src.mlm.varlen import VarlenLayout

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Настоящая CUDA, без подмен ядра:
#
#   шаг модели    параметры и все тензоры батча на карте, внимание
#                 через flash_attn_varlen_func (bf16, cu_seqlens
#                 int32, causal=False), параметры, градиенты и
#                 состояние AdamW — fp32, шаг двигает каждую часть;
#   обучение      auto на CUDA — строгий flash; с целью [USR] — тоже.
# ============================================================


pytestmark = pytest.mark.cuda

flash_only = pytest.mark.skipif(not world.flash_ready(), reason="нужны CUDA и библиотека flash-attn")

CUDA = torch.device("cuda")

PARTS = ("embedding", "event", "profile", "history", "head")


def tensors_of(data) -> dict[str, torch.Tensor]:
    """
    Все тензоры упакованного батча, включая раскладки varlen.
    """

    found = {}

    for item in fields(data):

        value = getattr(data, item.name)

        if isinstance(value, torch.Tensor):
            found[item.name] = value

        elif isinstance(value, VarlenLayout):

            for part in ("cu_seqlens", "lengths"):
                found[f"{item.name}.{part}"] = getattr(value, part)

            for number, group in enumerate(value.groups):
                found[f"{item.name}.group{number}.cu_seqlens"] = group.cu_seqlens

            for number, bucket in enumerate(value.buckets):
                for part in ("segments", "index", "mask"):
                    found[f"{item.name}.bucket{number}.{part}"] = getattr(bucket, part)

    return found


@flash_only
def test_full_model_step_on_cuda(stage, monkeypatch):
    """
    Модель этапа 12 — load_model — целиком на карте: проход через
    flash-attn в bf16, backward и шаг AdamW в fp32.
    """

    import flash_attn

    from src.mlm.model import load_model, pack
    from src.mlm.varlen import autocast

    settle(stage, train_people=many())

    model = load_model(3, 512, 0.0, CUDA, "flash")
    model.train()

    assert all(value.device.type == "cuda" and value.dtype == torch.float32
               for value in model.parameters())

    data = pack([made.client for made in many()], CUDA)

    found = tensors_of(data)

    assert all(value.device.type == "cuda" for value in found.values()), [
        name for name, value in found.items() if value.device.type != "cuda"
    ]
    assert found["events.group0.cu_seqlens"].dtype == torch.int32

    real = flash_attn.flash_attn_varlen_func
    calls = []

    def spy(query, key, value, cu_q, cu_k, max_q, max_k, dropout_p=0.0, causal=None, **kwargs):
        calls.append((query.dtype, query.device.type, cu_q.dtype, causal, query.dim()))
        return real(query, key, value, cu_q, cu_k, max_q, max_k, dropout_p=dropout_p, causal=causal, **kwargs)

    monkeypatch.setattr(flash_attn, "flash_attn_varlen_func", spy)

    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    with autocast(CUDA):
        out = model(data)

    blocks = len(model.profile.layers) + len(model.event.layers) + len(model.history.layers)

    assert len(calls) == blocks
    assert set(calls) == {(torch.bfloat16, "cuda", torch.int32, False, 3)}
    assert out.logits.dtype == torch.bfloat16
    assert bool(torch.isfinite(out.loss))

    out.loss.backward()

    for name, value in model.named_parameters():
        assert value.grad is not None and value.grad.dtype == torch.float32, name
        assert bool(torch.isfinite(value.grad).all()), name

    optimizer.step()

    for state in optimizer.state.values():
        assert all(item.dtype == torch.float32 for key, item in state.items() if key != "step")

    for part in PARTS:
        moved = [name for name, value in getattr(model, part).named_parameters()
                 if not torch.equal(value.detach(), before[f"{part}.{name}"])]
        assert moved, part


@flash_only
def test_training_on_cuda_turns_auto_into_strict_flash(stage, capsys):

    from src.mlm.train import train

    settle(stage, train_people=many())

    result = train(tiny(token_budget=6, device="auto", attention_backend="auto"),
                   epochs=1, max_steps=None, masking=every_value())

    described = result["model"]

    assert described["device"].startswith("cuda")
    assert described["attention"] == "flash" and described["bf16"]
    assert described["blocks"] == {"profile": world.LAYERS, "event": world.LAYERS, "history": world.LAYERS}

    printed = capsys.readouterr().out

    assert "внимание flash" in printed and described["gpu"] in printed


@flash_only
def test_the_usr_target_trains_on_cuda(stage, monkeypatch):
    """
    Цель [USR] на счётчиках и давности — тем же путём, что обучение:
    flash в bf16, параметры в fp32.
    """

    import src.mlm.model as model_module
    from src.mlm.settings import checkpoint_path
    from src.mlm.train import train

    from tests.test_recent_types import head

    monkeypatch.setattr(model_module, "recent_types", lambda artifacts, dim, seed: head(dim))

    settle(stage, train_people=many())

    result = train(
        tiny(token_budget=6, device="auto", attention_backend="auto", usr_aux_weight=1.0),
        epochs=1, max_steps=None, masking=every_value(),
    )

    assert result["model"]["attention"] == "flash"

    weights = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)["model_state_dict"]

    assert all(bool(torch.isfinite(value).all()) for value in weights.values())
    assert any(name.startswith("recent.") for name in weights)
