from __future__ import annotations

from dataclasses import fields, replace

import pyarrow.parquet as pq
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
#   этапы 10–12   считают на карте, веса — те же, что на CPU (они
#                 разыграны на CPU), векторы совпадают с CPU в
#                 пределах fp32;
#   этап 13       на карте идёт через настоящий flash-attn;
#   шаг модели    параметры и все тензоры батча на карте, внимание
#                 через flash_attn_varlen_func (bf16, cu_seqlens
#                 int32, causal=False), параметры, градиенты и
#                 состояние AdamW — fp32, шаг двигает каждую часть;
#   обучение      auto на CUDA — строгий flash.
# ============================================================


pytestmark = pytest.mark.cuda

flash_only = pytest.mark.skipif(not world.flash_ready(), reason="нужны CUDA и библиотека flash-attn")

CUDA = torch.device("cuda")

PARTS = ("embedding", "event", "profile", "history", "head")


def vectors(path, column: str) -> torch.Tensor:
    return torch.tensor(pq.read_table(path).column(column).to_pylist())


def weights(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


@flash_only
def test_diagnostic_stages_run_on_cuda_and_agree_with_the_cpu(stage, tmp_path):

    from src.event.build import build_group as build_events
    from src.event.settings import EVENTS_FILE, WEIGHTS_FILE, events_dir
    from src.history.build import build_group as build_history
    from src.history.settings import HISTORY_FILE, history_dir
    from src.profile.build import build_group as build_profiles
    from src.profile.settings import PROFILES_FILE, profiles_dir

    settle(stage, train_people=many())

    event, profile, history = world.encoder_configs()

    runs = (
        ("event", build_events, event, events_dir, EVENTS_FILE, "vector"),
        ("profile", build_profiles, profile, profiles_dir, PROFILES_FILE, "profile"),
        ("history", build_history, history, history_dir, HISTORY_FILE, "client"),
    )

    for name, build, config, directory, table, column in runs:

        on_cpu = build("train", config)
        on_cuda = build("train", replace(config, device="cuda"), directory=tmp_path / name)

        assert on_cpu["device"] == "cpu" and on_cuda["device"].startswith("cuda"), name

        cpu_weights = weights(directory("train") / WEIGHTS_FILE)
        cuda_weights = weights(tmp_path / name / WEIGHTS_FILE)

        for key, value in cpu_weights["state_dict"].items():
            assert torch.equal(value, cuda_weights["state_dict"][key]), (name, key)

        left = vectors(directory("train") / table, column)
        right = vectors(tmp_path / name / table, column)

        assert left.shape == right.shape and torch.isfinite(right).all(), name
        assert torch.allclose(left, right, atol=1e-4, rtol=1e-4), (name, float((left - right).abs().max()))


@flash_only
def test_stage_13_on_cuda_goes_through_flash(stage, tmp_path, monkeypatch):

    import flash_attn

    from src.mlm.build import build_group
    from src.mlm.settings import MlmConfig

    settle(stage, train_people=many())

    on_cpu = build_group("val", MlmConfig(device="cpu", attention_backend="sdpa"))

    real = flash_attn.flash_attn_varlen_func
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("causal"))
        return real(*args, **kwargs)

    monkeypatch.setattr(flash_attn, "flash_attn_varlen_func", spy)

    on_cuda = build_group("val", MlmConfig(device="auto"), directory=tmp_path)

    assert on_cuda["device"].startswith("cuda")
    assert calls and set(calls) == {False}
    assert on_cuda["targets"] == on_cpu["targets"]
    assert on_cuda["loss"] == pytest.approx(on_cpu["loss"], rel=2e-2)


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

            for part in ("cu_seqlens", "cu_seqlens_int32", "lengths"):
                found[f"{item.name}.{part}"] = getattr(value, part)

            for number, bucket in enumerate(value.buckets):
                for part in ("segments", "index", "mask"):
                    found[f"{item.name}.bucket{number}.{part}"] = getattr(bucket, part)

    return found


@flash_only
def test_full_model_step_on_cuda(stage, monkeypatch):
    """
    Модель этапа 14 — load_model — целиком на карте: проход через
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
    assert found["events.cu_seqlens_int32"].dtype == torch.int32

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
