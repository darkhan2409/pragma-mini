from __future__ import annotations

import pytest
import torch

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# Архитектура по умолчанию: d = 128, 4 головы, энкодер анкеты —
# 1 блок, события — 5, истории — 2; всего 8 блоков внимания.
# Этапы 10–12 записывают конфиг рядом с весами, и модель обучения
# собирается из него, поэтому проверяются сами конфиги, энкодеры из
# них и настоящий проход через flash-attn: 8 вызовов ядра, каждый
# двунаправленный, в bf16, по плоским тензорам и cu_seqlens int32.
# ============================================================


def configs():

    from src.embedding.settings import EmbeddingConfig
    from src.event.settings import EventConfig
    from src.history.settings import HistoryConfig
    from src.profile.settings import ProfileConfig

    return EmbeddingConfig(), ProfileConfig(), EventConfig(), HistoryConfig()


def build(dim: int, attention: str = "sdpa"):
    """
    Модель из конфигов по умолчанию; словарь — маленький словарь тестов.
    """

    from src.event.encoder import EventEncoder
    from src.history.encoder import HistoryEncoder
    from src.mlm.model import Mlm, Model
    from src.profile.encoder import ProfileEncoder

    _, profile, event, history = configs()

    return Model(
        embedding=world.embedding(world.VOCAB, dim, world.SEED),
        event=EventEncoder(dim, event.layers, event.heads, event.feedforward, 0.0, event.seed),
        profile=ProfileEncoder(dim, profile.layers, profile.heads, profile.feedforward, 0.0,
                               profile.rope_base, profile.seed),
        history=HistoryEncoder(dim, history.layers, history.heads, history.feedforward, 0.0,
                               history.rope_base, history.seed),
        head=Mlm(dim, world.SEED),
        attention=attention,
    )


def test_default_architecture_is_1_5_2_at_d128_with_4_heads():

    embedding, profile, event, history = configs()

    assert embedding.dim == 128
    assert (profile.layers, event.layers, history.layers) == (1, 5, 2)
    assert profile.heads == event.heads == history.heads == 4
    assert embedding.dim % event.heads == 0


def test_encoders_built_from_the_configs_have_1_5_2_blocks():

    model = build(128)

    assert (len(model.profile.layers), len(model.event.layers), len(model.history.layers)) == (1, 5, 2)


@pytest.mark.cuda
@pytest.mark.skipif(not world.flash_ready(), reason="нужны CUDA и библиотека flash-attn")
def test_forward_calls_the_real_kernel_once_per_block(clients, monkeypatch):
    """
    Настоящий flash_attn_varlen_func под шпионом: 1 + 5 + 2 = 8
    вызовов за проход, и backward доходит до всех параметров.
    """

    import flash_attn

    from src.mlm.model import pack
    from src.mlm.varlen import autocast

    real = flash_attn.flash_attn_varlen_func
    calls: list[tuple] = []

    def spy(query, key, value, cu_q, cu_k, max_q, max_k, dropout_p=0.0, causal=None, **kwargs):
        calls.append((query.dtype, query.dim(), int(query.shape[0]) == int(cu_q[-1]), cu_q.dtype, causal))
        return real(query, key, value, cu_q, cu_k, max_q, max_k, dropout_p=dropout_p, causal=causal, **kwargs)

    monkeypatch.setattr(flash_attn, "flash_attn_varlen_func", spy)

    device = torch.device("cuda")

    model = build(128, attention="flash").to(device)

    with autocast(device):
        out = model(pack(clients, device))

    assert len(calls) == 8
    assert set(calls) == {(torch.bfloat16, 3, True, torch.int32, False)}

    out.loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
