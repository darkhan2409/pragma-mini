from __future__ import annotations

import math

import pytest
import torch

from src.downstream.lora import (
    TARGETS_BLOCK,
    TARGETS_EVENT,
    ChurnHead,
    adapter_state,
    delta_ratio,
    inject,
    load_adapters,
    trainable,
)
from src.mlm.model import pack

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# Заплатка W + B·A при B = 0 не меняет эталон ни на бит; учатся только
# A, B и голова; состояние заплаток переносится между моделями точно.
# База и LoRA сравниваются в одном режиме и одном grad-состоянии:
# в eval/no_grad TransformerEncoderLayer идёт по fused fast path.
# ============================================================


CPU = torch.device("cpu")
RANK = 2
ALPHA = 4.0


def expected_targets(model) -> list[str]:
    names = []
    for part, targets in (("event", TARGETS_EVENT), ("profile", TARGETS_BLOCK), ("history", TARGETS_BLOCK)):
        for i in range(len(getattr(model, part).layers)):
            names += [f"{part}.layers.{i}.{t}" for t in targets]
    return names


def embed(model, clients) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model.client_embeddings(pack(clients, CPU))


def test_zero_adapters_are_the_reference_bitwise(clients):

    model = world.model()
    before = embed(model, clients)

    inject(model, RANK, ALPHA, seed=1)

    assert torch.equal(embed(model, clients), before)


def test_only_adapters_and_head_learn(clients):

    model = world.model()
    head = ChurnHead(world.DIM, 0.2)
    inject(model, RANK, ALPHA, seed=1)
    generator = torch.Generator().manual_seed(5)
    for name, p in model.named_parameters():
        if name.endswith(".B"):
            p.data = torch.randn(p.shape, generator=generator) * 0.1
    head.linear.weight.data = torch.randn(head.linear.weight.shape, generator=generator)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not n.endswith((".A", ".B"))}
    learning = trainable(model, head)

    optimizer = torch.optim.AdamW(learning, lr=1e-2)
    model.train()
    head(model.client_embeddings(pack(clients, CPU))).sum().backward()
    optimizer.step()

    ids = {id(p) for p in learning}
    for name, p in model.named_parameters():
        if id(p) in ids:
            assert p.grad is not None, name
        else:
            assert p.grad is None, name
            assert torch.equal(p, frozen[name]), name
    assert len(frozen) + len(learning) - len(list(head.parameters())) == len(list(model.parameters()))
    assert all(p.grad is not None for p in head.parameters())


def test_targets_and_count():

    model = world.model()
    head = ChurnHead(world.DIM, 0.2)
    state = model.state_dict()
    shapes = {name: state[name].shape for name in expected_targets(model)}

    names = inject(model, RANK, ALPHA, seed=1)

    assert names == expected_targets(model)
    total = sum(RANK * (i + o) for o, i in shapes.values())
    assert sum(p.numel() for p in trainable(model, head)) == total + world.DIM + 1


def test_adapters_survive_save_and_load(clients):

    model = world.model()
    head = ChurnHead(world.DIM, 0.2)
    inject(model, RANK, ALPHA, seed=1)
    generator = torch.Generator().manual_seed(9)
    for p in trainable(model, head):
        p.data = torch.randn(p.shape, generator=generator) * 0.1
    state = adapter_state(model, head)

    fresh = world.model()
    fresh_head = ChurnHead(world.DIM, 0.5)
    inject(fresh, RANK, ALPHA, seed=2)
    load_adapters(fresh, fresh_head, state)

    first, second = embed(model, clients), embed(fresh, clients)
    assert torch.equal(first, second)
    assert torch.equal(head(first), fresh_head(second))

    names = expected_targets(model)
    alien = dict(state)
    alien["event.layers.0.alien.A"] = torch.zeros(1)
    with pytest.raises(ValueError):
        load_adapters(fresh, fresh_head, alien)
    missing = {k: v for k, v in state.items() if k != f"{names[0]}.A"}
    with pytest.raises(ValueError):
        load_adapters(fresh, fresh_head, missing)
    wrong = dict(state)
    wrong[f"{names[0]}.B"] = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        load_adapters(fresh, fresh_head, wrong)


def test_head_starts_at_the_base_rate():

    head = ChurnHead(world.DIM, 0.087)
    x = torch.randn(7, world.DIM, generator=torch.Generator().manual_seed(3)) * 5

    probability = torch.sigmoid(head(x))

    assert head(x).shape == (7,)
    assert torch.allclose(probability, torch.full((7,), 0.087), atol=1e-6)


def test_inject_twice_is_refused():

    model = world.model()
    inject(model, RANK, ALPHA, seed=1)

    with pytest.raises(ValueError):
        inject(model, RANK, ALPHA, seed=1)


def test_delta_ratio_is_zero_then_grows():

    model = world.model()
    inject(model, RANK, ALPHA, seed=1)
    assert delta_ratio(model) == 0.0

    for name, p in model.named_parameters():
        if name.endswith(".B"):
            p.data.fill_(0.1)

    assert delta_ratio(model) > 0.0
    assert math.isfinite(delta_ratio(model))


# ============================================================
# CUDA
# ============================================================


@pytest.mark.cuda
@pytest.mark.skipif(not world.flash_ready(), reason="нужны CUDA и библиотека flash-attn")
def test_flash_and_sdpa_agree_with_adapters(clients):

    from src.mlm.varlen import autocast

    cuda = torch.device("cuda")
    models = {}
    for attention in ("flash", "sdpa"):
        model = world.model(attention=attention)
        inject(model, RANK, ALPHA, seed=1)
        # Ненулевые A и B: иначе заплатки ничего не добавляют и проверка пуста.
        generator = torch.Generator().manual_seed(5)
        for name, p in model.named_parameters():
            if name.endswith((".A", ".B")):
                p.data = torch.randn(p.shape, generator=generator) * 0.1
        models[attention] = model.to(cuda).eval()

    data = pack(clients, cuda)
    with torch.no_grad():
        with autocast(cuda):
            flash = models["flash"].client_embeddings(data)
        sdpa = models["sdpa"].client_embeddings(data)
        zero = world.model(attention="sdpa").to(cuda).eval().client_embeddings(data)

    # Заплатки действительно меняют результат.
    assert not torch.allclose(sdpa, zero, atol=1e-3)
    assert torch.allclose(flash.float(), sdpa.float(), atol=5e-2, rtol=5e-2)
