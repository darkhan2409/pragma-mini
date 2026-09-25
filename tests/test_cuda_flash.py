from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from src.mlm.model import pack
from src.mlm.varlen import VarlenLayout, attend, autocast

from tests import world
from tests.test_attention_backends import reference_attend
from tests.test_scheduler import many
from tests.test_training_math import every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Здесь всё, что нельзя проверить без железа.
#
# CUDA на этой машине есть, flash-attn нет, поэтому набор делится
# надвое: часть выполняется по-настоящему (autocast, размещение,
# состояние генератора CUDA при продолжении), часть честно
# пропускается и ждёт среды с библиотекой.
#
# Пропуск — не «проверено». В отчёте такие тесты называются
# отдельно.
# ============================================================


pytestmark = pytest.mark.cuda


CPU = torch.device("cpu")


cuda_only = pytest.mark.skipif(not world.cuda_ready(), reason="нужна CUDA с bf16")

flash_only = pytest.mark.skipif(
    not world.flash_ready(), reason="нужны CUDA и библиотека flash-attn"
)


def device() -> torch.device:
    return torch.device("cuda")


# ============================================================
# СМЕШАННАЯ ТОЧНОСТЬ И РАЗМЕЩЕНИЕ
# ============================================================


def test_autocast_is_off_on_cpu():

    assert isinstance(autocast(CPU), type(nullcontext()))


@cuda_only
def test_autocast_on_cuda_is_bfloat16():
    """
    Именно bf16: для него не нужно масштабирование потерь, и веса
    при этом остаются fp32.
    """

    context = autocast(device())

    assert isinstance(context, torch.autocast)
    assert context.fast_dtype == torch.bfloat16


@cuda_only
def test_pack_puts_the_batch_on_the_device(clients):
    """
    Всё, что участвует в графе, переезжает на устройство; разбивка
    по корзинам остаётся на CPU — она нужна индексами, а не
    арифметикой.
    """

    data = pack(clients, device())

    for name in ("key_ids", "value_ids", "positions", "labels", "calendar",
                 "history_positions", "target_token", "target_event"):
        assert getattr(data, name).device.type == "cuda", name

    for layout in (data.events, data.profiles, data.history):
        assert layout.cu_seqlens.device.type == "cuda"
        assert layout.buckets[0].index.device.type == "cuda"

    assert data.events.bucket_of.__class__.__module__ == "numpy"


@cuda_only
def test_the_same_model_gives_the_same_answer_on_both_devices(clients):
    """
    fp32 против fp32: разные ядра складывают в разном порядке, но
    ответ обязан совпасть в пределах float32.
    """

    built = world.model()
    built.eval()

    with torch.no_grad():
        here = built(pack(clients, CPU))

    moved = built.to(device())

    with torch.no_grad():
        there = moved(pack(clients, device()))

    assert torch.allclose(here.logits, there.logits.cpu(), atol=1e-4, rtol=1e-4)
    assert float(here.loss) == pytest.approx(float(there.loss), rel=1e-4)


@cuda_only
def test_bfloat16_pass_stays_finite_and_close(clients):
    """
    Под autocast счёт идёт в bf16: точность ниже, но ответ обязан
    остаться конечным и близким к fp32.
    """

    built = world.model().to(device())
    built.eval()

    data = pack(clients, device())

    with torch.no_grad():

        exact = built(data)

        with autocast(device()):
            rough = built(data)

    assert bool(torch.isfinite(rough.logits).all())
    assert torch.allclose(
        rough.logits.float(), exact.logits, atol=0.2, rtol=0.2
    )


# ============================================================
# ПРОДОЛЖЕНИЕ НА CUDA
# ============================================================


@cuda_only
def test_resume_on_cuda_restores_the_device_generator(stage):
    """
    Состояние генератора CUDA лежит в чекпойнте: без него dropout
    продолжения разыграл бы другие маски.
    """

    from src.mlm.settings import checkpoint_path
    from src.mlm.train import train

    settle(stage, train_people=many(), dropout=0.2)

    config = tiny(token_budget=6, device="cuda", warmup_steps=2)
    masking = every_value()

    train(config, epochs=2, max_steps=None, masking=masking)

    straight = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)

    assert straight["cuda_rng_state"] is not None

    settle(stage, train_people=many(), dropout=0.2)

    train(config, epochs=2, max_steps=6, masking=masking)
    train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    again = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)

    for name, value in straight["model_state_dict"].items():
        assert torch.equal(value, again["model_state_dict"][name]), name


# ============================================================
# FLASHATTENTION
# ============================================================


@flash_only
@pytest.mark.parametrize("lengths", [
    [1], [2], [3], [7], [31], [32], [33],
    [1, 2, 3, 7, 31, 32, 33],
    [1, 64, 2, 63, 3],
])
def test_flash_matches_sdpa_segment_by_segment(lengths: list[int]):
    """
    Настоящий flash_attn_varlen_func против посегментного SDPA.

    Допуск под bf16 разумный, но не настолько широкий, чтобы
    спрятать неправильную раскладку: ошибка в cu_seqlens смешала
    бы соседние сегменты и дала расхождение порядка единицы.
    """

    heads, head_dim = 2, 8

    layout = VarlenLayout.build(
        torch.tensor(lengths).numpy(), device(), "сегменты"
    )

    torch.manual_seed(4)

    query, key, value = (
        torch.randn(sum(lengths), heads, head_dim, device=device(), dtype=torch.bfloat16)
        for _ in range(3)
    )

    mine = attend(query, key, value, layout, 0.0)
    theirs = reference_attend(query, key, value, layout, 0.0)

    assert torch.allclose(mine.float(), theirs.float(), atol=2e-2, rtol=2e-2)


@flash_only
def test_flash_receives_flat_tensors_without_padding(clients, monkeypatch):
    """
    На настоящем flash проверяется то же, что на CPU с эталоном:
    Q/K/V плоские, cu_seqlens верные, прямоугольника нет.
    """

    import src.mlm.varlen as varlen

    seen: list[tuple] = []

    original = varlen.attend

    def spy(query, key, value, layout, dropout):
        seen.append((tuple(query.shape), layout.cu_seqlens.tolist()))
        return original(query, key, value, layout, dropout)

    monkeypatch.setattr("src.mlm.varlen.attend", spy)

    built = world.model(attention="flash").to(device())
    built.eval()

    data = pack(clients, device())

    with torch.no_grad(), autocast(device()):
        built(data)

    assert seen

    for shape, bounds in seen:
        assert len(shape) == 3
        assert shape[0] == bounds[-1]


@flash_only
def test_flash_and_buckets_agree_on_the_whole_pass(clients):
    """
    Весь проход двумя путями: событие, анкета, история и голова.
    """

    built = world.model(attention="flash").to(device())
    built.eval()

    data = pack(clients, device())

    with torch.no_grad():

        buckets = built(data)

        with autocast(device()):
            flash = built(data)

    assert torch.allclose(
        flash.logits.float(), buckets.logits.float(), atol=5e-2, rtol=5e-2
    )


@flash_only
def test_flash_sends_gradients_to_the_same_weights(clients):

    left = world.model(attention="flash").to(device())
    right = world.model(attention="sdpa").to(device())

    left.eval()
    right.eval()

    with autocast(device()):
        left(pack(clients, device())).logits.float().sum().backward()

    right(pack(clients, device())).logits.sum().backward()

    other = dict(right.named_parameters())

    for name, parameter in left.named_parameters():

        assert parameter.grad is not None, name

        scale = float(other[name].grad.abs().max()) + 1e-6

        assert torch.allclose(
            parameter.grad.float(), other[name].grad, atol=scale * 0.1, rtol=0.1
        ), name


@flash_only
def test_flash_survives_a_batch_without_targets(clients):
    """
    Окно без целей на настоящем ядре: ни падения, ни NaN.

    Проверяется то же, что на CPU эталоном, но на flash-attn: у
    ядра свои требования к формам, и пустой набор целей — как раз
    тот случай, где расходятся длины.
    """

    quiet = [client for client in clients if client.n_targets == 0]

    assert quiet, "в наборе должен быть клиент без целей"

    built = world.model(attention="flash").to(device())

    data = pack(quiet, device())

    assert int(data.target_token.numel()) == 0

    with autocast(device()):
        out = built(data)

    assert out.logits.shape == (0, world.VOCAB)
    assert float(out.loss.detach()) == 0.0
    assert bool(torch.isfinite(out.loss))

    out.loss.backward()

    for name, parameter in built.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


@flash_only
def test_flash_step_changes_the_weights(clients):
    """
    Настоящий шаг обучения на ядре: backward, обрезка, шаг AdamW.

    Без GradScaler: bf16 в нём не нуждается, а веса и состояние
    оптимизатора остаются fp32.
    """

    built = world.model(attention="flash", dropout=0.1).to(device())
    built.train()

    before = {name: value.detach().clone()
              for name, value in built.named_parameters()}

    optimizer = torch.optim.AdamW(built.parameters(), lr=1e-3)

    with autocast(device()):
        out = built(pack(clients, device()))

    assert out.logits.dtype is torch.bfloat16
    assert bool(torch.isfinite(out.loss))

    out.loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(built.parameters(), 1.0)

    assert bool(torch.isfinite(norm))

    optimizer.step()

    for name, value in built.named_parameters():
        assert value.dtype is torch.float32, name
        assert not torch.equal(value.detach(), before[name]), name
