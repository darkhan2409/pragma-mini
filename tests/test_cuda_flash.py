from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from src.mlm import varlen
from src.mlm.model import Mlm, mlm_loss, pack
from src.mlm.varlen import FLASH_PAD, VarlenLayout, attend, autocast

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


# ============================================================
# ПАМЯТЬ: ГРУППЫ ВЫЗОВОВ FLASH И ПОТЕРИ КУСКАМИ
# ============================================================


def event_lengths(count: int, total: int) -> np.ndarray:
    """
    count событий на total токенов, длины почти поровну.
    """

    lengths = np.full(count, total // count, dtype=np.int64)
    lengths[: total % count] += 1

    return lengths


@flash_only
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_groups_give_the_same_attention_bit_for_bit(monkeypatch, dropout: float):
    """
    Настоящий flash_attn_varlen_func: группы против одного вызова
    на все сегменты. Выход, градиент и состояние генератора после
    прохода совпадают бит в бит — в том числе с dropout: смещение
    Philox у группы то же, что у её части одного вызова.
    """

    from flash_attn import flash_attn_varlen_func

    monkeypatch.setattr(varlen, "FLASH_ROWS", 1 << 14)

    lengths = np.random.default_rng(3).integers(1, 38, size=3000)

    layout = VarlenLayout.build(lengths, device(), "события")

    assert len(layout.groups) > 1

    total = int(lengths.sum())

    torch.manual_seed(5)

    qkv = torch.randn(total, 3, 4, 32, device=device(), dtype=torch.bfloat16)
    grad = torch.randn(total, 4, 32, device=device(), dtype=torch.bfloat16)

    bounds = torch.as_tensor(
        np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32), device=device()
    )
    longest = int(lengths.max())

    def one(query, key, value):
        return flash_attn_varlen_func(
            query, key, value, bounds, bounds, longest, longest,
            dropout_p=dropout, causal=False,
        )

    def grouped(query, key, value):
        return attend(query, key, value, layout, dropout)

    def run(call):

        leaf = qkv.clone().requires_grad_(True)

        torch.cuda.manual_seed(9)

        out = call(leaf[:, 0], leaf[:, 1], leaf[:, 2])
        out.backward(grad)

        return out.detach(), leaf.grad, torch.cuda.get_rng_state()

    for left, right in zip(run(one), run(grouped)):
        assert torch.equal(left, right)


def backward_extra(layout: VarlenLayout, total: int) -> int:
    """
    Сколько памяти сверх уже живой занимает backward внимания.
    """

    query, key, value = (
        torch.randn(total, 4, 32, device=device(), dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )

    out = attend(query, key, value, layout, 0.0)
    grad = torch.ones_like(out)

    torch.cuda.synchronize()

    base = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()

    out.backward(grad)

    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated() - base


@flash_only
def test_backward_of_the_largest_client_needs_no_huge_buffer(monkeypatch):
    """
    Клиент на пределе — 12 000 событий, 58 000 токенов. Одним
    вызовом backward заводит dq_accum на 58 000 + 128·12 000 строк
    (около 0,8 ГБ, почти весь — запас по 128 строк на событие).
    Группами он не больше FLASH_ROWS строк: сверху остаются только
    градиенты Q/K/V и их сборка.
    """

    lengths = event_lengths(12000, 58000)

    limit = varlen.FLASH_ROWS

    grouped = backward_extra(VarlenLayout.build(lengths, device(), "события"), 58000)

    monkeypatch.setattr(varlen, "FLASH_ROWS", 1 << 30)

    whole = backward_extra(VarlenLayout.build(lengths, device(), "события"), 58000)

    # Контроль: одним вызовом буфер действительно огромный.
    assert whole > 700 * 2**20

    # Градиенты Q/K/V групп и их сборка — по 256 байт на токен
    # каждый из шести; dq_accum и softmax_d — 512 + 16 байт на
    # строку предела.
    assert grouped <= 6 * 256 * 58000 + (512 + 16) * limit + 16 * 2**20


def loss_extra(count: int) -> int:
    """
    Сколько памяти сверх уже живой занимают потери и их backward
    на count целях при словаре и d этапа 12.
    """

    generator = torch.Generator(device=device()).manual_seed(4)

    head = Mlm(128, 1).to(device())

    weight = torch.randn(5721, 128, device=device(), generator=generator, requires_grad=True)

    parts = [
        torch.randn(count, 128, device=device(), generator=generator, requires_grad=True)
        for _ in range(3)
    ]

    targets = torch.randint(0, 5721, (count,), device=device(), generator=generator)

    torch.cuda.synchronize()

    base = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()

    with autocast(device()):
        loss = mlm_loss(head, *parts, weight, targets, 0.1)

    loss.backward()

    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated() - base


@cuda_only
def test_loss_memory_does_not_grow_with_the_vocabulary_times_targets():
    """
    Целиком кросс-энтропия под bf16 держала бы около 12 байт на
    (цель, слово): на 30 000 целях — больше 2 ГБ. Кусками сверху
    остаётся один кусок [2048, словарь] и линейные по целям
    градиенты входов. Разница между 10 000 и 30 000 целей — только
    эти линейные члены.
    """

    from src.mlm.model import TARGETS_PER_CHUNK

    small = loss_extra(10000)
    large = loss_extra(30000)

    per_target = 8 * 512

    assert large <= 16 * 5721 * TARGETS_PER_CHUNK + per_target * 30000 + 8 * 2**20
    assert large - small <= per_target * 20000 + 16 * 2**20


@flash_only
def test_flash_model_is_the_same_with_and_without_groups(clients, monkeypatch):
    """
    Весь проход модели в обучении, с dropout: малый предел группы
    не меняет ни логиты, ни потери — маски dropout те же, и
    генератор дальше идёт тем же потоком.
    """

    built = world.model(attention="flash", dropout=0.1).to(device())
    built.train()

    def run():

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        data = pack(clients, device())

        with autocast(device()):
            out = built(data)

        return data, out

    _, whole = run()

    monkeypatch.setattr(varlen, "FLASH_ROWS", 3 * FLASH_PAD)

    data, split = run()

    assert len(data.events.groups) > 1

    assert torch.equal(split.logits, whole.logits)
    assert torch.equal(split.loss, whole.loss)
