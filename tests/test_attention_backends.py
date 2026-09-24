from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.mlm.model import Model, pack
from src.mlm.varlen import (
    BackendError,
    VarlenLayout,
    encoder_layer_varlen,
    flash_available,
    history_block_varlen,
    resolve_backend,
)

from tests import world
from tests.test_isolation import long, short
from tests.test_scheduler import many
from tests.test_training_math import every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Бэкенд внимания — это способ счёта, а не другая модель. auto
# молча берёт корзины, когда flash недоступен; явный flash без
# него обязан сказать об этом ошибкой, а не подменить путь тихо.
#
# Сам путь flash на этой машине недостижим: библиотеки нет, а
# Model._flash дополнительно требует autocast CUDA. Но его
# арифметика проверяема и на CPU: attend подменяется эталоном на
# SDPA, и тогда varlen-слои считают то же, что честный проход по
# одному сегменту.
#
# Эта же подмена даёт главное, что требуется от flash-пути:
# доказать, что в attend уходят ПЛОСКИЕ Q/K/V без заполнителя, а
# не прямоугольник [B, max_length, ...].
# ============================================================


CPU = torch.device("cpu")
CUDA = torch.device("cuda")


def fake_flash(monkeypatch, available: bool) -> None:
    monkeypatch.setattr("src.mlm.varlen.flash_available", lambda: available)


# ============================================================
# ВЫБОР БЭКЕНДА
# ============================================================


@pytest.mark.parametrize("available", [False, True])
def test_sdpa_is_never_overridden(monkeypatch, available: bool):
    """
    sdpa значит sdpa: библиотека при этом даже не опрашивается.
    """

    fake_flash(monkeypatch, available)

    assert resolve_backend("sdpa", CPU) == "sdpa"
    assert resolve_backend("sdpa", CUDA) == "sdpa"


def test_auto_falls_back_to_buckets_without_cuda(monkeypatch):

    fake_flash(monkeypatch, True)

    assert resolve_backend("auto", CPU) == "sdpa"


def test_auto_falls_back_to_buckets_without_the_library(monkeypatch):

    fake_flash(monkeypatch, False)

    assert resolve_backend("auto", CUDA) == "sdpa"


def test_auto_takes_flash_when_both_are_there(monkeypatch):

    fake_flash(monkeypatch, True)

    assert resolve_backend("auto", CUDA) == "flash"


def test_explicit_flash_without_cuda_is_an_error(monkeypatch):
    """
    Тихая подмена спрятала бы, что обучение идёт не тем путём,
    который просили.
    """

    fake_flash(monkeypatch, True)

    with pytest.raises(BackendError, match="нет CUDA"):
        resolve_backend("flash", CPU)


def test_explicit_flash_without_the_library_is_an_error(monkeypatch):

    fake_flash(monkeypatch, False)

    with pytest.raises(BackendError, match="flash-attn не установлена"):
        resolve_backend("flash", CUDA)


def test_unknown_backend_is_named_in_the_error():

    with pytest.raises(BackendError, match=r"\['auto', 'flash', 'sdpa'\]"):
        resolve_backend("triton", CPU)


# ============================================================
# ПОИСК БИБЛИОТЕКИ
# ============================================================


def test_missing_library_is_simply_absent(monkeypatch):

    monkeypatch.setitem(sys.modules, "flash_attn", None)

    assert flash_available() is False


def test_a_broken_extension_counts_as_absent(monkeypatch):
    """
    Несовместимая сборка CUDA-расширения падает при загрузке с
    OSError или RuntimeError, а не с ImportError. Сломанная
    библиотека — то же, что её нет.
    """

    broken = types.ModuleType("flash_attn")

    def explode(name):
        raise OSError("не та сборка CUDA")

    broken.__getattr__ = explode

    monkeypatch.setitem(sys.modules, "flash_attn", broken)

    assert flash_available() is False


def test_a_working_library_is_found(monkeypatch):

    ready = types.ModuleType("flash_attn")
    ready.flash_attn_varlen_func = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "flash_attn", ready)

    assert flash_available() is True


# ============================================================
# ЧЕРЕЗ КОМАНДУ
# ============================================================


def test_command_reports_an_impossible_backend_instead_of_crashing(stage, tmp_path, capsys):
    """
    CLI обязан поймать BackendError и вернуть код «сделать
    нельзя», а не вывалить traceback.
    """

    from src.mlm.train import build_parser, run_training
    from src.preprocessing.run import EXIT_BLOCKED

    settle(stage, train_people=many())

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"device": "cpu", "attention_backend": "flash"}), encoding="utf-8"
    )

    args = build_parser().parse_args(["--config", str(path)])

    assert run_training(args) == EXIT_BLOCKED
    assert "flash" in capsys.readouterr().out


# ============================================================
# STRICT: FLASH БЕЗ AUTOCAST
# ============================================================


def test_strict_flash_refuses_to_run_in_fp32(clients):
    """
    flash-attn в fp32 не считает. Явно запрошенный flash без
    autocast обязан сказать об этом, а не посчитать иначе.
    """

    built = world.model(attention="sdpa")

    strict = Model(
        embedding=built.embedding, event=built.event, profile=built.profile,
        history=built.history, head=built.head,
        attention="flash", strict=True,
    )

    with pytest.raises(BackendError, match="не под autocast"):
        strict(pack(clients, CPU))


def test_resolved_flash_without_strict_quietly_uses_buckets(clients, model):
    """
    auto, выбравший flash, но попавший в проход без autocast,
    считает корзинами — и обязан дать тот же ответ.
    """

    lenient = Model(
        embedding=model.embedding, event=model.event, profile=model.profile,
        history=model.history, head=model.head,
        attention="flash", strict=False,
    )
    lenient.eval()

    data = pack(clients, CPU)

    with torch.no_grad():
        assert torch.equal(lenient(data).logits, model(data).logits)


# ============================================================
# АРИФМЕТИКА VARLEN НА CPU
# ============================================================


def reference_attend(query, key, value, layout, dropout):
    """
    Эталон вместо flash_attn_varlen_func: то же внимание, но
    посегментно через SDPA.

    Масштаб у обоих один — 1/sqrt(head_dim), — поэтому сравнивать
    можно напрямую.
    """

    out = torch.zeros_like(query)

    bounds = layout.cu_seqlens.tolist()

    for number in range(len(bounds) - 1):

        first, last = bounds[number], bounds[number + 1]

        piece = [
            item[first:last].transpose(0, 1).unsqueeze(0) for item in (query, key, value)
        ]

        answer = F.scaled_dot_product_attention(*piece, dropout_p=dropout)

        out[first:last] = answer.squeeze(0).transpose(0, 1)

    return out


def test_varlen_encoder_layer_matches_the_layer_itself(monkeypatch):
    """
    varlen-слой переписывает формулу nn.TransformerEncoderLayer
    руками. Проверяется он против самого слоя, посегментно.
    """

    monkeypatch.setattr("src.mlm.varlen.attend", reference_attend)

    built = world.model()
    built.eval()

    layer = built.event.layers[0]

    lengths = [3, 4, 1, 4]

    layout = VarlenLayout.build(
        torch.tensor(lengths).numpy(), CPU, "события"
    )

    torch.manual_seed(1)

    flat = torch.randn(sum(lengths), world.DIM)

    with torch.no_grad():

        mine = encoder_layer_varlen(layer, flat, layout)

        bounds = layout.cu_seqlens.tolist()

        for number in range(len(lengths)):

            first, last = bounds[number], bounds[number + 1]

            theirs = layer(flat[first:last].unsqueeze(0))[0]

            assert torch.allclose(mine[first:last], theirs, atol=1e-6)


def test_varlen_history_block_matches_the_block_itself(monkeypatch):
    """
    То же для блока истории — вместе с поворотом TimeRoPE.
    """

    monkeypatch.setattr("src.mlm.varlen.attend", reference_attend)

    built = world.model()
    built.eval()

    block = built.history.layers[0]
    rope = built.history.rope

    lengths = [3, 4, 1]

    layout = VarlenLayout.build(torch.tensor(lengths).numpy(), CPU, "истории")

    torch.manual_seed(2)

    flat = torch.randn(sum(lengths), world.DIM)
    positions = torch.rand(sum(lengths)) * 10.0

    cos, sin = rope.angles(positions)

    with torch.no_grad():

        mine = history_block_varlen(block, rope, flat, cos, sin, layout)

        bounds = layout.cu_seqlens.tolist()

        for number in range(len(lengths)):

            first, last = bounds[number], bounds[number + 1]

            own_cos, own_sin = rope.angles(positions[first:last])

            theirs = block(flat[first:last].unsqueeze(0), rope, own_cos, own_sin, None)[0]

            assert torch.allclose(mine[first:last], theirs, atol=1e-6)


# ============================================================
# ПУТЬ FLASH НЕ СТРОИТ ПРЯМОУГОЛЬНИКОВ
# ============================================================


def test_flash_path_passes_flat_tensors_without_padding(monkeypatch, clients):
    """
    Главное требование к varlen-пути: в attend уходят ПЛОСКИЕ
    Q/K/V формы [сумма настоящих позиций, головы, head_dim].

    Прямоугольник [B, max_length, ...] означал бы, что заполнитель
    вернулся, а вместе с ним и вся работа, ради устранения которой
    varlen и заведён.
    """

    seen: list[dict] = []

    def spy(query, key, value, layout, dropout):

        seen.append(
            {
                "shape": tuple(query.shape),
                "same": query.shape == key.shape == value.shape,
                "bounds": layout.cu_seqlens.tolist(),
                "lengths": layout.lengths.tolist(),
            }
        )

        return reference_attend(query, key, value, layout, dropout)

    monkeypatch.setattr("src.mlm.varlen.attend", spy)

    built = world.model(attention="flash")
    built.eval()

    # Автокаста CUDA здесь нет, поэтому путь включается прямо.
    monkeypatch.setattr(built, "_flash", lambda: True)

    data = pack(clients, CPU)

    with torch.no_grad():
        out = built(data)

    assert seen, "путь flash не был пройден"

    heads = built.event.layers[0].self_attn.num_heads

    widths = {
        int(data.key_ids.numel()),
        int(data.profile_key_ids.numel()),
        int(data.clients + data.events.segments),
    }

    for call in seen:

        rows, count, head_dim = call["shape"]

        assert call["same"]
        assert count == heads
        assert head_dim == world.DIM // heads

        # Ровно сумма длин сегментов: ни одной строки заполнителя.
        assert rows == sum(call["lengths"])
        assert rows == call["bounds"][-1]
        assert rows in widths

    assert out.logits.shape == (int(data.target_token.numel()), world.VOCAB)


def test_flash_path_and_buckets_agree(monkeypatch, clients):
    """
    Две реализации одного вычисления: плоская и по корзинам.
    """

    monkeypatch.setattr("src.mlm.varlen.attend", reference_attend)

    built = world.model(attention="flash")
    built.eval()

    data = pack(clients, CPU)

    with torch.no_grad():

        buckets = built(data)

        monkeypatch.setattr(built, "_flash", lambda: True)

        flat = built(data)

    assert torch.allclose(flat.logits, buckets.logits, atol=1e-5, rtol=1e-5)
    assert float(flat.loss) == pytest.approx(float(buckets.loss), rel=1e-5)
