from __future__ import annotations

import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch

import src.mlm.model as model_module
from src.embedding.layer import InputEmbedding
from src.mlm.inputs import IGNORE
from src.mlm.model import RECENCY_CAP_DAYS, RECENT_DAYS, Model, Predicted, RecentTypes, pack
from src.mlm.settings import TELEMETRY_FILE, checkpoint_path, train_dir
from src.mlm.train import Scores, train
from src.mlm.varlen import to_device

from tests import world
from tests.test_attention_backends import reference_attend
from tests.test_checkpoint_resume import alike, read
from tests.test_recent_types import head as recent_head
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Ускорения прохода модели не имеют права менять результат. Прежний
# код (до ускорения) лежит здесь дословно — old_* — и проход
# сравнивается с ним побитно:
#
#   - счёт top-1/top-5 берётся из логитов кусков потерь, а не
#     отдельным проходом головы;
#   - число целей и цели RecentTypes считаются без синхронизации;
#   - embed на плоском пути не умножает на маску из единиц;
#   - cos/sin TimeRoPE приводятся к типу autocast один раз на
#     энкодер, а не в каждом повороте;
#   - pack переносит поля одним буфером на тип;
#   - цикл обучения читает числа прохода после нормы градиента.
#
# Побитно — loss, aux, hits, градиент каждого параметра, шаг AdamW,
# состояние генератора после прохода (dropout) и порядок вызовов
# ядра внимания. Путь flash на CPU идёт через посегментный эталон
# ядра под bf16 autocast CPU: так проверяются и приведения типов.
# ============================================================


# ------------------------------------------------------------
# прежний код, дословно
# ------------------------------------------------------------


def old_hits_in_pieces(head, token, event, client, weight, targets, k: int = 5) -> tuple[int, int]:

    first = five = targets.new_zeros(())

    with torch.no_grad():

        for piece in zip(*(part.split(model_module.TARGETS_PER_CHUNK) for part in (token, event, client, targets))):
            logits = head(piece[0], piece[1], piece[2], weight)
            labels = piece[3]
            top = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
            first = first + (top[:, 0] == labels).sum()
            five = five + (top == labels[:, None]).any(dim=-1).sum()

    return int(first), int(five)


def old_forward(self: Model, data, logits: bool = True) -> Predicted:

    token_vectors, event_vectors, client_vectors = self._encode(data)

    if token_vectors is None:
        token_vectors = client_vectors[:0]

    event_rows = event_vectors[data.target_event]
    client_rows = client_vectors[data.target_client]

    targets = data.labels[data.target_token]

    full = (
        self.head(token_vectors, event_rows, client_rows, self.embedding.weight)
        if logits else None
    )

    scored = None if logits else old_hits_in_pieces(
        self.head, token_vectors, event_rows, client_rows, self.embedding.weight, targets,
    )

    return Predicted(
        logits=full,
        hits=scored,
        aux=self.recent(data, client_vectors) if self.recent is not None else None,
        targets=targets,
        loss=model_module.mlm_loss(
            self.head, token_vectors, event_rows, client_rows,
            self.embedding.weight, targets, self.label_smoothing,
        ),
        place=data.target_place,
        event=data.target_local,
        client=data.target_client,
    )


def old_targets(self: RecentTypes, data) -> torch.Tensor:

    from src.temporal.position import TIME_SCALE

    original = torch.where(data.labels != IGNORE, data.labels, data.value_ids)

    typed = torch.nonzero(
        (data.key_ids == self.event_type_key) & (data.positions == 0), as_tuple=True
    )[0]

    device = data.event_time_log.device

    kind = torch.full((data.events.segments,), -1, dtype=torch.long, device=device)
    kind[data.event_of_token[typed]] = self.type_of_value[original[typed]]

    days = TIME_SCALE * torch.expm1(data.event_time_log.float() / TIME_SCALE) / 86_400.0

    known = kind >= 0
    windows = torch.tensor(RECENT_DAYS, dtype=days.dtype, device=device)
    inside = (days[:, None] <= windows[None, :]) & known[:, None]

    counts = torch.zeros(data.clients, len(RECENT_DAYS), self.types, dtype=torch.float32, device=device)
    event, window = torch.nonzero(inside, as_tuple=True)
    counts.index_put_(
        (data.user_of_event[event], window, kind[event]),
        torch.ones_like(event, dtype=torch.float32), accumulate=True,
    )

    recency = torch.full((data.clients, self.types), RECENCY_CAP_DAYS, dtype=torch.float32, device=device)
    chosen = torch.nonzero(known, as_tuple=True)[0]
    recency.view(-1).scatter_reduce_(
        0, data.user_of_event[chosen] * self.types + kind[chosen],
        days[chosen].clamp(max=RECENCY_CAP_DAYS), reduce="amin",
    )

    return torch.cat([counts, recency.unsqueeze(1)], dim=1).log1p()


def old_embed(self: InputEmbedding, key_ids, value_ids, positions, mask) -> torch.Tensor:

    # Прежние вызовы плоского пути подавали маску из единиц.
    if mask is None:
        mask = torch.ones_like(key_ids, dtype=torch.bool)

    visible = (~self.marker_of(key_ids)).unsqueeze(-1)

    rest = (self.table(value_ids) * self.scale + self.pieces_of(positions)) * visible

    return (self.table(key_ids) * self.scale + rest) * mask.unsqueeze(-1)


def old_code(patch) -> None:
    """Прежний проход: прежние embed, цели RecentTypes и углы TimeRoPE в fp32."""

    patch.setattr(InputEmbedding, "embed", old_embed)
    patch.setattr(RecentTypes, "targets", old_targets)
    patch.setattr(model_module, "_activation_type", lambda angles, x: angles)


# ------------------------------------------------------------
# проход обучения
# ------------------------------------------------------------


def run(path: str, dropout: float, old: bool, monkeypatch, calls: list) -> dict:
    """
    Один шаг обучения, как в train(): проход без логитов, backward
    по сумме потерь, деление градиентов на число целей, клип и AdamW.
    """

    built = world.model(attention=path, dropout=dropout)
    built.attach_recent(recent_head())
    built.train()

    if path == "flash":
        monkeypatch.setattr(built, "_flash", lambda: True)

    optimizer = torch.optim.AdamW(built.parameters(), lr=1e-2, weight_decay=0.01)
    data = pack(world.clients(), CPU)

    calls.clear()
    torch.manual_seed(11)

    autocast = torch.autocast("cpu", dtype=torch.bfloat16) if path == "flash" else nullcontext()

    with monkeypatch.context() as patch:

        if old:
            old_code(patch)

        with autocast:
            out = old_forward(built, data, logits=False) if old else built(data, logits=False)

        ((out.loss + 0.5 * out.aux) * out.count).backward()

    grads = {name: value.grad.clone() for name, value in built.named_parameters()}
    torch._foreach_div_([value.grad for value in built.parameters()], out.count)
    norm = torch.nn.utils.clip_grad_norm_(built.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": out.loss.detach(), "aux": out.aux.detach(), "targets": out.targets,
        "hits": tuple(int(value) for value in out.hits), "count": out.count,
        "grads": grads, "norm": norm, "rng": torch.get_rng_state(), "calls": list(calls),
        "model": built.state_dict(), "optimizer": optimizer.state_dict(),
    }


@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("path", ["sdpa", "flash"])
def test_training_pass_is_the_old_pass(monkeypatch, path: str, dropout: float):
    """
    Куски по две цели: счёт top-1/top-5 собирается из многих кусков.
    Равны побитно потери, aux, счёт, градиенты, шаг AdamW, генератор
    после прохода и вызовы ядра внимания.
    """

    monkeypatch.setattr(model_module, "TARGETS_PER_CHUNK", 2)

    calls: list = []

    def spy(query, key, value, layout, probability):
        calls.append((tuple(query.shape), query.dtype, layout.cu_seqlens.tolist(), probability))
        return reference_attend(query, key, value, layout, probability)

    monkeypatch.setattr("src.mlm.varlen.attend", spy)

    new = run(path, dropout, False, monkeypatch, calls)
    old = run(path, dropout, True, monkeypatch, calls)

    assert new["count"] > 2, "нужно несколько кусков"
    assert (path == "flash") == bool(new["calls"]), "путь flash не был пройден"

    alike(new, old)


def test_the_head_runs_once_per_chunk_and_once_more_in_backward(monkeypatch):
    """
    Прямой проход куска и его пересчёт в backward — два прохода
    головы на кусок; отдельного прохода ради счёта нет.
    """

    monkeypatch.setattr(model_module, "TARGETS_PER_CHUNK", 2)

    built = world.model()
    built.train()

    runs = []
    built.head.register_forward_hook(lambda module, items, out: runs.append(len(items[0])))

    out = built(pack(world.clients(), CPU), logits=False)
    (out.loss * out.count).backward()

    pieces = [len(piece) for piece in out.targets.split(2)]

    # backward пересчитывает куски в обратном порядке.
    assert len(pieces) > 1 and runs == pieces + pieces[::-1]


def test_validation_pass_is_the_old_pass(monkeypatch):
    """
    Проход val (eval, полные логиты) тот же: логиты, потери и aux.
    """

    data = pack(world.clients(), CPU)
    outs = []

    for old in (False, True):

        built = world.model(dropout=0.1)
        built.attach_recent(recent_head())
        built.eval()

        with monkeypatch.context() as patch, torch.no_grad():
            if old:
                old_code(patch)
            outs.append(old_forward(built, data) if old else built(data))

    new, old = outs

    for name in ("logits", "loss", "aux", "targets"):
        assert torch.equal(getattr(new, name), getattr(old, name)), name

    assert new.hits is None and old.hits is None


# ------------------------------------------------------------
# части
# ------------------------------------------------------------


def adversarial(data):
    """
    Цели на краях: давность у самых границ окон и за пределом, тип
    закрытого события из метки, испорченный тип без метки.
    """

    from dataclasses import replace

    from src.temporal.position import TIME_SCALE

    times = data.event_time_log.clone()
    edges = [*RECENT_DAYS, RECENCY_CAP_DAYS, 10 * RECENCY_CAP_DAYS]
    for number, days in enumerate(edges[: times.numel()]):
        times[number] = TIME_SCALE * np.log1p(days * 86_400.0 / TIME_SCALE)

    typed = torch.nonzero((data.key_ids == world.KEY_A) & (data.positions == 0))[:, 0]
    labels, values = data.labels.clone(), data.value_ids.clone()

    # Первое — закрыто маской, метка настоящая; второе — испорчено
    # в значение без типа и без метки.
    labels[typed[0]], values[typed[0]] = values[typed[0]], 0
    labels[typed[1]], values[typed[1]] = IGNORE, 0

    return replace(data, event_time_log=times, labels=labels, value_ids=values)


def test_recent_targets_are_the_old_targets():

    recent = recent_head()
    data = pack(world.clients(), CPU)

    for batch in (data, adversarial(data)):
        assert torch.equal(recent.targets(batch), old_targets(recent, batch))


def test_embed_without_a_mask_is_embed_with_ones():

    embedding = world.embedding()
    data = pack(world.clients(), CPU)

    outs, grads = [], []

    for mask in (None, torch.ones_like(data.key_ids, dtype=torch.bool)):
        embedding.zero_grad()
        out = embedding.embed(data.key_ids, data.value_ids, data.positions, mask)
        out.square().sum().backward()
        outs.append(out.detach())
        grads.append(embedding.weight.grad.clone())

    assert torch.equal(*outs) and torch.equal(*grads)


def test_angles_take_the_activation_type_only_under_autocast():
    """
    Под autocast cos/sin приходят в его типе — тот же тензор, что
    дал бы поворот, приводя fp32 сам; без autocast они остаются fp32.
    """

    angles = world.model().history.rope.angles(torch.linspace(0.0, 30.0, 17))
    x = torch.zeros(3)

    assert all(value.dtype == torch.float32 for value in model_module._activation_type(angles, x))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = model_module._activation_type(angles, x)

    assert all(torch.equal(one, two.to(torch.bfloat16)) for one, two in zip(cast, angles))


def test_one_transfer_per_type_keeps_every_array():

    arrays = {
        "long": np.arange(37, dtype=np.int64),
        "empty": np.zeros(0, dtype=np.int64),
        "table": np.arange(18, dtype=np.float32).reshape(3, 6),
        "short": np.array([5, 7], dtype=np.int32),
        "times": np.linspace(0, 1, 5, dtype=np.float32),
    }

    moved = to_device(arrays, CPU)

    assert set(moved) == set(arrays)

    for name, array in arrays.items():
        assert torch.equal(moved[name], torch.as_tensor(array)), name
        assert moved[name].is_contiguous()


def test_flash_targets_need_no_bucket_fields():
    """
    Поля корзин SDPA у прохода flash не строятся; у SDPA они те же,
    что считались в pack.
    """

    data = pack(world.clients(), CPU)

    assert {"target_inside", "target_bucket", "target_row"}.isdisjoint(vars(data))

    starts = data.events.cu_seqlens[:-1].numpy()
    target_event = data.target_event.numpy()

    assert data.target_inside.tolist() == (data.target_token.numpy() - starts[target_event]).tolist()
    assert data.target_bucket.tolist() == data.events.bucket_of[target_event].tolist()
    assert data.target_row.tolist() == data.events.row_of[target_event].tolist()


# ------------------------------------------------------------
# на CUDA: закреплённый буфер и плотные выборки
# ------------------------------------------------------------


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA")


@pytest.mark.cuda
@cuda_only
def test_one_transfer_per_type_on_cuda_keeps_every_array():
    """
    На CUDA буфер закреплённый, а перенос асинхронный. Долгое ядро
    впереди на потоке держит первый перенос в очереди, пока второй
    вызов берёт свой буфер: аллокатор не вправе отдать ему блок,
    который ещё не скопирован. Значения — те, что были в массивах в
    момент вызова, даже если массивы тут же переписаны.
    """

    cuda = torch.device("cuda")

    def arrays(shift: int) -> dict:
        return {
            "long": np.arange(300_000, dtype=np.int64) + shift,
            "empty": np.zeros(0, dtype=np.int64),
            "table": (np.arange(180_000, dtype=np.float32) + shift).reshape(-1, 6),
            "columns": (np.arange(24, dtype=np.int64) + shift).reshape(4, 6)[:, ::2],
            "short": np.array([5, 7], dtype=np.int32) + shift,
            "flags": (np.arange(9) + shift) % 3 == 0,
        }

    batches = [arrays(0), arrays(1000)]
    expected = [{name: np.ascontiguousarray(array).copy() for name, array in batch.items()} for batch in batches]

    torch.cuda.synchronize()
    torch.cuda._sleep(200_000_000)

    moved = [to_device(batch, cuda) for batch in batches]

    for batch in batches:
        for array in batch.values():
            array[...] = 0

    torch.cuda.synchronize()

    for out, want in zip(moved, expected):

        assert set(out) == set(want)

        for name, array in want.items():
            assert out[name].is_cuda and out[name].is_contiguous(), name
            assert torch.equal(out[name].cpu(), torch.from_numpy(array)), name


@pytest.mark.cuda
@cuda_only
def test_recent_targets_on_cuda_are_the_old_targets():
    """
    Плотные выборки на CUDA — накопление index_put_ и минимум
    scatter_reduce — дают те же цели, что прежние nonzero, до бита.
    """

    cuda = torch.device("cuda")

    recent = recent_head().to(cuda)
    data = pack(world.clients(), cuda)

    for batch in (data, adversarial(data)):
        assert torch.equal(recent.targets(batch), old_targets(recent, batch))


# ------------------------------------------------------------
# цикл обучения
# ------------------------------------------------------------


def test_training_reads_the_numbers_of_its_passes_in_order(stage, monkeypatch):
    """
    Числа прохода (loss, счёт, aux) читаются после нормы градиента,
    но складываются в том же порядке: loss шага в телеметрии, счёт
    эпохи и среднее aux те же, что при чтении сразу после прохода.
    """

    monkeypatch.setattr(model_module, "recent_types", lambda artifacts, dim, seed: recent_head(dim))

    settle(stage, train_people=many(), dropout=0.1)

    passes: list[Predicted] = []
    forward = Model.forward

    def look(self, data, logits=True):
        out = forward(self, data, logits=logits)
        if self.training:
            passes.append(out)
        return out

    monkeypatch.setattr(Model, "forward", look)

    train(tiny(token_budget=6, grad_accum_steps=2, usr_aux_weight=0.5), epochs=1, max_steps=None,
          masking=every_value())

    rows = [json.loads(line) for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()]

    expected, losses = Scores(), []

    for first in range(0, len(passes), 2):
        window = [out for out in passes[first:first + 2] if out.count]
        total = 0.0
        for out in window:
            total += out.loss.item() * out.count
            expected.add(out)
        losses.append(total / sum(out.count for out in window))

    aux = float(np.mean([float(out.aux.detach()) for out in passes if out.count]))

    assert len(passes) > 2
    assert [row["loss"] for row in rows if row["kind"] == "step"] == losses
    assert [row["usr_aux_mean"] for row in rows if row["kind"] == "epoch"] == [aux]

    history = read(checkpoint_path())["history"]

    assert history[0]["train"] == expected.summary()
