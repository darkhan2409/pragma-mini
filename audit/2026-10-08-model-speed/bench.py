"""
Замеры и сверки прохода PRAGMA (forward / backward / шаг оптимизатора) в
раскладке data/ во временном каталоге --root (как у
audit/2026-10-07-input-runtime/bench.py setup). Настоящий data/ не
трогается. Какой код мерить, задаёт PYTHONPATH.

    profile  шаг обучения на фиксированных micro-batch'ах эпохи 1: время частей
             (CUDA events и время CPU), число вызовов головы, синхронизации,
             переносы CPU→GPU, ядра (torch.profiler), пик памяти, по размерам;
    dump     выходы прохода, градиенты, шаг AdamW, val-проход и состояние
             генераторов случайности — в файл, для побайтной сверки двух версий
             кода; режимы cpu-sdpa, cpu-flash (путь flash с посегментным
             ядром-эталоном под bf16 autocast CPU), gpu-det (flash с
             deterministic=True и детерминированными алгоритмами), gpu;
    compare  два файла dump или два чекпойнта: побайтно, иначе max|Δ|;
    train    настоящий train() с --out в --root/12_train/<label>: телеметрия
             шагов, загрузка GPU, пик памяти; --deterministic — режим gpu-det,
             --cpu — CPU в один поток с конфигом --budget.

CPU-режимы запускать с MKL_CBWR=AUTO,STRICT; gpu-det — с
CUBLAS_WORKSPACE_CONFIG=:4096:8 (их ставит сам bench до импорта torch).
"""

from __future__ import annotations

import os
import sys

# До импорта torch: воспроизводимость BLAS на CPU и cuBLAS.
os.environ.setdefault("MKL_CBWR", "AUTO,STRICT")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import functools
import hashlib
import io
import json
import resource
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

PLACES = (
    ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
    ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
    ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ("src.dataset.settings", "DATASET_DIR", "05_dataset"),
    ("src.embedding.settings", "EMBEDDINGS_DIR", "06_embeddings"),
    ("src.mlm.settings", "BACKBONE_DIR", "07_backbone"),
    ("src.mlm.settings", "TRAIN_DIR", "12_train"),
)


def _outside_data(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(ROOT / "data"):
        raise SystemExit(f"{path}: замеры не пишут в data/")
    return path


def _redirect(root: Path) -> None:
    for module, attribute, folder in PLACES:
        setattr(import_module(module), attribute, root / folder)


def _code() -> str:
    return str(Path(import_module("src.mlm.model").__file__).resolve().parents[2])


# ------------------------------------------------------------
# режимы
# ------------------------------------------------------------


def _deterministic() -> None:
    """flash с детерминированным обратным проходом и детерминированные алгоритмы."""

    import flash_attn
    import torch

    original = flash_attn.flash_attn_varlen_func
    flash_attn.flash_attn_varlen_func = functools.partial(original, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _reference_attend(query, key, value, layout, dropout):
    """Эталон ядра flash на CPU: то же внимание посегментно через SDPA."""

    import torch
    import torch.nn.functional as F

    out = torch.zeros_like(query)
    bounds = layout.cu_seqlens.tolist()

    for number in range(len(bounds) - 1):
        first, last = bounds[number], bounds[number + 1]
        piece = [item[first:last].transpose(0, 1).unsqueeze(0) for item in (query, key, value)]
        out[first:last] = F.scaled_dot_product_attention(*piece, dropout_p=dropout).squeeze(0).transpose(0, 1)

    return out


def _setup(mode: str):
    """
    Устройство, бэкенд и контекст autocast режима. cpu-flash ведёт путь
    flash модели на CPU: ядро — посегментный эталон, autocast — bf16 CPU.
    """

    import torch

    from src.mlm import model as model_module
    from src.mlm import varlen

    if mode.startswith("cpu"):
        torch.set_num_threads(1)
        device = torch.device("cpu")
        if mode == "cpu-flash":
            varlen.attend = _reference_attend
            model_module.Model._flash = lambda self: True
            return device, "sdpa", "flash", lambda: torch.autocast("cpu", dtype=torch.bfloat16)
        return device, "sdpa", None, nullcontext

    device = torch.device("cuda")

    if mode == "gpu-det":
        _deterministic()

    return device, "flash", None, lambda: varlen.autocast(device)


def _build(device, backend: str, attention: str | None):
    """Модель и оптимизатор так же, как их собирает train()."""

    import torch

    from src.mlm.model import load_model
    from src.mlm.settings import MlmConfig
    from src.mlm.train import attach_recent

    config = MlmConfig()

    model = load_model(seed=config.seed, events_per_chunk=config.events_per_chunk,
                       label_smoothing=config.label_smoothing, device=device, attention_backend=backend)

    if config.usr_aux_weight > 0.0:
        attach_recent(model, config)

    if attention is not None:
        model.attention = attention

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, fused=device.type == "cuda")

    torch.manual_seed(config.seed)

    return model, optimizer, config


def _variant(name: str | None) -> None:
    """
    Экспериментальные варианты — только подменой в процессе замера;
    код src не меняется. Числа у них другие (кроме chunk-сдвига в
    пределах суммы), в безопасный набор они не входят.

      sdpa-events      энкодер события корзинами SDPA, анкета и история flash;
      no-checkpoint    потери кусками без пересчёта в backward;
      chunk=N          TARGETS_PER_CHUNK = N;
      compile-layers   torch.compile слоёв энкодеров (encoder_layer_varlen,
                       history_block_varlen), dynamic=True;
      compile-recent   torch.compile RecentTypes.forward;
      compile-embed    torch.compile InputEmbedding.embed.
    """

    if not name:
        return

    import torch

    from src.embedding.layer import InputEmbedding
    from src.mlm import model as model_module

    if name == "sdpa-events":
        model_module.Model._events_flash = model_module.Model._events
    elif name == "no-checkpoint":
        model_module.checkpoint = lambda function, *items, use_reentrant: function(*items)
    elif name.startswith("chunk="):
        model_module.TARGETS_PER_CHUNK = int(name.split("=")[1])
    elif name == "compile-layers":
        model_module.encoder_layer_varlen = torch.compile(model_module.encoder_layer_varlen, dynamic=True)
        model_module.history_block_varlen = torch.compile(model_module.history_block_varlen, dynamic=True)
    elif name == "compile-recent":
        model_module.RecentTypes.forward = torch.compile(model_module.RecentTypes.forward, dynamic=True)
    elif name == "compile-embed":
        InputEmbedding.embed = torch.compile(InputEmbedding.embed, dynamic=True)
    else:
        raise SystemExit(f"вариант {name}?")


def _batches(group: str, count: int, epoch: int = 1) -> list:
    """Первые count micro-batch'ей эпохи в порядке обучения (вход тот же у обеих версий)."""

    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Prefetch, micro_batches
    from src.mlm.settings import MlmConfig
    from src.mlm.train import train_source

    source = train_source(MlmConfig(), MaskingConfig(), epoch) if group == "train" else \
        import_module("src.mlm.inputs").Source(group, masking=MaskingConfig())

    out = []
    for batch in micro_batches(Prefetch(source, 0).clients(), MlmConfig().token_budget):
        out.append(batch)
        if len(out) >= count:
            break
    return out


# ------------------------------------------------------------
# шаг обучения
# ------------------------------------------------------------


def _step(model, optimizer, config, clients, device, context, timer=None):
    """
    Один шаг обучения, как в train(): проход, backward с весом count,
    деление градиентов на число целей, клип, шаг AdamW.
    Возвращает (loss, aux, count, hits, norm) как числа Python.
    """

    import torch

    from src.mlm.model import pack

    mark = timer or (lambda name: None)

    mark("start")
    data = pack(clients, device)
    mark("pack")

    with context():
        out = model(data, logits=False)
    mark("forward")

    loss = aux = None
    if out.count > 0:
        objective = out.loss
        if out.aux is not None:
            objective = objective + config.usr_aux_weight * out.aux
        (objective * out.count).backward()
    mark("backward")

    norm = None
    if out.count > 0:
        torch._foreach_div_([p.grad for p in model.parameters() if p.grad is not None], out.count)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    mark("clip")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mark("optimizer")

    loss = out.loss.detach()
    aux = out.aux.detach() if out.aux is not None else None

    return loss, aux, out.count, out.hits, norm


# ------------------------------------------------------------
# profile
# ------------------------------------------------------------


# Части прохода, которые меряются CUDA events: (модуль, класс или None, имя).
PARTS = [
    ("src.mlm.model", "Model", "_events_flash"),
    ("src.mlm.model", "Model", "_profiles_flash"),
    ("src.mlm.model", "Model", "_history_flash"),
    ("src.mlm.model", "Model", "_history_input"),
    ("src.mlm.model", None, "mlm_loss"),
    ("src.mlm.model", None, "hits_in_pieces"),
    ("src.mlm.model", "RecentTypes", "forward"),
    ("src.mlm.model", "RecentTypes", "targets"),
    ("src.mlm.model", "Mlm", "forward"),
    ("src.embedding.layer", "InputEmbedding", "embed"),
    ("src.mlm.varlen", None, "encoder_layer_varlen"),
    ("src.mlm.varlen", None, "history_block_varlen"),
    ("src.mlm.varlen", None, "attend"),
    ("src.mlm.varlen", "VarlenLayout", "build"),
]


def _wrap_parts(records, calls):
    """Обёртки с парой CUDA events на вызов; без синхронизаций внутри шага."""

    import torch

    restore = []

    for module_name, owner_name, name in PARTS:
        module = import_module(module_name)
        owner = getattr(module, owner_name) if owner_name else module
        original = owner.__dict__.get(name) if owner_name else getattr(module, name, None)
        if original is None:
            continue
        label = f"{owner_name + '.' if owner_name else ''}{name}"
        function = original.__func__ if isinstance(original, staticmethod) else original

        def wrapped(*items, _function=function, _label=label, **named):
            calls[_label] += 1
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            try:
                return _function(*items, **named)
            finally:
                end.record()
                records[_label].append((begin, end))

        setattr(owner, name, staticmethod(wrapped) if isinstance(original, staticmethod) else wrapped)
        restore.append((owner, name, original))

    return restore


def command_profile(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import numpy as np
    import torch

    device, backend, attention, context = _setup("gpu")
    _variant(args.variant)

    batches = _batches("train", args.warmup + args.steps)
    model, optimizer, config = _build(device, backend, attention)
    model.train()

    sizes = [sum(c.n_tokens + c.profile_n_tokens for c in batch) for batch in batches]

    # Прогрев: аллокатор, ядра, кэш autocast (и компиляция у compile-*).
    losses = []
    warm = time.perf_counter()
    for batch in batches[:args.warmup]:
        loss, *_ = _step(model, optimizer, config, batch, device, context)
        losses.append(loss)
    torch.cuda.synchronize()
    warm = time.perf_counter() - warm
    torch.cuda.reset_peak_memory_stats()

    records: dict = defaultdict(list)
    calls: dict = defaultdict(int)
    restore = _wrap_parts(records, calls) if args.parts else []

    phases = ("start", "pack", "forward", "backward", "clip", "optimizer")
    step_events = []
    host = defaultdict(float)
    targets = 0

    started = time.perf_counter()

    try:
        for batch in batches[args.warmup:]:
            events = {}
            stamps = {}

            def mark(name, _events=events, _stamps=stamps):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                _events[name] = event
                _stamps[name] = time.perf_counter()

            loss, aux, count, hits, norm = _step(model, optimizer, config, batch, device, context, mark)
            losses.append(loss)
            targets += count
            # Число Python из нормы — как в train(): одна синхронизация на шаг.
            float(norm) if norm is not None else None
            step_events.append(events)
            for left, right in zip(phases, phases[1:]):
                host[right] += stamps[right] - stamps[left]
    finally:
        torch.cuda.synchronize()
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)

    wall = time.perf_counter() - started
    measured = len(batches) - args.warmup

    gpu = defaultdict(float)
    per_step = []
    for events in step_events:
        total = events["start"].elapsed_time(events["optimizer"])
        per_step.append(total)
        for left, right in zip(phases, phases[1:]):
            gpu[right] += events[left].elapsed_time(events[right])

    tokens = sum(sizes[args.warmup:])

    record = {
        "label": args.label, "code": _code(), "variant": args.variant, "warmup_s": round(warm, 2),
        "losses": [float(loss) for loss in losses],
        "warmup": args.warmup, "steps": measured,
        "wall_s": round(wall, 3), "step_ms": round(1000 * wall / measured, 2),
        "tokens_per_s": round(tokens / wall, 1), "targets_per_s": round(targets / wall, 1),
        "steps_per_s": round(measured / wall, 3),
        "phase_ms_gpu": {name: round(value / measured, 3) for name, value in gpu.items()},
        "phase_ms_host": {name: round(1000 * value / measured, 3) for name, value in host.items()},
        "parts_ms_per_step": {label: round(sum(b.elapsed_time(e) for b, e in pairs) / measured, 3)
                              for label, pairs in records.items()},
        "calls_per_step": {label: round(count / measured, 2) for label, count in calls.items()},
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "tokens_per_step": round(tokens / measured, 1), "targets_per_step": round(targets / measured, 1),
    }

    # По размеру micro-batch: четверти по числу токенов.
    order = np.argsort(sizes[args.warmup:])
    quarters = np.array_split(order, 4)
    record["by_size"] = [
        {"tokens_from": int(sizes[args.warmup:][part[0]]), "tokens_to": int(sizes[args.warmup:][part[-1]]),
         "step_ms_gpu": round(float(np.mean([per_step[i] for i in part])), 2)}
        for part in quarters if len(part)
    ]

    if args.variant and args.variant.startswith("compile"):
        from torch._dynamo.utils import counters
        record["dynamo"] = {"stats": dict(counters["stats"]), "graph_breaks": sum(counters["graph_break"].values()),
                            "break_reasons": list(counters["graph_break"])[:5],
                            "recompiles": dict(counters["recompiles"]) if "recompiles" in counters else None}

    if args.syncs:
        record["syncs_per_step"] = _count_syncs(model, optimizer, config, batches[args.warmup:args.warmup + 5],
                                                device, context)

    if args.trace:
        record["kernels"] = _trace(model, optimizer, config, batches[args.warmup:args.warmup + args.trace],
                                   device, context, root / f"trace-{args.label}.txt")

    print(json.dumps(record, ensure_ascii=False, indent=1))

    (root / f"profile-{args.label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")


def _count_syncs(model, optimizer, config, batches, device, context) -> dict:
    """Синхронизации CPU↔GPU за шаг: предупреждения set_sync_debug_mode, по местам вызова."""

    import torch

    places = defaultdict(int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for batch in batches:
                _, _, _, _, norm = _step(model, optimizer, config, batch, device, context)
                float(norm) if norm is not None else None
        finally:
            torch.cuda.set_sync_debug_mode(0)

    for item in caught:
        if "synchroniz" not in str(item.message).lower():
            continue
        places[f"{Path(item.filename).name}:{item.lineno}"] += 1

    return {"per_step": round(sum(places.values()) / len(batches), 2),
            "places": {key: round(value / len(batches), 2) for key, value in sorted(places.items())}}


def _trace(model, optimizer, config, batches, device, context, path: Path) -> dict:
    """torch.profiler за несколько шагов: ядра, переносы, время CUDA по операциям."""

    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for batch in batches:
            _, _, _, _, norm = _step(model, optimizer, config, batch, device, context)
            float(norm) if norm is not None else None
        torch.cuda.synchronize()

    events = prof.events()
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    memcpy = [e for e in kernels if "Memcpy" in e.name or "memcpy" in e.name]

    path.write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60), encoding="utf-8")

    # Занятость GPU: объединение интервалов ядер к стене от первого до
    # последнего ядра; простои длиннее 50 мкс.
    spans = sorted((e.time_range.start, e.time_range.end) for e in kernels)
    busy, gaps, idle, reach = 0.0, 0, 0.0, spans[0][0]
    for start, end in spans:
        if start > reach:
            idle += start - reach
            gaps += start - reach > 50
        busy += max(0.0, end - max(start, reach))
        reach = max(reach, end)

    return {
        "gpu_busy_share": round(busy / (reach - spans[0][0]), 4),
        "gpu_idle_ms_per_step": round(idle / 1000 / len(batches), 2),
        "gaps_over_50us_per_step": round(gaps / len(batches), 1),
        "kernels_per_step": round(len(kernels) / len(batches), 1),
        "memcpy_per_step": round(len(memcpy) / len(batches), 1),
        "memcpy_h2d_per_step": round(sum(1 for e in memcpy if "HtoD" in e.name) / len(batches), 1),
        "memcpy_ms_per_step": round(sum(e.device_time for e in memcpy) / 1000 / len(batches), 3),
        "cuda_ms_per_step": round(sum(e.device_time for e in kernels) / 1000 / len(batches), 3),
    }


# ------------------------------------------------------------
# valpass: проход val на фиксированных micro-batch'ах
# ------------------------------------------------------------


def command_valpass(args) -> None:
    """
    Тело цикла validate() на одних и тех же micro-batch'ах val, без
    чтения: pack, проход с полными логитами, Scores.add и Detail.add.
    Повторяется --repeat раз, в отчёт — каждый повтор.
    """

    root = _outside_data(args.root)
    _redirect(root)

    import numpy as np
    import torch

    from src.mlm.model import pack
    from src.mlm.train import Detail, Scores
    from src.tokenization.finalvocab import load_final_vocab

    device, backend, attention, context = _setup("gpu")
    model, _, _ = _build(device, backend, attention)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
    model.eval()

    batches = _batches("val", 10**9)
    vocab = load_final_vocab()
    key_names = np.empty(len(vocab), dtype=object)
    for token, number in vocab.items():
        key_names[number] = token

    seconds, parts = [], []
    for _ in range(args.repeat):
        scores, detail = Scores(), Detail()
        spent = defaultdict(float)
        torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.no_grad():
            for clients in batches:
                mark = time.perf_counter()
                with context():
                    out = model(pack(clients, device))
                scores.add(out)
                if args.parts:
                    torch.cuda.synchronize()
                spent["model+scores"] += time.perf_counter() - mark
                mark = time.perf_counter()
                if not args.no_detail:
                    detail.add(out, clients, key_names)
                spent["detail"] += time.perf_counter() - mark
                del out
        torch.cuda.synchronize()
        seconds.append(round(time.perf_counter() - began, 3))
        parts.append({name: round(value, 3) for name, value in spent.items()})

    record = {"label": args.label, "code": _code(), "batches": len(batches), "seconds": seconds,
              "parts": parts, "scores": scores.as_dict(),
              "detail_sha256": hashlib.sha256(json.dumps(detail.summary(), sort_keys=True).encode()).hexdigest()[:16]}
    print(json.dumps(record, ensure_ascii=False))
    (root / f"valpass-{args.label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------
# breakdown: время CUDA по частям, прямой и обратный проход
# ------------------------------------------------------------


# Функции, вызовы которых становятся областями профиля.
SCOPES = PARTS + [
    ("src.history.encoder", "TimeRoPE", "rotate"),
    ("src.history.encoder", "TimeRoPE", "_rotate_half"),
    ("src.embedding.layer", "InputEmbedding", "marker_of"),
    ("src.embedding.layer", "InputEmbedding", "pieces_of"),
    ("src.mlm.model", None, "_piece_loss"),
]


def _scope_functions(restore: list) -> None:

    from torch.profiler import record_function

    for module_name, owner_name, name in SCOPES:
        module = import_module(module_name)
        owner = getattr(module, owner_name) if owner_name else module
        original = owner.__dict__.get(name) if owner_name else getattr(module, name, None)
        if original is None:
            continue
        label = f"{owner_name + '.' if owner_name else ''}{name}"
        function = original.__func__ if isinstance(original, staticmethod) else original

        def wrapped(*items, _function=function, _label=label, **named):
            with record_function(_label):
                return _function(*items, **named)

        setattr(owner, name, staticmethod(wrapped) if isinstance(original, staticmethod) else wrapped)
        restore.append((owner, name, original))


def _scope_modules(model) -> list:
    """Подмодули блоков — областями профиля через хуки (имя — роль в блоке)."""

    from torch.profiler import record_function

    handles, stack = [], []

    def roles():
        for encoder, layers in (("event", model.event.layers), ("profile", model.profile.layers),
                                ("history", model.history.layers)):
            for layer in layers:
                for role, module in layer.named_children():
                    if role == "self_attn":
                        yield f"{encoder}.out_proj", module.out_proj
                    elif role != "rope":
                        yield f"{encoder}.{role}", module
        yield "event.calendar", model.event.calendar
        yield "event.final_norm", model.event.norm
        yield "history.final_norm", model.history.norm
        if model.recent is not None:
            yield "RecentTypes.proj", model.recent.proj

    for label, module in roles():
        def enter(module, items, _label=label):
            scope = record_function(_label)
            scope.__enter__()
            stack.append(scope)

        def leave(module, items, out):
            stack.pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(leave))

    return handles


def command_breakdown(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    device, backend, attention, context = _setup("gpu")

    batches = _batches("train", args.warmup + args.steps)
    model, optimizer, config = _build(device, backend, attention)
    model.train()

    for batch in batches[:args.warmup]:
        _, _, _, _, norm = _step(model, optimizer, config, batch, device, context)
    torch.cuda.synchronize()

    restore: list = []
    _scope_functions(restore)
    handles = _scope_modules(model)

    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for batch in batches[args.warmup:]:
                with record_function("STEP"):
                    _, _, _, _, norm = _step(model, optimizer, config, batch, device, context)
                    float(norm) if norm is not None else None
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)

    labels = {label for _, label in _labels()} | {"STEP"}
    events = prof.events()

    def scopes_of(event) -> tuple[list[str], int | None]:
        names, backward, parent = [], None, event
        while parent is not None:
            if parent.name in labels:
                names.append(parent.name)
            if backward is None and parent.name.startswith("autograd::engine::evaluate_function"):
                backward = parent.sequence_nr
            parent = parent.cpu_parent
        return names, backward

    forward_scopes: dict[int, list[str]] = {}
    for event in events:
        if event.sequence_nr >= 0 and event.device_type == torch.autograd.DeviceType.CPU:
            names, backward = scopes_of(event)
            if backward is None and event.sequence_nr not in forward_scopes:
                forward_scopes[event.sequence_nr] = names

    forward, backward_time, other = defaultdict(float), defaultdict(float), defaultdict(float)
    total = 0.0
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CPU or not event.kernels:
            continue
        spent = sum(kernel.duration for kernel in event.kernels)
        total += spent
        names, sequence = scopes_of(event)
        if sequence is not None:
            for name in forward_scopes.get(sequence, ["?"]) or ["(вне частей)"]:
                backward_time[name] += spent
            backward_time["(весь backward)"] += spent
        elif names and names != ["STEP"]:
            for name in names:
                if name != "STEP":
                    forward[name] += spent
        else:
            other[event.name] += spent

    steps = args.steps

    def per_step(table):
        return {name: round(value / 1000 / steps, 3) for name, value in sorted(table.items(), key=lambda x: -x[1])}

    record = {
        "label": args.label, "code": _code(), "steps": steps,
        "cuda_ms_per_step": round(total / 1000 / steps, 3),
        "forward_ms": per_step(forward), "backward_ms": per_step(backward_time),
        "outside_ms": dict(list(per_step(other).items())[:15]),
    }

    print(json.dumps(record, ensure_ascii=False, indent=1))
    (root / f"breakdown-{args.label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")


def _labels():
    for module_name, owner_name, name in SCOPES:
        yield None, f"{owner_name + '.' if owner_name else ''}{name}"
    for encoder in ("event", "profile", "history"):
        for role in ("norm1", "norm2", "linear1", "linear2", "dropout", "dropout1", "dropout2", "out_proj",
                     "activation", "qkv", "out", "ffn", "drop"):
            yield None, f"{encoder}.{role}"
    for label in ("event.calendar", "event.final_norm", "history.final_norm", "RecentTypes.proj"):
        yield None, label


# ------------------------------------------------------------
# facts: данные и микрозамеры для решений по кандидатам
# ------------------------------------------------------------


def _cuda_ms(function, repeat: int = 50) -> float:

    import torch

    for _ in range(5):
        function()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeat):
        function()
    end.record()
    torch.cuda.synchronize()
    return round(begin.elapsed_time(end) / repeat, 4)


def command_facts(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import numpy as np
    import torch

    from src.dataset.tokenized import EVENT_TYPE_KEY
    from src.mlm.inputs import micro_batches
    from src.mlm.model import pack
    from src.tokenization.finalvocab import FrozenArtifacts

    device, backend, attention, context = _setup("gpu")
    model, _, config = _build(device, backend, attention)

    # Эпоха 1 целиком: формы micro-batch и номера кусков.
    batches = _batches("train", 10**9)
    shapes = set()
    positions_max, positions = 0, np.zeros(0, dtype=np.int64)
    typed_per_event, typed_offset = defaultdict(int), defaultdict(int)
    event_type_key = FrozenArtifacts.load().key_id(EVENT_TYPE_KEY)
    targets = []

    for batch in batches:
        T = sum(c.n_tokens for c in batch)
        E = sum(c.n_events for c in batch)
        M = sum(int((c.labels != -100).sum()) for c in batch)
        shapes.add((len(batch), T, E, M))
        targets.append(M)
        for client in batch:
            positions = np.union1d(positions, np.unique(client.positions))
            starts = np.repeat(client.event_starts, client.event_lengths)
            typed = (client.key_ids == event_type_key) & (client.positions == 0)
            event = np.repeat(np.arange(client.n_events), client.event_lengths)
            counts = np.bincount(event[typed], minlength=client.n_events)
            for value, number in zip(*np.unique(counts, return_counts=True)):
                typed_per_event[int(value)] += int(number)
            offsets = (np.arange(client.n_tokens) - starts)[typed]
            for value, number in zip(*np.unique(offsets, return_counts=True)):
                typed_offset[int(value)] += int(number)

    record = {
        "micro_batches": len(batches), "unique_shapes_BTEM": len(shapes),
        "unique_T": len({shape[1] for shape in shapes}),
        "targets_per_batch": {"min": int(min(targets)), "median": int(np.median(targets)), "max": int(max(targets))},
        "chunks_per_batch_mean": round(float(np.mean([-(-m // 2048) for m in targets])), 2),
        "piece_positions": {"max": int(positions.max()), "unique": int(positions.size)},
        "event_type_tokens_per_event": dict(sorted(typed_per_event.items())),
        "event_type_offset_in_event": dict(sorted(typed_offset.items())),
    }

    # Таблица синусоид номеров кусков против формулы — на GPU, тем же кодом.
    embedding = model.embedding
    every = torch.arange(int(positions.max()) + 1, device=device)
    table = embedding.pieces_of(every)
    data = pack(batches[len(batches) // 2], device)
    record["piece_table_equal"] = bool(torch.equal(table[data.positions], embedding.pieces_of(data.positions)))

    # Части embed на среднем micro-batch.
    key, value, place = data.key_ids, data.value_ids, data.positions
    marker = torch.zeros_like(key, dtype=torch.bool)
    record["embed_parts_ms"] = {
        "table(key_ids)": _cuda_ms(lambda: embedding.table(key)),
        "table(value_ids)": _cuda_ms(lambda: embedding.table(value)),
        "marker_of (isin)": _cuda_ms(lambda: embedding.marker_of(key)),
        "marker по cu_seqlens": _cuda_ms(lambda: marker.zero_().index_fill_(0, data.events.cu_seqlens[:-1], True)),
        "pieces_of": _cuda_ms(lambda: embedding.pieces_of(place)),
        "pieces по таблице": _cuda_ms(lambda: table[place]),
        "embed с маской": _cuda_ms(lambda: embedding.embed(key, value, place, torch.ones_like(key, dtype=torch.bool))),
        "embed без маски": _cuda_ms(lambda: embedding.embed(key, value, place, None)),
        "tokens": int(key.numel()),
    }
    is_marker = embedding.marker_of(key)
    by_layout = torch.zeros_like(is_marker).index_fill_(0, data.events.cu_seqlens[:-1], True)
    record["marker_by_cu_seqlens_equal"] = bool(torch.equal(is_marker, by_layout))

    # Голова: proj [M,3d]->[M,d] и [M,d]@[V,d]^T на куске 2048, прямой проход.
    weight = embedding.weight
    rows = torch.randn(2048, 3 * weight.shape[1], device=device)
    with context():
        hidden = model.head.proj(rows)
        record["head_chunk_ms"] = {
            "proj": _cuda_ms(lambda: model.head.proj(rows)),
            "logits_matmul": _cuda_ms(lambda: hidden @ weight.t()),
            "topk5": _cuda_ms(lambda: (hidden @ weight.t()).topk(5, dim=-1)),
            "vocab": int(weight.shape[0]),
        }

    print(json.dumps(record, ensure_ascii=False, indent=1))
    (root / "facts.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------
# dump / compare
# ------------------------------------------------------------


def _digest(value) -> str:

    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        return hashlib.sha256(str(data.dtype).encode() + str(tuple(data.shape)).encode()
                              + data.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
    if isinstance(value, np.ndarray):
        return hashlib.sha256(str(value.dtype).encode() + value.tobytes()).hexdigest()[:16]
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def command_dump(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import torch

    from src.mlm import varlen
    from src.mlm.model import pack
    from src.mlm.train import target_losses

    device, backend, attention, context = _setup(args.mode)

    # Порядок вызовов ядра внимания: (строк, групп, dropout) по порядку.
    calls = []
    attend = varlen.attend

    def spy(query, key, value, layout, dropout, _attend=attend):
        calls.append((int(query.shape[0]), len(layout.groups), float(dropout)))
        return _attend(query, key, value, layout, dropout)

    varlen.attend = spy

    if args.max_tokens:
        batches = [b for b in _batches("train", 300)[args.skip:]
                   if sum(c.n_tokens + c.profile_n_tokens for c in b) <= args.max_tokens][:args.count]
    else:
        batches = _batches("train", args.skip + args.count)[args.skip:]

    model, optimizer, config = _build(device, backend, attention)
    model.train()

    state = {"steps": [], "mode": args.mode, "code": _code(), "batches": len(batches)}

    for number, batch in enumerate(batches):

        before = len(calls)

        loss, aux, count, hits, norm = None, None, None, None, None

        data = pack(batch, device)
        with context():
            out = model(data, logits=False)
        if out.count > 0:
            objective = out.loss + (config.usr_aux_weight * out.aux if out.aux is not None else 0)
            (objective * out.count).backward()

        grads = {name: p.grad.detach().clone() if p.grad is not None else None
                 for name, p in model.named_parameters()}

        if out.count > 0:
            torch._foreach_div_([p.grad for p in model.parameters() if p.grad is not None], out.count)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        state["steps"].append({
            "loss": out.loss.detach().cpu(), "aux": out.aux.detach().cpu() if out.aux is not None else None,
            "count": int(out.count), "hits": tuple(int(x) for x in out.hits) if out.hits is not None else None,
            "norm": norm.detach().cpu() if norm is not None else None,
            "grads": {name: g.cpu() if g is not None else None for name, g in grads.items()}
            if number == 0 or args.all_grads else {name: _digest(g) for name, g in grads.items()},
            "attend": calls[before:],
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        })

    state["model"] = {name: value.cpu() for name, value in model.state_dict().items()}
    state["optimizer"] = optimizer.state_dict()

    # Val-проход: полные логиты, потери, top-1/top-5 и разбор целей.
    model.eval()
    evaluation = []
    with torch.no_grad():
        for batch in _batches("val", args.val)[:args.val]:
            data = pack(batch, device)
            with context():
                out = model(data)
            nll, first = target_losses(out.logits, out.targets) if out.count else (None, None)
            from src.mlm.model import hits as full_hits
            evaluation.append({
                "loss": out.loss.detach().cpu(), "count": int(out.count),
                "logits": out.logits.detach().cpu() if out.logits is not None else None,
                "hits": full_hits(out.logits, out.targets, 5) if out.count else None,
                "nll": nll, "first": first,
            })
    state["val"] = evaluation

    torch.save(state, args.out)
    print(json.dumps({"mode": args.mode, "code": state["code"], "steps": len(state["steps"]),
                      "losses": [float(s["loss"]) for s in state["steps"]],
                      "val": [float(e["loss"]) for e in evaluation]}, ensure_ascii=False))


def _equal(left, right, path: str, problems: list) -> None:

    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape:
            problems.append(f"{path}: {left.dtype}{tuple(left.shape)} против {right.dtype}{tuple(right.shape)}")
        elif not torch.equal(left, right):
            delta = (left.double() - right.double()).abs().max().item() if left.is_floating_point() else None
            problems.append(f"{path}: различается, max|Δ| {delta}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype or not np.array_equal(left, right):
            problems.append(f"{path}: массив различается")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            problems.append(f"{path}: ключи {sorted(set(left) ^ set(right), key=str)[:5]}")
        for key in left:
            if key in right:
                _equal(left[key], right[key], f"{path}.{key}", problems)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            problems.append(f"{path}: длина {len(left)} против {len(right)}")
        for index, (one, two) in enumerate(zip(left, right)):
            _equal(one, two, f"{path}[{index}]", problems)
        return
    if left != right:
        problems.append(f"{path}: {str(left)[:80]} против {str(right)[:80]}")


def command_compare(args) -> None:

    import torch

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)

    for name in args.ignore or ():
        for state in (left, right):
            state.pop(name, None)

    problems: list[str] = []
    _equal(left, right, "", problems)

    if problems:
        print(f"РАЗЛИЧИЯ ({len(problems)}):")
        for item in problems[:args.show]:
            print("  " + item)
        raise SystemExit(1)

    print("РАВНЫ побайтно")


# ------------------------------------------------------------
# train
# ------------------------------------------------------------


def command_train(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import torch

    from src.masking.settings import MaskingConfig
    from src.mlm.settings import MlmConfig
    from src.mlm.train import train

    config = MlmConfig()

    if args.cpu:
        torch.set_num_threads(1)
        config = replace(config, device="cpu", attention_backend="sdpa", loader_workers=0,
                         token_budget=args.budget or config.token_budget)

    if args.deterministic:
        _deterministic()

    directory = root / "12_train" / args.label

    started = time.perf_counter()
    train(config, epochs=args.epochs, max_steps=args.steps or None, masking=MaskingConfig(), directory=directory,
          resume=args.resume)
    wall = time.perf_counter() - started

    rows = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    if args.resume:
        last = max(number for number, row in enumerate(rows) if row.get("kind") == "run")
        rows = rows[last:]
    steps = [row for row in rows if row.get("kind") == "step"]
    epochs = [row for row in rows if row.get("kind") == "epoch"]
    measured = steps[args.warmup:]
    seconds = sum(item["seconds"] for item in measured)

    record = {
        "label": args.label, "code": _code(), "wall_s": round(wall, 2), "steps": len(steps),
        "step_ms": round(1000 * seconds / max(1, len(measured)), 1),
        "data_wait_share": round(sum(item["wait_seconds"] for item in measured) / seconds, 4) if seconds else None,
        "tokens_per_s": round(sum(item["tokens"] for item in measured) / seconds, 1) if seconds else None,
        "targets_per_s": round(sum(item["targets"] for item in measured) / seconds, 1) if seconds else None,
        "losses": [item["loss"] for item in steps], "grad_norms": [item["grad_norm"] for item in steps],
        "epochs": epochs,
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
    }

    if torch.cuda.is_available() and not args.cpu:
        record["cuda_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        record["cuda_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)

    (root / f"train-{args.label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({key: value for key, value in record.items() if key not in ("losses", "grad_norms", "epochs")},
                     ensure_ascii=False))


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile")
    profile.add_argument("--root", type=Path, required=True)
    profile.add_argument("--label", required=True)
    profile.add_argument("--warmup", type=int, default=15)
    profile.add_argument("--steps", type=int, default=80)
    profile.add_argument("--parts", action="store_true")
    profile.add_argument("--syncs", action="store_true")
    profile.add_argument("--trace", type=int, default=0)
    profile.add_argument("--variant")

    breakdown = sub.add_parser("breakdown")
    breakdown.add_argument("--root", type=Path, required=True)
    breakdown.add_argument("--label", required=True)
    breakdown.add_argument("--warmup", type=int, default=10)
    breakdown.add_argument("--steps", type=int, default=8)

    valpass = sub.add_parser("valpass")
    valpass.add_argument("--root", type=Path, required=True)
    valpass.add_argument("--label", required=True)
    valpass.add_argument("--checkpoint", type=Path)
    valpass.add_argument("--repeat", type=int, default=3)
    valpass.add_argument("--parts", action="store_true")
    valpass.add_argument("--no-detail", action="store_true")

    facts = sub.add_parser("facts")
    facts.add_argument("--root", type=Path, required=True)

    dump = sub.add_parser("dump")
    dump.add_argument("--root", type=Path, required=True)
    dump.add_argument("--mode", choices=("cpu-sdpa", "cpu-flash", "gpu-det", "gpu"), required=True)
    dump.add_argument("--out", type=Path, required=True)
    dump.add_argument("--count", type=int, default=4)
    dump.add_argument("--skip", type=int, default=0)
    dump.add_argument("--max-tokens", type=int, default=0)
    dump.add_argument("--val", type=int, default=2)
    dump.add_argument("--all-grads", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--ignore", action="append")
    compare.add_argument("--show", type=int, default=30)

    train = sub.add_parser("train")
    train.add_argument("--root", type=Path, required=True)
    train.add_argument("--label", required=True)
    train.add_argument("--steps", type=int, default=60)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--warmup", type=int, default=10)
    train.add_argument("--deterministic", action="store_true")
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--budget", type=int, default=0)
    train.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    {"profile": command_profile, "breakdown": command_breakdown, "valpass": command_valpass,
     "facts": command_facts, "dump": command_dump,
     "compare": command_compare, "train": command_train}[args.command](args)


if __name__ == "__main__":
    main()
