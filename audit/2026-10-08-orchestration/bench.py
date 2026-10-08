"""
Замеры обвязки обучения (src.mlm.train) во временном каталоге --root
с раскладкой data/ (как у audit/2026-10-07-input-runtime/bench.py
setup). Настоящий data/ не трогается. Какой код мерить, задаёт
PYTHONPATH.

    run       настоящий train() на --epochs эпох, выход в
              --root/12_train/<label>; подменами вокруг частей обвязки
              пишется время старта (модель, горизонт, sha256, словарь),
              val (модель, Scores, target_losses, Detail, summary),
              каждой записи чекпойнта (сериализация отдельно от
              os.replace), телеметрии и печати, плюс переносы GPU→CPU
              в val и запись на диск. --deterministic — flash с
              deterministic=True: две версии кода дают побайтно одни
              чекпойнты. --plain — без подмен: только стена.
    detail    validate() на весах чекпойнта: время частей и разбивка;
              sha256 итоговой разбивки — для сверки версий.
    files     размеры и время torch.save / torch.load файлов прогона.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import functools
import hashlib
import json
import resource
import time
from collections import defaultdict
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
    return str(Path(import_module("src.mlm.train").__file__).resolve().parents[2])


def _deterministic() -> None:

    import flash_attn
    import torch

    flash_attn.flash_attn_varlen_func = functools.partial(flash_attn.flash_attn_varlen_func, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _io() -> dict:
    """Байты чтения и записи процесса по /proc/self/io; без него (ядро WSL) — None."""

    if not Path("/proc/self/io").exists():
        return {"read": None, "write": None}

    values = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        name, value = line.split(":")
        values[name] = int(value)
    return {"read": values["read_bytes"], "write": values["write_bytes"]}


class Clock:
    """Время по подписям: сумма, число вызовов; CUDA синхронизируется по желанию."""

    def __init__(self, sync: bool):
        self.sync = sync
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.marks: list[tuple[str, float]] = []

    def now(self) -> float:
        if self.sync:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return time.perf_counter()

    def wrap(self, owner, name: str, label: str, sync: bool | None = None) -> None:

        original = getattr(owner, name)
        clock = self

        @functools.wraps(original)
        def timed(*items, **named):
            began = clock.now() if (clock.sync if sync is None else sync) else time.perf_counter()
            try:
                return original(*items, **named)
            finally:
                ended = clock.now() if (clock.sync if sync is None else sync) else time.perf_counter()
                clock.seconds[label] += ended - began
                clock.calls[label] += 1

        setattr(owner, name, timed)

    def mark(self, label: str) -> None:
        """Смена вида прохода (train/val) — по часам стены, как time в телеметрии."""
        if not self.marks or self.marks[-1][0] != label:
            self.marks.append((label, time.time()))


def _instrument(clock: Clock, directory: Path) -> None:
    """Подмены вокруг частей обвязки. Результат обучения они не меняют."""

    import torch

    import src.mlm.inputs as inputs
    import src.mlm.model as model_module
    import src.mlm.train as train_module
    import src.tokenization.finalvocab as finalvocab

    # Старт.
    clock.wrap(model_module, "load_model", "startup.load_model")
    clock.wrap(train_module, "attach_recent", "startup.attach_recent")
    clock.wrap(train_module, "horizon", "startup.horizon")
    clock.wrap(train_module, "data_record", "startup.data_record")
    clock.wrap(train_module, "file_digest", "startup.file_digest (sha256)", sync=False)
    clock.wrap(train_module, "backbone_record", "startup.backbone_record")
    clock.wrap(train_module, "limit_cuda_memory", "startup.limit_cuda_memory (поднимает CUDA)")
    clock.wrap(inputs.Source, "__init__", "source.open", sync=False)

    # Первая порция каждого Prefetch: запуск процесса подготовки и
    # первая группа строк.
    prefetch = inputs.Prefetch.clients

    def first_client(self):
        began = time.perf_counter()
        items = prefetch(self)
        for number, item in enumerate(items):
            if number == 0:
                clock.seconds[f"loader.first_client workers={self.workers}"] += time.perf_counter() - began
                clock.calls[f"loader.first_client workers={self.workers}"] += 1
            yield item

    inputs.Prefetch.clients = first_client
    clock.wrap(finalvocab, "load_final_vocab", "val.load_final_vocab", sync=False)

    # Сколько групп строк читает Source.sizes.
    sizes = inputs.Source.sizes

    def counted_sizes(self, *items, **named):
        for size in sizes(self, *items, **named):
            clock.calls["startup.sizes clients"] += 1
            yield size

    inputs.Source.sizes = counted_sizes

    # val: целиком и по частям; синхронизация CUDA на границах частей.
    clock.wrap(train_module, "validate", "val.validate")
    clock.wrap(train_module.Scores, "add", "val+train.Scores.add")
    clock.wrap(train_module.Detail, "add", "val.Detail.add")
    clock.wrap(train_module, "target_losses", "val.target_losses")
    clock.wrap(train_module.Detail, "summary", "val.Detail.summary")

    forward = model_module.Model.forward

    def timed_forward(self, data, logits=True):
        if self.training:
            clock.mark("train.forward")
            return forward(self, data, logits=logits)
        clock.mark("val.forward")
        began = clock.now()
        out = forward(self, data, logits=logits)
        clock.seconds["val.model"] += clock.now() - began
        clock.calls["val.model"] += 1
        return out

    model_module.Model.forward = timed_forward

    # Переносы GPU→CPU в val: .cpu() и .item() под validate.
    inside = {"val": False}
    validate = train_module.validate

    def flagged(*items, **named):
        inside["val"] = True
        try:
            return validate(*items, **named)
        finally:
            inside["val"] = False

    train_module.validate = flagged

    for method in ("cpu", "item", "tolist"):
        original = getattr(torch.Tensor, method)

        def moved(self, *items, _original=original, _method=method, **named):
            if inside["val"] and self.device.type == "cuda":
                clock.calls[f"val.D2H {_method}"] += 1
                clock.seconds[f"val.D2H bytes {_method}"] += self.numel() * self.element_size()
            return _original(self, *items, **named)

        setattr(torch.Tensor, method, moved)

    # Чекпойнты: сериализация и os.replace по видам файла.
    save = torch.save
    replace = os.replace

    def kind(path) -> str:
        name = Path(path).name.removesuffix(".tmp")
        return "epoch_weights" if name.startswith("epoch_") else name.removesuffix(".pt")

    def timed_save(state, path, *items, **named):
        began = time.perf_counter()
        save(state, path, *items, **named)
        clock.seconds[f"save.{kind(path)}.torch_save"] += time.perf_counter() - began
        clock.calls[f"save.{kind(path)}.torch_save"] += 1
        clock.seconds[f"save.{kind(path)}.mib"] = Path(path).stat().st_size / 2**20
        clock.seconds["save.written_mib"] += Path(path).stat().st_size / 2**20

    def timed_replace(source, target, *items, **named):
        began = time.perf_counter()
        replace(source, target, *items, **named)
        clock.seconds[f"save.{kind(target)}.os_replace"] += time.perf_counter() - began

    torch.save = timed_save
    os.replace = timed_replace

    clock.wrap(train_module, "save_checkpoint", "save.save_checkpoint total", sync=False)
    clock.wrap(train_module, "append_telemetry", "telemetry.append", sync=False)


def command_run(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import torch

    from src.masking.settings import MaskingConfig
    from src.mlm.settings import MlmConfig
    import src.mlm.train as train_module

    if args.deterministic:
        _deterministic()

    directory = root / "12_train" / args.label
    clock = Clock(sync=not args.plain)

    if not args.plain:
        _instrument(clock, directory)

    # Печать — отдельной подменой print модуля обучения.
    printed = {"seconds": 0.0, "lines": 0}
    original_print = print

    def timed_print(*items, **named):
        began = time.perf_counter()
        original_print(*items, **named)
        printed["seconds"] += time.perf_counter() - began
        printed["lines"] += 1

    train_module.print = timed_print

    io_before = _io()
    cpu_before = os.times()
    started = time.perf_counter()
    started_wall = time.time()

    from dataclasses import replace

    config = MlmConfig()
    if args.patience is not None:
        config = replace(config, early_stopping_patience=args.patience, early_stopping_min_delta=args.min_delta)

    result = train_module.train(config, epochs=args.epochs, max_steps=args.steps, masking=MaskingConfig(),
                                directory=directory, resume=args.resume)

    wall = time.perf_counter() - started
    cpu_after = os.times()
    io_after = _io()

    rows = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    if args.resume:
        rows = rows[max(number for number, row in enumerate(rows) if row.get("kind") == "run"):]
    epochs = [row for row in rows if row.get("kind") == "epoch"]

    # Старт — до первого прохода обучения.
    starts = [stamp for label, stamp in clock.marks if label == "train.forward"]
    first = starts[0] if starts else None

    # Пауза между эпохами: от строки эпохи (val уже прошла, чекпойнтов
    # ещё нет) до первого прохода следующей эпохи — запись файлов,
    # снимок и запуск подготовки следующей эпохи.
    pauses = [round(start - row["time"], 3) for row, start in zip(epochs, starts[1:])]

    record = {
        "label": args.label, "code": _code(), "plain": args.plain, "deterministic": args.deterministic,
        "result": {key: value for key, value in result.items() if key != "model"},
        "wall_s": round(wall, 2),
        "startup_s": round(first - started_wall, 3) if first else None,
        "checkpoint_pauses_s": pauses,
        "epochs": [{key: row[key] for key in ("epoch", "train_seconds", "val_seconds", "data_wait_seconds",
                                              "train_loss", "val_loss", "steps")} for row in epochs],
        "epoch_ends": [round(row["time"], 3) for row in epochs],
        "parts": {label: round(value, 4) for label, value in sorted(clock.seconds.items())},
        "calls": dict(sorted(clock.calls.items())),
        "print": {"seconds": round(printed["seconds"], 4), "lines": printed["lines"]},
        "cpu_user_s": round(cpu_after.user - cpu_before.user, 2),
        "cpu_system_s": round(cpu_after.system - cpu_before.system, 2),
        "children_cpu_s": round(cpu_after.children_user + cpu_after.children_system
                                - cpu_before.children_user - cpu_before.children_system, 2),
        "read_gib": round((io_after["read"] - io_before["read"]) / 2**30, 3) if io_before["read"] is not None else None,
        "written_gib": round((io_after["write"] - io_before["write"]) / 2**30, 3) if io_before["write"] is not None else None,
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
        "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }

    (root / f"orchestration-{args.label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=1))


def command_detail(args) -> None:
    """validate() на весах чекпойнта с разбивкой времени по частям."""

    root = _outside_data(args.root)
    _redirect(root)

    import torch

    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Prefetch, Source
    from src.mlm.model import load_model
    from src.mlm.settings import MlmConfig
    import src.mlm.train as train_module

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = MlmConfig.from_dict(state["config"])
    device = torch.device("cuda")

    # Как в train(): лимит памяти и порог сборки аллокатора.
    if args.limit:
        train_module.limit_cuda_memory(device)

    model = load_model(seed=config.seed, events_per_chunk=config.events_per_chunk,
                       label_smoothing=config.label_smoothing, device=device, attention_backend="flash")
    if config.usr_aux_weight > 0.0:
        train_module.attach_recent(model, config)
    model.load_state_dict(state["model_state_dict"])

    clock = Clock(sync=not args.plain)
    if not args.plain:
        _instrument(clock, root)

    source = Source("val", masking=MaskingConfig.from_dict(state["masking"]))

    rounds = []
    for _ in range(args.repeat):
        torch.cuda.synchronize()
        began = time.perf_counter()
        scores = train_module.validate(model, Prefetch(source, args.workers), device, config.token_budget)
        torch.cuda.synchronize()
        rounds.append(round(time.perf_counter() - began, 3))
    seconds = rounds[-1]

    detail = json.dumps(scores.detail, sort_keys=True)

    record = {
        "label": args.label, "code": _code(), "plain": args.plain, "workers": args.workers,
        "seconds": seconds, "rounds": rounds, "limit": args.limit, "summary": scores.summary(),
        "allocator": {key: torch.cuda.memory_stats()[key] for key in
                      ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms")},
        "scores": scores.as_dict(), "detail_sha256": hashlib.sha256(detail.encode()).hexdigest()[:16],
        "parts": {label: round(value, 4) for label, value in sorted(clock.seconds.items())},
        "calls": dict(sorted(clock.calls.items())),
    }

    (root / f"detail-{args.label}.json").write_text(json.dumps(dict(record, detail=scores.detail),
                                                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


def command_snapshot(args) -> None:
    """
    Снимок состояния и запись чекпойнта на модели и AdamW из чекпойнта
    на GPU: ссылки или копии, D2H при сохранении, сериализация против
    записи, os.replace и os.link.
    """

    import io

    import torch

    root = _outside_data(args.root)
    _redirect(root)

    from src.mlm.model import load_model
    from src.mlm.settings import MlmConfig
    import src.mlm.train as train_module

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = MlmConfig.from_dict(state["config"])
    device = torch.device("cuda")

    model = load_model(seed=config.seed, events_per_chunk=config.events_per_chunk,
                       label_smoothing=config.label_smoothing, device=device, attention_backend="flash")
    if config.usr_aux_weight > 0.0:
        train_module.attach_recent(model, config)
    model.load_state_dict(state["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
                                  fused=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])

    record = defaultdict(list)
    directory = root / "snapshot-bench"
    directory.mkdir(exist_ok=True)

    for _ in range(args.repeat):
        torch.cuda.synchronize()
        began = time.perf_counter()
        weights = model.state_dict()
        record["model.state_dict_s"].append(time.perf_counter() - began)
        began = time.perf_counter()
        adam = optimizer.state_dict()
        record["optimizer.state_dict_s"].append(time.perf_counter() - began)
        began = time.perf_counter()
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
        record["rng_s"].append(time.perf_counter() - began)

        shared = all(weights[name].data_ptr() == value.data_ptr() for name, value in model.named_parameters()
                     if name in weights)
        record["state_dict_shares_parameters"].append(shared)

        current = dict(state, model_state_dict=weights, optimizer_state_dict=adam,
                       rng_state=rng[0], cuda_rng_state=rng[1])

        buffer = io.BytesIO()
        began = time.perf_counter()
        torch.save(current, buffer)
        record["torch_save_to_memory_s (D2H + сериализация)"].append(time.perf_counter() - began)
        record["mib"].append(buffer.tell() / 2**20)

        temporary = directory / "checkpoint.pt.tmp"
        began = time.perf_counter()
        torch.save(current, temporary)
        record["torch_save_to_file_s"].append(time.perf_counter() - began)

        began = time.perf_counter()
        with open(directory / "raw.bin", "wb") as handle:
            handle.write(buffer.getbuffer())
            handle.flush()
        record["write_bytes_only_s"].append(time.perf_counter() - began)

        began = time.perf_counter()
        os.replace(temporary, directory / "checkpoint.pt")
        record["os_replace_s"].append(time.perf_counter() - began)

        (directory / "best.pt").unlink(missing_ok=True)
        began = time.perf_counter()
        os.link(directory / "checkpoint.pt", directory / "best.pt")
        record["os_link_s"].append(time.perf_counter() - began)

        began = time.perf_counter()
        torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        record["torch_load_s"].append(time.perf_counter() - began)

    summary = {key: (round(min(values), 4) if isinstance(values[0], float) else values[0])
               for key, values in record.items()}
    summary["all"] = {key: [round(value, 4) if isinstance(value, float) else value for value in values]
                      for key, values in record.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def command_files(args) -> None:
    """Размеры файлов прогона; время torch.save в память и на диск и torch.load."""

    import io

    import torch

    directory = _outside_data(args.directory)
    record = {}

    for path in sorted(directory.rglob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        began = time.perf_counter()
        torch.save(state, io.BytesIO())
        serialize = time.perf_counter() - began
        began = time.perf_counter()
        torch.load(path, map_location="cpu", weights_only=True)
        load = time.perf_counter() - began
        record[str(path.relative_to(directory))] = {
            "mib": round(path.stat().st_size / 2**20, 2), "serialize_in_memory_s": round(serialize, 3),
            "load_s": round(load, 3), "keys": sorted(state)}

    print(json.dumps(record, ensure_ascii=False, indent=1))


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--epochs", type=int, default=2)
    run.add_argument("--deterministic", action="store_true")
    run.add_argument("--plain", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--patience", type=int)
    run.add_argument("--steps", type=int, help="--max-steps обучения")
    run.add_argument("--min-delta", type=float, default=0.0)

    detail = sub.add_parser("detail")
    detail.add_argument("--root", type=Path, required=True)
    detail.add_argument("--checkpoint", type=Path, required=True)
    detail.add_argument("--label", required=True)
    detail.add_argument("--workers", type=int, default=1)
    detail.add_argument("--plain", action="store_true")
    detail.add_argument("--limit", action="store_true")
    detail.add_argument("--repeat", type=int, default=1)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--root", type=Path, required=True)
    snapshot.add_argument("--checkpoint", type=Path, required=True)
    snapshot.add_argument("--repeat", type=int, default=5)

    files = sub.add_parser("files")
    files.add_argument("directory", type=Path)

    args = parser.parse_args()

    {"run": command_run, "detail": command_detail, "snapshot": command_snapshot,
     "files": command_files}[args.command](args)


if __name__ == "__main__":
    main()
