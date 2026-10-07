"""
Замеры подготовки входа модели (05_dataset → маска → Client → micro-batch →
pack → GPU). Всё в раскладке data/ во временном каталоге --root; настоящий
data/ не трогается. Какой код мерить, задаёт PYTHONPATH.

    setup    --root: val (маленькая выгрузка окна val → 02 → 04 словарём
             train → 05), веса 06 и 07 по seed конвейера; train-набор и
             словарь подключаются ссылками;
    loader   проход Prefetch(Source train эпохи).clients() через
             micro_batches без модели: время, пик RSS; --timers — разбивка
             по компонентам (только workers 0: всё в этом процессе);
    digest   отпечатки каждого клиента (маска, метки, reason, время,
             календарь, анкета) и границ micro-batch'ей — для сверки двух
             версий кода и разных workers;
    pack     pack первых micro-batch'ей на устройство: время с
             синхронизацией CUDA, доля VarlenLayout.build, отпечатки
             тензоров и раскладок;
    train    настоящий train() с --max-steps в --root/12_train/<label>:
             телеметрия шагов (loss, норма градиента, ожидание данных,
             время), загрузка GPU по nvidia-smi, пик памяти CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
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


def _rss() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


# ------------------------------------------------------------
# setup
# ------------------------------------------------------------


def command_setup(args) -> None:

    root = _outside_data(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for name, target in (("03_vocab", args.vocab), ("05_dataset/train", args.train)):
        link = root / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(Path(target).resolve())

    _redirect(root)

    from src.generator import emit
    from src.generator.config import DATASETS
    from src.preprocessing.canonical.build import build_group as preprocess
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir

    val = DATASETS["val"]

    started = time.perf_counter()

    if not (raw_group_dir("val") / "profile.parquet").exists():
        emit.generate_dataset(
            total_clients=args.val_clients, out_dir=raw_group_dir("val"), seed=val.seed,
            history_start=val.history_start, history_end=val.history_end,
            registration_end=val.registration_end, workers=args.workers, quiet=True,
        )

    print(f"val RAW: {time.perf_counter() - started:.1f} с")

    preprocess(raw_group_dir("val"), group_dir("val"), PreprocessingConfig.load(None), "val")

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    artifacts = FrozenArtifacts.load()

    print(json.dumps(encode_group(artifacts, "val", TokenizerConfig.load(None))["counts"]))

    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig

    print(json.dumps(build_dataset(artifacts, "val", DatasetConfig.load(None))["counts"]))

    from src.embedding.build import build_group as build_embedding
    from src.embedding.settings import EmbeddingConfig

    with redirect_stdout(io.StringIO()):
        build_embedding("train", EmbeddingConfig.load(None))

    from src.event.settings import EventConfig
    from src.history.settings import HistoryConfig
    from src.mlm.backbone import init_backbone
    from src.profile.settings import ProfileConfig

    report = init_backbone(EventConfig.load(None), ProfileConfig.load(None), HistoryConfig.load(None))

    print(f"backbone: {report['encoders']}")
    print(f"готово за {time.perf_counter() - started:.1f} с")


# ------------------------------------------------------------
# loader
# ------------------------------------------------------------


TIMED = [
    ("src.temporal.samples", "TemporalGroup", "row_group"),
    ("src.temporal.samples", "TemporalGroup", "_with_time"),
    ("src.temporal.samples", None, "time_log"),
    ("src.temporal.samples", None, "profile_time_log"),
    ("src.temporal.samples", None, "check"),
    ("src.temporal.samples", None, "check_profile"),
    ("src.mlm.inputs", "Source", "batch"),
    ("src.mlm.inputs", "Source", "_mask"),
    ("src.mlm.inputs", "Source", "_client"),
    ("src.mlm.inputs", "Source", "sizes"),
    ("src.mlm.inputs", None, "choose"),
    ("src.mlm.inputs", None, "apply"),
    ("src.mlm.inputs", None, "_check"),
    ("src.masking.choose", None, "values_of"),
    ("src.masking.choose", None, "value_chance"),
    ("src.masking.choose", None, "value_chances"),
    ("src.masking.choose", None, "_parse"),
    ("src.mlm.inputs", None, "apply_selection"),
    ("src.temporal.samples", None, "time_logs"),
    ("src.temporal.samples", None, "profile_time_logs"),
    ("src.temporal.samples", None, "checks_hold"),
]


def _source(group: str, epoch: int):

    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Source
    from src.mlm.train import for_epoch

    masking = MaskingConfig()

    return Source(group, masking=for_epoch(masking, epoch) if group == "train" else masking)


def _patch_timers(spent, calls):

    restore = []

    def timer(label, function):
        def inner(*items, **named):
            start = time.perf_counter()
            try:
                return function(*items, **named)
            finally:
                spent[label] += time.perf_counter() - start
                calls[label] += 1
        return inner

    for module_name, owner_name, name in TIMED:
        module = import_module(module_name)
        owner = getattr(module, owner_name) if owner_name else module
        original = owner.__dict__.get(name) if owner_name else getattr(module, name, None)
        if original is None:
            continue
        setattr(owner, name, timer(f"{owner_name + '.' if owner_name else ''}{name}", original))
        restore.append((owner, name, original))

    return restore


def command_loader(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    from src.mlm.inputs import Prefetch, micro_batches
    from src.mlm.settings import MlmConfig

    if args.ahead is not None:
        Prefetch.AHEAD = args.ahead

    spent: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    restore = _patch_timers(spent, calls) if args.timers else []

    budget = MlmConfig().token_budget

    started = time.perf_counter()
    cpu = time.process_time()

    source = _source(args.group, args.epoch)

    batches = clients = tokens = 0
    waits: list[float] = []
    ready = time.perf_counter()
    first = None

    try:
        for batch in micro_batches(Prefetch(source, args.workers).clients(), budget):
            now = time.perf_counter()
            waits.append(now - ready)
            if first is None:
                first = now - started
            batches += 1
            clients += len(batch)
            tokens += sum(client.n_tokens + client.profile_n_tokens for client in batch)
            if args.limit and batches >= args.limit:
                break
            ready = time.perf_counter()
    finally:
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)

    wall = time.perf_counter() - started

    record = {
        "label": args.label, "code": str(Path(import_module("src.mlm.inputs").__file__).resolve().parents[2]),
        "group": args.group, "workers": args.workers, "ahead": args.ahead, "timers_on": bool(args.timers),
        "wall_s": round(wall, 2), "cpu_s_main": round(time.process_time() - cpu, 2), "peak_rss_gib_main": _rss(),
        "first_batch_s": round(first or 0.0, 2), "micro_batches": batches, "clients": clients, "tokens": tokens,
        "ms_per_micro_batch": round(1000 * wall / max(1, batches), 1),
        "steady_ms_per_micro_batch": round(1000 * sum(waits[5:]) / max(1, len(waits) - 5), 1),
        "timers": {label: [round(spent[label], 2), calls[label]] for label in spent},
    }

    print(json.dumps({key: value for key, value in record.items() if key != "timers"}, ensure_ascii=False))
    for label, (seconds, count) in sorted(record["timers"].items(), key=lambda item: -item[1][0]):
        print(f"  {label:34} {seconds:8.2f} с {100 * seconds / wall:5.1f}%  вызовов {count}")

    if args.out:
        Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------
# digest
# ------------------------------------------------------------


CLIENT_ARRAYS = ("key_ids", "value_ids", "positions", "labels", "event_starts", "event_lengths",
                 "event_time_log", "calendar", "profile_key_ids", "profile_value_ids", "profile_positions",
                 "profile_time_log")


def client_digest(client) -> dict:

    import numpy as np

    parts = {}

    for name in CLIENT_ARRAYS:
        value = np.ascontiguousarray(getattr(client, name))
        parts[name] = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                                     + value.tobytes()).hexdigest()[:16]

    parts["reason"] = hashlib.sha256("\x1f".join(client.reason).encode()).hexdigest()[:16]
    parts["event_time"] = hashlib.sha256(repr(list(client.event_time)).encode()).hexdigest()[:16]
    parts["ids"] = f"{client.batch_index}:{client.client_id}"

    return parts


def command_digest(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    from src.mlm.inputs import Prefetch, micro_batches
    from src.mlm.settings import MlmConfig

    if args.ahead is not None:
        Prefetch.AHEAD = args.ahead

    budget = MlmConfig().token_budget

    fields: dict[str, hashlib._Hash] = defaultdict(hashlib.sha256)
    boundaries = hashlib.sha256()
    batches = clients = 0

    for batch in micro_batches(Prefetch(_source(args.group, args.epoch), args.workers).clients(), budget):
        batches += 1
        boundaries.update(("|".join(client.client_id for client in batch) + "\n").encode())
        for client in batch:
            clients += 1
            for name, value in client_digest(client).items():
                fields[name].update(value.encode())

    record = {"group": args.group, "epoch": args.epoch, "workers": args.workers, "clients": clients,
              "micro_batches": batches, "boundaries": boundaries.hexdigest()[:16],
              "fields": {name: digest.hexdigest()[:16] for name, digest in sorted(fields.items())}}

    record["all"] = hashlib.sha256(json.dumps(record["fields"], sort_keys=True).encode()
                                   + record["boundaries"].encode()).hexdigest()[:16]

    print(json.dumps(record, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------
# pack
# ------------------------------------------------------------


def _tensor_digest(value) -> str:

    import torch

    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        return hashlib.sha256(str(data.dtype).encode() + str(tuple(data.shape)).encode()
                              + data.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def _layout_digest(layout) -> dict:

    import numpy as np

    out = {"cu_seqlens": _tensor_digest(layout.cu_seqlens), "lengths": _tensor_digest(layout.lengths),
           "groups": [(group.rows, group.max_seqlen, _tensor_digest(group.cu_seqlens)) for group in layout.groups]}

    if getattr(layout, "buckets", None) is not None:
        out["buckets"] = [(_tensor_digest(bucket.segments), _tensor_digest(bucket.index), _tensor_digest(bucket.mask))
                          for bucket in layout.buckets]
        out["bucket_of"] = hashlib.sha256(np.asarray(layout.bucket_of).tobytes()).hexdigest()[:16]
        out["row_of"] = hashlib.sha256(np.asarray(layout.row_of).tobytes()).hexdigest()[:16]

    return out


def command_pack(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import torch

    from src.mlm import model as model_module
    from src.mlm.inputs import Prefetch, micro_batches
    from src.mlm.settings import MlmConfig
    from src.mlm.varlen import VarlenLayout

    device = torch.device(args.device)
    budget = MlmConfig().token_budget

    built = {"seconds": 0.0, "calls": 0}
    original = VarlenLayout.build

    def timed_build(*items, **named):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            return original(*items, **named)
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize()
            built["seconds"] += time.perf_counter() - start
            built["calls"] += 1

    VarlenLayout.build = staticmethod(timed_build)

    digests = []
    pack_seconds = 0.0
    megabytes = 0.0
    batches = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)

    try:
        for batch in micro_batches(Prefetch(_source(args.group, args.epoch), 0).clients(), budget):

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            packed = model_module.pack(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            pack_seconds += time.perf_counter() - start
            batches += 1

            tensors = {name: value for name, value in vars(packed).items() if isinstance(value, torch.Tensor)}
            megabytes += sum(value.numel() * value.element_size() for value in tensors.values()) / 2**20

            if args.digest:
                digests.append({
                    "tensors": {name: _tensor_digest(value) for name, value in sorted(tensors.items())},
                    "layouts": {name: _layout_digest(getattr(packed, name)) for name in ("events", "profiles", "history")},
                    "clients": packed.clients,
                })

            del packed

            if args.limit and batches >= args.limit:
                break
    finally:
        VarlenLayout.build = staticmethod(original)

    record = {"label": args.label, "device": str(device), "micro_batches": batches,
              "pack_s": round(pack_seconds, 3), "pack_ms_per_batch": round(1000 * pack_seconds / max(1, batches), 2),
              "layout_build_s": round(built["seconds"], 3), "layout_calls": built["calls"],
              "tensor_mib_per_batch": round(megabytes / max(1, batches), 2)}

    if device.type == "cuda":
        record["cuda_peak_mib"] = round((torch.cuda.max_memory_allocated(device) - before) / 2**20, 1)

    if args.digest:
        record["digest"] = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:16]

    print(json.dumps(record, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(json.dumps(dict(record, batches=digests), ensure_ascii=False, indent=1),
                                  encoding="utf-8")


# ------------------------------------------------------------
# train
# ------------------------------------------------------------


class _GpuSampler:
    """nvidia-smi раз в 200 мс: загрузка GPU и занятая память."""

    def __init__(self):
        self.samples: list[tuple[float, int, int]] = []
        self._process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for line in self._process.stdout:
            try:
                util, memory = (int(item.strip()) for item in line.split(","))
            except ValueError:
                continue
            self.samples.append((time.time(), util, memory))

    def stop(self):
        self._process.terminate()
        self._process.wait()


def command_train(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import torch

    from dataclasses import replace

    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Prefetch
    from src.mlm.settings import MlmConfig
    from src.mlm.train import train

    if args.ahead is not None:
        Prefetch.AHEAD = args.ahead

    config = MlmConfig()

    if args.workers is not None:
        config = replace(config, loader_workers=args.workers)

    if args.device is not None:
        config = replace(config, device=args.device)

    directory = root / "12_train" / args.label

    sampler = _GpuSampler()
    started_at = time.time()
    started = time.perf_counter()

    try:
        train(config, epochs=1, max_steps=args.steps or None, masking=MaskingConfig(), directory=directory)
    finally:
        sampler.stop()

    wall = time.perf_counter() - started

    rows = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()] \
        if (directory / "telemetry.jsonl").exists() else []
    steps = [row for row in rows if row.get("kind") == "step"]
    epochs = [row for row in rows if row.get("kind") != "step"]

    measured = steps[args.warmup:]
    seconds = sum(item["seconds"] for item in measured)
    waited = sum(item["wait_seconds"] for item in measured)
    tokens = sum(item["tokens"] for item in measured)

    window = [sample for sample in sampler.samples
              if measured and sample[0] >= measured[0]["time"] - measured[0]["seconds"]]

    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False) \
        if (directory / "checkpoint.pt").exists() else None

    weights = hashlib.sha256()
    if state is not None:
        for name, value in sorted(state["model_state_dict"].items()):
            weights.update(name.encode() + value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

    record = {
        "label": args.label, "code": str(Path(import_module("src.mlm.inputs").__file__).resolve().parents[2]),
        "workers": config.loader_workers, "ahead": Prefetch.AHEAD, "steps": len(steps), "warmup": args.warmup,
        "wall_s": round(wall, 2), "measured_s": round(seconds, 2),
        "data_wait_share": round(waited / seconds, 4) if seconds else None,
        "step_ms": round(1000 * seconds / max(1, len(measured)), 1),
        "tokens_per_s": round(tokens / seconds, 1) if seconds else None,
        "gpu_util_mean": round(sum(sample[1] for sample in window) / len(window), 1) if window else None,
        "gpu_memory_used_max_mib": max((sample[2] for sample in window), default=None),
        "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else None,
        "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3) if torch.cuda.is_available() else None,
        "peak_rss_gib_main": _rss(),
        "losses": [round(item["loss"], 6) for item in steps],
        "grad_norms": [round(item["grad_norm"], 6) for item in steps],
        "targets": [item["targets"] for item in steps],
        "weights_sha256": weights.hexdigest()[:16] if state is not None else None,
        "epoch": {key: epochs[-1].get(key) for key in ("train_seconds", "data_wait_seconds", "val_seconds",
                                                       "train_loss", "val_loss", "steps", "grad_norm_mean",
                                                       "cuda_peak_allocated_gib", "cuda_peak_reserved_gib")}
        if epochs and epochs[-1].get("kind") == "epoch" else None,
        "val": state["history"][-1]["val"] if state is not None and state.get("history") else None,
        "val_detail_sha256": hashlib.sha256(json.dumps(state["history"][-1]["val_detail"], sort_keys=True,
                                                       default=str).encode()).hexdigest()[:16]
        if state is not None and state.get("history") else None,
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
    }

    print(json.dumps({key: value for key, value in record.items() if key not in ("losses", "grad_norms", "targets")},
                     ensure_ascii=False))

    (root / f"train-{args.label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


def command_validate(args) -> None:
    """
    validate() одной и той же модели (веса из чекпойнта) по val: метрики
    и разбивка по целям должны совпасть у двух версий подготовки входа.
    """

    root = _outside_data(args.root)
    _redirect(root)

    import torch

    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Prefetch, Source
    from src.mlm.model import load_model
    from src.mlm.settings import MlmConfig
    from src.mlm.train import attach_recent, validate

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = MlmConfig.from_dict(state["config"])
    device = torch.device("cuda")

    model = load_model(seed=config.seed, events_per_chunk=config.events_per_chunk,
                       label_smoothing=config.label_smoothing, device=device, attention_backend="flash")
    if config.usr_aux_weight > 0.0:
        attach_recent(model, config)
    model.load_state_dict(state["model_state_dict"])

    started = time.perf_counter()
    scores = validate(model, Prefetch(Source("val", masking=MaskingConfig.from_dict(state["masking"])), args.workers),
                      device, config.token_budget)
    seconds = time.perf_counter() - started

    record = {"label": args.label, "seconds": round(seconds, 2), "summary": scores.summary(),
              "detail_sha256": hashlib.sha256(json.dumps(scores.detail, sort_keys=True, default=str).encode()).hexdigest()[:16]}

    print(json.dumps(record, ensure_ascii=False, default=str))

    (root / f"validate-{args.label}.json").write_text(
        json.dumps(dict(record, detail=scores.detail), ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup")
    setup.add_argument("--root", type=Path, required=True)
    setup.add_argument("--vocab", type=Path, required=True)
    setup.add_argument("--train", type=Path, required=True)
    setup.add_argument("--val-clients", type=int, default=300)
    setup.add_argument("--workers", type=int, default=6)

    for name in ("loader", "digest", "pack"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--group", default="train")
        item.add_argument("--epoch", type=int, default=1)
        item.add_argument("--workers", type=int, default=0)
        item.add_argument("--ahead", type=int, default=None)
        item.add_argument("--limit", type=int, default=None)
        item.add_argument("--label", default="run")
        item.add_argument("--out", type=Path, default=None)
        if name == "loader":
            item.add_argument("--timers", action="store_true")
        if name == "pack":
            item.add_argument("--device", default="cuda")
            item.add_argument("--digest", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--root", type=Path, required=True)
    train.add_argument("--label", required=True)
    train.add_argument("--steps", type=int, default=60)
    train.add_argument("--warmup", type=int, default=10)
    train.add_argument("--workers", type=int, default=None)
    train.add_argument("--ahead", type=int, default=None)
    train.add_argument("--device", default=None)

    check = sub.add_parser("validate")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--label", required=True)
    check.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    {"setup": command_setup, "loader": command_loader, "digest": command_digest, "pack": command_pack,
     "train": command_train, "validate": command_validate}[args.command](args)


if __name__ == "__main__":
    main()
