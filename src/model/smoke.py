from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.preprocessing.config import DATASET_NAMES
from src.tokenizer.artifacts import Tokenizer
from src.tokenizer.config import artifacts_dir as tokenizer_artifacts_dir
from src.tokenizer.config import tokenized_dir, vocab_dir
from src.tokenizer.dataset import TokenizedDataset, collate

from .batching import events_from_batch, profiles_from_batch
from .config import config_from_tokenizer
from .encoders import build_encoders


# ============================================================
# ИДЕЯ
# ============================================================
#
# Замер на настоящих данных: сто клиентов, по одному примеру на
# клиента, истории не обрезаются.
#
# Загрузка данных и подготовка batch измеряются отдельно от
# forward, иначе замер энкодера превращается в замер parquet.
# На CUDA устройство синхронизируется до и после каждого
# измеряемого участка: без этого время получается фиктивным.
# ============================================================


TARGET_CLIENTS = 100


@dataclass(frozen=True)
class Selection:
    dataset: str
    index: int
    client_id: int
    cutoff: object


def select_examples(
    root: Path,
    vocab: Path,
    cutoff: str = "last",
    max_clients: int | None = None,
    datasets: tuple[str, ...] = DATASET_NAMES,
) -> tuple[list[Selection], str | None]:
    """
    По одному примеру на клиента из всех датасетов.

    Клиент встречается в нескольких датасетах и на нескольких
    cutoff; берётся ровно один пример, дублей нет.
    """

    if cutoff not in ("last", "first"):
        raise ValueError("cutoff должен быть last или first")

    best: dict[int, Selection] = {}

    for name in datasets:

        data = TokenizedDataset(root, name, vocab_dir=vocab)

        client_ids = data.examples.column("client_id").to_pylist()
        cutoffs = data.examples.column("cutoff").to_pylist()

        for index, (client_id, moment) in enumerate(zip(client_ids, cutoffs)):

            client_id = int(client_id)

            current = best.get(client_id)

            better = (
                current is None
                or (cutoff == "last" and moment > current.cutoff)
                or (cutoff == "first" and moment < current.cutoff)
            )

            if better:
                best[client_id] = Selection(name, index, client_id, moment)

    chosen = [best[client_id] for client_id in sorted(best)]

    warning = None

    if len(chosen) < TARGET_CLIENTS:
        warning = (
            f"уникальных клиентов {len(chosen)}, а требуется {TARGET_CLIENTS}; "
            "клиенты не дублируются, замер идёт на том, что есть"
        )

    if max_clients is not None:
        chosen = chosen[:max_clients]

    return chosen, warning


# ============================================================
# ЗАМЕР
# ============================================================


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    root: Path,
    vocab: Path,
    artifacts: Path,
    device: str = "cpu",
    batch_examples: int = 8,
    microbatch: int = 1024,
    cutoff: str = "last",
    max_clients: int | None = None,
    seed: int = 42,
    warmup: int = 1,
    datasets: tuple[str, ...] = DATASET_NAMES,
) -> dict:

    root = Path(root)

    torch_device = torch.device(device)

    tokenizer = Tokenizer.load(vocab, artifacts)

    config = config_from_tokenizer(tokenizer)

    # --------------------------------------------------------
    # ВЫБОР И ЗАГРУЗКА
    # --------------------------------------------------------

    started = time.perf_counter()

    chosen, warning = select_examples(root, vocab, cutoff, max_clients, datasets)

    if not chosen:
        raise RuntimeError("не нашлось ни одного примера")

    readers = {name: TokenizedDataset(root, name, vocab_dir=vocab) for name in sorted({s.dataset for s in chosen})}

    examples = [readers[item.dataset].load(item.index) for item in chosen]

    portions = [examples[start : start + batch_examples] for start in range(0, len(examples), batch_examples)]

    prepared = []

    for portion in portions:
        batch = collate(portion)
        prepared.append(
            (
                events_from_batch(batch, config).to(torch_device),
                profiles_from_batch(batch, config).to(torch_device),
            )
        )

    load_seconds = time.perf_counter() - started

    # --------------------------------------------------------
    # МОДЕЛЬ
    # --------------------------------------------------------

    pair = build_encoders(config, seed=seed, device=torch_device).eval()

    n_events = sum(len(events) for events, _ in prepared)
    n_profiles = sum(len(profiles) for _, profiles in prepared)

    max_event_tokens = max(events.max_length for events, _ in prepared)
    max_profile_tokens = max(profiles.max_length for _, profiles in prepared)

    event_shape = None
    profile_shape = None

    with torch.inference_mode():

        for _ in range(max(0, warmup)):
            events, profiles = prepared[0]
            pair.encode_events(events, microbatch)
            pair.encode_profiles(profiles, microbatch)

        _sync(torch_device)

        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)

        # Замеряются ВСЕ порции, включая нулевую: иначе число
        # событий в отчёте не совпало бы с измеренным.

        _sync(torch_device)
        started = time.perf_counter()

        for events, _ in prepared:
            out = pair.encode_events(events, microbatch)
            event_shape = tuple(out.shape[-1:]) if event_shape is None else event_shape

        _sync(torch_device)
        event_seconds = time.perf_counter() - started

        _sync(torch_device)
        started = time.perf_counter()

        for _, profiles in prepared:
            out = pair.encode_profiles(profiles, microbatch)
            profile_shape = tuple(out.shape[-1:]) if profile_shape is None else profile_shape

        _sync(torch_device)
        profile_seconds = time.perf_counter() - started

        peak_memory = (
            int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else None
        )

    return {
        "warning": warning,
        "device": str(torch_device),
        "clients": len({item.client_id for item in chosen}),
        "examples": len(examples),
        "portions": len(prepared),
        "events": n_events,
        "profiles": n_profiles,
        "max_event_tokens": max_event_tokens,
        "max_profile_tokens": max_profile_tokens,
        "event_output_shape": (n_events, config.d_model),
        "profile_output_shape": (n_profiles, config.d_model),
        "load_seconds": load_seconds,
        "event_seconds": event_seconds,
        "profile_seconds": profile_seconds,
        "events_per_second": n_events / event_seconds if event_seconds > 0 else float("inf"),
        "profiles_per_second": n_profiles / profile_seconds if profile_seconds > 0 else float("inf"),
        "peak_cuda_bytes": peak_memory,
        "parameters": pair.n_parameters(),
        "settings": {
            "batch_examples": batch_examples,
            "microbatch": microbatch,
            "cutoff": cutoff,
            "max_clients": max_clients,
            "seed": seed,
            "warmup": warmup,
        },
        "config": config.as_dict(),
    }


# ============================================================
# ОТЧЁТ
# ============================================================


def render(report: dict) -> str:

    lines: list[str] = []

    if report["warning"]:
        lines.append(f"ВНИМАНИЕ: {report['warning']}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("EVENT И PROFILE ENCODER")
    lines.append("=" * 60)

    def row(label: str, value) -> None:
        lines.append(f"  {label:32s}{value:>24}")

    row("устройство", report["device"])
    row("параметров модели", f"{report['parameters']:,}".replace(",", " "))
    lines.append("")

    row("клиентов", f"{report['clients']:,}".replace(",", " "))
    row("примеров", f"{report['examples']:,}".replace(",", " "))
    row("порций", f"{report['portions']:,}".replace(",", " "))
    row("событий", f"{report['events']:,}".replace(",", " "))
    row("профилей", f"{report['profiles']:,}".replace(",", " "))
    lines.append("")

    row("макс. токенов в событии", report["max_event_tokens"])
    row("макс. токенов в профиле", report["max_profile_tokens"])
    row("выход Event Encoder", str(report["event_output_shape"]))
    row("выход Profile Encoder", str(report["profile_output_shape"]))
    lines.append("")

    row("загрузка и подготовка, с", f"{report['load_seconds']:.3f}")
    row("Event Encoder, с", f"{report['event_seconds']:.3f}")
    row("Profile Encoder, с", f"{report['profile_seconds']:.3f}")
    row("событий в секунду", f"{report['events_per_second']:,.0f}".replace(",", " "))
    row("профилей в секунду", f"{report['profiles_per_second']:,.0f}".replace(",", " "))

    if report["peak_cuda_bytes"] is not None:
        row("пик памяти GPU, МБ", f"{report['peak_cuda_bytes'] / (1 << 20):.1f}")

    lines.append("")
    lines.append("  параметры запуска: " + ", ".join(f"{k}={v}" for k, v in sorted(report["settings"].items())))
    lines.append("  config: " + ", ".join(
        f"{k}={v}" for k, v in sorted(report["config"].items()) if k != "special_ids"
    ))

    return "\n".join(lines)


def main() -> None:

    parser = argparse.ArgumentParser(description="Замер Event и Profile Encoder на tokenized данных")

    parser.add_argument("--name", default="smoke")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-examples", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--cutoff", choices=("last", "first"), default="last")
    parser.add_argument("--max-clients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)

    args = parser.parse_args()

    report = run_benchmark(
        root=args.root or tokenized_dir(args.name),
        vocab=args.vocab or vocab_dir(args.name),
        artifacts=args.artifacts or tokenizer_artifacts_dir(args.name),
        device=args.device,
        batch_examples=args.batch_examples,
        microbatch=args.microbatch,
        cutoff=args.cutoff,
        max_clients=args.max_clients,
        seed=args.seed,
        warmup=args.warmup,
    )

    print(render(report))


if __name__ == "__main__":
    main()
