from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.tokenizer.artifacts import Tokenizer
from src.tokenizer.config import artifacts_dir as tokenizer_artifacts_dir
from src.tokenizer.config import tokenized_dir, vocab_dir
from src.tokenizer.build import iter_client_blocks
from src.tokenizer.dataset import TokenizedDataset, _events_of, collate

from .backbone import build_backbone
from .config import config_from_tokenizer
from .history_batching import metadata_from_examples, prepare_history_batch, to_model_inputs
from .smoke import select_examples


# ============================================================
# ИДЕЯ
# ============================================================
#
# Один и тот же набор примеров прогоняется при четырёх лимитах
# истории. Batch, dtype и microbatch зафиксированы и между
# замерами не меняются: иначе сравнивались бы разные вещи.
#
# Лимит это потолок, а не факт. Запуск с лимитом 4096 на
# историях длиной 3636 вместимость 4096 не проверяет, поэтому
# отчёт печатает фактические длины рядом с лимитом.
#
# Загрузка данных, подготовка batch и forward замеряются
# отдельно: подготовка это Python-цикл pad_records, и путать её
# со временем модели нельзя.
# ============================================================


DEFAULT_LIMITS: tuple[int, ...] = (512, 1024, 2048, 4096)

PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)


def summarize_lengths(values: np.ndarray) -> dict:
    """
    p50/p90/p95/p99/max. numeric_summary из preprocessing не
    подходит: там нет p90 и p95.
    """

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {f"p{p}": 0 for p in PERCENTILES} | {"max": 0, "mean": 0.0}

    result = {
        f"p{p}": int(np.percentile(values, p, method="inverted_cdf")) for p in PERCENTILES
    }

    result["max"] = int(values.max())
    result["mean"] = float(values.mean())

    return result


def iter_selected_examples(root: Path, vocab: Path, chosen):
    """
    Выбранные примеры потоком, по блокам клиентов.

    Читать примеры поодиночке дорого: на 10 000 клиентов это
    случайные обращения к row group. Здесь каждый файл событий
    проходится один раз по порядку.
    """

    by_dataset: dict[str, dict[int, int]] = {}

    for item in chosen:
        by_dataset.setdefault(item.dataset, {})[item.client_id] = item.index

    for dataset in sorted(by_dataset):

        wanted = by_dataset[dataset]

        data = TokenizedDataset(root, dataset, vocab_dir=vocab)

        for block in iter_client_blocks(data.events_path):

            client_id = block.column("client_id").to_numpy()

            if client_id.size == 0:
                continue

            change = np.flatnonzero(np.diff(client_id)) + 1

            starts = np.concatenate([[0], change])
            ends = np.concatenate([change, [client_id.size]])

            for lo, hi in zip(starts, ends):

                index = wanted.get(int(client_id[lo]))

                if index is None:
                    continue

                row = data.examples.slice(index, 1).to_pylist()[0]

                yield data._example(row, _events_of(block, int(lo), int(lo) + int(row["seq_end"])))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================
# ЗАМЕР
# ============================================================


def run_history_benchmark(
    root: Path,
    vocab: Path,
    artifacts: Path,
    device: str = "cpu",
    limits: tuple[int, ...] = DEFAULT_LIMITS,
    max_clients: int | None = None,
    cutoff: str = "last",
    batch_examples: int = 4,
    event_microbatch: int = 1024,
    seed: int = 42,
    warmup: int = 1,
    stream: bool = False,
    quiet: bool = True,
) -> dict:

    root = Path(root)

    torch_device = torch.device(device)

    tokenizer = Tokenizer.load(vocab, artifacts)

    config = config_from_tokenizer(tokenizer)

    # --------------------------------------------------------
    # ЗАГРУЗКА
    # --------------------------------------------------------

    started = time.perf_counter()

    chosen, warning = select_examples(root, vocab, cutoff, max_clients)

    if not chosen:
        raise RuntimeError("не нашлось ни одного примера")

    if stream:
        return _run_streaming(
            root, vocab, chosen, warning, config, torch_device,
            limits, batch_examples, event_microbatch, seed, warmup, cutoff, max_clients,
        )

    readers = {
        name: TokenizedDataset(root, name, vocab_dir=vocab)
        for name in sorted({item.dataset for item in chosen})
    }

    examples = [readers[item.dataset].load(item.index) for item in chosen]

    portions = [
        examples[start : start + batch_examples]
        for start in range(0, len(examples), batch_examples)
    ]

    collated = [(collate(portion), metadata_from_examples(portion)) for portion in portions]

    load_seconds = time.perf_counter() - started

    # --------------------------------------------------------
    # МОДЕЛЬ
    # --------------------------------------------------------

    backbone = build_backbone(config, seed=seed, device=torch_device).eval()

    results: list[dict] = []

    failure: dict | None = None

    with torch.inference_mode():

        for limit in limits:

            try:
                results.append(
                    _measure(
                        backbone,
                        config,
                        collated,
                        limit,
                        torch_device,
                        event_microbatch,
                        warmup,
                        len({item.client_id for item in chosen}),
                        len(examples),
                    )
                )
            except torch.OutOfMemoryError as error:
                failure = {
                    "limit": limit,
                    "batch_examples": batch_examples,
                    "event_microbatch": event_microbatch,
                    "device": str(torch_device),
                    "dtype": "float32",
                    "message": str(error).splitlines()[0],
                }
                break

            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "warning": warning,
        "device": str(torch_device),
        "load_seconds": load_seconds,
        "parameters": backbone.n_parameters(),
        "settings": {
            "batch_examples": batch_examples,
            "event_microbatch": event_microbatch,
            "dtype": "float32",
            "cutoff": cutoff,
            "max_clients": max_clients,
            "seed": seed,
            "warmup": warmup,
        },
        "config": config.as_dict(),
        "results": results,
        "failure": failure,
    }


def _run_streaming(
    root, vocab, chosen, warning, config, device,
    limits, batch_examples, event_microbatch, seed, warmup, cutoff, max_clients,
) -> dict:
    """
    Один проход по данным, все лимиты на каждой порции.

    Держать в памяти все примеры и все padded-тензоры нельзя:
    на 10 000 клиентов это десятки гигабайт. Набор примеров у
    всех лимитов один и тот же, меняется только порядок обхода:
    он идёт по хранилищу, а не по client_id.
    """

    backbone = build_backbone(config, seed=seed, device=device).eval()

    state = {
        limit: {
            "original": [],
            "used": [],
            "truncated": [],
            "events": 0,
            "longest": 0,
            "batch_rows": 0,
            "prepare": 0.0,
            "event": 0.0,
            "profile": 0.0,
            "history": 0.0,
            "peak": 0,
        }
        for limit in limits
    }

    load_seconds = 0.0

    failure = None

    source = iter_selected_examples(root, vocab, chosen)

    warmed = False

    with torch.inference_mode():

        while failure is None:

            moment = time.perf_counter()

            portion = []

            for example in source:
                portion.append(example)
                if len(portion) >= batch_examples:
                    break

            if not portion:
                load_seconds += time.perf_counter() - moment
                break

            batch = collate(portion)
            meta = metadata_from_examples(portion)

            load_seconds += time.perf_counter() - moment

            for limit in limits:

                cell = state[limit]

                try:

                    moment = time.perf_counter()
                    history = prepare_history_batch(batch, meta, limit)
                    inputs = to_model_inputs(history, config, device)
                    cell["prepare"] += time.perf_counter() - moment

                    if not warmed:
                        for _ in range(max(0, warmup)):
                            vectors = backbone.pair.encode_events(inputs.events, event_microbatch)
                            snapshots = backbone.pair.encode_profiles(inputs.profiles)
                            backbone.encode_history(inputs, vectors, snapshots)

                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)

                    _sync(device)
                    moment = time.perf_counter()
                    vectors = backbone.pair.encode_events(inputs.events, event_microbatch)
                    _sync(device)
                    cell["event"] += time.perf_counter() - moment

                    moment = time.perf_counter()
                    snapshots = backbone.pair.encode_profiles(inputs.profiles)
                    _sync(device)
                    cell["profile"] += time.perf_counter() - moment

                    moment = time.perf_counter()
                    out = backbone.encode_history(inputs, vectors, snapshots)
                    _sync(device)
                    cell["history"] += time.perf_counter() - moment

                    if device.type == "cuda":
                        cell["peak"] = max(cell["peak"], int(torch.cuda.max_memory_allocated(device)))

                    cell["original"].append(history.info.original_history_length)
                    cell["used"].append(history.info.used_history_length)
                    cell["truncated"].append(history.info.truncated)
                    cell["events"] += inputs.n_events

                    if out.max_length > cell["longest"]:
                        cell["longest"] = out.max_length
                        cell["batch_rows"] = out.n_examples

                    del inputs, history, vectors, snapshots, out

                except torch.OutOfMemoryError as error:
                    failure = {
                        "limit": limit,
                        "batch_examples": batch_examples,
                        "event_microbatch": event_microbatch,
                        "device": str(device),
                        "dtype": "float32",
                        "message": str(error).splitlines()[0],
                    }
                    break

            warmed = True

    results = []

    n_clients = len({item.client_id for item in chosen})

    for limit in limits:

        cell = state[limit]

        if not cell["used"]:
            continue

        original = np.concatenate(cell["original"])
        used = np.concatenate(cell["used"])
        truncated = np.concatenate(cell["truncated"])

        results.append(
            {
                "limit": limit,
                "clients": n_clients,
                "examples": int(used.size),
                "original": summarize_lengths(original),
                "used": summarize_lengths(used),
                "truncated_share": float(truncated.mean()),
                "reached_limit": bool(used.max() >= limit),
                "events": cell["events"],
                "client_embedding_shape": (int(used.size), config.d_model),
                "contextualized_shape": (cell["batch_rows"], cell["longest"], config.d_model),
                "prepare_seconds": cell["prepare"],
                "event_seconds": cell["event"],
                "profile_seconds": cell["profile"],
                "history_seconds": cell["history"],
                "forward_seconds": cell["event"] + cell["profile"] + cell["history"],
                "events_per_second": cell["events"] / max(cell["event"], 1e-9),
                "peak_cuda_bytes": cell["peak"] or None,
            }
        )

    return {
        "warning": warning,
        "device": str(device),
        "load_seconds": load_seconds,
        "parameters": backbone.n_parameters(),
        "settings": {
            "batch_examples": batch_examples,
            "event_microbatch": event_microbatch,
            "dtype": "float32",
            "cutoff": cutoff,
            "max_clients": max_clients,
            "seed": seed,
            "warmup": warmup,
            "stream": True,
        },
        "config": config.as_dict(),
        "results": results,
        "failure": failure,
    }


def _measure(
    backbone,
    config,
    collated,
    limit: int,
    device: torch.device,
    event_microbatch: int,
    warmup: int,
    n_clients: int,
    n_examples: int,
) -> dict:

    started = time.perf_counter()

    prepared = []

    original: list[np.ndarray] = []
    used: list[np.ndarray] = []
    truncated: list[np.ndarray] = []

    for batch, meta in collated:

        history = prepare_history_batch(batch, meta, limit)

        prepared.append(to_model_inputs(history, config, device))

        original.append(history.info.original_history_length)
        used.append(history.info.used_history_length)
        truncated.append(history.info.truncated)

    prepare_seconds = time.perf_counter() - started

    # --------------------------------------------------------

    for _ in range(max(0, warmup)):
        inputs = prepared[0]
        vectors = backbone.pair.encode_events(inputs.events, event_microbatch)
        snapshots = backbone.pair.encode_profiles(inputs.profiles)
        backbone.encode_history(inputs, vectors, snapshots)

    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    event_seconds = profile_seconds = history_seconds = 0.0

    n_events = 0
    longest = 0

    for inputs in prepared:

        _sync(device)
        moment = time.perf_counter()
        vectors = backbone.pair.encode_events(inputs.events, event_microbatch)
        _sync(device)
        event_seconds += time.perf_counter() - moment

        moment = time.perf_counter()
        snapshots = backbone.pair.encode_profiles(inputs.profiles)
        _sync(device)
        profile_seconds += time.perf_counter() - moment

        moment = time.perf_counter()
        out = backbone.encode_history(inputs, vectors, snapshots)
        _sync(device)
        history_seconds += time.perf_counter() - moment

        n_events += inputs.n_events
        longest = max(longest, out.max_length)

    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    original_all = np.concatenate(original)
    used_all = np.concatenate(used)
    truncated_all = np.concatenate(truncated)

    return {
        "limit": limit,
        "clients": n_clients,
        "examples": n_examples,
        "original": summarize_lengths(original_all),
        "used": summarize_lengths(used_all),
        "truncated_share": float(truncated_all.mean()),
        "reached_limit": bool(used_all.max() >= limit),
        "events": n_events,
        "client_embedding_shape": (n_examples, config.d_model),
        "contextualized_shape": (len(prepared[0].used_history_length), longest, config.d_model),
        "prepare_seconds": prepare_seconds,
        "event_seconds": event_seconds,
        "profile_seconds": profile_seconds,
        "history_seconds": history_seconds,
        "forward_seconds": event_seconds + profile_seconds + history_seconds,
        "events_per_second": n_events / max(event_seconds, 1e-9),
        "peak_cuda_bytes": peak,
    }


# ============================================================
# ОТЧЁТ
# ============================================================


def render(report: dict) -> str:

    lines: list[str] = []

    if report["warning"]:
        lines.append(f"ВНИМАНИЕ: {report['warning']}")
        lines.append("")

    lines.append("=" * 78)
    lines.append("HISTORY ENCODER: ЛИМИТЫ ИСТОРИИ")
    lines.append("=" * 78)
    lines.append(f"  устройство {report['device']}, параметров {report['parameters']:,}".replace(",", " "))
    lines.append(
        "  зафиксировано: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(report["settings"].items())
            if key in ("batch_examples", "event_microbatch", "dtype")
        )
    )
    lines.append(f"  загрузка данных, с {report['load_seconds']:.2f}")
    lines.append("")

    header = f"  {'лимит':>7s}{'обрезано':>11s}{'used p50':>10s}{'p90':>7s}{'p95':>7s}{'p99':>7s}{'max':>7s}{'событий':>10s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for item in report["results"]:
        lines.append(
            f"  {item['limit']:>7,}{item['truncated_share']:>11.2f}"
            f"{item['used']['p50']:>10,}{item['used']['p90']:>7,}{item['used']['p95']:>7,}"
            f"{item['used']['p99']:>7,}{item['used']['max']:>7,}{item['events']:>10,}".replace(",", " ")
        )

    if report["results"]:
        first = report["results"][0]["original"]
        lines.append("")
        lines.append(
            f"  исходные длины: p50 {first['p50']:,}  p90 {first['p90']:,}  p95 {first['p95']:,}  "
            f"p99 {first['p99']:,}  max {first['max']:,}".replace(",", " ")
        )

    lines.append("")

    header = (
        f"  {'лимит':>7s}{'подготовка':>12s}{'Event':>9s}{'Profile':>9s}{'History':>9s}"
        f"{'forward':>9s}{'соб/с':>11s}{'пик GPU МБ':>12s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for item in report["results"]:
        peak = f"{item['peak_cuda_bytes'] / (1 << 20):.0f}" if item["peak_cuda_bytes"] else "—"
        lines.append(
            f"  {item['limit']:>7,}{item['prepare_seconds']:>12.2f}{item['event_seconds']:>9.2f}"
            f"{item['profile_seconds']:>9.3f}{item['history_seconds']:>9.2f}{item['forward_seconds']:>9.2f}"
            f"{item['events_per_second']:>11,.0f}{peak:>12s}".replace(",", " ")
        )

    lines.append("")

    for item in report["results"]:
        note = "" if item["reached_limit"] else "  (данные короче лимита: вместимость не проверена)"
        lines.append(
            f"  лимит {item['limit']:>5}: client_embedding {item['client_embedding_shape']}, "
            f"contextualized {item['contextualized_shape']}{note}"
        )

    if report["failure"]:
        lines.append("")
        lines.append("  ОСТАНОВ ПО ПАМЯТИ")
        lines.append("  " + ", ".join(f"{key}={value}" for key, value in sorted(report["failure"].items())))

    return "\n".join(lines)


def main() -> None:

    parser = argparse.ArgumentParser(description="Замер History Encoder при разных лимитах истории")

    parser.add_argument("--name", default="dev")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limits", default=",".join(str(value) for value in DEFAULT_LIMITS))
    parser.add_argument("--max-clients", type=int, default=None)
    parser.add_argument("--cutoff", choices=("last", "first"), default="last")
    parser.add_argument("--batch-examples", type=int, default=4)
    parser.add_argument("--event-microbatch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--stream", action="store_true", help="потоковый режим: в памяти одна порция")

    args = parser.parse_args()

    report = run_history_benchmark(
        root=args.root or tokenized_dir(args.name),
        vocab=args.vocab or vocab_dir(args.name),
        artifacts=args.artifacts or tokenizer_artifacts_dir(args.name),
        device=args.device,
        limits=tuple(int(value) for value in args.limits.split(",")),
        max_clients=args.max_clients,
        cutoff=args.cutoff,
        batch_examples=args.batch_examples,
        event_microbatch=args.event_microbatch,
        seed=args.seed,
        warmup=args.warmup,
        stream=args.stream,
    )

    print(render(report))


if __name__ == "__main__":
    main()
