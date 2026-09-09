from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.generator.config import RUNS_DIR
from src.preprocessing.artifacts import write_json, write_text
from src.preprocessing.config import processed_dir as prep_processed_dir
from src.tokenizer.config import artifacts_dir, tokenized_dir, vocab_dir

from .config import EVENT_ARCHITECTURE, SESSION_ARCHITECTURE, STRUCTURE_EVENT, STRUCTURE_SESSION
from .data import ClientStore, session_keys_from_examples
from .history_batching import metadata_from_examples, prepare_history_batch
from .history_smoke import summarize_lengths
from .session_batching import group_events
from .trainer import TrainConfig, Trainer, load_environment
from src.tokenizer.dataset import collate


# ============================================================
# ИДЕЯ
# ============================================================
#
# Две вещи в одном отчёте.
#
# Первая: сколько позиций History Encoder экономит группировка.
# Считается по данным, без модели, на всех примерах набора.
#
# Вторая: чего это стоит по времени и памяти. Обе структуры
# гоняются на ОДНИХ И ТЕХ ЖЕ batch'ах, с одним seed и одними
# масками; равенство целей проверяется отпечатком.
#
# Обучения здесь нет: backward считается, optimizer.step не
# вызывается. Сокращение длины это сокращение длины, а не
# доказанное ускорение и тем более не качество.
# ============================================================


DATASETS: tuple[str, ...] = ("train", "val_client", "test_client", "val_time", "test_time")

APP_TYPES: tuple[str, ...] = ("app_screen", "app_operation", "banner")

DEFAULT_BATCH = 2
DEFAULT_BATCHES = 3


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device) -> float | None:
    if torch.device(device).type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1 << 20)


# ============================================================
# СТАТИСТИКА ГРУППИРОВКИ
# ============================================================


def dataset_statistics(
    root: Path,
    vocab_path: Path,
    datasets: tuple[str, ...] = DATASETS,
    batch_size: int = 64,
    max_clients: int | None = None,
) -> dict:
    """
    Сколько получается сессий и насколько короче история.

    Модель не участвует: только массивы. Считается на тех же
    примерах, что увидит обучение, то есть на префиксах до
    cutoff.
    """

    report: dict = {"datasets": {}}

    totals = {
        "examples": 0,
        "events": 0,
        "app_events": 0,
        "screens": 0,
        "grouped": 0,
        "sessions": 0,
    }

    before: list[np.ndarray] = []
    after: list[np.ndarray] = []
    lengths: list[np.ndarray] = []

    for name in datasets:

        store = ClientStore(root, name, vocab_path, max_clients=max_clients, sessions=True)

        local = {
            "examples": 0,
            "events": 0,
            "app_events": 0,
            "screens": 0,
            "grouped": 0,
            "sessions": 0,
        }

        local_before: list[np.ndarray] = []
        local_after: list[np.ndarray] = []
        local_lengths: list[np.ndarray] = []

        for start in range(0, len(store), batch_size):

            chunk = range(start, min(start + batch_size, len(store)))

            examples = store.examples(chunk)

            batch = collate(examples)
            meta = metadata_from_examples(examples)

            layout = group_events(
                example_of_event=batch.example_of_event,
                ts=batch.ts,
                seq=batch.seq,
                session_keys=session_keys_from_examples(examples),
                cutoffs=meta.cutoffs,
                n_examples=batch.n_examples,
            )

            types = np.asarray(batch.event_type, dtype=object)

            local["examples"] += batch.n_examples
            local["events"] += batch.n_events
            local["app_events"] += int(np.isin(types, APP_TYPES).sum())
            local["screens"] += int((types == "app_screen").sum())
            local["grouped"] += layout.n_grouped
            local["sessions"] += layout.n_sessions

            original = np.bincount(
                np.asarray(batch.example_of_event, dtype=np.int64), minlength=batch.n_examples
            )

            local_before.append(original.astype(np.int64))
            local_after.append(np.asarray(layout.used_history_length, dtype=np.int64))
            local_lengths.append(np.asarray(layout.session_length, dtype=np.int64))

        example_before = np.concatenate(local_before) if local_before else np.zeros(0, np.int64)
        example_after = np.concatenate(local_after) if local_after else np.zeros(0, np.int64)
        session_lengths = (
            np.concatenate(local_lengths) if local_lengths else np.zeros(0, np.int64)
        )

        report["datasets"][name] = {
            **local,
            "clients": len(store.client_ids),
            "grouped_share_of_app": _share(local["grouped"], local["app_events"]),
            "screens_share_of_app": _share(local["screens"], local["app_events"]),
            "history_before": summarize_lengths(example_before),
            "history_after": summarize_lengths(example_after),
            "reduction": _reduction(example_before, example_after),
            "session_length": _session_length(session_lengths),
        }

        for key in totals:
            totals[key] += local[key]

        before.append(example_before)
        after.append(example_after)
        lengths.append(session_lengths)

    all_before = np.concatenate(before) if before else np.zeros(0, np.int64)
    all_after = np.concatenate(after) if after else np.zeros(0, np.int64)
    all_lengths = np.concatenate(lengths) if lengths else np.zeros(0, np.int64)

    report["total"] = {
        **totals,
        "grouped_share_of_app": _share(totals["grouped"], totals["app_events"]),
        "screens_share_of_app": _share(totals["screens"], totals["app_events"]),
        "history_before": summarize_lengths(all_before),
        "history_after": summarize_lengths(all_after),
        "reduction": _reduction(all_before, all_after),
        "session_length": _session_length(all_lengths),
    }

    report["rule"] = (
        "сессия это события приложения с одним session_id внутри примера: "
        "экраны, операции и баннеры, рождённые в ней; отдельным событием "
        "остаётся то, у чего принадлежность неизвестна"
    )

    return report


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _reduction(before: np.ndarray, after: np.ndarray) -> dict:
    """
    Насколько меньше стало позиций History Encoder.
    """

    if before.size == 0:
        return {"positions_before": 0, "positions_after": 0, "share": 0.0, "per_example_p50": 0.0}

    total_before = int(before.sum())
    total_after = int(after.sum())

    per_example = 1.0 - after / np.maximum(before, 1)

    return {
        "positions_before": total_before,
        "positions_after": total_after,
        "share": round(1.0 - total_after / total_before, 4) if total_before else 0.0,
        "per_example_p50": round(float(np.percentile(per_example, 50)), 4),
    }


def _session_length(values: np.ndarray) -> dict:

    if values.size == 0:
        return {"n": 0}

    counts = np.bincount(values)

    return {
        "n": int(values.size),
        **summarize_lengths(values),
        "mean": round(float(values.mean()), 3),
        "share_of_one": round(float((values == 1).mean()), 4),
        "histogram": {
            str(index): int(counts[index]) for index in range(1, min(counts.size, 11))
        },
        "over_ten": int((values > 10).sum()),
    }


# ============================================================
# СРАВНЕНИЕ РЕЖИМОВ
# ============================================================


def pick_batches(store: ClientStore, n_batches: int, batch_size: int) -> list[list[int]]:
    """
    Самые длинные истории, медианные и короткие.

    Одни и те же примеры идут в обе структуры: иначе сравнение
    было бы про разные данные.
    """

    lengths = np.array([int(row["seq_end"]) for row in store.rows], dtype=np.int64)

    order = np.argsort(lengths)

    picks: list[list[int]] = []

    positions = (len(order) - 1, len(order) // 2, 0)

    for index in range(min(n_batches, len(positions))):

        anchor = positions[index]

        lo = max(0, min(anchor - batch_size + 1, len(order) - batch_size))

        picks.append([int(order[lo + step]) for step in range(min(batch_size, len(order)))])

    return picks


def measure_mode(
    env,
    config: TrainConfig,
    structure: str,
    batches: list[list[int]],
    device: str,
    warmup: int = 2,
    repeats: int = 3,
) -> dict:
    """
    Forward и backward одной структуры на заданных batch'ах.

    optimizer.step не вызывается: это замер, а не обучение.
    """

    architecture = SESSION_ARCHITECTURE if structure == STRUCTURE_SESSION else EVENT_ARCHITECTURE

    settings = replace(config, **architecture)

    trainer = Trainer(settings, env.tokenizer, env.table, env.unigram, device)

    trainer.train_mode()

    store = ClientStore(
        env.root,
        "train",
        env.vocab_dir,
        max_clients=config.max_train_clients,
        clients=None,
        sessions=settings.uses_sessions,
    )

    prepared = []

    for index, rows in enumerate(batches):
        inputs, targets, history = trainer.prepare(store.examples(rows), index)
        prepared.append((inputs, targets, history))

    shapes = []

    for inputs, targets, history in prepared:

        item = {
            "examples": int(inputs.n_examples),
            "events": int(inputs.n_events),
            # Ширина после padding: столько позиций реально
            # обсчитывает History Encoder.
            "padded_width": int(inputs.max_length),
            # Сумма занятых позиций по примерам: столько их
            # содержательно, без padding.
            "used_positions": int(inputs.used_history_length.sum()) + int(inputs.n_examples),
            "targets": int(targets.n),
            "truncated": bool(history.info.any_truncated),
        }

        if inputs.sessions is not None:
            item["sessions"] = int(inputs.sessions.n_sessions)
            item["max_session_length"] = int(inputs.sessions.max_session_length)

        shapes.append(item)

    digests = [targets.digest() for _, targets, _ in prepared]

    forward_seconds: list[float] = []
    backward_seconds: list[float] = []

    for index in range(warmup + repeats):

        for inputs, targets, _ in prepared:

            trainer.optimizer.zero_grad(set_to_none=True)

            _sync(device)

            if index == warmup and torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            started = time.perf_counter()

            loss, _, _ = trainer.compute(inputs, targets)

            _sync(device)

            middle = time.perf_counter()

            loss.field_balanced.backward()

            _sync(device)

            finished = time.perf_counter()

            if index >= warmup:
                forward_seconds.append(middle - started)
                backward_seconds.append(finished - middle)

    trainer.optimizer.zero_grad(set_to_none=True)

    session_parameters = (
        trainer.backbone.session.n_parameters() if trainer.backbone.session is not None else 0
    )

    fuse_session = (
        sum(p.numel() for p in trainer.head.fuse_session.parameters())
        if trainer.head.fuse_session is not None
        else 0
    )

    return {
        "structure": structure,
        "config": trainer.model_config.as_dict(),
        "parameters": {
            "backbone": trainer.backbone.n_parameters(),
            "head": trainer.head.n_parameters(),
            "total": trainer.n_parameters(),
            "session_encoder": session_parameters,
            "fuse_session": fuse_session,
        },
        "shapes": shapes,
        "targets_sha256": digests,
        "forward_seconds": round(float(np.mean(forward_seconds)), 5),
        "backward_seconds": round(float(np.mean(backward_seconds)), 5),
        "step_seconds": round(float(np.mean(forward_seconds) + np.mean(backward_seconds)), 5),
        "peak_mb": None if _memory(device) is None else round(_memory(device), 1),
    }


def examples_of_grouping(root: Path, vocab_path: Path, n: int = 3) -> list[dict]:
    """
    Несколько примеров группировки для отчёта.
    """

    store = ClientStore(root, "val_time", vocab_path, max_clients=n, sessions=True)

    shown: list[dict] = []

    for index in range(min(n, len(store))):

        example = store.example(index)

        batch = collate([example])
        meta = metadata_from_examples([example])

        layout = group_events(
            example_of_event=batch.example_of_event,
            ts=batch.ts,
            seq=batch.seq,
            session_keys=session_keys_from_examples([example]),
            cutoffs=meta.cutoffs,
            n_examples=1,
        )

        order = np.argsort(layout.session_slot)[:4]

        sessions = [
            {
                "slot": int(layout.session_slot[position]),
                "screens": int(layout.session_length[position]),
                "start": str(layout.session_start[position]),
                "end": str(layout.session_end[position]),
            }
            for position in order
        ]

        types = np.asarray(batch.event_type, dtype=object)

        shown.append(
            {
                "client_id": int(example.client_id),
                "cutoff": str(example.cutoff),
                "events": int(batch.n_events),
                "app_events": int(np.isin(types, APP_TYPES).sum()),
                "history_positions": int(layout.used_history_length[0]),
                "sessions": layout.n_sessions,
                "first_sessions": sessions,
                "standalone_app": int(
                    (np.isin(types, APP_TYPES) & (layout.session_of_event < 0)).sum()
                ),
            }
        )

    return shown


# ============================================================
# ЗАПУСК
# ============================================================


REMAINING = (
    "исправление состава MLM-целей",
    "temporal validation только на новых событиях",
    "обучение на 10 000 клиентах",
    "downstream-сравнение",
)


def run_session_smoke(
    env,
    out_dir: Path,
    config: TrainConfig,
    device: str = "cpu",
    n_batches: int = DEFAULT_BATCHES,
    batch_size: int = DEFAULT_BATCH,
    warmup: int = 2,
    repeats: int = 3,
    max_clients: int | None = None,
    quiet: bool = False,
) -> dict:

    out_dir = Path(out_dir)

    started = time.perf_counter()

    statistics = dataset_statistics(
        env.root, env.vocab_dir, max_clients=max_clients
    )

    grouping = examples_of_grouping(env.root, env.vocab_dir)

    store = ClientStore(
        env.root, "train", env.vocab_dir, max_clients=config.max_train_clients, sessions=True
    )

    batches = pick_batches(store, n_batches, batch_size)

    del store

    modes: dict = {}
    limit: dict | None = None

    attempt = batch_size

    while attempt >= 1:

        try:
            trial = {}

            for structure in (STRUCTURE_EVENT, STRUCTURE_SESSION):

                trial[structure] = measure_mode(
                    env=env,
                    config=config,
                    structure=structure,
                    batches=[rows[:attempt] for rows in batches],
                    device=device,
                    warmup=warmup,
                    repeats=repeats,
                )

                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()

            modes = trial
            break

        except torch.OutOfMemoryError as error:

            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()

            limit = {
                "batch_size": attempt,
                "message": str(error).splitlines()[0],
            }

            attempt -= 1

    report = {
        "device": device,
        "batch_size": attempt if modes else None,
        "requested_batch_size": batch_size,
        "n_batches": n_batches,
        "statistics": statistics,
        "grouping_examples": grouping,
        "modes": modes,
        "limit": limit,
        "seconds": round(time.perf_counter() - started, 1),
        "caveat": (
            "сокращение числа позиций это сокращение числа позиций. "
            "Ни ускорения, ни улучшения качества оно не доказывает: "
            "обучение здесь не запускалось, backward считался без optimizer.step"
        ),
        "remaining": list(REMAINING),
    }

    if modes:

        digests = {name: item["targets_sha256"] for name, item in modes.items()}

        report["same_targets"] = digests[STRUCTURE_EVENT] == digests[STRUCTURE_SESSION]

    write_json(out_dir / "session_smoke.json", report)
    write_text(out_dir / "session_smoke.md", render_session_smoke(report))

    if not quiet:
        print(render_session_smoke(report))

    return report


# ============================================================
# ОТЧЁТ
# ============================================================


def _number(value) -> str:
    return "—" if value is None else f"{value:,}".replace(",", " ")


def _lengths_row(label: str, item: dict) -> str:
    return (
        f"| {label} | {_number(item.get('p50'))} | {_number(item.get('p90'))} "
        f"| {_number(item.get('p95'))} | {_number(item.get('p99'))} | {_number(item.get('max'))} |"
    )


def render_session_smoke(report: dict) -> str:

    lines: list[str] = []

    lines.append("# Session Encoder: группировка и стоимость")
    lines.append("")
    lines.append(report["statistics"]["rule"] + ".")
    lines.append("")

    total = report["statistics"]["total"]

    lines.append("## Что сгруппировалось")
    lines.append("")
    lines.append("| Показатель | Значение |")
    lines.append("|---|---:|")
    lines.append(f"| примеров | {_number(total['examples'])} |")
    lines.append(f"| событий | {_number(total['events'])} |")
    lines.append(f"| app-событий | {_number(total['app_events'])} |")
    lines.append(f"| из них экранов | {_number(total['screens'])} |")
    lines.append(f"| сгруппировано | {_number(total['grouped'])} |")
    lines.append(f"| сессий | {_number(total['sessions'])} |")
    lines.append(f"| доля сгруппированных app-событий | {total['grouped_share_of_app']} |")
    lines.append(f"| из них экранов, доля | {total['screens_share_of_app']} |")
    lines.append("")

    lines.append("## Длина истории")
    lines.append("")
    lines.append("| | p50 | p90 | p95 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(_lengths_row("до группировки", total["history_before"]))
    lines.append(_lengths_row("после", total["history_after"]))
    lines.append("")

    reduction = total["reduction"]

    lines.append(
        f"Позиций History Encoder: {_number(reduction['positions_before'])} → "
        f"{_number(reduction['positions_after'])}, то есть на "
        f"{reduction['share']:.1%} меньше; медиана по примерам "
        f"{reduction['per_example_p50']:.1%}."
    )
    lines.append("")

    length = total["session_length"]

    lines.append("## Длина сессии")
    lines.append("")
    lines.append("| Событий | Сессий |")
    lines.append("|---|---:|")

    for key, value in (length.get("histogram") or {}).items():
        lines.append(f"| {key} | {_number(value)} |")

    lines.append(f"| больше 10 | {_number(length.get('over_ten'))} |")
    lines.append("")
    lines.append(
        f"Среднее {length.get('mean')}, медиана {length.get('p50')}, "
        f"p99 {length.get('p99')}, максимум {length.get('max')}; "
        f"доля сессий из одного события {length.get('share_of_one')}."
    )
    lines.append("")

    lines.append("## По сплитам")
    lines.append("")
    lines.append("| Сплит | примеров | событий | сессий | позиций до | после | сокращение |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for name, item in report["statistics"]["datasets"].items():
        lines.append(
            f"| {name} | {_number(item['examples'])} | {_number(item['events'])} "
            f"| {_number(item['sessions'])} | {_number(item['reduction']['positions_before'])} "
            f"| {_number(item['reduction']['positions_after'])} "
            f"| {item['reduction']['share']:.1%} |"
        )

    lines.append("")

    modes = report.get("modes") or {}

    if modes:

        lines.append("## Две структуры на одних batch")
        lines.append("")
        lines.append(
            f"Устройство {report['device']}, batch {report['batch_size']}, "
            f"{report['n_batches']} batch на режим, полные истории без обрезки."
        )

        if report.get("same_targets"):
            lines.append("Цели и маски совпадают: отпечатки targets одинаковы.")

        lines.append("")
        lines.append("| Показатель | event | session |")
        lines.append("|---|---:|---:|")

        left = modes.get(STRUCTURE_EVENT, {})
        right = modes.get(STRUCTURE_SESSION, {})

        for label, key in (
            ("параметров всего", "total"),
            ("backbone", "backbone"),
            ("head", "head"),
            ("Session Encoder", "session_encoder"),
            ("проекция сессий", "fuse_session"),
        ):
            lines.append(
                f"| {label} | {_number(left.get('parameters', {}).get(key))} "
                f"| {_number(right.get('parameters', {}).get(key))} |"
            )

        for label, key in (
            ("forward, с", "forward_seconds"),
            ("backward, с", "backward_seconds"),
            ("шаг, с", "step_seconds"),
            ("пик памяти, МБ", "peak_mb"),
        ):
            lines.append(f"| {label} | {left.get(key)} | {right.get(key)} |")

        lines.append("")
        lines.append(
            "| batch | события | позиций event | позиций session | ширина event "
            "| ширина session | сессий | S |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        for index, item in enumerate(left.get("shapes", [])):
            other = right.get("shapes", [])[index] if index < len(right.get("shapes", [])) else {}
            lines.append(
                f"| {index} | {_number(item.get('events'))} "
                f"| {_number(item.get('used_positions'))} "
                f"| {_number(other.get('used_positions'))} "
                f"| {_number(item.get('padded_width'))} "
                f"| {_number(other.get('padded_width'))} "
                f"| {_number(other.get('sessions'))} | {_number(other.get('max_session_length'))} |"
            )

        lines.append("")
        lines.append(
            "Ширина это длина после padding, её и обсчитывает History Encoder; "
            "позиции это сумма занятых по примерам. Когда самый длинный пример "
            "batch без сессий, ширина у обеих структур одинакова."
        )

        lines.append("")

    if report.get("limit"):
        lines.append(
            f"Память: при batch {report['limit']['batch_size']} произошло переполнение, "
            f"замер повторён меньшим batch."
        )
        lines.append("")

    lines.append("## Примеры группировки")
    lines.append("")

    for item in report["grouping_examples"]:
        lines.append(
            f"**Клиент {item['client_id']}**, cutoff {item['cutoff']}: событий "
            f"{_number(item['events'])}, из них app {_number(item['app_events'])}; "
            f"сессий {item['sessions']}, позиций истории {_number(item['history_positions'])}, "
            f"отдельных app-событий {_number(item['standalone_app'])}."
        )
        lines.append("")
        lines.append("| слот | событий | начало | конец |")
        lines.append("|---|---:|---|---|")

        for session in item["first_sessions"]:
            lines.append(
                f"| {session['slot']} | {session['screens']} | {session['start']} | {session['end']} |"
            )

        lines.append("")

    lines.append("## Чего этот отчёт не утверждает")
    lines.append("")
    lines.append(report["caveat"] + ".")
    lines.append("")
    lines.append("Вне этого этапа остаются: " + ", ".join(report["remaining"]) + ".")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================


def main() -> None:

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Session Encoder: замер группировки и стоимости")

    parser.add_argument("--name", default="v21_1000")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stat-clients", type=int, default=None)
    parser.add_argument("--train-clients", type=int, default=None)

    args = parser.parse_args()

    root = args.root or tokenized_dir(args.name)
    vocab = args.vocab or vocab_dir(args.name)
    artifacts = args.artifacts or artifacts_dir(args.name)

    out_dir = args.out or RUNS_DIR / args.name / "session_smoke"

    env = load_environment(root, vocab, artifacts)

    config = TrainConfig(
        max_train_clients=args.train_clients,
        max_val_clients=None,
        max_events_per_history=None,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )

    run_session_smoke(
        env=env,
        out_dir=out_dir,
        config=config,
        device=args.device,
        n_batches=args.batches,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        max_clients=args.stat_clients,
    )

    print(f"записано: {out_dir}")


if __name__ == "__main__":
    main()


__all__ = ["dataset_statistics", "render_session_smoke", "run_session_smoke"]
