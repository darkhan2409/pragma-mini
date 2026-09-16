from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

from src.generator.config import RUNS_DIR
from src.preprocessing.run import clean_directory
from src.tokenizer.config import artifacts_dir as default_artifacts_dir
from src.tokenizer.config import tokenized_dir as default_tokenized_dir
from src.tokenizer.config import vocab_dir as default_vocab_dir

from .compare import run_comparison

from .checkpoint import verify_checkpoint
from .targets import TARGET_POLICIES
from .trainer import (
    ARCHITECTURE_FIELDS,
    BEST_SCOPES,
    TrainConfig,
    TrainingAborted,
    TrainingInterrupted,
    benchmark,
    load_environment,
    run_training,
    tiny_overfit,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Порядок первого запуска задан не памятью, а командами:
#
#   overfit     может ли модель вообще запомнить два примера
#   benchmark   сколько стоит шаг именно в этой конфигурации
#   run         один короткий эксперимент с фиксированным бюджетом
#   compare     две обученные арки на одних примерах и масках
#
# Каждый режим строит СВОЮ модель: сравнивать скорость на
# частично обученных весах или продолжать run после overfit
# значит мерить и обучать разные вещи. Исключение это compare:
# он берёт и веса, и конфигурацию из checkpoint, потому что
# обязан воспроизвести обучение, а не задать его заново.
#
# Каталог запуска не перезаписывается молча: результат прошлого
# эксперимента дороже удобства.
# ============================================================


COMPARISON_DIR = "comparison"


def run_dir(name: str, mode: str, tag: str | None = None) -> Path:
    """
    Тег отделяет эксперименты друг от друга.

    Без тега это прежний путь, поэтому результаты первого
    эксперимента остаются там, где были.
    """

    return RUNS_DIR / name / mode if not tag else RUNS_DIR / name / tag / mode


def prepare_dir(path: Path, force: bool, keep: bool = False) -> Path:
    """
    keep=True оставляет каталог как есть.

    Так продолжают прогон: чистить каталог, из которого только
    что прочитан checkpoint, значит удалить и его, и лог, куда
    продолжение обязано дописывать.
    """

    path = Path(path)

    if keep:
        path.mkdir(parents=True, exist_ok=True)
        return path

    if path.exists() and any(path.iterdir()):

        if not force:
            raise SystemExit(
                f"каталог {path} уже содержит результат запуска; "
                "укажите --out или --force, если его действительно нужно перезаписать"
            )

        clean_directory(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


# ============================================================
# АРГУМЕНТЫ
# ============================================================


# Отличает «флаг не передавали» от «передали none»: у трёх
# лимитов None это значение, а не отсутствие значения.
NOT_GIVEN = object()

NULLABLE: tuple[str, ...] = (
    "max_events_per_history",
    "max_train_clients",
    "max_val_clients",
)


def int_or_none(text: str) -> int | None:
    """
    Число или слово none.

    None это не «побольше», а «лимита нет»: у обрезки истории
    и у числа клиентов это отдельная ветка кода, и выразить её
    большим числом нельзя. При этом «флаг не передан» по-прежнему
    значит «оставить умолчание».
    """

    if text.strip().lower() in ("none", "нет", "off"):
        return None

    value = int(text)

    if value < 1:
        raise argparse.ArgumentTypeError(f"ожидалось положительное число или none, получено {text!r}")

    return value


def split_list(text: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in text.split(",") if name.strip())


def add_common(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--name", default="dev")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tag", default=None, help="подкаталог эксперимента внутри data/runs/<name>")
    parser.add_argument("--force", action="store_true", help="перезаписать каталог запуска")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "float32", "bf16"), default="auto")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--max-events",
        type=int_or_none,
        default=NOT_GIVEN,
        dest="max_events_per_history",
        help="лимит истории или none",
    )
    parser.add_argument("--event-microbatch", type=int, default=None)
    parser.add_argument(
        "--masking-mode",
        choices=("token", "key", "event", "field_balanced", "combined"),
        default=None,
    )
    parser.add_argument("--balanced-share", type=float, default=None)
    parser.add_argument("--token-rate", type=float, default=None)
    parser.add_argument("--event-rate", type=float, default=None)
    parser.add_argument("--key-rate", type=float, default=None)
    parser.add_argument(
        "--train-clients", type=int_or_none, default=NOT_GIVEN, dest="max_train_clients"
    )
    parser.add_argument(
        "--val-clients", type=int_or_none, default=NOT_GIVEN, dest="max_val_clients"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--target-policy", choices=TARGET_POLICIES, default=None)
    parser.add_argument(
        "--vocab-tag",
        default=None,
        help="комплект словаря: data/artifacts/<name>/tokenizer__<tag> и data/tokenized/<name>__<tag>",
    )
    parser.add_argument(
        "--new-vocab-tag",
        default=None,
        help="только для compare: комплект словаря ВТОРОЙ арки, если он другой",
    )
    parser.add_argument(
        "--stream-validation",
        action="store_const",
        const=True,
        default=None,
        help="не держать готовые batch validation в памяти",
    )
    parser.add_argument("--best-metric", choices=BEST_SCOPES, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="как часто переписывать last.pt независимо от оценок",
    )
    parser.add_argument(
        "--final-splits",
        type=split_list,
        default=None,
        help="наборы для единственной оценки после обучения, через запятую",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # Архитектура запуска. Ничего не передали значит прежняя
    # модель: умолчания ModelConfig не менялись.
    parser.add_argument("--d-model", type=int, default=None, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=None, dest="n_heads")
    parser.add_argument("--dim-feedforward", type=int, default=None, dest="dim_feedforward")
    parser.add_argument("--event-layers", type=int, default=None, dest="n_event_layers")
    parser.add_argument("--profile-layers", type=int, default=None, dest="n_profile_layers")
    parser.add_argument("--history-layers", type=int, default=None, dest="n_history_layers")


OVERRIDES: tuple[str, ...] = (
    "seed",
    "val_seed",
    "lr",
    "weight_decay",
    "gradient_clip",
    "warmup_steps",
    "max_steps",
    "log_every",
    "eval_every",
    "batch_size",
    "eval_batch_size",
    "max_events_per_history",
    "event_microbatch",
    "masking_mode",
    "balanced_share",
    "token_rate",
    "event_rate",
    "key_rate",
    "max_train_clients",
    "max_val_clients",
    "epochs",
    "accumulation_steps",
    "target_policy",
    "vocab_tag",
    "stream_validation",
    "best_metric",
    "checkpoint_every",
    "final_splits",
    "top_k",
    "dropout",
    *ARCHITECTURE_FIELDS,
)


def config_from_args(args) -> TrainConfig:

    # NOT_GIVEN отличает «флаг не передавали» от «передали
    # none»: второе это осмысленное значение, а не отсутствие.
    changes = {
        name: getattr(args, name)
        for name in OVERRIDES
        if getattr(args, name, NOT_GIVEN) is not NOT_GIVEN
        and getattr(args, name, None) is not None
    }

    for name in NULLABLE:
        value = getattr(args, name, NOT_GIVEN)
        if value is None:
            changes[name] = None

    return replace(TrainConfig(precision=args.precision), **changes)


def paths_from_args(args) -> tuple[Path, Path, Path]:
    """
    Тег словаря выбирает КОМПЛЕКТ: свой каталог artifacts и свой
    токенизированный датасет. Без тега пути прежние.
    """

    tag = getattr(args, "vocab_tag", None)

    return (
        args.root or default_tokenized_dir(args.name, tag),
        args.vocab or default_vocab_dir(args.name, tag),
        args.artifacts or default_artifacts_dir(args.name),
    )


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="MLM-обучение mini_pragma_v2: проверка, замер, короткий эксперимент"
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    overfit = sub.add_parser("overfit", help="запоминание крошечного фиксированного batch")
    add_common(overfit)
    overfit.add_argument("--steps", type=int, default=300)
    overfit.add_argument("--examples", type=int, default=2)
    overfit.add_argument("--overfit-lr", type=float, default=1e-2)
    overfit.add_argument("--overfit-events", type=int, default=64)

    measure = sub.add_parser("benchmark", help="скорость и память полного шага")
    add_common(measure)
    measure.add_argument("--warmup", type=int, default=5)
    measure.add_argument("--measured", type=int, default=20)

    inspect = sub.add_parser(
        "check", help="что лежит в checkpoint: без данных, модели и обучения"
    )
    inspect.add_argument("--checkpoint", type=Path, required=True)

    short = sub.add_parser("run", help="короткий эксперимент с фиксированным бюджетом шагов")
    add_common(short)
    short.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="продолжить ЭТОТ ЖЕ прогон с checkpoint (last.pt или interrupted.pt)",
    )

    compare = sub.add_parser(
        "compare", help="сравнение двух checkpoint'ов на фиксированных наборах масок"
    )
    add_common(compare)
    compare.add_argument("--old", type=Path, default=None, help="по умолчанию run без тега")
    compare.add_argument("--new", type=Path, default=None, help="по умолчанию run с тегом")
    compare.add_argument("--min-targets", type=int, default=30)
    compare.add_argument("--bootstrap", type=int, default=2000)

    return parser


def default_checkpoint(name: str, tag: str | None = None) -> Path:
    """
    Результат короткого run, если явный путь не задан.
    """

    directory = run_dir(name, "run", tag)

    for candidate in ("best.pt", "last.pt"):
        if (directory / candidate).exists():
            return directory / candidate

    raise SystemExit(
        f"не нашёл checkpoint в {directory}; укажите путь явно или сначала выполните run"
    )


def describe_checkpoint(path: Path) -> dict:
    """
    Короткая выжимка из checkpoint: чем продолжать и с чего.
    """

    payload = verify_checkpoint(Path(path))

    counters = payload.get("counters") or {}
    progress = payload.get("progress") or {}
    train_config = payload.get("train_config") or {}
    model_config = payload.get("model_config") or {}

    micro_done = progress.get("micro_done")
    micro_budget = progress.get("micro_budget")

    share = micro_done / micro_budget if micro_done and micro_budget else None

    print(f"файл                    {Path(path)}")
    print(f"размер, МБ              {Path(path).stat().st_size / (1 << 20):.1f}")
    print(f"версия checkpoint       {payload.get('version')}")
    print(f"секции                  {', '.join(sorted(payload))}")
    print()
    print(f"шагов оптимизатора      {counters.get('n_steps')}")
    print(f"batch'ей                {counters.get('n_batches')}, пропущено {counters.get('n_skipped')}")
    # Бюджет эпох есть только у обучения по эпохам: у шагового
    # знаменателя нет, и печатать "из None" нечестно.
    print(
        f"micro-batch             {micro_done} из {micro_budget} "
        f"({share * 100:.1f} %)"
        if micro_budget
        else f"micro-batch             {micro_done}, бюджет в шагах"
    )
    print(f"sampler                 эпоха {(payload.get('sampler') or {}).get('epoch')}, "
          f"позиция {(payload.get('sampler') or {}).get('position')}")
    print(f"оценки на шагах         {progress.get('evaluated_steps')}")
    print(f"лучший                  {progress.get('best')}")
    print(f"финальные наборы        {'сделаны' if progress.get('final') else 'нет'}")
    print()
    print(f"d_model                 {model_config.get('d_model')}, "
          f"слоёв {model_config.get('n_event_layers')}/{model_config.get('n_history_layers')}")
    print(f"политика целей          {train_config.get('target_policy')}")
    print(f"seed / val_seed         {train_config.get('seed')} / {train_config.get('val_seed')}")
    print(f"эпох / batch            {train_config.get('epochs')} / {train_config.get('batch_size')}")
    print(f"обрезка истории         {train_config.get('max_events_per_history')}")
    print()
    print(f"наборы validation       {sorted((payload.get('splits') or {}))}")
    print(f"отпечатки artifacts     {len(payload.get('artifacts') or {})} шт")

    return payload


def main() -> None:

    # Консоль Windows по умолчанию cp1251 и падает на Δ, а отчёт
    # уже записан на диск: терять его из-за печати нельзя.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()

    # Осмотр checkpoint не трогает ни данные, ни модель: он
    # отвечает на вопрос «чем продолжать», а не «продолжить».
    if args.mode == "check":
        describe_checkpoint(args.checkpoint)
        return

    root, vocab, artifacts = paths_from_args(args)

    config = config_from_args(args)

    folder = COMPARISON_DIR if args.mode == "compare" else args.mode

    out = prepare_dir(
        args.out or run_dir(args.name, folder, args.tag),
        args.force,
        keep=getattr(args, "resume", None) is not None,
    )

    env = load_environment(root, vocab, artifacts, epsilon=config.epsilon)

    print(f"словарь {env.vocab.size}, обучаемых полей {len(env.table.trainable_field_ids)}, "
          f"вырожденных {len(env.table.degenerate_field_ids)}")

    if env.unigram.missing:
        print(f"без unigram-baseline: {env.unigram.missing}")

    print(f"каталог запуска: {out}")
    print()

    if args.mode == "overfit":

        report = tiny_overfit(
            env,
            config,
            out,
            steps=args.steps,
            n_examples=args.examples,
            lr=args.overfit_lr,
            max_events=args.overfit_events,
            device=args.device,
        )

        if not report["passed"]:
            raise SystemExit(
                "tiny overfit не пройден: к основному эксперименту переходить нельзя, "
                "нужно разбирать mapping целей, маскирование и градиенты"
            )

        return

    if args.mode == "benchmark":
        benchmark(env, config, out, warmup=args.warmup, measured=args.measured, device=args.device)
        return

    if args.mode == "compare":

        old = args.old or default_checkpoint(args.name)
        new = args.new or default_checkpoint(args.name, args.tag)

        if Path(old) == Path(new):
            raise SystemExit("старый и новый checkpoint это один файл: сравнивать нечего")

        print(f"старый checkpoint: {old}")
        print(f"новый checkpoint:  {new}")
        print("конфигурация и маски берутся из checkpoint'ов, флаги обучения игнорируются")
        print()

        # Арки с разными словарями живут в разных token-простран-
        # ствах: каждой нужен свой комплект artifacts, а равенство
        # задачи доказывается инвариантным отпечатком целей.
        new_env = None

        if args.new_vocab_tag is not None:

            new_env = load_environment(
                default_tokenized_dir(args.name, args.new_vocab_tag),
                default_vocab_dir(args.name, args.new_vocab_tag),
                default_artifacts_dir(args.name),
                epsilon=config.epsilon,
            )

        run_comparison(
            env,
            old,
            new,
            out,
            device=args.device,
            min_targets=args.min_targets,
            n_boot=args.bootstrap,
            new_env=new_env,
        )

        return

    # Полные истории заявляются, а не предполагаются: если
    # лимит снят, preflight проходит по всем примерам и
    # подтверждает, что ни один не обрезан.
    try:
        run_training(
            env,
            config,
            out,
            device=args.device,
            preflight=config.max_events_per_history is None,
            resume=getattr(args, "resume", None),
        )
    except TrainingInterrupted as error:
        # Остановка по просьбе это не ошибка запуска: код 130,
        # как у обычного Ctrl+C, и понятная строка о том, чем
        # продолжить.
        print(str(error))
        raise SystemExit(130) from None
    except TrainingAborted as error:
        print(str(error))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()


__all__ = ["main", "build_parser", "config_from_args", "run_dir"]
