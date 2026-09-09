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

from .ablation import run_ablation
from .compare import run_comparison
from .diagnostics import run_cluster_bootstrap
from .full_history import ABLATION_SPLITS, CLIENT_UNIVERSE, run_full_history
from .history_coverage import WINDOWS, run_history_coverage
from .history_encoder import ATTENTION_RULES
from .config import STRUCTURES
from src.tokenizer.masking import SCHEMES

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
#   ablate      диагностика внимания на готовом checkpoint
#
# Каждый режим строит СВОЮ модель: сравнивать скорость на
# частично обученных весах или продолжать run после overfit
# значит мерить и обучать разные вещи. Исключение это ablate:
# он берёт и веса, и конфигурацию из checkpoint, потому что
# обязан воспроизвести обучение, а не задать его заново.
#
# Каталог запуска не перезаписывается молча: результат прошлого
# эксперимента дороже удобства.
# ============================================================


MODES: tuple[str, ...] = (
    "check",
    "overfit", "benchmark", "run", "ablate", "compare", "diagnose", "full-history",
)

ABLATION_DIR = "ablation"
COMPARISON_DIR = "comparison"
DIAGNOSTICS_DIR = "diagnostics"
FULL_HISTORY_TAG = "full_history"


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
    parser.add_argument("--mask-scheme", choices=SCHEMES, default=None)
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
    parser.add_argument("--structure", choices=STRUCTURES, default=None)
    parser.add_argument("--d-model", type=int, default=None, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=None, dest="n_heads")
    parser.add_argument("--dim-feedforward", type=int, default=None, dest="dim_feedforward")
    parser.add_argument("--event-layers", type=int, default=None, dest="n_event_layers")
    parser.add_argument("--profile-layers", type=int, default=None, dest="n_profile_layers")
    parser.add_argument("--history-layers", type=int, default=None, dest="n_history_layers")
    parser.add_argument("--session-layers", type=int, default=None, dest="n_session_layers")


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
    "mask_scheme",
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
    return (
        args.root or default_tokenized_dir(args.name),
        args.vocab or default_vocab_dir(args.name),
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

    ablate = sub.add_parser(
        "ablate", help="диагностика внимания History Encoder на готовом checkpoint"
    )
    add_common(ablate)
    ablate.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="по умолчанию best.pt последнего run, иначе last.pt",
    )
    ablate.add_argument("--rules", default=",".join(ATTENTION_RULES))

    compare = sub.add_parser(
        "compare", help="сравнение двух checkpoint'ов на фиксированных наборах масок"
    )
    add_common(compare)
    compare.add_argument("--old", type=Path, default=None, help="по умолчанию run без тега")
    compare.add_argument("--new", type=Path, default=None, help="по умолчанию run с тегом")
    compare.add_argument("--min-targets", type=int, default=30)
    compare.add_argument("--bootstrap", type=int, default=2000)
    compare.add_argument("--rules", default=",".join(ATTENTION_RULES))

    diagnose = sub.add_parser(
        "diagnose", help="кластерный bootstrap и календарная глубина истории, без обучения"
    )
    add_common(diagnose)
    diagnose.add_argument("--old", type=Path, default=None)
    diagnose.add_argument("--new", type=Path, default=None)
    diagnose.add_argument("--bootstrap", type=int, default=2000)
    diagnose.add_argument("--min-targets", type=int, default=30)
    diagnose.add_argument("--rules", default=",".join(ATTENTION_RULES))
    diagnose.add_argument(
        "--windows", default=",".join("full" if w is None else str(w) for w in WINDOWS)
    )
    diagnose.add_argument("--train-coverage-clients", type=int, default=512)
    diagnose.add_argument("--skip-bootstrap", action="store_true")
    diagnose.add_argument("--skip-coverage", action="store_true")
    diagnose.add_argument("--skip-links", action="store_true")

    full = sub.add_parser(
        "full-history", help="полные истории без обрезки, одна эпоха на выбранном наборе клиентов"
    )
    add_common(full)
    full.add_argument("--clients", type=int, default=CLIENT_UNIVERSE)
    full.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="имя=путь; по умолчанию прежние combined и field_balanced checkpoint'ы",
    )
    full.add_argument("--ablation-splits", default=",".join(ABLATION_SPLITS))
    full.add_argument("--bootstrap", type=int, default=2000)
    full.add_argument("--skip-benchmark", action="store_true")

    return parser


def default_baselines(name: str) -> dict[str, Path]:
    """
    Прежние checkpoint'ы: combined отличается от нового только
    контекстом и объёмом, field_balanced это исходный эксперимент.
    """

    candidates = {
        "combined_128": run_dir(name, "run", "combined") / "best.pt",
        "field_balanced_128": run_dir(name, "run") / "best.pt",
    }

    return {label: path for label, path in candidates.items() if path.exists()}


def parse_windows(text: str) -> tuple[int | None, ...]:
    return tuple(
        None if value.strip() == "full" else int(value)
        for value in text.split(",")
        if value.strip()
    )


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
    print(f"micro-batch эпохи       {micro_done} из {micro_budget}"
          + (f" ({share * 100:.1f} %)" if share else ""))
    print(f"sampler                 эпоха {(payload.get('sampler') or {}).get('epoch')}, "
          f"позиция {(payload.get('sampler') or {}).get('position')}")
    print(f"оценки на шагах         {progress.get('evaluated_steps')}")
    print(f"лучший                  {progress.get('best')}")
    print(f"финальные наборы        {'сделаны' if progress.get('final') else 'нет'}")
    print()
    print(f"структура               {model_config.get('structure')}, d_model {model_config.get('d_model')}")
    print(f"политика целей          {train_config.get('target_policy')}")
    print(f"схема масок             {train_config.get('mask_scheme')}")
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

    folder = {
        "ablate": ABLATION_DIR,
        "compare": COMPARISON_DIR,
        "diagnose": DIAGNOSTICS_DIR,
    }.get(args.mode, args.mode)

    if args.mode == "full-history":
        out = prepare_dir(args.out or RUNS_DIR / args.name / FULL_HISTORY_TAG, args.force)
    else:
        out = prepare_dir(
            args.out or run_dir(args.name, folder, args.tag),
            args.force,
            keep=getattr(args, "resume", None) is not None,
        )

    env = load_environment(root, vocab, artifacts, epsilon=config.epsilon)

    print(f"словарь {env.vocab.size}, обучаемых полей {len(env.table.trainable_key_ids)}, "
          f"вырожденных {len(env.table.degenerate_key_ids)}")

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

    if args.mode == "full-history":

        settings = replace(
            config,
            client_universe=args.clients,
            max_train_clients=None,
            max_val_clients=None,
            max_events_per_history=None,
            epochs=1,
            masking_mode=args.masking_mode or "combined",
            token_rate=args.token_rate if args.token_rate is not None else 0.15,
            event_rate=args.event_rate if args.event_rate is not None else 0.10,
            key_rate=args.key_rate if args.key_rate is not None else 0.10,
            eval_batch_size=args.eval_batch_size or 2,
            log_every=args.log_every or 200,
            eval_every=args.eval_every or 2000,
        )

        baselines = (
            {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.baseline}
            if args.baseline
            else default_baselines(args.name)
        )

        print(f"клиентов: {settings.client_universe}, лимит истории: "
              f"{settings.max_events_per_history}, эпох: {settings.epochs}")
        print("baseline'ы: " + (", ".join(f"{k} -> {v}" for k, v in baselines.items()) or "нет"))
        print()

        run_full_history(
            env,
            out,
            settings,
            baselines,
            device=args.device,
            ablation_splits=tuple(
                value.strip() for value in args.ablation_splits.split(",") if value.strip()
            ),
            n_boot=args.bootstrap,
            skip_benchmark=args.skip_benchmark,
        )

        return

    if args.mode == "diagnose":

        if not args.skip_bootstrap:

            old = args.old or default_checkpoint(args.name)
            new = args.new or default_checkpoint(args.name, args.tag)

            print(f"старый checkpoint: {old}")
            print(f"новый checkpoint:  {new}")
            print("обучения нет: веса заморожены, маски и цели те же")
            print()

            run_cluster_bootstrap(
                env,
                old,
                new,
                out,
                device=args.device,
                rules=tuple(value.strip() for value in args.rules.split(",") if value.strip()),
                n_boot=args.bootstrap,
                min_targets=args.min_targets,
            )

        if not args.skip_coverage:

            from src.tokenizer.config import processed_dir as default_processed_dir

            print()
            print("календарная глубина recent-окон")
            print()

            run_history_coverage(
                default_processed_dir(args.name),
                out,
                splits={
                    "train": args.train_coverage_clients,
                    "val_client": config.max_val_clients,
                    "val_time": config.max_val_clients,
                },
                windows=parse_windows(args.windows),
                with_links=not args.skip_links,
            )

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

        run_comparison(
            env,
            old,
            new,
            out,
            device=args.device,
            rules=tuple(value.strip() for value in args.rules.split(",") if value.strip()),
            min_targets=args.min_targets,
            n_boot=args.bootstrap,
        )

        return

    if args.mode == "ablate":

        checkpoint = args.checkpoint or default_checkpoint(args.name, args.tag)

        print(f"checkpoint: {checkpoint}")
        print("конфигурация и маски validation берутся из него, флаги обучения игнорируются")
        print()

        run_ablation(
            env,
            checkpoint,
            out,
            device=args.device,
            rules=tuple(value.strip() for value in args.rules.split(",") if value.strip()),
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
