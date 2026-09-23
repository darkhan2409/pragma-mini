from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group
from src.tokenization.specials import SpecialsError

from .batch import BatchError
from .build import build_group
from .temporal import TemporalError
from .settings import BatchingConfig, ConfigError


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу, один видимый файл:
#
#   python -m src.batching.run train|val|test
#
# Вход: data/06_temporal/<group>/temporal.parquet и код [PAD] из
# data/03_vocab/special_tokens.json.
# Выход: data/07_batches/<group>/batches.parquet и ничего больше.
#
# Этап ничего не маскирует, не выбирает значения для
# предсказания и не обучается.
# ============================================================


FAILURES = (BatchError, ConfigError, SpecialsError, TemporalError)


def run_group(args) -> int:

    group = normalize_group(args.group)

    try:
        config = BatchingConfig.load(Path(args.config) if args.config else None)

        report = build_group(group, config)

    except FAILURES as error:
        print(f"[batching] группа {group}: {error}")
        return EXIT_BLOCKED

    counts = report["counts"]

    print(f"[batching] группа {group} → {report['file']}")
    print(
        f"    батчей {counts['batches']} по {report['batch_size']} клиентов "
        f"(в последнем {counts['last_batch']}); клиентов {counts['clients']}, "
        f"молчащих {counts['silent_clients']}"
    )
    print(
        f"    токенов настоящих {counts['tokens_real']}, заполнителей "
        f"{counts['tokens_slots'] - counts['tokens_real']} "
        f"({_share(counts['tokens_real'], counts['tokens_slots'])}); "
        f"ширина батча до {counts['max_width']}"
    )
    print(
        f"    событий настоящих {counts['events_real']}, заполнителей "
        f"{counts['events_slots'] - counts['events_real']} "
        f"({_share(counts['events_real'], counts['events_slots'])})"
    )
    print(
        f"    токенов анкеты настоящих {counts['profile_tokens_real']}, заполнителей "
        f"{counts['profile_tokens_slots'] - counts['profile_tokens_real']} "
        f"({_share(counts['profile_tokens_real'], counts['profile_tokens_slots'])})"
    )
    print(
        f"    окно сортировки {report['window_clients']} клиентов, "
        f"[PAD] {report['pad_id']}"
    )

    return EXIT_OK


def _share(real: int, slots: int) -> str:
    """
    Доля заполнителя строкой.
    """

    return "0.0%" if slots == 0 else f"{100.0 * (slots - real) / slots:.1f}%"


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.batching.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--config", type=Path, default=None, help="JSON с переопределениями конфига батчей"
    )

    parser.set_defaults(handler=run_group)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
