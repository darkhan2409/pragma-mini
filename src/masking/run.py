from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group
from src.tokenization.specials import SpecialsError

from .apply import MaskError
from .batches import BatchesError
from .build import build_group
from .settings import ConfigError, MaskingConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу, один видимый файл:
#
#   python -m src.masking.run train|val|test
#
# Вход: data/07_batches/<group>/batches.parquet и коды [MASK] и
# [UNK] из data/03_vocab/special_tokens.json.
# Выход: data/08_masked/<group>/masked.parquet и ничего больше.
#
# Этап решает, что спрятать и что предсказывать. Модели и
# обучения здесь нет.
# ============================================================


FAILURES = (BatchesError, ConfigError, MaskError, SpecialsError)


def run_group(args) -> int:

    group = normalize_group(args.group)

    try:
        config = MaskingConfig.load(Path(args.config) if args.config else None)

        report = build_group(group, config)

    except FAILURES as error:
        print(f"[masking] группа {group}: {error}")
        return EXIT_BLOCKED

    counts = report["counts"]

    print(f"[masking] группа {group} → {report['file']}")
    print(
        f"    клиентов {counts['clients']}, батчей {counts['batches']}; "
        f"допустимых событий {counts['events']}, значений {counts['values']}"
    )
    print(
        f"    выбрано {counts['chosen']} ({_share(counts['chosen'], counts['values'])}): "
        f"event {counts['by_event']}, key {counts['by_key']}, value {counts['by_value']}"
    )
    print(
        f"    под [MASK] {counts['masked']}, под [UNK] {counts['unknown']} "
        f"({_share(counts['unknown'], counts['chosen'])}); "
        f"в loss токенов {counts['labelled_tokens']}"
    )
    print(
        f"    без допустимых целей {counts['without_targets']} клиентов; "
        f"seed {report['seed']}"
    )

    return EXIT_OK


def _share(part: int, whole: int) -> str:
    """
    Доля строкой.
    """

    return "0.0%" if whole == 0 else f"{100.0 * part / whole:.1f}%"


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.masking.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига маскирования",
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
