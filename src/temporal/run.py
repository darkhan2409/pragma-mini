from __future__ import annotations

import argparse
import sys

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .build import build_group
from .position import TemporalError
from .samples import SamplesError


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу, один видимый файл:
#
#   python -m src.temporal.run train|val|test
#
# Вход: data/05_dataset/<group>/samples.parquet.
# Выход: data/06_temporal/<group>/temporal.parquet и ничего
# больше.
#
# Этап добавляет к примеру временные позиции событий. TimeRoPE
# и модель он не содержит.
# ============================================================


FAILURES = (SamplesError, TemporalError)


def run_group(args) -> int:

    group = normalize_group(args.group)

    try:
        report = build_group(group)

    except FAILURES as error:
        print(f"[temporal] группа {group}: {error}")
        return EXIT_BLOCKED

    counts = report["counts"]

    print(f"[temporal] группа {group} → {report['file']}")
    print(
        f"    клиентов {counts['clients']}, событий {counts['events']}; "
        f"молчащих {counts['silent_clients']}"
    )
    print(
        f"    самая дальняя позиция {counts['max_position']:.2f} "
        f"при истории до {counts['max_days']:.1f} суток"
    )

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.temporal.run")

    # Настроек у этапа нет, поэтому нет и --config: переопределять
    # тут нечего.
    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")

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
