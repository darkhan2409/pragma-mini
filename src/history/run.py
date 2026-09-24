from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .settings import ConfigError, HistoryConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу:
#
#   python -m src.history.run train|val|test
#
# Вход: временные позиции из data/07_batches, векторы событий из
# data/10_events и вектор анкеты из data/11_profiles.
# Выход: data/12_history/<group>/ — итоговый вектор клиента и
# веса энкодера. Векторы событий после истории остаются в памяти
# для MLM-головы и на диск не пишутся.
#
# MLM-голова и обучение — не здесь.
# ============================================================


def run_group(args) -> int:

    group = normalize_group(args.group)

    try:
        from .build import HistoryError, build_group
        from .inputs import InputError

    except ModuleNotFoundError as error:
        print(
            f"[history] нет модуля {error.name}: этап считает тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    try:
        config = HistoryConfig.load(Path(args.config) if args.config else None)

        report = build_group(group, config)

    except (ConfigError, HistoryError, InputError) as error:
        print(f"[history] группа {group}: {error}")
        return EXIT_BLOCKED

    print(f"[history] группа {group} → {report['table']}")
    print(f"    веса {report['weights']}")
    print(
        f"    клиентов {report['clients']}, событий {report['events']}, "
        f"самая длинная история {report['longest']}; {report['seconds']:.1f} с "
        f"на {report['device']}"
    )

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.history.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига энкодера истории",
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
