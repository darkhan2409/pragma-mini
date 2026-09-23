from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .settings import ConfigError, EventConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу:
#
#   python -m src.event.run train|val|test
#
# Вход: data/07_batches, data/08_masked и веса входного слоя из
# data/09_embeddings.
# Выход: data/10_events/<group>/ — вектор каждого настоящего
# события и веса энкодера.
#
# Этап сворачивает токены события в вектор события. History
# Encoder, профиль, TimeRoPE и обучение — не здесь.
# ============================================================


def run_group(args) -> int:

    group = normalize_group(args.group)

    # torch тянется здесь, а не наверху файла: он объявлен
    # необязательным дополнением, и без него команда обязана
    # сказать это внятно, а не упасть трассировкой импорта.
    try:
        from src.embedding.inputs import InputError
        from src.tokenization.specials import SpecialsError

        from .build import EventError, build_group
        from .gather import GatherError

    except ModuleNotFoundError as error:
        print(
            f"[event] нет модуля {error.name}: этап считает тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    try:
        config = EventConfig.load(Path(args.config) if args.config else None)

        report = build_group(group, config)

    except (ConfigError, EventError, GatherError, InputError, SpecialsError) as error:
        print(f"[event] группа {group}: {error}")
        return EXIT_BLOCKED

    print(f"[event] группа {group} → {report['table']}")
    print(f"    веса {report['weights']}")
    print(
        f"    клиентов {report['clients']}, событий {report['events']} "
        f"по {report['dim']} чисел; {report['size'] / (1 << 20):.1f} МиБ"
    )

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.event.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига энкодера события",
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
