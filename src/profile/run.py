from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .settings import ConfigError, ProfileConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу:
#
#   python -m src.profile.run train|val|test
#
# Вход: анкета из data/07_batches и веса входного слоя из
# data/09_embeddings.
# Выход: data/11_profiles/<group>/ — вектор на клиента и веса
# энкодера.
#
# Этап сворачивает анкету клиента в один вектор. History Encoder
# и обучение — не здесь.
# ============================================================


def run_group(args) -> int:

    group = normalize_group(args.group)

    # torch тянется здесь, а не наверху файла: он объявлен
    # необязательным дополнением.
    try:
        from src.embedding.inputs import InputError
        from src.tokenization.specials import SpecialsError

        from .build import ProfileError, build_group

    except ModuleNotFoundError as error:
        print(
            f"[profile] нет модуля {error.name}: этап считает тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    try:
        config = ProfileConfig.load(Path(args.config) if args.config else None)

        report = build_group(group, config)

    except (ConfigError, InputError, ProfileError, SpecialsError) as error:
        print(f"[profile] группа {group}: {error}")
        return EXIT_BLOCKED

    print(f"[profile] группа {group} → {report['table']}")
    print(f"    веса {report['weights']}")
    print(
        f"    клиентов {report['clients']}, токенов анкет {report['tokens']}; "
        f"{report['size'] / 1024:.0f} КиБ"
    )

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.profile.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига энкодера анкеты",
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
