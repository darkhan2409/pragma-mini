from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .settings import ConfigError, MlmConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу:
#
#   python -m src.mlm.run train|val|test
#
# Вход: data/07_batches и data/08_masked; веса энкодеров из
# data/09_embeddings, 10_events, 11_profiles и 12_history.
# Выход: data/13_mlm/<group>/ — результаты по целям, страница и
# веса головы.
#
# Этап считает потери необученной модели. Обучения здесь нет.
# ============================================================


def run_group(args) -> int:

    group = normalize_group(args.group)

    try:
        from .build import MlmError, build_group
        from .inputs import InputError

    except ModuleNotFoundError as error:
        print(
            f"[mlm] нет модуля {error.name}: этап считает тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    try:
        config = MlmConfig.load(Path(args.config) if args.config else None)

        report = build_group(group, config)

    except (ConfigError, InputError, MlmError, FileNotFoundError) as error:
        print(f"[mlm] группа {group}: {error}")
        return EXIT_BLOCKED

    print(f"[mlm] группа {group} → {report['targets_file']}")
    print(f"    страница {report['preview']}")
    print(f"    веса головы {report['weights']}")
    print(
        f"    d {report['dim']}, seed {report['seed']}, сглаживание "
        f"{report['label_smoothing']}; считано на {report['device']}"
    )
    print(
        f"    клиентов {report['clients']}, событий {report['events']}; "
        f"целей {report['targets']} ({report['by_reason']})"
    )
    print(
        f"    кросс-энтропия {report['loss']:.4f}, угадано {report['correct']} "
        f"({100.0 * report['correct'] / max(report['targets'], 1):.2f}%) — "
        "модель НЕ обучена, это не оценка качества"
    )
    print(
        f"    целей с исходным [UNK]: {report['unknown']} "
        f"({100.0 * report['unknown'] / max(report['targets'], 1):.1f}%)"
    )
    print(
        f"    на диске {report['size'] / 1024:.0f} КиБ"
        + (f", пик {report['peak']:.0f} МиБ" if report["peak"] else "")
    )

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.mlm.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига головы",
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
