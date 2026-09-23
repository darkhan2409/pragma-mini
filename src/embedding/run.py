from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .settings import ConfigError, EmbeddingConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда на группу, один видимый файл:
#
#   python -m src.embedding.run train|val|test
#
# Вход: data/07_batches/<group>/batches.parquet и
# data/08_masked/<group>/masked.parquet, оба сразу.
# Выход: data/09_embeddings/<group>/preview.html и веса слоя
# рядом с ним.
#
# Этап считает вход модели и показывает его человеку. Энкодеров,
# внимания, MLM-головы и обучения здесь нет.
# ============================================================


def run_group(args) -> int:

    group = normalize_group(args.group)

    # torch тянется здесь, а не наверху файла: он объявлен
    # необязательным дополнением, и без него команда обязана
    # сказать это внятно, а не упасть трассировкой импорта.
    try:
        from src.tokenization.finalvocab import VocabError
        from src.tokenization.specials import SpecialsError

        from .build import build_group
        from .inputs import InputError

    except ModuleNotFoundError as error:
        print(
            f"[embedding] нет модуля {error.name}: этап считает тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    try:
        config = EmbeddingConfig.load(Path(args.config) if args.config else None)

        report = build_group(group, config, args.batch)

    except (ConfigError, InputError, SpecialsError, VocabError) as error:
        print(f"[embedding] группа {group}: {error}")
        return EXIT_BLOCKED

    dim = report["dim"]

    print(f"[embedding] группа {group} → {report['table']}")
    print(f"    страница {report['preview']}")
    print(f"    веса {report['weights']}")
    print(
        f"    словарь {report['vocab_size']} ID, d {dim}, seed {report['seed']}; "
        f"батчей {report['batches']}, клиентов {report['clients']}"
    )
    print(
        f"    векторов {report['numbers'] // dim} по {dim} чисел "
        f"({report['numbers']} чисел, {report['size'] / (1 << 20):.1f} МиБ на диске)"
    )
    print(
        f"    в показанном батче {report['batch']}: сверено значений "
        f"{report['compared_values']}, маркеров {report['markers']}, "
        f"мест заполнителя {report['pad_slots']}"
    )
    print(
        f"    на странице клиент {report['client_id']}, событие {report['event']}"
    )

    if report["note"]:
        print(f"    {report['note']}")

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.embedding.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--batch", type=int, default=0, help="номер батча в файле группы"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига эмбеддингов",
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
