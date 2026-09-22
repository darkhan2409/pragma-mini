from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group
from src.tokenization.finalvocab import FrozenArtifacts, VocabError

from .build import BuildError, build_group
from .context import ContextError
from .sample import SampleError
from .settings import ConfigError, DatasetConfig
from .tokenized import TokenizedError


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда собирает одну группу:
#
#   python -m src.dataset.run train
#   python -m src.dataset.run val
#   python -m src.dataset.run test
#
# Вход: data/04_tokenized/<group>/ и словарь из data/03_vocab/.
# Выход: data/05_dataset/<group>/samples.parquet и ничего больше.
#
# Словарь один на все группы — тот, что обучен на train. Другого
# у набора быть не может: одинаковые ID у разных групп это и
# есть условие сравнимости.
# ============================================================


FAILURES = (BuildError, ConfigError, ContextError, SampleError, TokenizedError, VocabError)


def run_group(args) -> int:

    group = normalize_group(args.group)

    try:
        config = DatasetConfig.load(Path(args.config) if args.config else None)

        artifacts = FrozenArtifacts.load()

        report = build_group(artifacts, group, config)

    except FAILURES as error:
        print(f"[dataset] группа {group}: {error}")
        return EXIT_BLOCKED

    counts = report["counts"]

    print(f"[dataset] группа {group} на срез {report['cutoff'][:10]} → {report['file']}")
    print(
        f"    примеров {counts['samples']}, событий {counts['events']}, "
        f"значений {counts['values']}, токенов {counts['tokens']} "
        f"(из них на профили {counts['profile_tokens']})"
    )
    print(
        f"    в периоде целей событий {counts['eligible_events']} у "
        f"{counts['samples_with_targets']} примеров; молчащих клиентов {counts['silent_clients']}, "
        f"пустых анкет {counts['empty_profiles']}"
    )
    print(
        f"    усечено примеров {counts['truncated']} (событий за границей "
        f"{counts['excluded_events']}); токенов в примере до {counts['max_tokens']}"
    )

    print(f"    [UNK] значений {report['unknown_values']}")

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.dataset.run")

    parser.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    parser.add_argument(
        "--config", type=Path, default=None, help="JSON с переопределениями конфига датасета"
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
