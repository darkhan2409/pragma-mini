from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .settings import GROUPS, PreprocessingConfig, group_dir, normalize_group, raw_group_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Точка входа препроцессинга. Команда ровно одна:
#
#   python -m src.preprocessing.run preprocess <group>
#
# Она читает data/01_raw/<group>, проверяет выгрузку, раскрывает
# payload, приводит значения к объявленным типам, упорядочивает
# события клиента и кладёт ДВА файла в data/02_preprocessed/<group>.
#
# Ни истории на дату, ни разделения, ни смыслового слоя рядом
# больше нет: срез применяется позже, при сборке датасета, а
# производных признаков не считает никто.
#
# Состояния между запусками НЕТ: ни отпечатков, ни маркеров, ни
# манифеста. Каждый запуск переписывает каталог своей группы.
# ============================================================


EXIT_OK = 0
EXIT_BLOCKED = 2


def clean_directory(path: Path) -> None:

    if not path.exists():
        return

    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        else:
            item.rmdir()


def run_preprocess(args) -> int:

    from .canonical.build import STAGE, build_group
    from .canonical.events import CanonicalError
    from .rawdata import RawContractError

    config = PreprocessingConfig.load(Path(args.config) if args.config else None)

    group = normalize_group(args.group)

    raw_dir = raw_group_dir(group)
    target = group_dir(group)

    if not raw_dir.exists():
        print(f"[{STAGE}] группа {group}: нет выгрузки {raw_dir}")
        return EXIT_BLOCKED

    try:
        result = build_group(raw_dir, target, config, group)
    except (CanonicalError, RawContractError) as error:
        # Непригодный вход: этап останавливается с названием
        # строки и причиной, а частичного слоя не оставляет.
        print(f"[{STAGE}] группа {group}: {error}")
        clean_directory(target)
        return EXIT_BLOCKED

    print(
        f"[{STAGE}] группа {group}: событий {result.events_rows}, "
        f"клиентов {result.clients} → {target}"
    )
    print("    анкета не копируется: следующие этапы читают её из "
          f"{raw_dir / 'profile.parquet'}")

    return EXIT_OK


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.preprocessing.run")

    subparsers = parser.add_subparsers(dest="stage", required=True)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="проверка выгрузки и очищенная группа в data/02_preprocessed/<group>",
    )
    preprocess.add_argument("group", choices=GROUPS, help="группа: train, val или test")
    preprocess.add_argument("--config", type=Path, default=None, help="JSON с переопределениями конфига")
    preprocess.set_defaults(handler=run_preprocess)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
