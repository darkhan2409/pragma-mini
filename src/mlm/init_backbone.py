from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK


# ============================================================
# ИДЕЯ
# ============================================================
#
# Одна команда, без группы:
#
#   python -m src.mlm.init_backbone [--event-config путь]
#                                   [--profile-config путь]
#                                   [--history-config путь]
#
# Вход: веса входного слоя data/09_embeddings/train и словарь.
# Выход: data/09_backbone/ — начальные веса энкодеров события,
# анкеты и истории и lineage.json.
#
# Данные не читаются и через энкодеры ничего не проходит: это
# секунды, а не проход по всем событиям. Конфиги — те же классы и
# тот же формат JSON, что у этапов 10–12; без файла — значения по
# умолчанию, они же итоговая архитектура.
# ============================================================


def run(args) -> int:

    # torch тянется здесь, а не наверху файла: без него команда
    # обязана сказать, что поставить, а не упасть трассировкой.
    try:
        from src.event.settings import ConfigError as EventConfigError
        from src.event.settings import EventConfig
        from src.history.settings import ConfigError as HistoryConfigError
        from src.history.settings import HistoryConfig
        from src.profile.settings import ConfigError as ProfileConfigError
        from src.profile.settings import ProfileConfig

        from .backbone import BackboneError, init_backbone

    except ModuleNotFoundError as error:
        print(
            f"[init-backbone] нет модуля {error.name}: веса — тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    def path(value: str | None) -> Path | None:
        return Path(value) if value else None

    try:
        report = init_backbone(
            EventConfig.load(path(args.event_config)),
            ProfileConfig.load(path(args.profile_config)),
            HistoryConfig.load(path(args.history_config)),
        )

    except (BackboneError, EventConfigError, ProfileConfigError, HistoryConfigError) as error:
        print(f"[init-backbone] {error}")
        return EXIT_BLOCKED

    print(f"[init-backbone] начальные веса → {report['directory']}")

    for name, item in report["encoders"].items():
        config = item["config"]
        print(
            f"    {name}: блоков {item['blocks']}, голов {config['heads']}, FFN "
            f"{config['feedforward']}, seed {config['seed']}; параметров {item['parameters']:,}"
        )

    print(
        f"    d {report['dim']}; на диске {report['bytes'] / (1 << 20):.2f} МиБ; "
        f"{report['seconds']:.2f} с"
    )

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.mlm.init_backbone")

    parser.add_argument("--event-config", default=None,
                        help="JSON с переопределениями конфига энкодера события")
    parser.add_argument("--profile-config", default=None,
                        help="JSON с переопределениями конфига энкодера анкеты")
    parser.add_argument("--history-config", default=None,
                        help="JSON с переопределениями конфига энкодера истории")

    parser.set_defaults(handler=run)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
