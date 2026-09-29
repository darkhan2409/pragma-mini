from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# КОНВЕЙЕР ОДНОЙ КОМАНДОЙ
# ============================================================
#
#   python -m src.pipeline                          02 → 05, 09 и init_backbone, все группы
#   python -m src.pipeline --from encode --to dataset --groups val
#   python -m src.pipeline --args dataset="--config ctx.json"
#
# Цепочка этапов записана один раз, здесь. Каждый этап — та же
# команда, что запускают руками, отдельным процессом: память
# между этапами возвращается системе, а поведение этапа не
# отличается от ручного запуска.
#
# Строится только то, что читает обучение и оценка. Промежуточное
# после 04 и 05 на диске не хранится: временные позиции и маски
# считаются при чтении набора (src.temporal.samples, src.masking).
# Веса входного слоя нужны только train — модель одна. Генерация
# (01) и обучение (14) сюда не входят: это долгие запуски отдельной
# командой.
# ============================================================


GROUPS = ("train", "val", "test")

# (имя, модуль и аргументы, группы). "{group}" заменяется группой;
# None — этап один на все группы.
STAGES: tuple[tuple[str, tuple[str, ...], tuple[str, ...] | None], ...] = (
    ("preprocess", ("src.preprocessing.run", "preprocess", "{group}"), GROUPS),
    ("fit", ("src.tokenization.run", "fit"), None),
    ("encode", ("src.tokenization.run", "encode", "{group}"), GROUPS),
    ("dataset", ("src.dataset.run", "{group}"), GROUPS),
    ("embeddings", ("src.embedding.run", "train"), None),
    ("backbone", ("src.mlm.init_backbone",), None),
)

NAMES = tuple(name for name, _, _ in STAGES)


def commands(first: str, last: str, groups: tuple[str, ...], extra: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """
    Команды этапов от first до last включительно, по порядку.
    """

    begin, end = NAMES.index(first), NAMES.index(last)

    if begin > end:
        raise ValueError(f"этап {first} идёт после {last}")

    planned = []

    for name, template, applies in STAGES[begin:end + 1]:

        targets = [group for group in (applies or (None,)) if applies is None or group in groups]

        for group in targets:
            argv = [part.format(group=group) for part in template]
            planned.append((name, [sys.executable, "-m", *argv, *extra.get(name, [])]))

    return planned


def run(args) -> int:

    extra: dict[str, list[str]] = {}

    for item in args.args:
        name, _, text = item.partition("=")
        if name not in NAMES:
            print(f"[pipeline] нет этапа {name!r}: есть {', '.join(NAMES)}")
            return 2
        extra[name] = shlex.split(text)

    try:
        planned = commands(args.start, args.stop, tuple(args.groups), extra)
    except ValueError as error:
        print(f"[pipeline] {error}")
        return 2

    timings = []

    for name, argv in planned:

        print(f"[pipeline] {name}: {' '.join(argv[2:])}", flush=True)

        started = time.perf_counter()

        code = subprocess.call(argv)

        seconds = time.perf_counter() - started

        timings.append({"stage": name, "command": argv[2:], "seconds": seconds, "code": code})

        if code != 0:
            print(f"[pipeline] остановлено на {name} (код {code}); дальше не идём")
            break

    print("[pipeline] время по этапам:")

    for item in timings:
        print(f"    {item['stage']:<11} {' '.join(item['command'][1:]):<22} {item['seconds'] / 60:6.1f} мин")

    print(f"    всего {sum(item['seconds'] for item in timings) / 60:.1f} мин")

    if args.report is not None:
        args.report.write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")

    return timings[-1]["code"] if timings else 0


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.pipeline")
    parser.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0])
    parser.add_argument("--to", dest="stop", choices=NAMES, default=NAMES[-1])
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument(
        "--args", action="append", default=[], metavar="ЭТАП=АРГУМЕНТЫ",
        help='дополнительные аргументы этапа, например dataset="--config ctx.json"',
    )
    parser.add_argument("--report", type=Path, default=None, help="JSON с временем этапов")
    parser.set_defaults(handler=run)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()


__all__ = ["NAMES", "STAGES", "commands", "main"]
