from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from src.preprocessing.artifacts import write_json, write_text
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import processed_dir
from src.tokenization.settings import vocab_dir

from .build import BuildError, build_dataset
from .check import CheckError, check_dataset, write_check
from .context import ContextError
from .inputs import DatasetInputs, InputsError
from .measure import measure_group, write_measurement
from .reader import Dataset, DatasetError
from .report import render_contract_md
from .sample import SampleError
from .settings import ConfigError, DatasetConfig, datasets_dir
from .storage import (
    MASKER_COLUMNS,
    MODEL_COLUMNS,
    SERVICE_COLUMNS,
    StorageError,
)
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Четыре команды, каждая со своим вопросом:
#
#   contract   что разрешено читать и каким будет пример
#   measure    сколько истории у клиентов на самом деле
#   build      собрать набор
#   check      проверить собранный набор
#
# Ни одна не запускает следующую сама: бюджет контекста это
# решение человека, принятое после измерения.
# ============================================================


CONTRACT_DIRNAME = "contract"
CONTRACT_JSON_FILE = "dataset_contract.json"
CONTRACT_MD_FILE = "dataset_contract.md"


FAILURES = (
    BuildError,
    CheckError,
    ConfigError,
    ContextError,
    DatasetError,
    InputsError,
    SampleError,
    StorageError,
)


def dataset_name(args) -> str:

    if args.name:
        return args.name

    root = getattr(args, "raw_root", None)

    if root is None:
        raise SystemExit("нужно имя набора: укажите --name")

    return Path(root).resolve().name


def load_config(args) -> DatasetConfig:
    """
    Конфигурация из файла и правки командной строки.

    Флаги существуют ради двух наборов на одних данных: политика
    контекста входит в имя набора, и менять её файлом ради одного
    прогона неудобно. Действующая конфигурация целиком уезжает в
    манифест, поэтому короткая команда не делает решение
    невидимым.
    """

    config = DatasetConfig.load(Path(args.config) if args.config else None)

    context = config.context

    if getattr(args, "policy", None):
        context = replace(context, policy=args.policy)

    if getattr(args, "max_events", None) is not None:
        context = replace(context, max_events=args.max_events)

    if getattr(args, "max_tokens", None) is not None:
        context = replace(context, max_tokens=args.max_tokens)

    if getattr(args, "group", None):
        config = replace(config, groups=(args.group,))

    config = replace(config, context=context)

    config.validate()

    return config


def open_inputs(args, config: DatasetConfig) -> tuple[DatasetInputs, Path]:

    name = dataset_name(args)

    processed = Path(args.processed) if args.processed else processed_dir(name)
    target = Path(args.vocab) if args.vocab else vocab_dir(name)
    root = Path(args.out) if args.out else datasets_dir(name)

    inputs = DatasetInputs.open(processed, Path(args.raw_root), target, config)

    return inputs, root


def run_contract(args) -> int:

    try:
        config = load_config(args)
        inputs, root = open_inputs(args, config)
    except FAILURES as error:
        print(f"[contract] {error}")
        return EXIT_BLOCKED

    report = {
        "stage": "contract",
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_id": inputs.dataset_id(),
        "readiness": inputs.readiness,
        "inputs": inputs.as_dict(),
        "config": config.as_dict(),
        "identity": inputs.identity(),
        "sources": list(inputs.sources()),
        "channels": {
            "model": list(MODEL_COLUMNS),
            "masker": list(MASKER_COLUMNS),
            "service": list(SERVICE_COLUMNS),
        },
        "contract": {
            "sample": "один пример это один клиент на один срез",
            "masks": "True это настоящее значение, а не выравнивание",
            "target_candidate_mask": "область допустимых целей, а не выбранные цели",
            "weight": "единица, делённая на число срезов клиента",
            "dataset_id": "потребитель называет набор явно",
        },
    }

    directory = root / CONTRACT_DIRNAME

    write_json(directory / CONTRACT_JSON_FILE, report)
    write_text(directory / CONTRACT_MD_FILE, render_contract_md(report))

    print(f"[contract] контракт входа → {directory}")
    print(
        f"    словарь {inputs.artifacts.manifest['artifact_id']}, набор получит имя "
        f"{report['dataset_id']}, готовность {inputs.readiness['status']}"
    )

    for group, item in sorted(inputs.as_dict()["groups"].items()):
        print(
            f"    {group}: клиентов {item['clients']}, срезы "
            f"{', '.join(value[:10] for value in item['cutoffs'])}, вес {item['weight']:.4f}"
        )

    for item in inputs.readiness["reasons"][:3]:
        print(f"    диагностика: {item}")

    return EXIT_OK


def run_measure(args) -> int:

    try:
        config = load_config(args)
        inputs, root = open_inputs(args, config)

        group = args.group or inputs.artifacts.manifest["group"]

        if group not in inputs.groups:
            raise InputsError(f"группы {group!r} нет среди собираемых: {list(inputs.groups)}")

        report = measure_group(inputs, group)

    except FAILURES as error:
        print(f"[measure] {error}")
        return EXIT_BLOCKED

    outputs = write_measurement(root, report)

    lengths = report["lengths"]

    print(f"[measure] группа {group} на срезе {report['cutoff'][:10]} → {outputs[0].parent}")
    print(
        f"    клиентов {report['clients']}, событий {report['events']}, "
        f"токенов {report['tokens']}"
    )

    for name in ("events_per_client", "tokens_per_client", "tokens_per_event"):
        item = lengths[name]
        print(
            f"    {name}: p50 {item['p50']}, p90 {item['p90']}, p95 {item['p95']}, "
            f"p99 {item['p99']}, максимум {item['max']}"
        )

    return EXIT_OK


def run_build(args) -> int:

    try:
        config = load_config(args)
        inputs, root = open_inputs(args, config)

        result = build_dataset(inputs, config, root, force=args.force)

    except FAILURES as error:
        print(f"[build] {error}")
        return EXIT_BLOCKED

    counts = result.report["counts"]

    print(f"[build] набор {result.dataset_id} → {result.directory}")
    print(
        f"    примеров {counts['samples']}, событий {counts['events']}, "
        f"токенов {counts['tokens']}, целей {counts['eligible_events']}"
    )
    print(
        f"    пустых историй {counts['empty_history']}, без целей "
        f"{counts['samples_without_targets']}, усечённых {counts['truncated']}"
    )

    for group, item in sorted(counts["by_group"].items()):
        print(
            f"    {group}: примеров {item['samples']}, исключено событий "
            f"{item['excluded_events']}, потеряно целей {item['excluded_eligible']}"
        )

    for group, item in sorted(result.report["eligible_agreement"].items()):
        if item["comparable"] and not item["agree"]:
            print(
                f"    расхождение целей в {group}: у набора {item['dataset']}, "
                f"у разделения {item['corpus_manifest']}"
            )

    print(f"    готовность {result.report['readiness']['status']}")

    return EXIT_OK


def run_check(args) -> int:

    try:
        config = load_config(args)
        inputs, root = open_inputs(args, config)

        directory = Path(args.dataset) if args.dataset else root / args.dataset_id

        dataset = Dataset.open(directory, artifacts=inputs.artifacts)

        report = check_dataset(dataset, inputs, recompute=args.recompute)

    except FAILURES as error:
        print(f"[check] {error}")
        return EXIT_BLOCKED

    outputs = write_check(dataset.directory, report)

    print(f"[check] набор {report['dataset_id']} → {outputs[1]}")

    for item in report["checks"]:
        mark = "ок" if item["ok"] else "НЕТ"
        print(f"    [{mark}] {item['name']}: {item['detail']}")

    return EXIT_OK if report["ok"] else EXIT_BLOCKED


def _add_common(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--raw-root", type=Path, required=True,
                        help="корень RAW с подкаталогами train/val/test")
    parser.add_argument("--name", default=None,
                        help="имя набора: data/processed/<name> и data/datasets/<name>")
    parser.add_argument("--processed", type=Path, default=None,
                        help="каталог обработанных данных вместо data/processed/<name>")
    parser.add_argument("--vocab", type=Path, default=None,
                        help="каталог словаря вместо data/artifacts/<name>/tokenizer")
    parser.add_argument("--out", type=Path, default=None,
                        help="каталог наборов вместо data/datasets/<name>")
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON с переопределениями конфигурации датасета")


def _add_policy(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--policy", default=None, choices=("all", "recent_plus_milestones"),
                        help="политика отбора истории")
    parser.add_argument("--max-events", type=int, default=None, help="предел событий в примере")
    parser.add_argument("--max-tokens", type=int, default=None, help="предел токенов в примере")


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.dataset.run")

    subparsers = parser.add_subparsers(dest="stage", required=True)

    contract = subparsers.add_parser("contract", help="контракт входа и формат примера")
    _add_common(contract)
    _add_policy(contract)
    contract.add_argument("--group", default=None, help="собрать только одну группу")
    contract.set_defaults(handler=run_contract)

    measure = subparsers.add_parser("measure", help="длины историй до выбора бюджетов")
    _add_common(measure)
    _add_policy(measure)
    measure.add_argument("--group", default=None, help="группа измерения; по умолчанию fit-группа")
    measure.set_defaults(handler=run_measure)

    build = subparsers.add_parser("build", help="собрать набор примеров")
    _add_common(build)
    _add_policy(build)
    build.add_argument("--group", default=None, help="собрать только одну группу")
    build.add_argument("--force", action="store_true", help="пересобрать уже готовый набор")
    build.set_defaults(handler=run_build)

    check = subparsers.add_parser("check", help="проверить собранный набор")
    _add_common(check)
    _add_policy(check)
    check.add_argument("--group", default=None, help="ограничить проверку одной группой входов")
    check.add_argument("--dataset-id", default=None, help="имя набора внутри data/datasets/<name>")
    check.add_argument("--dataset", type=Path, default=None, help="каталог набора целиком")
    check.add_argument("--recompute", type=int, default=0,
                       help="сколько примеров пересчитать из семантики")
    check.set_defaults(handler=run_check)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.stage == "check" and not (args.dataset or args.dataset_id):
        raise SystemExit("нужно имя набора: укажите --dataset-id или --dataset")

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
