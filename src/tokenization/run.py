from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK, clean_directory
from src.preprocessing.settings import PreprocessingConfig, normalize_group, processed_dir

from .categorical import ValuesError, build_values
from .contract import CONFIG_FILE, ContractError, build_contract
from .corpus import CorpusError
from .layout import LayoutError, build_layout
from .numeric import BucketsError, build_buckets
from .scan import ScanError
from .schema import SchemaError
from .settings import ConfigError, TokenizerConfig, tokenized_dir, vocab_dir
from .text import TextError
from .transform import TransformError, group_corpus, open_artifacts, resolve_cutoffs, transform_group


# ============================================================
# ИДЕЯ
# ============================================================
#
# Каждый этап токенизатора это отдельная команда и отдельная
# остановка. Ни одна не запускает следующую сама: словарь это
# решение, а не побочный эффект.
#
#   contract   что разрешено читать и на чём разрешено учиться
#   values     смыслы ключей и коды категорий
#   buckets    границы чисел
#   freeze     BPE и общее пространство ID
#   encode     кодирование замороженными артефактами
#
# Выход первых четырёх: data/artifacts/<name>/tokenizer/.
# ============================================================


FAILURES = (
    BucketsError,
    ConfigError,
    ContractError,
    CorpusError,
    LayoutError,
    SchemaError,
    ScanError,
    TextError,
    ValuesError,
)


def dataset_name(args) -> str:

    if args.name:
        return args.name

    if getattr(args, "raw", None) is not None:
        return Path(args.raw).resolve().name

    root = getattr(args, "raw_root", None)

    if root is None:
        raise SystemExit("нужно имя набора: укажите --name")

    return Path(root).resolve().name


def resolve_raw(args, group: str) -> Path:
    """
    Каталог RAW той группы, на которой работаем.

    Справочники продуктов и мерчантов лежат там, и смысловой
    слой без них не расшифрует ни продукт, ни торговую точку.
    """

    if args.raw is not None:
        return Path(args.raw)

    root = Path(args.raw_root)

    candidates = [root / group] + ([root / "validation"] if group == "val" else [])

    found = next((path for path in candidates if path.exists()), None)

    if found is None:
        raise SystemExit(f"в {root} нет подкаталога группы {group}")

    return found


def run_contract(args) -> int:

    try:
        config = TokenizerConfig.load(Path(args.config) if args.config else None)
    except ConfigError as error:
        print(f"[contract] конфигурация: {error}")
        return EXIT_BLOCKED

    group = normalize_group(args.group) if args.group else config.fit_group

    name = dataset_name(args)

    processed = Path(args.processed) if args.processed else processed_dir(name)

    target = Path(args.out) if args.out else vocab_dir(name)

    raw_dir = resolve_raw(args, group)

    # Контракт это основание всего, что ниже: изменился он —
    # прежние словари недействительны, и оставлять их рядом
    # нельзя.
    clean_directory(target)

    try:
        result = build_contract(
            processed_dir=processed,
            raw_dir=raw_dir,
            target=target,
            config=config,
            group=group,
            allow_short_horizon=args.allow_short_horizon,
        )
    except FAILURES as error:
        print(f"[contract] группа {group}: {error}")
        clean_directory(target)
        return EXIT_BLOCKED

    report = result.report

    print(f"[contract] группа {group}: контракт входа → {target}")
    print(
        f"    корпус на {report['fit_end'][:10]}: клиентов {report['corpus']['clients']}, "
        f"событий {report['corpus']['events']}, значений {report['corpus']['values']}, "
        f"действующих анкет {report['corpus']['profiles_as_of']}"
    )
    print(
        f"    ключей объявлено {report['keys']['declared']}: код получают "
        f"{report['keys']['model_feature']}, связями остаются {report['keys']['link']}; "
        f"встретилось на train {report['keys']['observed_in_train']}"
    )
    print(
        f"    отпечаток смыслового содержимого {report['fit_content_sha256'][:16]}, "
        f"готовность {report['readiness']['status']}"
    )

    for item in report["contradictions"]:
        print(f"    противоречие: {item['key']} — {item['detail']}")

    for item in report["limitations"][:3]:
        print(f"    ограничение: {item}")

    return EXIT_OK


def _later_stage(args, what: str, builder) -> int:
    """
    Общий путь этапов, которые читают готовые артефакты и ничего
    не читают из данных заново.
    """

    try:
        config = TokenizerConfig.load(Path(args.config) if args.config else None)
    except ConfigError as error:
        print(f"[{what}] конфигурация: {error}")
        return EXIT_BLOCKED

    group = normalize_group(args.group) if args.group else config.fit_group

    name = dataset_name(args)

    processed = Path(args.processed) if args.processed else processed_dir(name)
    target = Path(args.out) if args.out else vocab_dir(name)

    try:
        result = builder(target, processed, config, group)
    except FAILURES as error:
        print(f"[{what}] группа {group}: {error}")
        return EXIT_BLOCKED

    return result


def run_values(args) -> int:

    result = _later_stage(args, "values", build_values)

    if isinstance(result, int):
        return result

    report = result.report
    counts = report["counts"]

    print(f"[values] группа {report['group']}: каталог значений собран")
    print(
        f"    ключей с кодом {counts['keys_model_feature']}: чисел {counts['numeric']}, "
        f"категорий {counts['categorical']}, текстов {counts['text']}; ссылок {counts['link']}"
    )
    print(
        f"    доменов {counts['domains']} (объединённых {counts['domains_shared']}), "
        f"различных значений {counts['values']}, редких {counts['rare']}"
    )
    print(f"    без наблюдений на train: {counts['unobserved_in_train']} ключей")

    return EXIT_OK


def run_buckets(args) -> int:

    result = _later_stage(args, "buckets", build_buckets)

    if isinstance(result, int):
        return result

    report = result.report
    counts = report["counts"]

    print(f"[buckets] группа {report['group']}: числовые диапазоны собраны")
    print(f"    ключей {counts['keys']}, диапазонов {counts['buckets']}; источники границ {counts['by_source']}")

    for item in report["warnings"][:5]:
        print(f"    предупреждение: {item}")

    return EXIT_OK


def run_freeze(args) -> int:

    result = _later_stage(args, "freeze", build_layout)

    if isinstance(result, int):
        return result

    manifest = result.report
    sizes = manifest["layout"]["sizes"]

    print(f"[freeze] группа {manifest['group']}: словарь заморожен, отпечаток {manifest['artifact_id']}")
    print(
        f"    специальных {sizes['special']}, ключей {sizes['keys']}, категорий {sizes['categorical']}, "
        f"диапазонов {sizes['buckets']}, кусков BPE {sizes['bpe']}; всего ID {sizes['total']}"
    )

    bpe = manifest["bpe"]

    if bpe.get("enabled"):
        print(
            f"    BPE на ключах {', '.join(bpe['keys'])}: {bpe['corpus']['texts']} текстов, "
            f"словарь {bpe['vocab_size']['actual']} из запрошенных {bpe['vocab_size']['requested']}"
        )
    else:
        print(f"    BPE выключен: {bpe.get('reason')}")

    return EXIT_OK


def frozen_config(target: Path, artifacts) -> TokenizerConfig:
    """
    Конфигурация, которой заморожен этот словарь.
    """

    config = TokenizerConfig.load(target / CONFIG_FILE)

    if config.sha256() != artifacts.manifest["config_sha256"]:
        raise ConfigError(
            f"{CONFIG_FILE} в каталоге артефактов не тот, которым заморожен словарь: "
            "соберите словарь заново"
        )

    return config


def group_cutoff(corpus, group: str):
    """
    Конечный cutoff группы.

    Берётся из фактического разделения, а не из значений по
    умолчанию: на нестандартных окнах умолчание указало бы не на
    тот момент, и разница заметна не сразу.
    """

    if corpus.final_cutoff is not None:
        return corpus.final_cutoff

    return PreprocessingConfig.load(None).windows[group].final_cutoff


def run_dirname(group: str, cutoffs: list, client: str | None) -> str:
    """
    Имя каталога результата, различающее разные прогоны.

    Полный прогон группы и разбор одного клиента не должны
    попадать в один каталог: диагностический запуск затёр бы
    полноценный результат.
    """

    name = f"{group}__{cutoffs[-1].date()}"

    if len(cutoffs) > 1:
        name = f"{group}__{cutoffs[0].date()}__{cutoffs[-1].date()}__{len(cutoffs)}"

    if client:
        name = f"{name}__client_{client}"

    return name


def prepare_directory(directory: Path, force: bool) -> None:
    """
    Готовит каталог результата, не стирая чужой молча.
    """

    existing = [path for path in directory.rglob("*") if path.is_file()] if directory.exists() else []

    if existing and not force:
        raise TransformError(
            f"в {directory} уже лежит результат из {len(existing)} файлов. "
            "Перезапись стирает его целиком: укажите --force, если это то, чего вы хотите, "
            "или задайте другой --tokenized"
        )

    clean_directory(directory)


def run_encode(args) -> int:

    name = dataset_name(args)

    processed = Path(args.processed) if args.processed else processed_dir(name)
    target = Path(args.out) if args.out else vocab_dir(name)

    try:
        artifacts = open_artifacts(target)

        # Конфигурация едет вместе со словарём. Читается она из
        # замороженного комплекта, а не из аргументов: иначе
        # кодирование шло бы по одним правилам, а словарь был бы
        # построен по другим.
        config = frozen_config(target, artifacts)

        if args.config is not None:

            asked = TokenizerConfig.load(Path(args.config))

            if asked.sha256() != config.sha256():
                raise ConfigError(
                    "переданная конфигурация отличается от той, которой заморожен словарь. "
                    "Кодирование правил не меняет: чтобы применить другие, соберите словарь заново"
                )

        group = normalize_group(args.group) if args.group else config.fit_group

        raw_dir = resolve_raw(args, group)

        corpus = group_corpus(processed, raw_dir, group)

        cutoffs = resolve_cutoffs(args.cutoff, group_cutoff(corpus, group))

        directory = (
            Path(args.tokenized)
            if args.tokenized
            else tokenized_dir(name) / artifacts.manifest["artifact_id"] / run_dirname(group, cutoffs, args.client)
        )

        prepare_directory(directory, args.force)

        result = transform_group(
            artifacts=artifacts,
            corpus=corpus,
            cutoffs=cutoffs,
            directory=directory,
            config=config,
            clients=[args.client] if args.client else None,
        )

    except (*FAILURES, TransformError) as error:
        group = normalize_group(args.group) if args.group else "—"
        print(f"[encode] группа {group}: {error}")
        return EXIT_BLOCKED

    report = result.report
    counts = report["counts"]

    print(f"[encode] группа {group}: срезы {', '.join(item[:10] for item in report['cutoffs'])} → {result.directory}")
    print(
        f"    клиентов {counts['clients']} на {counts['client_slices']} срезах, "
        f"событий {counts['events']}, значений {counts['values']}, токенов {counts['tokens']}"
    )
    print(
        f"    [MISSING] {report['specials']['missing']}, [UNK] {report['specials']['unknown']}, "
        f"[INVALID] {report['specials']['invalid']}, [EMPTY] {report['specials']['empty']}; "
        f"токенов в событии до {report['lengths']['event_tokens']['max']}"
    )
    print(f"    словарь {report['artifact_id']} не изменился: проверено до и после")

    for item in report["skipped_cutoffs"]:
        print(f"    пропущен срез {item['cutoff'][:10]}: {item['reason']}")

    return EXIT_OK


def _add_common(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--name", default=None, help="имя набора: data/processed/<name> и data/artifacts/<name>")
    parser.add_argument("--group", default=None, help="группа fit; по умолчанию train")
    parser.add_argument("--processed", type=Path, default=None, help="каталог набора вместо data/processed/<name>")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="каталог артефактов вместо data/artifacts/<name>/tokenizer",
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON с переопределениями конфига токенизатора")


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.tokenization.run")

    subparsers = parser.add_subparsers(dest="stage", required=True)

    contract = subparsers.add_parser("contract", help="этап 1: контракт входа и разрешённый train-корпус")

    source = contract.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="каталог RAW группы, на которой идёт fit")
    source.add_argument("--raw-root", type=Path, help="корень RAW с подкаталогами train/val/test")

    _add_common(contract)

    contract.add_argument(
        "--allow-short-horizon",
        action="store_true",
        help="согласиться на короткий горизонт наблюдения: результат помечается как диагностика",
    )
    contract.set_defaults(handler=run_contract)

    values = subparsers.add_parser("values", help="этап 2: смыслы ключей и коды категорий")
    _add_common(values)
    values.set_defaults(handler=run_values, raw=None, raw_root=None)

    buckets = subparsers.add_parser("buckets", help="этап 3: числовые диапазоны")
    _add_common(buckets)
    buckets.set_defaults(handler=run_buckets, raw=None, raw_root=None)

    freeze = subparsers.add_parser("freeze", help="этап 4: BPE и общее пространство ID")
    _add_common(freeze)
    freeze.set_defaults(handler=run_freeze, raw=None, raw_root=None)

    encode = subparsers.add_parser("encode", help="этап 5: кодирование замороженными артефактами")

    source = encode.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, help="каталог RAW группы, которую кодируем")
    source.add_argument("--raw-root", type=Path, help="корень RAW с подкаталогами train/val/test")

    _add_common(encode)

    encode.add_argument(
        "--cutoff", action="append", default=None,
        help="момент среза ISO; можно повторять. По умолчанию конечный cutoff группы",
    )
    encode.add_argument("--client", default=None, help="закодировать одного клиента")
    encode.add_argument(
        "--tokenized", type=Path, default=None,
        help="каталог результата вместо data/tokenized/<name>/<artifact_id>/<группа>__<дата>",
    )
    encode.add_argument(
        "--force", action="store_true",
        help="перезаписать непустой каталог результата",
    )
    encode.set_defaults(handler=run_encode)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
