from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.artifacts import write_json
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .categorical import ValuesError, build_value_vocab, load_value_vocab
from .fit import FitError, read_train, unit_warnings
from .keyvocab import KeyVocabError, build_key_vocab, load_key_vocab
from .layout import FrozenArtifacts, LayoutError, build_tokenizer
from .numeric import BucketsError, build_buckets, load_buckets
from .scan import ScanError
from .schema import SchemaError, SemanticSchema
from .settings import (
    BPE_FILE,
    BUCKETS_FILE,
    KEY_VOCAB_FILE,
    SPECIAL_TOKENS_FILE,
    TOKENIZER_DIR,
    TOKENIZER_FILE,
    VALUE_VOCAB_FILE,
    ConfigError,
    TokenizerConfig,
    tokenizer_path,
)
from .specials import SpecialsError, build_special_tokens, load_special_tokens
from .text import TextError, build_bpe, load_bpe_file
from .transform import TransformError, encode_group


# ============================================================
# ИДЕЯ
# ============================================================
#
# Словарь строится по шагам, и каждый шаг это отдельная команда
# с одним видимым результатом. Автоматической цепочки нет
# намеренно: словарь это решение, а не побочный эффект запуска.
#
#   special-tokens data/tokenizer/special_tokens.json служебные токены и их ID
#   key-vocab     data/tokenizer/key_vocab.json     поля модели и их ID
#   value-vocab   data/tokenizer/value_vocab.json   категории train и их ID
#   buckets       data/tokenizer/buckets.json       границы чисел и их токены
#   bpe           data/tokenizer/bpe.json           разбиение текста
#   final-vocab   data/tokenizer/tokenizer.json     единое пространство ID
#   encode <g>    data/tokenized/<g>/               два файла закодированной группы
#
# Для прода есть fit: он вызывает те же шесть этапов подряд и
# останавливается на первой же ошибке, называя этап.
#
# Учатся только value-vocab, buckets и bpe, и только на
# data/preprocessed/train.
# Кодирование применяет готовый словарь и не меняет его.
# ============================================================


FAILURES = (
    BucketsError,
    ConfigError,
    FitError,
    KeyVocabError,
    LayoutError,
    SchemaError,
    ScanError,
    SpecialsError,
    TextError,
    ValuesError,
)


def _config(args) -> TokenizerConfig:
    return TokenizerConfig.load(Path(args.config) if args.config else None)


def _write(name: str, payload: dict) -> Path:

    path = tokenizer_path(name)

    write_json(path, payload)

    return path


# ------------------------------------------------------------
# ЭТАПЫ
# ------------------------------------------------------------


def run_special_tokens(args) -> int:

    tokens = build_special_tokens()

    path = _write(SPECIAL_TOKENS_FILE, tokens)

    print(f"[special-tokens] служебные токены → {path}")
    print(f"    токенов {tokens['size']}: ID с 0 по {tokens['next_id'] - 1}")

    for row in tokens["tokens"]:
        print(f"    {row['id']:>2} {row['token']:<10} {row['role']}")

    return EXIT_OK


def run_key_vocab(args) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        specials = load_special_tokens()
        vocab = build_key_vocab(specials, config, schema)
    except FAILURES as error:
        print(f"[key-vocab] {error}")
        return EXIT_BLOCKED

    path = _write(KEY_VOCAB_FILE, vocab)

    print(f"[key-vocab] поля модели → {path}")
    print(
        f"    ключей {vocab['size']}: ID с {vocab['first_key_id']} по {vocab['next_id'] - 1}; "
        f"ссылок без кода {len(vocab['link_keys'])}"
    )

    kinds: dict[str, int] = {}

    for row in vocab["keys"]:
        kinds[row["value_kind"]] = kinds.get(row["value_kind"], 0) + 1

    print("    по видам значения: " + ", ".join(f"{name} {count}" for name, count in sorted(kinds.items())))

    return EXIT_OK


def run_value_vocab(args) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        key_vocab = load_key_vocab()
        train = read_train(config, schema)
        vocab = build_value_vocab(train, key_vocab, config, schema)
    except FAILURES as error:
        print(f"[value-vocab] {error}")
        return EXIT_BLOCKED

    path = _write(VALUE_VOCAB_FILE, vocab)

    counts = vocab["counts"]

    print(f"[value-vocab] категории train → {path}")
    print(
        f"    значений {counts['values']} в {counts['domains']} доменах "
        f"(объединённых {counts['domains_shared']}), редких {counts['rare']}"
    )
    print(
        f"    ID с {vocab['first_value_id']} по {vocab['next_id'] - 1}; "
        f"ключей без наблюдений на train {counts['unobserved_in_train']}"
    )

    return EXIT_OK


def run_buckets(args) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        value_vocab = load_value_vocab()
        train = read_train(config, schema)
        registry = build_buckets(train, value_vocab, config, schema)
    except FAILURES as error:
        print(f"[buckets] {error}")
        return EXIT_BLOCKED

    path = _write(BUCKETS_FILE, registry)

    counts = registry["counts"]

    print(f"[buckets] числовые диапазоны → {path}")
    print(
        f"    ключей {counts['keys']}, диапазонов {counts['buckets']}; "
        f"источники границ {counts['by_source']}"
    )
    print(f"    ID с {registry['first_bucket_id']} по {registry['next_id'] - 1}")

    for item in registry["warnings"][:5]:
        print(f"    предупреждение: {item}")

    for item in unit_warnings(schema, config)[:5]:
        print(f"    единица: {item}")

    return EXIT_OK


def run_bpe(args) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        key_vocab = load_key_vocab()
        train = read_train(config, schema)
        model = build_bpe(train, key_vocab, config)
    except FAILURES as error:
        print(f"[bpe] {error}")
        return EXIT_BLOCKED

    path = _write(BPE_FILE, model)

    print(f"[bpe] разбиение текста → {path}")

    if model.get("enabled"):
        print(
            f"    ключи {', '.join(model['keys'])}: {model['corpus']['texts']} текстов, "
            f"словарь {model['vocab_size']['actual']} из запрошенных {model['vocab_size']['requested']}"
        )
        print(f"    кусков на текст: медиана {model['pieces_per_text']['median']}, "
              f"максимум {model['pieces_per_text']['max']}")
    else:
        print(f"    BPE выключен: {model.get('reason')}")

    return EXIT_OK


def run_final_vocab(args) -> int:

    try:
        key_vocab = load_key_vocab()
        value_vocab = load_value_vocab()
        buckets = load_buckets()
        bpe = load_bpe_file()
        specials = load_special_tokens()
        bundle = build_tokenizer(specials, key_vocab, value_vocab, buckets, bpe)
        FrozenArtifacts.from_bundle(bundle)
    except FAILURES as error:
        print(f"[final-vocab] {error}")
        return EXIT_BLOCKED

    path = _write(TOKENIZER_FILE, bundle)

    sizes = bundle["layout"]["sizes"]
    ranges = bundle["layout"]["ranges"]

    print(f"[final-vocab] словарь собран → {path}")
    print(f"    токенов в таблице {len(bundle['tokens'])}")
    print(
        f"    специальных {sizes['special']}, ключей {sizes['keys']}, категорий {sizes['categorical']}, "
        f"диапазонов {sizes['buckets']}, кусков BPE {sizes['bpe']}; всего ID {sizes['total']}"
    )
    print(
        "    границы: специальные " + str(ranges["special"]) + ", ключи " + str(ranges["keys"])
        + ", значения " + str(ranges["values"]) + ", BPE " + str(ranges["bpe"])
    )

    return EXIT_OK


def run_encode(args) -> int:

    group = normalize_group(args.group)

    try:
        config = _config(args)
        artifacts = FrozenArtifacts.load()

        if config.sha256() != artifacts.bundle["config_sha256"]:
            raise ConfigError(
                "конфигурация не та, которой собран словарь: кодирование пошло бы по одним "
                "правилам, а словарь построен по другим. Соберите словарь заново"
            )

        report = encode_group(artifacts, group, config)

    except (*FAILURES, TransformError) as error:
        print(f"[encode] группа {group}: {error}")
        return EXIT_BLOCKED

    counts = report["counts"]

    print(f"[encode] группа {group} до {report['cutoff'][:10]} → data/tokenized/{group}")
    print(
        f"    клиентов {counts['clients']}, событий {counts['events']}, значений {counts['values']}, "
        f"токенов {counts['tokens']}"
    )
    print(
        f"    [MISSING] {report['specials']['missing']}, [UNK] {report['specials']['unknown']}, "
        f"[INVALID] {report['specials']['invalid']}, [EMPTY] {report['specials']['empty']}; "
        f"токенов в событии до {counts['max_event_tokens']}"
    )
    print(
        f"    строк: events {report['rows']['events']}, profile {report['rows']['profile']}; "
        f"молчащих клиентов {counts['silent_clients']}"
    )

    return EXIT_OK


# ------------------------------------------------------------
# ВЕСЬ FIT ОДНОЙ КОМАНДОЙ
# ------------------------------------------------------------


# Этапы обучения в том же порядке, в каком их запускают
# руками. Список один: второго описания цепочки нет.
FIT_STAGES: tuple[tuple[str, object], ...] = (
    ("special-tokens", run_special_tokens),
    ("key-vocab", run_key_vocab),
    ("value-vocab", run_value_vocab),
    ("buckets", run_buckets),
    ("bpe", run_bpe),
    ("final-vocab", run_final_vocab),
)


def run_fit(args) -> int:
    """
    Все этапы обучения подряд, одной командой.

    Собственной логики здесь нет: вызываются те же функции,
    что и у отдельных команд, поэтому поэтапный запуск и
    fit дают один и тот же результат.

    Готовый словарь убирается ДО первого этапа: если
    цепочка оборвётся, рядом не останется tokenizer.json от
    прежней сборки, который выглядел бы собранным из уже
    пересчитанных частей.
    """

    stale = tokenizer_path(TOKENIZER_FILE)

    if stale.exists():
        stale.unlink()

    for number, (name, handler) in enumerate(FIT_STAGES, start=1):

        print(f"[fit] этап {number}/{len(FIT_STAGES)}: {name}")

        code = handler(args)

        if code != EXIT_OK:
            print(f"[fit] остановлено на этапе {name}: словарь не собран")
            return code

    print(f"[fit] словарь готов: {TOKENIZER_DIR}")

    return EXIT_OK


# ------------------------------------------------------------
# КОМАНДЫ
# ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.tokenization.run")

    subparsers = parser.add_subparsers(dest="stage", required=True)

    def add(name: str, help_text: str, handler) -> argparse.ArgumentParser:

        item = subparsers.add_parser(name, help=help_text)
        item.add_argument(
            "--config", type=Path, default=None, help="JSON с переопределениями конфига токенизатора"
        )
        item.set_defaults(handler=handler)

        return item

    add("fit", "все этапы обучения словаря подряд", run_fit)

    add("special-tokens", "служебные токены и их ID", run_special_tokens)
    add("key-vocab", "поля модели и их ID", run_key_vocab)
    add("value-vocab", "категориальные значения train и их ID", run_value_vocab)
    add("buckets", "границы чисел и их токены", run_buckets)
    add("bpe", "разбиение текста на train", run_bpe)
    add("final-vocab", "единое пространство ID: tokenizer.json", run_final_vocab)

    encode = add("encode", "кодирование группы готовым словарём", run_encode)
    encode.add_argument("group", choices=GROUPS, help="группа: train, val или test")

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
