from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.masking.settings import MaskingConfig
from src.masking.weights import WeightsError, build
from src.preprocessing.artifacts import write_json
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK
from src.preprocessing.settings import GROUPS, normalize_group

from .categorical import ValuesError, build_value_vocab, load_value_vocab
from .finalvocab import FrozenArtifacts, VocabError, build_final_vocab, vocabulary_digest
from .fit import FitError, TrainCorpus, read_train, unit_warnings
from .keyvocab import KeyVocabError, build_key_vocab, load_key_vocab
from .numeric import BucketsError, build_buckets, load_buckets
from .scan import ScanError
from .schema import SchemaError, SemanticSchema
from .settings import (
    BPE_FILE,
    BUCKETS_FILE,
    FINAL_VOCAB_FILE,
    KEY_VOCAB_FILE,
    SPECIAL_TOKENS_FILE,
    VOCAB_DIR,
    VALUE_VOCAB_FILE,
    VALUE_WEIGHTS_FILE,
    ConfigError,
    TokenizerConfig,
    vocab_path,
)
from .specials import SpecialsError, build_special_tokens, load_special_tokens
from .text import TextError, build_bpe, load_bpe
from .transform import TransformError, encode_group
from .valuestats import value_counts


# ============================================================
# ИДЕЯ
# ============================================================
#
# Словарь строится по шагам, и каждый шаг это отдельная команда
# с одним видимым файлом. Автоматической цепочки нет намеренно:
# словарь это решение, а не побочный эффект запуска.
#
#   special-tokens  data/03_vocab/special_tokens.json  служебные токены
#   key-vocab       data/03_vocab/key_vocab.json       поля модели
#   value-vocab     data/03_vocab/value_vocab.json     категории train
#   buckets         data/03_vocab/buckets.json         диапазоны чисел
#   bpe             data/03_vocab/bpe.json             разбиение текста
#   final-vocab     data/03_vocab/final_vocab.json     имя токена -> ID
#   value-weights   data/03_vocab/value_weights.json   веса value-маскирования
#   encode <group>  data/04_tokenized/<group>/         два файла группы
#
# Для прода есть fit: он вызывает те же семь функций подряд и
# останавливается на первой же ошибке, называя этап.
#
# Учатся только value-vocab, buckets, bpe и value-weights, и только
# на data/02_preprocessed/train вместе с data/01_raw/train/profile.parquet.
# Кодирование применяет готовый словарь и не меняет его.
# ============================================================


FAILURES = (
    BucketsError,
    ConfigError,
    FitError,
    KeyVocabError,
    SchemaError,
    ScanError,
    SpecialsError,
    TextError,
    ValuesError,
    VocabError,
)


def _config(args) -> TokenizerConfig:
    return TokenizerConfig.load(Path(args.config) if args.config else None)


def _write(name: str, payload: dict) -> Path:

    path = vocab_path(name)

    write_json(path, payload)

    return path


def _span(mapping: dict) -> str:
    """
    Отрезок номеров словаря для строки терминала.
    """

    ids = [number for item in mapping.values() for number in _numbers(item)]

    return f"ID с {min(ids)} по {max(ids)}" if ids else "номеров нет"


def _numbers(value) -> list[int]:

    if isinstance(value, bool):
        return []

    if isinstance(value, int):
        return [value]

    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]

    return []


# ------------------------------------------------------------
# ЭТАПЫ
# ------------------------------------------------------------


def run_special_tokens(args) -> int:

    tokens = build_special_tokens()

    path = _write(SPECIAL_TOKENS_FILE, tokens)

    print(f"[special-tokens] служебные токены → {path}")

    for name, token_id in tokens.items():
        print(f"    {token_id:>2} {name}")

    return EXIT_OK


def run_key_vocab(args) -> int:

    try:
        schema = SemanticSchema.open()
        specials = load_special_tokens()
        vocab = build_key_vocab(specials, schema)
    except FAILURES as error:
        print(f"[key-vocab] {error}")
        return EXIT_BLOCKED

    path = _write(KEY_VOCAB_FILE, vocab)

    print(f"[key-vocab] поля модели → {path}")
    print(f"    ключей {len(vocab)}: {_span(vocab)}")

    return EXIT_OK


def run_value_vocab(args, train: TrainCorpus | None = None) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        key_vocab = load_key_vocab()
        train = train if train is not None else read_train(config, schema)
        vocab = build_value_vocab(train, key_vocab, config, schema)
    except FAILURES as error:
        print(f"[value-vocab] {error}")
        return EXIT_BLOCKED

    path = _write(VALUE_VOCAB_FILE, vocab)

    values = sum(len(item) for item in vocab.values())
    silent = sorted(key for key, item in vocab.items() if not item)

    print(f"[value-vocab] категории train → {path}")
    print(f"    ключей {len(vocab)}, значений {values}: {_span(vocab)}")
    print(f"    без наблюдений на train: {len(silent)} ключей")

    return EXIT_OK


def run_buckets(args, train: TrainCorpus | None = None) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        value_vocab = load_value_vocab()
        train = train if train is not None else read_train(config, schema)
        buckets, warnings = build_buckets(train, value_vocab, config, schema)
    except FAILURES as error:
        print(f"[buckets] {error}")
        return EXIT_BLOCKED

    path = _write(BUCKETS_FILE, buckets)

    total = sum(len(item) for item in buckets.values())

    print(f"[buckets] числовые диапазоны → {path}")
    print(f"    ключей {len(buckets)}, диапазонов {total}: {_span(buckets)}")

    for item in warnings[:5]:
        print(f"    предупреждение: {item}")

    for item in unit_warnings(schema, config)[:5]:
        print(f"    единица: {item}")

    return EXIT_OK


def run_bpe(args, train: TrainCorpus | None = None) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        key_vocab = load_key_vocab()
        train = train if train is not None else read_train(config, schema)
        model, warnings = build_bpe(train, key_vocab, config, schema)
    except FAILURES as error:
        print(f"[bpe] {error}")
        return EXIT_BLOCKED

    path = vocab_path(BPE_FILE)

    model.save(path)

    print(f"[bpe] разбиение текста → {path}")
    print(f"    кусков {model.size}, из них слияний {model.merges}")

    for item in warnings:
        print(f"    {item}")

    return EXIT_OK


def run_final_vocab(args) -> int:

    try:
        specials = load_special_tokens()
        key_vocab = load_key_vocab()
        value_vocab = load_value_vocab()
        buckets = load_buckets()
        bpe = load_bpe()

        vocab = build_final_vocab(specials, key_vocab, value_vocab, buckets, bpe)

    except FAILURES as error:
        print(f"[final-vocab] {error}")
        return EXIT_BLOCKED

    path = _write(FINAL_VOCAB_FILE, vocab)

    try:
        FrozenArtifacts.load()
    except FAILURES as error:
        print(f"[final-vocab] словарь собран, но не читается обратно: {error}")
        return EXIT_BLOCKED

    values = sum(len(item) for item in value_vocab.values())
    ranges = sum(len(item) for item in buckets.values())

    print(f"[final-vocab] словарь собран → {path}")
    print(
        f"    специальных {len(specials)}, ключей {len(key_vocab)}, категорий {values}, "
        f"диапазонов {ranges}, кусков BPE {bpe.size}; всего ID {len(vocab)}"
    )

    return EXIT_OK


def run_value_weights(args, train: TrainCorpus | None = None) -> int:

    try:
        config = _config(args)
        schema = SemanticSchema.open()
        artifacts = FrozenArtifacts.load()
        train = train if train is not None else read_train(config, schema)
        found = value_counts(train, artifacts, schema)
        payload = build(
            found.counts, found.key_ids, found.labels, vocabulary_digest(), frozenset(found.estimated)
        )
    except (*FAILURES, WeightsError) as error:
        print(f"[value-weights] {error}")
        return EXIT_BLOCKED

    path = _write(VALUE_WEIGHTS_FILE, payload)

    keys = payload["keys"]
    values = sum(item["distinct"] for item in keys.values())

    print(f"[value-weights] веса value-маскирования по train → {path}")
    print(
        f"    ключей событий {len(keys)}, значений {values}, вхождений {payload['occurrences']}; "
        f"масштаб {payload['scale']:.3f}"
    )

    # Вероятность самого частого значения при конфиге маски по
    # умолчанию: так видно, какие поля почти не станут целями.
    masking = MaskingConfig()
    scale = payload["scale"]

    def top_probability(item: dict) -> float:
        raw = masking.value_probability * scale * item["key_weight"] * item["values"][0]["value_weight"]
        return min(masking.max_value_probability, max(masking.min_value_probability, raw))

    quiet = sorted(keys.items(), key=lambda pair: (pair[1]["key_weight"], pair[0]))[:5]

    print(
        "    наименее информативные: "
        + ", ".join(
            f"{key} (энтропия {item['normalized_entropy']:.2f}, P частого {top_probability(item):.3f})"
            for key, item in quiet
        )
    )

    if found.estimated:
        print(f"    по выборке fit: {', '.join(sorted(found.estimated))}")

    return EXIT_OK


def run_encode(args) -> int:

    group = normalize_group(args.group)

    try:
        config = _config(args)
        artifacts = FrozenArtifacts.load()
        report = encode_group(artifacts, group, config)

    except (*FAILURES, TransformError) as error:
        print(f"[encode] группа {group}: {error}")
        return EXIT_BLOCKED

    counts = report["counts"]

    print(f"[encode] группа {group} до {report['cutoff'][:10]} → data/04_tokenized/{group}")
    print(
        f"    клиентов {counts['clients']}, событий {counts['events']}, значений {counts['values']}, "
        f"токенов {counts['tokens']}"
    )
    print(
        f"    [UNK] {report['unknown_values']}; токенов в событии до {counts['max_event_tokens']}"
    )
    print(
        f"    строк: events {report['rows']['events']}, profile {report['rows']['profile']}; "
        f"молчащих клиентов {counts['silent_clients']}"
    )

    return EXIT_OK


# ------------------------------------------------------------
# ВЕСЬ FIT ОДНОЙ КОМАНДОЙ
# ------------------------------------------------------------


# Этапы обучения в том же порядке, в каком их запускают руками.
# Список один: второго описания цепочки нет.
FIT_STAGES: tuple[tuple[str, object], ...] = (
    ("special-tokens", run_special_tokens),
    ("key-vocab", run_key_vocab),
    ("value-vocab", run_value_vocab),
    ("buckets", run_buckets),
    ("bpe", run_bpe),
    ("final-vocab", run_final_vocab),
    ("value-weights", run_value_weights),
)


# Этапы, которые учатся на корпусе train: fit читает его один раз.
CORPUS_STAGES = ("value-vocab", "buckets", "bpe", "value-weights")


def run_fit(args) -> int:
    """
    Все этапы обучения подряд, одной командой.

    Собственной логики здесь нет: вызываются те же функции, что
    и у отдельных команд, поэтому поэтапный запуск и fit дают
    один и тот же результат. Одно отличие — train читается один
    раз: словарь значений, диапазоны, BPE и веса значений учатся на
    одном и том же корпусе (fit.read_train), а отдельные проходы по
    группе отнимали бы большую часть времени fit.

    Готовый словарь и веса убираются ДО первого этапа: если цепочка
    оборвётся, рядом не останется final_vocab.json или
    value_weights.json от прежней сборки, которые выглядели бы
    собранными из уже пересчитанных частей.
    """

    for name in (FINAL_VOCAB_FILE, VALUE_WEIGHTS_FILE):

        stale = vocab_path(name)

        if stale.exists():
            stale.unlink()

    train: TrainCorpus | None = None

    for number, (name, handler) in enumerate(FIT_STAGES, start=1):

        print(f"[fit] этап {number}/{len(FIT_STAGES)}: {name}")

        if name in CORPUS_STAGES:

            if train is None:
                try:
                    train = read_train(_config(args), SemanticSchema.open())
                except FAILURES as error:
                    print(f"[fit] train не прочитан: {error}")
                    return EXIT_BLOCKED

            code = handler(args, train)

        else:
            code = handler(args)

        if code != EXIT_OK:
            print(f"[fit] остановлено на этапе {name}: словарь не собран")
            return code

    print(f"[fit] словарь готов: {VOCAB_DIR}")

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
    add("buckets", "числовые диапазоны и их токены", run_buckets)
    add("bpe", "разбиение текста на train", run_bpe)
    add("final-vocab", "имя токена → глобальный ID: final_vocab.json", run_final_vocab)
    add("value-weights", "веса value-маскирования по train: value_weights.json", run_value_weights)

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
