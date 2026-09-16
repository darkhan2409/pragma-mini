from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.preprocessing.artifacts import sha256_file, write_json
from src.preprocessing.build import events_schema, profile_schema
from src.preprocessing.config import CLIENT_GROUPS, DATASET_NAMES
from src.preprocessing.run import clean_directory

from .artifacts import Tokenizer, write_config
from .build import assert_no_mask, write_group
from .config import (
    CONFIG_FILE,
    DATASET_MANIFEST_FILE,
    FORMAT_VERSION,
    GOLDEN_FILE,
    STATS_FILE,
    DEFAULT_SETTINGS,
    IncompatibleArtifactsError,
    TokenizerSettings,
    artifacts_dir as default_artifacts_dir,
    processed_dir as default_processed_dir,
    tokenized_dir as default_tokenized_dir,
    vocab_dir as default_vocab_dir,
)
from .dataset import TokenizedDataset, first_example, golden_example
from .semantics import (
    DEFAULT_KEY_MODE,
    DEFAULT_VALUE_MODE,
    KEY_MODES,
    VALUE_MODES,
    check_mode,
    is_baseline,
)
from .vocab import FitCounters, fit_counters


# ============================================================
# ИДЕЯ
# ============================================================
#
# Оркестратор: processed → словарь на train → токенизированные
# датасеты → статистика и golden-примеры.
#
#     python -m src.tokenizer.run --name smoke
#
# Preprocessing не переписывается и не перезапускается: его
# artifacts только читаются, а их хэши уходят в config.
# ============================================================


def check_processed(processed_dir: Path) -> None:
    """
    Вход обязан быть тем самым processed, а не похожим на него.
    """

    processed_dir = Path(processed_dir)

    expected = {
        "clients/train_clients/events.parquet": events_schema(),
        "clients/train_clients/profile.parquet": profile_schema(),
    }

    for relative, schema in expected.items():

        path = processed_dir / relative

        if not path.exists():
            raise IncompatibleArtifactsError(f"нет {relative} в {processed_dir}")

        if not pq.read_schema(path).remove_metadata().equals(schema):
            raise IncompatibleArtifactsError(f"{relative} имеет чужую схему")

    for group in CLIENT_GROUPS:
        for name in ("events", "profile"):
            path = processed_dir / "clients" / f"{group}_clients" / f"{name}.parquet"
            if not path.exists():
                raise IncompatibleArtifactsError(f"нет clients/{group}_clients/{name}.parquet")

    for dataset in DATASET_NAMES:
        if not (processed_dir / dataset / "examples.parquet").exists():
            raise IncompatibleArtifactsError(f"нет {dataset}/examples.parquet")


def run(
    processed_in: Path,
    artifacts_in: Path,
    out_dir: Path,
    vocab_out: Path,
    settings: TokenizerSettings = DEFAULT_SETTINGS,
    quiet: bool = False,
    key_mode: str = DEFAULT_KEY_MODE,
    value_mode: str = DEFAULT_VALUE_MODE,
    counters: FitCounters | None = None,
    fit_only: bool = False,
) -> dict:

    processed_in = Path(processed_in)
    artifacts_in = Path(artifacts_in)
    out_dir = Path(out_dir)
    vocab_out = Path(vocab_out)

    def say(text: str) -> None:
        if not quiet:
            print(text)

    check_processed(processed_in)

    if not fit_only:
        clean_directory(out_dir)

    clean_directory(vocab_out)

    # --------------------------------------------------------
    # СЛОВАРЬ
    # --------------------------------------------------------

    # Счётчики считаются по ПОЛЯМ и от режима не зависят,
    # поэтому четыре словаря строятся из одного прохода.
    if counters is None:
        counters = fit_counters(processed_in, artifacts_in)

    vocab = counters.build(key_mode, value_mode)
    fit = counters.report

    from src.preprocessing.artifacts import read_json

    split_manifest = read_json(artifacts_in / "split_manifest.json")

    limits = settings.resolve(split_manifest["raw"])

    vocab.save(vocab_out)

    config = write_config(vocab, vocab_out, artifacts_in, limits, fit)

    sharing = vocab.sharing_report()

    say(
        f"словарь [{vocab.key_mode}/{vocab.value_mode}]: полей {vocab.n_fields}, "
        f"key token {vocab.n_key_tokens} (склеено {sharing['n_merged_key_tokens']}), "
        f"value token {vocab.n_values} (склеено {sharing['n_merged_value_tokens']}), "
        f"размер {vocab.size}; fit-клиентов {fit.n_clients}"
    )

    if fit_only:
        say(f"словарь:  {vocab_out}")
        return {"counts": {}, "config": config, "stats": None, "manifest": None, "vocab": vocab}

    # --------------------------------------------------------
    # ДАТАСЕТЫ
    # --------------------------------------------------------

    counts: dict[str, int] = {}
    summaries: dict[str, dict] = {}
    records: dict[str, dict] = {}

    for group in CLIENT_GROUPS:

        group_counts, group_summaries, event_stats, profile_stats = write_group(
            processed_in, out_dir, group, vocab, limits
        )

        counts.update(group_counts)
        summaries.update(group_summaries)

        records[f"{group}_clients"] = {
            "events": event_stats.as_dict(),
            "profile": profile_stats.as_dict(),
        }

    assert_no_mask(out_dir)

    # --------------------------------------------------------
    # МАНИФЕСТ, СТАТИСТИКА, GOLDEN
    # --------------------------------------------------------

    manifest = {
        "format_version": FORMAT_VERSION,
        "tokenizer_config_sha256": sha256_file(vocab_out / CONFIG_FILE),
        "vocab_size": vocab.size,
        "modes": vocab.modes,
        "rows": dict(sorted(counts.items())),
        "rule": (
            "записи хранятся один раз на группу клиентов; пример это ссылка "
            "(client_id, cutoff, seq_end, snapshot_ts), история это префикс seq < seq_end"
        ),
    }

    write_json(out_dir / DATASET_MANIFEST_FILE, manifest)

    golden = {}

    for dataset in DATASET_NAMES:

        data = TokenizedDataset(out_dir, dataset, vocab_dir=vocab_out)

        example = first_example(data)

        if example is not None:
            golden[dataset] = golden_example(vocab, example)

    write_json(vocab_out / GOLDEN_FILE, {"note": "первый пример каждого датасета в порядке (client_id, cutoff)", "examples": golden})

    stats = {
        "format_version": FORMAT_VERSION,
        "fit": fit.as_dict(),
        "vocab": {
            "modes": vocab.modes,
            "n_fields": vocab.n_fields,
            "n_keys": vocab.n_key_tokens,
            "n_values": vocab.n_values,
            "size": vocab.size,
            "values_by_key": {entry.key: entry.n_values for entry in vocab.fields},
            "sharing": sharing,
        },
        "limits": limits.as_dict(),
        "records": records,
        "examples": summaries,
        "note": (
            "records считает каждую запись один раз; examples показывает, как их видит модель, "
            "то есть с повторами по cutoff"
        ),
    }

    write_json(vocab_out / STATS_FILE, stats)

    if not quiet:
        print()
        print("=" * 60)
        print("TOKENIZED")
        print("=" * 60)
        for key, value in sorted(counts.items()):
            print(f"  {key:42s}{value:>10,}".replace(",", " "))
        print()
        print(f"датасеты: {out_dir}")
        print(f"словарь:  {vocab_out}")

    return {"counts": counts, "config": config, "stats": stats, "manifest": manifest}


# ============================================================
# ЧЕТЫРЕ РЕЖИМА ИЗ ОДНОГО ПРОХОДА
# ============================================================
#
# Счётчики значений считаются ПО ПОЛЯМ, а склейка меняет только
# выдачу token_id. Значит проход по train нужен один, а словарей
# из него получается четыре.
# ============================================================

MODE_TAGS: tuple[tuple[str, str, str], ...] = (
    ("baseline", DEFAULT_KEY_MODE, DEFAULT_VALUE_MODE),
    ("semkeys", "semantic", DEFAULT_VALUE_MODE),
    ("shared", DEFAULT_KEY_MODE, "shared"),
    ("semkeys_shared", "semantic", "shared"),
)


def fit_all_modes(
    processed_in: Path,
    artifacts_in: Path,
    name: str,
    settings: TokenizerSettings = DEFAULT_SETTINGS,
    quiet: bool = False,
) -> dict:
    """
    Четыре словаря из одного прохода по train.

    Датасеты не токенизируются: это отчёт о размерах и склейке,
    а не подготовка арок.
    """

    processed_in = Path(processed_in)
    artifacts_in = Path(artifacts_in)

    counters = fit_counters(processed_in, artifacts_in)

    report: dict[str, dict] = {}

    for tag, key_mode, value_mode in MODE_TAGS:

        result = run(
            processed_in=processed_in,
            artifacts_in=artifacts_in,
            out_dir=default_tokenized_dir(name, tag),
            vocab_out=default_vocab_dir(name, tag),
            settings=settings,
            quiet=True,
            key_mode=key_mode,
            value_mode=value_mode,
            counters=counters,
            fit_only=True,
        )

        vocab = result["vocab"]

        report[tag] = {
            "tag": tag,
            "modes": vocab.modes,
            "vocab_dir": str(default_vocab_dir(name, tag)),
            **vocab.sharing_report(),
        }

    if not quiet:
        print()
        print("=" * 78)
        print("VOCAB MODES")
        print("=" * 78)
        header = f"  {'tag':16s}{'key_mode':12s}{'value_mode':16s}{'keys':>6s}{'values':>8s}{'size':>7s}{'merged k/v':>13s}"
        print(header)
        for tag, _, _ in MODE_TAGS:
            item = report[tag]
            merged = f"{item['n_merged_key_tokens']}/{item['n_merged_value_tokens']}"
            print(
                f"  {tag:16s}{item['modes']['key_mode']:12s}"
                f"{item['modes']['categorical_value_mode']:16s}"
                f"{item['n_key_tokens']:6d}{item['n_value_tokens']:8d}{item['size']:7d}{merged:>13s}"
            )
        print()
        print(f"  полей во всех режимах: {report['baseline']['n_fields']}")

    return report


def main() -> None:

    # Отчёт содержит кириллицу, а консоль Windows по умолчанию
    # cp1251: без этого печать падает уже ПОСЛЕ записи файлов.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Tokenizer: processed → словарь и токенизированные датасеты"
    )

    parser.add_argument("--name", default="smoke", help="имя набора")
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--vocab-out", type=Path, default=None)
    parser.add_argument("--max-tokens-per-event", type=int, default=None)
    parser.add_argument("--max-events-per-history", type=int, default=None)

    parser.add_argument("--key-mode", choices=KEY_MODES, default=DEFAULT_KEY_MODE)
    parser.add_argument("--categorical-value-mode", choices=VALUE_MODES, default=DEFAULT_VALUE_MODE)
    parser.add_argument(
        "--vocab-tag",
        default=None,
        help="суффикс каталогов словаря и датасета; обязателен для режима со склейкой",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="остановиться после словаря: датасеты не токенизировать",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="все четыре режима из одного прохода по train (только с --fit-only)",
    )

    args = parser.parse_args()

    settings = TokenizerSettings(
        max_tokens_per_event=args.max_tokens_per_event,
        max_events_per_history=args.max_events_per_history,
    )

    processed_in = args.processed or default_processed_dir(args.name)
    artifacts_in = args.artifacts or default_artifacts_dir(args.name)

    if args.all_modes:

        if not args.fit_only:
            parser.error("--all-modes имеет смысл только с --fit-only: иначе это четыре токенизации")

        fit_all_modes(processed_in, artifacts_in, args.name, settings)
        return

    check_mode(args.key_mode, args.categorical_value_mode)

    # Словарь со склейкой обязан называть свой тег: без него он
    # записался бы поверх baseline, и старые checkpoint перестали
    # бы открываться.
    if not is_baseline(args.key_mode, args.categorical_value_mode) and args.vocab_tag is None:
        parser.error(
            f"режим {args.key_mode}/{args.categorical_value_mode} требует --vocab-tag: "
            "без него словарь затёр бы baseline-артефакты"
        )

    run(
        processed_in=processed_in,
        artifacts_in=artifacts_in,
        out_dir=args.out or default_tokenized_dir(args.name, args.vocab_tag),
        vocab_out=args.vocab_out or default_vocab_dir(args.name, args.vocab_tag),
        settings=settings,
        key_mode=args.key_mode,
        value_mode=args.categorical_value_mode,
        fit_only=args.fit_only,
    )


if __name__ == "__main__":
    main()


__all__ = ["run", "main", "Tokenizer", "check_processed", "fit_all_modes", "MODE_TAGS"]
