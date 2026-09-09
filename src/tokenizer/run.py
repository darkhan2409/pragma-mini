from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from src.preprocessing.artifacts import sha256_file, write_json
from src.generator.version import RAW_SCHEMA_REVISION, REVISION_KEY
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
from .vocab import fit_vocab


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


def check_processed(processed_dir: Path, revision: int = RAW_SCHEMA_REVISION) -> None:
    """
    Вход обязан быть тем самым processed, а не похожим на него.
    """

    processed_dir = Path(processed_dir)

    expected = {
        "clients/train_clients/events.parquet": events_schema(revision),
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
) -> dict:

    processed_in = Path(processed_in)
    artifacts_in = Path(artifacts_in)
    out_dir = Path(out_dir)
    vocab_out = Path(vocab_out)

    def say(text: str) -> None:
        if not quiet:
            print(text)

    from src.preprocessing.artifacts import read_json as _read_json

    revision = int(
        _read_json(Path(artifacts_in) / "split_manifest.json")
        .get("raw", {})
        .get(REVISION_KEY, 1)
    )

    check_processed(processed_in, revision)

    clean_directory(out_dir)
    clean_directory(vocab_out)

    # --------------------------------------------------------
    # СЛОВАРЬ
    # --------------------------------------------------------

    vocab, fit = fit_vocab(processed_in, artifacts_in)

    from src.preprocessing.artifacts import read_json

    split_manifest = read_json(artifacts_in / "split_manifest.json")

    limits = settings.resolve(split_manifest["raw"])

    vocab.save(vocab_out)

    config = write_config(vocab, vocab_out, artifacts_in, limits, fit)

    say(
        f"словарь: ключей {vocab.n_keys}, значений {vocab.n_values}, "
        f"размер {vocab.size}; fit-клиентов {fit.n_clients}"
    )

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
            "n_keys": vocab.n_keys,
            "n_values": vocab.n_values,
            "size": vocab.size,
            "values_by_key": {entry.key: entry.n_values for entry in vocab.keys},
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


def main() -> None:

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

    args = parser.parse_args()

    run(
        processed_in=args.processed or default_processed_dir(args.name),
        artifacts_in=args.artifacts or default_artifacts_dir(args.name),
        out_dir=args.out or default_tokenized_dir(args.name),
        vocab_out=args.vocab_out or default_vocab_dir(args.name),
        settings=TokenizerSettings(
            max_tokens_per_event=args.max_tokens_per_event,
            max_events_per_history=args.max_events_per_history,
        ),
    )


if __name__ == "__main__":
    main()


__all__ = ["run", "main", "Tokenizer", "check_processed"]
