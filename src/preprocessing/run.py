from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa

from .artifacts import render_validation_md, sha256_ints, write_json, write_table, write_text
from .build import write_processed
from .config import (
    CLIENT_GROUPS,
    DATASET_NAMES,
    DEFAULT_RAW_DIR,
    DEFAULT_SETTINGS,
    SCHEMA_VERSION,
    Settings,
    artifacts_dir,
    processed_dir,
)
from .cutoffs import build_cutoff_index, cutoff_of, cutoff_summary, month_grid, skip_summary
from .fit import run_fit
from .raw import RawDataset
from .splits import assign_groups, month_roles, split_description
from .validate import ValidationError, validate_raw


# ============================================================
# ИДЕЯ
# ============================================================
#
# Оркестратор: RAW → проверка → сплиты и сетка месяцев →
# кандидаты → обучаемые artifacts на train → processed.
#
#     python -m src.preprocessing.run --raw data/raw/smoke --name smoke
#
# Вывод: data/processed/<name>/ и data/artifacts/<name>/.
# В artifacts не попадают ни пути, ни время запуска: два
# запуска на одних данных дают одинаковые байты.
# ============================================================


def clean_directory(path: Path) -> None:

    if not path.exists():
        return

    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        else:
            item.rmdir()


def split_manifest(
    raw: RawDataset,
    settings: Settings,
    groups: dict[int, str],
    roles: dict,
    cutoff_index: pa.Table,
    scope,
    counts: dict[str, int],
) -> dict:

    months = sorted(roles)

    by_group: dict[str, list[int]] = {group: [] for group in CLIENT_GROUPS}

    for client_id, group in sorted(groups.items()):
        by_group[group].append(client_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "split": split_description(settings.split_seed, settings.split_shares),
        "clients": {
            "total": len(groups),
            "counts": {group: len(ids) for group, ids in by_group.items()},
            "shares": {group: len(ids) / len(groups) for group, ids in by_group.items()} if groups else {},
            "sha256": {group: sha256_ints(ids) for group, ids in by_group.items()},
            "fit_clients": scope.n_clients,
        },
        "months": {
            "history_start": raw.manifest.history_start.isoformat(),
            "feature_end": raw.manifest.feature_end.isoformat(),
            "observation_months": [month.strftime("%Y-%m") for month in months],
            "roles": {month.strftime("%Y-%m"): roles[month] for month in months},
            "cutoffs": {month.strftime("%Y-%m"): cutoff_of(month).isoformat() for month in months},
            "by_role": {
                role: [cutoff_of(month).isoformat() for month in months if roles[month] == role]
                for role in sorted(set(roles.values()))
            },
        },
        "datasets": {
            name: {
                "clients": next((group for (group, role), dataset in _dataset_pairs() if dataset == name), None),
                "months": next((role for (group, role), dataset in _dataset_pairs() if dataset == name), None),
                "examples": counts.get(f"{name}/examples", 0),
            }
            for name in DATASET_NAMES
        },
        "candidates": {
            "total": cutoff_index.num_rows,
            "valid": int(sum(1 for value in cutoff_index.column("valid").to_pylist() if value)),
            "skipped": skip_summary(cutoff_index),
            "by_cutoff": cutoff_summary(cutoff_index),
        },
        "rules": {
            "example": "пара (client_id, cutoff); cutoff это начало следующего месяца",
            "events": "ts < cutoff, то есть префикс ленты seq < seq_end",
            "profile": "последний снимок с ts < cutoff",
            "min_observation_days": settings.min_observation_days,
            "observation_start": "max(history_start, first_seen[transactions])",
            "fit_events": "train-клиенты с хотя бы одним валидным train-примером; события с ts < последний валидный train-cutoff клиента, каждое один раз",
            "fit_profile": "as-of снимки, выбранные валидными train-примерами, каждый один раз",
            "consumer": "история примера это seq < seq_end; события train-клиентов намеренно содержат val- и test-месяцы для val_time и test_time",
        },
        "raw": raw.manifest.echo(),
        "preprocessing_config": settings.as_dict(),
        "rows": dict(sorted(counts.items())),
    }


def _dataset_pairs():
    from .config import DATASETS

    return DATASETS.items()


def run(
    raw_dir: Path,
    name: str,
    out_dir: Path | None = None,
    artifacts_out: Path | None = None,
    settings: Settings = DEFAULT_SETTINGS,
    quiet: bool = False,
) -> dict:

    raw = RawDataset(raw_dir)

    out_dir = Path(out_dir) if out_dir else processed_dir(name)
    artifacts_out = Path(artifacts_out) if artifacts_out else artifacts_dir(name)

    clean_directory(out_dir)
    clean_directory(artifacts_out)

    def say(text: str) -> None:
        if not quiet:
            print(text)

    # --------------------------------------------------------
    # ПРОВЕРКА
    # --------------------------------------------------------

    try:
        report = validate_raw(raw)
    except ValidationError as error:
        write_json(artifacts_out / "validation_report.json", error.report)
        write_text(artifacts_out / "validation_report.md", render_validation_md(error.report))
        raise

    write_json(artifacts_out / "validation_report.json", report)
    write_text(artifacts_out / "validation_report.md", render_validation_md(report))

    say(f"проверка RAW: {report['status']}, проверок {len(report['checks'])}")

    # --------------------------------------------------------
    # СПЛИТЫ И КАНДИДАТЫ
    # --------------------------------------------------------

    groups = assign_groups(raw.client_ids(), settings.split_seed, settings.split_shares)

    months = month_grid(raw.manifest.history_start, raw.manifest.feature_end)

    roles = month_roles(months)

    cutoff_index = build_cutoff_index(raw, groups, roles, settings)

    say(f"месяцев {len(months)}, кандидатов {cutoff_index.num_rows}")

    # --------------------------------------------------------
    # ОБУЧАЕМЫЕ ARTIFACTS
    # --------------------------------------------------------

    scope, buckets, bucket_edges, field_stats, unigram_baselines, distributions = run_fit(
        raw, cutoff_index, settings
    )

    say(f"fit: клиентов {scope.n_clients}, снимков {len(scope.snapshots)}")

    write_json(artifacts_out / "bucket_edges.json", bucket_edges)
    write_json(artifacts_out / "field_stats.json", field_stats)
    write_json(artifacts_out / "unigram_baselines.json", unigram_baselines)

    for relative, table in sorted(distributions.items()):
        write_table(artifacts_out / relative, table)

    # --------------------------------------------------------
    # PROCESSED
    # --------------------------------------------------------

    counts = write_processed(raw, cutoff_index, groups, buckets, out_dir)

    manifest = split_manifest(raw, settings, groups, roles, cutoff_index, scope, counts)

    write_json(artifacts_out / "split_manifest.json", manifest)

    if not quiet:
        print()
        print("=" * 60)
        print("PROCESSED")
        print("=" * 60)
        for key, value in sorted(counts.items()):
            print(f"  {key:42s}{value:>10,}".replace(",", " "))
        print()
        print(f"processed: {out_dir}")
        print(f"artifacts: {artifacts_out}")

    return {"counts": counts, "manifest": manifest, "validation": report}


def main() -> None:

    parser = argparse.ArgumentParser(description="Preprocessing: RAW → месячные as-of примеры и artifacts")

    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--name", default=None, help="имя набора; по умолчанию имя каталога RAW")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--buckets", type=int, default=DEFAULT_SETTINGS.default_buckets)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SETTINGS.split_seed)
    parser.add_argument("--min-observation-days", type=int, default=DEFAULT_SETTINGS.min_observation_days)

    args = parser.parse_args()

    settings = Settings(
        split_seed=args.split_seed,
        split_shares=DEFAULT_SETTINGS.split_shares,
        min_observation_days=args.min_observation_days,
        default_buckets=args.buckets,
        bucket_overrides=DEFAULT_SETTINGS.bucket_overrides,
    )

    run(
        raw_dir=args.raw,
        name=args.name or Path(args.raw).name,
        out_dir=args.out,
        artifacts_out=args.artifacts,
        settings=settings,
    )


if __name__ == "__main__":
    main()
