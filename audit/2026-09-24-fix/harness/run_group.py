"""
Регрессионная генерация объявленной группы В КАТАЛОГ АУДИТА.

    python audit/2026-09-24-fix/harness/run_group.py --group train

Параметры группы берутся из config.DATASETS и здесь не
переопределяются: проверяется ровно то, чем будут учить модель.
Каталог — runs/regression/<группа>, data/ не трогается.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT / "harness"))

from run_gen import inside_runs, peak_memory  # noqa: E402
from sources import SourcesChanged, verify  # noqa: E402


def main() -> int:

    parser = argparse.ArgumentParser(prog="run_group")

    parser.add_argument("--group", required=True)
    parser.add_argument("--root", default="regression")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-clients", type=int, default=256)

    args = parser.parse_args()

    root = inside_runs(RUNS / args.root)

    try:
        before = verify("до прогона")
    except SourcesChanged as error:
        print(error)
        return 2

    from src.generator import config, emit

    started = time.perf_counter()

    counts = emit.generate_group(
        args.group,
        workers=args.workers,
        chunk_clients=args.chunk_clients,
        quiet=True,
        root=root,
    )

    seconds = time.perf_counter() - started

    try:
        after = verify("после прогона")
    except SourcesChanged as error:
        print(error)
        return 2

    if after != before:
        print(f"состояние кода изменилось во время прогона: {before} -> {after}")
        return 2

    out = root / args.group

    settings = config.DATASETS[args.group]

    record = {
        "group": args.group,
        "code_state": before,
        "out": str(out),
        "clients": settings.clients,
        "seed": settings.seed,
        "world_seed": config.WORLD_SEED,
        "start": settings.history_start.isoformat(),
        "end": settings.history_end.isoformat(),
        "workers": args.workers,
        "chunk_clients": args.chunk_clients,
        "counts": counts,
        "seconds": round(seconds, 2),
        "peak_bytes": peak_memory(),
        "files": {
            path.name: path.stat().st_size for path in sorted(out.iterdir()) if path.is_file()
        },
    }

    (out / "run_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(record, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
