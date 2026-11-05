"""Разворачивает выгрузку генератора в кэш по источникам для проверок реализма.

    python flatten.py <raw_dir> <cache_dir> [--workers 4]

raw_dir — каталог с events.parquet (например, scratch-мир или data/01_raw/train), cache_dir — вне data/.
Кэш: <cache_dir>/<source>/part-NNN.parquet (одна часть — одна row group выгрузки): client_id, seq,
ts (наивное местное время UTC+5), поля payload. Ничего не пишет рядом с выгрузкой.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]


def _task(arg: tuple[str, str, int]) -> None:
    raw, out, rg = arg
    f = pq.ParquetFile(Path(raw) / "events.parquet")
    offset = sum(f.metadata.row_group(i).num_rows for i in range(rg))
    t = f.read_row_group(rg).to_pandas()
    t["seq"] = range(offset, offset + len(t))
    if not t.event_time.str.endswith("+05:00").all():
        raise ValueError("смещение времени не +05:00")
    t["ts"] = pd.to_datetime(t.event_time.str.slice(0, -6), format="ISO8601")
    for source, g in t.groupby("source", sort=False):
        p = pd.DataFrame.from_records([json.loads(x) for x in g.payload])
        for c in p.columns:
            if p[c].map(lambda v: isinstance(v, (dict, list))).any():
                p[c] = p[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        df = pd.concat([g[["client_id", "seq", "ts"]].reset_index(drop=True), p.reset_index(drop=True)], axis=1)
        d = Path(out) / source
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"part-{rg:03d}.parquet", index=False, compression="zstd")


def flatten(raw: Path, out: Path, workers: int = 4) -> None:
    raw, out = Path(raw).resolve(), Path(out).resolve()
    if out.is_relative_to(REPO / "data"):
        raise ValueError("кэш не пишется в data/ проекта")
    groups = pq.ParquetFile(raw / "events.parquet").num_row_groups
    with Pool(workers) as pool:
        pool.map(_task, [(str(raw), str(out), i) for i in range(groups)])
    (out / "SOURCE").write_text(str(raw))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    flatten(a.raw, a.cache, a.workers)
    print("ok", a.cache)
