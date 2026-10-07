"""
Замеры препроцессинга (этап canonical): читает RAW и пишет только в
каталог из --out (data/ запрещён). Какой код мерить, задаёт PYTHONPATH:
снимок старого кода или рабочее дерево.

    run      build_group: время, CPU, пик RSS, объём чтения и записи,
             sha256 файла, логический хеш строк, группы строк, схема;
    profile  cProfile build_group по компонентам;
    counts   сколько раз разбирается type, время, читается лента и
             профиль, вызывается json.loads;
    errors   класс и текст ошибки на каждой негодной выгрузке из
             tests/test_preprocessing_errors.py.

Перехватчики только считают: результат этапа от них не зависит.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

# src берётся из PYTHONPATH, если он задан; tests — из рабочего дерева.
sys.path.append(str(ROOT))


def _outside_data(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(ROOT / "data"):
        raise SystemExit(f"{path}: замеры не пишут в data/")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical(path: Path) -> dict:
    """
    Строки, схема и метаданные файла независимо от его раскладки.
    """

    parquet = pq.ParquetFile(path)
    table = parquet.read().combine_chunks()
    sink = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(None), sink, compression="none",
                   row_group_size=max(1, table.num_rows), write_statistics=False)
    return {
        "rows": table.num_rows,
        "row_groups": parquet.num_row_groups,
        "schema": str(table.schema.remove_metadata()),
        "metadata": {key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()
                     if key != b"ARROW:schema"},
        "logical_sha256": hashlib.sha256(sink.getvalue()).hexdigest(),
        "file_sha256": file_sha256(path),
    }


def _config(batch: int):
    from dataclasses import replace

    from src.preprocessing.settings import PreprocessingConfig

    return replace(PreprocessingConfig.load(None), batch_clients=batch)


def command_run(args) -> None:

    from src.preprocessing.canonical.build import build_group

    out = _outside_data(args.out)
    config = _config(args.batch)

    cpu = time.process_time()
    start = time.perf_counter()

    result = build_group(args.raw, out, config, "train")

    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu

    import src.preprocessing.canonical.build as build_module

    record = {
        "label": args.label,
        "code": str(Path(build_module.__file__).resolve().parents[3]),
        "batch_clients": args.batch,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
        # Учёта ввода-вывода по процессу в WSL2 нет: объём записи — размер
        # файла, объём чтения — проходы по ленте (counts) на размер RAW.
        "raw_events_mib": round((Path(args.raw) / "events.parquet").stat().st_size / 2**20, 1),
        "written_mib": round((out / "events.parquet").stat().st_size / 2**20, 1),
        "events_rows": result.events_rows,
        "clients": result.clients,
    }

    if args.hash:
        record["events"] = logical(out / "events.parquet")

    (out / "bench.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({key: value for key, value in record.items() if key != "events"}, ensure_ascii=False))


# Компоненты: (подпись, конец имени файла или "~" для встроенных, имя функции).
COMPONENTS = [
    ("build_group (всё)", "build.py", "build_group"),
    ("check_raw", "rawdata.py", "check_raw"),
    ("  _check_payloads (type)", "rawdata.py", "_check_payloads"),
    ("  _check_profile_moments", "rawdata.py", "_check_profile_moments"),
    ("read_row_group (parquet)", "~", "read_row_group"),
    ("read_table (parquet)", "~", "read_table"),
    ("iter_client_batches", "events.py", "iter_client_batches"),
    ("_refuse_split_clients", "events.py", "_refuse_split_clients"),
    ("to_pylist (все)", "~", "to_pylist"),
    ("event_types_of", "rawdata.py", "event_types_of"),
    ("iter_event_types", "rawdata.py", "iter_event_types"),
    ("build_batch", "events.py", "build_batch"),
    ("parse_batch", "events.py", "parse_batch"),
    ("parse_payloads", "rawdata.py", "parse_payloads"),
    ("_fast_parse", "rawdata.py", "_fast_parse"),
    ("_slow_parse", "rawdata.py", "_slow_parse"),
    ("parse_event_time", "rawdata.py", "parse_event_time"),
    ("normalize_text", "events.py", "normalize_text"),
    ("lifelong_sources", "events.py", "lifelong_sources"),
    ("json.loads (все)", "__init__.py", "loads"),
    ("concat_tables", "~", "concat_tables"),
    ("pa.table / from_pylist", "~", "table"),
    ("cast", "~", "cast"),
    ("_order (сортировка)", "events.py", "_order"),
    ("sort_indices", "~", "sort_indices"),
    ("TableWriter.write", "artifacts.py", "write"),
    ("ParquetWriter.write_table", "core.py", "write_table"),
    ("_clients", "build.py", "_clients"),
]


def command_profile(args) -> None:

    from src.preprocessing.canonical.build import build_group

    out = _outside_data(args.out)
    config = _config(args.batch)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    build_group(args.raw, out, config, "train")
    profiler.disable()
    wall = time.perf_counter() - start

    stats = pstats.Stats(profiler)
    stats.dump_stats(str(out / "build.prof"))

    table = stats.stats
    total = max(ct for (_, _, name), (_, _, _, ct, _) in table.items() if name == "build_group")

    print(f"build_group под cProfile: {wall:.1f} с")
    print(f"{'компонент':34} {'cumtime, с':>11} {'%':>6} {'вызовов':>10} {'мс/вызов':>10}")

    for label, file_part, name in COMPONENTS:
        cumulative = 0.0
        count = 0
        for (filename, _line, function), (_cc, nc, _tt, ct, _callers) in table.items():
            if file_part == "~":
                if not (function == f"<built-in method {name}>" or function.endswith(f".{name}>")
                        or f"'{name}'" in function or function == name):
                    continue
            elif not filename.endswith(file_part) or function != name:
                continue
            cumulative += ct
            count += nc
        print(f"{label:34} {cumulative:11.2f} {100 * cumulative / total:6.1f} {count:10d} "
              f"{1000 * cumulative / count if count else 0:10.2f}")

    print()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("tottime").print_stats(25)
    print("\n".join(line for line in buffer.getvalue().splitlines()[6:] if line.strip()))


def command_counts(args) -> None:

    import json as json_module

    from src.preprocessing import rawdata
    from src.preprocessing.canonical import build as build_module
    from src.preprocessing.canonical import events

    out = _outside_data(args.out)
    config = _config(args.batch)

    counters: dict[str, int] = defaultdict(int)

    def counted(module, name, rows=None):
        original = getattr(module, name)

        def inner(*items, **named):
            counters[f"{name} вызовов"] += 1
            if rows is not None:
                counters[f"{name} строк"] += rows(*items, **named)
            return original(*items, **named)

        setattr(module, name, inner)
        return original

    saved = []

    # Модули импортируют функции по имени: подменяются все места.
    for module in (rawdata, events):
        if hasattr(module, "event_types_of"):
            saved.append((module, "event_types_of", counted(module, "event_types_of", lambda payload: len(payload))))
        if hasattr(module, "parse_event_time"):
            saved.append((module, "parse_event_time", counted(module, "parse_event_time")))

    for name in ("_fast_parse", "_slow_parse"):
        saved.append((rawdata, name, counted(rawdata, name)))

    saved.append((events, "normalize_text", counted(events, "normalize_text")))

    original_iter = rawdata.RawDataset.iter_row_groups
    original_read = rawdata.RawDataset.read

    def iter_row_groups(self, table, columns=None):
        counters[f"проходов {table} (iter_row_groups), колонки {','.join(columns) if columns else 'все'}"] += 1
        return original_iter(self, table, columns)

    def read(self, table, columns=None):
        counters[f"чтений {table} целиком (read), колонки {','.join(columns) if columns else 'все'}"] += 1
        return original_read(self, table, columns)

    rawdata.RawDataset.iter_row_groups = iter_row_groups
    rawdata.RawDataset.read = read

    original_loads = json_module.loads

    def loads(*items, **named):
        counters["json.loads вызовов"] += 1
        return original_loads(*items, **named)

    json_module.loads = loads

    try:
        build_module.build_group(args.raw, out, config, "train")
    finally:
        json_module.loads = original_loads
        rawdata.RawDataset.iter_row_groups = original_iter
        rawdata.RawDataset.read = original_read
        for module, name, original in saved:
            setattr(module, name, original)

    for key in sorted(counters):
        print(f"{key:80} {counters[key]:>12,}")


def command_errors(args) -> None:

    import tempfile

    from tests import test_preprocessing_errors as cases

    for name, events, snapshot, edit, _expected, _fragment in cases.CASES:
        with tempfile.TemporaryDirectory(dir=_outside_data(args.out)) as directory:
            try:
                cases.run_case(Path(directory), events, snapshot, edit)
                outcome = "ПРОШЛО БЕЗ ОШИБКИ"
            except Exception as error:  # noqa: BLE001 — нужен любой исход
                outcome = f"{type(error).__name__}: {error}"
        print(f"{name}\t{outcome}")


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "profile", "counts", "errors"):
        item = sub.add_parser(name)
        item.add_argument("--out", type=Path, required=True)
        if name != "errors":
            item.add_argument("--raw", type=Path, required=True)
            item.add_argument("--batch", type=int, default=64)
        if name == "run":
            item.add_argument("--label", default="")
            item.add_argument("--hash", action="store_true")

    args = parser.parse_args()

    {"run": command_run, "profile": command_profile, "counts": command_counts,
     "errors": command_errors}[args.command](args)


if __name__ == "__main__":
    main()
