"""
Замеры сборки набора (05_dataset): группа train из --root (раскладка data/
во временном каталоге: 03_vocab и 04_tokenized/train), результат — в --out.
Настоящий data/ не трогается. Какой код мерить, задаёт PYTHONPATH.

    run      build_group: время, CPU, пик RSS, логический отпечаток
             samples.parquet (не зависит от раскладки по группам строк),
             sha256 файлов, meta, отчёт; --timers добавляет лёгкие таймеры
             и счётчики вызовов (с их накладными);
    profile  cProfile build_group по компонентам;
    compare  две выдачи: логический отпечаток, схема, meta, отчёт и
             TemporalGroup строка в строку (с посчитанным временем);
    read     чтение набора теми, кто его читает: TemporalGroup по группам
             строк, Source.sizes() и Source.clients() (маска обучения);
             время и пик RSS.

--set модуль:имя=значение подменяет константу модуля только в этом процессе;
--row-group-samples задаёт размер группы строк набора.
"""

from __future__ import annotations

import argparse
import ast
import cProfile
import hashlib
import io
import json
import pstats
import resource
import sys
import time
from collections import defaultdict
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

GROUP = "train"


def _outside_data(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(ROOT / "data"):
        raise SystemExit(f"{path}: замеры не пишут в data/")
    return path


def _redirect(root: Path, out: Path | None = None) -> None:
    """Каталоги этапов — во временный корень, как в tests/conftest.py."""

    for module, attribute, folder in (
        ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
        ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
        ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
        ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
        ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ):
        setattr(import_module(module), attribute, root / folder)

    # Набор читается из каталога группы: выдача --out и есть он.
    if out is not None:
        setattr(import_module("src.dataset.settings"), "DATASET_DIR", out.parent)


def _apply_sets(items: list[str] | None) -> dict:
    applied = {}
    for item in items or ():
        target, value = item.split("=", 1)
        module, name = target.split(":")
        parsed = ast.literal_eval(value)
        setattr(import_module(module), name, parsed)
        applied[target] = parsed
    return applied


def _config(row_group_samples: int | None):

    from dataclasses import replace

    from src.dataset.settings import DatasetConfig

    config = DatasetConfig.load(None)

    return config if row_group_samples is None else replace(config, row_group_samples=row_group_samples)


def _build(out: Path, row_group_samples: int | None) -> dict:

    from src.dataset.build import build_group
    from src.tokenization.finalvocab import FrozenArtifacts

    return build_group(FrozenArtifacts.load(), GROUP, _config(row_group_samples), directory=out)


# ------------------------------------------------------------
# логический отпечаток
# ------------------------------------------------------------


def logical_digest(path: Path, batch_rows: int = 64) -> dict:
    """
    sha256 содержимого таблицы по колонкам, без зависимости от раскладки
    по группам строк и кускам: строки идут потоком, у списков отдельно
    длины и плоские значения, у времени отдельно маска null.
    """

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    hashes = {field.name: hashlib.sha256() for field in schema}
    rows = 0

    def flat(column: pa.Array, digest) -> None:
        if pa.types.is_timestamp(column.type):
            valid = pc.is_valid(column).to_numpy(zero_copy_only=False)
            digest.update(np.packbits(valid).tobytes() + b"|")
            digest.update(pc.fill_null(column.cast(pa.int64()), -1).to_numpy().astype("<i8").tobytes())
        elif pa.types.is_boolean(column.type):
            if column.null_count:
                raise SystemExit(f"{path}: null в булевой колонке")
            digest.update(np.packbits(column.to_numpy(zero_copy_only=False)).tobytes() + b"|")
        elif pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
            if column.null_count:
                raise SystemExit(f"{path}: null в числовой колонке")
            digest.update(column.to_numpy().tobytes())
        else:
            for value in column.to_pylist():
                digest.update(b"\x00" if value is None else b"\x01" + value.encode("utf-8") + b"\x02")

    for batch in parquet.iter_batches(batch_size=batch_rows):
        rows += batch.num_rows
        for field in schema:
            column = batch.column(field.name)
            digest = hashes[field.name]
            if pa.types.is_list(field.type):
                if column.null_count:
                    raise SystemExit(f"{path}: null-список в {field.name}")
                digest.update(b"L" + pc.list_value_length(column).to_numpy().astype("<i8").tobytes())
                flat(pc.list_flatten(column), digest)
            else:
                flat(column, digest)

    parts = {name: digest.hexdigest()[:16] for name, digest in hashes.items()}
    total = hashlib.sha256((schema.to_string(show_schema_metadata=True) + json.dumps(parts, sort_keys=True)
                            + str(rows)).encode()).hexdigest()

    metadata = parquet.metadata

    return {"sha256": total, "rows": rows, "parts": parts,
            "row_groups": metadata.num_row_groups,
            "file_bytes": path.stat().st_size,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _outputs(out: Path) -> dict:
    return {
        "samples": logical_digest(out / "samples.parquet"),
        "meta_sha256": hashlib.sha256((out / "meta.json").read_bytes()).hexdigest(),
    }


# ------------------------------------------------------------
# run
# ------------------------------------------------------------


# Лёгкие таймеры: (модуль, класс или None, имя).
TIMED = [
    ("src.tokenization.finalvocab", "FrozenArtifacts", "load"),
    ("src.dataset.tokenized", "TokenizedGroup", "__init__"),
    ("src.dataset.tokenized", "TokenizedGroup", "_client"),
    ("src.dataset.tokenized", None, "event_type_of"),
    ("src.dataset.build", None, "build_sample"),
    ("src.dataset.sample", None, "select"),
    ("src.dataset.sample", None, "eligible"),
    ("src.dataset.sample", "Sample", "check"),
    ("src.dataset.sample", None, "_check_positions"),
    ("src.dataset.sample", None, "_check_profile_time"),
    ("src.dataset.sample", None, "_utc_moments"),
    ("src.dataset.sample", None, "_ints"),
    ("src.dataset.build", None, "_count_unknown"),
    ("src.dataset.build", None, "_row"),
    ("src.dataset.build", None, "_table"),
    ("src.dataset.build", None, "_write"),
    ("src.dataset.tokenized", "TokenizedGroup", "_client_rows"),
    ("src.dataset.sample", None, "_events_hold"),
    ("src.dataset.sample", None, "border"),
]


class _TableProxy:

    def __init__(self, timer):
        import pyarrow as pa
        self._table = pa.Table
        self.from_pylist = timer("pa.Table.from_pylist", pa.Table.from_pylist)
        self.from_pydict = timer("pa.Table.from_pydict", pa.Table.from_pydict)
        self.from_arrays = timer("pa.Table.from_arrays", pa.Table.from_arrays)

    def __getattr__(self, name):
        return getattr(self._table, name)


class _ArrowProxy:

    def __init__(self, timer):
        import pyarrow as pa
        self._pa = pa
        self.Table = _TableProxy(timer)

    def __getattr__(self, name):
        return getattr(self._pa, name)


def command_run(args) -> None:

    root = _outside_data(args.root)
    out = _outside_data(args.out)
    _redirect(root)
    applied = _apply_sets(args.set)

    import pyarrow.parquet as pq

    spent: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    restore = []

    def timer(label, function):
        def inner(*items, **named):
            start = time.perf_counter()
            try:
                return function(*items, **named)
            finally:
                spent[label] += time.perf_counter() - start
                calls[label] += 1
        return inner

    if args.timers:

        for module_name, owner_name, name in TIMED:
            module = import_module(module_name)
            owner = getattr(module, owner_name) if owner_name else module
            original = owner.__dict__.get(name) if owner_name else getattr(module, name, None)
            if original is None:
                continue
            if isinstance(original, staticmethod):
                wrapped = staticmethod(timer(f"{owner_name}.{name}", original.__func__))
            else:
                wrapped = timer(f"{owner_name + '.' if owner_name else ''}{name}", original)
            setattr(owner, name, wrapped)
            restore.append((owner, name, original))

        build = import_module("src.dataset.build")
        restore.append((build, "pa", build.pa))
        build.pa = _ArrowProxy(timer)

        for owner, name in ((pq.ParquetFile, "read_row_group"), (pq.ParquetFile, "read_row_groups"),
                            (pq.ParquetFile, "iter_batches"), (pq.ParquetWriter, "write_table")):
            original = getattr(owner, name)
            setattr(owner, name, timer(f"{owner.__name__}.{name}", original))
            restore.append((owner, name, original))

    cpu = time.process_time()
    start = time.perf_counter()

    try:
        report = _build(out, args.row_group_samples)
    finally:
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)

    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    from src.dataset import build

    record = {
        "label": args.label,
        "code": str(Path(build.__file__).resolve().parents[2]),
        "set": applied,
        "row_group_samples": args.row_group_samples,
        "timers_on": bool(args.timers),
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "peak_rss_gib": round(rss, 3),
        "report": {key: report[key] for key in ("counts", "unknown_values")},
        "report_meta": report["meta"],
        "outputs": _outputs(out),
        "timers": {label: [round(spent[label], 2), calls[label]] for label in spent},
    }

    path = out.parent / f"bench-{args.label}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    outputs = record["outputs"]
    print(json.dumps({"label": args.label, "wall_s": record["wall_s"], "cpu_s": record["cpu_s"],
                      "peak_rss_gib": record["peak_rss_gib"], "set": applied,
                      "samples": [outputs["samples"]["sha256"][:16], outputs["samples"]["row_groups"],
                                  outputs["samples"]["file_bytes"], outputs["samples"]["file_sha256"][:16]],
                      "meta": outputs["meta_sha256"][:16]}, ensure_ascii=False))
    for label, (seconds, count) in sorted(record["timers"].items(), key=lambda item: -item[1][0]):
        print(f"  {label:34} {seconds:8.2f} с {100 * seconds / wall:5.1f}%  вызовов {count}")


# ------------------------------------------------------------
# profile
# ------------------------------------------------------------


COMPONENTS = [
    ("build_group (всё)", "build.py", "build_group"),
    ("FrozenArtifacts.load", "finalvocab.py", "load"),
    ("TokenizedGroup.__init__", "tokenized.py", "__init__"),
    ("TokenizedGroup.clients", "tokenized.py", "clients"),
    ("  _client_rows", "tokenized.py", "_client_rows"),
    ("  read_row_group", "~", "read_row_group"),
    ("  _client", "tokenized.py", "_client"),
    ("  event_type_of", "tokenized.py", "event_type_of"),
    ("  describe", "finalvocab.py", "describe"),
    ("build_sample", "sample.py", "build_sample"),
    ("  eligible", "targets.py", "eligible"),
    ("  can_be_target", "targets.py", "can_be_target"),
    ("  select", "context.py", "select"),
    ("  border", "context.py", "border"),
    ("  _check_limits", "context.py", "_check_limits"),
    ("  _tail", "context.py", "_tail"),
    ("  _split", "context.py", "_split"),
    ("  _ints", "sample.py", "_ints"),
    ("  _utc_moments", "sample.py", "_utc_moments"),
    ("  Sample.check", "sample.py", "check"),
    ("    _events_hold", "sample.py", "_events_hold"),
    ("    _check_positions", "sample.py", "_check_positions"),
    ("    _check_profile_time", "sample.py", "_check_profile_time"),
    ("n_values / profile_n_values", "sample.py", "n_values"),
    ("_count_unknown", "build.py", "_count_unknown"),
    ("_row", "build.py", "_row"),
    ("_table", "build.py", "_table"),
    ("_write", "build.py", "_write"),
    ("  from_pylist", "~", "from_pylist"),
    ("  ParquetWriter.write_table", "parquet/core.py", "write_table"),
]


def command_profile(args) -> None:

    root = _outside_data(args.root)
    out = _outside_data(args.out)
    _redirect(root)
    _apply_sets(args.set)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    _build(out, args.row_group_samples)
    profiler.disable()
    wall = time.perf_counter() - start

    target = out.parent / f"dataset-{args.label}.prof"
    stats = pstats.Stats(profiler)
    stats.dump_stats(str(target))

    table = stats.stats
    total = max(ct for (_, _, name), (_, _, _, ct, _) in table.items() if name == "build_group")

    print(f"build_group под cProfile: {wall:.1f} с")
    print(f"{'компонент':36} {'cumtime, с':>11} {'%':>6} {'вызовов':>11} {'мкс/вызов':>11}")

    for label, file_part, name in COMPONENTS:
        cumulative, count = 0.0, 0
        for (filename, _line, function), (_cc, nc, _tt, ct, _callers) in table.items():
            if file_part == "~":
                if name not in function:
                    continue
            elif not filename.endswith(file_part) or function != name:
                continue
            cumulative += ct
            count += nc
        print(f"{label:36} {cumulative:11.2f} {100 * cumulative / total:6.1f} {count:11d} "
              f"{1e6 * cumulative / count if count else 0:11.1f}")

    print()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("tottime").print_stats(25)
    print("\n".join(line for line in buffer.getvalue().splitlines()[6:] if line.strip()))


# ------------------------------------------------------------
# compare / read
# ------------------------------------------------------------


def command_compare(args) -> None:

    root = _outside_data(args.root)
    left, right = _outside_data(args.left), _outside_data(args.right)
    _redirect(root)

    import pyarrow.parquet as pq

    from src.temporal.samples import TemporalGroup

    problems: list[str] = []

    a, b = logical_digest(left / "samples.parquet"), logical_digest(right / "samples.parquet")

    if a["sha256"] != b["sha256"]:
        problems.append(f"samples.parquet: логический отпечаток {a['parts']} против {b['parts']}")

    if not pq.ParquetFile(left / "samples.parquet").schema_arrow.equals(
            pq.ParquetFile(right / "samples.parquet").schema_arrow, check_metadata=True):
        problems.append("samples.parquet: схемы различаются")

    print(f"samples.parquet: строк {a['rows']}/{b['rows']}, групп строк {a['row_groups']}/{b['row_groups']}, "
          f"файл {a['file_bytes']}/{b['file_bytes']} байт, "
          f"байты {'равны' if a['file_sha256'] == b['file_sha256'] else 'различаются'}")

    if (left / "meta.json").read_bytes() != (right / "meta.json").read_bytes():
        problems.append("meta.json различается")

    if args.left_label and args.right_label:
        reports = [json.loads((path.parent / f"bench-{label}.json").read_text())["report"]
                   for path, label in ((left, args.left_label), (right, args.right_label))]
        if reports[0] != reports[1]:
            problems.append(f"отчёт build_group различается: {reports[0]} против {reports[1]}")

    def rows(directory: Path):
        group = TemporalGroup(GROUP, directory)
        for number in range(group.count):
            yield from group.row_group(number).to_pylist()

    count = 0
    for one, two in zip(rows(left), rows(right), strict=True):
        count += 1
        if one != two:
            problems.append(f"TemporalGroup: клиент {one['client_id']} различается")
            if len(problems) > 20:
                break

    print(f"TemporalGroup: строк {count}")

    if problems:
        print("РАЗЛИЧИЯ:")
        for item in problems:
            print("  " + item)
        raise SystemExit(1)

    print("РАВНЫ: логическое содержимое samples, схема, meta"
          + (", отчёт" if args.left_label else "") + ", TemporalGroup строка в строку")


def command_read(args) -> None:

    root = _outside_data(args.root)
    out = _outside_data(args.out)
    _redirect(root, out)

    from src.mlm.inputs import Source
    from src.temporal.samples import TemporalGroup

    timings = {}

    start = time.perf_counter()
    group = TemporalGroup(GROUP, out)
    rows = 0
    for number in range(group.count):
        rows += group.row_group(number).num_rows
    timings["temporal_row_groups_s"] = round(time.perf_counter() - start, 2)

    source = Source(GROUP)

    start = time.perf_counter()
    sizes = sum(1 for _ in source.sizes())
    timings["source_sizes_s"] = round(time.perf_counter() - start, 2)

    start = time.perf_counter()
    clients = tokens = 0
    for client in source.clients():
        clients += 1
        tokens += client.n_tokens
    timings["source_clients_s"] = round(time.perf_counter() - start, 2)

    print(json.dumps({"out": out.name, "row_groups": group.count, "rows": rows, "sizes": sizes,
                      "clients": clients, "tokens": tokens, **timings,
                      "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)}))


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "profile"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--label", default="run")
        item.add_argument("--set", action="append")
        item.add_argument("--row-group-samples", type=int, default=None)
        if name == "run":
            item.add_argument("--timers", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("--root", type=Path, required=True)
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--left-label")
    compare.add_argument("--right-label")

    read = sub.add_parser("read")
    read.add_argument("--root", type=Path, required=True)
    read.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    {"run": command_run, "profile": command_profile, "compare": command_compare,
     "read": command_read}[args.command](args)


if __name__ == "__main__":
    main()
