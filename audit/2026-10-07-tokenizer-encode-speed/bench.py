"""
Замеры encode токенизатора: группа train из --root (раскладка data/ во
временном каталоге со словарём в --root/03_vocab), результат — в --out.
Настоящий data/ не трогается. Какой код мерить, задаёт PYTHONPATH.

    run      encode_group: время, CPU, пик RSS, логические отпечатки
             events/profile (не зависят от раскладки по группам строк),
             sha256 файлов, meta, счётчики отчёта; --timers добавляет
             лёгкие таймеры и счётчики вызовов (с их накладными);
    profile  cProfile encode_group по компонентам;
    compare  две выдачи: логические отпечатки, схемы, meta, отчёт и
             TokenizedGroup.clients() клиент в клиент;
    read     проход TokenizedGroup.clients() по выдаче: время и пик RSS
             читателя датасета (раскладка групп строк влияет на него);
    errors   маленький мир генератора в --work и испорченные входы
             encode_group: исход каждого (ошибка с текстом или
             отпечатки выдачи) строкой JSON — для сверки двух версий кода.

--set модуль:имя=значение подменяет константу модуля (размер пачки
записи и т. п.) только в этом процессе.
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


def _redirect(root: Path) -> None:
    """Каталоги этапов — во временный корень, как в tests/conftest.py."""

    for module, attribute, folder in (
        ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
        ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
        ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
        ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
        ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ):
        setattr(import_module(module), attribute, root / folder)


def _apply_sets(items: list[str]) -> dict:
    applied = {}
    for item in items or ():
        target, value = item.split("=", 1)
        module, name = target.split(":")
        parsed = ast.literal_eval(value)
        setattr(import_module(module), name, parsed)
        applied[target] = parsed
    return applied


def _encode(out: Path) -> dict:

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    return encode_group(FrozenArtifacts.load(), GROUP, TokenizerConfig.load(None), directory=out)


# ------------------------------------------------------------
# логический отпечаток
# ------------------------------------------------------------


def logical_digest(path: Path, batch_rows: int = 65536) -> dict:
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
            "max_row_group_rows": max((metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)), default=0),
            "file_bytes": path.stat().st_size,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _outputs(out: Path) -> dict:
    return {
        "events": logical_digest(out / "events.parquet"),
        "profile": logical_digest(out / "profile.parquet"),
        "meta_sha256": hashlib.sha256((out / "meta.json").read_bytes()).hexdigest(),
    }


# ------------------------------------------------------------
# run
# ------------------------------------------------------------


# Лёгкие таймеры: (модуль, класс или None, имя).
TIMED = [
    ("src.preprocessing.read", "Group", "history"),
    ("src.preprocessing.read", "Group", "_addresses"),
    ("src.preprocessing.read", "Group", "events_table"),
    ("src.preprocessing.read", "Group", "_build"),
    ("src.preprocessing.read", "_ClientCursor", "take"),
    ("src.preprocessing.read", "_ClientCursor", "skip_before"),
    ("src.preprocessing.read", None, "client_events"),
    ("src.preprocessing.read", None, "profile_at"),
    ("src.preprocessing.read", None, "calendar_features"),
    ("src.preprocessing.read", None, "event_values"),
    ("src.tokenization.transform", None, "encode_event"),
    ("src.tokenization.transform", None, "encode_profile"),
    ("src.tokenization.transform", None, "_count_unknown"),
    ("src.tokenization.encode", None, "normalize_text"),
    ("src.tokenization.text", "BpeModel", "pieces"),
    ("src.tokenization.finalvocab", "FrozenArtifacts", "bucket_id"),
    ("src.preprocessing.read", "ClientEvent", "model_values"),
    ("src.preprocessing.artifacts", "TableWriter", "write"),
    ("src.preprocessing.artifacts", "TableWriter", "close"),
]


class _TableProxy:
    """pa.Table для transform: from_pylist и from_arrays с таймером."""

    def __init__(self, timer):
        import pyarrow as pa
        self._table = pa.Table
        self.from_pylist = timer("pa.Table.from_pylist", pa.Table.from_pylist)
        self.from_arrays = timer("pa.Table.from_arrays", pa.Table.from_arrays)
        self.from_pydict = timer("pa.Table.from_pydict", pa.Table.from_pydict)

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
    reads: dict[str, int] = defaultdict(int)
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
            setattr(owner, name, timer(f"{owner_name + '.' if owner_name else ''}{name}", original))
            restore.append((owner, name, original))

        transform = import_module("src.tokenization.transform")
        restore.append((transform, "pa", transform.pa))
        transform.pa = _ArrowProxy(timer)

        events_path = root / "02_preprocessed" / GROUP / "events.parquet"
        rows_total = pq.read_metadata(events_path).num_rows

        for name in ("read_row_groups", "read_row_group", "iter_batches"):
            original = getattr(pq.ParquetFile, name)

            def counted(self, *items, _name=name, _original=original, **named):
                if self.metadata.num_rows == rows_total:
                    groups = named.get("row_groups") if _name == "iter_batches" else items[0] if items else named.get("row_groups", named.get("i"))
                    reads[f"лента: {_name}, вызовов"] += 1
                    reads[f"лента: {_name}, групп строк"] += len(groups) if isinstance(groups, (list, tuple)) else 1
                return _original(self, *items, **named)

            setattr(pq.ParquetFile, name, counted)
            restore.append((pq.ParquetFile, name, original))

        write_table = pq.ParquetWriter.write_table
        setattr(pq.ParquetWriter, "write_table", timer("ParquetWriter.write_table", write_table))
        restore.append((pq.ParquetWriter, "write_table", write_table))

    cpu = time.process_time()
    start = time.perf_counter()

    try:
        report = _encode(out)
    finally:
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)

    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    from src.tokenization import transform

    record = {
        "label": args.label,
        "code": str(Path(transform.__file__).resolve().parents[2]),
        "set": applied,
        "timers_on": bool(args.timers),
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "peak_rss_gib": round(rss, 3),
        "report": {key: report[key] for key in ("rows", "counts", "unknown_values", "unknown_keys")},
        "report_meta": report["meta"],
        "outputs": _outputs(out),
        "timers": {label: [round(spent[label], 2), calls[label]] for label in spent},
        "reads": dict(reads),
    }

    path = out.parent / f"bench-{args.label}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    outputs = record["outputs"]
    print(json.dumps({"label": args.label, "wall_s": record["wall_s"], "cpu_s": record["cpu_s"],
                      "peak_rss_gib": record["peak_rss_gib"], "set": applied,
                      "events": [outputs["events"]["sha256"][:16], outputs["events"]["row_groups"],
                                 outputs["events"]["file_bytes"]],
                      "profile": [outputs["profile"]["sha256"][:16], outputs["profile"]["row_groups"],
                                  outputs["profile"]["file_bytes"]],
                      "meta": outputs["meta_sha256"][:16]}, ensure_ascii=False))
    for label, (seconds, count) in sorted(record["timers"].items(), key=lambda item: -item[1][0]):
        print(f"  {label:34} {seconds:8.2f} с {100 * seconds / wall:5.1f}%  вызовов {count}")
    for label, count in sorted(reads.items()):
        print(f"  {label:34} {count}")


# ------------------------------------------------------------
# profile
# ------------------------------------------------------------


COMPONENTS = [
    ("encode_group (всё)", "transform.py", "encode_group"),
    ("Group.__init__", "read.py", "__init__"),
    ("Group._addresses (индекс)", "read.py", "_addresses"),
    ("Group.history", "read.py", "history"),
    ("Group.histories / _stream", "read.py", "_stream"),
    ("Group.events_table", "read.py", "events_table"),
    ("read_row_groups / read_row_group", "~", "read_row_group"),
    ("Group._build", "read.py", "_build"),
    ("  to_pylist (все)", "~", "to_pylist"),
    ("  calendar_features", "calendar.py", "calendar_features"),
    ("  event_values", "read.py", "event_values"),
    ("  model_event", "projection.py", "model_event"),
    ("  profile_at", "profile_state.py", "profile_at"),
    ("encode_event", "encode.py", "encode_event"),
    ("  model_values (копия)", "read.py", "model_values"),
    ("  encode_values", "encode.py", "encode_values"),
    ("  _value_ids", "encode.py", "_value_ids"),
    ("  _text_value_ids", "encode.py", "_text_value_ids"),
    ("  normalize_text", "events.py", "normalize_text"),
    ("  BpeModel.pieces", "text.py", "pieces"),
    ("  piece_id", "finalvocab.py", "piece_id"),
    ("  categorical_id", "finalvocab.py", "categorical_id"),
    ("  value_text", "scan.py", "value_text"),
    ("  bucket_id", "finalvocab.py", "bucket_id"),
    ("  locate", "numeric.py", "locate"),
    ("  EncodedRecord.add", "encode.py", "add"),
    ("  EncodedRecord.check", "encode.py", "check"),
    ("  EncodedRecord.n_values", "encode.py", "n_values"),
    ("  kind", "finalvocab.py", "kind"),
    ("  key_id", "finalvocab.py", "key_id"),
    ("  special", "finalvocab.py", "special"),
    ("encode_profile", "encode.py", "encode_profile"),
    ("_count_unknown", "transform.py", "_count_unknown"),
    ("from_pylist", "~", "from_pylist"),
    ("TableWriter.write", "artifacts.py", "write"),
    ("  ParquetWriter.write_table", "parquet/core.py", "write_table"),
    ("TableWriter.close", "artifacts.py", "close"),
]


def command_profile(args) -> None:

    root = _outside_data(args.root)
    out = _outside_data(args.out)
    _redirect(root)
    _apply_sets(args.set)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    _encode(out)
    profiler.disable()
    wall = time.perf_counter() - start

    target = out.parent / f"encode-{args.label}.prof"
    stats = pstats.Stats(profiler)
    stats.dump_stats(str(target))

    table = stats.stats
    total = max(ct for (_, _, name), (_, _, _, ct, _) in table.items() if name == "encode_group")

    print(f"encode_group под cProfile: {wall:.1f} с")
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


def _clients(root: Path, out: Path):

    from src.dataset.tokenized import TokenizedGroup
    from src.tokenization.finalvocab import FrozenArtifacts

    return TokenizedGroup(GROUP, FrozenArtifacts.load(), directory=out).clients()


def command_compare(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    left, right = _outside_data(args.left), _outside_data(args.right)

    problems: list[str] = []

    import pyarrow.parquet as pq

    for name in ("events.parquet", "profile.parquet"):
        a, b = logical_digest(left / name), logical_digest(right / name)
        if a["sha256"] != b["sha256"]:
            problems.append(f"{name}: логический отпечаток {a['parts']} против {b['parts']}")
        if not pq.ParquetFile(left / name).schema_arrow.equals(pq.ParquetFile(right / name).schema_arrow,
                                                                check_metadata=True):
            problems.append(f"{name}: схемы различаются")
        print(f"{name}: строк {a['rows']}/{b['rows']}, групп строк {a['row_groups']}/{b['row_groups']}, "
              f"файл {a['file_bytes']}/{b['file_bytes']} байт, "
              f"байты {'равны' if a['file_sha256'] == b['file_sha256'] else 'различаются'}")

    if (left / "meta.json").read_bytes() != (right / "meta.json").read_bytes():
        problems.append("meta.json различается")

    reports = [json.loads((path.parent / f"bench-{label}.json").read_text())["report"]
               for path, label in ((left, args.left_label), (right, args.right_label)) if args.left_label]
    if reports and reports[0] != reports[1]:
        problems.append(f"отчёт encode_group различается: {reports[0]} против {reports[1]}")

    clients = 0
    events = 0
    typed = 0
    sources = 0
    for one, two in zip(_clients(root, left), _clients(root, right), strict=True):
        clients += 1
        events += one.n_events
        typed += sum(1 for event in one.events if event.event_type is not None)
        sources += sum(1 for event in one.events if event.lifelong_source is not None)
        if one != two:
            problems.append(f"TokenizedGroup: клиент {one.client_id} различается")
            if len(problems) > 20:
                break

    print(f"TokenizedGroup: клиентов {clients}, событий {events}, с типом {typed}, "
          f"источников вех {sources}")

    if problems:
        print("РАЗЛИЧИЯ:")
        for item in problems:
            print("  " + item)
        raise SystemExit(1)

    print("РАВНЫ: логическое содержимое events/profile, схемы, meta"
          + (", отчёт" if reports else "") + ", TokenizedGroup клиент в клиент")


def command_read(args) -> None:

    root = _outside_data(args.root)
    out = _outside_data(args.out)
    _redirect(root)

    start = time.perf_counter()
    clients = events = 0
    for client in _clients(root, out):
        clients += 1
        events += client.n_events
    wall = time.perf_counter() - start

    print(json.dumps({"out": str(out), "clients": clients, "events": events, "wall_s": round(wall, 2),
                      "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)}))


# ------------------------------------------------------------
# errors
# ------------------------------------------------------------


def command_errors(args) -> None:

    import dataclasses
    import shutil
    from datetime import datetime
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    work = _outside_data(args.work)
    shutil.rmtree(work, ignore_errors=True)
    root = work / "data"
    _redirect(root)

    from src.generator import emit
    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.settings import PreprocessingConfig, group_dir, raw_group_dir
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.run import run_fit
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    emit.generate_dataset(total_clients=12, out_dir=raw_group_dir(GROUP), seed=77, world_seed=42,
                          history_start=datetime(2025, 9, 1), history_end=datetime(2026, 1, 1),
                          workers=1, community_size=4, quiet=True)
    build_group(raw_group_dir(GROUP), group_dir(GROUP), PreprocessingConfig.load(None), GROUP)

    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        assert run_fit(SimpleNamespace(config=None)) == 0

    tape_path = group_dir(GROUP) / "events.parquet"
    profile_path = raw_group_dir(GROUP) / "profile.parquet"
    tape = pq.read_table(tape_path)
    profile = pq.read_table(profile_path)
    vocab = {name: hashlib.sha256((root / "03_vocab" / name).read_bytes()).hexdigest()[:16]
             for name in sorted(path.name for path in (root / "03_vocab").iterdir())}
    print(json.dumps({"case": "vocab", "outcome": vocab}, ensure_ascii=False))

    ids = tape.column("client_id").to_pylist()
    first = min(ids)

    def column(table, name, values):
        index = table.schema.get_field_index(name)
        return table.set_column(index, table.schema.field(index), pa.array(values, table.schema.field(index).type))

    def edit_tape(function):
        return lambda: pq.write_table(function(tape), tape_path)

    def edit_profile(function):
        return lambda: pq.write_table(function(profile), profile_path)

    def nan_number(table):
        # Первое заполненное дробное поле (rate): NaN до словаря доходить не должен.
        name = next(field.name for field in table.schema if pa.types.is_floating(field.type)
                    and table.column(field.name).null_count < table.num_rows)
        numbers = table.column(name).to_pylist()
        row = next(index for index, value in enumerate(numbers) if value is not None)
        numbers[row] = float("nan")
        return column(table, name, numbers)

    def false_source(table):
        marks = table.column("lifelong_source").to_pylist()
        types = table.column("type").to_pylist()
        row = next(index for index, kind in enumerate(types) if kind in ("purchase", "app_screen"))
        marks[row] = "first_loan_opened"
        return column(table, "lifelong_source", marks)

    def unknown_source(table):
        sources = table.column("source").to_pylist()
        return column(table, "source", [f"unknown_{index % 3}" if index % 97 == 5 else value
                                        for index, value in enumerate(sources)])

    def early_as_of(table):
        moments = table.column("as_of").to_pylist()
        moments[3] = datetime(2025, 10, 1, tzinfo=moments[3].tzinfo)
        return column(table, "as_of", moments)

    def descending(table):
        order = sorted(set(ids), reverse=True)
        return table.take([index for client in order for index, value in enumerate(ids) if value == client])

    def strangers(table):
        extra = [table.slice(0, 5).set_column(0, "client_id", pa.array([name] * 5)) for name in (f"{first}~", "~z")]
        return pa.concat_tables([table, *extra]).sort_by("client_id")

    artifacts = FrozenArtifacts.load()
    config = TokenizerConfig.load(None)
    event_key = "merchant_city" if "merchant_city" in artifacts.keys else sorted(artifacts.keys)[-1]

    cases = [
        ("clean", None, artifacts, config),
        ("stale vocab: ключ, которого больше нет", None,
         dataclasses.replace(artifacts, keys={**artifacts.keys, "profile_pensioner": 999}), config),
        ("vocab без ключа анкеты", None,
         dataclasses.replace(artifacts, keys={key: value for key, value in artifacts.keys.items()
                                              if key != "profile_job_tenure_months"}), config),
        (f"vocab без ключа события {event_key}: неизвестные ключи", None,
         dataclasses.replace(artifacts, keys={key: value for key, value in artifacts.keys.items() if key != event_key}),
         config),
        ("NaN в числе", edit_tape(nan_number), artifacts, config),
        ("предел кусков BPE 1", None, artifacts, dataclasses.replace(config, max_pieces_per_value=1)),
        ("ложная пометка источника вехи", edit_tape(false_source), artifacts, config),
        ("поле без смысла у источника", edit_tape(unknown_source), artifacts, config),
        ("cutoff позже as_of анкеты", edit_profile(early_as_of), artifacts, config),
        ("клиенты ленты не по порядку", edit_tape(descending), artifacts, config),
        ("клиенты ленты без анкеты", edit_tape(strangers), artifacts, config),
        ("клиент на границе групп строк", lambda: pq.write_table(tape, tape_path, row_group_size=97), artifacts, config),
    ]

    for number, (name, prepare, items, settings) in enumerate(cases):

        pq.write_table(tape, tape_path)
        pq.write_table(profile, profile_path)

        if prepare is not None:
            prepare()

        out = work / f"case-{number}"

        try:
            report = encode_group(items, GROUP, settings, directory=out)
        except Exception as error:  # noqa: BLE001 — сверяется любой исход
            outcome = {"error": type(error).__name__, "message": str(error),
                       "cause": type(error.__cause__).__name__ if error.__cause__ else None}
        else:
            outputs = _outputs(out)
            outcome = {"events": outputs["events"]["sha256"][:16], "profile": outputs["profile"]["sha256"][:16],
                       "meta": outputs["meta_sha256"][:16],
                       "report": {key: report[key] for key in ("counts", "unknown_values", "unknown_keys")}}

        print(json.dumps({"case": name, "outcome": outcome}, ensure_ascii=False, sort_keys=True))


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "profile"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--label", default="run")
        item.add_argument("--set", action="append")
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

    errors = sub.add_parser("errors")
    errors.add_argument("--work", type=Path, required=True)

    args = parser.parse_args()

    {"run": command_run, "profile": command_profile, "compare": command_compare,
     "read": command_read, "errors": command_errors}[args.command](args)


if __name__ == "__main__":
    main()
