"""
Замеры fit токенизатора: читает train из --root (раскладка data/ во
временном каталоге) и пишет словари туда же, в --root/03_vocab.
Настоящий data/ не трогается. Какой код мерить, задаёт PYTHONPATH.

    prepare  раскладка --root: лента препроцессинга и анкета RAW;
    run      run_fit: время, CPU, пик RSS, sha256 артефактов и слепок
             FitStatistics (всё, что увидел проход по корпусу);
    profile  cProfile run_fit по компонентам;
    counts   чтения групп строк ленты, байты, копии значений событий,
             хэши выборки, _trim, корпус BPE.

Перехватчики только считают: результат fit от них не зависит.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import resource
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

ARTIFACTS = ("special_tokens.json", "key_vocab.json", "value_vocab.json", "buckets.json",
             "bpe.json", "final_vocab.json", "value_weights.json")


def _outside_data(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(ROOT / "data"):
        raise SystemExit(f"{path}: замеры не пишут в data/")
    return path


def _redirect(root: Path) -> None:
    """Каталоги этапов — во временный корень, как в tests/conftest.py."""

    from importlib import import_module

    for module, attribute, folder in (
        ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
        ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
        ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
        ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
        ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ):
        setattr(import_module(module), attribute, root / folder)


def _fit(root: Path) -> int:

    from src.tokenization import run

    shutil.rmtree(root / "03_vocab", ignore_errors=True)

    return run.run_fit(argparse.Namespace(config=None))


def statistics_digest(stats) -> dict:
    """
    Всё содержимое FitStatistics в каноническом виде и его sha256.
    Выборки чисел — вместе с хэшами единиц, после отбора k наименьших.
    """

    def sketch(item):
        item._trim()
        return {"summary": item.summary(), "items": sorted(item._items),
                "distinct": sorted(item._distinct) if item.distinct_exact else None,
                "clients": item.clients}

    payload = {
        "categorical": sorted((list(key), entry.count, entry.clients) for key, entry in stats.categorical.items()),
        "numeric": {key: sketch(value) for key, value in sorted(stats.numeric.items())},
        "split_numeric": [([key, condition], sketch(value)) for (key, condition), value in
                          sorted(stats.split_numeric.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or ""))],
        "text": {key: sorted((text, entry.count, entry.clients, entry.example_raw, entry.max_bytes)
                             for text, entry in bucket.items()) for key, bucket in sorted(stats.text.items())},
        "text_empty": sorted(stats.text_empty.items()),
        "key_counts": sorted((key, value.events, value.clients) for key, value in stats.key_counts.items()),
        "key_types": sorted((key, sorted(value)) for key, value in stats.key_types.items()),
        "event_types": sorted(stats.event_types.items()),
        "unknown_event_types": sorted(stats.unknown_event_types.items()),
        "missing": sorted((list(key), value) for key, value in stats.missing.items()),
        "unknown_keys": sorted(stats.unknown_keys.items()),
        "limitations": sorted(stats.limitations.items()),
        "totals": [stats.clients, stats.events, stats.values, stats.clients_without_profile, stats.profiles],
    }

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    parts = {name: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]
             for name, value in payload.items()}

    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "parts": parts}


def command_prepare(args) -> None:

    root = _outside_data(args.root)

    for path, target in ((args.events, root / "02_preprocessed" / "train" / "events.parquet"),
                         (args.profile, root / "01_raw" / "train" / "profile.parquet")):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    print(f"{root}: лента и анкета на месте")


# Крупные части fit для лёгких таймеров: (модуль, объект или None, имя).
TIMED = [
    ("src.tokenization.run", None, "read_train"),
    ("src.preprocessing.read", "Group", "__init__"),
    ("src.preprocessing.read", "Group", "_addresses"),
    ("src.preprocessing.read", "Group", "history"),
    ("src.preprocessing.read", "Group", "events_table"),
    ("src.preprocessing.read", "Group", "_build"),
    ("src.tokenization.fit", None, "scan"),
    ("src.tokenization.run", None, "build_value_vocab"),
    ("src.tokenization.run", None, "build_buckets"),
    ("src.tokenization.run", None, "build_bpe"),
    ("src.tokenization.text", None, "train_bpe"),
    ("src.tokenization.text", None, "check_roundtrip"),
    ("src.tokenization.run", None, "build_final_vocab"),
    ("src.tokenization.run", None, "value_counts"),
    ("src.tokenization.run", None, "write_json"),
]


def command_run(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    from importlib import import_module

    import pyarrow.parquet as pq

    from src.tokenization import run

    captured = {}
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

    for module_name, owner_name, name in TIMED:
        module = import_module(module_name)
        owner = getattr(module, owner_name) if owner_name else module
        original_item = owner.__dict__.get(name) if owner_name else getattr(module, name, None)
        if original_item is None:
            continue
        label = f"{owner_name + '.' if owner_name else ''}{name}"
        if name == "read_train":
            def keep(*items, _inner=timer(label, original_item), **named):
                captured["train"] = _inner(*items, **named)
                return captured["train"]
            setattr(owner, name, keep)
        else:
            setattr(owner, name, timer(label, original_item))
        restore.append((owner, name, original_item))

    reads = defaultdict(int)
    events_path = root / "02_preprocessed" / "train" / "events.parquet"
    rows_total = pq.read_metadata(events_path).num_rows
    original_groups = pq.ParquetFile.read_row_groups
    original_batches = pq.ParquetFile.iter_batches

    def read_row_groups(self, row_groups, *items, **named):
        if self.metadata.num_rows == rows_total:
            reads["групп строк ленты прочитано (read_row_groups)"] += len(row_groups)
        return original_groups(self, row_groups, *items, **named)

    def iter_batches(self, *items, **named):
        if self.metadata.num_rows == rows_total:
            reads["групп строк ленты открыто курсором (iter_batches)"] += len(named.get("row_groups") or [])
        return original_batches(self, *items, **named)

    pq.ParquetFile.read_row_groups = read_row_groups
    pq.ParquetFile.iter_batches = iter_batches

    cpu = time.process_time()
    start = time.perf_counter()

    try:
        code = _fit(root)
    finally:
        for owner, name, original_item in restore:
            setattr(owner, name, original_item)
        pq.ParquetFile.read_row_groups = original_groups
        pq.ParquetFile.iter_batches = original_batches

    wall = time.perf_counter() - start
    cpu = time.process_time() - cpu

    from src.tokenization import settings as tokenizer_settings

    vocab = Path(tokenizer_settings.VOCAB_DIR)

    record = {
        "label": args.label,
        "code": str(Path(run.__file__).resolve().parents[2]),
        "exit": code,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
        "timers": {label: [round(spent[label], 2), calls[label]] for label in spent},
        "reads": dict(reads),
        "artifacts": {name: hashlib.sha256((vocab / name).read_bytes()).hexdigest()
                      for name in ARTIFACTS if (vocab / name).exists()},
    }

    if "train" in captured:
        record["statistics"] = statistics_digest(captured["train"].statistics)

    out = root / f"bench-{args.label}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({key: value for key, value in record.items() if key not in ("artifacts", "statistics", "timers")},
                     ensure_ascii=False))
    for label, (seconds, count) in sorted(record["timers"].items(), key=lambda item: -item[1][0]):
        print(f"  {label:28} {seconds:8.2f} с {100 * seconds / wall:5.1f}%  вызовов {count}")


COMPONENTS = [
    ("run_fit (всё)", "run.py", "run_fit"),
    ("read_train", "fit.py", "read_train"),
    ("Group.__init__", "read.py", "__init__"),
    ("Group._addresses (индекс)", "read.py", "_addresses"),
    ("Group.histories", "read.py", "histories"),
    ("Group.history", "read.py", "history"),
    ("Group.events_table", "read.py", "events_table"),
    ("read_row_groups / read_row_group", "~", "read_row_group"),
    ("to_pylist (все)", "~", "to_pylist"),
    ("calendar_features", "calendar.py", "calendar_features"),
    ("event_values", "read.py", "event_values"),
    ("model_event", "projection.py", "model_event"),
    ("profile_at", "profile_state.py", "profile_at"),
    ("scan", "scan.py", "scan"),
    ("  NumericSketch.add", "scan.py", "add"),
    ("  _unit_hash (blake2b)", "scan.py", "_unit_hash"),
    ("  NumericSketch._trim", "scan.py", "_trim"),
    ("  _add_categorical", "scan.py", "_add_categorical"),
    ("  _add_text", "scan.py", "_add_text"),
    ("  normalize_text", "events.py", "normalize_text"),
    ("  model_values (копия)", "read.py", "model_values"),
    ("build_value_vocab", "categorical.py", "build_value_vocab"),
    ("build_buckets", "numeric.py", "build_buckets"),
    ("build_bpe (всё)", "text.py", "build_bpe"),
    ("  train_bpe", "text.py", "train_bpe"),
    ("  corpus_rows", "text.py", "corpus_rows"),
    ("  check_roundtrip", "text.py", "check_roundtrip"),
    ("build_final_vocab", "finalvocab.py", "build_final_vocab"),
    ("value_counts", "valuestats.py", "value_counts"),
    ("write_json", "artifacts.py", "write_json"),
]


def command_profile(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    _fit(root)
    profiler.disable()
    wall = time.perf_counter() - start

    out = root / f"fit-{args.label}.prof"
    stats = pstats.Stats(profiler)
    stats.dump_stats(str(out))

    table = stats.stats
    total = max(ct for (_, _, name), (_, _, _, ct, _) in table.items() if name == "run_fit")

    print(f"run_fit под cProfile: {wall:.1f} с")
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
    pstats.Stats(profiler, stream=buffer).sort_stats("tottime").print_stats(22)
    print("\n".join(line for line in buffer.getvalue().splitlines()[6:] if line.strip()))


def command_counts(args) -> None:

    root = _outside_data(args.root)
    _redirect(root)

    import pyarrow.parquet as pq

    from src.preprocessing import read as read_module
    from src.tokenization import scan as scan_module
    from src.tokenization import text as text_module

    counters: dict[str, int] = defaultdict(int)
    groups_read: dict[int, int] = defaultdict(int)
    trim_sizes: list[int] = []

    events_path = root / "02_preprocessed" / "train" / "events.parquet"
    metadata = pq.read_metadata(events_path)
    group_bytes = [metadata.row_group(index).total_byte_size for index in range(metadata.num_row_groups)]
    group_compressed = [sum(metadata.row_group(index).column(column).total_compressed_size
                            for column in range(metadata.num_columns))
                        for index in range(metadata.num_row_groups)]

    original_groups = pq.ParquetFile.read_row_groups
    original_group = pq.ParquetFile.read_row_group

    def read_row_groups(self, row_groups, columns=None, **named):
        # Лента опознаётся по числу строк и групп: анкету не считаем.
        if (self.metadata.num_rows, self.metadata.num_row_groups) == (metadata.num_rows, metadata.num_row_groups):
            counters["read_row_groups вызовов"] += 1
            for index in row_groups:
                groups_read[index] += 1
                counters["прочитано групп строк (ленты)"] += 1
                counters["прочитано байт сжатых"] += group_compressed[index] if columns is None else 0
        return original_groups(self, row_groups, columns=columns, **named)

    def read_row_group(self, index, columns=None, **named):
        counters[f"read_row_group вызовов, колонки {','.join(columns) if columns else 'все'}"] += 1
        return original_group(self, index, columns=columns, **named)

    pq.ParquetFile.read_row_groups = read_row_groups
    pq.ParquetFile.read_row_group = read_row_group

    original_values = read_module.ClientEvent.model_values

    def model_values(self):
        counters["model_values (копия dict)"] += 1
        return original_values(self)

    read_module.ClientEvent.model_values = model_values

    original_hash = scan_module._unit_hash

    def unit_hash(text):
        counters["_unit_hash (blake2b)"] += 1
        return original_hash(text)

    scan_module._unit_hash = unit_hash

    original_trim = scan_module.NumericSketch._trim

    def trim(self):
        trim_sizes.append(len(self._items))
        return original_trim(self)

    scan_module.NumericSketch._trim = trim

    original_normalize = scan_module.normalize_text

    def normalize(value):
        counters["normalize_text в scan"] += 1
        result = original_normalize(value)
        if result != value:
            counters["normalize_text в scan изменил значение"] += 1
        return result

    scan_module.normalize_text = normalize

    original_rows = text_module.corpus_rows
    corpus = {}

    def corpus_rows(stats, keys):
        rows = original_rows(stats, keys)
        corpus["unique"] = len(rows)
        corpus["occurrences"] = sum(count for _key, _text, count in rows)
        return rows

    text_module.corpus_rows = corpus_rows

    try:
        _fit(root)
    finally:
        pq.ParquetFile.read_row_groups = original_groups
        pq.ParquetFile.read_row_group = original_group
        read_module.ClientEvent.model_values = original_values
        scan_module._unit_hash = original_hash
        scan_module.NumericSketch._trim = original_trim
        scan_module.normalize_text = original_normalize
        text_module.corpus_rows = original_rows

    print(f"лента: групп строк {metadata.num_row_groups}, строк {metadata.num_rows}, "
          f"файл {events_path.stat().st_size / 2**20:.1f} МиБ, сжатых байт по группам {sum(group_compressed) / 2**20:.1f} МиБ")
    reads = sorted(groups_read.items())
    print("чтений каждой группы строк:", ", ".join(f"{index}:{count}" for index, count in reads[:20])
          + (" …" if len(reads) > 20 else ""))
    for key in sorted(counters):
        value = counters[key]
        shown = f"{value / 2**20:,.1f} МиБ" if "байт" in key else f"{value:,}"
        print(f"{key:60} {shown:>16}")
    if trim_sizes:
        print(f"_trim: вызовов {len(trim_sizes)}, размер перед отбором — среднее {sum(trim_sizes) / len(trim_sizes):.0f}, "
              f"максимум {max(trim_sizes)}")
    if corpus:
        print(f"корпус BPE: уникальных текстов {corpus['unique']:,}, вхождений {corpus['occurrences']:,}, "
              f"отношение {corpus['occurrences'] / max(1, corpus['unique']):.1f}")


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--events", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)

    for name in ("run", "profile", "counts"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--label", default="run")

    args = parser.parse_args()

    {"prepare": command_prepare, "run": command_run, "profile": command_profile,
     "counts": command_counts}[args.command](args)


if __name__ == "__main__":
    main()
