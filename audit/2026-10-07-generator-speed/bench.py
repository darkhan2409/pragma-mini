"""
Замеры скорости генератора: только читает код генератора и пишет в
каталог из --out (data/ запрещён).

    run      полный прогон generate_dataset: время, память дерева
             процессов, загрузка CPU, своп, диск, sha256 файлов и
             логический хеш строк (не зависит от раскладки parquet);
    profile  cProfile одной пачки в процессе: симуляция, Arrow,
             parquet, sha256 частей — по компонентам;
    stats    счётчики одной пачки: прогрев, пустые дни, действия на
             клиент-день, очередь и её пересортировки.

Перехватчики только считают и засекают время: розыгрыши, порядок и
состояние генератора они не трогают.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def _outside_data(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to(ROOT / "data"):
        raise SystemExit(f"{path}: замеры не пишут в data/")
    return path


# ------------------------------------------------------------
# система
# ------------------------------------------------------------


def _meminfo() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            name, rest = line.split(":", 1)
            values[name] = int(rest.split()[0]) * 1024
    return values


def _cpu_jiffies() -> tuple[int, int]:
    with open("/proc/stat", encoding="ascii") as handle:
        fields = [int(item) for item in handle.readline().split()[1:]]
    idle = fields[3] + fields[4]
    return sum(fields) - idle, sum(fields)


def _disk_sectors() -> tuple[int, int]:
    read = written = 0
    with open("/proc/diskstats", encoding="ascii") as handle:
        for line in handle:
            parts = line.split()
            name = parts[2]
            if name.startswith(("sd", "nvme", "vd")) and not name[-1].isdigit():
                read += int(parts[5])
                written += int(parts[9])
    return read * 512, written * 512


def _tree_rss(root: int) -> tuple[int, int, int, int]:
    """
    Сумма RSS процесса и всех его потомков, RSS самого процесса,
    наибольший RSS потомка и число процессов дерева.
    """

    parent: dict[int, int] = {}
    rss: dict[int, int] = {}

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="ascii") as handle:
                stat = handle.read()
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            with open(f"/proc/{entry}/status", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        rss[int(entry)] = int(line.split()[1]) * 1024
                        break
            parent[int(entry)] = ppid
        except (OSError, ValueError, IndexError):
            continue

    tree = {root}
    grew = True
    while grew:
        grew = False
        for pid, ppid in parent.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                grew = True

    sizes = [rss.get(pid, 0) for pid in tree]
    children = [rss.get(pid, 0) for pid in tree if pid != root]

    return sum(sizes), rss.get(root, 0), max(children, default=0), len(tree)


class Sampler(threading.Thread):

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop = threading.Event()
        self.peak_tree = 0
        self.peak_main = 0
        self.peak_child = 0
        self.peak_processes = 0
        self.min_available = None
        self.min_swap_free = None

    def run(self) -> None:
        while not self.stop.is_set():
            total, main, child, count = _tree_rss(os.getpid())
            self.peak_tree = max(self.peak_tree, total)
            self.peak_main = max(self.peak_main, main)
            self.peak_child = max(self.peak_child, child)
            self.peak_processes = max(self.peak_processes, count)
            info = _meminfo()
            available = info.get("MemAvailable", 0)
            swap = info.get("SwapFree", 0)
            self.min_available = available if self.min_available is None else min(self.min_available, available)
            self.min_swap_free = swap if self.min_swap_free is None else min(self.min_swap_free, swap)
            self.stop.wait(self.interval)


# ------------------------------------------------------------
# хеши
# ------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(path: Path) -> tuple[int, str]:
    """
    Хеш строк таблицы, не зависящий от раскладки файла: таблица
    перезаписывается одной группой строк без сжатия в память.
    Одинаковые строки в одинаковом порядке дают одинаковый хеш.
    """

    table = pq.read_table(path).combine_chunks()
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="none", row_group_size=max(1, table.num_rows),
                   write_statistics=False, store_schema=True)
    return table.num_rows, hashlib.sha256(sink.getvalue()).hexdigest()


def digests(out: Path) -> dict:
    from src.generator.emit import TABLES

    result = {}
    for name, (relative, _) in TABLES.items():
        path = out / relative
        rows, logical = logical_sha256(path)
        result[name] = {"rows": rows, "file_sha256": file_sha256(path), "logical_sha256": logical}
    return result


# ------------------------------------------------------------
# run
# ------------------------------------------------------------


def command_run(args) -> None:

    from src.generator import emit

    out = _outside_data(args.out)

    phase: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)

    def timed(name, function):
        def inner(*items, **named):
            start = time.perf_counter()
            try:
                return function(*items, **named)
            finally:
                phase[name] += time.perf_counter() - start
                calls[name] += 1
        return inner

    # Только главный процесс: склейка и итоговые sha256.
    original_merge, original_sha = emit._merge_parts, emit._file_sha256
    emit._merge_parts = timed("merge_parts", original_merge)
    emit._file_sha256 = timed("file_sha256", original_sha)

    sampler = Sampler()
    swap_before = _meminfo()
    disk_before = _disk_sectors()
    cpu_before = _cpu_jiffies()
    times_before = os.times()

    sampler.start()
    start = time.perf_counter()

    try:
        emit.generate_dataset(
            total_clients=args.clients,
            out_dir=out,
            seed=args.seed,
            world_seed=42,
            history_start=datetime.fromisoformat(args.start),
            history_end=datetime.fromisoformat(args.end),
            registration_end=datetime.fromisoformat(args.registration or args.end),
            workers=args.workers,
            chunk_clients=args.chunk,
            quiet=True,
        )
    finally:
        wall = time.perf_counter() - start
        sampler.stop.set()
        sampler.join()
        emit._merge_parts, emit._file_sha256 = original_merge, original_sha

    cpu_after = _cpu_jiffies()
    disk_after = _disk_sectors()
    times_after = os.times()

    busy = (cpu_after[0] - cpu_before[0]) / max(1, cpu_after[1] - cpu_before[1])

    record = {
        "label": args.label,
        "code": str(Path(emit.__file__).resolve().parents[2]),
        "clients": args.clients,
        "seed": args.seed,
        "horizon": [args.start, args.end, args.registration or args.end],
        "workers": args.workers,
        "chunk_clients": args.chunk,
        "wall_s": round(wall, 2),
        "merge_s": round(phase["merge_parts"], 2),
        "main_sha256_s": round(phase["file_sha256"], 2),
        "main_process_cpu_s": round((times_after.user - times_before.user) + (times_after.system - times_before.system), 2),
        "system_cpu_busy": round(busy, 3),
        "cpus": os.cpu_count(),
        "peak_tree_rss_gib": round(sampler.peak_tree / 2**30, 3),
        "peak_main_rss_gib": round(sampler.peak_main / 2**30, 3),
        "peak_worker_rss_gib": round(sampler.peak_child / 2**30, 3),
        "peak_processes": sampler.peak_processes,
        "min_available_gib": round((sampler.min_available or 0) / 2**30, 3),
        "swap_used_delta_gib": round(
            ((swap_before["SwapTotal"] - (sampler.min_swap_free or swap_before["SwapFree"]))
             - (swap_before["SwapTotal"] - swap_before["SwapFree"])) / 2**30, 3),
        "disk_read_mib": round((disk_after[0] - disk_before[0]) / 2**20, 1),
        "disk_written_mib": round((disk_after[1] - disk_before[1]) / 2**20, 1),
    }

    if args.hash:
        record["tables"] = digests(out)

    (out / "bench.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(record, ensure_ascii=False))


# ------------------------------------------------------------
# пачка в процессе (profile, stats)
# ------------------------------------------------------------


def _prepare_batch(args):
    """
    Окружение воркера и задание пачки — так же, как generate_dataset.
    """

    from src.generator import config, emit
    from src.generator import params as params_module
    from src.generator import rng as rng_module
    from src.generator.world import communities

    out = _outside_data(args.out)

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    registration = datetime.fromisoformat(args.registration or args.end)

    config.activate_horizon(start, end, registration)

    horizon = (config.HISTORY_START.isoformat(), config.HISTORY_END.isoformat(),
               config.REGISTRATION_END.isoformat())

    settings = emit._build_params(None, None, None)
    params_module.activate(settings)
    rng_module.configure(args.seed, settings.fingerprint(), 42)

    emit._worker_init(args.seed, None, None, None, rng_module.current_world_seed(), horizon)

    size = settings.relationships.community_size
    count = communities.community_count(args.clients)
    per_batch = max(1, args.chunk // size)

    ids = tuple(range(args.batch * per_batch, min((args.batch + 1) * per_batch, count)))

    out.mkdir(parents=True, exist_ok=True)

    return (args.batch, 0, ids, args.clients, str(out)), out


# Компоненты: функция генератора -> подпись в таблице. Время — cumtime
# функции (вложенные вызовы входят в родителя).
COMPONENTS = [
    ("run_batch", "emit.py", "_run_batch"),
    ("run_community", "engine.py", "run_community"),
    ("prehistory", "simulate.py", "_prehistory"),
    ("burn_in", "engagement.py", "burn_in"),
    ("engagement._step (все)", "engagement.py", "_step"),
    ("_plan_day", "engine.py", "_plan_day"),
    ("engagement.advance", "engagement.py", "advance"),
    ("needs.daily_intents (покупки)", "needs.py", "daily_intents"),
    ("_execute (все действия)", "engine.py", "_execute"),
    ("  _on_purchase", "engine.py", "_on_purchase"),
    ("  _on_transfer", "engine.py", "_on_transfer"),
    ("  _on_session (app)", "engine_app.py", "_on_session"),
    ("  _on_income", "engine.py", "_on_income"),
    ("  _on_bill", "engine.py", "_on_bill"),
    ("  _on_cash", "engine.py", "_on_cash"),
    ("_month_end", "engine_month.py", "_month_end"),
    ("credit (engine_credit, всё)", "engine_credit.py", None),
    ("ClientState.emit", "simulate.py", "emit"),
    ("engagement.note", "engagement.py", "note"),
    ("_finish (строки, профиль, truth)", "engine_month.py", "finish"),
    ("queue: list.sort", "~", "<method 'sort' of 'list' objects>"),
    ("queue: sorted", "~", "<built-in method builtins.sorted>"),
    ("_action_order (ключ сортировки)", "engine.py", "_action_order"),
    ("keyed_rng / event_rng", "rng.py", None),
    ("Table.from_pylist", "~", "from_pylist"),
    ("ParquetWriter.write_table", "core.py", "write_table"),
    ("ParquetWriter.close", "core.py", "close"),
    ("_file_sha256 (части)", "emit.py", "_file_sha256"),
]


def _component_rows(stats: pstats.Stats) -> list[tuple]:

    table = stats.stats  # (file, line, name) -> (cc, nc, tt, ct, callers)

    total = max(ct for (_, _, name), (_, _, _, ct, _) in table.items() if name == "_run_batch")

    rows = []

    for label, file_part, name in COMPONENTS:

        cumulative = 0.0
        count = 0

        for (filename, _line, function), (cc, nc, tt, ct, _callers) in table.items():

            if file_part != "~" and not filename.endswith(file_part):
                continue
            if name is None:
                # Модуль целиком: сумма собственного времени его функций.
                cumulative += tt
                count += nc
                continue
            if file_part == "~":
                if name not in function:
                    continue
            elif function != name:
                continue

            cumulative += ct
            count += nc

        rows.append((label, cumulative, 100.0 * cumulative / total if total else 0.0, count,
                     1e6 * cumulative / count if count else 0.0))

    return rows


def command_profile(args) -> None:

    from src.generator import emit

    job, out = _prepare_batch(args)

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    emit._run_batch(job)
    profiler.disable()
    wall = time.perf_counter() - start

    stats = pstats.Stats(profiler)
    stats.dump_stats(str(out / f"batch{args.batch}.prof"))

    print(f"пачка {args.batch}: {len(job[2])} сообществ, {wall:.1f} с под cProfile")
    print(f"{'компонент':40} {'cumtime, с':>11} {'%':>6} {'вызовов':>11} {'мкс/вызов':>10}")
    for label, cumulative, share, count, average in _component_rows(stats):
        print(f"{label:40} {cumulative:11.2f} {share:6.1f} {count:11d} {average:10.1f}")

    print()
    print("Собственное время, топ-25 функций:")
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("tottime").print_stats(25)
    print("\n".join(line for line in buffer.getvalue().splitlines()[6:] if line.strip()))

    print()
    print("Что вызывает _plan_day (cumtime):")
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumtime").print_callees("_plan_day")
    print("\n".join(buffer.getvalue().splitlines()[:60]))


def command_stats(args) -> None:

    from src.generator import config, emit, engine
    from src.generator.behaviour import engagement
    from src.generator.simulate import CommunitySimulation

    job, out = _prepare_batch(args)

    counters: dict[str, float] = defaultdict(float)
    day_actions: dict[tuple, int] = defaultdict(int)
    empty_days = 0
    plan_days = 0
    before_start = 0
    tails: list[int] = []
    queue_lengths: list[int] = []
    in_burn_in = [False]
    burn_steps: list[int] = []
    kinds: dict[str, int] = defaultdict(int)
    kind_time: dict[str, float] = defaultdict(float)

    original_plan = engine._plan_day
    original_burn = engagement.burn_in
    original_step = engagement._step
    original_execute = engine._execute

    def plan(sim, state, day):
        nonlocal empty_days, plan_days, before_start
        start = time.perf_counter()
        actions = original_plan(sim, state, day)
        counters["plan_day_s"] += time.perf_counter() - start
        plan_days += 1
        if day < state.persona.relationship_start.replace(hour=0, minute=0, second=0, microsecond=0):
            before_start += 1
        if not actions:
            empty_days += 1
        day_actions[(id(sim), day)] += len(actions)
        for action in actions:
            kinds[action.kind] += 1
        return actions

    def burn(state, ties, obligated):
        in_burn_in[0] = True
        steps_before = counters["steps"]
        start = time.perf_counter()
        try:
            return original_burn(state, ties, obligated)
        finally:
            counters["burn_in_s"] += time.perf_counter() - start
            in_burn_in[0] = False
            burn_steps.append(int(counters["steps"] - steps_before))

    def step(state, day, *rest):
        counters["steps"] += 1
        if in_burn_in[0]:
            start = time.perf_counter()
            try:
                return original_step(state, day, *rest)
            finally:
                counters["burn_step_s"] += time.perf_counter() - start
        return original_step(state, day, *rest)

    executed: dict[int, list] = {}

    def execute(sim, action):
        mark = executed.get(id(sim))
        if mark is None or mark[0] is not sim.queue:
            if mark is not None:
                queue_lengths.append(len(mark[0]))
            mark = [sim.queue, 0]
            executed[id(sim)] = mark
        mark[1] += 1
        start = time.perf_counter()
        original_execute(sim, action)
        kind_time[action.kind] += time.perf_counter() - start
        if sim.queue_changed:
            tails.append(len(sim.queue) - mark[1])

    engine._plan_day = plan
    engagement.burn_in = burn
    engagement._step = step
    engine._execute = execute

    start = time.perf_counter()
    try:
        emit._run_batch(job)
    finally:
        engine._plan_day = original_plan
        engagement.burn_in = original_burn
        engagement._step = original_step
        engine._execute = original_execute
    wall = time.perf_counter() - start

    for mark in executed.values():
        queue_lengths.append(len(mark[0]))

    def quantiles(values):
        if not values:
            return "—"
        ordered = sorted(values)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        return f"mean {sum(ordered) / len(ordered):.1f}  p50 {pick(0.5)}  p90 {pick(0.9)}  p99 {pick(0.99)}  max {ordered[-1]}"

    per_day = list(day_actions.values())
    clients = sum(len(__import__("src.generator.world.communities", fromlist=["x"]).members(c, args.clients)) for c in job[2])
    days = (config.HISTORY_END - config.HISTORY_START).days

    print(f"пачка {args.batch}: клиентов {clients}, дней {days}, {wall:.1f} с (с перехватчиками)")
    print(f"прогрев: клиентов {len(burn_steps)}, шагов в среднем {sum(burn_steps) / max(1, len(burn_steps)):.0f}, "
          f"полных (≥1095) {sum(1 for item in burn_steps if item >= 1095)}, "
          f"время {counters['burn_in_s']:.1f} с, из них _step {counters['burn_step_s']:.1f} с")
    print(f"_plan_day: вызовов {plan_days}, до прихода в банк {before_start}, пустых {empty_days} "
          f"({100 * empty_days / max(1, plan_days):.1f}%), время {counters['plan_day_s']:.1f} с")
    print(f"действий на клиент-день: {sum(kinds.values()) / max(1, plan_days - before_start):.2f}")
    print("действий по видам (план дня):", dict(sorted(kinds.items(), key=lambda item: -item[1])))
    print(f"очередь сообщества за день: {quantiles(queue_lengths)}")
    print(f"пересортировок хвоста: {len(tails)} ({len(tails) / max(1, len(queue_lengths)):.1f} на сообщество-день), "
          f"длина хвоста: {quantiles(tails)}")
    print("время исполнения по видам, с:", {key: round(value, 1) for key, value in sorted(kind_time.items(), key=lambda item: -item[1])})


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(item):
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--clients", type=int, default=1000)
        item.add_argument("--seed", type=int, default=11)
        item.add_argument("--start", default="2024-01-01")
        item.add_argument("--end", default="2026-01-01")
        item.add_argument("--registration", default=None)
        item.add_argument("--chunk", type=int, default=256)

    run = sub.add_parser("run")
    common(run)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--label", default="")
    run.add_argument("--hash", action="store_true", help="sha256 файлов и логический хеш строк")

    for name in ("profile", "stats"):
        item = sub.add_parser(name)
        common(item)
        item.add_argument("--batch", type=int, default=0)

    args = parser.parse_args()

    {"run": command_run, "profile": command_profile, "stats": command_stats}[args.command](args)


if __name__ == "__main__":
    main()
