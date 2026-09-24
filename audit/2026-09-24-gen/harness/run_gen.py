"""
Безопасный прогон генератора для аудита.

Запускается ОТДЕЛЬНЫМ процессом, чтобы процессные глобалы
генератора (горизонт, параметры, состояние RNG) не переезжали
между прогонами незаметно.

    python audit/2026-09-24-gen/harness/run_gen.py --name p0-pilot \
        --clients 64 --seed 100 --start 2024-01-01 --end 2026-01-01

Каталог вывода всегда внутри runs/: generate_dataset при
resume=False делает shutil.rmtree(out_dir) до всякой проверки
параметров (src/generator/emit.py:342-343), поэтому путь
проверяется абсолютным ДО вызова.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT / "harness"))

from sources import SourcesChanged, verify  # noqa: E402


def peak_memory() -> int | None:
    """
    Пиковый рабочий набор процесса в байтах.

    psutil в проекте нет, ставить зависимость ради измерения
    нельзя, поэтому на Windows берётся PSAPI напрямую.
    """

    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)

    kernel = ctypes.windll.kernel32

    kernel.GetCurrentProcess.restype = wintypes.HANDLE

    handle = kernel.GetCurrentProcess()

    # argtypes обязательны: без них ctypes передаёт псевдо-хендл
    # (-1) как 32-битное число, на 64-битной Windows он обрезается
    # и вызов молча возвращает ноль.
    call = kernel.K32GetProcessMemoryInfo

    call.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    call.restype = wintypes.BOOL

    if not call(handle, ctypes.byref(counters), counters.cb):
        return None

    return int(counters.PeakWorkingSetSize)


def inside_runs(path: Path) -> Path:
    """
    Путь обязан лежать внутри runs/ и никуда больше.
    """

    resolved = path.resolve()

    if RUNS.resolve() not in resolved.parents:
        raise SystemExit(f"отказ: {resolved} вне {RUNS}")

    if resolved == ROOT / "data" or (ROOT / "data").resolve() in resolved.parents:
        raise SystemExit(f"отказ: {resolved} внутри data/")

    return resolved


def main() -> int:

    parser = argparse.ArgumentParser(prog="run_gen")

    parser.add_argument("--name", required=True, help="имя каталога под runs/")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, default=42)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-clients", type=int, default=256)
    parser.add_argument("--community-size", type=int, default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    out = inside_runs(RUNS / args.name)

    # Прогон привязан к состоянию кода: сверка ДО работы. При
    # расхождении прогона не будет вовсе.
    try:
        before_id = verify("до прогона")
    except SourcesChanged as error:
        print(error)
        return 2

    from src.generator import emit

    # Доказательство, какой код исполняется.
    where = Path(emit.__file__).resolve()

    started = time.perf_counter()

    counts = emit.generate_dataset(
        total_clients=args.clients,
        out_dir=out,
        seed=args.seed,
        world_seed=args.world_seed,
        history_start=datetime.fromisoformat(args.start),
        history_end=datetime.fromisoformat(args.end),
        workers=args.workers,
        chunk_clients=args.chunk_clients,
        community_size=args.community_size,
        params_path=args.params,
        resume=args.resume,
        quiet=True,
    )

    seconds = time.perf_counter() - started

    # И сверка ПОСЛЕ: прогон не имеет права быть посчитанным
    # наполовину одним кодом, наполовину другим.
    try:
        after_id = verify("после прогона")
    except SourcesChanged as error:
        print(error)
        return 2

    if after_id != before_id:
        print(f"состояние кода изменилось во время прогона: {before_id} -> {after_id}")
        return 2

    sizes = {
        path.name: path.stat().st_size
        for path in sorted(out.iterdir())
        if path.is_file()
    }

    record = {
        "name": args.name,
        "code_state": before_id,
        "module": str(where),
        "out": str(out),
        "clients": args.clients,
        "seed": args.seed,
        "world_seed": args.world_seed,
        "start": args.start,
        "end": args.end,
        "workers": args.workers,
        "chunk_clients": args.chunk_clients,
        "community_size": args.community_size,
        "params": args.params,
        "resume": args.resume,
        "counts": counts,
        "seconds": round(seconds, 2),
        "peak_bytes": peak_memory(),
        "files": sizes,
    }

    (out / "run_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(record, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
