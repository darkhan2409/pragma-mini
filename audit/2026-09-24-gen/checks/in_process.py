"""
D5: прогоны в одном процессе против свежих процессов.

    python audit/2026-09-24-gen/checks/in_process.py

generate_dataset перед работой ставит три процессных глобала —
горизонт, параметры и состояние RNG (emit.py:329-341), а ключ
кэша state_cache горизонта не содержит (rng.py:57-58, 80) и
activate_horizon кэши не чистит.

Отсюда гипотеза: второй прогон в том же процессе может
переиспользовать значения, посчитанные под прежним окном.

Проверка прямая: в одном процессе выполняются два прогона с
разными окнами, и каждый сравнивается с таким же прогоном,
сделанным свежим процессом.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT / "harness"))

from sources import verify  # noqa: E402

from src.generator import emit  # noqa: E402

CLIENTS = 64
SEED = 100

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str) -> None:
    RESULTS.append({"check": name, "verdict": verdict, "detail": detail})
    print(f"[{verdict}] {name}: {detail}")


def fresh(name: str) -> Path:
    out = (RUNS / name).resolve()
    if RUNS.resolve() not in out.parents:
        raise SystemExit(f"отказ: {out} вне {RUNS}")
    if out.exists():
        shutil.rmtree(out)
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(out: Path) -> dict:
    return {
        name: digest(out / name)
        for name in ("events.parquet", "profile.parquet")
    }


def here(name: str, end: str) -> dict:
    """
    Прогон в ЭТОМ процессе.
    """

    out = fresh(name)

    emit.generate_dataset(
        total_clients=CLIENTS,
        out_dir=out,
        seed=SEED,
        world_seed=42,
        history_start=datetime.fromisoformat("2024-01-01"),
        history_end=datetime.fromisoformat(end),
        workers=1,
        quiet=True,
    )

    return fingerprint(out)


def elsewhere(name: str, end: str) -> dict:
    """
    Прогон в СВЕЖЕМ процессе, через обвязку.
    """

    fresh(name)

    subprocess.run(
        [
            sys.executable,
            str(AUDIT / "harness" / "run_gen.py"),
            "--name", name,
            "--clients", str(CLIENTS),
            "--seed", str(SEED),
            "--start", "2024-01-01",
            "--end", end,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    return fingerprint(RUNS / name)


def main() -> int:

    code_state = verify("до D5")

    # Свежие процессы — эталон.
    clean_short = elsewhere("d5-clean-short", "2026-01-01")
    clean_long = elsewhere("d5-clean-long", "2026-09-01")

    record("эталоны", "СПРАВКА",
           "два прогона свежими процессами: окно до 2026-01-01 и до 2026-09-01")

    # Тот же порядок, но в одном процессе.
    same_first = here("d5-same-first", "2026-01-01")
    same_second = here("d5-same-second", "2026-09-01")

    first_ok = same_first == clean_short
    second_ok = same_second == clean_long

    record(
        "D5 первый прогон в процессе",
        "PASS" if first_ok else "FAIL",
        "совпал со свежим процессом" if first_ok else "отличается от свежего процесса",
    )

    record(
        "D5 второй прогон в том же процессе",
        "PASS" if second_ok else "FAIL",
        "совпал со свежим процессом"
        if second_ok
        else "ОТЛИЧАЕТСЯ от свежего процесса: состояние первого прогона повлияло на второй",
    )

    # Обратный порядок: длинное окно первым.
    back_first = here("d5-back-first", "2026-09-01")
    back_second = here("d5-back-second", "2026-01-01")

    back_first_ok = back_first == clean_long
    back_second_ok = back_second == clean_short

    record(
        "D5 обратный порядок, первый",
        "PASS" if back_first_ok else "FAIL",
        "совпал со свежим процессом" if back_first_ok else "отличается",
    )

    record(
        "D5 обратный порядок, второй",
        "PASS" if back_second_ok else "FAIL",
        "совпал со свежим процессом"
        if back_second_ok
        else "ОТЛИЧАЕТСЯ: порядок прогонов в процессе влияет на результат",
    )

    if verify("после D5") != code_state:
        record("состояние кода", "FAIL", "исходники изменились во время проверки")

    destination = AUDIT / "evidence" / "in-process.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
