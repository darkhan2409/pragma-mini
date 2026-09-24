"""
Фиксация и сверка состояния исходников аудита.

    python audit/2026-09-24-gen/state/fingerprint.py            записать
    python audit/2026-09-24-gen/state/fingerprint.py --verify   сверить

Вся логика живёт в harness/sources.py — здесь только команда,
чтобы у обвязки прогона и у этой команды был один и тот же
отпечаток, а не две похожие реализации.

Запись отпечатка — осознанное действие: при расхождении команда
--verify возвращает 1 и НИЧЕГО не перезаписывает.
"""

from __future__ import annotations

import json
import multiprocessing
import platform
import subprocess
import sys
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

sys.path.insert(0, str(AUDIT / "harness"))

from sources import STORED, SourcesChanged, snapshot, state_id, verify  # noqa: E402


def environment() -> dict:

    versions = {}

    for name in ("numpy", "pandas", "pyarrow", "sklearn", "scipy", "tokenizers"):
        module = __import__(name)
        versions[name] = getattr(module, "__version__", "?")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": multiprocessing.cpu_count(),
        "start_methods": multiprocessing.get_all_start_methods(),
        "libraries": versions,
    }


def main() -> int:

    if "--verify" in sys.argv[1:]:

        try:
            identifier = verify("сверка")
        except SourcesChanged as error:
            print(error)
            return 1

        print(f"исходники не менялись, состояние {identifier}")

        return 0

    current = snapshot()

    STORED.write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    identifier = state_id(current)

    (AUDIT / "state" / "environment.json").write_text(
        json.dumps(
            {"code_state": identifier, **environment()}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    (AUDIT / "state" / "code_state.txt").write_text(identifier + "\n", encoding="utf-8")

    print(f"файлов {len(current)}, состояние {identifier} -> {STORED}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
