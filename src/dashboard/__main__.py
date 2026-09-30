from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ============================================================
# ЗАПУСК ДАШБОРДА
# ============================================================
#
#   python -m src.dashboard [--run каталог прогона] [--port 8501]
#
# Отдельный процесс streamlit run со страницей src/dashboard/app.py.
# Слежение за исходниками выключено, статистика использования в
# Streamlit не уходит.
# ============================================================


APP = Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(prog="python -m src.dashboard")
    parser.add_argument("--run", default=None, help="каталог прогона, например data/runs/w4-b0")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)

    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError:
        print("[dashboard] нет streamlit: установите командой pip install -e .[dashboard]")
        return 2

    command = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
    ]

    if args.run is not None:
        command += ["--", "--run", args.run]

    print(f"[dashboard] http://localhost:{args.port}")

    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
