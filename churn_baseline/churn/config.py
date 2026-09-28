from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


# Проект лежит в корне репозитория PRAGMA, но от её кода не зависит:
# отсюда читается только выгрузка генератора data/01_raw/<group>/.
PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
RAW_DIR = REPO / "data" / "01_raw"

DATA_DIR = PROJECT / "data"
MODELS_DIR = PROJECT / "models"
REPORTS_DIR = PROJECT / "reports"

# Группы клиентов — ровно группы выгрузки PRAGMA. Своего разбиения
# проект не делает: клиент train PRAGMA остаётся train здесь.
GROUPS: tuple[str, ...] = ("train", "val", "test")

# Окно наблюдения target: (T, T + HORIZON].
HORIZON = timedelta(days=60)

# Все времена RAW записаны со смещением Казахстана. Календарный день
# клиента — местный: сутки считаются от местной полуночи.
LOCAL_OFFSET = timedelta(hours=5)
RAW_OFFSET_SUFFIX = "+05:00"

# Окна агрегатов, в сутках.
WINDOWS: tuple[int, ...] = (7, 30, 60, 90)

SEED = 42


def manifest(group: str, raw_dir: Path = RAW_DIR) -> dict:
    return json.loads((raw_dir / group / "manifest.json").read_text())


def period_end(group: str, raw_dir: Path = RAW_DIR) -> datetime:
    """
    Конец выгрузки группы: полуоткрытая граница, событий в ней и позже нет.
    """
    return datetime.fromisoformat(manifest(group, raw_dir)["period_end"])


def cutoff(group: str, raw_dir: Path = RAW_DIR) -> datetime:
    """
    T группы: самый поздний момент, у которого окно (T, T + HORIZON]
    целиком лежит внутри выгрузки. T — местная полночь, как и конец
    выгрузки.
    """
    return period_end(group, raw_dir) - HORIZON
