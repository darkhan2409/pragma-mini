from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Проект лежит в корне репозитория PRAGMA, но её код не использует:
# признаки строятся только из выгрузки data/01_raw/<group>/. Генератор
# импортирует один replay.py — ради target, не ради признаков.
PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
RAW_DIR = REPO / "data" / "01_raw"

DATA_DIR = PROJECT / "data"
MODELS_DIR = PROJECT / "models"
REPORTS_DIR = PROJECT / "reports"

# Группы клиентов — ровно группы выгрузки PRAGMA.
GROUPS: tuple[str, ...] = ("train", "val", "test")

LOCAL = timezone(timedelta(hours=5))
LOCAL_OFFSET = timedelta(hours=5)
RAW_OFFSET_SUFFIX = "+05:00"

# Строки датасета — операции не раньше этого момента: все окна истории
# до 90 дней у них наблюдаются целиком (выгрузка начинается 2024-01-01).
ELIGIBLE_FROM = datetime(2024, 4, 1, tzinfo=LOCAL)

# В обучение идёт одна отрицательная строка train из NEGATIVE_ONE_IN, с
# весом NEGATIVE_ONE_IN: иначе 5,5 млн строк не помещаются в память.
# Позитивы берутся все; val и test не прореживаются.
NEGATIVE_ONE_IN = 10

SEED = 42


def manifest(group: str, raw_dir: Path = RAW_DIR) -> dict:
    return json.loads((raw_dir / group / "manifest.json").read_text())
