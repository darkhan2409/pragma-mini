from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


# Проект лежит в корне репозитория PRAGMA, но от её кода не зависит:
# отсюда читается только выгрузка генератора data/01_raw/<group>/ и
# продолжение групп для меток (FUTURE_DIR).
PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
RAW_DIR = REPO / "data" / "01_raw"

DATA_DIR = PROJECT / "data"

# Продолжение группы после конца её выгрузки — те же клиенты,
# прожитые дальше (python -m src.generator.continuation). Только для
# меток: признаки и популяция — из выгрузки. Лежит здесь, вне data/
# репозитория, — этапы PRAGMA его не видят.
FUTURE_DIR = DATA_DIR / "future"
MODELS_DIR = PROJECT / "models"
REPORTS_DIR = PROJECT / "reports"

# Группы клиентов — ровно группы выгрузки PRAGMA. Своего разбиения
# проект не делает: клиент train PRAGMA остаётся train здесь.
GROUPS: tuple[str, ...] = ("train", "val", "test")

# Пока идут эксперименты, train учит, а val оценивает; test
# строится и оценивается только в финальной оценке (--final-test).
FINAL_GROUP = "test"

# Окно наблюдения target: (T, T + HORIZON].
HORIZON = timedelta(days=60)

# Группы, у которых T — конец выгрузки, а окно target лежит в
# продолжении. Конец выгрузки train — конец окна, на котором учился
# backbone PRAGMA: с T раньше него окно target train попало бы в
# историю предобучения.
FUTURE_LABEL_GROUPS: tuple[str, ...] = ("train",)

# Задача churn_active90: действие клиента в [T − RECENT, T).
RECENT = timedelta(days=90)

# Все времена RAW записаны со смещением Казахстана. Календарный день
# клиента — местный: сутки считаются от местной полуночи.
LOCAL_OFFSET = timedelta(hours=5)
RAW_OFFSET_SUFFIX = "+05:00"

# Окна агрегатов, в сутках.
WINDOWS: tuple[int, ...] = (7, 30, 60, 90)

SEED = 42


def groups(final_test: bool) -> tuple[str, ...]:
    """
    Группы одного запуска: test — только в финальной оценке.
    """
    return GROUPS if final_test else tuple(group for group in GROUPS if group != FINAL_GROUP)


def manifest(group: str, raw_dir: Path = RAW_DIR) -> dict:
    return json.loads((raw_dir / group / "manifest.json").read_text())


def period_end(group: str, raw_dir: Path = RAW_DIR) -> datetime:
    """
    Конец выгрузки группы: полуоткрытая граница, событий в ней и позже нет.
    """
    return datetime.fromisoformat(manifest(group, raw_dir)["period_end"])


def cutoff(group: str, raw_dir: Path = RAW_DIR) -> datetime:
    """
    T группы, местная полночь, как и конец выгрузки:
    - группа с метками из продолжения — сам конец выгрузки;
    - остальные — самый поздний момент, у которого окно (T, T + HORIZON]
      целиком лежит внутри выгрузки.
    """
    end = period_end(group, raw_dir)
    return end if group in FUTURE_LABEL_GROUPS else end - HORIZON
