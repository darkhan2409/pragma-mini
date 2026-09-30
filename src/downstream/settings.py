from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.generator.config import DATA_DIR


# ============================================================
# ИДЕЯ
# ============================================================
#
# Оценка обученной модели на задачах, ради которых она учится:
# вектор клиента на момент T, строго из прошлого, и простая голова
# над ним — против CatBoost-бейзлайна на тех же клиентах.
#
# Момент T у группы один — конец её выгрузки минус HORIZON_DAYS:
# так окно метки (T, T + HORIZON_DAYS] целиком лежит внутри
# выгрузки. Это ровно T churn_baseline: сравнение идёт на одних и
# тех же клиентах, с одним и тем же моментом и одной меткой.
#
#   data/13_downstream/<тег>/<group>.parquet   векторы на T
#   data/13_downstream/<тег>/meta.json         из чего посчитаны
#   data/13_downstream/<тег>/report.json       пробы и сравнение
# ============================================================


DOWNSTREAM_DIR = DATA_DIR / "13_downstream"

# Окно метки после T, в сутках.
HORIZON_DAYS = 60

# Строки и прогнозы churn-бейзлайна: какие клиенты, на какой T и с
# какой меткой. Бейзлайн их только пишет; здесь они только читаются.
CHURN_REPORTS = DATA_DIR.parent / "churn_baseline" / "reports"

EMBEDDINGS_META = "meta.json"

REPORT_FILE = "report.json"


def cutoff(group: str) -> datetime:
    """
    Момент T группы в UTC: конец выгрузки минус HORIZON_DAYS.

    Конец выгрузки — final_cutoff окна группы, местная полночь.
    Вычитание в UTC сохраняет её: у Asia/Almaty нет перехода часов.
    """

    from src.preprocessing.settings import PreprocessingConfig

    window = PreprocessingConfig.load(None).windows[group]

    return (window.final_cutoff - timedelta(days=HORIZON_DAYS)).astimezone(timezone.utc)


def downstream_dir(tag: str) -> Path:
    """
    Каталог векторов и отчёта одной модели: тег — имя чекпойнта или
    init для начальных весов.
    """

    return DOWNSTREAM_DIR / tag


__all__ = [
    "CHURN_REPORTS",
    "DOWNSTREAM_DIR",
    "EMBEDDINGS_META",
    "HORIZON_DAYS",
    "REPORT_FILE",
    "cutoff",
    "downstream_dir",
]
