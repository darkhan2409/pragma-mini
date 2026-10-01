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
# Момент T у группы один, и это ровно T churn_baseline: сравнение
# идёт на одних и тех же клиентах, с одним и тем же моментом и одной
# меткой.
#
#   train      конец окна группы — конец истории, на которой учился
#              backbone. С T раньше него окно метки train лежало бы
#              в истории предобучения. Метку churn_baseline берёт из
#              продолжения тех же клиентов после конца выгрузки;
#              вход модели на T — вся история, которую модель знает.
#   val, test  конец выгрузки минус HORIZON_DAYS: окно метки
#              (T, T + HORIZON_DAYS] целиком лежит внутри выгрузки.
#
#   data/13_downstream/<тег>/<group>.parquet   векторы на T
#   data/13_downstream/<тег>/meta.json         из чего посчитаны
#   data/13_downstream/<тег>/report.json       пробы и сравнение
#
# Пока идут эксперименты, train учит голову, а val выбирает
# эксперимент. test не считается и не показывается: он нужен один
# раз, в финальной оценке, и включается явно (--final-test).
# ============================================================


DOWNSTREAM_DIR = DATA_DIR / "13_downstream"

# Окно метки после T, в сутках.
HORIZON_DAYS = 60

# Строки и прогнозы churn-бейзлайна: какие клиенты, на какой T и с
# какой меткой. Бейзлайн их только пишет; здесь они только читаются.
CHURN_REPORTS = DATA_DIR.parent / "churn_baseline" / "reports"

# Продолжения групп для меток (src.generator.continuation). Отсюда
# читается только future.json — сверить, из чего бейзлайн взял метку.
CHURN_FUTURE = DATA_DIR.parent / "churn_baseline" / "data" / "future"

# Группы, у которых T — конец окна, а метка — в продолжении.
FUTURE_LABEL_GROUPS = ("train",)

EMBEDDINGS_META = "meta.json"

GROUPS = ("train", "val")

FINAL_GROUPS = ("train", "val", "test")

REPORT_FILE = "report.json"


def cutoff(group: str) -> datetime:
    """
    Момент T группы в UTC: у train — конец окна группы, у остальных —
    конец окна минус HORIZON_DAYS.

    Конец окна — final_cutoff группы, местная полночь. Вычитание в
    UTC сохраняет её: у Asia/Almaty нет перехода часов.
    """

    from src.preprocessing.settings import PreprocessingConfig

    window = PreprocessingConfig.load(None).windows[group]

    if group in FUTURE_LABEL_GROUPS:
        return window.final_cutoff.astimezone(timezone.utc)

    return (window.final_cutoff - timedelta(days=HORIZON_DAYS)).astimezone(timezone.utc)


def groups(final_test: bool) -> tuple[str, ...]:
    """
    Группы оценки: test — только в финальной оценке.
    """

    return FINAL_GROUPS if final_test else GROUPS


def downstream_dir(tag: str) -> Path:
    """
    Каталог векторов и отчёта одной модели: тег — имя прогона или
    чекпойнта.
    """

    return DOWNSTREAM_DIR / tag


__all__ = [
    "CHURN_FUTURE",
    "CHURN_REPORTS",
    "DOWNSTREAM_DIR",
    "EMBEDDINGS_META",
    "FINAL_GROUPS",
    "FUTURE_LABEL_GROUPS",
    "GROUPS",
    "HORIZON_DAYS",
    "REPORT_FILE",
    "cutoff",
    "downstream_dir",
    "groups",
]
