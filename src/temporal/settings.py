from __future__ import annotations

from pathlib import Path

from src.generator.config import DATA_DIR


# ============================================================
# ИДЕЯ
# ============================================================
#
# Решение у этапа одно — точка отсчёта времени событий (--anchor):
#
#   last_event  от последнего события примера (по умолчанию);
#   cutoff      от cutoff T примера, как у анкеты: давность
#               «последнее событие → T» модели тогда видна.
#
# Выбор — эксперимент волны 4 (audit/2026-09-28-project): после
# решения останется один вариант. Он записан в meta.json этапа, и
# сборка входа на T (src.downstream.at_cutoff) берёт его оттуда.
#
# Формула временной позиции задана форматом, а не вкусом.
# Отбор событий сделал датасет. Группы строк повторяют входные:
# этап переносит пример как есть и добавляет одну колонку, и
# перекладывать строки по своему размеру ему незачем.
# ============================================================


# Один каталог на группу и один файл в нём.
#
#   data/06_temporal/<group>/temporal.parquet
TEMPORAL_DIR = DATA_DIR / "06_temporal"

TEMPORAL_FILE = "temporal.parquet"

META_FILE = "meta.json"

TIME_ANCHORS = ("last_event", "cutoff")

DEFAULT_ANCHOR = "last_event"


def temporal_dir(group: str) -> Path:
    """
    Каталог временных позиций группы.
    """

    return TEMPORAL_DIR / group


__all__ = [
    "DEFAULT_ANCHOR",
    "META_FILE",
    "TEMPORAL_DIR",
    "TEMPORAL_FILE",
    "TIME_ANCHORS",
    "temporal_dir",
]
