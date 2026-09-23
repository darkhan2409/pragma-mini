from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR, RAW_DIR, TIMEZONE, TIMEZONE_NAME


# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что влияет на результат препроцессинга и может меняться
# между запусками, живёт здесь: часовой пояс контракта, окна
# групп, пороги. Значения по умолчанию — договорённости плана;
# JSON-файл переопределяет отдельные поля.
#
# Состояния между запусками у препроцессинга нет: смена
# настройки действует со следующего запуска этапа, а прежние
# результаты просто перезаписываются.
# ============================================================


# Пути стандартны и в командах не задаются.
#
#   data/01_raw/<group>/            выгрузка генератора
#   data/02_preprocessed/<group>/   очищенная лента, один файл
#
# Анкета остаётся в выгрузке: препроцессингу в ней нечего менять.
# Отчётов рядом с данными нет: что случилось на этапе, говорит
# сама команда.
PREPROCESSED_DIR = DATA_DIR / "02_preprocessed"

# Имена групп совпадают с именами подкаталогов RAW.
GROUPS: tuple[str, ...] = ("train", "val", "test")

GROUP_ALIASES: dict[str, str] = {"validation": "val", "valid": "val", "testing": "test"}

# ------------------------------------------------------------
# КАЛЕНДАРЬ
# ------------------------------------------------------------
#
# Отдельный временной канал модели: час суток, день недели и день
# месяца события на единичной окружности, шесть чисел на событие.
#
# Правила лежат здесь, а не в коде модели: смена пояса, длины
# цикла или порядка колонок требует пересборки данных.
#
# Считается календарь по МЕСТНОМУ времени банка. event_time
# хранится и сравнивается в UTC, но суточный и месячный циклы
# принадлежат человеку: полночь клиента это полночь в Алматы, и
# зарплата приходит в своё местное число месяца. Перевод пояса
# делается только ради этих шести чисел и нигде не сохраняется.
#
# Календарных полей нет ни в RAW, ни в canonical: значения
# считаются из event_time при подготовке входа модели. В словари,
# BPE, бакеты, маски и предсказываемые поля они не входят — это
# числовой канал, а не токены.
# ------------------------------------------------------------

# Смещение банковского пояса от UTC в часах. Берётся из
# контракта генератора: им же подписано время в выгрузке, и
# расходиться этим двум числам нельзя.
TIMEZONE_HOURS: float = TIMEZONE.utcoffset(None).total_seconds() / 3600.0

CALENDAR_VERSION = "1.0.0"

CALENDAR_ENCODING: dict = {
    "version": CALENDAR_VERSION,
    "source": "event_time",
    "channel": "отдельный числовой вход модели, не поле события",
    "cycles": {"hour_of_day": 24, "day_of_week": 7, "day_of_month": 31},
    "week_starts_on": "monday",
    "day_of_month_base": 0,
    "hour_is_fractional": True,
    "features": (
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_month_sin",
        "day_of_month_cos",
    ),
    "excluded_from": ("key_vocab", "value_vocab", "bpe", "buckets", "masker", "prediction_targets"),
}


def _dt(value: str | datetime) -> datetime:
    """
    Момент конфигурации в UTC.

    Границы окон объявляются в местном времени банка
    (полночь первого числа), а сравниваются с нормализованным
    временем события. Смешивать наивное и осознанное время
    нельзя, поэтому приведение стоит в одном месте.
    """

    moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)

    stamped = moment.replace(tzinfo=TIMEZONE) if moment.tzinfo is None else moment

    return stamped.astimezone(timezone.utc)


@dataclass(frozen=True)
class GroupWindow:
    """
    Окно группы: начало доступной истории включительно, конечный
    cutoff исключительно, период будущих целей MLM [start, end).
    """

    history_start: datetime
    final_cutoff: datetime
    target_start: datetime
    target_end: datetime

    def as_dict(self) -> dict:
        return {
            "history_start": self.history_start.isoformat(),
            "final_cutoff": self.final_cutoff.isoformat(),
            "target_start": self.target_start.isoformat(),
            "target_end": self.target_end.isoformat(),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "GroupWindow":
        return GroupWindow(
            history_start=_dt(data["history_start"]),
            final_cutoff=_dt(data["final_cutoff"]),
            target_start=_dt(data["target_start"]),
            target_end=_dt(data["target_end"]),
        )


def default_windows() -> dict[str, GroupWindow]:
    """
    Окна групп. Границы названы полночью местного времени
    и хранятся в UTC.
    """

    start = _dt(datetime(2024, 1, 1))

    return {
        "train": GroupWindow(start, _dt(datetime(2026, 1, 1)), start, _dt(datetime(2026, 1, 1))),
        "val": GroupWindow(start, _dt(datetime(2026, 5, 1)), _dt(datetime(2026, 1, 1)),
                           _dt(datetime(2026, 5, 1))),
        "test": GroupWindow(start, _dt(datetime(2026, 9, 1)), _dt(datetime(2026, 5, 1)),
                            _dt(datetime(2026, 9, 1))),
    }


@dataclass(frozen=True)
class PreprocessingConfig:

    windows: dict[str, GroupWindow] = field(default_factory=default_windows)

    # Часовой пояс банка одним смещением от UTC в часах. По нему
    # считаются календарные признаки события.
    #
    # Смещение, а не имя зоны: выгрузка подписывает время
    # фиксированным смещением, и база часовых поясов, которая
    # для Asia/Almaty до марта 2024 года даёт UTC+6, развела бы
    # эти два времени на час.
    timezone_hours: float = TIMEZONE_HOURS

    # Сколько клиентов кладётся в один row group ленты.
    batch_clients: int = 64

    # --------------------------------------------------------

    def bank_timezone(self) -> timezone:
        """
        Пояс банка для расчёта календаря.
        """

        return timezone(timedelta(hours=self.timezone_hours), TIMEZONE_NAME)

    # --------------------------------------------------------
    # СЕКЦИИ
    # --------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "windows": {name: window.as_dict() for name, window in sorted(self.windows.items())},
            "timezone_hours": self.timezone_hours,
            "calendar": {
                **{key: value for key, value in CALENDAR_ENCODING.items() if key != "features"},
                "features": list(CALENDAR_ENCODING["features"]),
                "excluded_from": list(CALENDAR_ENCODING["excluded_from"]),
            },
            "batch_clients": self.batch_clients,
        }

    # --------------------------------------------------------
    # ЗАГРУЗКА
    # --------------------------------------------------------

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PreprocessingConfig":

        base = PreprocessingConfig()

        unknown = set(data) - set(base.as_dict())
        if unknown:
            raise ValueError(f"неизвестные ключи конфига: {sorted(unknown)}")

        windows = dict(base.windows)
        for name, window in data.get("windows", {}).items():
            windows[normalize_group(name)] = GroupWindow.from_dict(window)

        return replace(
            base,
            windows=windows,
            timezone_hours=float(data.get("timezone_hours", base.timezone_hours)),
            batch_clients=int(data.get("batch_clients", base.batch_clients)),
        )

    @staticmethod
    def load(path: Path | None) -> "PreprocessingConfig":

        if path is None:
            return PreprocessingConfig()

        data = json.loads(Path(path).read_text(encoding="utf-8"))

        return PreprocessingConfig.from_dict(data)


def normalize_group(name: str) -> str:

    key = GROUP_ALIASES.get(name.lower(), name.lower())

    if key not in GROUPS:
        raise ValueError(f"группа должна быть одной из {GROUPS}, получено {name!r}")

    return key


def group_dir(group: str) -> Path:
    """
    Каталог очищенной группы: data/02_preprocessed/<group>.
    """

    return PREPROCESSED_DIR / normalize_group(group)


def raw_group_dir(group: str) -> Path:
    """
    Каталог выгрузки группы: data/01_raw/<group>.
    """

    return RAW_DIR / normalize_group(group)

