from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR

from .artifacts import dumps_json, sha256_bytes


# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что влияет на результат препроцессинга и может меняться
# между запусками, живёт здесь: часовой пояс контракта, окна
# групп, пороги. Значения по умолчанию — договорённости плана;
# JSON-файл переопределяет отдельные поля.
#
# Отпечаток конфига входит в отпечаток каждого этапа, поэтому
# смена настройки пересчитывает затронутые этапы, а не оставляет
# старые результаты незаметно.
# ============================================================


PROCESSED_DIR = DATA_DIR / "processed"

# Имена групп совпадают с именами подкаталогов RAW.
GROUPS: tuple[str, ...] = ("train", "val", "test")

GROUP_ALIASES: dict[str, str] = {"validation": "val", "valid": "val", "testing": "test"}

# Источники, которые обязаны быть доступны с начала согласованного
# горизонта: без них истории 2023 года нет.
BASE_SOURCES: tuple[str, ...] = (
    "transactions",
    "product_events",
    "loans",
    "applications",
    "profile",
    "app_operations",
)

SCHEMA_VERSION = 1

# ------------------------------------------------------------
# КАЛЕНДАРЬ
# ------------------------------------------------------------
#
# Отдельный временной канал модели: час суток, день недели и день
# месяца события на единичной окружности, шесть чисел на событие.
#
# Правила лежат здесь, а не в коде модели, потому что входят в
# отпечаток этапа и в манифест: смена пояса, длины цикла или
# порядка колонок требует пересборки данных.
#
# Календарных полей нет ни в RAW, ни в canonical: значения
# считаются из event_time при подготовке входа модели. В словари,
# BPE, бакеты, маски и предсказываемые поля они не входят — это
# числовой канал, а не токены.
# ------------------------------------------------------------

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
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


@dataclass(frozen=True)
class AmbiguousInterval:
    """
    Отрезок местного времени, который прожит дважды: часы
    переводили назад. Такие значения помечаются, а не чинятся.
    """

    start: datetime
    end: datetime
    reason: str

    def as_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "reason": self.reason}

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AmbiguousInterval":
        return AmbiguousInterval(_dt(data["start"]), _dt(data["end"]), str(data["reason"]))


def default_ambiguous_intervals() -> tuple[AmbiguousInterval, ...]:
    """
    Казахстан перешёл с UTC+6 на UTC+5 1 марта 2024 года: часы
    перевели назад в полночь, и час перед ней случился дважды.
    """

    return (
        AmbiguousInterval(datetime(2024, 2, 29, 23, 0), datetime(2024, 3, 1, 0, 0), "kz_utc6_to_utc5"),
    )


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

    start = datetime(2024, 1, 1)

    return {
        "train": GroupWindow(start, datetime(2026, 1, 1), start, datetime(2026, 1, 1)),
        "val": GroupWindow(start, datetime(2026, 5, 1), datetime(2026, 1, 1), datetime(2026, 5, 1)),
        "test": GroupWindow(start, datetime(2026, 9, 1), datetime(2026, 5, 1), datetime(2026, 9, 1)),
    }


@dataclass(frozen=True)
class PreprocessingConfig:

    # Часовой пояс контракта RAW: значения остаются наивными, как
    # в выгрузке; пояс — объявленная метаданная.
    timezone: str = "Asia/Almaty"

    # Согласованное начало истории для всех групп.
    required_history_start: datetime = datetime(2024, 1, 1)

    windows: dict[str, GroupWindow] = field(default_factory=default_windows)

    # Объявленные неоднозначные отрезки местного времени.
    ambiguous_local_intervals: tuple[AmbiguousInterval, ...] = field(
        default_factory=default_ambiguous_intervals
    )

    # Сколько клиентов кладётся в один row group canonical.
    batch_clients: int = 64

    # Размер выборки для проверки правил чтения истории.
    history_sample_clients: int = 25
    history_report_cutoffs: int = 6

    # Сколько клиентов показывает смысловой слой в диагностике.
    semantic_sample_clients: int = 10

    base_sources: tuple[str, ...] = BASE_SOURCES

    # --------------------------------------------------------
    # СЕКЦИИ
    # --------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "timezone": self.timezone,
            "required_history_start": self.required_history_start.isoformat(),
            "windows": {name: window.as_dict() for name, window in sorted(self.windows.items())},
            "base_sources": list(self.base_sources),
            "ambiguous_local_intervals": [item.as_dict() for item in self.ambiguous_local_intervals],
            "calendar": {
                **{key: value for key, value in CALENDAR_ENCODING.items() if key != "features"},
                "features": list(CALENDAR_ENCODING["features"]),
                "excluded_from": list(CALENDAR_ENCODING["excluded_from"]),
                "timezone": self.timezone,
            },
            "batch_clients": self.batch_clients,
            "history_sample_clients": self.history_sample_clients,
            "history_report_cutoffs": self.history_report_cutoffs,
            "semantic_sample_clients": self.semantic_sample_clients,
        }

    def section(self, stage: str) -> dict:
        """
        Часть конфига, от которой зависит этап. Только она входит
        в его отпечаток: смена настройки чужого этапа ничего не
        пересчитывает.
        """

        whole = self.as_dict()

        sections = {
            "passport": (
                "schema_version",
                "timezone",
                "required_history_start",
                "windows",
                "base_sources",
            ),
            "canonical": (
                "schema_version",
                "timezone",
                "ambiguous_local_intervals",
                "batch_clients",
                # Правила календаря описывают границу модели, а её
                # реестр пишет этот этап: смена правил обязана
                # пересобрать слой.
                "calendar",
            ),
            "history": (
                "schema_version",
                "windows",
                "history_sample_clients",
                "history_report_cutoffs",
            ),
            "corpus": (
                "schema_version",
                "windows",
                "required_history_start",
            ),
            "semantic": (
                "schema_version",
                "timezone",
                "windows",
                "calendar",
                "semantic_sample_clients",
            ),
        }

        if stage not in sections:
            raise KeyError(f"неизвестный этап конфигурации: {stage}")

        return {key: whole[key] for key in sections[stage]}

    def sha256(self) -> str:
        return sha256_bytes(dumps_json(self.as_dict()).encode("utf-8"))

    def section_sha256(self, stage: str) -> str:
        return sha256_bytes(dumps_json(self.section(stage)).encode("utf-8"))

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

        intervals = (
            tuple(AmbiguousInterval.from_dict(item) for item in data["ambiguous_local_intervals"])
            if "ambiguous_local_intervals" in data
            else base.ambiguous_local_intervals
        )

        return replace(
            base,
            timezone=str(data.get("timezone", base.timezone)),
            required_history_start=_dt(data.get("required_history_start", base.required_history_start)),
            windows=windows,
            base_sources=tuple(data.get("base_sources", base.base_sources)),
            ambiguous_local_intervals=intervals,
            batch_clients=int(data.get("batch_clients", base.batch_clients)),
            history_sample_clients=int(data.get("history_sample_clients", base.history_sample_clients)),
            history_report_cutoffs=int(data.get("history_report_cutoffs", base.history_report_cutoffs)),
            semantic_sample_clients=int(data.get("semantic_sample_clients", base.semantic_sample_clients)),
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


def processed_dir(name: str) -> Path:
    return PROCESSED_DIR / name


DEFAULT_CONFIG = PreprocessingConfig()
