from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR



# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что решает человек, а не данные, живёт здесь: сколько
# истории берётся в один пример и какие старые события
# считаются важными.
#
# Здесь нет ни одного значения, посчитанного по данным.
# Бюджеты контекста выбирает человек, и подбирать их по
# validation или test запрещено.
# ============================================================


# Один каталог на группу и один файл в нём.
#
#   data/05_dataset/<group>/samples.parquet
DATASET_DIR = DATA_DIR / "05_dataset"

SAMPLES_FILE = "samples.parquet"
META_FILE = "meta.json"

# 1 — анкета примера описывает конец выгрузки (прежний формат);
# 2 — анкета описывает начало периода целей группы;
# 3 — анкета на cutoff событий (state_at_event_cutoff), возраст и
#     признак пенсионера посчитаны от даты рождения;
# 4 — анкета это Attributes на cutoff и вехи Lifelong раньше
#     него, с временем каждого токена (profile_time);
# 5 — признака пенсионера среди Attributes нет; возраст — только
#     от birth_date на cutoff и точным числом полных лет;
# 6 — стаж на месте работы в Attributes, новые вехи Lifelong;
#     событие-источник вехи целью не бывает;
# 7 — событие-источник вехи находит ссылка вехи (source_id), а не
#     совпадение времени и типа; стаж только при наёмном виде
#     дохода на cutoff;
# 8 — в meta окна группы (контекст и маскирование), в lineage —
#     окна всех групп.
DATASET_FORMAT = 8

GROUPS: tuple[str, ...] = ("train", "val", "test")

# Политика отбора истории в один пример.
#
#   all     вся видимая история;
#   recent  последние max_events событий, в исходном порядке.
POLICY_ALL = "all"
POLICY_RECENT = "recent"

POLICIES: tuple[str, ...] = (POLICY_ALL, POLICY_RECENT)

# Сколько последних событий попадает в пример по умолчанию.
#
# Предел нужен шагу обучения: память forward+backward растёт
# линейно с длиной истории. Самый длинный клиент train — 20 300
# событий, 121 623 токена — требует 3,49 ГиБ, больше свободных
# 3,2 ГиБ у RTX 3050 под WSL, и уходит в системную память. При
# 12 000 самый тяжёлый шаг train занимает 2,53 ГиБ (flash-attn,
# bf16). Долгосрочные факты о клиенте при этом не теряются: они
# лежат в Lifelong анкеты, а не в ленте.
MAX_EVENTS = 12000


class ConfigError(ValueError):
    """
    Конфигурация датасета невозможна.
    """


@dataclass(frozen=True)
class ContextPolicy:
    """
    Сколько истории попадает в один пример.
    """

    policy: str = POLICY_RECENT

    # Предел числа событий. У recent он обязателен; у all это
    # объявленная граница: история длиннее неё — ошибка, а не
    # повод обрезать молча.
    max_events: int | None = MAX_EVENTS

    # Предел на одну запись. Он действует ВСЕГДА, даже когда
    # берётся вся история: значение, не помещающееся в одно
    # событие, это ошибка настройки, а не повод обрезать.
    max_event_tokens: int = 4096
    max_profile_tokens: int = 4096

    def validate(self) -> None:

        if self.policy not in POLICIES:
            raise ConfigError(f"неизвестная политика контекста {self.policy!r}: известны {list(POLICIES)}")

        if self.max_events is not None and self.max_events < 1:
            raise ConfigError("max_events обязан быть положительным или None")

        if self.policy == POLICY_RECENT and self.max_events is None:
            raise ConfigError("политика recent без max_events отбирать нечего")

        for name in ("max_event_tokens", "max_profile_tokens"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} обязан быть положительным")

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "max_events": self.max_events,
            "max_event_tokens": self.max_event_tokens,
            "max_profile_tokens": self.max_profile_tokens,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ContextPolicy":

        base = ContextPolicy()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи политики контекста: {sorted(unknown)}")

        policy = replace(
            base,
            policy=str(data.get("policy", base.policy)),
            max_events=_optional_int(data, "max_events", base.max_events),
            max_event_tokens=int(data.get("max_event_tokens", base.max_event_tokens)),
            max_profile_tokens=int(data.get("max_profile_tokens", base.max_profile_tokens)),
        )

        policy.validate()

        return policy


def _optional_int(data: Mapping[str, Any], name: str, default: int | None) -> int | None:

    if name not in data:
        return default

    value = data[name]

    return None if value is None else int(value)


@dataclass(frozen=True)
class DatasetConfig:
    """
    Решения человека о сборке набора.
    """

    context: ContextPolicy = field(default_factory=ContextPolicy)

    # Сколько примеров лежит в одной группе строк parquet.
    row_group_samples: int = 32

    def validate(self) -> None:

        if self.row_group_samples < 1:
            raise ConfigError("row_group_samples обязан быть положительным")

        self.context.validate()

    def as_dict(self) -> dict:
        return {
            "context": self.context.as_dict(),
            "row_group_samples": self.row_group_samples,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "DatasetConfig":

        base = DatasetConfig()

        unknown = set(data) - set(base.as_dict())

        if unknown:
            raise ConfigError(f"неизвестные ключи конфига датасета: {sorted(unknown)}")

        config = replace(
            base,
            context=(
                ContextPolicy.from_dict(data["context"]) if "context" in data else base.context
            ),
            row_group_samples=int(data.get("row_group_samples", base.row_group_samples)),
        )

        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "DatasetConfig":

        if path is None:
            config = DatasetConfig()
            config.validate()
            return config

        return DatasetConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def dataset_dir(group: str) -> Path:
    """
    Каталог собранной группы.
    """

    return DATASET_DIR / group


__all__ = [
    "DATASET_DIR",
    "GROUPS",
    "MAX_EVENTS",
    "POLICIES",
    "POLICY_ALL",
    "POLICY_RECENT",
    "DATASET_FORMAT",
    "META_FILE",
    "SAMPLES_FILE",
    "ConfigError",
    "ContextPolicy",
    "DatasetConfig",
    "dataset_dir",
]
