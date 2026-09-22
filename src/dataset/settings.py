from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR
from src.preprocessing.artifacts import dumps_json, sha256_bytes

from .version import SCHEMA_VERSION


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

GROUPS: tuple[str, ...] = ("train", "val", "test")

# Политика отбора истории в один пример.
POLICY_ALL = "all"
POLICY_RECENT_PLUS_MILESTONES = "recent_plus_milestones"

POLICIES: tuple[str, ...] = (POLICY_ALL, POLICY_RECENT_PLUS_MILESTONES)


# Важные старые события в порядке приоритета. Порядок и есть
# решение: при нехватке резерва берутся те, что выше.
#
# Сверху то, что меняет положение клиента необратимо (просрочка,
# мошенничество, блокировка), ниже жизненный цикл продукта, и в
# конце изменение анкеты.
DEFAULT_MILESTONES: tuple[str, ...] = (
    "delinquency_registered",
    "installment_missed",
    "arrears_cleared",
    "fraud_alert",
    "fraud_decision",
    "card_blocked",
    "card_unblocked",
    "loan_disbursement",
    "loan_restructured",
    "loan_closed",
    "early_repayment",
    "application_decision",
    "product_opened",
    "product_closed",
    "product_migrated",
    "account_opened",
    "profile_change",
)


class ConfigError(ValueError):
    """
    Конфигурация датасета невозможна.
    """


@dataclass(frozen=True)
class ContextPolicy:
    """
    Сколько истории попадает в один пример.

    Лимит событий и лимит токенов это РАЗНЫЕ настройки: сто
    коротких экранов приложения и сто кредитных событий стоят
    модели по-разному, и один предел через другой не выражается.
    """

    policy: str = POLICY_ALL

    # None значит «предела нет». У политики all это норма.
    max_events: int | None = None
    max_tokens: int | None = None

    # Предел на одну запись. Он действует ВСЕГДА, даже когда
    # берётся вся история: значение, не помещающееся в одно
    # событие, это ошибка настройки, а не повод обрезать.
    max_event_tokens: int = 4096
    max_profile_tokens: int = 4096

    milestone_event_types: tuple[str, ...] = DEFAULT_MILESTONES

    # Какая доля обоих бюджетов резервируется под важные старые
    # события.
    milestone_share: float = 0.25

    # Отдать недоизрасходованный резерв недавним событиям.
    return_unused_budget: bool = True

    def validate(self) -> None:

        if self.policy not in POLICIES:
            raise ConfigError(f"неизвестная политика контекста {self.policy!r}: известны {list(POLICIES)}")

        for name in ("max_events", "max_tokens"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ConfigError(f"{name} обязан быть положительным или None")

        for name in ("max_event_tokens", "max_profile_tokens"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} обязан быть положительным")

        if not 0.0 <= self.milestone_share < 1.0:
            raise ConfigError("доля резерва вех лежит в [0, 1): весь бюджет под вехи не отдаётся")

        if len(set(self.milestone_event_types)) != len(self.milestone_event_types):
            raise ConfigError("тип события назван вехой дважды: приоритет тогда неоднозначен")

        if self.policy == POLICY_RECENT_PLUS_MILESTONES:

            if self.max_events is None and self.max_tokens is None:
                raise ConfigError(
                    "политика recent_plus_milestones без единого бюджета отбирать нечего: "
                    "задайте max_events, max_tokens или обе"
                )

            if self.milestone_share > 0.0 and not self.milestone_event_types:
                raise ConfigError("резерв под вехи объявлен, а список важных типов пуст")

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "max_events": self.max_events,
            "max_tokens": self.max_tokens,
            "max_event_tokens": self.max_event_tokens,
            "max_profile_tokens": self.max_profile_tokens,
            "milestone_event_types": list(self.milestone_event_types),
            "milestone_share": self.milestone_share,
            "return_unused_budget": self.return_unused_budget,
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
            max_tokens=_optional_int(data, "max_tokens", base.max_tokens),
            max_event_tokens=int(data.get("max_event_tokens", base.max_event_tokens)),
            max_profile_tokens=int(data.get("max_profile_tokens", base.max_profile_tokens)),
            milestone_event_types=tuple(
                str(item) for item in data.get("milestone_event_types", base.milestone_event_types)
            ),
            milestone_share=float(data.get("milestone_share", base.milestone_share)),
            return_unused_budget=bool(data.get("return_unused_budget", base.return_unused_budget)),
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

    schema_version: int = SCHEMA_VERSION

    context: ContextPolicy = field(default_factory=ContextPolicy)

    # Сколько примеров лежит в одной группе строк parquet.
    row_group_samples: int = 32

    def validate(self) -> None:

        if self.row_group_samples < 1:
            raise ConfigError("row_group_samples обязан быть положительным")

        self.context.validate()

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "context": self.context.as_dict(),
            "row_group_samples": self.row_group_samples,
        }

    def sha256(self) -> str:
        return sha256_bytes(dumps_json(self.as_dict()).encode("utf-8"))

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
    "DEFAULT_MILESTONES",
    "GROUPS",
    "POLICIES",
    "POLICY_ALL",
    "POLICY_RECENT_PLUS_MILESTONES",
    "SAMPLES_FILE",
    "ConfigError",
    "ContextPolicy",
    "DatasetConfig",
    "dataset_dir",
]
