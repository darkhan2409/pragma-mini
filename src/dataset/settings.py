from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.generator.config import DATA_DIR
from src.preprocessing.artifacts import dumps_json, sha256_bytes

from .version import SCHEMA_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Всё, что решает человек, а не данные, живёт здесь: какие
# группы собирать, на каких срезах, сколько истории брать в один
# пример и какие старые события считаются важными.
#
# Отпечаток конфигурации входит в dataset_id, поэтому смена
# любого решения даёт другой набор явно, а не молча поверх
# прежнего.
#
# Здесь нет ни одного значения, посчитанного по данным. Бюджеты
# контекста выбирает человек, посмотрев на отчёт измерения, и
# подбирать их по validation или test запрещено.
# ============================================================


DATASETS_DIR = DATA_DIR / "datasets"

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

    schema_version: int = SCHEMA_VERSION

    groups: tuple[str, ...] = GROUPS

    # Дополнительные срезы по группам.
    #
    # СЕЙЧАС ОНИ ЗАПРЕЩЕНЫ, и поле оставлено только затем, чтобы
    # отказ был внятным. Анкета клиента одна — итоговая, на
    # границу выгрузки. На конечном срезе группы это честно: срез
    # и есть граница. На любом более раннем срезе та же анкета
    # была бы знанием из будущего, и пример перестал бы быть
    # честным молча.
    #
    # Вернуть ранние срезы можно, когда профиль снова научится
    # отвечать на вопрос «что банк знал к этой дате».
    extra_cutoffs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    context: ContextPolicy = field(default_factory=ContextPolicy)

    # Сколько примеров лежит в одном файле и в одной группе строк.
    shard_samples: int = 512
    row_group_samples: int = 32

    # Сколько читаемых примеров положить рядом с набором.
    golden_limit: int = 6

    def validate(self) -> None:

        if not self.groups:
            raise ConfigError("собирать нечего: список групп пуст")

        unknown = [group for group in self.groups if group not in GROUPS]

        if unknown:
            raise ConfigError(f"неизвестные группы: {unknown}; известны {list(GROUPS)}")

        if len(set(self.groups)) != len(self.groups):
            raise ConfigError("группа названа дважды")

        declared = sorted(
            group for group, moments in self.extra_cutoffs.items() if moments
        )

        if declared:
            raise ConfigError(
                "дополнительные срезы запрещены (объявлены у групп "
                + ", ".join(declared)
                + "): профиль это одна итоговая строка на границу выгрузки, и на более раннем "
                "срезе он был бы знанием из будущего. Пример строится только на конечном "
                "срезе группы"
            )

        for group, moments in sorted(self.extra_cutoffs.items()):

            if group not in self.groups:
                raise ConfigError(f"срезы объявлены для группы {group!r}, которую не собираем")

            parsed = [_moment(group, item) for item in moments]

            if len(set(parsed)) != len(parsed):
                raise ConfigError(f"группа {group}: один и тот же дополнительный срез назван дважды")

        if self.shard_samples < 1:
            raise ConfigError("в одном файле обязан лежать хотя бы один пример")

        if self.row_group_samples < 1:
            raise ConfigError("в одной группе строк обязан лежать хотя бы один пример")

        if self.row_group_samples > self.shard_samples:
            raise ConfigError("группа строк не может быть больше файла")

        if self.golden_limit < 0:
            raise ConfigError("число читаемых примеров не бывает отрицательным")

        self.context.validate()

    def cutoffs_for(self, group: str, final_cutoff: datetime) -> list[datetime]:
        """
        Срезы группы: её конечный момент и объявленные явно.

        Порядок хронологический и от порядка записи в конфиге не
        зависит: вес примера считается по их числу, и перестановка
        строк в файле не должна менять ничего.
        """

        moments = {final_cutoff}

        for item in self.extra_cutoffs.get(group, ()):
            moments.add(_moment(group, item))

        return sorted(moments)

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "groups": list(self.groups),
            "extra_cutoffs": {
                group: list(moments) for group, moments in sorted(self.extra_cutoffs.items())
            },
            "context": self.context.as_dict(),
            "shard_samples": self.shard_samples,
            "row_group_samples": self.row_group_samples,
            "golden_limit": self.golden_limit,
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
            groups=tuple(str(item) for item in data.get("groups", base.groups)),
            extra_cutoffs={
                str(group): tuple(str(item) for item in moments)
                for group, moments in data.get("extra_cutoffs", {}).items()
            },
            context=(
                ContextPolicy.from_dict(data["context"]) if "context" in data else base.context
            ),
            shard_samples=int(data.get("shard_samples", base.shard_samples)),
            row_group_samples=int(data.get("row_group_samples", base.row_group_samples)),
            golden_limit=int(data.get("golden_limit", base.golden_limit)),
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


def _moment(group: str, value: str) -> datetime:

    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise ConfigError(f"группа {group}: срез {value!r} не читается как момент времени") from error


def datasets_dir(name: str) -> Path:
    return DATASETS_DIR / name


__all__ = [
    "DATASETS_DIR",
    "DEFAULT_MILESTONES",
    "GROUPS",
    "POLICIES",
    "POLICY_ALL",
    "POLICY_RECENT_PLUS_MILESTONES",
    "ConfigError",
    "ContextPolicy",
    "DatasetConfig",
    "datasets_dir",
]
