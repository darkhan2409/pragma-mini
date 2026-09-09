from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pyarrow as pa

from src.generator.config import (
    ARTIFACTS_DIR,
    EVENT_TYPE_BY_SOURCE,
    EVENT_TYPES,
    PROCESSED_DIR,
    PROFILE_DYNAMIC_FIELDS,
    PROFILE_FIELDS,
    RAW_DIR,
    SOURCES,
)
from src.generator.emit import PROFILE_FIELD_TYPES, SCHEMAS, schemas_for
from src.generator.version import RAW_SCHEMA_REVISION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Единственное место, где сказано, ЧТО есть каждое поле RAW:
# тип (categorical | numeric | boolean | metadata), роль и
# флаг predictable. Поля различаются парой (namespace, field).
#
# Namespaces:
#   timeline          служебные колонки ленты и event_type
#   <event_type>      поля payload каждого типа события
#   profile           полный профиль из 20 полей (контекст)
#   source_coverage   покрытие, роль coverage, в статистики не идёт
#   labels            метка, роль label, никогда не читается для признаков
#
# Реестр строится из схем генератора, поэтому не может
# разойтись с RAW: тест проверяет, что покрыта каждая колонка.
# ============================================================


# ------------------------------------------------------------
# ВИДЫ И РОЛИ
# ------------------------------------------------------------

KIND_CATEGORICAL = "categorical"
KIND_NUMERIC = "numeric"
KIND_BOOLEAN = "boolean"
KIND_METADATA = "metadata"

ROLE_FEATURE = "feature"
ROLE_METADATA = "metadata"
ROLE_CONTAINER = "container"
ROLE_COVERAGE = "coverage"
ROLE_LABEL = "label"


@dataclass(frozen=True)
class FieldSpec:
    namespace: str
    field: str
    kind: str
    arrow_type: pa.DataType
    predictable: bool = False
    role: str = ROLE_FEATURE
    note: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.namespace, self.field)

    @property
    def is_feature(self) -> bool:
        return self.role == ROLE_FEATURE

    @property
    def is_numeric(self) -> bool:
        return self.kind == KIND_NUMERIC

    @property
    def column(self) -> str:
        """
        Имя колонки в широкой таблице processed.
        """

        return f"{self.namespace}__{self.field}"

    @property
    def bucket_column(self) -> str:
        return f"{self.column}__bucket"


# ------------------------------------------------------------
# ПРАВИЛА ПО УМОЛЧАНИЮ
# ------------------------------------------------------------

# Поля с непрерывным или широким числовым смыслом.
NUMERIC_FIELDS = frozenset(
    {
        "amount",
        "amount_or_limit",
        "term",
        "age",
        "declared_income",
        "relationship_months",
        "contracts_count",
        "active_contracts",
        "credit_limit",
        "credit_utilization",
    }
)

# Служебные поля: хранятся, не bucketize, не предсказываются.
METADATA_FIELDS = frozenset({"client_id", "ts", "seq", "session_id", "snapshot_month"})

# Содержательные поля событий, которые не являются целью:
# производные от ts и атрибут качества данных.
NOT_PREDICTABLE_EVENT_FIELDS = frozenset({"timestamp_quality", "day_of_week", "hour"})

METADATA_NOTES = {
    "client_id": "идентификатор клиента: ключ, не признак",
    "ts": "служебная метка времени: порядок и окна, не признак",
    "seq": "порядковый номер в ленте: tie-break, информации не несёт",
    "session_id": "идентификатор сессии: хранится, не bucketize, не предсказывается",
    "snapshot_month": "месяц снимка профиля: ключ as-of",
    "payload": "контейнер JSON, разбирается в типизированные колонки",
}

# Имена скрытых полей генератора: их не должно быть ни среди
# колонок RAW, ни среди ключей payload.
LATENT_NAMES = frozenset(
    {
        "activity",
        "digital_affinity",
        "mobility",
        "credit_need",
        "risk",
        "volatility",
        "push_reachable",
        "campaign",
        "clicked",
        "outcome",
        "activity_scenario",
        "stress_scenario",
        "credit_stress",
        "browsed_products",
        "clicked_offers",
        "activity_multiplier",
        "spending_multiplier",
        "discretionary_multiplier",
        "utilization_pressure",
        "decline_pressure",
        "baseline_stress",
        "birth_date",
        "relationship_start",
    }
)


# ------------------------------------------------------------
# ПОРЯДОК ПОЛЕЙ PAYLOAD
# ------------------------------------------------------------

SOURCE_BY_EVENT_TYPE: dict[str, str] = {
    event_type: source for source, event_type in EVENT_TYPE_BY_SOURCE.items()
}

EVENT_TYPE_PROFILE = EVENT_TYPE_BY_SOURCE["profile"]


def payload_fields(
    event_type: str, revision: int = RAW_SCHEMA_REVISION
) -> tuple[str, ...]:
    """
    Поля payload в порядке, в котором их пишет генератор.

    Порядок это не украшение: ключи payload обязаны идти в том
    же порядке, что колонки таблицы, и ревизия схемы меняет
    оба списка одновременно.

    Умолчание это последняя ревизия: реестр строится по
    надмножеству полей, чтобы одна сборка preprocessing читала
    любой RAW. Проверки контракта передают ревизию явно.
    """

    if event_type == EVENT_TYPE_PROFILE:
        return tuple(PROFILE_DYNAMIC_FIELDS)

    source = SOURCE_BY_EVENT_TYPE[event_type]

    # Первые две колонки таблицы это client_id и ts.
    return tuple(schemas_for(revision)[source].names[2:])


def payload_arrow_type(
    event_type: str, field_name: str, revision: int = RAW_SCHEMA_REVISION
) -> pa.DataType:

    if event_type == EVENT_TYPE_PROFILE:
        return PROFILE_FIELD_TYPES[field_name]

    return schemas_for(revision)[SOURCE_BY_EVENT_TYPE[event_type]].field(field_name).type


def payload_schema(
    event_type: str, revision: int = RAW_SCHEMA_REVISION
) -> pa.Schema:
    """
    Схема разбора payload.

    Берётся надмножество полей: pyarrow.json с явной схемой
    делает отсутствующий ключ null, а лишний ключ ошибкой.
    Поэтому надмножество читает и старый RAW, и новый, а вот
    сужение схемы сломалось бы на новом.
    """

    return pa.schema(
        [
            (name, payload_arrow_type(event_type, name, revision))
            for name in payload_fields(event_type, revision)
        ]
    )


# ------------------------------------------------------------
# РЕЕСТР
# ------------------------------------------------------------


def _kind(field_name: str, arrow_type: pa.DataType) -> str:

    if field_name in METADATA_FIELDS:
        return KIND_METADATA

    if field_name in NUMERIC_FIELDS:
        return KIND_NUMERIC

    if pa.types.is_boolean(arrow_type):
        return KIND_BOOLEAN

    return KIND_CATEGORICAL


def _metadata(namespace: str, field_name: str, arrow_type: pa.DataType, role: str = ROLE_METADATA, note: str = "") -> FieldSpec:
    return FieldSpec(
        namespace=namespace,
        field=field_name,
        kind=KIND_METADATA,
        arrow_type=arrow_type,
        predictable=False,
        role=role,
        note=note or METADATA_NOTES.get(field_name, ""),
    )


def build_registry() -> dict[tuple[str, str], FieldSpec]:

    specs: list[FieldSpec] = []

    # --------------------------------------------------------
    # ЛЕНТА
    # --------------------------------------------------------

    timeline = SCHEMAS["timeline"]

    specs.append(_metadata("timeline", "client_id", timeline.field("client_id").type))
    specs.append(_metadata("timeline", "ts", timeline.field("ts").type))
    specs.append(_metadata("timeline", "seq", timeline.field("seq").type))
    specs.append(
        FieldSpec(
            namespace="timeline",
            field="event_type",
            kind=KIND_CATEGORICAL,
            arrow_type=timeline.field("event_type").type,
            predictable=True,
            note="тип следующего события: основная цель последовательности",
        )
    )
    specs.append(
        _metadata("timeline", "payload", timeline.field("payload").type, role=ROLE_CONTAINER)
    )

    # --------------------------------------------------------
    # СОБЫТИЯ
    # --------------------------------------------------------

    for event_type in EVENT_TYPES:

        for field_name in payload_fields(event_type):

            arrow_type = payload_arrow_type(event_type, field_name)
            kind = _kind(field_name, arrow_type)

            if kind == KIND_METADATA:
                specs.append(_metadata(event_type, field_name, arrow_type))
                continue

            specs.append(
                FieldSpec(
                    namespace=event_type,
                    field=field_name,
                    kind=kind,
                    arrow_type=arrow_type,
                    predictable=field_name not in NOT_PREDICTABLE_EVENT_FIELDS,
                )
            )

    # --------------------------------------------------------
    # ПОЛНЫЙ ПРОФИЛЬ
    # --------------------------------------------------------

    profile = SCHEMAS["profile"]

    for field_name in ("client_id", "ts", "snapshot_month"):
        specs.append(_metadata("profile", field_name, profile.field(field_name).type))

    for field_name in PROFILE_FIELDS:
        arrow_type = PROFILE_FIELD_TYPES[field_name]
        specs.append(
            FieldSpec(
                namespace="profile",
                field=field_name,
                kind=_kind(field_name, arrow_type),
                arrow_type=arrow_type,
                predictable=False,
                note="контекст as-of, не цель",
            )
        )

    # --------------------------------------------------------
    # СЛУЖЕБНЫЕ ТАБЛИЦЫ
    # --------------------------------------------------------

    for field_name in SCHEMAS["source_coverage"].names:
        specs.append(
            _metadata(
                "source_coverage",
                field_name,
                SCHEMAS["source_coverage"].field(field_name).type,
                role=ROLE_COVERAGE,
                note="покрытие источника: отличает «события не было» от «источник не видел»",
            )
        )

    for field_name in SCHEMAS["labels"].names:
        specs.append(
            _metadata(
                "labels",
                field_name,
                SCHEMAS["labels"].field(field_name).type,
                role=ROLE_LABEL,
                note="downstream-метка: preprocessing её никогда не читает для признаков",
            )
        )

    registry = {spec.key: spec for spec in specs}

    assert len(registry) == len(specs), "дубликат (namespace, field) в реестре"

    return registry


REGISTRY: dict[tuple[str, str], FieldSpec] = build_registry()


def specs_for(namespace: str) -> list[FieldSpec]:
    return [spec for spec in REGISTRY.values() if spec.namespace == namespace]


def feature_specs() -> list[FieldSpec]:
    return [spec for spec in REGISTRY.values() if spec.is_feature]


def numeric_specs() -> list[FieldSpec]:
    return [spec for spec in feature_specs() if spec.is_numeric]


def predictable_specs() -> list[FieldSpec]:
    return [spec for spec in feature_specs() if spec.predictable]


# Namespaces, чьи записи это события ленты.
EVENT_NAMESPACES: tuple[str, ...] = tuple(EVENT_TYPES)

# Все namespaces в порядке вывода.
NAMESPACES: tuple[str, ...] = ("timeline",) + EVENT_NAMESPACES + ("profile", "source_coverage", "labels")

# Таблица RAW, из которой читается namespace (для реестра и проверок).
NAMESPACE_TABLE: dict[str, str] = {
    "timeline": "timeline",
    "profile": "profile",
    "source_coverage": "source_coverage",
    "labels": "labels",
    **{event_type: SOURCE_BY_EVENT_TYPE[event_type] for event_type in EVENT_TYPES},
}

# Источник, чей first_seen определяет начало наблюдения клиента.
OBSERVATION_SOURCE = "transactions"

assert OBSERVATION_SOURCE in SOURCES


# ------------------------------------------------------------
# СПЛИТЫ И ДАТАСЕТЫ
# ------------------------------------------------------------

CLIENT_GROUPS: tuple[str, ...] = ("train", "val", "test")

MONTH_ROLES: tuple[str, ...] = ("train_period", "val_month", "test_month")

# (группа клиентов, роль месяца) -> датасет. Остальные
# сочетания в V1 не используются.
DATASETS: dict[tuple[str, str], str] = {
    ("train", "train_period"): "train",
    ("val", "train_period"): "val_client",
    ("test", "train_period"): "test_client",
    ("train", "val_month"): "val_time",
    ("train", "test_month"): "test_time",
}

DATASET_NAMES: tuple[str, ...] = ("train", "val_client", "test_client", "val_time", "test_time")

FIT_DATASET = "train"

# Причины, по которым кандидат (client, cutoff) не стал примером.
# Порядок это приоритет проверки; все причины монотонны по cutoff.
SKIP_REASONS: tuple[str, ...] = (
    "no_transactions_coverage",
    "insufficient_observation",
    "no_events",
    "no_profile_snapshot",
    "unused_combination",
)


# ------------------------------------------------------------
# НАСТРОЙКИ ЗАПУСКА
# ------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """
    Всё, что влияет на результат и может меняться между
    запусками. Пишется в split_manifest.json.
    """

    split_seed: int = 20240601
    split_shares: tuple[float, float, float] = (0.8, 0.1, 0.1)

    min_observation_days: int = 90

    default_buckets: int = 16
    bucket_overrides: Mapping[tuple[str, str], int] = field(default_factory=dict)

    rare_count_threshold: int = 20

    high_missing: float = 0.90
    near_constant: float = 0.99
    low_entropy_bits: float = 0.1

    # Информационный порог: max_bucket_share / expected_share.
    uneven_buckets: float = 2.0

    def buckets_for(self, spec: FieldSpec) -> int:
        return int(self.bucket_overrides.get(spec.key, self.default_buckets))

    def as_dict(self) -> dict:
        return {
            "split_seed": self.split_seed,
            "split_shares": dict(zip(CLIENT_GROUPS, self.split_shares)),
            "min_observation_days": self.min_observation_days,
            "default_buckets": self.default_buckets,
            "bucket_overrides": {
                f"{namespace}.{field_name}": count
                for (namespace, field_name), count in sorted(self.bucket_overrides.items())
            },
            "rare_count_threshold": self.rare_count_threshold,
            "flag_thresholds": {
                "high_missing": self.high_missing,
                "near_constant": self.near_constant,
                "low_entropy_bits": self.low_entropy_bits,
                "uneven_buckets": self.uneven_buckets,
            },
        }


DEFAULT_SETTINGS = Settings()


# ------------------------------------------------------------
# ПУТИ
# ------------------------------------------------------------

DEFAULT_RAW_DIR = RAW_DIR / "smoke"


def processed_dir(name: str):
    return PROCESSED_DIR / name


def artifacts_dir(name: str):
    return ARTIFACTS_DIR / name


SCHEMA_VERSION = 1
