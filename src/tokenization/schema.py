from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.artifacts import read_json
from src.preprocessing.canonical.build import REGISTRY_FILE as CANONICAL_REGISTRY_FILE
from src.preprocessing.canonical.build import STAGE as CANONICAL_STAGE
from src.preprocessing.projection import EVENT_TYPE_FIELD, INITIATOR_EVENT_TYPES, INITIATOR_FIELD
from src.preprocessing.semantic.build import REGISTRY_FILE as SEMANTIC_REGISTRY_FILE
from src.preprocessing.semantic.build import STAGE as SEMANTIC_STAGE
from src.preprocessing.semantic.keys import (
    CATEGORICAL,
    DYNAMIC_FIELDS,
    ENVELOPE_KEYS,
    NUMERIC,
    REFERENCE,
    TEXT,
    KeysError,
    key_for,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Токенизатор не переосмысливает поля: смысл ему приносят два
# реестра препроцессинга.
#
#   semantic_registry.json   что значит ключ, какого он вида и в
#                            каких единицах;
#   field_registry.json      какое физическое поле у какого типа
#                            события объявлено.
#
# Второй нужен ровно для одного вопроса: какие ключи у этого
# типа события ОБЪЯВЛЕНЫ. Без него «значения нет» и «поля здесь
# не бывает» выглядели бы одинаково, а это разные вещи: у
# покупки отсутствие причины отказа значит «отказа не было», а у
# экрана приложения причины отказа не предусмотрено вовсе.
# ============================================================


ORIGIN_PAYLOAD = "payload"
ORIGIN_ENVELOPE = "envelope"
ORIGIN_PROFILE = "profile"
ORIGIN_PROFILE_CHANGE = "profile_change"
ORIGIN_CATALOG = "catalog"
ORIGIN_DERIVED = "derived"
ORIGIN_REFERENCE = "reference"

# Ключи профиля наблюдаются один раз на клиента: версия профиля
# на fit_end это один факт, а не сто повторов по числу покупок.
WEIGHT_PER_EVENT = "per_event"
WEIGHT_PER_CLIENT = "per_client"

# Виды значения, которые получают код в словаре. Ссылка кода не
# получает: она метаданная связи.
MODEL_FEATURE_KINDS: tuple[str, ...] = (NUMERIC, CATEGORICAL, TEXT)


class SchemaError(ValueError):
    """
    Реестры смысла не дают собрать контракт входа.
    """


@dataclass(frozen=True)
class KeyInfo:
    key: str
    value_kind: str
    unit: str | None
    temporal: str | None
    derived_from: tuple[str, ...]
    description: str
    physical_fields: tuple[str, ...]
    origin: str
    weight_rule: str

    @property
    def is_model_feature(self) -> bool:
        return self.value_kind in MODEL_FEATURE_KINDS

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "value_kind": self.value_kind,
            "unit": self.unit,
            "temporal": self.temporal,
            "derived_from": list(self.derived_from),
            "description": self.description,
            "physical_fields": list(self.physical_fields),
            "origin": self.origin,
            "weight_rule": self.weight_rule,
        }


def _origin(physical_field: str) -> str:
    """
    Откуда ключ берётся. Определяется по происхождению, которое
    смысловой слой записал сам.
    """

    if physical_field == "envelope":
        return ORIGIN_ENVELOPE

    if physical_field == "derived:local_ref":
        return ORIGIN_REFERENCE

    if physical_field.startswith("derived:"):
        return ORIGIN_DERIVED

    if physical_field.startswith("catalog:"):
        return ORIGIN_CATALOG

    if physical_field.startswith("profile_change["):
        return ORIGIN_PROFILE_CHANGE

    if physical_field.startswith("profile."):
        return ORIGIN_PROFILE

    return ORIGIN_PAYLOAD


class SemanticSchema:
    """
    Реестры смысла глазами токенизатора.
    """

    def __init__(self, semantic_registry: dict, field_registry: dict):

        self.semantic_registry = semantic_registry
        self.field_registry = field_registry

        registry = semantic_registry.get("registry")

        if not registry or not registry.get("keys"):
            raise SchemaError("в semantic_registry.json нет реестра ключей: соберите этап 5 заново")

        self.keys_version = registry["keys_version"]
        self.semantic_version = semantic_registry["semantic_version"]
        self.projection_version = semantic_registry["projection_version"]
        self.stage_version = semantic_registry["stage_version"]
        self.group = semantic_registry.get("group")
        self.cutoff = semantic_registry.get("cutoff")

        self.ambiguous = tuple(
            (tuple(item["keys"]), item["reason"]) for item in registry.get("ambiguous", ())
        )
        self.allowed_sharing = tuple(
            (item["key"], tuple(item["sources"]), item["reason"]) for item in registry.get("allowed_sharing", ())
        )

        self.keys: dict[str, KeyInfo] = {}

        for key, row in sorted(registry["keys"].items()):

            physical = tuple(row.get("physical_fields") or ())

            if not physical:
                raise SchemaError(f"у ключа {key} нет происхождения: реестр собран старой версией этапа 5")

            origin = _origin(physical[0])

            self.keys[key] = KeyInfo(
                key=key,
                value_kind=row["value_kind"],
                unit=row.get("unit"),
                temporal=row.get("temporal"),
                derived_from=tuple(row.get("derived_from") or ()),
                description=row.get("description", ""),
                physical_fields=physical,
                origin=origin,
                weight_rule=WEIGHT_PER_CLIENT if origin == ORIGIN_PROFILE else WEIGHT_PER_EVENT,
            )

        self.declared_payload, self.dynamic_event_types = self._declared(field_registry)

    # --- разбор реестра полей ---

    @staticmethod
    def _declared(field_registry: dict) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
        """
        Какие смысловые ключи объявлены у каждого типа события.

        Читается позитивно: в расчёт идёт только то, что сам
        препроцессинг пометил смысловым полем. Локальные ссылки и
        внутренние поля сюда не попадают — ни те, ни другие
        значением модели не становятся.
        """

        declared: dict[str, set[str]] = {}
        dynamic: set[str] = set()
        problems: list[str] = []

        rows = field_registry.get("fields")

        if not rows:
            raise SchemaError(f"в {CANONICAL_REGISTRY_FILE} нет полей: соберите canonical заново")

        for row in rows:

            if row.get("role") != "payload" or row.get("model_role") != "semantic_field":
                continue

            owner = row["owner"]
            name = row["name"]

            declared.setdefault(owner, set())

            if name in DYNAMIC_FIELDS:
                # Смысл такого поля задаёт значение соседнего
                # field_name, поэтому заранее он не объявлен.
                dynamic.add(owner)
                continue

            try:
                declared[owner].add(key_for(name, row["source"]).key)
            except KeysError as error:
                problems.append(str(error))

        if problems:
            raise SchemaError("; ".join(sorted(set(problems))))

        event_type_key = ENVELOPE_KEYS[EVENT_TYPE_FIELD].key
        initiator_key = ENVELOPE_KEYS[INITIATOR_FIELD].key

        for owner, keys in declared.items():
            keys.add(event_type_key)
            if owner in INITIATOR_EVENT_TYPES:
                keys.add(initiator_key)

        return (
            {owner: tuple(sorted(keys)) for owner, keys in sorted(declared.items())},
            tuple(sorted(dynamic)),
        )

    # --- выборки ---

    def of_kind(self, kind: str) -> tuple[str, ...]:
        return tuple(key for key, info in self.keys.items() if info.value_kind == kind)

    @property
    def model_feature_keys(self) -> tuple[str, ...]:
        return tuple(key for key, info in self.keys.items() if info.is_model_feature)

    @property
    def link_keys(self) -> tuple[str, ...]:
        return self.of_kind(REFERENCE)

    @property
    def numeric_keys(self) -> tuple[str, ...]:
        return self.of_kind(NUMERIC)

    @property
    def categorical_keys(self) -> tuple[str, ...]:
        return self.of_kind(CATEGORICAL)

    @property
    def text_keys(self) -> tuple[str, ...]:
        return self.of_kind(TEXT)

    @property
    def event_types(self) -> tuple[str, ...]:
        return tuple(self.declared_payload)

    def declared(self, event_type: str) -> tuple[str, ...]:
        """
        Ключи, объявленные у этого типа события. Неизвестный тип
        это пустой набор, а не ошибка: такое событие целиком
        попадёт в отчёт как неизвестное.
        """

        return self.declared_payload.get(event_type, ())

    def info(self, key: str) -> KeyInfo:

        info = self.keys.get(key)

        if info is None:
            raise SchemaError(f"ключа {key!r} нет в смысловом реестре")

        return info

    def weight_rule(self, key: str) -> str:
        return self.info(key).weight_rule

    # --- открытие ---

    @staticmethod
    def paths(processed_dir: Path, group: str) -> tuple[Path, Path]:
        processed = Path(processed_dir)
        return (
            processed / SEMANTIC_STAGE / group / SEMANTIC_REGISTRY_FILE,
            processed / CANONICAL_STAGE / group / CANONICAL_REGISTRY_FILE,
        )

    @staticmethod
    def open(processed_dir: Path, group: str) -> "SemanticSchema":

        semantic_path, canonical_path = SemanticSchema.paths(processed_dir, group)

        for path in (semantic_path, canonical_path):
            if not path.exists():
                raise SchemaError(f"нет файла {path}: этапы препроцессинга не собраны для группы {group}")

        return SemanticSchema(read_json(semantic_path), read_json(canonical_path))


__all__ = [
    "MODEL_FEATURE_KINDS",
    "ORIGIN_CATALOG",
    "ORIGIN_DERIVED",
    "ORIGIN_ENVELOPE",
    "ORIGIN_PAYLOAD",
    "ORIGIN_PROFILE",
    "ORIGIN_PROFILE_CHANGE",
    "ORIGIN_REFERENCE",
    "WEIGHT_PER_CLIENT",
    "WEIGHT_PER_EVENT",
    "KeyInfo",
    "SchemaError",
    "SemanticSchema",
]
