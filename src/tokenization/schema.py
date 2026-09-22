from __future__ import annotations

from dataclasses import dataclass

from src.generator.config import key_catalogue
from src.preprocessing.keys import (
    CATEGORICAL,
    DYNAMIC_FIELDS,
    DIRECT_KEYS,
    NUMERIC,
    REFERENCE,
    TEXT,
    KeysError,
    key_for,
    keys_registry,
)
from src.preprocessing.projection import PROJECTION_VERSION, EVENT_TYPE_FIELD, model_role


# ============================================================
# ИДЕЯ
# ============================================================
#
# Токенизатор не переосмысливает поля: что значит ключ, какого
# он вида и в каких единицах, говорит реестр смыслов, собранный
# ИЗ КОДА по каталогу ключей payload. Файлов-реестров рядом с
# данными больше нет.
#
# Какие ключи ОБЪЯВЛЕНЫ у каждого типа события, читается оттуда
# же. Различать это важно: «значения нет» и «поля здесь не
# бывает» разные вещи. У покупки отсутствие причины отказа
# значит «отказа не было», а у экрана приложения причины отказа
# не предусмотрено вовсе.
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

    def __init__(self, registry: dict | None = None):

        registry = registry or keys_registry(key_catalogue())

        if not registry.get("keys"):
            raise SchemaError("реестр смыслов пуст: каталог ключей не дал ни одного ключа")

        self.registry = registry

        self.keys_version = registry["keys_version"]
        self.projection_version = PROJECTION_VERSION

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
                raise SchemaError(f"у ключа {key} нет происхождения: реестр собран старой версией смыслового слоя")

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

        self.declared_payload, self.dynamic_event_types = self._declared()

    # --- разбор реестра полей ---

    @staticmethod
    def _declared() -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
        """
        Какие смысловые ключи объявлены у каждого типа события.

        Читается позитивно из каталога ключей payload: в расчёт
        идёт только то, что модельная проекция назвала смысловым
        полем. Локальные ссылки и внутренние поля сюда не
        попадают — ни те, ни другие значением модели не
        становятся.
        """

        declared: dict[str, set[str]] = {}
        dynamic: set[str] = set()
        problems: list[str] = []

        for owner, info in key_catalogue().items():

            declared.setdefault(owner, set())

            for item in info["fields"]:

                name = item["name"]

                if model_role(name) != "semantic_field":
                    continue

                if name in DYNAMIC_FIELDS:
                    # Смысл такого поля задаёт значение соседнего
                    # field_name, поэтому заранее он не объявлен.
                    dynamic.add(owner)
                    continue

                try:
                    declared[owner].add(key_for(name, info["source"]).key)
                except KeysError as error:
                    problems.append(str(error))

        if problems:
            raise SchemaError("; ".join(sorted(set(problems))))

        event_type_key = DIRECT_KEYS[EVENT_TYPE_FIELD].key

        for owner, keys in declared.items():
            keys.add(event_type_key)

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
    def open(group: str | None = None) -> "SemanticSchema":
        """
        Реестр смыслов из кода. Группа на смысл не влияет: он
        один на весь конвейер.
        """

        return SemanticSchema()


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
