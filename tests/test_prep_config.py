"""
Реестр полей: он не может разойтись со схемами RAW.
"""

from __future__ import annotations

import pytest

from src.generator.config import EVENT_TYPES, PROFILE_DYNAMIC_FIELDS, PROFILE_FIELDS
from src.generator.emit import SCHEMAS, schemas_for
from src.generator.timeline import PAYLOAD_BUILDERS, payload_builders
from src.generator.version import RAW_SCHEMA_REVISION, REVISIONS
from src.preprocessing.config import (
    EVENT_TYPE_PROFILE,
    KIND_CATEGORICAL,
    KIND_METADATA,
    KIND_NUMERIC,
    LATENT_NAMES,
    NAMESPACE_TABLE,
    REGISTRY,
    Settings,
    feature_specs,
    numeric_specs,
    payload_fields,
    payload_schema,
    predictable_specs,
    specs_for,
)
from tests.test_raw_schema import LATENT_COLUMNS


# ============================================================
# ПОКРЫТИЕ
# ============================================================


def test_registry_covers_every_raw_column():
    """
    Каждая колонка каждой таблицы RAW описана в реестре.
    """

    for table, schema in SCHEMAS.items():

        if table == "timeline":
            namespaces = ["timeline"]
        elif table == "profile":
            namespaces = ["profile"]
        else:
            namespaces = [
                namespace for namespace, name in NAMESPACE_TABLE.items() if name == table
            ]

        for namespace in namespaces:

            described = {spec.field for spec in specs_for(namespace)}

            if namespace in EVENT_TYPES:
                # У события описаны поля payload, ключи лежат в timeline.
                expected = set(schema.names[2:])
                if namespace == EVENT_TYPE_PROFILE:
                    expected = set(PROFILE_DYNAMIC_FIELDS)
            else:
                expected = set(schema.names)

            assert expected <= described, (namespace, sorted(expected - described))


@pytest.mark.parametrize("revision", REVISIONS)
def test_payload_fields_follow_generator_order(revision: int):
    """
    Порядок полей payload совпадает с порядком генератора.

    Проверяется каждая ревизия схемы: ключи payload и колонки
    таблицы обязаны меняться вместе, иначе лента разойдётся
    с таблицами именно там, где это труднее всего заметить.
    """

    schemas = schemas_for(revision)

    for source, builder in payload_builders(revision).items():

        from src.generator.config import EVENT_TYPE_BY_SOURCE

        event_type = EVENT_TYPE_BY_SOURCE[source]

        if source == "profile":
            assert payload_fields(event_type, revision) == tuple(PROFILE_DYNAMIC_FIELDS)
            continue

        assert payload_fields(event_type, revision) == tuple(schemas[source].names[2:])


@pytest.mark.parametrize("revision", REVISIONS)
def test_payload_schema_types_match_raw(revision: int):

    schemas = schemas_for(revision)

    for event_type in EVENT_TYPES:

        schema = payload_schema(event_type, revision)

        table = NAMESPACE_TABLE[event_type]

        for name in schema.names:

            expected = (
                schemas["profile"].field(name).type
                if event_type == EVENT_TYPE_PROFILE
                else schemas[table].field(name).type
            )

            assert schema.field(name).type == expected, (event_type, name)



def test_profile_namespace_holds_all_twenty_fields():
    described = {spec.field for spec in specs_for("profile")}

    assert set(PROFILE_FIELDS) <= described
    assert len(described) == len(PROFILE_FIELDS) + 3


# ============================================================
# ВИДЫ И ФЛАГИ
# ============================================================


def test_metadata_is_never_predictable():
    for spec in REGISTRY.values():
        if spec.kind == KIND_METADATA:
            assert not spec.predictable, spec.key


def test_metadata_is_not_a_feature():
    for spec in REGISTRY.values():
        if spec.kind == KIND_METADATA:
            assert not spec.is_feature or spec.role == "feature", spec.key
            assert spec.namespace and spec.field


def test_mcc_is_categorical():
    assert REGISTRY[("transaction", "mcc")].kind == KIND_CATEGORICAL


def test_service_fields_are_metadata():
    assert REGISTRY[("app_screen", "session_id")].kind == KIND_METADATA
    assert REGISTRY[("timeline", "seq")].kind == KIND_METADATA
    assert REGISTRY[("timeline", "client_id")].kind == KIND_METADATA
    assert REGISTRY[("profile", "snapshot_month")].kind == KIND_METADATA


def test_event_type_is_predictable():
    spec = REGISTRY[("timeline", "event_type")]

    assert spec.predictable
    assert spec.kind == KIND_CATEGORICAL


def test_derived_and_quality_fields_are_not_predictable():
    for key in (
        ("communication", "day_of_week"),
        ("communication", "hour"),
        ("product_event", "timestamp_quality"),
    ):
        assert not REGISTRY[key].predictable, key


def test_full_profile_is_context_not_target():
    for spec in specs_for("profile"):
        assert not spec.predictable, spec.key


def test_profile_snapshot_fields_are_predictable():
    predictable = {spec.field for spec in predictable_specs() if spec.namespace == EVENT_TYPE_PROFILE}

    assert predictable == set(PROFILE_DYNAMIC_FIELDS)


def test_labels_and_coverage_are_excluded():
    for namespace in ("labels", "source_coverage"):
        for spec in specs_for(namespace):
            assert not spec.is_feature, spec.key
            assert not spec.predictable, spec.key


def test_numeric_fields_are_known():
    numeric = {spec.key for spec in numeric_specs()}

    assert ("transaction", "amount") in numeric
    assert ("product_event", "term") in numeric
    assert ("profile", "credit_utilization") in numeric
    assert ("transaction", "mcc") not in numeric


def test_column_names_are_unique():
    columns = [spec.column for spec in feature_specs()]

    assert len(columns) == len(set(columns))


# ============================================================
# СКРЫТЫЕ ПОЛЯ
# ============================================================


def test_latent_names_cover_generator_test_list():
    assert LATENT_COLUMNS <= LATENT_NAMES


def test_latent_names_do_not_intersect_registry():
    described = {spec.field for spec in REGISTRY.values()}

    assert not (described & LATENT_NAMES)


# ============================================================
# НАСТРОЙКИ
# ============================================================


def test_bucket_override_applies():
    spec = REGISTRY[("transaction", "amount")]

    settings = Settings(bucket_overrides={spec.key: 4})

    assert settings.buckets_for(spec) == 4
    assert settings.buckets_for(REGISTRY[("product_event", "term")]) == settings.default_buckets


def test_settings_serialize_without_tuples():
    data = Settings(bucket_overrides={("transaction", "amount"): 8}).as_dict()

    assert data["bucket_overrides"] == {"transaction.amount": 8}
    assert set(data["split_shares"]) == {"train", "val", "test"}
