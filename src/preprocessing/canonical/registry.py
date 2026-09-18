from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pyarrow as pa

from ..artifacts import dumps_json, sha256_bytes
from ..projection import model_role, projection_registry
from ..rawdata import DTYPE_MAP, RawManifest


# ============================================================
# ИДЕЯ
# ============================================================
#
# Реестр физических полей: единственное место, где сказано, что
# есть каждое поле RAW.
#
# Физическое поле это ПАРА (владелец, имя), а не одно имя.
# Проверено на выгрузке: dtype одного имени совпадает во всех
# типах событий, а вот nullable и level расходятся — например,
# channel обязателен у заявки и необязателен у операции, а его
# уровень бывает операцией, коммуникацией и обращением. Поэтому
# колонка в таблице общая (по имени), а идентичность и правила
# живут на паре.
#
# field_id это устойчивый индекс трассировки, а НЕ словарь
# токенов: он ничего не кодирует и не участвует в модели.
#
# Единицы не угадываются: они взяты из описаний самого каталога
# ключей генератора («сумма операции в тенге», «срок договора в
# месяцах», «дней просрочки») и перечислены здесь явно.
# ============================================================


REGISTRY_VERSION = "1.1.0"

# Владельцы полей.
OWNER_ENVELOPE = "envelope"
OWNER_PROFILE = "profile"
OWNER_COVERAGE = "coverage"
OWNER_DERIVED = "derived"

# Роли: что поле значит для дальнейших этапов.
ROLE_PAYLOAD = "payload"
ROLE_ENVELOPE = "envelope"
ROLE_PROFILE = "profile"
ROLE_COVERAGE = "coverage"
ROLE_DERIVED = "derived"
ROLE_TRACE = "trace"
ROLE_INTERNAL = "internal"


# ------------------------------------------------------------
# ЕДИНИЦЫ
# ------------------------------------------------------------
#
# Только то, что прямо сказано в описании поля. Там, где единицы
# нет (код, идентификатор, категория), стоит None.
# ------------------------------------------------------------

UNITS: dict[str, str] = {
    # деньги: генератор объявляет тенге и хранит целые
    "amount": "KZT",
    "amount_or_limit": "KZT",
    "amount_due": "KZT",
    "amount_paid": "KZT",
    "principal_outstanding": "KZT",
    "balance_after": "KZT",
    "requested_amount": "KZT",
    "approved_amount": "KZT",
    "declared_income": "KZT",
    "credit_limit": "KZT",
    # сумма в валюте страны покупки: единица лежит в original_currency
    "original_amount": "original_currency",
    # сроки
    "term": "months",
    "requested_term": "months",
    "approved_term": "months",
    "relationship_months": "months",
    "days_past_due": "days",
    # доли
    "rate": "fraction_per_year",
    "credit_utilization": "fraction",
    # счётчики
    "children": "count",
    "contracts_count": "count",
    "active_contracts": "count",
    "installment_no": "count",
    "age": "years",
    # код календаря: число это код, а не величина. Час и день
    # недели события сюда не входят: они не поля выгрузки, а
    # отдельный временной канал, считаемый из event_time.
    "salary_day": "day_of_month_code",
}

# Поля, чей текст нормализуется отдельной копией.
NORMALIZED_TEXT: tuple[str, ...] = ("merchant_name", "counterparty")

# Ссылочные поля: связывают события с сущностями и друг с другом.
REFERENCE_FIELDS: dict[str, str] = {
    "account_id": "account",
    "card_id": "card",
    "contract_id": "contract",
    "application_id": "application",
    "case_id": "case",
    "offer_id": "offer",
    "cause_event_id": "event",
    "merchant_id": "merchant",
    "outlet_id": "outlet",
    "product_id": "product",
    "previous_product_id": "product",
}


@dataclass(frozen=True)
class FieldEntry:
    field_id: int
    owner: str
    name: str
    dtype: str
    nullable: bool
    level: str
    role: str
    # Судьба поля за границей модели: смысловое значение,
    # локальная ссылка или внутреннее поле слоя.
    model_role: str
    unit: str | None
    reference_to: str | None
    column: str
    description: str
    # Источник события, которому принадлежит поле payload. Смысл
    # поля это пара (имя, источник), и без источника реестр не
    # даёт восстановить каталог ключей.
    source: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.owner, self.name)

    @property
    def arrow_type(self) -> pa.DataType:
        return DTYPE_MAP[self.dtype]


def _arrow_name(field_type: pa.DataType) -> str:
    for name, value in DTYPE_MAP.items():
        if value.equals(field_type):
            return name
    if pa.types.is_timestamp(field_type):
        return "timestamp"
    if pa.types.is_int32(field_type) or pa.types.is_int64(field_type):
        return "int"
    if pa.types.is_floating(field_type):
        return "float"
    if pa.types.is_boolean(field_type):
        return "bool"
    return str(field_type)


def build_registry(manifest: RawManifest, extra: Iterable[tuple[str, str, str, str]] = ()) -> list[FieldEntry]:
    """
    Реестр в устойчивом порядке: конверт, поля payload по типам
    событий в порядке каталога, профиль, покрытие, производные
    колонки canonical.

    extra: (owner, name, dtype, description) для производных
    колонок, которых в RAW нет.
    """

    from ..rawdata import COVERAGE_SCHEMA, ENVELOPE_SCHEMA

    entries: list[FieldEntry] = []

    def add(owner: str, name: str, dtype: str, nullable: bool, level: str, role: str, description: str, column: str | None = None, source: str | None = None) -> None:
        entries.append(
            FieldEntry(
                field_id=len(entries),
                owner=owner,
                name=name,
                dtype=dtype,
                nullable=nullable,
                level=level,
                role=role,
                model_role=model_role(name),
                unit=UNITS.get(name),
                reference_to=REFERENCE_FIELDS.get(name),
                column=column or name,
                description=description,
                source=source,
            )
        )

    # --- конверт ---

    for field in ENVELOPE_SCHEMA:
        if field.name == "payload":
            continue
        add(
            OWNER_ENVELOPE,
            field.name,
            _arrow_name(field.type),
            field.name in ("effective_at", "correlation_id", "link_type"),
            "event",
            ROLE_ENVELOPE,
            "поле конверта события",
        )

    # --- payload по типам событий ---

    for event_type, info in manifest.catalogue.items():
        for item in info.fields:
            add(
                event_type,
                item.name,
                item.dtype,
                item.nullable,
                item.level,
                ROLE_PAYLOAD,
                item.description,
                source=info.source,
            )

    # --- профиль ---

    from src.generator.profile import PROFILE_SCHEMA

    for field in PROFILE_SCHEMA:
        add(
            OWNER_PROFILE,
            field.name,
            _arrow_name(field.type),
            field.name not in ("client_id", "profile_version", "valid_from"),
            "client",
            ROLE_PROFILE,
            "поле версии профиля",
        )

    # --- покрытие ---

    for field in COVERAGE_SCHEMA:
        add(
            OWNER_COVERAGE,
            field.name,
            _arrow_name(field.type),
            field.name in ("last_available_at", "first_seen", "coverage_reason", "opening_state"),
            "client_source",
            ROLE_COVERAGE,
            "поле покрытия источника",
        )

    # --- производные колонки canonical ---

    for owner, name, dtype, description in extra:
        role = ROLE_TRACE if name.startswith("raw_") else ROLE_DERIVED
        add(owner, name, dtype, True, "event", role, description)

    return entries


def registry_as_dict(entries: list[FieldEntry], timezone: str | None = None) -> dict:

    return {
        "registry_version": REGISTRY_VERSION,
        "model_projection": projection_registry(
            (item.name for item in entries if item.role == ROLE_PAYLOAD),
            timezone=timezone,
        ),
        "rules": {
            "identity": "физическое поле это пара (владелец, имя): dtype у одного имени совпадает, nullable и level различаются",
            "field_id": "устойчивый индекс трассировки, не словарь токенов и не вход модели",
            "column": "колонка canonical называется по имени поля; принадлежность даёт event_type строки",
            "units": "взяты из описаний каталога ключей генератора; там, где единицы нет, стоит null",
            "reference_to": "поле связывает событие с сущностью этого вида",
            "model_role": "что происходит с полем на границе модели: см. блок model_projection",
        },
        "counts": {
            "total": len(entries),
            "by_role": _count(entries, lambda item: item.role),
            "by_owner_kind": {
                "envelope": sum(1 for item in entries if item.owner == OWNER_ENVELOPE),
                "payload": sum(1 for item in entries if item.role == ROLE_PAYLOAD),
                "profile": sum(1 for item in entries if item.owner == OWNER_PROFILE),
                "coverage": sum(1 for item in entries if item.owner == OWNER_COVERAGE),
                "derived": sum(1 for item in entries if item.owner == OWNER_DERIVED),
            },
            "distinct_payload_names": len({item.name for item in entries if item.role == ROLE_PAYLOAD}),
        },
        "fields": [asdict(item) for item in entries],
    }


def catalogue_from_registry(registry: dict) -> dict:
    """
    Каталог ключей, восстановленный из реестра полей canonical.

    Смысловой слой читает свой ВХОД, а не импортирует генератор:
    иначе реестр смыслов описывал бы код, которым данные могли
    быть и не собраны, и подмена генератора осталась бы
    незамеченной.
    """

    catalogue: dict[str, dict] = {}

    for item in registry.get("fields", ()):

        if item.get("role") != ROLE_PAYLOAD:
            continue

        event_type = item["owner"]

        entry = catalogue.setdefault(event_type, {"source": item.get("source"), "fields": []})
        entry["fields"].append({"name": item["name"]})

    return catalogue


def _count(entries: list[FieldEntry], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in entries:
        counts[key(item)] = counts.get(key(item), 0) + 1
    return dict(sorted(counts.items()))


def registry_digest(entries: list[FieldEntry]) -> str:
    return sha256_bytes(dumps_json(registry_as_dict(entries)).encode("utf-8"))


__all__ = [
    "catalogue_from_registry",
    "NORMALIZED_TEXT",
    "OWNER_COVERAGE",
    "OWNER_DERIVED",
    "OWNER_ENVELOPE",
    "OWNER_PROFILE",
    "REFERENCE_FIELDS",
    "REGISTRY_VERSION",
    "UNITS",
    "FieldEntry",
    "build_registry",
    "registry_as_dict",
    "registry_digest",
]
