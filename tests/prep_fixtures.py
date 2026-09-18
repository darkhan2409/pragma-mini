from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.generator.config import (
    EVENT_TYPE_PRIORITY,
    SCHEMA_VERSION,
    EVENT_TYPE_SOURCE,
    HISTORY_END,
    HISTORY_START,
    REGISTRY_START,
    SOURCE_AVAILABILITY,
    SOURCE_PRECISION,
    SOURCES,
    TIME_PRECISIONS,
    key_catalogue,
)
from src.generator.emit import (
    COVERAGE_SCHEMA,
    EVENTS_SCHEMA,
    GEOGRAPHY_SCHEMA,
    MERCHANTS_SCHEMA,
    PRODUCTS_SCHEMA,
    ContentDigest,
    generate_dataset,
)
from src.generator.profile import PROFILE_SCHEMA


# ============================================================
# ИДЕЯ
# ============================================================
#
# Два источника данных для тестов препроцессинга:
#
#   tiny_raw   один раз на сессию сгенерированные 24 клиента
#              (три сообщества, ~25 с) — настоящий контракт v5;
#   MiniRaw    ручной набор из нескольких строк в формате v5 с
#              честными контрольными суммами — для целевых
#              случаев: граница cutoff, пустой месяц, лишний
#              ключ, короткий горизонт.
# ============================================================


TINY_CLIENTS = 24
TINY_COMMUNITY = 8
TINY_CATALOG_SCALE = 0.05


@pytest.fixture(scope="session")
def tiny_raw(tmp_path_factory) -> Path:

    out = tmp_path_factory.mktemp("raw_tiny")

    generate_dataset(
        total_clients=TINY_CLIENTS,
        out_dir=out,
        seed=42,
        workers=1,
        chunk_clients=TINY_COMMUNITY,
        catalog_scale=TINY_CATALOG_SCALE,
        community_size=TINY_COMMUNITY,
        quiet=True,
    )

    return out


CHECK_RAW = Path("data/raw/check")


@pytest.fixture(scope="session")
def check_raw() -> Path:

    if not (CHECK_RAW / "manifest.json").exists():
        pytest.skip("нет data/raw/check: локальный набор из 300 клиентов не сгенерирован")

    return CHECK_RAW


# ============================================================
# РУЧНОЙ RAW
# ============================================================


def _ts(value) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


# Отличает «поставь значение по умолчанию» от «поставь null».
DEFAULT = object()


class MiniRaw:
    """
    Строит каталог RAW v5 из ручных строк. Контрольные суммы
    считаются так же, как у генератора, поэтому паспорт
    принимает набор как подлинный.
    """

    def __init__(
        self,
        out: Path,
        history_start: datetime = HISTORY_START,
        history_end: datetime = HISTORY_END,
        extract_time: datetime | None = None,
        seed: int = 7,
        world_seed: int | None = None,
        availability: dict | None = None,
        schema_changes: tuple = (),
    ):
        self.out = Path(out)
        self.history_start = history_start
        self.history_end = history_end
        self.extract_time = extract_time or history_end
        self.seed = seed
        self.world_seed = world_seed
        # Источники «с начала истории» следуют за history_start набора,
        # как в генераторе; остальные хранят реальные даты запуска.
        self.availability = {
            source: (history_start if value == HISTORY_START else value)
            for source, value in SOURCE_AVAILABILITY.items()
        }
        if availability:
            self.availability.update(availability)
        self.schema_changes = tuple(schema_changes)

        self.events: list[dict] = []
        self.profile: list[dict] = []
        self.coverage: list[dict] = []
        self.products: list[dict] = []
        self._ids = 0

    # --- строки ---

    def event(
        self,
        client_id: str,
        event_type: str,
        event_time,
        payload: dict | None = None,
        event_id: str | None = None,
        version: int = 1,
        link_type: str | None = None,
        correlation_id: str | None = None,
        initiator: str = "client",
        precision: str | None = None,
        is_test_account: bool = False,
        effective_at=None,
        raw_payload: str | None = None,
    ) -> str:

        source = EVENT_TYPE_SOURCE.get(event_type, "transactions")

        event_time = _ts(event_time)

        if event_id is None:
            self._ids += 1
            event_id = f"ev{self._ids:015d}"

        self.events.append(
            {
                "event_id": event_id,
                "client_id": client_id,
                "event_type": event_type,
                "source": source,
                "event_time": event_time,
                "effective_at": _ts(effective_at) or event_time,
                "time_precision": precision or SOURCE_PRECISION.get(source, "second"),
                "event_version": version,
                "change_initiator": initiator,
                "correlation_id": correlation_id,
                "link_type": link_type,
                "is_test_account": is_test_account,
                "payload": raw_payload if raw_payload is not None else json.dumps(payload or {}, ensure_ascii=False),
            }
        )

        return event_id

    def product(
        self,
        product_id: str,
        product_code: str,
        product_name: str,
        product_family: str = "card",
        product_version: int = 1,
        tariff_version: int = 1,
        valid_from=None,
        status: str = "active",
    ) -> None:
        """
        Строка справочника продуктов: по ней слой расшифровывает
        название продукта на дату.
        """

        row = {name: None for name in PRODUCTS_SCHEMA.names}
        row.update(
            {
                "product_id": product_id,
                "product_code": product_code,
                "product_family": product_family,
                "product_name": product_name,
                "product_version": product_version,
                "tariff_version": tariff_version,
                "status": status,
                "valid_from": _ts(valid_from) or self.history_start,
                "valid_to": None,
                "unresolved_source": False,
                "is_synthetic": True,
            }
        )

        self.products.append(row)

    def profile_version(self, client_id: str, version: int, valid_from, **fields) -> None:

        row = {name: None for name in PROFILE_SCHEMA.names}
        row.update(
            {
                "client_id": client_id,
                "profile_version": version,
                "valid_from": _ts(valid_from),
                "valid_to": None,
                "change_source": "system",
                "confirmed": True,
                "change_reason": "monthly_recalculation",
            }
        )
        row.update(fields)
        self.profile.append(row)

    def cover(
        self,
        client_id: str,
        source: str,
        first_seen=DEFAULT,
        status: str = "full",
        reason: str | None = None,
        last_available_at=None,
        opening_state: str | None = None,
    ) -> None:

        self.coverage.append(
            {
                "client_id": client_id,
                "source": source,
                "first_available_at": self.availability[source],
                "last_available_at": _ts(last_available_at),
                "first_seen": (
                    max(self.availability[source], self.history_start)
                    if first_seen is DEFAULT
                    else _ts(first_seen)
                ),
                "coverage_status": status,
                "coverage_reason": reason,
                "opening_state": opening_state,
            }
        )

    def cover_all(self, client_id: str, first_seen=DEFAULT) -> None:
        for source in SOURCES:
            self.cover(client_id, source, first_seen=first_seen)

    # --- запись ---

    def write(self) -> Path:

        out = self.out
        (out / "catalog").mkdir(parents=True, exist_ok=True)
        (out / "truth").mkdir(parents=True, exist_ok=True)

        tables = {
            "events": ("events.parquet", EVENTS_SCHEMA, self.events),
            "profile": ("profile.parquet", PROFILE_SCHEMA, self.profile),
            "source_coverage": ("source_coverage.parquet", COVERAGE_SCHEMA, self.coverage),
        }

        rows: dict[str, int] = {}
        content: dict[str, str] = {}

        for name, (relative, schema, records) in tables.items():
            table = pa.Table.from_pylist(records, schema=schema)
            pq.write_table(table, out / relative, compression="zstd")
            digest = ContentDigest()
            digest.extend(table.to_pylist())
            rows[name] = table.num_rows
            content[name] = digest.value()

        catalogs = {
            "catalog/products.parquet": PRODUCTS_SCHEMA,
            "catalog/merchants.parquet": MERCHANTS_SCHEMA,
            "catalog/geography.parquet": GEOGRAPHY_SCHEMA,
        }

        for relative, schema in catalogs.items():
            rows_of = self.products if relative.endswith("products.parquet") else []
            pq.write_table(pa.Table.from_pylist(rows_of, schema=schema), out / relative, compression="zstd")

        clients = sorted({row["client_id"] for row in self.events} | {row["client_id"] for row in self.coverage})

        manifest = {
            "generator_version": "mini",
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "total_clients": len(clients),
            "community_size": 1,
            "communities": len(clients),
            "chunk_clients": 1,
            "history_start": self.history_start.isoformat(),
            "history_end": self.history_end.isoformat(),
            "registry_start": REGISTRY_START.isoformat(),
            "extract_time": self.extract_time.isoformat(),
            "sources": {
                source: {
                    "available_from": self.availability[source].isoformat(),
                    "time_precision": SOURCE_PRECISION[source],
                    "defect_profile": {},
                }
                for source in SOURCES
            },
            "time_precisions": list(TIME_PRECISIONS),
            "event_type_priority": EVENT_TYPE_PRIORITY,
            "key_catalogue": key_catalogue(),
            "schema_changes": [dict(item) for item in self.schema_changes],
            "conflict_rules": [],
            "bank_timeline": [],
            "product_timeline_sha256": "0" * 64,
            "unresolved_sources": 0,
            "catalog_rows": {"products": len(self.products), "geography": 0, "merchants": 0},
            "rows": {**rows, "truth_events": 0, "truth_relationships": 0, "truth_clients": 0},
            "content_sha256": content,
            "generation_config": {},
            "generation_config_sha256": "mini",
            "calibration_targets": [],
        }

        if self.world_seed is not None:
            manifest["world_seed"] = self.world_seed

        manifest["file_sha256"] = {
            path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out.rglob("*.parquet"))
        }

        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

        return out


def purchase_payload(amount: int = 12500, **overrides) -> dict:
    """
    Полный payload покупки по каталогу: все ключи присутствуют,
    необязательные — null.
    """

    payload = {
        "amount": amount,
        "currency": "KZT",
        "original_amount": None,
        "original_currency": None,
        "direction": "debit",
        "status": "approved",
        "decline_reason": None,
        "channel": "pos",
        "account_id": "acc_1",
        "card_id": "crd_1",
        "contract_id": None,
        "cause_event_id": None,
        "accrual_period": None,
        "reason": "purchase",
        "merchant_id": "mc_1",
        "outlet_id": "ot_1",
        "merchant_name": "Europharma 24",
        "mcc": "5912",
        "merchant_city": "Almaty",
        "merchant_country": "KZ",
        "is_online": False,
        "is_subscription": False,
        "counterparty": None,
        "balance_after": 100000,
    }

    payload.update(overrides)

    return payload
