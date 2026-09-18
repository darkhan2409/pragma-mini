from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.preprocessing.canonical.build import REGISTRY_FILE, build_group
from src.preprocessing.history import CanonicalStore, history_as_of
from src.preprocessing.projection import (
    ENTITY_REFS,
    INTERNAL_FIELDS,
    SEMANTIC_PAYLOAD_FIELDS,
    ProjectionError,
    model_history,
    validate_projection,
)
from src.preprocessing.settings import PreprocessingConfig

from tests.prep_fixtures import MiniRaw, purchase_payload  # noqa: F401


CONFIG = PreprocessingConfig()

FULL_HORIZON = datetime(2023, 1, 1)


def _card_payload(**overrides) -> dict:
    payload = {
        "product_id": "prd_card",
        "product_code": "CARD",
        "product_version": 1,
        "tariff_version": 1,
        "product_family": "debit_card",
        "contract_id": "ctr_1",
        "account_id": "acc_1",
        "card_id": "crd_1",
        "offer_id": None,
        "previous_product_id": None,
        "migration_reason": None,
        "amount_or_limit": None,
        "term": None,
        "rate": None,
        "reason": "opened",
        "timestamp_quality": "exact",
    }
    payload.update(overrides)
    return payload


def test_model_event_passes_only_allowed_fields(tmp_path):
    """
    Граница модели: наружу выходят client_id, event_time, source
    и смысловые поля. Сырые идентификаторы заменены локальными
    ссылками, технические поля не проходят вовсе.
    """

    mini = MiniRaw(tmp_path / "raw", history_start=FULL_HORIZON)
    mini.cover_all("c1", first_seen="2023-01-01")

    mini.event("c1", "card_activated", "2023-02-01 10:00:00", payload=_card_payload())
    mini.event("c1", "card_blocked", "2023-02-05 10:00:00", payload=_card_payload(reason="blocked"), initiator="bank")

    purchase = purchase_payload(amount=12500)
    purchase["account_id"] = "acc_1"
    purchase["card_id"] = "crd_1"
    purchase["merchant_id"] = "mer_1"
    purchase["outlet_id"] = "out_1"

    original = mini.event("c1", "purchase", "2023-03-05 10:00:00", payload=purchase)
    mini.event(
        "c1", "purchase", "2023-03-05 10:00:00",
        payload={**purchase, "amount": 13000}, event_id=original, version=2,
    )

    # Вторая покупка в той же точке: ссылка должна совпасть.
    mini.event("c1", "purchase", "2023-03-09 10:00:00", payload={**purchase, "amount": 700})

    out = tmp_path / "canonical"
    build_group(mini.write(), out, CONFIG, "train")

    history = history_as_of(CanonicalStore(out), "c1", datetime(2023, 4, 1))

    events = model_history(history)

    assert [item.client_id for item in events] == ["c1"] * 4

    for item in events:
        assert set(item.as_dict()) == {"client_id", "event_time", "source", "fields"}
        assert "event_type" in item.fields
        # Ни одного технического поля и ни одного сырого идентификатора.
        assert not set(item.fields) & set(INTERNAL_FIELDS)
        assert not set(item.fields) & set(ENTITY_REFS)
        assert set(item.fields) <= set(SEMANTIC_PAYLOAD_FIELDS) | {"event_type", "change_initiator"} | {
            name for name, _ in ENTITY_REFS.values()
        }

    activated, blocked, corrected, second = events

    # Действует исправленная версия, событие осталось на своём месте.
    assert corrected.fields["amount"] == 13000
    assert corrected.fields["event_type"] == "purchase"

    # Одна сущность — одна ссылка во всех событиях клиента.
    assert activated.fields["card_ref"] == corrected.fields["card_ref"] == "CARD_1"
    assert activated.fields["account_ref"] == "ACCOUNT_1"
    assert corrected.fields["merchant_ref"] == second.fields["merchant_ref"] == "MERCHANT_1"
    assert corrected.fields["outlet_ref"] == "OUTLET_1"

    # Инициатор проходит там, где различие несёт смысл.
    assert blocked.fields["change_initiator"] == "bank"
    assert "change_initiator" not in corrected.fields

    # Реестр называет судьбу каждого поля и попадает в артефакт.
    registry = json.loads((out / REGISTRY_FILE).read_text(encoding="utf-8"))
    projection = registry["model_projection"]

    assert projection["counts"]["local_refs"] == len(ENTITY_REFS)
    assert "timestamp_quality" in projection["internal_fields"]
    assert "amount" in projection["semantic_fields"]

    roles = {item["name"]: item["model_role"] for item in registry["fields"]}
    assert roles["amount"] == "semantic_field"
    assert roles["card_id"] == "local_ref"
    assert roles["raw_row"] == "internal"

    # Календарь это отдельный канал: час и день недели из fields
    # убраны, время события остаётся единственным источником.
    assert not {"hour", "day_of_week", "day_of_month"} & set(SEMANTIC_PAYLOAD_FIELDS)
    for item in events:
        assert not {"hour", "day_of_week", "day_of_month"} & set(item.fields)

    assert projection["calendar"]["source"] == "event_time"
    assert len(projection["calendar"]["features"]) == 6


def test_unknown_payload_field_is_refused():
    """
    Новое поле выгрузки не проходит в модель само собой: его
    обязаны назвать.
    """

    with pytest.raises(ProjectionError, match="не классифицированы"):
        validate_projection(["amount", "внезапное_поле"])
