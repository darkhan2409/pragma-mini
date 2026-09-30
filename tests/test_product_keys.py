from __future__ import annotations

from tests.test_profile_state import EARLY, QUIET_SNAPSHOT, RAW_CLIENT, prepare, raw_event


# ============================================================
# ИДЕЯ
# ============================================================
#
# product_id и previous_product_id — коды каталога, а не
# идентификаторы сущностей: модель получает их категориями
# product и previous_product. Проверяется, что:
#
#   - продуктовое событие после 02 несёт свой продукт, а не
#     только reason;
#   - ключ один во всех источниках, где поле встречается;
#   - значения словарь учит только на train, и продукт, которого
#     train не видел, — [UNK].
# ============================================================


MIGRATED = raw_event(RAW_CLIENT, "2025-07-01T10:00:00", {
    "type": "product_migrated", "product_id": "prd_new", "previous_product_id": "prd_old",
    "migration_reason": "upgrade", "reason": "client_request",
})


def test_product_events_carry_their_product(stage):

    from src.preprocessing.keys import key_for

    events = prepare(stage, EARLY + [MIGRATED], QUIET_SNAPSHOT).events

    kinds = {event.values["event_type"]: event.values for event in events}

    assert kinds["product_opened"]["product"] == "prd_test"
    assert kinds["product_migrated"]["product"] == "prd_new"
    assert kinds["product_migrated"]["previous_product"] == "prd_old"

    # Ни сырого поля, ни идентификатора договора в модель не идёт.
    assert not any("product_id" in values or "contract_id" in values for values in kinds.values())

    sources = ("product_events", "applications", "communications", "banners", "app_screens")
    assert {key_for("product_id", source).key for source in sources} == {"product"}


def test_product_values_are_learned_on_train_only(stage):

    from src.tokenization.categorical import build_value_vocab
    from src.tokenization.fit import read_train
    from src.tokenization.keyvocab import build_key_vocab
    from src.tokenization.schema import SemanticSchema
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.specials import build_special_tokens

    prepare(stage, EARLY[:1] + [MIGRATED], QUIET_SNAPSHOT, group="train")

    config = TokenizerConfig.load(None)
    schema = SemanticSchema.open()

    keys = build_key_vocab(build_special_tokens(), schema)
    values = build_value_vocab(read_train(config, schema), keys, config, schema)

    assert {"product", "previous_product"} <= set(keys)
    assert sorted(values["product"]) == ["prd_new"]
    assert sorted(values["previous_product"]) == ["prd_old"]
