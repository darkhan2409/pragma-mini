from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.tokenization.settings import ConfigError, NumericEncoder, TokenizerConfig

from tests.test_profile_state import QUIET_SNAPSHOT, RAW_CLIENT, prepare, raw_event


# ============================================================
# ИДЕЯ
# ============================================================
#
# Две настройки кодировщика числа (эксперимент волны 4, по
# умолчанию выключены — словарь прежний):
#
#   split_by       своя шкала на каждое значение ключа того же
#                  события: сумма зарплаты ищется среди диапазонов
#                  зачислений, покупка — среди списаний, операция
#                  без direction — среди своих; невиданное train
#                  условие — [UNK];
#   negative_bins  минус учится отдельной шкалой с границей в
#                  нуле: ни один диапазон не держит разом долг и
#                  остаток.
#
# Словарь учится функциями fit на крошечном train.
# ============================================================


MONEY = dict(method="quantile", bins=4, fallback=[1000.0, 5000.0], zero_policy="separate")

SPLIT = dict(MONEY, negative_policy="invalid", split_by="direction")

SIGNED = dict(MONEY, negative_policy="allowed", negative_bins=2)


def tape() -> list[dict]:

    events = []

    for number in range(12):
        events.append(raw_event(RAW_CLIENT, f"2025-03-{number + 1:02d}T10:00:00", {
            "type": "purchase", "amount": 100 * (number + 1), "direction": "debit",
            "status": "approved",
            # Долг редок, как в данных: доля минуса меньше корзины.
            "balance_after": (-400_000, -100_000)[number] if number < 2 else 1_000 * number,
        }))

    for number in range(6):
        events.append(raw_event(RAW_CLIENT, f"2025-04-{number + 1:02d}T10:00:00", {
            "type": "salary_credit", "amount": 100_000 * (number + 1), "direction": "credit",
            "status": "approved",
        }))

    for number in range(5):
        events.append(raw_event(RAW_CLIENT, f"2025-05-{number + 1:02d}T10:00:00", {
            "type": "app_operation", "domain": "transfers", "operation": "transfer_phone",
            "status": "success", "amount": 2_000 * (number + 1), "device_new": False,
            "session_id": f"ses_{number}",
        }))

    return events


def fitted(stage, **encoders):
    """
    Диапазоны, выученные на train с данными кодировщиками.
    """

    from src.tokenization.categorical import build_value_vocab
    from src.tokenization.fit import read_train
    from src.tokenization.keyvocab import build_key_vocab
    from src.tokenization.numeric import build_buckets
    from src.tokenization.schema import SemanticSchema
    from src.tokenization.specials import build_special_tokens

    prepare(stage, tape(), QUIET_SNAPSHOT, group="train")

    config = TokenizerConfig.from_dict(
        {"numeric_min_values": 1, "numeric_min_clients": 1, "numeric_encoders": encoders}
    )
    schema = SemanticSchema.open()

    train = read_train(config, schema)
    values = build_value_vocab(train, build_key_vocab(build_special_tokens(), schema), config, schema)

    buckets, _ = build_buckets(train, values, config, schema)

    return train, buckets


def lookup(buckets: dict):
    """
    Поиск диапазона тем же методом словаря, что и в кодировании.
    """

    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.numeric import read_buckets

    artifacts = SimpleNamespace(buckets=read_buckets(buckets))

    return lambda key, value, record=None: FrozenArtifacts.bucket_id(artifacts, key, value, record)


def test_by_default_nothing_is_split_and_the_file_keeps_its_format():

    from src.tokenization.numeric import Bucket

    encoders = TokenizerConfig.load(None).numeric_encoders

    assert all(spec.split_by is None and spec.negative_bins is None for spec in encoders.values())
    assert Bucket("amount_due_bucket_2", 0.0, 10.0, 7).as_dict() == {"id": 7, "min": 0.0, "max": 10.0}


def test_a_split_scale_is_learned_per_direction(stage):

    train, buckets = fitted(stage, transaction_amount=SPLIT)

    stats = train.statistics.split_numeric

    assert {condition: stats[("transaction_amount", condition)].n for _, condition in stats} == {
        "debit": 12, "credit": 6, None: 5,
    }

    entries = buckets["transaction_amount"]

    assert {item["when"] for item in entries.values()} == {"debit", "credit", None}
    assert all(item["split_by"] == "direction" for item in entries.values())
    assert {name.rsplit("_bucket_", 1)[0] for name in entries} == {
        "transaction_amount_credit", "transaction_amount_debit", "transaction_amount_no_direction",
    }

    # Зарплаты разошлись по своей шкале, а не легли в один диапазон.
    find = lookup(buckets)
    salaries = {find("transaction_amount", 100_000 * n, {"direction": "credit"}) for n in range(1, 7)}
    assert len(salaries) >= 3


def test_the_same_amount_is_coded_by_its_own_direction(stage):

    _, buckets = fitted(stage, transaction_amount=SPLIT)

    find = lookup(buckets)
    entries = buckets["transaction_amount"]

    debit = find("transaction_amount", 1_200, {"direction": "debit"})
    credit = find("transaction_amount", 1_200, {"direction": "credit"})
    silent = find("transaction_amount", 1_200, {"operation": "transfer_phone"})

    when = {item["id"]: item["when"] for item in entries.values()}

    assert (when[debit], when[credit], when[silent]) == ("debit", "credit", None)

    # Условие, которого train не видел, шкалы не имеет: [UNK].
    assert find("transaction_amount", 1_200, {"direction": "sideways"}) is None

    # Без записи условие неизвестно, и угадывать его нельзя.
    from src.tokenization.numeric import BucketsError

    with pytest.raises(BucketsError, match="записи нет"):
        find("transaction_amount", 1_200)


def test_negative_values_get_their_own_ranges(stage):

    _, buckets = fitted(stage, balance_after=SIGNED)

    entries = list(buckets["balance_after"].values())

    for item in entries:
        low = float("-inf") if item["min"] is None else item["min"]
        high = float("inf") if item["max"] is None else item["max"]
        assert high <= 0.0 or low >= 0.0, item

    find = lookup(buckets)

    ranges = {item["id"]: item for item in entries}

    for debt in (-1_000_000, -400_000, -100_000, -1):
        found = ranges[find("balance_after", debt)]
        assert found["max"] is not None and found["max"] <= 0.0

    assert find("balance_after", -1) != find("balance_after", 100)


def test_without_negative_bins_debt_shares_a_range_with_a_small_balance(stage):

    _, buckets = fitted(stage, balance_after=dict(MONEY, negative_policy="allowed"))

    find = lookup(buckets)

    assert find("balance_after", -400_000) == find("balance_after", 100)


@pytest.mark.parametrize("spec, message", [
    (dict(MONEY, negative_policy="invalid", negative_bins=2), "negative_bins"),
    (dict(method="fixed", boundaries=[1.0, 2.0], split_by="direction"), "split_by"),
    (dict(MONEY, split_by="direction", fit_source="amount_due"), "split_by"),
])
def test_contradictory_encoders_are_refused(spec: dict, message: str):

    with pytest.raises(ConfigError, match=message):
        NumericEncoder.from_dict(spec).validate("key")


def test_a_split_by_a_non_categorical_key_is_refused(stage):

    from src.tokenization.fit import FitError

    with pytest.raises(FitError, match="split_by"):
        fitted(stage, transaction_amount=dict(SPLIT, split_by="balance_after"))


def test_the_fitted_vocabulary_codes_an_event_by_its_direction(stage, tmp_path):
    """
    Вся цепочка fit на диск и кодирование события: сумма ищется
    среди диапазонов направления этого события.
    """

    import json

    from src.preprocessing.read import Group
    from src.preprocessing.settings import PreprocessingConfig
    from src.tokenization.encode import encode_event
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.run import run_fit

    prepare(stage, tape(), QUIET_SNAPSHOT, group="train")

    config = tmp_path / "tokenizer.json"
    config.write_text(json.dumps({
        "numeric_min_values": 1, "numeric_min_clients": 1,
        "numeric_encoders": {"transaction_amount": SPLIT},
    }))

    assert run_fit(SimpleNamespace(config=config)) == 0

    artifacts = FrozenArtifacts.load()

    cutoff = PreprocessingConfig.load(None).windows["train"].final_cutoff
    events = Group("train").history(RAW_CLIENT, cutoff).events

    amount = artifacts.key_id("transaction_amount")

    for event in events:

        record = encode_event(artifacts, event, 4)
        value = record.value_ids[record.key_ids.index(amount)]

        name = artifacts.describe(value)
        direction = event.values.get("direction")

        expected = f"transaction_amount_{direction}_" if direction else "transaction_amount_no_direction_"
        assert expected in name, (event.values, name)
