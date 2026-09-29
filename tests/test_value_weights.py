from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.masking.apply import IGNORE, apply
from src.masking.choose import KEY, VALUE, choose, value_chance, values_of
from src.masking.settings import ConfigError, MaskingConfig
from src.masking.weights import (
    VALUE_WEIGHT_MAX,
    ValueWeights,
    WeightsError,
    build,
    key_weight,
    load_value_weights,
    value_weight,
)

from tests import world
from tests.test_key_hiding import val_row
from tests.test_masking import row_of
from tests.test_profile_state import QUIET_SNAPSHOT, RAW_CLIENT, prepare, raw_event


# ============================================================
# ИДЕЯ
# ============================================================
#
# Механизм value выбирает цель с вероятностью, зависящей от
# статистики TRAIN: у почти константного ключа и частого значения
# она ниже, у разнообразного ключа и редкого значения — выше, но
# всегда в [min_value_probability, max_value_probability].
# Проверяется, что:
#
#   - веса следуют энтропии ключа и частоте значения и ограничены;
#   - статистика собрана только по train и заморожена для val/test;
#   - розыгрыш остаётся розыгрышем: seed и клиент дают ту же маску,
#     другой seed — другую, значение BPE выбирается целиком;
#   - механизмы event и key, порча контекста и сама маска при
#     выключенном взвешивании не меняются ни на бит.
# ============================================================


CURRENCY, MERCHANT = 100, 101

KZT, USD, EUR = 1000, 1001, 1002


def merchants() -> dict[tuple[int, ...], float]:
    """
    Разнообразный ключ: 40 значений с убывающей частотой.
    """

    return {(2000 + number,): float(200 // (number + 1)) for number in range(40)}


def weights_of(counts: dict[str, dict], key_ids: dict[str, int] | None = None) -> ValueWeights:

    ids = key_ids or {"currency": CURRENCY, "merchant": MERCHANT}

    return ValueWeights.from_payload(build(counts, ids, {}, "test"))


def market() -> ValueWeights:
    return weights_of({
        "currency": {(KZT,): 980.0, (USD,): 15.0, (EUR,): 5.0},
        "merchant": merchants(),
    })


def weighted(**overrides) -> MaskingConfig:
    return MaskingConfig(**{"event_probability": 0.0, "key_probability": 0.0, "unknown_probability": 0.0,
                            **overrides})


# ============================================================
# ВЕСА
# ============================================================


def test_dominant_kzt_is_masked_less_than_rare_currencies():

    weights = market()
    config = MaskingConfig()

    kzt, usd, eur = (weights.probability(CURRENCY, (token,), config) for token in (KZT, USD, EUR))

    assert kzt < usd <= eur
    assert kzt < config.value_probability


def test_a_low_entropy_key_weighs_less_than_a_diverse_one():

    assert key_weight([1000.0]) == 0.0
    assert key_weight([5.0, 5.0, 5.0, 5.0]) == pytest.approx(1.0)
    assert key_weight([980.0, 15.0, 5.0]) < key_weight(list(merchants().values()))

    payload = build({"currency": {(KZT,): 980.0, (USD,): 15.0, (EUR,): 5.0}, "merchant": merchants()},
                    {"currency": CURRENCY, "merchant": MERCHANT}, {}, "test")

    assert payload["keys"]["currency"]["key_weight"] < payload["keys"]["merchant"]["key_weight"]


def test_a_very_rare_value_gets_a_bounded_weight():

    entropy = 2.0

    assert value_weight(1.0, 1e6, entropy) == VALUE_WEIGHT_MAX
    assert value_weight(1.0, 1e12, entropy) == VALUE_WEIGHT_MAX

    weights = market()
    config = MaskingConfig()

    rare = weights.probability(MERCHANT, (2039,), config)
    unseen = weights.probability(MERCHANT, (9999,), config)
    typical = weights.probability(MERCHANT, (2000,), config)

    assert rare <= config.max_value_probability
    assert unseen <= config.max_value_probability
    # Сюрприз растёт логарифмом: редкое выше частого, но не в
    # десятки раз.
    assert typical < rare <= typical * VALUE_WEIGHT_MAX / 0.25


@pytest.mark.parametrize("limits", [(0.05, 0.35), (0.1, 0.2), (0.0, 1.0)])
def test_every_probability_lies_within_the_configured_limits(limits):

    low, high = limits

    generator = np.random.default_rng(3)

    counts = {}
    ids = {}

    for number in range(30):
        size = int(generator.integers(1, 60))
        values = generator.pareto(float(generator.uniform(0.3, 3.0)), size) + 1.0
        counts[f"k{number}"] = {(10_000 + number * 100 + place,): float(value) for place, value in enumerate(values)}
        ids[f"k{number}"] = 200 + number

    weights = ValueWeights.from_payload(build(counts, ids, {}, "test"))

    config = MaskingConfig(min_value_probability=low, max_value_probability=high)

    probabilities = [
        weights.probability(ids[key], tokens, config) for key in counts for tokens in counts[key]
    ]
    probabilities += [weights.probability(200, (1,), config), weights.probability(999, (1,), config)]

    assert all(low <= value <= high for value in probabilities)


def test_the_value_mechanism_keeps_its_average_rate_on_train():
    """
    Масштаб g возвращает средней вероятности до обрезки
    value_probability: веса перераспределяют цели, а не убавляют их.
    """

    counts = {"currency": {(KZT,): 980.0, (USD,): 15.0, (EUR,): 5.0}, "merchant": merchants()}

    weights = weights_of(counts)
    open_limits = MaskingConfig(min_value_probability=0.0, max_value_probability=1.0)

    ids = {"currency": CURRENCY, "merchant": MERCHANT}

    total = sum(count for table in counts.values() for count in table.values())
    mean = sum(
        count * weights.probability(ids[key], tokens, open_limits)
        for key, table in counts.items() for tokens, count in table.items()
    ) / total

    assert mean == pytest.approx(open_limits.value_probability, rel=1e-9)


def test_zero_value_probability_switches_the_mechanism_off():

    row = val_row()
    config = weighted(value_probability=0.0)

    weights = market()

    assert choose("val", row, config, weights).choices == []

    for value in values_of(row):
        assert value_chance(value, row, config, weights) == 0.0


def test_limits_are_checked():

    with pytest.raises(ConfigError, match="min_value_probability"):
        MaskingConfig.from_dict({"min_value_probability": 0.5, "max_value_probability": 0.2})

    assert MaskingConfig().informativeness_weighted_masking is True
    assert (MaskingConfig().min_value_probability, MaskingConfig().max_value_probability) == (0.05, 0.35)


# ============================================================
# РОЗЫГРЫШ
# ============================================================


def world_weights() -> ValueWeights:
    """
    Веса крошечного мира: key_a почти константен, key_b разнообразен.
    """

    return weights_of(
        {
            "key_a": {(10,): 990.0, (14,): 5.0, (25,): 5.0},
            "key_b": {(11, 12, 13): 30.0, (16,): 30.0, (28, 29): 30.0, (19,): 30.0},
        },
        {"key_a": world.KEY_A, "key_b": world.KEY_B},
    )


def read(group: str, row: dict, config: MaskingConfig, weights: ValueWeights | None) -> dict:

    selection = choose(group, row, config, weights)

    return apply(row["client_id"], row, selection.choices, world.MASK, world.UNK, selection.corrupted)


def test_one_seed_and_one_client_give_the_same_mask():

    row = row_of(world.population()[0])
    config = weighted(seed=5)

    assert read("train", row, config, world_weights()) == read("train", row, config, world_weights())


def test_another_seed_can_choose_differently():

    row = row_of(world.population()[0])

    masks = {
        json.dumps(read("train", row, weighted(seed=seed), world_weights())["labels"])
        for seed in range(30)
    }

    assert len(masks) > 1


def test_a_bpe_value_is_chosen_whole():

    made = world.make("bpe", [[(world.KEY_B, [11, 12, 13], False)]] * 6, [(world.KEY_A, [20])])
    row = row_of(made)

    outcomes = set()

    for seed in range(40):

        masked = read("train", row, weighted(seed=seed, max_value_probability=0.9), world_weights())

        for value in values_of(row):
            labelled = [masked["labels"][index] != IGNORE for index in range(value.start, value.start + value.length)]
            assert all(labelled) or not any(labelled)
            outcomes.add(all(labelled))

    assert outcomes == {True, False}


def test_probabilities_drive_the_draw():
    """
    Много вхождений одного значения: доля выбранных близка к его
    вероятности. Розыгрыш детерминирован, поэтому и число целей тоже.
    """

    config = weighted(seed=9)
    weights = world_weights()

    for tokens, pieces in (((10,), [10]), ((19,), [19])):

        made = world.make("rate", [[(world.KEY_A if tokens == (10,) else world.KEY_B, pieces, False)]] * 400,
                          [(world.KEY_A, [20])])
        row = row_of(made)

        key = world.KEY_A if tokens == (10,) else world.KEY_B
        expected = weights.probability(key, tokens, config)

        chosen = sum(1 for choice in choose("train", row, config, weights).choices if choice.reason == VALUE)

        spread = 4 * math.sqrt(400 * expected * (1 - expected))

        assert abs(chosen - 400 * expected) <= spread, (tokens, chosen, expected)


# ============================================================
# ОСТАЛЬНЫЕ МЕХАНИЗМЫ И ПРЕЖНЕЕ ПОВЕДЕНИЕ
# ============================================================


# Отпечаток масок, снятый кодом до взвешивания: value_ids, labels и
# reason населения мира и клиента с частью целей по трём группам и
# трём seed.
BEFORE_WEIGHTING = "d9e81dc48cbb1e6e3df23b0262b68af738977b738562a0304acaa16371b2438a"


def fingerprint(weights: ValueWeights | None) -> str:

    digest = hashlib.sha256()

    rows = [row_of(made) for made in world.population("g")] + [val_row()]

    for group in ("train", "val", "test"):
        for seed in (1, 7, 42):

            config = MaskingConfig(
                seed=seed, value_probability=0.3, event_probability=0.1, key_probability=0.2,
                unknown_probability=0.1, key_context_corruption_probability=0.5,
                informativeness_weighted_masking=False,
            )

            for row in rows:
                masked = read(group, row, config, weights)
                digest.update(json.dumps([masked["value_ids"], masked["labels"], masked["reason"]]).encode())

    return digest.hexdigest()


def test_without_weighting_the_mask_is_bit_for_bit_the_previous_one():

    assert fingerprint(None) == BEFORE_WEIGHTING
    assert fingerprint(world_weights()) == BEFORE_WEIGHTING


@pytest.mark.parametrize("seed", range(20))
def test_event_key_and_context_draws_do_not_depend_on_weighting(seed: int):

    row = val_row()
    base = MaskingConfig(seed=seed, value_probability=0.3, event_probability=0.2, key_probability=0.3,
                         unknown_probability=0.0, informativeness_weighted_masking=False)

    plain = choose("val", row, base)
    weighted_selection = choose("val", row, dataclasses.replace(base, informativeness_weighted_masking=True),
                                world_weights())

    def others(selection) -> list:
        return [(choice.value, choice.reason) for choice in selection.choices if choice.reason != VALUE]

    assert others(plain) == others(weighted_selection)
    assert plain.corrupted == weighted_selection.corrupted


def test_the_comparison_above_is_not_empty():

    base = MaskingConfig(seed=0, value_probability=0.3, event_probability=0.2, key_probability=0.3,
                         unknown_probability=0.0, informativeness_weighted_masking=False)

    reasons = {
        choice.reason
        for seed in range(20)
        for choice in choose("val", val_row(), dataclasses.replace(base, seed=seed)).choices
    }

    assert {KEY, VALUE} <= reasons


def test_weighting_without_weights_is_an_error():

    row = val_row()

    with pytest.raises(WeightsError, match="весов train"):
        choose("val", row, weighted())


# ============================================================
# ФАЙЛ ВЕСОВ И ЧИТАТЕЛИ
# ============================================================


def test_readers_refuse_weights_of_another_vocabulary(stage):

    from src.embedding.inputs import InputError as EmbeddingInputError
    from src.embedding.inputs import Source as EmbeddingSource
    from src.mlm.inputs import InputError, Source
    from src.tokenization.settings import VALUE_WEIGHTS_FILE

    world.write_samples("train", [world.population()])

    path = stage / "03_vocab" / VALUE_WEIGHTS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vocabulary"] = "другой"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="другой словарь"):
        Source("train")

    with pytest.raises(EmbeddingInputError, match="другой словарь"):
        EmbeddingSource("train")

    path.unlink()

    with pytest.raises(InputError, match="python -m src.tokenization.run fit"):
        Source("train")

    # Без взвешивания файл не нужен вовсе.
    assert list(Source("train", masking=MaskingConfig(informativeness_weighted_masking=False)).clients())


def test_the_reader_uses_the_frozen_weights(stage):

    from src.mlm.inputs import Source

    world.write_samples("val", [world.population()])

    config = MaskingConfig(seed=3)
    source = Source("val", masking=config)
    weights = load_value_weights()

    for client, made in zip(source.clients(), world.population()):
        expected = read("val", row_of(made), config, weights)
        assert client.value_ids.tolist() == expected["value_ids"]
        assert client.labels.tolist() == expected["labels"]


def test_resume_refuses_changed_value_weights(stage):

    from src.mlm.settings import checkpoint_path
    from src.mlm.train import CheckpointError, train
    from src.tokenization.settings import VALUE_WEIGHTS_FILE

    from tests.test_scheduler import many
    from tests.test_training_math import settle, tiny

    settle(stage, train_people=many())

    # Цели почти у каждого клиента: иначе шагов мало, и прогон
    # закончил бы все эпохи раньше паузы.
    config = tiny(token_budget=6)
    masking = MaskingConfig(seed=5, value_probability=0.9, max_value_probability=0.9)

    train(config, epochs=3, max_steps=2, masking=masking)

    paused = checkpoint_path().read_bytes()

    path = stage / "03_vocab" / VALUE_WEIGHTS_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scale"] = payload["scale"] * 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError, match="данные изменились") as error:
        train(config, epochs=3, max_steps=None, masking=masking, resume=True)

    assert "03_vocab/value_weights" in str(error.value)
    assert checkpoint_path().read_bytes() == paused


def test_the_report_accounts_for_every_target(stage):

    from src.masking.report import masking_report

    world.write_samples("train", [world.population(), world.population("x")])

    report = masking_report("train", MaskingConfig(seed=2, value_probability=0.5))

    assert report["targets"] == sum(item["targets"] for item in report["keys"])
    assert report["values"] == sum(item["values"] for item in report["keys"])
    assert 0.0 <= report["key_tv"] <= 1.0

    for item in report["keys"]:
        assert 0.0 <= item["data_share"] <= 1.0 and 0.0 <= item["target_share"] <= 1.0
        assert 0.05 - 1e-9 <= item["mean_value_probability"] <= 0.35 + 1e-9
        assert item["value_tv"] is None or 0.0 <= item["value_tv"] <= 1.0


# ============================================================
# СТАТИСТИКА ТОЛЬКО ПО TRAIN
# ============================================================


def purchase(moment: str, merchant: str, currency: str) -> dict:
    return raw_event(RAW_CLIENT, moment, {
        "type": "purchase", "amount": 700, "direction": "debit", "status": "approved",
        "merchant_name": merchant, "currency": currency,
    })


def train_tape() -> list[dict]:

    events = [purchase(f"2025-03-{day:02d}T10:00:00", "Magnum", "KZT") for day in range(1, 21)]
    events += [purchase("2025-04-01T10:00:00", "Small Shop", "USD")]

    return events


def fitted_weights(stage, tmp_path, val_currency: str) -> dict:
    """
    fit на train; val в той же выгрузке — со своей валютой и своим
    мерчантом, которых в train нет.
    """

    from src.tokenization.run import run_fit
    from src.tokenization.settings import VALUE_WEIGHTS_FILE, vocab_path

    prepare(stage, train_tape(), QUIET_SNAPSHOT, group="train")
    prepare(stage, [purchase(f"2026-02-{day:02d}T10:00:00", "Starbucks", val_currency) for day in range(1, 11)],
            QUIET_SNAPSHOT, group="val")

    config = tmp_path / "tokenizer.json"
    config.write_text(json.dumps({"numeric_min_values": 1, "numeric_min_clients": 1}))

    assert run_fit(SimpleNamespace(config=config)) == 0

    return json.loads(vocab_path(VALUE_WEIGHTS_FILE).read_text(encoding="utf-8"))


def test_statistics_come_from_train_only(stage, tmp_path):

    payload = fitted_weights(stage, tmp_path, "EUR")

    currency = {row["label"]: row["count"] for row in payload["keys"]["currency"]["values"]}
    merchant = {row["label"]: row["count"] for row in payload["keys"]["merchant_name"]["values"]}

    assert currency == {"KZT": 20, "USD": 1}
    assert merchant == {"magnum": 20, "small shop": 1}

    # val с другими значениями на статистику не влияет: fit его не
    # читает, и файл выходит тем же.
    again = fitted_weights(stage, tmp_path, "GBP")

    assert again == payload


def test_statistics_use_the_tokens_of_the_encoding(stage, tmp_path):
    """
    Значение в статистике записано теми же номерами, что даёт
    кодирование 04: иначе маскер не нашёл бы свой вес.
    """

    from collections import Counter

    from src.preprocessing.read import Group
    from src.preprocessing.settings import PreprocessingConfig
    from src.tokenization.encode import encode_event
    from src.tokenization.finalvocab import FrozenArtifacts

    payload = fitted_weights(stage, tmp_path, "EUR")

    table = {
        (item["key_id"], tuple(row["tokens"])): row["count"]
        for item in payload["keys"].values() for row in item["values"]
    }

    artifacts = FrozenArtifacts.load()
    cutoff = PreprocessingConfig.load(None).windows["train"].final_cutoff

    seen: Counter = Counter()

    for event in Group("train").history(RAW_CLIENT, cutoff).events:

        record = encode_event(artifacts, event, 16)

        starts = [index for index, position in enumerate(record.positions) if position == 0][1:]

        for number, start in enumerate(starts):
            end = starts[number + 1] if number + 1 < len(starts) else len(record.positions)
            identity = (record.key_ids[start], tuple(record.value_ids[start:end]))
            if identity in table:
                seen[identity] += 1

    assert seen
    assert all(seen[identity] == table[identity] for identity in seen)
    assert {artifacts.key_id("currency"), artifacts.key_id("merchant_name")} <= {key for key, _ in seen}


def test_val_uses_the_frozen_train_weights(stage, tmp_path):

    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig
    from src.mlm.inputs import Source
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    payload = fitted_weights(stage, tmp_path, "EUR")

    artifacts = FrozenArtifacts.load()

    encode_group(artifacts, "val", TokenizerConfig.load(None))
    build_dataset(artifacts, "val", DatasetConfig.load(None))

    source = Source("val")

    assert source.weights == load_value_weights()

    # Валюта val в train не встречалась: вес невиданного, а не частота
    # в val, где она у каждого события.
    currency = payload["keys"]["currency"]
    config = MaskingConfig()

    unseen = source.weights.probability(currency["key_id"], (artifacts.special("[UNK]"),), config)
    expected = min(
        config.max_value_probability,
        max(config.min_value_probability,
            config.value_probability * payload["scale"] * currency["key_weight"] * VALUE_WEIGHT_MAX),
    )

    assert unseen == pytest.approx(expected)
