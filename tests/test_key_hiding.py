from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from src.masking.apply import IGNORE, apply
from src.masking.choose import KEY, NONE, choose, values_of
from src.masking.settings import ConfigError, MaskingConfig
from src.mlm.model import pack

from tests import world
from tests.test_masking import rates, row_of


# ============================================================
# ИДЕЯ
# ============================================================
#
# Две настройки против подсказок, которые делают MLM задачей
# копирования:
#
#   key_context_corruption_probability (маска, по умолчанию 0.5)
#       значение ключа, выбранного механизмом key, в событии вне
#       целей независимо портится в [UNK] — целиком, без метки.
#       Цели и их розыгрыш от настройки не меняются ни на бит;
#   hide_event_keys (модель, эксперимент волны 4, по умолчанию
#       выключен)              у значений события под маской event
#       ключ во входе закрыт: энкодеры не различают такие события
#       по набору ключей. Какой ключ предсказывать, голова узнаёт
#       запросом — ключом цели.
# ============================================================


CPU = torch.device("cpu")

A, B, C = world.KEY_A, world.KEY_B, world.KEY_C


def client() -> world.Made:
    """
    Четыре события: два вне периода целей, два внутри.
    """

    return world.make(
        "k-1",
        [
            [(A, [10], False), (B, [11], False)],
            [(A, [12], False), (C, [13, 14], False)],
            [(A, [15], False), (B, [16], False)],
            [(A, [17], False), (C, [18, 19], False)],
        ],
        [(A, [20])],
    )


def val_row() -> dict:
    return dict(row_of(client()), target_event_mask=[False, False, True, True])


# ============================================================
# КЛЮЧ В КОНТЕКСТЕ
# ============================================================


def corruption(probability: float, **overrides) -> MaskingConfig:
    """
    Выбран каждый ключ; контекст портится с вероятностью probability.
    """

    return rates(**{"key_probability": 1.0, "key_context_corruption_probability": probability, **overrides})


def read(group: str, row: dict, config: MaskingConfig) -> dict:

    selection = choose(group, row, config)

    return apply(row["client_id"], row, selection.choices, world.MASK, world.UNK, selection.corrupted)


def context_values(row: dict) -> list:
    """
    Значения событий вне целей: первые два события клиента.
    """

    return [value for value in values_of(row, targets_only=False) if value.event < 2]


def pieces(masked: dict, value) -> list[int]:
    return masked["value_ids"][value.start:value.start + value.length]


def test_the_old_setting_is_gone_and_the_new_one_is_on_by_default():

    assert "key_hides_context" not in MaskingConfig().as_dict()
    assert not hasattr(MaskingConfig(), "key_hides_context")

    with pytest.raises(ConfigError, match="неизвестные ключи"):
        MaskingConfig.from_dict({"key_hides_context": True})

    assert MaskingConfig().key_context_corruption_probability == 0.5

    with pytest.raises(ConfigError, match="key_context_corruption_probability"):
        MaskingConfig.from_dict({"key_context_corruption_probability": 1.5})


def test_without_corruption_the_context_of_a_chosen_key_stays_as_it_was():

    row = val_row()
    masked = read("val", row, corruption(0.0))

    assert any(choice.reason == KEY for choice in choose("val", row, corruption(0.0)).choices)

    for value in context_values(row):
        assert pieces(masked, value) == row["value_ids"][value.start:value.start + value.length]


def test_full_corruption_spoils_every_past_value_and_keeps_the_targets():

    row = val_row()
    masked = read("val", row, corruption(1.0))

    context = context_values(row)

    assert {value.key_id for value in context} == {A, B, C}

    # Контекст: все куски в [UNK], без метки и без причины.
    for value in context:
        span = range(value.start, value.start + value.length)
        assert all(masked["value_ids"][index] == world.UNK for index in span)
        assert all(masked["labels"][index] == IGNORE for index in span)
        assert all(masked["reason"][index] == NONE for index in span)

    # Цели: [MASK] и метка — исходное значение, как без порчи.
    for value in values_of(row):
        span = range(value.start, value.start + value.length)
        assert all(masked["value_ids"][index] == world.MASK for index in span)
        assert [masked["labels"][index] for index in span] == row["value_ids"][value.start:value.start + value.length]
        assert all(masked["reason"][index] == KEY for index in span)


@pytest.mark.parametrize("seed", range(20))
def test_targets_and_their_draws_do_not_depend_on_corruption(seed: int):

    config = rates(seed=seed, key_probability=0.5, value_probability=0.3, unknown_probability=0.3)

    plain = choose("val", val_row(), dataclasses.replace(config, key_context_corruption_probability=0.0))
    spoiled = choose("val", val_row(), dataclasses.replace(config, key_context_corruption_probability=1.0))

    assert plain.choices == spoiled.choices
    assert plain.corrupted == ()


def test_a_bpe_value_is_corrupted_whole_or_not_at_all():

    row = val_row()

    outcomes = set()

    for seed in range(40):

        masked = read("val", row, corruption(0.5, seed=seed))

        for value in context_values(row):

            spoiled = [piece == world.UNK for piece in pieces(masked, value)]

            assert all(spoiled) or not any(spoiled), (seed, value)

            if value.length > 1:
                outcomes.add(all(spoiled))

    # Многокусковое значение ключа C бывает и испорчено, и цело.
    assert outcomes == {True, False}


def test_keys_not_chosen_are_not_touched():

    row = val_row()

    for seed in range(100):

        config = rates(seed=seed, key_probability=0.5, key_context_corruption_probability=1.0)
        selection = choose("val", row, config)

        chosen = {choice.value.key_id for choice in selection.choices if choice.reason == KEY}

        if chosen and chosen != {A, B, C}:
            break
    else:
        pytest.fail("не нашлось seed, где выбрана лишь часть ключей")

    masked = read("val", row, config)

    assert {value.key_id for value in selection.corrupted} == chosen

    for value in context_values(row):
        if value.key_id not in chosen:
            assert pieces(masked, value) == row["value_ids"][value.start:value.start + value.length]


def test_rereading_with_the_same_seed_is_identical_and_another_seed_may_differ():

    row = val_row()

    assert read("val", row, corruption(0.5, seed=3)) == read("val", row, corruption(0.5, seed=3))

    spoiled = {
        tuple(value.start for value in choose("val", row, corruption(0.5, seed=seed)).corrupted)
        for seed in range(20)
    }

    assert len(spoiled) > 1


def test_on_train_the_whole_context_is_the_target_period_and_nothing_is_corrupted():

    row = row_of(client())

    assert choose("train", row, corruption(1.0)).corrupted == ()


def test_train_val_and_test_read_the_context_through_one_implementation(stage):
    """
    Вход обучения и слой эмбеддингов этапов 06, 08 и 09 портят контекст
    одним и тем же кодом для всех трёх групп.
    """

    from src.embedding.inputs import Source as EmbeddingSource
    from src.mlm.inputs import Source

    made = dataclasses.replace(client(), targetable=(False, False, True, True))

    config = corruption(1.0)

    for group in ("train", "val", "test"):

        world.write_samples(group, [[made]])

        (read_back,) = list(Source(group, masking=config).clients())

        expected = read(group, row_of(made), config)

        assert read_back.value_ids.tolist() == expected["value_ids"], group
        assert read_back.labels.tolist() == expected["labels"], group
        assert read_back.reason == expected["reason"], group
        assert world.UNK in read_back.value_ids.tolist(), group

        layer = EmbeddingSource(group, masking=config).batch(0).model

        assert layer.value_ids[0, :read_back.n_tokens].tolist() == read_back.value_ids.tolist(), group


# ============================================================
# КЛЮЧИ СОБЫТИЯ ПОД МАСКОЙ EVENT
# ============================================================


def masked_event(second_key: int) -> world.Made:
    """
    Второе событие закрыто механизмом event целиком; у его второго
    значения ключ second_key.
    """

    made = world.make(
        "h-1",
        [
            [(A, [10], False), (B, [11], False)],
            [(A, [12], True), (second_key, [13], True)],
            [(A, [15], False)],
        ],
        [(A, [20])],
    )

    reason = ["event" if value == "value" else value for value in made.client.reason]

    return dataclasses.replace(made.client, reason=reason)


def encoded(model, one) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        return model._encode(pack([one], CPU))


def test_pack_marks_the_values_of_an_event_masked_event():

    data = pack([masked_event(B)], CPU)

    expected = np.asarray(masked_event(B).reason, dtype=object) == "event"

    assert torch.equal(data.event_masked, torch.as_tensor(expected))
    assert int(data.event_masked.sum()) == 2


def test_with_hidden_keys_the_encoders_do_not_see_the_keys_of_the_masked_event():

    model = world.model().eval()

    visible = [encoded(model, masked_event(key)) for key in (B, C)]

    assert not torch.equal(visible[0][2], visible[1][2]), "без закрытия ключ виден"

    model.hide_event_keys(world.MASK)

    hidden = [encoded(model, masked_event(key)) for key in (B, C)]

    for one, other in zip(*hidden):
        assert torch.equal(one, other)


def test_the_head_is_asked_for_the_key_of_its_target():

    model = world.model().eval()
    model.hide_event_keys(world.MASK)

    with torch.no_grad():
        losses = [model(pack([masked_event(key)], CPU)).loss for key in (B, C)]

    assert not torch.equal(losses[0], losses[1])


def test_readouts_have_no_masked_event_and_do_not_change():

    model = world.model().eval()
    plain = world.population("r")[1].client

    with torch.no_grad():
        before = model.readouts(pack([plain], CPU))
        model.hide_event_keys(world.MASK)
        after = model.readouts(pack([plain], CPU))

    for name in before:
        assert torch.equal(before[name], after[name]), name


def test_a_trained_model_loads_with_its_keys_hidden(stage):

    from src.mlm.settings import checkpoint_path
    from src.mlm.train import load_trained, train

    from tests.test_scheduler import many
    from tests.test_training_math import every_value, settle, tiny

    settle(stage, train_people=many())

    train(tiny(token_budget=6, hide_event_keys=True), epochs=1, max_steps=None, masking=every_value())

    model, state = load_trained(checkpoint_path(), CPU)

    assert state["config"]["hide_event_keys"] is True
    assert model.hidden_key == world.MASK
