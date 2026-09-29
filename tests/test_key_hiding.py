from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from src.masking.apply import IGNORE, apply
from src.masking.choose import NONE, choose
from src.masking.settings import MaskingConfig
from src.mlm.model import pack

from tests import world
from tests.test_masking import rates, row_of


# ============================================================
# ИДЕЯ
# ============================================================
#
# Две настройки против подсказок, которые делают MLM задачей
# копирования (эксперимент волны 4, по умолчанию выключены):
#
#   key_hides_context (маска)  ключ, выбранный механизмом key,
#       закрывается и в событиях вне периода целей — без метки.
#       Цели и их розыгрыш от настройки не меняются ни на бит, а
#       видимых значений выбранного ключа у клиента не остаётся;
#   hide_event_keys (модель)   у значений события под маской event
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


def test_by_default_the_context_stays_open():

    assert MaskingConfig().key_hides_context is False
    assert choose("val", val_row(), rates(key_probability=1.0)).hidden == ()


@pytest.mark.parametrize("seed", range(20))
def test_targets_and_their_draws_do_not_depend_on_the_setting(seed: int):

    config = rates(seed=seed, key_probability=0.5, value_probability=0.3, unknown_probability=0.3)

    plain = choose("val", val_row(), config)
    closed = choose("val", val_row(), dataclasses.replace(config, key_hides_context=True))

    assert plain.choices == closed.choices


def test_a_chosen_key_leaves_no_visible_value_anywhere():

    row = val_row()
    config = rates(key_probability=1.0, key_hides_context=True)

    selection = choose("val", row, config)
    masked = apply("k-1", row, selection.choices, world.MASK, world.UNK, selection.hidden)

    keys = np.asarray(row["key_ids"])
    values = np.asarray(masked["value_ids"])
    labels = np.asarray(masked["labels"])

    for key in (A, B, C):
        assert (values[keys == key] == world.MASK).all(), key

    # Контекст закрыт без метки и без причины, цели — с меткой.
    context = np.arange(len(keys)) < row["event_starts"][2]
    closed = context & (values == world.MASK)

    assert closed.any()
    assert (labels[closed] == IGNORE).all()
    assert all(masked["reason"][index] == NONE for index in np.nonzero(closed)[0])
    assert (labels[~context & (values == world.MASK)] != IGNORE).all()


def test_on_train_the_whole_context_is_the_target_period_and_nothing_changes():

    row = row_of(client())
    config = rates(key_probability=1.0)

    plain = choose("train", row, config)
    closed = choose("train", row, dataclasses.replace(config, key_hides_context=True))

    assert closed.hidden == ()
    assert apply("k-1", row, plain.choices, world.MASK, world.UNK) == apply(
        "k-1", row, closed.choices, world.MASK, world.UNK, closed.hidden
    )


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
