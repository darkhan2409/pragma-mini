from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.downstream.finetune import ZERO_MASK
from src.masking.choose import EVENT, VALUE
from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source
from src.mlm.model import pack
from src.mlm.settings import MlmConfig, checkpoint_path
from src.mlm.train import load_trained, train
from src.mlm.varlen import autocast

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# E6 (hide_event_keys): у значений события, закрытого механизмом
# event, во входе энкодера события закрыт и ключ — [MASK]. Иначе
# набор видимых ключей выдаёт тип события и схему его полей.
# Какой ключ предсказывать, голова узнаёт из запроса — эмбеддинга
# исходного ключа цели. Проверяется, что:
#
#   - во вход идут ровно ключи с [MASK] на местах причины event, а
#     голова и data.key_ids видят исходные ключи;
#   - без флага проход тот же до бита, что и прежде;
#   - чекпойнт с флагом грузится со скрытием;
#   - без целей (ZERO_MASK: embed, LoRA) флаг ни на что не влияет;
#   - пути sdpa и flash видят одни и те же скрытые ключи (CUDA).
# ============================================================


def closed(client):
    """
    Тот же клиент, но события, у которых закрыто каждое значение,
    закрыты механизмом event: причина event вместо value. Маркер
    [EVT] остаётся без причины, как у настоящей маски.
    """

    reason = list(client.reason)

    for start, length in zip(client.event_starts.tolist(), client.event_lengths.tolist()):

        span = range(start + 1, start + length)

        if len(span) and all(reason[index] == VALUE for index in span):
            for index in span:
                reason[index] = EVENT

    return replace(client, reason=reason)


def event_clients() -> list:
    """
    Мир с событиями под маской event и целями механизма value рядом.
    """

    clients = [closed(client) for client in world.clients()]

    reasons = {reason for client in clients for reason in client.reason}

    assert {EVENT, VALUE} <= reasons

    return clients


def hidden(data) -> torch.Tensor:
    """
    Ключи входа, посчитанные здесь независимо от модели.
    """

    return torch.where(data.event_masked, torch.full_like(data.key_ids, world.MASK), data.key_ids)


def test_event_masked_keys_are_masked_in_the_input(stage):

    # Причину event пишет настоящая маска при чтении набора.
    settle(stage, train_people=world.population("s"))

    masking = MaskingConfig(
        seed=11,
        value_probability=0.5,
        event_probability=0.5,
        key_probability=0.0,
        unknown_probability=0.0,
        informativeness_weighted_masking=False,
    )

    drawn = list(Source("train", masking=masking).clients())
    data = pack(drawn, CPU)

    expected = [reason == EVENT for client in drawn for reason in client.reason]

    assert data.event_masked.dtype == torch.bool
    assert data.event_masked.tolist() == expected
    assert any(expected) and not all(expected)

    # Скрытие в модели — на клиентах с обеими причинами.
    data = pack(event_clients(), CPU)
    original = data.key_ids.clone()

    built = world.model()
    built.eval()

    built.hide_event_keys(world.MASK)

    assert built.hidden_key == world.MASK

    keys = built._event_keys(data)

    assert torch.equal(keys, hidden(data))
    assert torch.equal(keys[data.event_masked], torch.full_like(keys[data.event_masked], world.MASK))
    assert torch.equal(keys[~data.event_masked], original[~data.event_masked])

    with torch.no_grad():
        on = built.client_embeddings(data)

    # Ни data.key_ids (их читают RecentTypes и голова), ни маркеры не
    # тронуты: замена — новый тензор.
    assert torch.equal(data.key_ids, original)

    # Энкодер видит ровно скрытые ключи: тот же вход без флага с
    # закрытыми ключами в самих данных даёт тот же ответ до бита.
    plain = world.model()
    plain.eval()

    with torch.no_grad():
        same = plain.client_embeddings(replace(data, key_ids=hidden(data)))
        open_keys = plain.client_embeddings(data)

    assert torch.equal(on, same)
    assert not torch.equal(on, open_keys)

    # Голова читает исходный ключ цели: подмена ключа под маской event
    # энкодер не видит, а логиты меняет.
    target = int(data.target_token[data.event_masked[data.target_token]][0])
    keys = data.key_ids.clone()
    keys[target] = world.KEY_C if int(keys[target]) != world.KEY_C else world.KEY_B
    swapped = replace(data, key_ids=keys)

    # Всё под no_grad, как и on: с графом и без него CPU считает
    # внимание разными ядрами, и последний бит разный сам по себе.
    with torch.no_grad():
        first = built(data)
        second = built(swapped)
        unseen = built.client_embeddings(swapped)

    assert torch.equal(unseen, on)
    assert not torch.equal(first.logits, second.logits)


def test_unknown_value_of_a_masked_event_hides_its_key_too():
    """
    Значение события под маской event, ушедшее в [UNK], причины не
    получает (masking.apply), но ключ его закрыт: скрытие — по событию,
    а не по токену. Маркер [EVT] остаётся маркером.
    """

    made = world.make(
        "u-1",
        [
            # Событие под маской event: key_a закрыт, key_b ушёл в [UNK],
            # key_c из двух кусков закрыт.
            [(world.KEY_A, [10], True), (world.KEY_B, [11], False), (world.KEY_C, [12, 13], True)],
            # Соседнее событие с целью value — ключи видны.
            [(world.KEY_A, [14], True), (world.KEY_B, [15], False)],
            # Событие без целей.
            [(world.KEY_C, [16], False)],
        ],
        [(world.KEY_A, [20])],
    )

    client = made.client
    reason = list(client.reason)
    value_ids = client.value_ids.copy()

    # Первое событие: токены 0..4, маркер 0; [UNK] у key_b (токен 2).
    for index in (1, 3, 4):
        reason[index] = EVENT
    value_ids[2] = world.UNK

    client = replace(client, reason=reason, value_ids=value_ids)

    data = pack([client], CPU)

    assert data.event_masked.tolist() == [False, True, True, True, True, False, False, False, False, False]

    built = world.model()
    built.hide_event_keys(world.MASK)

    keys = built._event_keys(data).tolist()

    assert keys[:5] == [world.EVT] + [world.MASK] * 4
    assert keys[5:] == data.key_ids[5:].tolist()
    assert keys[5] == world.EVT and keys[8] == world.EVT


def test_train_hides_keys_when_asked(stage, monkeypatch):
    """
    train включает скрытие по флагу конфига, а без флага — нет.
    """

    from src.mlm.model import Model

    calls: list[int] = []
    original = Model.hide_event_keys

    def recorded(self, hidden: int) -> None:
        calls.append(int(hidden))
        original(self, hidden)

    monkeypatch.setattr(Model, "hide_event_keys", recorded)

    settle(stage, train_people=many())

    train(tiny(), epochs=1, max_steps=1, masking=every_value())

    assert calls == []

    train(tiny(hide_event_keys=True), epochs=1, max_steps=1, masking=every_value(),
          directory=stage / "hidden")

    assert calls == [world.MASK]


def test_hide_keys_off_is_the_previous_forward():
    """
    По умолчанию флаг выключен, и причина event на проход не влияет:
    логиты, потери, вектор клиента и градиенты те же до бита, что у
    тех же клиентов без единой причины event.
    """

    assert MlmConfig().hide_event_keys is False
    assert MlmConfig.from_dict({}).hide_event_keys is False

    with_event = event_clients()
    without = [replace(client, reason=[VALUE if reason == EVENT else reason for reason in client.reason])
               for client in with_event]

    outputs = []

    for clients in (with_event, without):

        built = world.model(events_per_chunk=2)
        built.eval()

        assert built.hidden_key is None

        data = pack(clients, CPU)
        predicted = built(data)
        predicted.loss.backward()

        outputs.append((
            predicted.logits.detach(), predicted.loss.detach(),
            built.client_embeddings(data).detach(),
            [parameter.grad.clone() for parameter in built.parameters() if parameter.grad is not None],
        ))

    (logits, loss, usr, grads), (logits_, loss_, usr_, grads_) = outputs

    assert torch.equal(logits, logits_)
    assert torch.equal(loss, loss_)
    assert torch.equal(usr, usr_)
    assert len(grads) == len(grads_) and all(torch.equal(a, b) for a, b in zip(grads, grads_))


def test_load_trained_restores_hide_keys(stage):

    settle(stage, train_people=many())

    train(tiny(hide_event_keys=True), epochs=1, max_steps=None, masking=every_value())

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)

    assert state["config"]["hide_event_keys"] is True

    model, _ = load_trained(checkpoint_path(), CPU)

    assert model.hidden_key == world.MASK

    # Чекпойнт без ключа — прежнее поведение: ключи видны.
    state["config"].pop("hide_event_keys")
    torch.save(state, checkpoint_path())

    model, _ = load_trained(checkpoint_path(), CPU)

    assert model.hidden_key is None


def test_zero_mask_input_is_unaffected(stage):
    """
    Маска без целей (embed, LoRA): причины event нет, и с флагом
    вектор клиента, readouts и проход те же до бита.
    """

    settle(stage, train_people=world.population("z"))

    drawn = list(Source("train", masking=ZERO_MASK).clients())
    data = pack(drawn, CPU)

    assert not bool(data.event_masked.any())

    outputs = []

    for hide in (False, True):

        built = world.model()
        built.eval()

        if hide:
            built.hide_event_keys(world.MASK)

        with torch.no_grad():
            predicted = built(data)
            outputs.append((
                built.client_embeddings(data), built.readouts(data),
                predicted.logits, predicted.loss,
            ))

    (usr, readouts, logits, loss), (usr_, readouts_, logits_, loss_) = outputs

    assert torch.equal(usr, usr_)
    assert all(torch.equal(readouts[name], readouts_[name]) for name in readouts)
    assert torch.equal(logits, logits_)
    assert torch.equal(loss, loss_) or bool(torch.isnan(loss) & torch.isnan(loss_))


@pytest.mark.cuda
@pytest.mark.skipif(not world.flash_ready(), reason="нужны CUDA и библиотека flash-attn")
def test_flash_and_sdpa_agree_with_hidden_keys():
    """
    Весь проход двумя путями со скрытыми ключами — в допуске
    test_cuda_flash.test_flash_and_buckets_agree_on_the_whole_pass.
    Плоский путь flash и корзины sdpa берут ключи из одного места.
    """

    device = torch.device("cuda")

    built = world.model(attention="flash").to(device)
    built.eval()
    built.hide_event_keys(world.MASK)

    data = pack(event_clients(), device)

    assert data.event_masked.device.type == "cuda"
    assert bool(data.event_masked.any())

    with torch.no_grad():

        buckets = built(data)
        usr_buckets = built.client_embeddings(data)

        with autocast(device):
            flash = built(data)
            usr_flash = built.client_embeddings(data)

            # Без скрытия вектор клиента на flash другой: плоский путь
            # действительно читает скрытые ключи, а не исходные.
            built.hidden_key = None
            usr_visible = built.client_embeddings(data)

    assert torch.allclose(flash.logits.float(), buckets.logits.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(usr_flash.float(), usr_buckets.float(), atol=5e-2, rtol=5e-2)
    assert not torch.equal(usr_visible, usr_flash)
