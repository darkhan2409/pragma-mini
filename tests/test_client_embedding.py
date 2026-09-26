from __future__ import annotations

import numpy as np
import torch

from src.mlm.diagnostics import VARIANTS, attention_to_events, cosine_l2, embeddings, variant
from src.mlm.model import pack

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# client_embedding — [USR] после последнего блока энкодера истории:
# ровно тот вектор клиента, что получает голова, и не выход
# энкодера анкеты. Диагностика строит варианты клиента только из
# его же данных и снимает внимание [USR] так, что строка внимания
# воспроизводит настоящий выход блока.
# ============================================================


CPU = torch.device("cpu")


def busy(clients) -> list:
    return [client for client in clients if client.n_events >= 3]


def test_embedding_is_one_vector_per_client_of_model_width(clients):

    model = world.model().eval()

    with torch.no_grad():
        found = model.client_embeddings(pack(clients, CPU))

    assert found.shape == (len(clients), world.DIM)


def test_embedding_of_the_final_architecture_has_width_128(clients):

    from tests.test_architecture import build

    with torch.no_grad():
        found = build(128).eval().client_embeddings(pack(clients, CPU))

    assert found.shape == (len(clients), 128)


def test_embedding_is_what_the_head_receives_after_the_history(clients):
    """
    Голова получает вектор клиента каждой цели; он совпадает с
    client_embedding владельца цели. Вектор энкодера анкеты — другой.
    """

    model = world.model().eval()
    data = pack(clients, CPU)

    received = []
    handle = model.head.register_forward_pre_hook(lambda module, args: received.append(args[2]))

    with torch.no_grad():
        model(data)
        found = model.client_embeddings(data)
        profile = model._profiles(data)

    handle.remove()

    assert torch.equal(received[0], found[data.target_client])
    assert not torch.allclose(found, profile, atol=1e-3)


def test_variants_use_only_the_clients_own_data(clients):

    rng = np.random.default_rng(0)
    client = busy(clients)[0]

    empty = variant(client, "no_events", rng)
    assert empty.n_events == 0 and empty.n_tokens == 0
    assert np.array_equal(empty.profile_value_ids, client.profile_value_ids)

    half = variant(client, "partial_50", rng)
    kept = half.n_events
    assert kept == int(np.ceil(client.n_events / 2))
    assert np.array_equal(half.event_time_log, client.event_time_log[-kept:])
    assert half.value_ids.tolist() == client.value_ids[int(client.event_starts[-kept]):].tolist()

    mixed = variant(client, "shuffled_content", rng)
    assert np.array_equal(mixed.event_time_log, client.event_time_log)
    assert sorted(mixed.value_ids.tolist()) == sorted(client.value_ids.tolist())
    assert mixed.event_starts.tolist() == np.concatenate([[0], np.cumsum(mixed.event_lengths)[:-1]]).tolist()

    alone = variant(client, "no_profile", rng)
    assert alone.profile_n_tokens == 1 and alone.profile_key_ids[0] == client.profile_key_ids[0]


def test_order_of_rows_alone_does_not_move_the_embedding(clients):
    """
    Позиция в истории — время, а не номер строки: переставленные
    строки с теми же моментами дают тот же вектор. Переставленное по
    моментам содержимое — уже другой вектор.
    """

    model = world.model().eval()
    rng = np.random.default_rng(1)
    chosen = busy(clients)

    full = embeddings(model, chosen, CPU)
    order = embeddings(model, [variant(c, "shuffled_order", rng) for c in chosen], CPU)
    content = embeddings(model, [variant(c, "shuffled_content", rng) for c in chosen], CPU)
    empty = embeddings(model, [variant(c, "no_events", rng) for c in chosen], CPU)

    assert torch.allclose(full, order, atol=1e-5)

    cos_content, _ = cosine_l2(full, content)
    cos_empty, _ = cosine_l2(full, empty)

    assert (cos_empty < 1 - 1e-4).all()
    assert (cos_content < 1 - 1e-6).any()


def test_every_variant_passes_through_the_model(clients):

    model = world.model().eval()
    rng = np.random.default_rng(2)

    for name in VARIANTS:
        found = embeddings(model, [variant(c, name, rng) for c in busy(clients)], CPU)
        assert torch.isfinite(found).all(), name


def test_attention_row_reproduces_the_real_attention_output(clients):
    """
    Строка softmax(QKᵀ/√d) запроса [USR] даёт тот же выход внимания,
    что настоящий проход; массы на [USR] и на события в сумме 1.
    """

    model = world.model().eval()

    for client in busy(clients):

        found = attention_to_events(model, client, CPU)

        assert found["row_error"] < 1e-5

        layers = len(model.history.layers)
        heads = model.history.layers[0].heads

        assert np.asarray(found["usr"]).shape == (layers, heads)

        total = np.asarray(found["usr"]) + np.asarray(found["events"])

        assert np.allclose(total, 1.0, atol=1e-5)
