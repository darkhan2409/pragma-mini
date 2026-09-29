from __future__ import annotations

import pickle

import torch

from src.masking.settings import MaskingConfig
from src.mlm.inputs import Prefetch, Source
from src.mlm.model import hits, hits_in_pieces, pack
from src.mlm.settings import checkpoint_path
from src.mlm.train import train

from tests import world
from tests.test_checkpoint_resume import compare, read
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Ускорения обучения не имеют права менять результат:
#
#   - подготовка групп строк в отдельном процессе (Prefetch) даёт
#     тех же клиентов в том же порядке и с той же маской, а
#     обучение с ней — побитно тот же чекпойнт;
#   - счёт top-1/top-5 кусками без полных логитов равен счёту по
#     полным логитам.
#
# Процесс подготовки заново импортирует модули и о подменённых
# тестами каталогах не знает: источник обязан приезжать к нему
# своими путями. Если бы он читал глобалы, он пошёл бы в
# настоящий data/ — а совпадение чекпойнтов это сразу выдало бы.
# ============================================================


def test_training_with_a_loader_process_is_bit_for_bit_the_same(stage):
    """
    Два прогона на двух эпохах: подготовка в процессе обучения и в
    отдельном процессе. Чекпойнты равны целиком — веса, AdamW,
    расписание, история и генераторы: с dropout чужой розыгрыш из
    глобального генератора сдвинул бы маски dropout.
    """

    settle(stage, train_people=many(), dropout=0.2)

    masking = MaskingConfig(seed=5, value_probability=0.5, event_probability=0.2,
                            key_probability=0.2, unknown_probability=0.1)

    train(tiny(token_budget=6, early_stopping_patience=10), epochs=2, max_steps=None, masking=masking)
    inline = read(checkpoint_path())

    train(tiny(token_budget=6, early_stopping_patience=10, loader_workers=1), epochs=2,
          max_steps=None, masking=masking)
    prefetched = read(checkpoint_path())

    config = dict(inline["config"])
    config["loader_workers"] = 1

    assert prefetched["config"] == config
    prefetched["config"] = inline["config"]

    compare(inline, prefetched)


def test_prefetch_keeps_the_order_and_the_masks_with_several_processes(stage):
    """
    Три процесса и группы строк по кругу: порядок клиентов и маски
    те же, что у чтения подряд.
    """

    world.install(stage, {"train": [many(4), many(3), world.population("x"), many(2)]})

    masking = MaskingConfig(seed=9, value_probability=0.6)

    alone = list(Source("train", masking=masking).clients())
    together = list(Prefetch(Source("train", masking=masking), 3).clients())

    assert [client.client_id for client in alone] == [client.client_id for client in together]

    for left, right in zip(alone, together):
        assert left.reason == right.reason
        for name in ("key_ids", "value_ids", "labels", "positions", "event_time_log"):
            assert torch.equal(torch.as_tensor(getattr(left, name)), torch.as_tensor(getattr(right, name)))


def test_a_source_travels_by_its_paths_not_by_the_settings(stage):
    """
    Распакованный источник читает те же файлы, что упакованный, —
    даже если глобал каталога уже смотрит в другое место.
    """

    settle(stage, train_people=many())

    source = Source("train", masking=every_value())
    packed = pickle.dumps(source)

    import src.dataset.settings as dataset

    original = dataset.DATASET_DIR
    dataset.DATASET_DIR = stage / "elsewhere"

    try:
        restored = pickle.loads(packed)
    finally:
        dataset.DATASET_DIR = original

    assert restored.directory == source.directory
    assert [client.client_id for client in restored.clients()] == [
        client.client_id for client in source.clients()
    ]


def test_hits_in_pieces_equal_hits_on_the_full_logits(monkeypatch):
    """
    Куски по две цели вместо 2048: сумма по кускам та же, что по
    полным логитам.
    """

    import src.mlm.model as model_module

    monkeypatch.setattr(model_module, "TARGETS_PER_CHUNK", 2)

    model = world.model().eval()
    data = pack(world.clients(), CPU)

    with torch.no_grad():
        token, events, clients = model._encode(data)
        event_rows = events[data.target_event]
        client_rows = clients[data.target_client]
        targets = data.labels[data.target_token]
        full = model.head(token, event_rows, client_rows, model.embedding.weight)

    assert targets.numel() > 2, "нужно несколько кусков"

    pieces = hits_in_pieces(model.head, token, event_rows, client_rows, model.embedding.weight, targets)

    assert pieces == hits(full, targets)

    with torch.no_grad():
        out = model(data, logits=False)

    assert out.logits is None and out.hits == hits(full, targets)
