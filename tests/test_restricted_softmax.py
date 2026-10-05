from __future__ import annotations

import pytest
import torch

from src.mlm.model import candidate_table, hits_in_pieces, mlm_loss, pack
from src.mlm.settings import checkpoint_path
from src.mlm.train import load_trained, train
from src.tokenization.finalvocab import FrozenArtifacts

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Softmax по кандидатам ключа — та же кросс-энтропия, только по
# значениям своего ключа, и сглаживание меток — по ним же:
#
#   - если кандидаты — весь словарь, потеря побитно (до округления)
#     та же, что у обычной кросс-энтропии со сглаживанием;
#   - с настоящими множествами — та, что считается руками;
#   - top-1 выбирает только среди кандидатов;
#   - флаг едет в чекпойнте, и загрузка восстанавливает множества.
# ============================================================


def inputs(model, data):

    with torch.no_grad():
        token, events, clients = model._encode(data)

    return token, events[data.target_event], clients[data.target_client], data.labels[data.target_token]


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_all_candidates_is_the_plain_cross_entropy(smoothing: float):

    model = world.model().eval()
    data = pack(world.clients(), CPU)
    token, event, client, targets = inputs(model, data)

    everything = torch.ones((1, world.VOCAB), dtype=torch.bool)
    rows = torch.zeros_like(targets)

    with torch.no_grad():
        plain = mlm_loss(model.head, token, event, client, model.embedding.weight, targets, smoothing)
        restricted = mlm_loss(model.head, token, event, client, model.embedding.weight, targets,
                              smoothing, allowed=everything, rows=rows)

    assert torch.allclose(plain, restricted, atol=1e-6)


def test_restricted_loss_and_hits_follow_the_candidates(monkeypatch):

    import src.mlm.model as model_module

    monkeypatch.setattr(model_module, "TARGETS_PER_CHUNK", 2)

    model = world.model().eval()
    data = pack(world.clients(), CPU)
    token, event, client, targets = inputs(model, data)

    # Кандидаты: у каждой цели — её значение, соседнее и ещё одно.
    allowed = torch.zeros((targets.numel(), world.VOCAB), dtype=torch.bool)
    allowed[torch.arange(targets.numel()), targets] = True
    allowed[torch.arange(targets.numel()), (targets + 1) % world.VOCAB] = True
    allowed[:, 0] = True
    rows = torch.arange(targets.numel())

    with torch.no_grad():
        logits = model.head(token, event, client, model.embedding.weight)
        loss = mlm_loss(model.head, token, event, client, model.embedding.weight, targets, 0.1,
                        allowed=allowed, rows=rows)
        found = hits_in_pieces(model.head, token, event, client, model.embedding.weight, targets,
                               allowed=allowed, rows=rows)

    logp = logits.masked_fill(~allowed, float("-inf")).log_softmax(dim=-1)
    nll = -logp[torch.arange(targets.numel()), targets]
    spread = -logp.masked_fill(~allowed, 0.0).sum(dim=-1) / allowed.sum(dim=-1)

    assert torch.allclose(loss, (0.9 * nll + 0.1 * spread).mean(), atol=1e-6)

    masked = logits.masked_fill(~allowed, float("-inf"))
    top = masked.topk(5, dim=-1).indices
    assert allowed.gather(1, top[:, :1]).all(), "первый ответ обязан быть кандидатом"
    assert found[0] == int((top[:, 0] == targets).sum())

    assert model_module.TARGETS_PER_CHUNK == 2 and targets.numel() > 2, "нужно несколько кусков"


def test_candidates_come_from_the_vocabulary_of_each_key(stage):

    settle(stage, train_people=many())

    artifacts = FrozenArtifacts.load()
    key_row, allowed = candidate_table(artifacts)

    for key, token_id in artifacts.keys.items():
        row = int(key_row[token_id])
        ids = set(torch.nonzero(allowed[row]).flatten().tolist())
        assert ids == set(artifacts.values[key].values()) | {artifacts.special("[UNK]")}, key


def test_the_flag_travels_in_the_checkpoint(stage, monkeypatch):
    """
    Флаг едет в конфиге чекпойнта, и загрузка восстанавливает
    множества. Значения синтетического мира со словарём ключей не
    согласованы, поэтому кандидаты здесь — весь словарь у каждого
    ключа: проверяется перенос флага, а не множества (их — тест выше).
    """

    import src.mlm.model as model_module

    def everything(artifacts):
        key_row, allowed = candidate_table(artifacts)
        return key_row, torch.ones_like(allowed)

    monkeypatch.setattr(model_module, "candidate_table", everything)

    settle(stage, train_people=many())

    train(tiny(token_budget=6, restricted_softmax=True), epochs=1, max_steps=None, masking=every_value())

    model, state = load_trained(checkpoint_path(), CPU)

    assert state["config"]["restricted_softmax"] is True
    assert model.allowed is not None and model.allowed.dtype == torch.bool
    assert "allowed" not in state["model_state_dict"], "кандидаты не веса и в чекпойнт не пишутся"
