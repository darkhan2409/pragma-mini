from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import src.mlm.model as model_module
from src.mlm.inputs import IGNORE
from src.mlm.train import Detail, events_bin, target_losses


# ============================================================
# ИДЕЯ
# ============================================================
#
# Разбивка val по целям (Detail) группирует цели целыми кодами и
# только на позициях целей, а target_losses переносит на CPU два
# массива на micro-batch, а не два на кусок. Итог обязан совпасть с
# прежним кодом (old_*, дословно) до бита: те же подмножества целей
# в том же порядке дают те же суммы.
# ============================================================


def old_target_losses(logits, targets) -> tuple[np.ndarray, np.ndarray]:

    nll, first = [], []

    for piece, labels in zip(
        logits.detach().split(model_module.TARGETS_PER_CHUNK), targets.split(model_module.TARGETS_PER_CHUNK)
    ):
        scores = piece.float()
        nll.append(F.cross_entropy(scores, labels, reduction="none").cpu().numpy())
        first.append((scores.argmax(dim=-1) == labels).cpu().numpy())

    return np.concatenate(nll).astype(np.float64), np.concatenate(first)


@dataclass
class OldDetail:

    nll_sum: float = 0.0
    targets: int = 0
    groups: dict = field(default_factory=dict)

    def add(self, out, clients: list, key_names: np.ndarray) -> None:

        if out.count == 0:
            return

        nll, first = old_target_losses(out.logits, out.targets)

        chosen = [client.labels != IGNORE for client in clients]

        labels = {
            "reason": np.concatenate(
                [np.asarray(client.reason, dtype=object)[mask] for client, mask in zip(clients, chosen)]
            ),
            "events": np.concatenate(
                [
                    np.full(int(mask.sum()), events_bin(client.n_events), dtype=object)
                    for client, mask in zip(clients, chosen)
                ]
            ),
            "key": key_names[
                np.concatenate([client.key_ids[mask] for client, mask in zip(clients, chosen)])
            ],
        }

        for kind, names in labels.items():

            if len(names) != len(nll):
                raise ValueError("порядок целей pack разошёлся с клиентами")

            table = self.groups.setdefault(kind, {})

            for name in np.unique(names):
                mask = names == name
                row = table.setdefault(str(name), [0, 0.0, 0])
                row[0] += int(mask.sum())
                row[1] += float(nll[mask].sum())
                row[2] += int(first[mask].sum())

        self.nll_sum += float(nll.sum())
        self.targets += len(nll)

    summary = Detail.summary


# ------------------------------------------------------------
# синтетические проходы
# ------------------------------------------------------------


VOCAB = 50
REASONS = ("event", "key", "value")

# Ключи — номера словаря 1..9; у номера 0 имени нет.
KEY_NAMES = np.array([None] + [f"k{number}" for number in range(1, VOCAB)], dtype=object)


def client(generator: np.random.Generator, tokens: int, n_events: int, keys: tuple[int, ...]):

    labels = np.where(generator.random(tokens) < 0.4, generator.integers(0, VOCAB, tokens), IGNORE)
    reason = [str(generator.choice(REASONS)) if label != IGNORE else "none" for label in labels]

    return SimpleNamespace(
        labels=labels, reason=reason, key_ids=generator.choice(keys, tokens), n_events=n_events,
    )


def passes(seed: int = 3):
    """
    Micro-batch'и: все пять корзин длины, три причины, ключ только в
    одном batch, batch без целей, логиты bf16 с равными значениями.
    """

    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)

    batches = [
        [client(generator, 30, 50, (1, 2, 3)), client(generator, 21, 150, (2, 3))],
        [client(generator, 40, 500, (1, 4)), client(generator, 9, 1500, (5,)), client(generator, 17, 5000, (1, 2))],
        [client(generator, 33, 50, (3, 6)), client(generator, 25, 3000, (1, 2, 3))],
    ]

    out = []

    for clients in batches:
        labels = np.concatenate([item.labels for item in clients])
        targets = torch.as_tensor(labels[labels != IGNORE])
        # bf16 — как под autocast: равные логиты нередки.
        logits = torch.randn(len(targets), VOCAB).to(torch.bfloat16)
        out.append((SimpleNamespace(logits=logits, targets=targets, count=len(targets)), clients))

    silent = SimpleNamespace(labels=np.full(5, IGNORE), reason=["none"] * 5, key_ids=np.ones(5, np.int64), n_events=7)
    out.append((SimpleNamespace(logits=torch.zeros(0, VOCAB), targets=torch.zeros(0, dtype=torch.long), count=0),
                [silent]))

    return out


@pytest.mark.parametrize("chunk", [2, 7, 2048])
def test_detail_is_the_old_detail(monkeypatch, chunk: int):

    monkeypatch.setattr(model_module, "TARGETS_PER_CHUNK", chunk)

    new, old = Detail(), OldDetail()

    for out, clients in passes():
        new.add(out, clients, KEY_NAMES)
        old.add(out, clients, KEY_NAMES)

    mine, theirs = new.summary(), old.summary()

    assert set(mine["events"]) == {"0-100", "100-300", "300-1000", "1000-3000", "3000+"}
    assert set(mine["reason"]) == set(REASONS)
    assert mine == theirs
    assert json.dumps(mine, sort_keys=True) == json.dumps(theirs, sort_keys=True)


@pytest.mark.parametrize("chunk", [2, 2048])
def test_target_losses_are_the_old_ones(monkeypatch, chunk: int):

    monkeypatch.setattr(model_module, "TARGETS_PER_CHUNK", chunk)

    for out, _ in passes()[:-1]:
        for one, two in zip(target_losses(out.logits, out.targets), old_target_losses(out.logits, out.targets)):
            assert one.dtype == two.dtype and np.array_equal(one, two)


def test_a_target_key_without_a_name_is_an_error():

    out, clients = passes()[0]
    clients[0].key_ids[:] = 0

    with pytest.raises(ValueError, match="без имени"):
        Detail().add(out, clients, KEY_NAMES)


def test_a_feature_count_unlike_the_targets_is_an_error():

    out, clients = passes()[0]

    with pytest.raises(ValueError, match="разошёлся"):
        Detail().add(out, clients[:1], KEY_NAMES)
