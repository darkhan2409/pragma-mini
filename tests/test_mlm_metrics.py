from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.mlm.model import hits, pack
from src.mlm.train import Scores

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# Точность MLM — Top-1 и Top-5 — только по настоящим целям
# (метка != -100). Проверяется:
#
#   счёт        точные логиты с известным ответом;
#   знаменатель метка -100 и позиции контекста в него не входят;
#   ноль целей  доли нет (None), деления на ноль нет;
#   сумма       доля эпохи — по всем целям, а не среднее батчей;
#   модель      в проходе целей ровно столько, сколько меток != -100;
#   обучение    строка эпохи и история несут train и val точность.
# ============================================================


CPU = torch.device("cpu")


def logits(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


# Словарь из 7 значений. Порядок логитов у каждой строки — по убыванию:
#   строка 0: 3 > 1 > 0 > 4 > 2 > 5 > 6
#   строка 1: 6 > 5 > 4 > 3 > 2 > 1 > 0
#   строка 2: 0 > 1 > 2 > 3 > 4 > 5 > 6
#   строка 3: 2 > 0 > 1 > 3 > 4 > 5 > 6
LOGITS = logits(
    [0.5, 0.6, 0.2, 0.9, 0.4, 0.1, 0.0],
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    [0.8, 0.7, 0.9, 0.3, 0.2, 0.1, 0.0],
)


def test_top1_and_top5_on_exact_logits():
    """
    Цели 3, 0, 6, 2: первым угаданы строки 0 и 3; у строк 1 и 2 цель
    седьмая — мимо и пятёрки. Итог: top1 = 2, top5 = 2.
    """

    assert hits(LOGITS, torch.tensor([3, 0, 6, 2]), 5) == (2, 2)

    # Цели 1, 4, 2, 5: первым — никто; в пятёрку входят строка 0 (1 —
    # второй), строка 1 (4 — третий), строка 2 (2 — третий), строка 3
    # (5 — шестой, нет).
    assert hits(LOGITS, torch.tensor([1, 4, 2, 5]), 5) == (0, 3)

    # Граница пятёрки: у строки 3 цель 4 стоит ровно пятой — входит;
    # у строки 1 цель 6 первая. Строки 0 и 2 — седьмые.
    assert hits(LOGITS, torch.tensor([6, 6, 6, 4]), 5) == (1, 2)


def test_ignored_labels_never_enter_the_count():

    assert hits(LOGITS, torch.tensor([3, -100, -100, 2]), 5) == (2, 2)
    assert hits(LOGITS, torch.tensor([-100, -100, -100, -100]), 5) == (0, 0)


def test_no_targets_is_no_share_not_zero_or_division():

    assert hits(LOGITS[:0], torch.tensor([], dtype=torch.long), 5) == (0, 0)

    scores = Scores()

    scores.add(SimpleNamespace(count=0, logits=LOGITS[:0], targets=torch.tensor([], dtype=torch.long),
                               loss=torch.tensor(0.0)))

    assert scores.targets == 0
    assert scores.loss is None and scores.top1_accuracy is None and scores.top5_accuracy is None


def test_epoch_share_is_over_all_targets_not_a_mean_of_batches():
    """
    Батч из одной угаданной цели и батч из трёх неугаданных: доля
    1/4, а не среднее (1 + 0) / 2.
    """

    scores = Scores()

    scores.add(SimpleNamespace(count=1, logits=LOGITS[:1], targets=torch.tensor([3]), loss=torch.tensor(0.5)))
    scores.add(SimpleNamespace(count=3, logits=LOGITS[1:], targets=torch.tensor([0, 6, 5]), loss=torch.tensor(2.0)))

    assert scores.targets == 4
    assert scores.top1_accuracy == pytest.approx(0.25)
    assert scores.top5_accuracy == pytest.approx(0.25)
    assert scores.loss == pytest.approx((0.5 + 3 * 2.0) / 4)


def test_model_counts_only_real_targets(clients):
    """
    В проходе модели строк логитов ровно столько, сколько меток
    != -100: контекст, незакрытые токены и клиенты без целей в
    знаменатель не входят.
    """

    model = world.model()
    model.eval()

    labelled = sum(int((client.labels != -100).sum()) for client in clients)
    tokens = sum(client.n_tokens for client in clients)

    assert 0 < labelled < tokens

    with torch.no_grad():
        out = model(pack(clients, CPU))

    scores = Scores()
    scores.add(out)

    assert out.count == scores.targets == labelled
    assert 0 <= scores.top1 <= scores.top5 <= labelled


def test_epoch_line_and_history_carry_train_and_val_accuracy(stage, capsys):

    from src.mlm.settings import checkpoint_path
    from src.mlm.train import train

    from tests.test_scheduler import many
    from tests.test_training_math import every_value, settle, tiny

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=2, max_steps=None, masking=every_value())

    printed = [line for line in capsys.readouterr().out.splitlines() if " train_loss=" in line]

    assert len(printed) == 2

    for name in ("train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5", "lr"):
        assert all(f" {name}=" in line for line in printed), name

    history = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)["history"]

    assert [item["epoch"] for item in history] == [1, 2]

    for item in history:
        for part in ("train", "val"):
            assert item[part]["targets"] > 0
            assert 0.0 <= item[part]["top1"] <= item[part]["top5"] <= 1.0
        assert item["learning_rate"] > 0
