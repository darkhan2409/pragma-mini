"""
Отбор срезов обучения.

Схема «один пример на клиента» должна забирать именно
последний cutoff и не трогать валидацию: иначе две схемы
мерились бы разными линейками и сравнение потеряло бы смысл.
"""

from __future__ import annotations

import pytest

from src.model.data import CUTOFFS_ALL, CUTOFFS_LAST, ClientStore
from src.model.trainer import TrainConfig, store_for
from src.tokenizer.config import IncompatibleArtifactsError

from tests.test_trainer import env  # noqa: F401


def rows_by_client(store: ClientStore) -> dict[int, list]:
    out: dict[int, list] = {}
    for row in store.rows:
        out.setdefault(int(row["client_id"]), []).append(row["cutoff"])
    return out


def test_last_cutoff_keeps_one_latest_example_per_client_and_spares_validation(env):

    everything = TrainConfig(max_train_clients=None, max_val_clients=None)
    latest = TrainConfig(
        max_train_clients=None, max_val_clients=None, train_cutoffs=CUTOFFS_LAST
    )

    assert everything.train_cutoffs == CUTOFFS_ALL

    full = store_for(env, everything, "train")
    thin = store_for(env, latest, "train")

    by_full = rows_by_client(full)
    by_thin = rows_by_client(thin)

    # --- те же клиенты, по одному примеру ------------------
    assert set(by_thin) == set(by_full)
    assert all(len(value) == 1 for value in by_thin.values())

    # --- и это самый поздний срез каждого ------------------
    for client, cutoffs in by_full.items():
        assert by_thin[client] == [max(cutoffs)]

    # На фикстуре у клиента больше одного среза: иначе тест
    # проходил бы, ничего не проверяя.
    assert max(len(value) for value in by_full.values()) > 1

    assert len(thin) == len(by_thin)
    assert len(thin) < len(full)

    # --- ленты событий не урезаны --------------------------
    assert set(thin.events) == set(full.events)

    for client, events in thin.events.items():
        assert events.n_events == full.events[client].n_events

    # --- валидация не изменилась ---------------------------
    for split in ("val_client", "val_time"):
        assert [row["cutoff"] for row in store_for(env, latest, split).rows] == [
            row["cutoff"] for row in store_for(env, everything, split).rows
        ]


def test_cutoff_policy_travels_into_the_checkpoint_and_blocks_a_foreign_resume(env, tmp_path):

    from src.model.trainer import run_training

    config = TrainConfig(
        max_train_clients=4,
        max_val_clients=2,
        batch_size=2,
        max_events_per_history=32,
        max_steps=2,
        warmup_steps=1,
        eval_every=100,
        precision="float32",
        train_cutoffs=CUTOFFS_LAST,
    )

    out = tmp_path / "run"

    report = run_training(env, config, out, device="cpu", quiet=True)

    assert report["config"]["train_cutoffs"] == CUTOFFS_LAST

    # Продолжить прогон другой схемой срезов нельзя: это
    # другой эксперимент под тем же именем.
    other = TrainConfig(
        max_train_clients=4,
        max_val_clients=2,
        batch_size=2,
        max_events_per_history=32,
        max_steps=2,
        warmup_steps=1,
        eval_every=100,
        precision="float32",
    )

    with pytest.raises(IncompatibleArtifactsError, match="train_cutoffs"):
        run_training(
            env, other, tmp_path / "other", device="cpu", quiet=True,
            resume=out / "last.pt",
        )
