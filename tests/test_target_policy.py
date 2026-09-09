"""
Состав MLM-целей.

Политика history выводит из задачи два множества полей:
timeline__event_type восстанавливается из состава самого
события, profile_snapshot__* дублируют as-of профиль. Оба
остаются ВХОДОМ: значения на месте, маски на них не ставятся,
головы полей никуда не деваются.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.data import ClientStore, FixedSplit
from src.model.history_batching import metadata_from_examples, prepare_history_batch
from src.model.metrics import STATUS_EXCLUDED, MetricAccumulator
from src.model.mlm_batching import build_targets
from src.model.targets import (
    HISTORY_EXCLUDES,
    POLICY_ALL,
    POLICY_HISTORY,
    exclude_patterns,
    excluded_fields,
)
from src.model.trainer import TrainConfig, Trainer
from src.tokenizer.dataset import collate
from src.tokenizer.masking import MODES, Masker, MaskingConfig

from tests.test_trainer import env, small_config  # noqa: F401


def targets_of(env, store, policy: str, mode: str, indices=range(4)):
    """
    Цели одного batch при заданной политике и режиме масок.
    """

    config = MaskingConfig(
        mode=mode, seed=777, exclude_fields=exclude_patterns(policy)
    )

    masker = Masker(env.vocab, config)

    examples = store.examples(indices)

    batch = collate(examples)

    # Без обрезки: позиции batch и history совпадают один
    # к одному, и «значение не тронуто» проверяется прямо.
    history = prepare_history_batch(
        batch, metadata_from_examples(examples), None, masker=masker, step=0
    )

    return masker, batch, history, build_targets(history, env.table)


def test_history_policy_removes_event_type_and_profile_snapshot_from_targets_but_not_from_input(
    env, tok_run
):

    store = ClientStore(env.root, "train", env.vocab_dir, max_clients=6)

    excluded = Masker(
        env.vocab, MaskingConfig(exclude_fields=HISTORY_EXCLUDES)
    ).excluded_names

    # Исключено ровно то, что заявлено, и это непустое множество.
    assert "timeline__event_type" in excluded
    assert any(name.startswith("profile_snapshot__") for name in excluded)
    assert all(
        name == "timeline__event_type" or name.startswith("profile_snapshot__")
        for name in excluded
    )

    excluded_ids = {env.vocab.key_entry(name).id for name in excluded}

    for mode in MODES:

        masker, batch, history, targets = targets_of(env, store, POLICY_HISTORY, mode)

        assert targets.n > 0, mode

        # --- целей по исключённым полям нет ----------------
        assert not np.isin(targets.key_ids, sorted(excluded_ids)).any(), mode

        # --- но значения на месте: это по-прежнему вход ----
        keys = np.asarray(batch.key_ids, dtype=np.int64)
        before = np.asarray(batch.value_ids, dtype=np.int64)
        after = np.asarray(history.tokens.value_ids, dtype=np.int64)

        touched = np.isin(keys, sorted(excluded_ids))

        assert np.array_equal(before[touched], after[touched]), mode

        # И хоть одна такая позиция в batch действительно есть.
        assert touched.sum() > 0, mode

        # --- политика all оставляет их целями --------------
        _, _, _, everything = targets_of(env, store, POLICY_ALL, mode)

        assert np.isin(everything.key_ids, sorted(excluded_ids)).any(), mode

        # Прежняя политика не тронута: маски те же, что были.
        assert everything.digest() != targets.digest(), mode

    # --- головы полей остаются -----------------------------
    trainer = Trainer(
        small_config(target_policy=POLICY_HISTORY),
        env.tokenizer,
        env.table,
        env.unigram,
        "cpu",
    )

    assert trainer.excluded == excluded

    for name in excluded:
        key_id = env.vocab.key_entry(name).id
        if env.table.trainable[key_id]:
            assert str(key_id) in trainer.head.heads

    # --- отчёт называет их исключёнными --------------------
    split = FixedSplit.build(
        name="val_client",
        store=ClientStore(env.root, "val_client", env.vocab_dir, max_clients=2),
        vocab=env.vocab,
        table=env.table,
        model_config=trainer.model_config,
        masking=trainer.config.masking(seed=trainer.config.val_seed),
        max_events=32,
        batch_size=2,
    )

    report = trainer.evaluate({"val_client": split}, exclude=trainer.excluded)

    item = report["val_client"]

    assert set(item["excluded_fields"]) <= excluded

    by_name = {field["field"]: field for field in item["fields"]}

    for name in item["excluded_fields"]:
        assert by_name[name]["status"] == STATUS_EXCLUDED
        assert by_name[name]["n_targets"] == 0

    # Ни одна исключённая голова не получила целей, поэтому
    # срез без них равен основному агрегату: это и есть смысл
    # политики, а не совпадение.
    assert item["subset"] is not None
    assert item["subset"]["field_balanced_ce"] == pytest.approx(
        item["field_balanced_ce"]
    )
    assert item["subset"]["n_fields_with_targets"] == item["n_fields_with_targets"]

    # --- опечатка в паттерне это ошибка --------------------
    with pytest.raises(ValueError):
        Masker(env.vocab, MaskingConfig(exclude_fields=("profile_snapshot",)))

    # --- конфиг прежних запусков не изменился --------------
    assert "exclude_fields" not in TrainConfig().masking().as_dict()
    assert TrainConfig(target_policy=POLICY_HISTORY).masking().as_dict()[
        "exclude_fields"
    ] == list(HISTORY_EXCLUDES)

    # excluded_fields по таблице отвечает тому же множеству.
    assert excluded_fields(env.table, POLICY_HISTORY) <= excluded
    assert excluded_fields(env.table, POLICY_ALL) == frozenset()
