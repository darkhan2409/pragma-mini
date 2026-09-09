"""
Схема масок по идентичности примера.

Прежняя схема тянет один поток чисел по всему batch: маска
примера зависит от того, с кем он попал в batch и каким по
счёту оказался. Значит смена eval_batch_size меняет задачу,
а не только скорость.

Схема example засевает поток парой (client_id, cutoff) и
разыгрывает маску примера отдельно. Прежняя схема при этом
обязана остаться прежней побайтово: на ней сняты цифры всех
существующих запусков.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.data import ClientStore
from src.model.history_batching import metadata_from_examples, prepare_history_batch
from src.tokenizer.dataset import collate
from src.tokenizer.masking import (
    MODE_COMBINED,
    MODE_FIELD_BALANCED,
    MODES,
    SCHEME_BATCH,
    SCHEME_EXAMPLE,
    Masker,
    MaskingConfig,
)

from tests.test_trainer import env  # noqa: F401


MAX_EVENTS = 96


def masks_by_example(env, store, indices, mode, scheme, step=0):
    """
    Замаскированные позиции каждого примера, адресованные
    внутри самого примера.

    Ключ это идентичность примера, а не его номер: именно она
    и должна определять маску.
    """

    examples = store.examples(indices)

    masker = Masker(env.vocab, MaskingConfig(mode=mode, seed=31337, scheme=scheme))

    history = prepare_history_batch(
        collate(examples),
        metadata_from_examples(examples),
        MAX_EVENTS,
        masker=masker,
        step=step,
    )

    owner = np.asarray(history.tokens.example_ids, dtype=np.int64)

    out = {}

    for position, example in enumerate(examples):
        inside = owner == position
        out[(example.client_id, example.cutoff)] = np.flatnonzero(
            np.asarray(history.mask)[inside]
        )

    return out, history


def test_example_scheme_is_independent_of_batch_composition_for_every_mode(env, tok_run):

    store = ClientStore(env.root, "val_client", env.vocab_dir, max_clients=6)

    assert len(store) >= 6

    for mode in MODES:

        # --- один и тот же пример в разных batch -----------
        alone, _ = masks_by_example(env, store, [1], mode, SCHEME_EXAMPLE)
        pair, _ = masks_by_example(env, store, [0, 1], mode, SCHEME_EXAMPLE)
        other_order, _ = masks_by_example(env, store, [1, 0], mode, SCHEME_EXAMPLE)
        wider, _ = masks_by_example(env, store, [2, 1, 0, 3], mode, SCHEME_EXAMPLE)

        key = next(iter(alone))

        for other in (pair, other_order, wider):
            assert np.array_equal(alone[key], other[key]), mode

        # Маска непустая: иначе равенство ничего не значило бы.
        assert alone[key].size > 0, mode

        # Соседи тоже не поехали от смены порядка и размера.
        for shared in set(pair) & set(wider):
            assert np.array_equal(pair[shared], wider[shared]), mode

        # --- прежняя схема как раз зависит от batch --------
        batch_alone, _ = masks_by_example(env, store, [1], mode, SCHEME_BATCH)
        batch_wider, _ = masks_by_example(env, store, [2, 1, 0, 3], mode, SCHEME_BATCH)

        assert not np.array_equal(batch_alone[key], batch_wider[key]), mode

        # --- разные примеры получают разные маски ----------
        assert len({tuple(value.tolist()) for value in wider.values()}) > 1, mode

    # --- прежняя схема не изменилась -----------------------
    #
    # Сравнение с независимым воспроизведением прежнего правила:
    # один поток default_rng([seed, step]) на весь batch.
    examples = store.examples(range(4))

    batch = collate(examples)

    masker = Masker(env.vocab, MaskingConfig(mode="token", seed=31337, token_rate=0.15))

    eligible = masker.eligible(batch.key_ids, batch.value_ids)

    indices = np.flatnonzero(eligible)

    expected = indices[
        np.random.default_rng([31337, 5]).random(indices.size) < 0.15
    ]

    masked = masker.apply(batch, step=5)

    assert np.array_equal(np.flatnonzero(masked.mask), expected)

    # --- бюджет field_balanced теперь на пример ------------
    _, history = masks_by_example(
        env, store, [0, 1, 2], MODE_FIELD_BALANCED, SCHEME_EXAMPLE
    )

    owner = np.asarray(history.tokens.example_ids, dtype=np.int64)

    masker = Masker(
        env.vocab, MaskingConfig(mode=MODE_FIELD_BALANCED, seed=31337, scheme=SCHEME_EXAMPLE)
    )

    eligible = masker.eligible(history.tokens.key_ids, history.tokens.value_ids)

    for example in range(3):

        inside = owner == example

        # Маска исходная, поэтому доступные позиции считаются
        # до подстановки [MASK]: берём их из целей.
        available = int((np.asarray(history.targets) >= 0)[inside].sum()) + int(
            (eligible & inside).sum()
        )

        chosen = int(np.asarray(history.mask)[inside].sum())

        assert chosen == pytest.approx(round(0.15 * available), abs=1)

    # --- combined остаётся объединением трёх выборок -------
    _, combined = masks_by_example(env, store, [0, 1], MODE_COMBINED, SCHEME_EXAMPLE)

    selection = combined.masking["selection"]

    assert selection["scheme"] == SCHEME_EXAMPLE
    assert set(selection["strategies"]) == {"token", "event", "key"}
    assert selection["unique"] >= max(selection["strategies"].values())

    # --- схема example требует идентичности ----------------
    with pytest.raises(ValueError):
        Masker(env.vocab, MaskingConfig(scheme=SCHEME_EXAMPLE)).apply(batch, 0)
