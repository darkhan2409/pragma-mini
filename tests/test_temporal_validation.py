"""
Temporal validation: метрика по новому месяцу и потоковая сборка.

Метрика на полной истории отвечает «как модель в среднем читает
прошлое клиента». Метрика месяца наблюдения отвечает «как она
читает то, что месяц добавил». В полной истории свежих событий
единицы процентов, и улучшение на них там растворяется.

Поток нужен, чтобы не держать весь набор в памяти. Он честен
ровно постольку, поскольку маска привязана к примеру: второй
проход обязан дать те же цели при любом размере batch.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.model.data import ClientStore, FixedSplit
from src.model.history_batching import metadata_from_examples, prepare_history_batch
from src.model.mlm_batching import build_targets
from src.model.trainer import (
    BEST_SCOPE_FULL,
    BEST_SCOPE_RECENT,
    Trainer,
    best_value,
)
from src.tokenizer.dataset import collate
from src.tokenizer.masking import (
    MODE_COMBINED,
    SCHEME_BATCH,
    SCHEME_EXAMPLE,
    Masker,
    MaskingConfig,
)

from tests.test_trainer import env, small_config  # noqa: F401


def val_masking(scheme: str = SCHEME_EXAMPLE) -> MaskingConfig:
    return MaskingConfig(mode=MODE_COMBINED, seed=99, scheme=scheme)


def targets_of(env, store, indices, max_events=64, scheme=SCHEME_EXAMPLE, step=0):

    examples = store.examples(indices)

    masker = Masker(env.vocab, val_masking(scheme))

    history = prepare_history_batch(
        collate(examples),
        metadata_from_examples(examples),
        max_events,
        masker=masker,
        step=step,
    )

    return examples, history, build_targets(history, env.table)


def described(examples, history, targets) -> dict:
    """
    Цели каждого примера в виде, не зависящем от состава batch.

    Событие адресуется номером ВНУТРИ примера, а не строкой
    batch: иначе сравнение двух разных batch было бы сравнением
    нумерации, а не целей.
    """

    example_of_event = np.asarray(history.tokens.example_of_event, dtype=np.int64)

    counts = np.bincount(example_of_event, minlength=len(examples))

    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    out: dict = {}

    for position, example in enumerate(examples):

        chosen = targets.example == position

        local = targets.event_row[chosen] - starts[position]

        out[(example.client_id, example.cutoff)] = sorted(
            zip(
                local.tolist(),
                targets.col[chosen].tolist(),
                targets.key_ids[chosen].tolist(),
                targets.global_targets[chosen].tolist(),
                targets.recent[chosen].tolist(),
            )
        )

    return out


# ============================================================
# 5. ЦЕЛИ НОВОГО МЕСЯЦА
# ============================================================


def test_recent_targets_mark_only_the_observation_month_and_drive_best_selection(
    env, tok_run
):

    store = ClientStore(env.root, "val_client", env.vocab_dir, max_clients=6)

    examples, history, targets = targets_of(env, store, range(6))

    # --- окно это month наблюдения, а не выдумка -----------
    #
    # Сверяется с колонкой observation_month самого набора:
    # правило выводится из cutoff, и совпасть оно обязано с
    # тем, что записал preprocessing.
    table = pq.read_table(
        env.root / "val_client" / "examples.parquet",
        columns=["client_id", "cutoff", "observation_month"],
    ).to_pydict()

    declared = {
        (int(client), np.datetime64(cutoff, "us")): np.datetime64(month, "us")
        for client, cutoff, month in zip(
            table["client_id"], table["cutoff"], table["observation_month"]
        )
    }

    window = history.meta.window_start.astype("datetime64[us]")

    for position, example in enumerate(examples):
        key = (example.client_id, np.datetime64(example.cutoff, "us"))
        assert window[position] == declared[key]

    # --- recent истинно ровно у событий этого месяца -------
    ts = np.asarray(history.tokens.ts).astype("datetime64[us]")

    expected = ts[targets.event_row] >= window[targets.example]

    assert np.array_equal(targets.recent, expected)

    # Свежие цели есть, но их меньшинство: иначе метрика
    # месяца была бы просто копией общей.
    assert 0 < targets.n_recent < targets.n

    # --- метрика месяца это срез той же оценки -------------
    trainer = Trainer(
        small_config(mask_scheme=SCHEME_EXAMPLE),
        env.tokenizer,
        env.table,
        env.unigram,
        "cpu",
    )

    split = FixedSplit.build(
        name="val_client",
        store=store,
        vocab=env.vocab,
        table=env.table,
        model_config=trainer.model_config,
        masking=val_masking(),
        max_events=64,
        batch_size=2,
    )

    report = trainer.evaluate({"val_client": split})["val_client"]

    fresh = report["recent"]

    assert 0 < fresh["n_targets"] < report["n_targets"]
    assert fresh["scope"]

    # Число свежих целей это ровно сумма флагов по batch.
    counted = sum(int(item[1].n_recent) for item in split.prepared)

    assert fresh["n_targets"] == counted

    # Маски общие: полный отчёт не изменился от появления среза.
    assert report["n_masked_positions"] == fresh["n_masked_positions"]

    # --- критерий best читает то, что просили --------------
    reports = {
        "val_time": {
            "field_balanced_ce": 1.5,
            "recent": {"field_balanced_ce": 2.5},
        }
    }

    assert best_value(reports, BEST_SCOPE_FULL) == 1.5
    assert best_value(reports, BEST_SCOPE_RECENT) == 2.5
    assert best_value({}, BEST_SCOPE_RECENT) is None
    assert best_value({"val_time": {"field_balanced_ce": 1.0}}, BEST_SCOPE_RECENT) is None


# ============================================================
# 6. ПОТОК ПОВТОРЯЕТ НАБОР
# ============================================================


def test_streaming_split_reproduces_the_stored_split_at_any_batch_size(env, tok_run):

    trainer = Trainer(
        small_config(mask_scheme=SCHEME_EXAMPLE),
        env.tokenizer,
        env.table,
        env.unigram,
        "cpu",
    )

    def build(batch_size: int, stream: bool, shared=None) -> tuple:
        store = ClientStore(
            env.root, "val_client", env.vocab_dir, max_clients=6, shared=shared
        )
        split = FixedSplit.build(
            name="val_client",
            store=store,
            vocab=env.vocab,
            table=env.table,
            model_config=trainer.model_config,
            masking=val_masking(),
            max_events=64,
            batch_size=batch_size,
            stream=stream,
        )
        return store, split

    _, stored = build(2, stream=False)
    _, streamed = build(2, stream=True)
    _, wider = build(3, stream=True)

    assert stored.prepared is not None
    assert streamed.prepared is None and streamed.streamed
    assert streamed.source is not None

    # --- тот же размер batch: всё совпадает ----------------
    assert streamed.digest == stored.digest
    assert streamed.n_targets == stored.n_targets
    assert streamed.n_batches == stored.n_batches
    assert np.array_equal(streamed.used_lengths, stored.used_lengths)

    assert streamed.settings["stream"] is True
    assert stored.settings["stream"] is False

    # --- поток воспроизводится при повторном чтении --------
    first = [
        (inputs.n_examples, item.digest())
        for inputs, item in streamed.iter_batches(trainer.model_config)
    ]
    second = [
        (inputs.n_examples, item.digest())
        for inputs, item in streamed.iter_batches(trainer.model_config)
    ]

    assert first == second
    assert [digest for _, digest in first] == [
        item.digest() for _, item in stored.prepared
    ]

    # --- другой размер batch: те же цели у тех же примеров --
    #
    # Digest здесь совпасть не может: он считается по batch.
    # Совпасть обязаны сами цели примеров.
    assert wider.n_targets == stored.n_targets

    store = ClientStore(env.root, "val_client", env.vocab_dir, max_clients=6)

    narrow_targets: dict = {}
    wide_targets: dict = {}

    for start in range(0, len(store), 2):
        chunk = range(start, min(start + 2, len(store)))
        narrow_targets.update(described(*targets_of(env, store, chunk)))

    for start in range(0, len(store), 3):
        chunk = range(start, min(start + 3, len(store)))
        wide_targets.update(described(*targets_of(env, store, chunk)))

    assert narrow_targets.keys() == wide_targets.keys()
    assert narrow_targets == wide_targets

    # --- метрики совпадают ---------------------------------
    left = trainer.evaluate({"val_client": stored})["val_client"]
    right = trainer.evaluate({"val_client": streamed})["val_client"]

    assert left["field_balanced_ce"] == pytest.approx(right["field_balanced_ce"])
    assert left["n_targets"] == right["n_targets"]
    assert left["recent"]["field_balanced_ce"] == pytest.approx(
        right["recent"]["field_balanced_ce"]
    )

    # --- общая память: те же примеры, без второго чтения ----
    train_store = ClientStore(env.root, "train", env.vocab_dir, max_clients=6)

    shared_store, shared_split = build(2, stream=True, shared=train_store)

    if shared_store.shared_events:
        assert shared_split.digest == stored.digest

    # Чужая группа клиентов по ссылке не берётся: val_client
    # это группа val, а train_store держит группу train, и
    # молча выдать одни ленты за другие нельзя.
    assert train_store.data.group != store.data.group
    assert not shared_store.shared_events
    assert shared_split.digest == stored.digest

    # --- поток при прежней схеме запрещён ------------------
    with pytest.raises(ValueError):
        FixedSplit.build(
            name="val_client",
            store=store,
            vocab=env.vocab,
            table=env.table,
            model_config=trainer.model_config,
            masking=val_masking(SCHEME_BATCH),
            max_events=64,
            batch_size=2,
            stream=True,
        )
