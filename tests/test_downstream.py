from __future__ import annotations

from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.downstream.probe import paired
from src.downstream.settings import cutoff

from tests import world
from tests.test_profile_state import (
    AFTER,
    BUSY_SNAPSHOT,
    EARLY,
    QUIET_SNAPSHOT,
    RAW_CLIENT,
    prepare,
    write_profile_vocab,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вектор клиента на момент T честен, только если вход на T собран
# из прошлого и собран ТАК ЖЕ, как вход обучения. Поэтому:
#
#   - на конце окна группы вход в памяти обязан побитно совпасть с
#     тем, что этапы 04–07 кладут на диск, — второй реализации
#     цепочки здесь нет;
#   - всё, что случилось после T, — события, переезд, продукт, —
#     вход на T не меняет ни на бит, а анкета откатывается на T;
#   - в метках задач граница T строгая: событие ровно в T не
#     признак и не метка, а конец окна метки в него входит.
# ============================================================


NOTHING = dict(value_probability=0.0, event_probability=0.0, key_probability=0.0,
               unknown_probability=0.0)

FIELDS = (
    "key_ids", "value_ids", "positions", "labels", "event_starts", "event_lengths",
    "event_time_log", "calendar", "profile_key_ids", "profile_value_ids",
    "profile_positions", "profile_time_log",
)


def chain(stage, tape: list[dict], snapshot: dict) -> None:
    """
    Выгрузка → 02 → 04 → 05 → 06 → 07 для train и val: теми же
    функциями этапов, что и в бою. train нужен ради отбора истории,
    на котором «училась модель» (05_dataset/train/meta.json).
    """

    from src.batching.build import build_group as build_batches
    from src.batching.settings import BatchingConfig
    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig
    from src.temporal.build import build_group as build_temporal
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    write_profile_vocab(stage, event_types=("purchase", "profile_change", "product_opened"))

    artifacts = FrozenArtifacts.load()

    for group in ("train", "val"):
        prepare(stage, tape, snapshot, group=group)
        encode_group(artifacts, group, TokenizerConfig.load(None))
        build_dataset(artifacts, group, DatasetConfig.load(None))
        build_temporal(group)
        build_batches(group, BatchingConfig.load(None))


def same(left, right) -> None:

    for name in FIELDS:
        one, other = getattr(left, name), getattr(right, name)
        assert one.dtype == other.dtype and one.shape == other.shape, name
        assert np.array_equal(one, other), name

    assert left.reason == right.reason


# ============================================================
# ВХОД НА T
# ============================================================


def test_input_at_the_group_cutoff_is_what_stage_07_stores(stage):
    """
    T = конец окна val: вход в памяти побитно равен клиенту, которого
    модель читает из 07_batches без масок.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff
    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Source
    from src.preprocessing.settings import PreprocessingConfig

    chain(stage, EARLY, QUIET_SNAPSHOT)

    (stored,) = list(Source("val", masking=MaskingConfig(**NOTHING)).clients())

    end = PreprocessingConfig.load(None).windows["val"].final_cutoff

    built = ClientsAtCutoff("val", end).client(RAW_CLIENT)

    assert built.n_events == len(EARLY) and built.n_events > 0
    same(built, stored)


def test_the_future_after_the_cutoff_does_not_change_the_input(stage):
    """
    Две выгрузки с общим прошлым до T: во второй после T ещё продукт
    и переезд, и снимок анкеты уже знает о переезде. Вход на T у них
    побитно один.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff

    moment = cutoff("val")

    chain(stage, EARLY, QUIET_SNAPSHOT)
    quiet = ClientsAtCutoff("val", moment).client(RAW_CLIENT)

    chain(stage, EARLY + AFTER, BUSY_SNAPSHOT)
    busy = ClientsAtCutoff("val", moment).client(RAW_CLIENT)

    same(quiet, busy)

    # T раньше конца окна: переезд 1 февраля и продукт 1 марта в
    # ленту на T не попадают.
    assert quiet.n_events == 2
    assert all(event_time < moment for event_time in quiet.event_time)
    assert quiet.event_time_log[-1] == 0.0


def test_the_input_at_an_earlier_cutoff_differs_from_the_end_of_the_window(stage):
    """
    Обратная сторона: T раньше конца окна действительно отрезает
    события и откатывает анкету — иначе прошлый тест ничего бы не
    проверял.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff
    from src.preprocessing.settings import PreprocessingConfig

    chain(stage, EARLY, QUIET_SNAPSHOT)

    end = PreprocessingConfig.load(None).windows["val"].final_cutoff

    early = ClientsAtCutoff("val", cutoff("val")).client(RAW_CLIENT)
    late = ClientsAtCutoff("val", end).client(RAW_CLIENT)

    assert early.n_events < late.n_events
    assert not np.array_equal(early.profile_value_ids, late.profile_value_ids)


def test_a_cutoff_outside_the_window_is_refused(stage):

    from src.downstream.at_cutoff import CutoffError, window_at
    from src.preprocessing.settings import PreprocessingConfig

    window = PreprocessingConfig.load(None).windows["val"]

    with pytest.raises(CutoffError, match="вне окна"):
        window_at("val", window.final_cutoff + timedelta(seconds=1))

    with pytest.raises(CutoffError, match="без пояса"):
        window_at("val", window.final_cutoff.replace(tzinfo=None))


# ============================================================
# ВЕКТОРЫ
# ============================================================


def test_readouts_are_the_client_embedding_and_its_events():
    """
    usr — тот же вектор, что client_embeddings; mean_event и
    last_event — среднее и последнее из векторов событий того же
    прохода.
    """

    from src.mlm.model import pack

    model = world.model().eval()
    clients = [client for client in world.clients() if client.n_events]
    data = pack(clients, torch.device("cpu"))

    # Всё без графа, как при съёме: с графом SDPA выбирает другое
    # ядро, и последние биты расходятся.
    with torch.no_grad():
        vectors = model.readouts(data)
        embedded = model.client_embeddings(data)
        _, events, usr = model._encode(data)

    assert torch.equal(vectors["usr"], embedded)
    assert torch.equal(vectors["usr"], usr)

    owner = data.user_of_event.numpy()

    for number, client in enumerate(clients):
        mine = events[torch.as_tensor(np.flatnonzero(owner == number))]
        assert torch.allclose(vectors["mean_event"][number], mine.mean(dim=0), atol=1e-6)
        assert torch.equal(vectors["last_event"][number], mine[-1])


# ============================================================
# ЗАДАЧИ
# ============================================================


def write_events(rows: list[tuple[str, object, str]]) -> None:
    """
    Лента 02 группы val из троек (клиент, время, тип). Группы строк
    по три: клиент ложится в две соседние, суммы обязаны сложиться.
    """

    from src.preprocessing.canonical.build import EVENTS_FILE
    from src.preprocessing.settings import group_dir

    path = group_dir("val") / EVENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "client_id": [row[0] for row in rows],
            "event_time": pa.array([row[1] for row in rows], pa.timestamp("us", tz="UTC")),
            "type": [row[2] for row in rows],
        }
    )

    pq.write_table(table, path, row_group_size=3)


def test_task_windows_have_strict_edges(stage):
    """
    Признаки — строго раньше T, метка — (T, T + 60 дней]. Событие в
    миг T не признак и не метка; ровно T + 60 дней — ещё метка,
    микросекундой позже — уже нет.
    """

    from src.downstream.tasks import build_table

    moment = cutoff("val")
    day, tick = timedelta(days=1), timedelta(microseconds=1)

    write_events([
        ("a", moment - timedelta(seconds=1), "purchase"),
        ("a", moment - 10 * day, "installment_due"),
        ("a", moment - 40 * day, "purchase"),
        ("a", moment, "purchase"),
        ("a", moment + 60 * day, "delinquency_registered"),
        ("a", moment + 60 * day + tick, "application_submitted"),
        ("b", moment - 100 * day, "purchase"),
        ("b", moment - 10 * day, "installment_missed"),
        ("b", moment - 5 * day, "loan_payment"),
        ("b", moment + tick, "application_submitted"),
        ("c", moment + day, "purchase"),
    ])

    table = build_table("val", moment)

    # c без событий до T в таблицу не входит.
    assert list(table.index) == ["a", "b"]

    a, b = table.loc["a"], table.loc["b"]

    assert a["n_events"] == 3
    assert a["gap_seconds"] == 1.0
    assert a["n_30d_purchase"] == 1 and a["n_90d_purchase"] == 2
    assert b["n_90d_purchase"] == 0 and b["n_365d_purchase"] == 1

    assert a["ndq_population"] and a["ndq"]
    assert not b["ndq_population"], "пропуск за 45 дней до T выводит из популяции"

    assert not a["a1"] and b["a1"]


# ============================================================
# СРАВНЕНИЕ
# ============================================================


def test_paired_bootstrap_is_zero_for_the_same_scores_and_positive_for_a_better_one():

    rng = np.random.default_rng(1)

    y = (rng.random(400) < 0.2).astype(int)
    noise = rng.random(400)

    same_scores = paired(y, noise, noise, draws=200, seed=0)

    assert same_scores["pr_auc"]["mean"] == 0.0
    assert same_scores["roc_auc"]["not_better"] == 1.0

    better = paired(y, y + 0.1 * noise, noise, draws=200, seed=0)

    assert better["roc_auc"]["low"] > 0.0
    assert better["pr_auc"]["not_better"] == 0.0
