from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from tests.test_profile_state import (
    QUIET_SNAPSHOT,
    RAW_CLIENT,
    prepare,
    raw_event,
    when,
    write_profile_vocab,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Веха о продукте выдала бы закрытое событие, из которого она
# получена. Какое это событие, говорит ссылка вехи source_id на
# саму карту или договор, а не совпадение времени:
#
#   T1  card_activated crd_1   источник первой карты
#       card_activated crd_2   тот же момент, другая карта — цель
#       purchase по crd_1      тот же момент и карта, другой тип — цель
#   T2  card_activated crd_3   следующая карта — цель
#   T3  account_opened  dep_1  источник первого вклада: акт открытия
#       product_opened  dep_1  это две строки одного момента
#       account_opened  dep_2  тот же момент, другой вклад — цели
#       product_opened  dep_2
#   T4  product_opened  loan_1 источник первого кредита
#   T5  product_opened  loan_2 второй кредит — цель
#   T6  product_migrated loan_3 третий кредит — цель
#   T7  account_opened, product_opened dep_3  второй вклад — цели
#
# Проверяется путь целиком: препроцессинг помечает ровно акт
# источника; пометка доезжает до маски целей 05 и переживает отбор
# контекста, раскладку по batch, маскирование и упаковку
# micro-batch; сами вехи остаются в анкете контекстом.
# Сломанная ссылка останавливает препроцессинг.
#
# Группа train: её период целей — вся история до cutoff (1 января
# 2026 у банка), и отбор контекста законно режет её начало.
# ============================================================


OLD = when("2021-05-17T00:00:00")

T1, T2, T3, T4, T5, T6, T7 = (
    "2025-06-10T10:00:00", "2025-07-01T10:00:00", "2025-08-01T12:00:00",
    "2025-09-01T12:00:00", "2025-10-01T12:00:00", "2025-11-01T12:00:00",
    "2025-11-15T12:00:00",
)

EVENT_TYPES = ("purchase", "card_activated", "account_opened", "product_opened", "product_migrated")


def purchase(moment: str, **fields) -> dict:
    return raw_event(RAW_CLIENT, moment, {
        "type": "purchase", "amount": 700, "direction": "debit", "status": "approved", **fields})


def card(moment: str, card_id: str, reason: str) -> dict:
    return raw_event(RAW_CLIENT, moment, {
        "type": "card_activated", "product_id": "prd_card", "card_id": card_id, "reason": reason})


def opened(moment: str, kind: str, contract_id: str, amount: int, **fields) -> dict:
    return raw_event(RAW_CLIENT, moment, {
        "type": kind, "product_id": "prd_product", "contract_id": contract_id,
        "amount_or_limit": amount, **fields})


TAPE = [
    purchase("2025-03-01T09:00:00"),
    purchase("2025-03-02T09:00:00"),
    card(T1, "crd_1", "application_approved"),
    card(T1, "crd_2", "opened"),
    purchase(T1, card_id="crd_1"),
    card(T2, "crd_3", "opened"),
    opened(T3, "account_opened", "dep_1", 100_000),
    opened(T3, "product_opened", "dep_1", 100_000),
    opened(T3, "account_opened", "dep_2", 200_000),
    opened(T3, "product_opened", "dep_2", 200_000),
    opened(T4, "product_opened", "loan_1", 300_000),
    opened(T5, "product_opened", "loan_2", 400_000),
    opened(T6, "product_migrated", "loan_3", 500_000, migration_reason="refinance"),
    opened(T7, "account_opened", "dep_3", 600_000),
    opened(T7, "product_opened", "dep_3", 600_000),
]

MILESTONES = [
    {"type": "bank_registered", "event_time": OLD, "source_id": None},
    {"type": "first_card_activated", "event_time": when(T1), "source_id": "crd_1"},
    {"type": "first_deposit_opened", "event_time": when(T3), "source_id": "dep_1"},
    {"type": "first_loan_opened", "event_time": when(T4), "source_id": "loan_1"},
]

# Акт источника, узнанный по содержимому строки, а не по пометке:
# (тип события, момент, сумма договора или причина) -> веха.
SOURCES = {
    ("card_activated", when(T1), "application_approved"): "first_card_activated",
    ("account_opened", when(T3), 100_000): "first_deposit_opened",
    ("product_opened", when(T3), 100_000): "first_deposit_opened",
    ("product_opened", when(T4), 300_000): "first_loan_opened",
}


def history(stage, tape: list[dict] = TAPE, milestones: list[dict] = MILESTONES):
    return prepare(stage, tape, dict(QUIET_SNAPSHOT, lifelong=milestones), group="train")


def identity(event) -> tuple:
    """
    Строка события по содержимому: тип, момент и сумма договора или
    причина — чем различаются события одного момента.
    """

    from src.preprocessing.keys import key_for

    reason = key_for("reason", "product_events").key

    values = event.values

    return (
        values["event_type"],
        event.event_time,
        values.get("amount_or_limit", values.get(reason)),
    )


def expected_targets(events) -> list[bool]:
    """
    Может ли событие быть целью: всё, кроме акта источника.
    """

    return [identity(event) not in SOURCES for event in events]


# ============================================================
# ПРЕПРОЦЕССИНГ: ССЫЛКА НАХОДИТ АКТ ИСТОЧНИКА
# ============================================================


def test_link_marks_exactly_the_act_of_the_source(stage):

    events = history(stage).events

    assert len(events) == len(TAPE)

    marked = {identity(event): event.lifelong_source for event in events if event.lifelong_source}

    assert marked == SOURCES


def test_neighbours_of_the_same_moment_are_not_sources(stage):
    """
    В момент T1 три события, в T3 четыре: источник среди них только
    тот, на кого указывает ссылка. Покупка по той же карте — не
    активация, и источником не становится.
    """

    events = history(stage).events

    at = {moment: [event for event in events if event.event_time == when(moment)] for moment in (T1, T3)}

    assert [event.lifelong_source for event in at[T1]].count("first_card_activated") == 1
    assert [event.lifelong_source for event in at[T1]].count(None) == 2

    assert [event.lifelong_source for event in at[T3]].count("first_deposit_opened") == 2
    assert [event.lifelong_source for event in at[T3]].count(None) == 2


def test_later_cards_loans_and_deposits_are_not_sources(stage):

    later = [event for event in history(stage).events if event.event_time > when(T4)]

    assert {event.values["event_type"] for event in later} == {
        "product_opened", "product_migrated", "account_opened",
    }
    assert all(event.lifelong_source is None for event in later)


def test_milestone_before_the_window_has_no_act_in_the_tape(stage):
    """
    Первая карта активирована в 2021 году: её акта в ленте нет, и
    все активации ленты — обычные события.
    """

    milestones = [dict(item, event_time=OLD) if item["type"] == "first_card_activated" else item
                  for item in MILESTONES]

    tape = [event for event in TAPE if '"crd_1"' not in event["payload"] or "purchase" in event["payload"]]

    events = history(stage, tape, milestones).events

    assert not [event for event in events if event.lifelong_source == "first_card_activated"]


def test_link_to_another_moment_is_refused(stage):
    """
    Ссылка указала на активацию другого момента: выгрузка сломана,
    и совпадением времени её не починить.
    """

    from src.preprocessing.canonical.events import CanonicalError

    milestones = [dict(item, event_time=when(T2)) if item["type"] == "first_card_activated" else item
                  for item in MILESTONES]

    with pytest.raises(CanonicalError, match="записан в"):
        history(stage, milestones=milestones)


def test_milestone_inside_the_window_without_its_act_is_refused(stage):

    from src.preprocessing.canonical.events import CanonicalError

    milestones = [dict(item, source_id="loan_9") if item["type"] == "first_loan_opened" else item
                  for item in MILESTONES]

    with pytest.raises(CanonicalError, match="в ленте нет"):
        history(stage, milestones=milestones)


def test_preprocessed_layer_without_the_marks_is_refused(stage):
    """
    Слой 02 прежнего кода меток не несёт: прочитай его молча, и
    источники стали бы целями.
    """

    from src.preprocessing.canonical.build import EVENTS_FILE
    from src.preprocessing.read import Group, ReadError
    from src.preprocessing.settings import group_dir

    history(stage)

    path = group_dir("train") / EVENTS_FILE

    pq.write_table(pq.read_table(path).drop(["lifelong_source"]), path)

    with pytest.raises(ReadError, match="lifelong_source"):
        Group("train")


# ============================================================
# 04 → 05: КОНТЕКСТ, НО НЕ ЦЕЛЬ
# ============================================================


def encoded(stage, max_events: int | None = None):
    """
    Выгрузка → 02 → 04 → 05 группы train; пример клиента и его
    история на cutoff.
    """

    from src.dataset.build import build_group
    from src.dataset.settings import ContextPolicy, DatasetConfig, dataset_dir, SAMPLES_FILE
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    write_profile_vocab(stage, event_types=EVENT_TYPES)

    client = history(stage)

    artifacts = FrozenArtifacts.load()

    encode_group(artifacts, "train", TokenizerConfig.load(None))

    policy = ContextPolicy() if max_events is None else ContextPolicy(max_events=max_events)

    build_group(artifacts, "train", DatasetConfig(context=policy))

    row = pq.read_table(dataset_dir("train") / SAMPLES_FILE).to_pylist()[0]

    return artifacts, client, row


def test_tokenized_events_carry_the_mark(stage):

    from src.tokenization.settings import tokenized_dir
    from src.tokenization.transform import EVENTS_FILE

    _, client, _ = encoded(stage)

    rows = pq.read_table(tokenized_dir("train") / EVENTS_FILE).to_pylist()

    assert [row["lifelong_source"] for row in rows] == [event.lifelong_source for event in client.events]


def test_sources_stay_in_the_history_and_milestones_in_the_profile(stage):

    artifacts, client, row = encoded(stage)

    assert len(row["event_time"]) == len(TAPE)
    assert row["target_event_mask"] == expected_targets(client.events)
    assert row["target_event_mask"].count(False) == len(SOURCES)

    # Вехи видны анкетой: четыре токена под своим ключом, со своим
    # временем.
    key = artifacts.key_id("profile_lifelong")

    moments = [moment for key_id, moment in zip(row["profile_key_ids"], row["profile_time"])
               if key_id == key]

    assert moments == [item["event_time"] for item in MILESTONES]


@pytest.mark.parametrize("max_events", [14, 13, 12, 9, 4])
def test_prohibition_survives_the_context_limit(stage, max_events: int):
    """
    Отбор оставляет хвост истории. Пределы режут старые покупки, сам
    момент T1 (посередине его трёх событий), первый вклад и всё до
    второго кредита: маска — это маска оставшихся событий, а не
    сдвинутая чужая.
    """

    _, client, row = encoded(stage, max_events)

    kept = client.events[-max_events:]

    assert row["event_time"] == [event.event_time for event in kept]
    assert row["target_event_mask"] == expected_targets(kept)


# ============================================================
# 06 → 07 → 08 → MICRO-BATCH
# ============================================================


def test_masking_and_micro_batches_never_label_a_source(stage):
    """
    Маскер закрывает каждое допустимое событие целиком. У актов
    источника нет ни одной метки, у остальных событий — есть; при
    упаковке двух клиентов в один micro-batch цели каждого — ровно
    его события не-источники.
    """

    from src.batching.build import build_group as build_batches
    from src.batching.settings import BatchingConfig
    from src.masking.apply import IGNORE
    from src.masking.build import build_group as build_masks
    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Source
    from src.mlm.model import pack
    from src.temporal.build import build_group as build_temporal

    max_events = 12

    _, client, _ = encoded(stage, max_events)

    build_temporal("train")
    build_batches("train", BatchingConfig.load(None))

    every_event = MaskingConfig(
        event_probability=1.0, value_probability=0.0, key_probability=0.0, unknown_probability=0.0
    )

    build_masks("train", every_event)

    expected = expected_targets(client.events[-max_events:])

    (one,) = list(Source("train").clients())

    labelled = [
        bool((one.labels[start:start + length] != IGNORE).any())
        for start, length in zip(one.event_starts, one.event_lengths)
    ]

    assert labelled == expected

    # Тот же маскер при чтении эпохи train даёт тот же запрет.
    (drawn,) = list(Source("train", masking=every_event).clients())

    assert np.array_equal(drawn.labels, one.labels)

    data = pack([one, drawn], torch.device("cpu"))

    wanted = [number for number, flag in enumerate(expected) if flag]

    for client_number in (0, 1):
        mine = data.target_local[data.target_client == client_number].tolist()
        assert sorted(set(mine)) == wanted
