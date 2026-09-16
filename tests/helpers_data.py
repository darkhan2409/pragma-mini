"""
Общие строители данных для тестов.

Здесь живут игрушечные словари и синтетические примеры, которые
нужны больше чем одному файлу. Раньше они лежали в самих тестах и
импортировались друг у друга: файл нельзя было удалить, не
обрушив соседние, а порядок импорта тянул за собой чужие
константы.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from src.tokenizer.config import EVT_ID, USR_ID
from src.tokenizer.dataset import Events, Example, TokenBatch, collate
from src.tokenizer.encode import encode_pairs
from src.tokenizer.vocab import NO_FIELD, FieldEntry, KeyToken, ValueEntry, Vocab
from src.model.history_batching import metadata_from_examples


# Общая точка отсчёта синтетических лент и их значения.
BASE = datetime(2025, 1, 1)

COLORS = ("red", "blue")
SIZES = (1, 2)

# Скрытые поля генератора: в RAW их не бывает никогда.
LATENT_COLUMNS = {
    "activity",
    "digital_affinity",
    "mobility",
    "credit_need",
    "risk",
    "volatility",
    "push_reachable",
    "campaign",
    "clicked",
    "outcome",
    "activity_scenario",
    "stress_scenario",
    "credit_stress",
    "browsed_products",
}


def toy_vocab() -> Vocab:
    """
    Три ключа, шесть значений. Константы заданы здесь, а не
    прочитаны из файлов: тест обязан ломаться от изменения
    правил, а не от изменения данных.
    """

    fields = [
        FieldEntry(0, "toy__color", "toy", "color", "categorical", True, "string", 6, (9, 10)),
        FieldEntry(1, "toy__size", "toy", "size", "numeric", True, "int64", 7, (11, 12)),
        FieldEntry(2, "toy__flag", "toy", "flag", "boolean", False, "bool", 8, (13, 14)),
    ]

    key_tokens = [
        KeyToken(6, "toy__color", (0,)),
        KeyToken(7, "toy__size", (1,)),
        KeyToken(8, "toy__flag", (2,)),
    ]

    values = [
        ValueEntry(9, "blue", "string", False, 5, (0,)),
        ValueEntry(10, "red", "string", False, 3, (0,)),
        ValueEntry(11, "0", "int64", False, 4, (1,)),
        ValueEntry(12, "1", "int64", False, 4, (1,)),
        ValueEntry(13, "false", "bool", False, 2, (2,)),
        ValueEntry(14, "true", "bool", False, 6, (2,)),
    ]

    return Vocab(fields, key_tokens, values)


def unbalanced_vocab() -> Vocab:
    """
    Два predictable-поля и одно нет: A широкое, B узкое.
    """

    fields = [
        FieldEntry(0, "s__a", "s", "a", "categorical", True, "string", 6, (9, 10, 11)),
        FieldEntry(1, "s__b", "s", "b", "categorical", True, "string", 7, (12, 13, 14)),
        FieldEntry(2, "s__quiet", "s", "quiet", "categorical", False, "string", 8, (15, 16, 17)),
    ]

    key_tokens = [
        KeyToken(6, "s__a", (0,)),
        KeyToken(7, "s__b", (1,)),
        KeyToken(8, "s__quiet", (2,)),
    ]

    values = [
        ValueEntry(9 + offset, f"v{offset % 3}", "string", False, 1, (offset // 3,))
        for offset in range(9)
    ]

    return Vocab(fields, key_tokens, values)


def fields_of(key_ids: np.ndarray) -> np.ndarray:
    """
    field_id синтетического batch.

    Мини-словари собраны как baseline, поэтому key token это
    6 + field_id, а специальная позиция остаётся без поля.
    """

    keys = np.asarray(key_ids, dtype=np.int64)

    return np.where(keys >= 6, keys - 6, NO_FIELD).astype(np.int16)


def make_batch(key_ids, value_ids, event_ids=None, example_ids=None, profile=None) -> TokenBatch:
    """
    Batch из готовых массивов: маскирование не зависит от того,
    откуда пришли токены.
    """

    key_ids = np.asarray(key_ids, dtype=np.int32)
    value_ids = np.asarray(value_ids, dtype=np.int32)

    size = key_ids.size

    event_ids = np.arange(size, dtype=np.int64) if event_ids is None else np.asarray(event_ids, np.int64)
    example_ids = np.zeros(size, dtype=np.int64) if example_ids is None else np.asarray(example_ids, np.int64)

    n_events = int(event_ids.max()) + 1 if size else 0

    widths = np.bincount(event_ids, minlength=n_events)

    offsets = np.zeros(n_events + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    profile = np.zeros(0, dtype=np.int32) if profile is None else np.asarray(profile, np.int32)

    return TokenBatch(
        key_ids=key_ids,
        value_ids=value_ids,
        positions=np.zeros(size, dtype=np.int16),
        event_ids=event_ids,
        example_ids=example_ids,
        event_offsets=offsets,
        example_of_event=np.zeros(n_events, dtype=np.int64),
        event_type=np.array(["synthetic"] * n_events, dtype=object),
        ts=np.zeros(n_events, dtype="datetime64[us]"),
        seq=np.arange(n_events, dtype=np.int64),
        profile_key_ids=np.zeros(profile.size, dtype=np.int32),
        profile_value_ids=profile,
        profile_positions=np.zeros(profile.size, dtype=np.int16),
        profile_example_ids=np.zeros(profile.size, dtype=np.int64),
        n_examples=int(example_ids.max()) + 1 if size else 0,
        field_ids=fields_of(key_ids),
        profile_field_ids=fields_of(np.zeros(profile.size, dtype=np.int32)),
    )


def ident(batch) -> np.ndarray:
    """
    Идентичности синтетического batch.

    Маска привязана к паре (client_id, cutoff), поэтому каждый
    пример получает свою: подставляется его номер. Так свойства
    маскирования проверяются на том же пути, что и в бою.
    """

    n = int(batch.n_examples)

    return np.stack(
        [np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)], axis=1
    )


def build_example(
    vocab,
    client_id: int,
    hours: list[float],
    cutoff_hours: float,
    colors: list[str] | None = None,
) -> Example:
    """
    Пример с событиями в заданные часы от базы.
    """

    colors = colors or [COLORS[index % len(COLORS)] for index in range(len(hours))]

    records = [encode_pairs(vocab, [("toy__color", color)], EVT_ID) for color in colors]

    widths = np.array([len(record.key_ids) for record in records], dtype=np.int64)

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    events = Events(
        key_ids=np.concatenate([record.key_ids for record in records]),
        value_ids=np.concatenate([record.value_ids for record in records]),
        positions=np.concatenate([record.positions for record in records]),
        offsets=offsets,
        event_type=np.array(["toy"] * len(records), dtype=object),
        ts=np.array([np.datetime64(BASE + timedelta(hours=h), "us") for h in hours]),
        seq=np.arange(len(records), dtype=np.int64),
        field_ids=np.concatenate([record.field_ids for record in records]),
    )

    # Ключи профиля в реальном словаре никогда не predictable:
    # берём toy__flag, иначе masker справедливо ругается.
    profile = encode_pairs(vocab, [("toy__flag", True)], USR_ID)

    return Example(
        client_id=client_id,
        cutoff=BASE + timedelta(hours=cutoff_hours),
        dataset="toy",
        client_group="train",
        seq_end=len(records),
        snapshot_ts=BASE - timedelta(hours=1),
        profile=profile,
        events=events,
    )


def mlm_example(
    vocab,
    client_id: int,
    hours: list[float],
    cutoff_hours: float,
    colors: list[str] | None = None,
    sizes: list[int] | None = None,
) -> Example:
    """
    Событие это [EVT] плюс два предсказуемых поля.
    """

    count = len(hours)

    colors = colors or [COLORS[index % len(COLORS)] for index in range(count)]
    sizes = sizes or [index % 2 for index in range(count)]

    records = [
        encode_pairs(vocab, [("toy__color", color), ("toy__size", size)], EVT_ID)
        for color, size in zip(colors, sizes)
    ]

    widths = np.array([len(record.key_ids) for record in records], dtype=np.int64)

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    events = Events(
        key_ids=np.concatenate([record.key_ids for record in records]),
        value_ids=np.concatenate([record.value_ids for record in records]),
        positions=np.concatenate([record.positions for record in records]),
        offsets=offsets,
        event_type=np.array(["toy"] * len(records), dtype=object),
        ts=np.array([np.datetime64(BASE + timedelta(hours=h), "us") for h in hours]),
        seq=np.arange(len(records), dtype=np.int64),
        field_ids=np.concatenate([record.field_ids for record in records]),
    )

    return Example(
        client_id=client_id,
        cutoff=BASE + timedelta(hours=cutoff_hours),
        dataset="toy",
        client_group="train",
        seq_end=len(records),
        snapshot_ts=BASE - timedelta(hours=1),
        profile=encode_pairs(vocab, [("toy__flag", True)], USR_ID),
        events=events,
    )


LEDGER: tuple[tuple[int, str, int], ...] = (
    (-3600, "communication", -1),
    (0, "app_operation", -1),
    (10, "app_operation", -1),
    (14, "app_screen", 1),
    (30, "app_screen", 1),
    (31, "banner", -1),
    (50, "transaction", -1),
    (70, "app_screen", 1),
    (95, "app_operation", -1),
    (7200, "product_event", -1),
    (20000, "app_screen", 2),
    (30000, "app_screen", 3),
    (30060, "app_screen", 3),
)


def ledger(vocab, client_id: int, spec, cutoff_s: float) -> Example:
    """
    Пример с событиями в заданные секунды и ключами сессий.
    """

    records = [
        encode_pairs(
            vocab,
            [("toy__color", COLORS[index % 2]), ("toy__size", SIZES[index % 2])],
            EVT_ID,
        )
        for index in range(len(spec))
    ]

    widths = np.array([len(record.key_ids) for record in records], dtype=np.int64)

    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    np.cumsum(widths, out=offsets[1:])

    events = Events(
        key_ids=np.concatenate([record.key_ids for record in records]),
        value_ids=np.concatenate([record.value_ids for record in records]),
        positions=np.concatenate([record.positions for record in records]),
        offsets=offsets,
        event_type=np.array([kind for _, kind, _ in spec], dtype=object),
        ts=np.array(
            [np.datetime64(BASE + timedelta(seconds=t), "us") for t, _, _ in spec]
        ),
        seq=np.arange(len(spec), dtype=np.int64),
        field_ids=np.concatenate([record.field_ids for record in records]),
    )

    return Example(
        client_id=client_id,
        cutoff=BASE + timedelta(seconds=cutoff_s),
        dataset="toy",
        client_group="train",
        seq_end=len(spec),
        snapshot_ts=BASE - timedelta(hours=2),
        profile=encode_pairs(vocab, [("toy__flag", True)], USR_ID),
        events=events,
    )


def prefix(spec, cutoff_s: float):
    return [row for row in spec if row[0] < cutoff_s]


def examples(vocab=None):
    """
    Пять примеров: полный, два обрезанных cutoff, без сессий
    и с единственным экраном.
    """

    vocab = vocab or toy_vocab()

    return {
        "ex0": ledger(vocab, 1, LEDGER, 40000),
        "ex1": ledger(vocab, 1, prefix(LEDGER, 32), 32),
        "ex2": ledger(vocab, 1, prefix(LEDGER, 12), 12),
        "ex3": ledger(vocab, 2, ((0, "transaction", -1), (100, "transaction", -1)), 40000),
        "ex4": ledger(vocab, 3, ((0, "app_screen", 5),), 40000),
    }


def synthetic_batch(histories: list[list[float]], cutoff_hours: float, sort: bool = True):
    """
    TokenBatch и metadata из списка историй, заданных часами.
    """

    vocab = toy_vocab()

    examples = [
        build_example(vocab, index, sorted(hours) if sort else hours, cutoff_hours)
        for index, hours in enumerate(histories)
    ]

    return collate(examples), metadata_from_examples(examples)
