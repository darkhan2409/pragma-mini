from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.tokenizer.dataset import Record, TokenBatch

from .config import ModelConfig


# ============================================================
# ИДЕЯ
# ============================================================
#
# TokenBatch хранит токены плоско, без padding: события лежат
# подряд, границы задаёт event_offsets, профили размечены
# profile_example_ids. Модели нужна прямоугольная матрица, и
# превращение одного в другое живёт здесь, а не в tokenizer:
# сохранённые датасеты не меняются.
#
# padding_mask: True означает padding. Padded позиция несёт
# [PAD] в обоих массивах ID и позицию 0.
#
# Вход проверяется до модели. Ошибка формата обязана называть
# запись и причину, а не всплывать индексом за границей внутри
# CUDA-ядра.
# ============================================================


class BatchError(ValueError):
    """
    Вход не соответствует контракту tokenizer.
    """


@dataclass(frozen=True)
class PaddedRecords:
    """
    Прямоугольный batch записей: [n, L] в torch.long.
    """

    key_ids: torch.Tensor
    value_ids: torch.Tensor
    positions: torch.Tensor
    padding_mask: torch.Tensor
    lengths: torch.Tensor

    def __len__(self) -> int:
        return int(self.key_ids.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.key_ids.shape[1])

    @property
    def device(self) -> torch.device:
        return self.key_ids.device

    def to(self, device) -> "PaddedRecords":
        return PaddedRecords(
            key_ids=self.key_ids.to(device),
            value_ids=self.value_ids.to(device),
            positions=self.positions.to(device),
            padding_mask=self.padding_mask.to(device),
            lengths=self.lengths.to(device),
        )

    def slice(self, start: int, stop: int) -> "PaddedRecords":
        """
        Кусок batch с шириной по самой длинной записи куска.
        """

        lengths = self.lengths[start:stop]

        width = int(lengths.max().item())

        return PaddedRecords(
            key_ids=self.key_ids[start:stop, :width],
            value_ids=self.value_ids[start:stop, :width],
            positions=self.positions[start:stop, :width],
            padding_mask=self.padding_mask[start:stop, :width],
            lengths=lengths,
        )


# ============================================================
# НАРЕЗКА ПЛОСКОГО BATCH
# ============================================================


def check_batch(batch: TokenBatch) -> None:
    """
    Плоские массивы согласованы между собой.
    """

    if batch.n_examples <= 0:
        raise BatchError("пустой TokenBatch: нет ни одного примера")

    sizes = {
        "key_ids": batch.key_ids.size,
        "value_ids": batch.value_ids.size,
        "positions": batch.positions.size,
        "event_ids": batch.event_ids.size,
        "example_ids": batch.example_ids.size,
    }

    if len(set(sizes.values())) != 1:
        raise BatchError(f"плоские массивы событий разной длины: {sizes}")

    if batch.event_offsets.size == 0 or int(batch.event_offsets[-1]) != batch.key_ids.size:
        raise BatchError(
            f"event_offsets заканчивается на {int(batch.event_offsets[-1]) if batch.event_offsets.size else None}, "
            f"а токенов событий {batch.key_ids.size}"
        )

    profile_sizes = {
        "profile_key_ids": batch.profile_key_ids.size,
        "profile_value_ids": batch.profile_value_ids.size,
        "profile_positions": batch.profile_positions.size,
        "profile_example_ids": batch.profile_example_ids.size,
    }

    if len(set(profile_sizes.values())) != 1:
        raise BatchError(f"плоские массивы профиля разной длины: {profile_sizes}")


def split_events(batch: TokenBatch) -> list[Record]:
    """
    Плоская история обратно в отдельные события.
    """

    check_batch(batch)

    offsets = np.asarray(batch.event_offsets, dtype=np.int64)

    return [
        Record(
            key_ids=batch.key_ids[lo:hi],
            value_ids=batch.value_ids[lo:hi],
            positions=batch.positions[lo:hi],
        )
        for lo, hi in zip(offsets[:-1], offsets[1:])
    ]


def split_profiles(batch: TokenBatch) -> list[Record]:
    """
    Профили разделяются по номеру примера, а не по
    предполагаемой ширине: ширина это свойство словаря, а не
    контракта batch.
    """

    check_batch(batch)

    owner = np.asarray(batch.profile_example_ids, dtype=np.int64)

    if owner.size == 0:
        return []

    if (np.diff(owner) < 0).any():
        raise BatchError("profile_example_ids не возрастает: профили перемешаны между примерами")

    change = np.flatnonzero(np.diff(owner)) + 1

    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [owner.size]])

    return [
        Record(
            key_ids=batch.profile_key_ids[lo:hi],
            value_ids=batch.profile_value_ids[lo:hi],
            positions=batch.profile_positions[lo:hi],
        )
        for lo, hi in zip(starts, ends)
    ]


# ============================================================
# ВЫРАВНИВАНИЕ
# ============================================================


def _check_record(record: Record, index: int, lead_id: int, config: ModelConfig) -> None:

    keys = np.asarray(record.key_ids, dtype=np.int64)
    values = np.asarray(record.value_ids, dtype=np.int64)
    positions = np.asarray(record.positions, dtype=np.int64)

    if not (keys.size == values.size == positions.size):
        raise BatchError(
            f"запись {index}: key_ids, value_ids и positions разной длины "
            f"({keys.size}, {values.size}, {positions.size})"
        )

    if keys.size == 0:
        raise BatchError(f"запись {index} пустая: у неё нет даже ведущего токена")

    if (keys == config.pad_id).any() or (values == config.pad_id).any():
        raise BatchError(
            f"запись {index} содержит [PAD] внутри содержимого; "
            "[PAD] допустим только как выравнивание, tokenizer его не пишет"
        )

    for name, array in (("key_ids", keys), ("value_ids", values)):
        if array.min() < 0 or array.max() >= config.vocab_size:
            raise BatchError(
                f"запись {index}: {name} выходит за словарь [0, {config.vocab_size}): "
                f"диапазон {int(array.min())}..{int(array.max())}"
            )

    if positions.min() < 0 or positions.max() >= config.max_position_embeddings:
        raise BatchError(
            f"запись {index}: позиция {int(positions.max())} не помещается в таблицу позиций "
            f"размера {config.max_position_embeddings}; ничего не обрезается, увеличьте max_position_embeddings"
        )

    if keys[0] != lead_id or values[0] != lead_id:
        raise BatchError(
            f"запись {index} начинается с ({int(keys[0])}, {int(values[0])}), "
            f"а ведущий токен обязан быть {lead_id} в обоих массивах"
        )

    if positions[0] != 0:
        raise BatchError(f"запись {index}: ведущий токен стоит на позиции {int(positions[0])}, а должен на 0")


def pad_records(records: Sequence[Record], *, lead_id: int, config: ModelConfig) -> PaddedRecords:
    """
    Список записей в прямоугольный batch.
    """

    if len(records) == 0:
        raise BatchError("нечего кодировать: список записей пуст")

    lengths = np.empty(len(records), dtype=np.int64)

    for index, record in enumerate(records):
        _check_record(record, index, lead_id, config)
        lengths[index] = len(record.key_ids)

    width = int(lengths.max())

    keys = np.full((len(records), width), config.pad_id, dtype=np.int64)
    values = np.full((len(records), width), config.pad_id, dtype=np.int64)
    positions = np.zeros((len(records), width), dtype=np.int64)

    for index, record in enumerate(records):
        size = int(lengths[index])
        keys[index, :size] = record.key_ids
        values[index, :size] = record.value_ids
        positions[index, :size] = record.positions

    mask = np.arange(width, dtype=np.int64)[None, :] >= lengths[:, None]

    return PaddedRecords(
        key_ids=torch.from_numpy(keys),
        value_ids=torch.from_numpy(values),
        positions=torch.from_numpy(positions),
        padding_mask=torch.from_numpy(mask),
        lengths=torch.from_numpy(lengths),
    )


def pad_flat(
    key_ids: np.ndarray,
    value_ids: np.ndarray,
    positions: np.ndarray,
    offsets: np.ndarray,
    *,
    lead_id: int,
    config: ModelConfig,
) -> PaddedRecords:
    """
    Тот же результат, что pad_records, но прямо из плоских
    массивов и без цикла на Python.

    На 19 млн событий цикл по записям стоил минуты; здесь все
    проверки и раскладка векторные.
    """

    keys = np.asarray(key_ids, dtype=np.int64)
    values = np.asarray(value_ids, dtype=np.int64)
    places = np.asarray(positions, dtype=np.int64)
    bounds = np.asarray(offsets, dtype=np.int64)

    if not (keys.size == values.size == places.size):
        raise BatchError(
            f"key_ids, value_ids и positions разной длины ({keys.size}, {values.size}, {places.size})"
        )

    if bounds.size < 2 or int(bounds[-1]) != keys.size:
        raise BatchError(f"offsets заканчиваются на {int(bounds[-1]) if bounds.size else None}, а токенов {keys.size}")

    lengths = np.diff(bounds)

    if lengths.size == 0:
        raise BatchError("нечего кодировать: список записей пуст")

    empty = np.flatnonzero(lengths < 1)

    if empty.size:
        raise BatchError(f"запись {int(empty[0])} пустая: у неё нет даже ведущего токена")

    if keys.size:

        if bool((keys == config.pad_id).any()) or bool((values == config.pad_id).any()):
            position = int(np.flatnonzero((keys == config.pad_id) | (values == config.pad_id))[0])
            raise BatchError(
                f"запись {int(np.searchsorted(bounds, position, side='right') - 1)} содержит [PAD] "
                "внутри содержимого; [PAD] допустим только как выравнивание, tokenizer его не пишет"
            )

        for name, array in (("key_ids", keys), ("value_ids", values)):
            if int(array.min()) < 0 or int(array.max()) >= config.vocab_size:
                raise BatchError(
                    f"{name} выходит за словарь [0, {config.vocab_size}): "
                    f"диапазон {int(array.min())}..{int(array.max())}"
                )

        if int(places.min()) < 0 or int(places.max()) >= config.max_position_embeddings:
            raise BatchError(
                f"позиция {int(places.max())} не помещается в таблицу позиций "
                f"размера {config.max_position_embeddings}; ничего не обрезается, "
                "увеличьте max_position_embeddings"
            )

    starts = bounds[:-1]

    wrong = np.flatnonzero((keys[starts] != lead_id) | (values[starts] != lead_id))

    if wrong.size:
        index = int(wrong[0])
        raise BatchError(
            f"запись {index} начинается с ({int(keys[starts[index]])}, {int(values[starts[index]])}), "
            f"а ведущий токен обязан быть {lead_id} в обоих массивах"
        )

    late = np.flatnonzero(places[starts] != 0)

    if late.size:
        index = int(late[0])
        raise BatchError(f"запись {index}: ведущий токен стоит на позиции {int(places[starts[index]])}, а должен на 0")

    # --------------------------------------------------------

    rows = lengths.size
    width = int(lengths.max())

    mask = np.arange(width, dtype=np.int64)[None, :] >= lengths[:, None]

    padded_keys = np.full((rows, width), config.pad_id, dtype=np.int64)
    padded_values = np.full((rows, width), config.pad_id, dtype=np.int64)
    padded_positions = np.zeros((rows, width), dtype=np.int64)

    # Позиции настоящих токенов в порядке строк совпадают с
    # порядком плоского массива.
    slots = np.flatnonzero(~mask)

    padded_keys.ravel()[slots] = keys
    padded_values.ravel()[slots] = values
    padded_positions.ravel()[slots] = places

    return PaddedRecords(
        key_ids=torch.from_numpy(padded_keys),
        value_ids=torch.from_numpy(padded_values),
        positions=torch.from_numpy(padded_positions),
        padding_mask=torch.from_numpy(mask),
        lengths=torch.from_numpy(lengths),
    )


def events_from_batch(batch: TokenBatch, config: ModelConfig) -> PaddedRecords:

    check_batch(batch)

    return pad_flat(
        batch.key_ids,
        batch.value_ids,
        batch.positions,
        batch.event_offsets,
        lead_id=config.evt_id,
        config=config,
    )


def profiles_from_batch(batch: TokenBatch, config: ModelConfig) -> PaddedRecords:
    return pad_records(split_profiles(batch), lead_id=config.usr_id, config=config)
