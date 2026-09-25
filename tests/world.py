from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa

from src.dataset.lineage import write_lineage
from src.batching.build import BATCHES_SCHEMA
from src.embedding.layer import InputEmbedding
from src.embedding.settings import EmbeddingConfig
from src.event.encoder import EventEncoder
from src.event.settings import EventConfig
from src.history.encoder import HistoryEncoder
from src.history.settings import HistoryConfig
from src.masking.build import MASKED_SCHEMA
from src.mlm.inputs import IGNORE, Client
from src.mlm.model import Mlm, Model
from src.preprocessing.artifacts import TableWriter
from src.profile.encoder import ProfileEncoder
from src.profile.settings import ProfileConfig
from src.tokenization.specials import SPECIAL_TOKENS_FILE, build_special_tokens


# ============================================================
# ИДЕЯ
# ============================================================
#
# Тестам нужен не датасет, а мир из нескольких клиентов, у
# которого видно каждое число. Клиент собирается прямо в памяти:
# Client это обычный frozen dataclass, и единственное, что его
# ограничивает, — инварианты pack и VarlenLayout.
#
# Те же клиенты умеют лечь в parquet схемами боя. Тогда Source,
# micro_batches, load_model и настоящий train читают файлы, а не
# заглушки, и проверяется в том числе чтение.
#
# Размеры крошечные намеренно: dim 8 при двух головах даёт
# head_dim 4, чётный, как требует TimeRoPE. Словарь на 32 номера
# позволяет писать ожидаемые числа руками.
# ============================================================


PAD, UNK, MASK, EVT, USR = 0, 1, 2, 3, 4

VOCAB = 64

# Сколько номеров словаря занимают куски BPE.
PIECES = 2

DIM = 8
HEADS = 2
LAYERS = 1
FEEDFORWARD = 16
ROPE_BASE = 10000.0

SEED = 7

# Ключи мира. Номера начинаются после специальных.
KEY_A, KEY_B, KEY_C = 5, 6, 7

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ============================================================
# КЛИЕНТ
# ============================================================
#
# Событие описывается списком полей, поле — тройкой
# (ключ, куски значения, цель ли это). Маркер [EVT] ставится сам
# и занимает оба слота; анкета так же начинается с [USR].
# ============================================================

Field = tuple[int, list[int], bool]
Event = list[Field]


@dataclass(frozen=True)
class Made:
    """
    Клиент и то, что маскирование от него скрыло.
    """

    client: Client
    value_ids_source: np.ndarray

    # Разрешено ли маскировать события этого клиента. False даёт
    # клиента, у которого целей не бывает ни при какой маске.
    targetable: bool = True


def make(
    client_id: str,
    events: list[Event],
    profile: list[tuple[int, list[int]]],
    *,
    batch_index: int = 0,
    targetable: bool = True,
) -> Made:
    """
    Клиент из описания событий и анкеты.

    События кладутся подряд, event_starts — исключающая сумма
    длин: ровно то, чего требует pack. Время убывает к нулю у
    самого свежего события, как в бою.
    """

    key_ids: list[int] = []
    value_ids: list[int] = []
    source: list[int] = []
    positions: list[int] = []
    labels: list[int] = []
    reason: list[str] = []

    starts: list[int] = []
    lengths: list[int] = []

    for fields in events:

        starts.append(len(key_ids))

        key_ids.append(EVT)
        value_ids.append(EVT)
        source.append(EVT)
        positions.append(0)
        labels.append(IGNORE)
        reason.append("")

        for key, pieces, target in fields:

            for number, piece in enumerate(pieces):

                key_ids.append(key)
                positions.append(number)
                source.append(piece)

                if target:
                    value_ids.append(MASK)
                    labels.append(piece)
                    reason.append("value")
                else:
                    value_ids.append(piece)
                    labels.append(IGNORE)
                    reason.append("")

        lengths.append(len(key_ids) - starts[-1])

    profile_key_ids: list[int] = [USR]
    profile_value_ids: list[int] = [USR]
    profile_positions: list[int] = [0]

    for key, pieces in profile:
        for number, piece in enumerate(pieces):
            profile_key_ids.append(key)
            profile_value_ids.append(piece)
            profile_positions.append(number)

    count = len(events)

    # Лог-секунды до последнего события: убывают к нулю.
    event_time_log = np.array(
        [float(count - 1 - number) for number in range(count)], dtype=np.float32
    )

    client = Client(
        batch_index=batch_index,
        client_id=client_id,
        key_ids=np.asarray(key_ids, dtype=np.int64),
        value_ids=np.asarray(value_ids, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        reason=reason,
        event_starts=np.asarray(starts, dtype=np.int64),
        event_lengths=np.asarray(lengths, dtype=np.int64),
        event_time_log=event_time_log,
        calendar=calendar_of(count),
        event_time=[EPOCH + timedelta(hours=number) for number in range(count)],
        profile_key_ids=np.asarray(profile_key_ids, dtype=np.int64),
        profile_value_ids=np.asarray(profile_value_ids, dtype=np.int64),
        profile_positions=np.asarray(profile_positions, dtype=np.int64),
    )

    return Made(
        client=client,
        value_ids_source=np.asarray(source, dtype=np.int64),
        targetable=targetable,
    )


def calendar_of(count: int) -> np.ndarray:
    """
    Шесть чисел календаря на событие: три пары sin/cos.

    Значения не настоящие, но лежат там же, где настоящие, — на
    единичных окружностях, и различаются от события к событию.
    """

    number = np.arange(count, dtype=np.float32)[:, None]

    angles = number * np.array([0.7, 1.3, 2.1], dtype=np.float32)

    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)


# ============================================================
# НАБОРЫ КЛИЕНТОВ
# ============================================================


def population(prefix: str = "c") -> list[Made]:
    """
    Пять клиентов разной формы.

    Разная форма здесь и есть смысл: одно событие и много,
    значение из одного куска и из трёх, клиент вовсе без целей,
    событие из одного маркера, анкета из одного [USR].
    """

    return [
        # Многокусковое значение и цель в середине ленты.
        make(
            f"{prefix}-long",
            [
                [(KEY_A, [10], False), (KEY_B, [11, 12, 13], True)],
                [(KEY_A, [14], True)],
                [(KEY_C, [15, 16], False)],
            ],
            [(KEY_A, [20]), (KEY_B, [21, 22])],
        ),
        # Одно событие, одна цель.
        make(f"{prefix}-one", [[(KEY_C, [17], True)]], [(KEY_A, [23])]),
        # Целей нет вовсе.
        make(
            f"{prefix}-quiet",
            [[(KEY_A, [18], False)], [(KEY_B, [19], False)]],
            [(KEY_C, [24])],
        ),
        # Событие из одного маркера рядом с обычным.
        make(
            f"{prefix}-bare",
            [[], [(KEY_A, [25], True)], []],
            [(KEY_A, [26])],
        ),
        # Анкета из одного [USR] и две цели подряд.
        make(
            f"{prefix}-thin",
            [[(KEY_A, [27], True), (KEY_B, [28, 29], True)]],
            [],
        ),
    ]


def clients(prefix: str = "c") -> list[Client]:
    return [made.client for made in population(prefix)]


# ============================================================
# PARQUET СХЕМАМИ БОЯ
# ============================================================


def write_batches(path: Path, batches: list[list[Made]]) -> None:
    """
    batches.parquet: одна группа строк на батч, ряды выровнены.

    Ширина группы — максимум по её клиентам, как делает этап 07:
    настоящие длины лежат в n_tokens, n_events и profile_n_tokens,
    а хвост закрыт масками.
    """

    writer = TableWriter(path, BATCHES_SCHEMA)

    try:
        for index, batch in enumerate(batches):
            writer.write(_batch_table(index, batch))
    finally:
        writer.close()

    # Отметка происхождения, как у настоящего этапа: без неё
    # читатель каталог отвергнет.
    write_lineage(path.parent)


def write_masked(path: Path, batches: list[list[Made]]) -> None:
    """
    masked.parquet: те же батчи, те же группы строк.
    """

    writer = TableWriter(path, MASKED_SCHEMA)

    try:
        for index, batch in enumerate(batches):
            writer.write(_masked_table(index, batch))
    finally:
        writer.close()

    # Отметка происхождения, как у настоящего этапа: без неё
    # читатель каталог отвергнет.
    write_lineage(path.parent)


def _batch_table(index: int, batch: list[Made]) -> pa.Table:

    width = max(made.client.n_tokens for made in batch)
    events = max(made.client.n_events for made in batch)
    profile = max(made.client.profile_n_tokens for made in batch)

    columns: dict[str, list] = {name: [] for name in BATCHES_SCHEMA.names}

    for made in batch:

        client = made.client

        columns["batch_index"].append(index)
        columns["client_id"].append(client.client_id)
        columns["n_tokens"].append(client.n_tokens)
        columns["n_events"].append(client.n_events)
        columns["profile_n_tokens"].append(client.profile_n_tokens)

        columns["key_ids"].append(_pad(client.key_ids, width, PAD))
        columns["value_ids"].append(_pad(made.value_ids_source, width, PAD))
        columns["positions"].append(_pad(client.positions, width, 0))
        columns["token_mask"].append(_flags(client.n_tokens, width))

        columns["event_starts"].append(_pad(client.event_starts, events, 0))
        columns["event_lengths"].append(_pad(client.event_lengths, events, 0))
        columns["event_time"].append(
            list(client.event_time) + [None] * (events - client.n_events)
        )
        columns["event_time_log"].append(_pad(client.event_time_log, events, 0.0))
        columns["calendar"].append(
            _pad(client.calendar.reshape(-1), events * 6, 0.0)
        )
        columns["event_mask"].append(_flags(client.n_events, events))
        columns["target_event_mask"].append(
            _flags(client.n_events if made.targetable else 0, events)
        )

        columns["profile_key_ids"].append(_pad(client.profile_key_ids, profile, PAD))
        columns["profile_value_ids"].append(
            _pad(client.profile_value_ids, profile, PAD)
        )
        columns["profile_positions"].append(_pad(client.profile_positions, profile, 0))
        columns["profile_token_mask"].append(
            _flags(client.profile_n_tokens, profile)
        )

    return pa.table(columns, schema=BATCHES_SCHEMA)


def _masked_table(index: int, batch: list[Made]) -> pa.Table:

    width = max(made.client.n_tokens for made in batch)

    columns: dict[str, list] = {name: [] for name in MASKED_SCHEMA.names}

    for made in batch:

        client = made.client

        columns["batch_index"].append(index)
        columns["client_id"].append(client.client_id)
        columns["value_ids_source"].append(_pad(made.value_ids_source, width, PAD))
        columns["value_ids"].append(_pad(client.value_ids, width, PAD))
        columns["labels"].append(_pad(client.labels, width, IGNORE))
        columns["reason"].append(
            list(client.reason) + [""] * (width - client.n_tokens)
        )

    return pa.table(columns, schema=MASKED_SCHEMA)


def _pad(values: np.ndarray, width: int, filler) -> list:

    tail = [filler] * (width - int(values.size))

    return list(values.tolist()) + tail


def _flags(length: int, width: int) -> list[bool]:
    return [True] * length + [False] * (width - length)


# ============================================================
# МОДЕЛЬ И ВЕСА
# ============================================================


def embedding(vocab: int = VOCAB, dim: int = DIM, seed: int = SEED) -> InputEmbedding:
    return InputEmbedding(vocab_size=vocab, dim=dim, seed=seed, markers=(EVT, USR))


def model(
    *,
    vocab: int = VOCAB,
    dim: int = DIM,
    heads: int = HEADS,
    layers: int = LAYERS,
    dropout: float = 0.0,
    seed: int = SEED,
    events_per_chunk: int = 512,
    label_smoothing: float = 0.1,
    attention: str = "sdpa",
) -> Model:
    """
    Все пять частей напрямую, без load_model и без data/.
    """

    return Model(
        embedding=embedding(vocab, dim, seed),
        event=EventEncoder(dim, layers, heads, FEEDFORWARD, dropout, seed),
        profile=ProfileEncoder(dim, layers, heads, FEEDFORWARD, dropout, seed),
        history=HistoryEncoder(dim, layers, heads, FEEDFORWARD, dropout, ROPE_BASE, seed),
        head=Mlm(dim, seed),
        events_per_chunk=events_per_chunk,
        label_smoothing=label_smoothing,
        attention=attention,
    )


def install(
    root: Path,
    groups: dict[str, list[list[Made]]],
    *,
    dropout: float = 0.0,
) -> None:
    """
    Весь мир на диск: батчи, маски и веса этапов 09-12.

    Вызывается ПОСЛЕ подмены каталогов: batches_dir и masked_dir
    читают свои глобалы в момент вызова.
    """

    from src.batching.settings import BATCHES_FILE, batches_dir
    from src.masking.settings import MASKED_FILE, masked_dir

    for group, batches in groups.items():

        write_batches(batches_dir(group) / BATCHES_FILE, batches)
        write_masked(masked_dir(group) / MASKED_FILE, batches)
        write_weights(root, group, dropout=dropout)


def write_weights(
    root: Path,
    group: str,
    *,
    vocab: int = VOCAB,
    dim: int = DIM,
    heads: int = HEADS,
    layers: int = LAYERS,
    dropout: float = 0.0,
    seed: int = SEED,
) -> None:
    """
    Веса этапов 09-12 теми ключами, которых ждёт load_model.

    Каждый файл несёт и конфигурацию, и состояние: размерности в
    конфиге этапа 13 не дублируются, и разойтись им негде.
    """

    import torch

    parts = {
        "09_embeddings": (
            {
                "vocab_size": vocab,
                "dim": dim,
                "seed": seed,
                "state_dict": embedding(vocab, dim, seed).state_dict(),
            },
        ),
        "10_events": (
            {
                "dim": dim,
                "config": EventConfig(
                    seed=seed, layers=layers, heads=heads,
                    feedforward=FEEDFORWARD, dropout=dropout,
                ).as_dict(),
                "state_dict": EventEncoder(
                    dim, layers, heads, FEEDFORWARD, dropout, seed
                ).state_dict(),
            },
        ),
        "11_profiles": (
            {
                "dim": dim,
                "config": ProfileConfig(
                    seed=seed, layers=layers, heads=heads,
                    feedforward=FEEDFORWARD, dropout=dropout,
                ).as_dict(),
                "state_dict": ProfileEncoder(
                    dim, layers, heads, FEEDFORWARD, dropout, seed
                ).state_dict(),
            },
        ),
        "12_history": (
            {
                "dim": dim,
                "config": HistoryConfig(
                    seed=seed, layers=layers, heads=heads,
                    feedforward=FEEDFORWARD, dropout=dropout, rope_base=ROPE_BASE,
                ).as_dict(),
                "state_dict": HistoryEncoder(
                    dim, layers, heads, FEEDFORWARD, dropout, ROPE_BASE, seed
                ).state_dict(),
            },
        ),
    }

    for name, (saved,) in parts.items():

        directory = root / name / group
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(saved, directory / "weights.pt")


def write_vocab(root: Path) -> None:
    """
    Настоящий словарь из шести файлов, только крошечный.

    Модель читает из него лишь специальные токены, но этап 09
    берёт размер словаря через FrozenArtifacts, а тот сверяет все
    шесть файлов между собой. Поэтому словарь собирается тем же
    build_final_vocab, что и в бою: номера идут подряд, специальные
    занимают 0-4, куски BPE замыкают пространство.
    """

    from tokenizers import Tokenizer, models

    from src.tokenization.finalvocab import build_final_vocab
    from src.tokenization.settings import (
        BPE_FILE, BUCKETS_FILE, FINAL_VOCAB_FILE, KEY_VOCAB_FILE, VALUE_VOCAB_FILE,
    )
    from src.tokenization.text import BpeModel

    directory = root / "03_vocab"
    directory.mkdir(parents=True, exist_ok=True)

    specials = build_special_tokens()

    keys = {"key_a": KEY_A, "key_b": KEY_B, "key_c": KEY_C}

    # Значения заполняют пространство до последних двух номеров,
    # которые занимают куски BPE.
    free = VOCAB - len(specials) - len(keys) - PIECES
    share = free // len(keys)

    values = {
        name: {
            f"v{number}": len(specials) + len(keys) + place * share + number
            for number in range(share)
        }
        for place, name in enumerate(keys)
    }

    pieces = {"ab": 0, "cd": 1}

    bpe = BpeModel(tokenizer=Tokenizer(models.BPE(vocab=pieces, merges=[])))

    vocab = build_final_vocab(specials, keys, values, {}, bpe)

    for name, data in (
        (FINAL_VOCAB_FILE, vocab),
        (SPECIAL_TOKENS_FILE, specials),
        (KEY_VOCAB_FILE, keys),
        (VALUE_VOCAB_FILE, values),
        (BUCKETS_FILE, {}),
    ):
        (directory / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    bpe.save(directory / BPE_FILE)


# ============================================================
# СРЕДА
# ============================================================


def cuda_ready() -> bool:
    """
    Есть ли CUDA, которая умеет bf16: без неё autocast обучения не
    включается, и сравнивать нечего.
    """

    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def flash_ready() -> bool:
    """
    Есть ли CUDA и библиотека flash-attn.
    """

    from src.mlm.varlen import flash_available

    return cuda_ready() and flash_available()


__all__ = [
    "DIM", "EVT", "FEEDFORWARD", "HEADS", "IGNORE", "KEY_A", "KEY_B", "KEY_C",
    "LAYERS", "MASK", "PAD", "ROPE_BASE", "SEED", "UNK", "USR", "VOCAB",
    "Made", "calendar_of", "clients", "cuda_ready", "embedding",
    "flash_ready", "make", "model",
    "install", "population", "write_batches", "write_masked", "write_vocab",
    "write_weights",
]
