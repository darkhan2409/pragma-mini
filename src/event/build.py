from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from src.dataset.lineage import lineage_problem
from src.embedding.inputs import Loaded, Source
from src.embedding.layer import InputEmbedding
from src.embedding.settings import WEIGHTS_FILE as EMBEDDING_WEIGHTS
from src.embedding.settings import embeddings_dir
from src.mlm.backbone import initial_event, payload
from src.preprocessing.artifacts import TableWriter
from src.tokenization.specials import EVT, USR, load_special_tokens

from .encoder import Encoded, EventEncoder
from .gather import Chunk, Events, calendar_of, chunks, gather
from .settings import EVENTS_FILE, WEIGHTS_FILE, EventConfig, events_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# У этапа два файла на выходе:
#
#   events.parquet — вектор каждого настоящего события группы;
#   weights.pt     — веса энкодера.
#
# СТРОКА ЭТО ОДНО СОБЫТИЕ. Не клиент: у клиента событий бывают
# десятки тысяч, и держать их одной ячейкой значило бы делать
# файл, который не открывается. Заполнителя в файле нет вовсе —
# только настоящие события, в хронологическом порядке, по клиенту
# подряд.
#
# Вектор один: итоговый, ПОСЛЕ прибавления календаря. Он же
# уходит в историю. Промежуточный вектор до календаря наружу не
# пишется.
#
# Ключи, значения, границы событий и маски здесь не дублируются —
# они лежат рядом в data/07_batches, и связь по паре
# (client_id, event).
#
# ВАЖНО, чем этот файл НЕ является. Векторы посчитаны начальным
# розыгрышем весов. При обучении они считаются заново, в прямом
# проходе; замороженным входом обучения файл не является.
#
# Этап — диагностика: обучению он не нужен, его начальные веса
# даёт python -m src.mlm.init_backbone без прохода по данным. Веса
# здесь разыгрываются той же функцией (backbone.initial_event),
# поэтому при одном конфиге они те же, что у init_backbone.
#
# Устройство — config.device: auto берёт CUDA, если она есть. Веса
# разыгрываются на CPU и только потом переезжают; векторы
# возвращаются на CPU перед записью.
# ============================================================


EVENTS_SCHEMA = pa.schema(
    [
        # --- где лежит событие ---
        ("batch_index", pa.int32()),
        ("client_id", pa.string()),
        ("event", pa.int32()),

        # --- когда оно случилось: чтобы файл читался глазами ---
        ("event_time", pa.timestamp("us", tz="UTC")),

        # --- итоговый вектор события, d чисел ---
        ("vector", pa.list_(pa.float32())),
    ]
)


class EventError(ValueError):
    """
    Векторы событий собрать нельзя.
    """


def build_group(
    group: str,
    config: EventConfig,
    directory: Path | None = None,
) -> dict:
    """
    Векторы событий всей группы.
    """

    source = Source(group, with_calendar=True)

    specials = load_special_tokens()

    device = _device(config.device)

    embedding = _embedding(group, specials).to(device)

    encoder = initial_event(config, embedding.dim)

    encoder.eval()

    encoder.to(device)

    directory = Path(directory) if directory is not None else events_dir(group)

    _clear(directory)

    table_path = directory / EVENTS_FILE
    weights_path = directory / WEIGHTS_FILE

    writer = TableWriter(table_path, EVENTS_SCHEMA)

    clients = 0
    counted = 0
    tokens = 0

    try:
        for number in range(source.count):

            loaded = source.batch(number)

            events = gather(loaded)

            with torch.no_grad():
                dated = encode(embedding, encoder, loaded, events, config)

            writer.write(_table(loaded, events, dated))

            clients += loaded.model.clients
            counted += events.count
            tokens += events.tokens

    finally:
        rows = writer.close()

    _save(encoder, config, embedding.dim, weights_path)

    return {
        "group": group,
        "table": str(table_path),
        "weights": str(weights_path),
        "dim": embedding.dim,
        "seed": config.seed,
        "layers": config.layers,
        "heads": config.heads,
        "device": str(device),
        "rows": rows,
        "batches": source.count,
        "clients": clients,
        "events": counted,
        "tokens": tokens,
        "size": table_path.stat().st_size,
    }


def encode(
    embedding: InputEmbedding,
    encoder: EventEncoder,
    loaded: Loaded,
    events: Events,
    config: EventConfig,
    sort: bool = True,
) -> np.ndarray:
    """
    Итоговые векторы всех настоящих событий батча, в порядке
    плоского списка.

    Порядок ОБХОДА при этом другой: события идут от коротких к
    длинным, и каждая порция кладётся на свои места по chunk.where.
    """

    dated = np.zeros((events.count, embedding.dim), dtype=np.float32)

    for chunk in chunks(events, config.events_per_chunk, sort=sort):

        dated[chunk.where] = encode_chunk(
            embedding, encoder, loaded, chunk
        ).dated.detach().cpu().numpy()

    return dated


def encode_chunk(
    embedding: InputEmbedding,
    encoder: EventEncoder,
    loaded: Loaded,
    chunk: Chunk,
) -> Encoded:
    """
    Одна порция событий через оба слоя, на устройстве энкодера.

    Порция собирается на CPU — батч лежит там — и переезжает
    целиком: внутри слоёв копий между устройствами нет.
    """

    model = loaded.model

    device = next(encoder.parameters()).device

    rows = torch.from_numpy(chunk.client)[:, None]
    column = torch.from_numpy(chunk.column)
    pad = torch.from_numpy(chunk.pad)

    tokens = embedding.embed(
        model.key_ids[rows, column].to(device),
        model.value_ids[rows, column].to(device),
        model.positions[rows, column].to(device),
        (~pad).to(device),
    )

    calendar = torch.from_numpy(calendar_of(loaded.calendar, chunk)).to(device)

    return encoder(tokens, pad.to(device), calendar)


def _table(loaded: Loaded, events: Events, dated: np.ndarray) -> pa.Table:
    """
    Строки одного батча: по настоящему событию на строку.
    """

    model = loaded.model

    moments = [
        loaded.rows[int(client)]["event_time"][int(slot)]
        for client, slot in zip(events.client, events.slot)
    ]

    return pa.table(
        {
            "batch_index": pa.array([model.batch_index] * events.count, pa.int32()),
            "client_id": pa.array(
                [model.client_ids[int(client)] for client in events.client], pa.string()
            ),
            "event": pa.array([int(slot) for slot in events.slot], pa.int32()),
            "event_time": pa.array(moments, pa.timestamp("us", tz="UTC")),
            "vector": pa.array(list(dated), type=pa.list_(pa.float32())),
        },
        schema=EVENTS_SCHEMA,
    )


def _embedding(group: str, specials: dict) -> InputEmbedding:
    """
    Входной слой этапа 09 со своими весами.
    """

    path = embeddings_dir(group) / EMBEDDING_WEIGHTS

    if not path.exists():
        raise EventError(f"нет {path}: выполните python -m src.embedding.run {group}")

    problem = lineage_problem(path.parent, f"python -m src.embedding.run {group}")

    if problem:
        raise EventError(problem)

    saved = torch.load(path, map_location="cpu", weights_only=True)

    layer = InputEmbedding(
        vocab_size=int(saved["vocab_size"]),
        dim=int(saved["dim"]),
        seed=int(saved["seed"]),
        markers=(specials[EVT], specials[USR]),
    )

    layer.load_state_dict(saved["state_dict"])

    layer.eval()

    return layer


def _save(encoder: EventEncoder, config: EventConfig, dim: int, path: Path) -> None:
    """
    Веса энкодера рядом с векторами — в формате backbone.payload.

    Веса входного слоя сюда не копируются: они лежат в
    data/09_embeddings и остаются одним файлом на всю модель.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(payload(encoder, config, dim), path)


def _device(name: str) -> torch.device:
    """
    Где считать.
    """

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise EventError("device cuda запрошен, но CUDA недоступна")

    return torch.device(name)


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только свои два файла: прежний
    результат стирается целиком.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "EVENTS_SCHEMA",
    "EventError",
    "build_group",
    "encode",
    "encode_chunk",
]
