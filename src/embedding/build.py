from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from src.preprocessing.artifacts import TableWriter, write_text
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import EVT, MASK, UNK, USR, load_special_tokens

from .inputs import Loaded, Source
from .layer import InputEmbedding
from .report import Names, Page, Shapes, render
from .select import Shown, select
from .settings import (
    EMBEDDINGS_FILE,
    PREVIEW_FILE,
    WEIGHTS_FILE,
    EmbeddingConfig,
    embeddings_dir,
)
from .version import IMPLEMENTATION_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# У этапа три файла на выходе, и путать их нельзя:
#
#   embeddings.parquet — настоящий выход: векторы всех токенов
#       группы, батч за батчем;
#   preview.html       — та же работа, показанная человеку на
#       одном событии;
#   weights.pt         — веса, которыми эти векторы посчитаны.
#
# Строка это клиент, и это ТА ЖЕ строка, что в batches.parquet и
# masked.parquet: тот же порядок, та же группа строк на батч.
# Поэтому ключи, маски, границы событий и время здесь не
# дублируются — они уже лежат рядом.
#
# ВАЖНО, чем этот файл НЕ является. Векторы посчитаны весами из
# weights.pt, то есть начальным розыгрышем. При обучении веса
# меняются на каждом шаге, и модель считает эмбеддинги сама, в
# прямом проходе. Этот файл — снимок входа, а не замена forward.
#
# Заполнитель лежит в файле как есть, нулями: ширина строки та же,
# что в батче, и читать её можно теми же масками. Нули сжимаются
# почти в ничто, поэтому платы за честную ширину почти нет.
# ============================================================


EMBEDDINGS_SCHEMA = pa.schema(
    [
        # --- где лежит клиент ---
        ("batch_index", pa.int32()),
        ("client_id", pa.string()),

        # --- длина одного вектора; ширина строки выводится делением ---
        ("dim", pa.int32()),

        # Вектора хранятся плоско, подряд по токенам: T * dim
        # чисел и P * dim чисел. large_list, а не list: у обычного
        # смещения 32-битные, а тут счёт идёт на сотни миллионов
        # чисел в одной группе строк.
        ("tokens", pa.large_list(pa.float32())),
        ("profile", pa.large_list(pa.float32())),
    ]
)


def build_group(
    group: str,
    config: EmbeddingConfig,
    index: int = 0,
    directory: Path | None = None,
) -> dict:
    """
    Векторы всей группы и отчёт по одному её батчу.
    """

    source = Source(group)

    vocab = FrozenArtifacts.load()

    specials = load_special_tokens()

    layer = InputEmbedding(
        vocab_size=vocab.size,
        dim=config.dim,
        seed=config.seed,
        markers=(specials[EVT], specials[USR]),
    )

    layer.eval()

    if index < 0 or index >= source.count:
        index = 0

    directory = Path(directory) if directory is not None else embeddings_dir(group)

    _clear(directory)

    table_path = directory / EMBEDDINGS_FILE
    preview_path = directory / PREVIEW_FILE
    weights_path = directory / WEIGHTS_FILE

    writer = TableWriter(table_path, EMBEDDINGS_SCHEMA)

    shown_batch: Loaded | None = None

    clients = 0
    numbers = 0

    try:
        for number in range(source.count):

            loaded = source.batch(number)

            with torch.no_grad():
                writer.write(_table(layer, loaded, config.dim))

            clients += loaded.model.clients
            numbers += loaded.model.clients * (
                loaded.model.width + loaded.model.profile_width
            ) * config.dim

            if number == index:
                shown_batch = loaded

    finally:
        rows = writer.close()

    _save(layer, config, vocab.size, weights_path)

    shown = _preview(
        group, index, config, vocab, specials, layer, shown_batch,
        preview_path, table_path, weights_path,
    )

    return {
        "group": group,
        "batch": index,
        "table": str(table_path),
        "preview": str(preview_path),
        "weights": str(weights_path),
        "dim": config.dim,
        "seed": config.seed,
        "vocab_size": vocab.size,
        "batches": source.count,
        "rows": rows,
        "clients": clients,
        "numbers": numbers,
        "size": table_path.stat().st_size,
        "width": shown_batch.model.width,
        "profile_width": shown_batch.model.profile_width,
        "compared_values": shown_batch.checks.compared_values,
        "markers": shown_batch.checks.markers,
        "pad_slots": shown_batch.checks.pad_slots,
        "client_id": shown.client_id,
        "event": shown.event,
        "note": shown.note,
    }


def _table(layer: InputEmbedding, loaded: Loaded, dim: int) -> pa.Table:
    """
    Строки одного батча.

    Слой считается по одному клиенту: батч целиком это сотни
    мегабайт, и держать их в памяти незачем — в файл они уходят
    всё равно построчно.
    """

    model = loaded.model

    tokens = []
    profile = []

    for client in range(model.clients):

        tokens.append(
            _flat(
                layer.embed(
                    model.key_ids[client: client + 1],
                    model.value_ids[client: client + 1],
                    model.positions[client: client + 1],
                    model.token_mask[client: client + 1],
                )
            )
        )

        profile.append(
            _flat(
                layer.embed(
                    model.profile_key_ids[client: client + 1],
                    model.profile_value_ids[client: client + 1],
                    model.profile_positions[client: client + 1],
                    model.profile_token_mask[client: client + 1],
                )
            )
        )

    return pa.table(
        {
            "batch_index": pa.array(
                [model.batch_index] * model.clients, pa.int32()
            ),
            "client_id": pa.array(list(model.client_ids), pa.string()),
            "dim": pa.array([dim] * model.clients, pa.int32()),
            "tokens": pa.array(tokens, type=pa.large_list(pa.float32())),
            "profile": pa.array(profile, type=pa.large_list(pa.float32())),
        },
        schema=EMBEDDINGS_SCHEMA,
    )


def _flat(vectors: torch.Tensor) -> np.ndarray:
    """
    [1, N, d] -> N * d чисел подряд.
    """

    return vectors[0].reshape(-1).numpy()


def _preview(
    group: str,
    index: int,
    config: EmbeddingConfig,
    vocab: FrozenArtifacts,
    specials: dict,
    layer: InputEmbedding,
    loaded: Loaded,
    preview_path: Path,
    table_path: Path,
    weights_path: Path,
) -> Shown:
    """
    Страница для человека по одному батчу.

    Слой применяется ТОЛЬКО к показанным позициям: их несколько
    десятков, и пересчитывать ради картинки весь батч незачем.
    """

    shown = select(
        rows=loaded.rows,
        key_ids=loaded.model.key_ids.numpy(),
        visible=loaded.model.value_ids.numpy(),
        positions=loaded.model.positions.numpy(),
        mask_id=specials[MASK],
        unknown_id=specials[UNK],
    )

    places = _places(shown)
    profile_places = list(shown.profile)

    with torch.no_grad():
        vectors = _slice(layer, loaded, shown.client, places, profile=False)
        profile_vectors = _slice(
            layer, loaded, shown.client, profile_places, profile=True
        )

    client = shown.client

    page = Page(
        names=Names(vocab),
        shown=shown,
        vectors=vectors,
        places=places,
        profile_vectors=profile_vectors,
        profile_places=profile_places,
        key_ids=loaded.model.key_ids[client].numpy(),
        visible=loaded.model.value_ids[client].numpy(),
        source=loaded.source_value_ids[client],
        positions=loaded.model.positions[client].numpy(),
        profile_key_ids=loaded.model.profile_key_ids[client].numpy(),
        profile_value_ids=loaded.model.profile_value_ids[client].numpy(),
        profile_positions=loaded.model.profile_positions[client].numpy(),
    )

    write_text(
        preview_path,
        render(
            group=group,
            index=index,
            config=config,
            implementation=IMPLEMENTATION_VERSION,
            names=page.names,
            checks=loaded.checks,
            shapes=Shapes(
                clients=loaded.model.clients,
                width=loaded.model.width,
                profile_width=loaded.model.profile_width,
                dim=config.dim,
            ),
            shown=shown,
            layer=layer,
            page=page,
            batches_path=loaded.batches_path,
            masked_path=loaded.masked_path,
            table_path=table_path,
            weights_path=weights_path,
        ),
    )

    return shown


def _places(shown: Shown) -> list[int]:
    """
    Позиции, которые попадут в отчёт, в порядке чтения.

    Этот же список и есть всё, что считает слой ради страницы.
    """

    places = [shown.marker]

    for value in shown.values:
        places.extend(value.places)

    if shown.pad is not None:
        places.append(shown.pad)

    seen: set[int] = set()

    return [place for place in places if not (place in seen or seen.add(place))]


def _slice(layer: InputEmbedding, loaded: Loaded, client: int, places: list[int],
           profile: bool) -> torch.Tensor:
    """
    Векторы узкого среза одного клиента.
    """

    model = loaded.model

    index = torch.tensor(places, dtype=torch.int64)

    if profile:
        return layer.embed(
            model.profile_key_ids[client, index].unsqueeze(0),
            model.profile_value_ids[client, index].unsqueeze(0),
            model.profile_positions[client, index].unsqueeze(0),
            model.profile_token_mask[client, index].unsqueeze(0),
        )

    return layer.embed(
        model.key_ids[client, index].unsqueeze(0),
        model.value_ids[client, index].unsqueeze(0),
        model.positions[client, index].unsqueeze(0),
        model.token_mask[client, index].unsqueeze(0),
    )


def _save(layer: InputEmbedding, config: EmbeddingConfig, vocab_size: int,
          path: Path) -> None:
    """
    Веса, которыми посчитан выход.

    В файле ровно одна таблица: лестница частот и номера маркеров
    задаются формулой и конструктором, и хранить их незачем.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "vocab_size": vocab_size,
            "dim": config.dim,
            "seed": config.seed,
            "state_dict": layer.state_dict(),
        },
        path,
    )


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только свои три файла: прежний
    результат стирается целиком.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "EMBEDDINGS_SCHEMA",
    "build_group",
]
