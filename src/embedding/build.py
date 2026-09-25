from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from src.dataset.lineage import write_lineage
from src.preprocessing.artifacts import TableWriter
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import EVT, USR, load_special_tokens

from .inputs import Loaded, Source
from .layer import InputEmbedding
from .settings import EMBEDDINGS_FILE, WEIGHTS_FILE, EmbeddingConfig, embeddings_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# У этапа два файла на выходе:
#
#   embeddings.parquet — векторы всех токенов группы, батч за
#       батчем;
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
    directory: Path | None = None,
) -> dict:
    """
    Векторы всей группы.
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

    directory = Path(directory) if directory is not None else embeddings_dir(group)

    _clear(directory)

    table_path = directory / EMBEDDINGS_FILE
    weights_path = directory / WEIGHTS_FILE

    writer = TableWriter(table_path, EMBEDDINGS_SCHEMA)

    clients = 0
    vectors = 0

    try:
        for number in range(source.count):

            loaded = source.batch(number)

            with torch.no_grad():
                tokens, profile = _vectors(layer, loaded)

            writer.write(_table(loaded, tokens, profile, config.dim))

            clients += loaded.model.clients
            vectors += loaded.model.clients * (
                loaded.model.width + loaded.model.profile_width
            )

    finally:
        rows = writer.close()

    _save(layer, config, vocab.size, weights_path)

    # Только после полной записи. Таблица этапа собрана под
    # словарь и анкету текущего кода, и потребители весов это
    # сверяют: веса прежнего словаря иначе читались бы молча.
    write_lineage(directory)

    return {
        "group": group,
        "table": str(table_path),
        "weights": str(weights_path),
        "dim": config.dim,
        "seed": config.seed,
        "vocab_size": vocab.size,
        "batches": source.count,
        "rows": rows,
        "clients": clients,
        "vectors": vectors,
        "size": table_path.stat().st_size,
    }


def _vectors(layer: InputEmbedding, loaded: Loaded) -> tuple[list, list]:
    """
    Векторы всех токенов батча, по клиенту на элемент.

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

    return tokens, profile


def _flat(vectors: torch.Tensor) -> np.ndarray:
    """
    [1, N, d] -> N * d чисел подряд.
    """

    return vectors[0].reshape(-1).numpy()


def _table(loaded: Loaded, tokens: list, profile: list, dim: int) -> pa.Table:
    """
    Строки одного батча: по клиенту на строку.
    """

    model = loaded.model

    return pa.table(
        {
            "batch_index": pa.array([model.batch_index] * model.clients, pa.int32()),
            "client_id": pa.array(list(model.client_ids), pa.string()),
            "dim": pa.array([dim] * model.clients, pa.int32()),
            "tokens": pa.array(tokens, type=pa.large_list(pa.float32())),
            "profile": pa.array(profile, type=pa.large_list(pa.float32())),
        },
        schema=EMBEDDINGS_SCHEMA,
    )


def _save(layer: InputEmbedding, config: EmbeddingConfig, vocab_size: int,
          path: Path) -> None:
    """
    Веса инициализированного слоя.

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
    Каталог группы держит только свои файлы: прежний результат
    стирается целиком.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "EMBEDDINGS_SCHEMA",
    "build_group",
]
