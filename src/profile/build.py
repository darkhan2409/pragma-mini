from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from src.dataset.lineage import lineage_problem, write_lineage
from src.embedding.inputs import BatchInput, Loaded, Source
from src.embedding.layer import InputEmbedding
from src.embedding.settings import WEIGHTS_FILE as EMBEDDING_WEIGHTS
from src.embedding.settings import embeddings_dir
from src.preprocessing.artifacts import TableWriter
from src.tokenization.specials import EVT, USR, load_special_tokens

from .encoder import ProfileEncoder
from .settings import PROFILES_FILE, WEIGHTS_FILE, ProfileConfig, profiles_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# У этапа два файла на выходе:
#
#   profiles.parquet — один вектор на клиента;
#   weights.pt       — веса энкодера, которыми выход посчитан.
#
# Строка это клиент, и это ТА ЖЕ строка, что в batches.parquet:
# тот же порядок, та же группа строк на батч. Сами токены анкеты
# здесь не дублируются — они лежат рядом в 07, вместе с временем
# каждого токена (profile_time_log): у вех это давность до cutoff,
# у [USR] и Attributes ноль.
#
# Рядом с весами лежит lineage.json: веса собраны под анкету и
# словарь текущего кода, и потребители это сверяют.
#
# Входной слой не разыгрывается заново: он грузится из
# data/09_embeddings/<group>/weights.pt вместе с vocab_size, dim
# и seed. Оттуда же берётся d.
#
# ВАЖНО, чем этот файл НЕ является. Векторы посчитаны начальным
# розыгрышем весов обоих слоёв. При обучении веса меняются на
# каждом шаге, и вектор клиента считается заново, вместе с
# InputEmbedding. Файл — снимок для просмотра, а не вход обучения.
# ============================================================


PROFILES_SCHEMA = pa.schema(
    [
        # --- где лежит клиент ---
        ("batch_index", pa.int32()),
        ("client_id", pa.string()),

        # --- длина вектора ---
        ("dim", pa.int32()),

        # Ровно d чисел на клиента, поэтому обычный list_:
        # large_list нужен там, где счёт идёт на сотни миллионов.
        ("profile", pa.list_(pa.float32())),
    ]
)


class ProfileError(ValueError):
    """
    Векторы клиентов собрать нельзя.
    """


def build_group(
    group: str,
    config: ProfileConfig,
    directory: Path | None = None,
) -> dict:
    """
    Векторы клиентов всей группы.
    """

    # Календарь не просится: анкета его не видит. Время её
    # токенов — просится.
    source = Source(group, with_profile_time=True)

    specials = load_special_tokens()

    embedding = _embedding(group, specials)

    config.check_dim(embedding.dim)

    encoder = ProfileEncoder(
        dim=embedding.dim,
        layers=config.layers,
        heads=config.heads,
        feedforward=config.feedforward,
        dropout=config.dropout,
        rope_base=config.rope_base,
        seed=config.seed,
    )

    encoder.eval()

    directory = Path(directory) if directory is not None else profiles_dir(group)

    _clear(directory)

    table_path = directory / PROFILES_FILE
    weights_path = directory / WEIGHTS_FILE

    writer = TableWriter(table_path, PROFILES_SCHEMA)

    clients = 0
    tokens = 0

    try:
        for number in range(source.count):

            loaded = source.batch(number)

            with torch.no_grad():
                vectors = encode(
                    embedding, encoder, loaded.model, torch.tensor(loaded.profile_time_log)
                )

            if bool(vectors.isnan().any()):
                raise ProfileError(f"батч {number}: в векторах клиентов появился NaN")

            writer.write(_table(loaded.model, vectors, embedding.dim))

            clients += loaded.model.clients
            tokens += int(loaded.model.profile_token_mask.sum())

    finally:
        rows = writer.close()

    _save(encoder, config, embedding.dim, weights_path)

    # Только после полной записи: прерванная сборка отметки не
    # получает, и читатель её отвергнет.
    write_lineage(directory)

    return {
        "group": group,
        "table": str(table_path),
        "weights": str(weights_path),
        "dim": embedding.dim,
        "seed": config.seed,
        "layers": config.layers,
        "heads": config.heads,
        "rows": rows,
        "batches": source.count,
        "clients": clients,
        "tokens": tokens,
        "size": table_path.stat().st_size,
    }


def encode(
    embedding: InputEmbedding,
    encoder: ProfileEncoder,
    model: BatchInput,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Векторы всех клиентов батча: [B, d]. times — время токенов
    анкеты [B, P].

    Анкета крошечная — два десятка токенов на клиента, — поэтому
    батч идёт одним куском, без нарезки на порции.
    """

    tokens = embedding.embed(
        model.profile_key_ids,
        model.profile_value_ids,
        model.profile_positions,
        model.profile_token_mask,
    )

    return encoder(tokens, times, model.profile_token_mask)


def _table(model: BatchInput, vectors: torch.Tensor, dim: int) -> pa.Table:
    """
    Строки одного батча: по клиенту на строку.
    """

    rows = [row for row in vectors.numpy()]

    return pa.table(
        {
            "batch_index": pa.array([model.batch_index] * model.clients, pa.int32()),
            "client_id": pa.array(list(model.client_ids), pa.string()),
            "dim": pa.array([dim] * model.clients, pa.int32()),
            "profile": pa.array(rows, type=pa.list_(pa.float32())),
        },
        schema=PROFILES_SCHEMA,
    )


def _embedding(group: str, specials: dict) -> InputEmbedding:
    """
    Входной слой этапа 09 со своими весами.

    Тот же загрузчик есть в этапе 10. Повтор намеренный: этапы
    держатся отдельно и друг друга не правят. Когда появится
    третий потребитель, его стоит вынести в сам пакет эмбеддингов.
    """

    path = embeddings_dir(group) / EMBEDDING_WEIGHTS

    if not path.exists():
        raise ProfileError(f"нет {path}: выполните python -m src.embedding.run {group}")

    problem = lineage_problem(path.parent, f"python -m src.embedding.run {group}")

    if problem:
        raise ProfileError(problem)

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


def _save(encoder: ProfileEncoder, config: ProfileConfig, dim: int, path: Path) -> None:
    """
    Веса энкодера рядом с векторами.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "dim": dim,
            "config": config.as_dict(),
            "state_dict": encoder.state_dict(),
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
    "PROFILES_SCHEMA",
    "ProfileError",
    "build_group",
    "encode",
]
