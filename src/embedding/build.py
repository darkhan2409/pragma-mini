from __future__ import annotations

from pathlib import Path

import torch

from src.dataset.lineage import write_lineage
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import EVT, USR, load_special_tokens

from .inputs import Source
from .layer import InputEmbedding
from .settings import WEIGHTS_FILE, EmbeddingConfig, embeddings_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# У этапа два файла на выходе:
#
#   weights.pt   — начальный розыгрыш общей таблицы эмбеддингов;
#   lineage.json — под какие словарь и анкету он собран.
#
# Векторов токенов этап не пишет. Их никто не читает: этапы 10-13
# берут номера токенов из батчей, веса — из weights.pt и считают
# вход сами, в прямом проходе. При обучении градиент идёт в ту же
# таблицу, и снимок векторов устарел бы на первом же шаге. Стоил
# же он гигабайты памяти при сборке и сотни мегабайт на диске.
#
# Батчи группы всё равно читаются целиком: Source сверяет пару
# 07/08 по каждому батчу, и веса не кладутся рядом с входом,
# который модель прочитать не сможет.
# ============================================================


def build_group(
    group: str,
    config: EmbeddingConfig,
    directory: Path | None = None,
) -> dict:
    """
    Веса входного слоя группы.
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

    directory = Path(directory) if directory is not None else embeddings_dir(group)

    # Прежний результат стирается целиком, вместе со снимком
    # векторов, который писала прежняя версия этапа.
    _clear(directory)

    clients = 0
    tokens = 0

    for number in range(source.count):

        model = source.batch(number).model

        clients += model.clients
        tokens += int(model.token_mask.sum()) + int(model.profile_token_mask.sum())

    weights_path = directory / WEIGHTS_FILE

    _save(layer, config, vocab.size, weights_path)

    # Только после полной записи. Веса собраны под словарь и
    # анкету текущего кода, и потребители это сверяют: веса
    # прежнего словаря иначе читались бы молча.
    write_lineage(directory)

    return {
        "group": group,
        "weights": str(weights_path),
        "dim": config.dim,
        "seed": config.seed,
        "vocab_size": vocab.size,
        "batches": source.count,
        "clients": clients,
        "tokens": tokens,
        "size": weights_path.stat().st_size,
    }


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
    "build_group",
]
