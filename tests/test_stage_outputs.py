from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests import world
from tests.test_source_inputs import three


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 06 сверяет каждую группу строк набора — время, маску и
# маркеры — и пишет только веса входного слоя и отметку
# происхождения. Проверяется не то, что файл появился, а то, что в
# нём: веса — ровно розыгрыш слоя по seed, сверены все батчи и все
# настоящие токены, маска — та же, что у прежнего обхода apply, а
# испорченный маркер останавливает этап до записи весов.
# ============================================================


def settle(root: Path, people: list[list[world.Made]] | None = None) -> list[world.Made]:

    flat = three()

    world.install(
        root,
        {
            "train": people if people is not None else [flat[:2], flat[2:]],
            "val": [world.population("v")],
        },
    )

    return flat


# ============================================================
# ВХОД СЛОЯ ЭМБЕДДИНГОВ — СВЕРКА
# ============================================================


def old_visible(self, row: dict) -> list[int]:
    """
    Видимые значения прежним обходом apply — эталон, дословно.
    """

    from src.masking.apply import apply
    from src.masking.choose import choose
    from src.tokenization.specials import MASK, UNK

    selection = choose(self.group, row, self.masking, self._weights)

    masked = apply(
        row["client_id"], row, selection.choices,
        self._specials[MASK], self._specials[UNK], selection.corrupted,
    )

    return masked["value_ids"]


def test_visible_values_are_the_values_of_the_old_apply(stage, monkeypatch):
    """
    Маска массивами (apply_selection) даёт слою те же значения, что
    прежний обход apply, и маскер в этих батчах действительно работал.
    """

    from src.embedding.inputs import Source

    settle(stage)

    new = [[row["visible_value_ids"].tolist() for row in Source("train").rows(number)] for number in range(2)]

    monkeypatch.setattr(Source, "_visible", old_visible)

    old = [[list(row["visible_value_ids"]) for row in Source("train").rows(number)] for number in range(2)]

    assert new == old
    assert any(world.MASK in values for batch in new for values in batch)


def test_a_spoiled_marker_stops_the_stage_before_the_weights(stage, monkeypatch):
    """
    Маркер, который маскер испортил в слоте значения, останавливает
    этап: отказ называет батч, клиента и позицию, а весов рядом с
    таким входом не остаётся.
    """

    from src.embedding.build import build_group
    from src.embedding.inputs import InputError, Source
    from src.embedding.settings import WEIGHTS_FILE, EmbeddingConfig, embeddings_dir

    settle(stage)

    first = Source("train").rows(1)[0]
    position = int(np.flatnonzero(np.asarray(first["key_ids"]) == world.EVT)[0])

    visible = Source._visible

    def spoiled(self, row: dict) -> np.ndarray:
        values = np.array(visible(self, row))
        values[int(np.flatnonzero(np.asarray(row["key_ids"]) == world.EVT)[0])] = world.PAD
        return values

    monkeypatch.setattr(Source, "_visible", spoiled)

    with pytest.raises(InputError) as error:
        Source("train").rows(1)

    assert str(error.value) == (
        f"батч 1, клиент {first['client_id']}, позиция {position}: "
        f"маркер {world.EVT} в слоте ключа, но {world.PAD} в слоте значения"
    )

    with pytest.raises(InputError, match="маркер"):
        build_group("train", EmbeddingConfig(dim=world.DIM, seed=world.SEED))

    assert not (embeddings_dir("train") / WEIGHTS_FILE).exists()


# ============================================================
# ЭТАП 06 — ЭМБЕДДИНГИ
# ============================================================


def test_embedding_stage_writes_weights_not_vectors(stage):
    """
    Этап оставляет веса и отметку происхождения, и только их.
    Снимок векторов прежней сборки стирается: векторы считает
    модель по номерам токенов и этим весам.
    """

    import torch

    from src.dataset.lineage import LINEAGE_FILE, lineage_problem
    from src.embedding.build import build_group
    from src.embedding.layer import InputEmbedding
    from src.embedding.settings import WEIGHTS_FILE, EmbeddingConfig, embeddings_dir
    from src.tokenization.finalvocab import FrozenArtifacts

    people = settle(stage)

    directory = embeddings_dir("train")
    directory.mkdir(parents=True, exist_ok=True)

    # Так выглядел каталог прежней версии этапа.
    (directory / "embeddings.parquet").write_bytes(b"old snapshot")

    report = build_group("train", EmbeddingConfig(dim=world.DIM, seed=world.SEED))

    assert sorted(path.name for path in directory.iterdir()) == [LINEAGE_FILE, WEIGHTS_FILE]
    assert lineage_problem(directory, "python -m src.embedding.run train") is None

    # Сверены все батчи и все настоящие токены.
    assert report["batches"] == 2
    assert report["clients"] == len(people)
    assert report["tokens"] == sum(
        made.client.n_tokens + made.client.profile_n_tokens for made in people
    )

    # Веса — ровно розыгрыш слоя по seed: тем же весам модель
    # потом посчитает те же векторы.
    saved = torch.load(directory / WEIGHTS_FILE, map_location="cpu", weights_only=True)

    size = FrozenArtifacts.load().size

    assert (saved["vocab_size"], saved["dim"], saved["seed"]) == (size, world.DIM, world.SEED)

    expected = InputEmbedding(size, world.DIM, world.SEED, markers=(world.EVT, world.USR))

    assert saved["state_dict"].keys() == expected.state_dict().keys()

    for name, value in expected.state_dict().items():
        assert torch.equal(saved["state_dict"][name], value), name


def test_input_layer_sums_three_terms_and_zeroes_padding():
    """
    Обычный токен — E[key]·√d + E[value]·√d + P[кусок], маркер —
    один E[маркер]·√d, заполнитель — ноль. Прежде это было видно
    только в снимке векторов этапа 06; теперь проверяется на
    самом слое.
    """

    import math

    import torch

    from src.embedding.layer import InputEmbedding

    layer = InputEmbedding(world.VOCAB, world.DIM, world.SEED, markers=(world.EVT, world.USR))

    table = layer.weight.detach()
    scale = math.sqrt(world.DIM)

    key_ids = torch.tensor([[world.EVT, world.KEY_A, world.KEY_A, world.KEY_B]])
    value_ids = torch.tensor([[world.EVT, 10, 11, world.PAD]])
    positions = torch.tensor([[0, 0, 1, 0]])
    mask = torch.tensor([[True, True, True, False]])

    with torch.no_grad():
        out = layer.embed(key_ids, value_ids, positions, mask)[0]

    def piece(number: int) -> torch.Tensor:
        return layer.pieces_of(torch.tensor(number))

    assert torch.equal(out[0], table[world.EVT] * scale)
    assert torch.allclose(out[1], table[world.KEY_A] * scale + table[10] * scale + piece(0))
    assert torch.allclose(out[2], table[world.KEY_A] * scale + table[11] * scale + piece(1))
    assert torch.equal(out[3], torch.zeros(world.DIM))
