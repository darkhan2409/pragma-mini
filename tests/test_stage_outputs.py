from __future__ import annotations

from pathlib import Path

from tests import world
from tests.test_source_inputs import three


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 06 выравнивает группу строк набора в памяти и пишет только
# веса входного слоя и отметку происхождения. Проверяется не то,
# что файл появился, а то, что в нём: веса — ровно розыгрыш слоя по
# seed, сверены все батчи и все настоящие токены, заполнитель —
# только справа и только в памяти.
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
# ВХОД СЛОЯ ЭМБЕДДИНГОВ — ВЫРАВНИВАНИЕ В ПАМЯТИ
# ============================================================


def test_embedding_input_pads_a_row_group_on_the_right_only(stage):
    """
    Набор хранит клиентов без заполнителя; слой считает [B, T] и
    [B, P]. Группа строк дополняется до самого длинного клиента:
    настоящее слева, маска — ровно «номер меньше длины», хвост —
    только [PAD], смещение пустого события — конец последовательности.
    """

    from src.embedding.inputs import Source

    people = settle(stage)

    loaded = Source("train").batch(0)
    model = loaded.model
    batch = people[:2]

    width = max(made.client.n_tokens for made in batch)
    profile = max(made.client.profile_n_tokens for made in batch)
    events = max(made.client.n_events for made in batch)

    assert model.key_ids.shape == (2, width)
    assert model.profile_key_ids.shape == (2, profile)

    for row, made in enumerate(batch):

        client = made.client
        n, p = client.n_tokens, client.profile_n_tokens

        assert model.token_mask[row].tolist() == [True] * n + [False] * (width - n)
        assert model.profile_token_mask[row].tolist() == [True] * p + [False] * (profile - p)

        assert model.key_ids[row, :n].tolist() == client.key_ids.tolist()
        assert model.key_ids[row, n:].tolist() == [world.PAD] * (width - n)
        assert model.value_ids[row, n:].tolist() == [world.PAD] * (width - n)
        assert model.profile_value_ids[row, :p].tolist() == client.profile_value_ids.tolist()

        structure = loaded.rows[row]

        assert structure["event_starts"] == client.event_starts.tolist() + [n] * (events - client.n_events)
        assert structure["event_mask"] == [True] * client.n_events + [False] * (events - client.n_events)


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
