from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from tests import world
from tests.test_source_inputs import three


# ============================================================
# ИДЕЯ
# ============================================================
#
# Каждый этап пишет один файл, и его схема — это контракт со
# следующим этапом. Проверяется не то, что файл появился, а то,
# что в нём: одна строка на то, что этап описывает, номера батчей
# на месте, и повторный прогон даёт тот же результат.
#
# Этапы 10 и 12 отдельно: у них строка значит разное. У события —
# одно настоящее событие, у истории — один клиент. Заполнитель ни
# в тот, ни в другой файл не попадает.
#
# Команды проверяются по коду выхода: нехватка входа обязана дать
# понятное сообщение и «сделать нельзя», а не traceback.
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli(module: str, *args: str) -> int:
    """
    Команда этапа через её собственный разбор аргументов.
    """

    from importlib import import_module

    run = import_module(module)

    return run.run_group(run.build_parser().parse_args(list(args)))


# ============================================================
# ЭТАП 08 — МАСКИРОВАНИЕ
# ============================================================


def test_masking_writes_one_row_per_client_in_the_same_batches(stage):

    from src.masking.build import MASKED_SCHEMA, build_group
    from src.masking.settings import MASKED_FILE, MaskingConfig, masked_dir
    from src.batching.settings import BATCHES_FILE, batches_dir

    people = settle(stage)

    report = build_group("train", MaskingConfig(seed=4))

    path = masked_dir("train") / MASKED_FILE

    written = pq.ParquetFile(path)
    source = pq.ParquetFile(batches_dir("train") / BATCHES_FILE)

    assert written.schema_arrow.equals(MASKED_SCHEMA, check_metadata=False)
    assert written.num_row_groups == source.num_row_groups
    assert written.metadata.num_rows == len(people)

    table = pq.read_table(path).to_pylist()

    assert [row["client_id"] for row in table] == [
        made.client.client_id for made in people
    ]
    assert [row["batch_index"] for row in table] == [0, 0, 1]
    assert report["rows"] == len(people)


def test_masking_never_changes_the_length_of_a_row(stage):

    from src.masking.build import build_group
    from src.masking.settings import MASKED_FILE, MaskingConfig, masked_dir

    settle(stage)

    build_group("train", MaskingConfig(seed=4, value_probability=1.0))

    for row in pq.read_table(masked_dir("train") / MASKED_FILE).to_pylist():

        width = len(row["value_ids_source"])

        assert len(row["value_ids"]) == width
        assert len(row["labels"]) == width
        assert len(row["reason"]) == width


def test_masking_repeats_byte_for_byte(stage):
    """
    Тот же вход и тот же конфиг дают тот же файл: розыгрыш маски
    выведен из seed и client_id, а не из порядка чтения.
    """

    from src.masking.build import build_group
    from src.masking.settings import MASKED_FILE, MaskingConfig, masked_dir

    settle(stage)

    config = MaskingConfig(seed=4)

    build_group("train", config)

    first = digest(masked_dir("train") / MASKED_FILE)

    build_group("train", config)

    assert digest(masked_dir("train") / MASKED_FILE) == first


def test_masking_command_reports_a_missing_input(stage, capsys):

    from src.batching.settings import BATCHES_FILE, batches_dir

    settle(stage)

    (batches_dir("train") / BATCHES_FILE).unlink()

    assert cli("src.masking.run", "train") == EXIT_BLOCKED
    assert "python -m src.batching.run train" in capsys.readouterr().out


def test_masking_command_succeeds(stage):

    settle(stage)

    assert cli("src.masking.run", "train") == EXIT_OK


# ============================================================
# ЭТАП 09 — ЭМБЕДДИНГИ
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
    только в снимке векторов этапа 09; теперь проверяется на
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


def test_embeddings_command_succeeds(stage):

    settle(stage)

    assert cli("src.embedding.run", "train") == EXIT_OK


# ============================================================
# ЭТАП 10 — СОБЫТИЯ
# ============================================================


def ready(root: Path) -> list[world.Made]:
    """
    Мир, доведённый до этапа 10: веса 09 уже лежат на месте.
    """

    people = settle(root)

    world.write_weights(root, "train")

    return people


def test_one_row_is_one_real_event(stage):

    from src.event.build import EVENTS_SCHEMA, build_group
    from src.event.settings import EVENTS_FILE, events_dir

    people = ready(stage)

    config = world.encoder_configs()[0]

    report = build_group("train", config)

    path = events_dir("train") / EVENTS_FILE

    written = pq.ParquetFile(path)

    assert written.schema_arrow.equals(EVENTS_SCHEMA, check_metadata=False)

    events = sum(made.client.n_events for made in people)

    assert written.metadata.num_rows == events
    assert report["events"] == events

    table = pq.read_table(path).to_pylist()

    # Строки идут по клиентам, а события внутри клиента — по
    # порядку, и заполнителя среди них нет.
    expected = [
        (made.client.client_id, number)
        for made in people
        for number in range(made.client.n_events)
    ]

    assert [(row["client_id"], row["event"]) for row in table] == expected

    for row in table:
        assert len(row["vector"]) == world.DIM
        assert all(np.isfinite(row["vector"]))


def test_events_command_succeeds(stage):

    ready(stage)

    assert cli("src.event.run", "train") == EXIT_OK


def test_events_command_reports_missing_weights(stage, capsys):

    from src.embedding.settings import WEIGHTS_FILE, embeddings_dir

    settle(stage)

    (embeddings_dir("train") / WEIGHTS_FILE).unlink()

    assert cli("src.event.run", "train") == EXIT_BLOCKED
    assert "python -m src.embedding.run train" in capsys.readouterr().out


# ============================================================
# ЭТАП 11 — АНКЕТЫ
# ============================================================


def test_profiles_are_one_row_per_client(stage):

    from src.profile.build import PROFILES_SCHEMA, build_group
    from src.profile.settings import PROFILES_FILE, profiles_dir

    people = ready(stage)

    config = world.encoder_configs()[1]

    build_group("train", config)

    path = profiles_dir("train") / PROFILES_FILE

    written = pq.ParquetFile(path)

    assert written.schema_arrow.equals(PROFILES_SCHEMA, check_metadata=False)
    assert written.metadata.num_rows == len(people)

    for row, made in zip(pq.read_table(path).to_pylist(), people):
        assert row["client_id"] == made.client.client_id
        assert row["dim"] == world.DIM
        assert len(row["profile"]) == world.DIM


def test_profiles_command_succeeds(stage):

    ready(stage)

    assert cli("src.profile.run", "train") == EXIT_OK


def test_profiles_command_reports_a_missing_input(stage, capsys):
    """
    Regression: раньше обработчик ошибки печатал строку успеха и
    падал на несвязанном report.
    """

    from src.embedding.settings import WEIGHTS_FILE, embeddings_dir

    settle(stage)

    (embeddings_dir("train") / WEIGHTS_FILE).unlink()

    assert cli("src.profile.run", "train") == EXIT_BLOCKED
    assert "[profile] группа train:" in capsys.readouterr().out


# ============================================================
# ЭТАП 12 — ИСТОРИЯ
# ============================================================


def grown(root: Path) -> list[world.Made]:
    """
    Мир, доведённый до этапа 12: события и анкеты посчитаны.
    """

    from src.event.build import build_group as build_events
    from src.profile.build import build_group as build_profiles

    people = ready(root)

    build_events(
        "train",
        world.encoder_configs()[0],
    )

    build_profiles(
        "train",
        world.encoder_configs()[1],
    )

    return people


def history_config():

    return world.encoder_configs()[2]


def test_one_row_is_one_client_and_only_the_final_vector(stage):
    """
    Обновлённые векторы событий голове нужны в памяти, но на диск
    не идут: в файле истории только итоговый вектор клиента.
    """

    from src.history.build import HISTORY_SCHEMA, build_group
    from src.history.settings import HISTORY_FILE, history_dir

    people = grown(stage)

    report = build_group("train", history_config())

    path = history_dir("train") / HISTORY_FILE

    written = pq.ParquetFile(path)

    assert written.schema_arrow.equals(HISTORY_SCHEMA, check_metadata=False)
    assert written.metadata.num_rows == len(people)
    assert set(HISTORY_SCHEMA.names) == {"batch_index", "client_id", "dim", "client"}

    for row, made in zip(pq.read_table(path).to_pylist(), people):
        assert row["client_id"] == made.client.client_id
        assert row["dim"] == world.DIM
        assert len(row["client"]) == world.DIM
        assert all(np.isfinite(row["client"]))

    assert report["clients"] == len(people)


def test_history_repeats_itself(stage):
    """
    Тот же вход и те же веса дают те же векторы: повторный прогон
    сверяется по числам, а не по байтам файла — контейнер torch и
    сжатие parquet к смыслу не относятся.
    """

    from src.history.build import build_group
    from src.history.settings import HISTORY_FILE, history_dir

    grown(stage)

    config = history_config()

    build_group("train", config)

    first = pq.read_table(history_dir("train") / HISTORY_FILE).to_pylist()

    build_group("train", config)

    second = pq.read_table(history_dir("train") / HISTORY_FILE).to_pylist()

    assert [row["client_id"] for row in first] == [row["client_id"] for row in second]

    for left, right in zip(first, second):
        assert left["client"] == right["client"]


def test_history_command_succeeds(stage):

    grown(stage)

    assert cli("src.history.run", "train") == EXIT_OK


def test_history_command_reports_a_missing_input(stage, capsys):
    """
    Regression: тот же сломанный обработчик, что и в этапе 11.
    """

    ready(stage)

    # События посчитаны, анкеты нет: истории собирать не из чего.
    from src.event.build import build_group as build_events

    build_events(
        "train",
        world.encoder_configs()[0],
    )

    assert cli("src.history.run", "train") == EXIT_BLOCKED
    assert "[history] группа train:" in capsys.readouterr().out


# ============================================================
# ЧТО ЭТАПЫ ПИШУТ РЯДОМ
# ============================================================


@pytest.mark.parametrize(
    "module, directory, weights",
    [
        ("src.embedding", "embeddings_dir", "src.embedding.settings"),
        ("src.event", "events_dir", "src.event.settings"),
        ("src.profile", "profiles_dir", "src.profile.settings"),
    ],
)
def test_every_encoder_stage_saves_its_weights(stage, module, directory, weights):
    """
    Рядом с таблицей лежит weights.pt: из него следующий этап
    берёт и размерность, и конфигурацию, и состояние.
    """

    from importlib import import_module

    import torch

    ready(stage)

    assert cli(f"{module}.run", "train") == EXIT_OK

    settings = import_module(weights)

    path = getattr(settings, directory)("train") / settings.WEIGHTS_FILE

    saved = torch.load(path, map_location="cpu", weights_only=True)

    assert "state_dict" in saved
    assert saved["dim" if "dim" in saved else "vocab_size"]
