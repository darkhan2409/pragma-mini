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


def test_embeddings_are_one_row_per_client(stage):

    from src.embedding.build import EMBEDDINGS_SCHEMA, build_group
    from src.embedding.settings import (
        EMBEDDINGS_FILE, WEIGHTS_FILE, EmbeddingConfig, embeddings_dir,
    )

    people = settle(stage)

    config = EmbeddingConfig(dim=world.DIM, seed=world.SEED)

    report = build_group("train", config)

    path = embeddings_dir("train") / EMBEDDINGS_FILE

    written = pq.ParquetFile(path)

    assert written.schema_arrow.equals(EMBEDDINGS_SCHEMA, check_metadata=False)
    assert written.metadata.num_rows == len(people)

    table = pq.read_table(path).to_pylist()

    # Строка хранит весь прямоугольник батча: настоящие токены и
    # за ними заполнитель. Заполнитель обязан быть ровно нулём —
    # это единственное, чем [PAD] отличается от значения.
    batches = [people[:2], people[2:]]

    place = 0

    for batch in batches:

        width = max(made.client.n_tokens for made in batch)
        profile = max(made.client.profile_n_tokens for made in batch)

        for made in batch:

            row = table[place]
            place += 1

            assert row["client_id"] == made.client.client_id
            assert row["dim"] == world.DIM

            assert len(row["tokens"]) == width * world.DIM
            assert len(row["profile"]) == profile * world.DIM

            assert not any(row["tokens"][made.client.n_tokens * world.DIM :])
            assert not any(row["profile"][made.client.profile_n_tokens * world.DIM :])

    assert (embeddings_dir("train") / WEIGHTS_FILE).exists()
    assert report["dim"] == world.DIM


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
    from src.event.settings import EVENTS_FILE, EventConfig, events_dir

    people = ready(stage)

    config = EventConfig(
        seed=world.SEED, layers=world.LAYERS, heads=world.HEADS,
        feedforward=world.FEEDFORWARD, dropout=0.0,
    )

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
    from src.profile.settings import PROFILES_FILE, ProfileConfig, profiles_dir

    people = ready(stage)

    config = ProfileConfig(
        seed=world.SEED, layers=world.LAYERS, heads=world.HEADS,
        feedforward=world.FEEDFORWARD, dropout=0.0,
    )

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
    from src.event.settings import EventConfig
    from src.profile.build import build_group as build_profiles
    from src.profile.settings import ProfileConfig

    people = ready(root)

    build_events(
        "train",
        EventConfig(
            seed=world.SEED, layers=world.LAYERS, heads=world.HEADS,
            feedforward=world.FEEDFORWARD, dropout=0.0,
        ),
    )

    build_profiles(
        "train",
        ProfileConfig(
            seed=world.SEED, layers=world.LAYERS, heads=world.HEADS,
            feedforward=world.FEEDFORWARD, dropout=0.0,
        ),
    )

    return people


def history_config():

    from src.history.settings import HistoryConfig

    return HistoryConfig(
        seed=world.SEED, layers=world.LAYERS, heads=world.HEADS,
        feedforward=world.FEEDFORWARD, dropout=0.0,
        rope_base=world.ROPE_BASE, device="cpu",
    )


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
    from src.event.settings import EventConfig

    build_events(
        "train",
        EventConfig(
            seed=world.SEED, layers=world.LAYERS, heads=world.HEADS,
            feedforward=world.FEEDFORWARD, dropout=0.0,
        ),
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
