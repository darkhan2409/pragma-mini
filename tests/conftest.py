from __future__ import annotations

from pathlib import Path

import pytest

from tests import world


# ============================================================
# ИДЕЯ
# ============================================================
#
# Настоящего data/ у тестов нет и быть не должно. Вместо него —
# временный каталог, на который переставлены глобалы settings
# каждого этапа: ..._dir(group) читает свой X_DIR в момент
# вызова, поэтому подмена глобала перенаправляет этап целиком.
#
# Подмена автоматическая и на всю сессию: так ни один тест не
# сможет случайно записать в настоящий data/, даже если забудет
# попросить фикстуру.
# ============================================================


# (модуль, имя глобала, подкаталог)
#
# Перечислены ВСЕ этапы, а не только те, что нужны сегодняшним
# тестам: пропущенный глобал означает, что этап пишет в
# настоящий data/, и заметно это становится только по следам.
PLACES = (
    ("src.preprocessing.settings", "RAW_DIR", "01_raw"),
    ("src.preprocessing.settings", "PREPROCESSED_DIR", "02_preprocessed"),
    ("src.tokenization.settings", "VOCAB_DIR", "03_vocab"),
    # Копия, снятая при импорте: FrozenArtifacts.load читает её, а
    # не settings, и без этой строки словарь искался бы в
    # настоящем data/.
    ("src.tokenization.finalvocab", "VOCAB_DIR", "03_vocab"),
    ("src.tokenization.settings", "TOKENIZED_DIR", "04_tokenized"),
    ("src.dataset.settings", "DATASET_DIR", "05_dataset"),
    ("src.temporal.settings", "TEMPORAL_DIR", "06_temporal"),
    ("src.batching.settings", "BATCHES_DIR", "07_batches"),
    ("src.masking.settings", "MASKED_DIR", "08_masked"),
    ("src.embedding.settings", "EMBEDDINGS_DIR", "09_embeddings"),
    ("src.event.settings", "EVENTS_DIR", "10_events"),
    ("src.profile.settings", "PROFILES_DIR", "11_profiles"),
    ("src.history.settings", "HISTORY_DIR", "12_history"),
    ("src.mlm.settings", "MLM_DIR", "13_mlm"),
    ("src.mlm.settings", "TRAIN_DIR", "14_train"),
)


def pytest_sessionstart(session) -> None:
    """
    Один поток на тензоры.

    Тензоры здесь крошечные, и на них диспетчеризация потоков
    стоит дороже самой арифметики: набор с шестью потоками идёт
    в разы медленнее.
    """

    import torch

    torch.set_num_threads(1)


@pytest.fixture(scope="session", autouse=True)
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Временный data/ на всю сессию.
    """

    from importlib import import_module

    root = tmp_path_factory.mktemp("data")

    patch = pytest.MonkeyPatch()

    for name, attribute, folder in PLACES:

        module = import_module(name)

        # Глобал обязан существовать: иначе подмена создала бы
        # новое имя и молча ничего не перенаправила.
        assert hasattr(module, attribute), f"{name}.{attribute} больше нет"

        patch.setattr(module, attribute, root / folder)

    world.write_vocab(root)

    yield root

    patch.undo()


@pytest.fixture(scope="session")
def groups(data_root: Path) -> dict[str, list[world.Made]]:
    """
    Мир на диске: батчи, маски и веса этапов 09-12 для train и val.

    Клиенты разложены по двум батчам, поэтому проверяется и
    переход между группами строк, и нумерация batch_index.
    """

    from src.batching.settings import BATCHES_FILE, batches_dir
    from src.masking.settings import MASKED_FILE, masked_dir

    made: dict[str, list[world.Made]] = {}

    for group, prefix in (("train", "t"), ("val", "v")):

        people = world.population(prefix)

        batches = [people[:3], people[3:]]

        world.write_batches(batches_dir(group) / BATCHES_FILE, batches)
        world.write_masked(masked_dir(group) / MASKED_FILE, batches)
        world.write_weights(data_root, group)

        made[group] = people

    return made


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Свой временный data/ на один тест.

    Нужен там, где тест сам решает, что лежит в train: сессионный
    мир общий, и переписывать его под себя нельзя.
    """

    from importlib import import_module

    root = tmp_path / "data"

    for name, attribute, folder in PLACES:
        monkeypatch.setattr(import_module(name), attribute, root / folder)

    world.write_vocab(root)

    return root


@pytest.fixture
def made() -> list[world.Made]:
    """
    Тот же набор клиентов, но только в памяти.
    """

    return world.population()


@pytest.fixture
def clients(made: list[world.Made]) -> list:
    return [item.client for item in made]


@pytest.fixture
def cpu():
    import torch

    return torch.device("cpu")


@pytest.fixture
def model():
    """
    Крошечная модель без dropout, в режиме проверки.
    """

    built = world.model()
    built.eval()

    return built
