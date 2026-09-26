from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn

from src.dataset.lineage import LINEAGE_FILE, lineage, lineage_problem
from src.embedding.settings import WEIGHTS_FILE as EMBEDDING_WEIGHTS
from src.embedding.settings import embeddings_dir
from src.event.encoder import EventEncoder
from src.event.settings import EventConfig
from src.event.version import IMPLEMENTATION_VERSION as EVENT_VERSION
from src.history.encoder import HistoryEncoder
from src.history.settings import HistoryConfig
from src.history.version import IMPLEMENTATION_VERSION as HISTORY_VERSION
from src.preprocessing.artifacts import read_json, write_json
from src.profile.encoder import ProfileEncoder
from src.profile.settings import ProfileConfig
from src.profile.version import IMPLEMENTATION_VERSION as PROFILE_VERSION
from src.tokenization.finalvocab import VocabError, load_final_vocab

from .settings import BACKBONE_FILES, backbone_dir


# ============================================================
# НАЧАЛЬНЫЕ ВЕСА BACKBONE
# ============================================================
#
# Подготовка к обучению:
#
#   07 батчи -> 08 маски -> 09 входной слой -> init_backbone -> 14
#
# init_backbone создаёт начальные веса энкодеров события, анкеты и
# истории — и только их. Ни одного прохода по данным: веса
# энкодера определяются его конфигом, длиной вектора d из весов
# этапа 09 и seed, а не клиентами. Разыгрываются они на CPU — так
# «тот же seed» значит «те же веса» на любой машине, — и обучение
# уже само переносит модель на карту.
#
# ЕДИНСТВЕННОЕ место, где конфиг и d превращаются в веса, — функции
# initial_* ниже. Ими же пользуются диагностические этапы 10–12,
# поэтому init_backbone и python -m src.event.run при одном конфиге
# дают одни и те же веса: второго розыгрыша нет. Формат файла весов
# — payload — тоже общий.
#
# Этапы 10–13 обучению не нужны: их векторы посчитаны начальными
# весами и при обучении всё равно считаются заново, в прямом
# проходе. Они остаются диагностикой.
#
# Модель одна и учится на train, поэтому и начальные веса одни: из
# весов этапа 09 группы train. val, test и отчёты считаются той же
# моделью, а не своим случайным backbone.
#
# lineage.json каталога называет, из чего он собран: происхождение
# набора, отпечатки словаря и весов входного слоя, версии кода
# энкодеров. Модель собирается только из каталога, где всё это
# совпадает с текущим; иначе — отказ с командой пересборки.
# ============================================================


# Группа, на которой учится модель: её входной слой и её веса.
MODEL_GROUP = "train"

# Формат каталога backbone: набор файлов и ключи lineage.json.
BACKBONE_FORMAT = 1

INIT_COMMAND = "python -m src.mlm.init_backbone"


class BackboneError(ValueError):
    """
    Начальные веса backbone собрать или прочитать нельзя.
    """


def initial_event(config: EventConfig, dim: int) -> EventEncoder:
    """
    Начальный энкодер события.
    """

    config.check_dim(dim)

    return EventEncoder(dim, config.layers, config.heads, config.feedforward,
                        config.dropout, config.seed)


def initial_profile(config: ProfileConfig, dim: int) -> ProfileEncoder:
    """
    Начальный энкодер анкеты.
    """

    config.check_dim(dim)

    return ProfileEncoder(dim, config.layers, config.heads, config.feedforward,
                          config.dropout, config.rope_base, config.seed)


def initial_history(config: HistoryConfig, dim: int) -> HistoryEncoder:
    """
    Начальный энкодер истории.
    """

    config.check_dim(dim)

    return HistoryEncoder(dim, config.layers, config.heads, config.feedforward,
                          config.dropout, config.rope_base, config.seed)


def payload(encoder: nn.Module, config, dim: int) -> dict:
    """
    Файл весов энкодера: d, конфиг и состояние на CPU. Состояние
    переносится на CPU, чтобы файл не зависел от того, где считали.
    """

    return {
        "dim": int(dim),
        "config": config.as_dict(),
        "state_dict": {name: value.detach().cpu() for name, value in encoder.state_dict().items()},
    }


def state_digest(state: dict) -> str:
    """
    Отпечаток содержимого весов: имена, типы, формы и байты тензоров.

    По содержимому, а не по файлу: пересохранение тех же весов его
    не меняет.
    """

    digest = hashlib.sha256()

    for name in sorted(state):

        tensor = state[name].detach().cpu().contiguous()

        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

    return digest.hexdigest()


def vocabulary_digest() -> str:
    """
    Отпечаток финального словаря data/03_vocab.
    """

    try:
        vocab = load_final_vocab()
    except VocabError as error:
        raise BackboneError(str(error)) from error

    text = json.dumps(vocab, sort_keys=True, ensure_ascii=False)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_embedding() -> dict:
    """
    Веса входного слоя модели — этап 09 группы train — с проверкой
    происхождения и словаря.
    """

    path = embeddings_dir(MODEL_GROUP) / EMBEDDING_WEIGHTS

    command = f"python -m src.embedding.run {MODEL_GROUP}"

    if not path.exists():
        raise BackboneError(f"нет {path}: выполните {command}")

    problem = lineage_problem(path.parent, command)

    if problem:
        raise BackboneError(problem)

    saved = torch.load(path, map_location="cpu", weights_only=True)

    try:
        size = len(load_final_vocab())
    except VocabError as error:
        raise BackboneError(str(error)) from error

    # Номера токенов берутся из словаря: таблица другого размера
    # собрана под другой словарь.
    if int(saved["vocab_size"]) != size:
        raise BackboneError(
            f"{path} собран под словарь из {saved['vocab_size']} токенов, а в "
            f"data/03_vocab их {size}: выполните {command} заново"
        )

    return saved


def stamp(saved: dict) -> dict:
    """
    Что у каталога backbone обязано совпасть с текущим кодом и
    данными. saved — веса этапа 09.
    """

    return {
        "format": BACKBONE_FORMAT,
        "dataset": lineage(),
        "vocabulary": vocabulary_digest(),
        "embedding": {
            "group": MODEL_GROUP,
            "vocab_size": int(saved["vocab_size"]),
            "dim": int(saved["dim"]),
            "seed": int(saved["seed"]),
            "state": state_digest(saved["state_dict"]),
        },
        "implementation": {
            "event": EVENT_VERSION,
            "profile": PROFILE_VERSION,
            "history": HISTORY_VERSION,
        },
    }


def init_backbone(
    event: EventConfig,
    profile: ProfileConfig,
    history: HistoryConfig,
    directory: Path | None = None,
) -> dict:
    """
    Начальные веса трёх энкодеров и lineage.json — без прохода по
    данным.
    """

    started = time.perf_counter()

    saved = read_embedding()

    dim = int(saved["dim"])

    encoders = {
        "event": (initial_event(event, dim), event),
        "profile": (initial_profile(profile, dim), profile),
        "history": (initial_history(history, dim), history),
    }

    directory = Path(directory) if directory is not None else backbone_dir()

    _clear(directory)

    sizes = {}

    for name, (encoder, config) in encoders.items():

        path = directory / BACKBONE_FILES[name]

        torch.save(payload(encoder, config, dim), path)

        sizes[name] = path.stat().st_size

    described = {
        name: {
            "config": config.as_dict(),
            "blocks": len(encoder.layers),
            "parameters": sum(value.numel() for value in encoder.parameters()),
        }
        for name, (encoder, config) in encoders.items()
    }

    # Отметка — последней: прерванная сборка её не получает, и
    # читатель такой каталог отвергнет.
    write_json(directory / LINEAGE_FILE, dict(stamp(saved), encoders=described))

    return {
        "directory": str(directory),
        "dim": dim,
        "encoders": described,
        "bytes": sum(sizes.values()) + (directory / LINEAGE_FILE).stat().st_size,
        "files": sizes,
        "seconds": time.perf_counter() - started,
    }


def load_backbone(saved: dict) -> tuple[EventEncoder, ProfileEncoder, HistoryEncoder]:
    """
    Начальные энкодеры модели из каталога backbone. saved — веса
    этапа 09, под которые каталог обязан быть собран.
    """

    directory = backbone_dir()

    path = directory / LINEAGE_FILE

    if not path.exists():
        raise BackboneError(
            f"нет {path}: начальные веса backbone не собраны — выполните {INIT_COMMAND}"
        )

    found = read_json(path)

    expected = stamp(saved)

    differ = {key: (found.get(key), value) for key, value in expected.items() if found.get(key) != value}

    if differ:
        shown = "; ".join(f"{key}: собран {was}, нужно {now}" for key, (was, now) in differ.items())
        raise BackboneError(
            f"{directory} собран не под текущие {', '.join(differ)} ({shown}) — "
            f"выполните {INIT_COMMAND} заново"
        )

    dim = int(saved["dim"])

    built = []

    for name, kind, initial in (
        ("event", EventConfig, initial_event),
        ("profile", ProfileConfig, initial_profile),
        ("history", HistoryConfig, initial_history),
    ):

        file = directory / BACKBONE_FILES[name]

        if not file.exists():
            raise BackboneError(f"нет {file}: выполните {INIT_COMMAND} заново")

        weights = torch.load(file, map_location="cpu", weights_only=True)

        # Файл и отметка пишутся одним запуском: чужой файл рядом с
        # отметкой — это смешанный каталог.
        recorded = found.get("encoders", {}).get(name, {}).get("config")

        if weights["config"] != recorded or int(weights["dim"]) != dim:
            raise BackboneError(
                f"{file} не из того запуска, что {path}: выполните {INIT_COMMAND} заново"
            )

        try:
            config = kind.from_dict(weights["config"])
            encoder = initial(config, dim)
            encoder.load_state_dict(weights["state_dict"])
        except (ValueError, RuntimeError) as error:
            raise BackboneError(f"{file} не подходит к коду энкодера: {error}") from error

        built.append(encoder)

    return built[0], built[1], built[2]


def _clear(directory: Path) -> None:
    """
    Каталог держит только свои файлы: прежний результат стирается.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "BACKBONE_FORMAT",
    "INIT_COMMAND",
    "MODEL_GROUP",
    "BackboneError",
    "init_backbone",
    "initial_event",
    "initial_history",
    "initial_profile",
    "load_backbone",
    "payload",
    "read_embedding",
    "stamp",
    "state_digest",
    "vocabulary_digest",
]
