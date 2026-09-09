from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from src.tokenizer.config import IncompatibleArtifactsError

from .config import STRUCTURE_EVENT


# ============================================================
# ИДЕЯ
# ============================================================
#
# Checkpoint обязан восстанавливать не только веса, но и весь
# контекст, от которого зависит следующий шаг: оптимизатор,
# scheduler, счётчики, состояние sampler и генераторы
# случайных чисел. Иначе продолженное обучение это другое
# обучение, а «дообучил с того же места» ничем не проверяется.
#
# Отдельно хранятся отпечатки artifacts и описание
# фиксированной validation. Веса, обученные на другом словаре
# или сравниваемые с другим набором масок, молча дают
# бессмысленные метрики; это должно быть ошибкой загрузки.
# ============================================================


CHECKPOINT_VERSION = 1

# Секции, без которых checkpoint бесполезен. Проверяются сразу
# после записи: файл, который не открывается или неполон, не
# должен занять место рабочего.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "version",
    "backbone",
    "head",
    "optimizer",
    "scheduler",
    "counters",
    "train_config",
    "model_config",
    "masking_config",
    "artifacts",
    "rng",
)

# Поля as_dict, которые не про воспроизводимость, а про чтение
# человеком: сверять их при resume незачем.
IGNORED_ON_RESUME: frozenset[str] = frozenset({"scheduler", "budget"})


def rng_state() -> dict:
    """
    Состояние всех генераторов, влияющих на обучение.
    """

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:

    random.setstate(state["python"])

    np.random.set_state(state["numpy"])

    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())

    saved = state.get("torch_cuda") or []

    if torch.cuda.is_available() and len(saved) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([torch.as_tensor(item, dtype=torch.uint8).cpu() for item in saved])


# ============================================================
# ЗАПИСЬ
# ============================================================


def save_checkpoint(
    path: Path,
    *,
    backbone,
    head,
    optimizer,
    scheduler,
    counters: dict,
    train_config: dict,
    model_config: dict,
    masking_config: dict,
    sampler_state: dict | None,
    splits: dict,
    artifacts: dict,
    metrics: dict | None = None,
    progress: dict | None = None,
) -> Path:
    """
    Атомарная запись: сначала во временный файл рядом, потом
    проверка, потом подмена.

    Порядок именно такой, потому что упасть можно посередине
    записи. Перезаписывать рабочий файл напрямую значит менять
    «есть checkpoint» на «есть обрубок» ровно в тот момент,
    когда он нужнее всего.
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": CHECKPOINT_VERSION,
        "backbone": backbone.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "counters": dict(counters),
        "train_config": dict(train_config),
        "model_config": dict(model_config),
        "masking_config": dict(masking_config),
        "sampler": sampler_state,
        "splits": splits,
        "artifacts": artifacts,
        "metrics": metrics,
        # Где остановилось обучение: micro-batch эпохи, история
        # оценок, лучший результат и что уже сделано. Без этого
        # resume начал бы эпоху заново.
        "progress": dict(progress) if progress else None,
        "rng": rng_state(),
    }

    # Временный файл рядом, а не в TEMP: os.replace атомарен
    # только внутри одной файловой системы.
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")

    try:

        torch.save(payload, temporary)

        verify_checkpoint(temporary)

        os.replace(temporary, path)

    except BaseException:

        # Прежний checkpoint остаётся целым: его не трогали.
        temporary.unlink(missing_ok=True)

        raise

    return path


def verify_checkpoint(path: Path) -> dict:
    """
    Файл открывается и содержит всё обязательное.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise IncompatibleArtifactsError(f"{path}: не похоже на checkpoint")

    missing = [name for name in REQUIRED_SECTIONS if name not in payload]

    if missing:
        raise IncompatibleArtifactsError(f"{path}: в checkpoint нет секций {missing}")

    return payload


# ============================================================
# ЧТЕНИЕ
# ============================================================


def _compare(name: str, saved, actual) -> None:

    if saved != actual:
        raise IncompatibleArtifactsError(
            f"checkpoint несовместим: {name} отличается от текущего окружения"
        )


# ------------------------------------------------------------
# СОВМЕСТИМОСТЬ ПРИ ДОБАВЛЕНИИ ПОЛЕЙ
# ------------------------------------------------------------
#
# Точное равенство словарей ломало бы все прежние checkpoints
# при одном лишь добавлении ключа. Правило мягче и по-прежнему
# строгое:
#
#   ключ есть у обоих   значения обязаны совпадать
#   ключ только сейчас  он обязан равняться унаследованному
#                       значению, иначе окружение другое
#   ключ только в файле  checkpoint новее кода: отказ
#
# Унаследованное значение это то, что подразумевал checkpoint,
# записанный до появления ключа. Для structure это "event":
# Session Encoder тогда не существовал. Для sessions это None:
# sidecar сессий тогда не собирали.
# ------------------------------------------------------------

LEGACY_MODEL_CONFIG: dict = {"structure": STRUCTURE_EVENT}

LEGACY_ARTIFACTS: dict = {"sessions": None}


def _compare_with_legacy(name: str, saved: dict, actual: dict, legacy: dict) -> None:

    # Ключи с известным унаследованным значением проверяются
    # первыми: иначе отказ назвал бы сопутствующий ключ вместо
    # настоящей причины (например n_session_layers вместо
    # структуры, которая его и притащила).
    order = sorted(set(saved) | set(actual), key=lambda key: (key not in legacy, key))

    for key in order:

        if key in saved and key in actual:

            if saved[key] != actual[key]:
                raise IncompatibleArtifactsError(
                    f"checkpoint несовместим: {name}, поле {key}: "
                    f"сохранено {saved[key]!r}, сейчас {actual[key]!r}"
                )

            continue

        if key in actual:

            if key not in legacy:
                raise IncompatibleArtifactsError(
                    f"checkpoint несовместим: {name}, поле {key} появилось позже "
                    f"checkpoint, и унаследованное значение для него не определено; "
                    f"сейчас {actual[key]!r}"
                )

            if actual[key] != legacy[key]:
                raise IncompatibleArtifactsError(
                    f"checkpoint несовместим: {name}, поля {key} в нём нет, значит "
                    f"он собран как {legacy[key]!r}, а сейчас {actual[key]!r}"
                )

            continue

        raise IncompatibleArtifactsError(
            f"checkpoint несовместим: {name}, в нём есть поле {key}, "
            f"которого нет в текущем окружении"
        )


def compare_model_config(saved: dict, actual: dict) -> None:
    _compare_with_legacy("конфигурация модели", saved, actual, LEGACY_MODEL_CONFIG)


def compare_artifacts(saved: dict, actual: dict) -> None:
    _compare_with_legacy("отпечатки artifacts", saved, actual, LEGACY_ARTIFACTS)


# ------------------------------------------------------------
# СОВМЕСТИМОСТЬ ПРИ ПРОДОЛЖЕНИИ
# ------------------------------------------------------------
#
# Продолжить обучение можно только тем же обучением. Другой
# seed, другая политика целей, другая схема масок или другой
# словарь дают уже другой эксперимент, и «дообучил с того же
# места» становится неправдой.
#
# Поэтому сверяется ВСЯ конфигурация, а не выборочные поля:
# перечислять исключения безопаснее, чем перечислять то, что
# важно, и однажды забыть новое поле.
# ------------------------------------------------------------


def compare_train_config(saved: dict, actual: dict) -> None:

    names = (set(saved) | set(actual)) - IGNORED_ON_RESUME

    differ = [
        name
        for name in sorted(names)
        if saved.get(name) != actual.get(name)
    ]

    if differ:
        lines = ", ".join(
            f"{name}: было {saved.get(name)!r}, стало {actual.get(name)!r}"
            for name in differ[:6]
        )
        raise IncompatibleArtifactsError(
            f"продолжение невозможно: конфигурация обучения отличается ({lines})"
        )


def compare_masking_config(saved: dict, actual: dict) -> None:

    differ = [
        name
        for name in sorted(set(saved) | set(actual))
        if saved.get(name) != actual.get(name)
    ]

    if differ:
        raise IncompatibleArtifactsError(
            f"продолжение невозможно: маскирование отличается по полям {differ[:6]}"
        )


def check_resume(
    payload: dict,
    *,
    train_config: dict,
    model_config: dict,
    masking_config: dict,
    artifacts: dict,
) -> dict:
    """
    Всё, от чего зависит продолжение, обязано совпасть.

    Вызывается ДО загрузки набора и до первого шага: несовпадение
    seed, политики целей или словаря должно стоить секунды, а не
    полчаса чтения событий.
    """

    if payload.get("version") != CHECKPOINT_VERSION:
        raise IncompatibleArtifactsError(
            f"checkpoint версии {payload.get('version')}, ожидалась {CHECKPOINT_VERSION}"
        )

    missing = [name for name in REQUIRED_SECTIONS if name not in payload]

    if missing:
        raise IncompatibleArtifactsError(f"в checkpoint нет секций {missing}")

    # Словарь, artifacts preprocessing и sidecar сессий это один
    # отпечаток: он покрывает и токенизацию, и ключи сессий.
    compare_artifacts(payload["artifacts"], artifacts)

    compare_model_config(payload["model_config"], model_config)

    compare_train_config(payload["train_config"], train_config)

    compare_masking_config(payload["masking_config"], masking_config)

    return payload


def check_resume_splits(payload: dict, splits: dict) -> None:
    """
    Наборы validation те же самые: те же маски и те же цели.

    Отдельно от check_resume, потому что описания наборов
    появляются только после их сборки.
    """

    stored = payload.get("splits") or {}

    for name, description in splits.items():

        saved = stored.get(name)

        if saved is None:
            raise IncompatibleArtifactsError(
                f"продолжение невозможно: checkpoint не знает набора {name}"
            )

        if saved.get("targets_sha256") != description.get("targets_sha256"):
            raise IncompatibleArtifactsError(
                f"продолжение невозможно: набор {name} собран иначе, "
                "маски и цели не совпадают с сохранёнными"
            )


def load_checkpoint(
    path: Path,
    *,
    backbone,
    head,
    optimizer=None,
    scheduler=None,
    sampler=None,
    model_config: dict | None = None,
    artifacts: dict | None = None,
    splits: dict | None = None,
    map_location="cpu",
    restore_random: bool = True,
) -> dict:
    """
    Восстанавливает обучение и проверяет, что окружение то же.
    """

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)

    if payload.get("version") != CHECKPOINT_VERSION:
        raise IncompatibleArtifactsError(
            f"checkpoint версии {payload.get('version')}, ожидалась {CHECKPOINT_VERSION}"
        )

    if model_config is not None:
        compare_model_config(payload["model_config"], dict(model_config))

    if artifacts is not None:
        compare_artifacts(payload["artifacts"], artifacts)

    if splits is not None:

        saved = payload.get("splits") or {}

        for name, description in splits.items():

            stored = saved.get(name)

            if stored is None:
                raise IncompatibleArtifactsError(f"checkpoint не содержит описания набора {name}")

            if stored.get("targets_sha256") != description.get("targets_sha256"):
                raise IncompatibleArtifactsError(
                    f"набор {name} собран иначе: маски и цели не совпадают с сохранёнными"
                )

    backbone.load_state_dict(payload["backbone"])
    head.load_state_dict(payload["head"])

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])

    if sampler is not None and payload.get("sampler"):
        sampler.load_state(payload["sampler"])

    if restore_random:
        restore_rng(payload["rng"])

    return payload
