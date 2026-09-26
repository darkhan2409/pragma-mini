from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json, write_json
from src.preprocessing.profile_state import LIFELONG_TYPES, PROFILE_SEMANTICS
from src.preprocessing.settings import PreprocessingConfig

from .settings import DATASET_FORMAT


# ============================================================
# ПРОИСХОЖДЕНИЕ ЭТАПОВ 06-09 И 11
# ============================================================
#
# Этапы 06 (временные позиции), 07 (батчи) и 08 (маски) только
# переносят анкету из набора 05. Каталог прежней сборки выглядел
# бы исправным и молча вернул бы в модель анкету прежнего смысла.
# Веса этапов 09 (таблица эмбеддингов по словарю) и 11 (энкодер
# анкеты) собраны под ту же анкету и тот же словарь.
#
# Поэтому каждый из них пишет рядом с результатом lineage.json —
# формат набора, смысл анкеты, набор её вех и окна групп
# (контекст и маскирование), из которых он собран, — а читатель
# сверяет его с текущими. Нет файла или он другой — каталог
# отвергается с командой пересборки. Окно маскирования решает,
# какие события вообще могут стать целями: маска 08 прежнего окна
# молча оценивала бы не тот период.
# ============================================================


LINEAGE_FILE = "lineage.json"


def lineage() -> dict:
    """
    Из чего собирается каталог текущим кодом.
    """

    return {
        "dataset_format": DATASET_FORMAT,
        "profile_semantics": PROFILE_SEMANTICS,
        "profile_lifelong_types": list(LIFELONG_TYPES),
        "windows": {
            group: window.as_dict()
            for group, window in sorted(PreprocessingConfig.load(None).windows.items())
        },
    }


def write_lineage(directory: Path) -> None:

    write_json(Path(directory) / LINEAGE_FILE, lineage())


def lineage_problem(directory: Path, command: str) -> str | None:
    """
    Почему каталог нельзя читать, или None, если можно.
    """

    path = Path(directory) / LINEAGE_FILE

    if not path.exists():
        return f"нет {path}: каталог собран прежним кодом — выполните {command} заново"

    found = read_json(path)

    if found != lineage():
        return f"{path}: собран из {found}, а нужно {lineage()} — выполните {command} заново"

    return None


__all__ = ["LINEAGE_FILE", "lineage", "lineage_problem", "write_lineage"]
