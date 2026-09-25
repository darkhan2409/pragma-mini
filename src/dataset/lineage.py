from __future__ import annotations

from pathlib import Path

from src.preprocessing.artifacts import read_json, write_json
from src.preprocessing.profile_state import PROFILE_SEMANTICS

from .settings import DATASET_FORMAT


# ============================================================
# ПРОИСХОЖДЕНИЕ ЭТАПОВ 06-08
# ============================================================
#
# Этапы 06 (временные позиции), 07 (батчи) и 08 (маски) только
# переносят анкету из набора 05, и их схемы от смысла анкеты не
# зависят. Каталог прежней сборки выглядел бы исправным и молча
# вернул бы в модель анкету прежнего смысла.
#
# Поэтому каждый из них пишет рядом с результатом lineage.json —
# формат набора и смысл анкеты, из которых он собран, — а
# читатель сверяет его с текущими. Нет файла или он другой —
# каталог отвергается с командой пересборки.
# ============================================================


LINEAGE_FILE = "lineage.json"


def lineage() -> dict:
    """
    Из чего собирается каталог текущим кодом.
    """

    return {"dataset_format": DATASET_FORMAT, "profile_semantics": PROFILE_SEMANTICS}


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
