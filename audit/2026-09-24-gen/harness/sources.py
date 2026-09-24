"""
Отпечаток исходников аудита и его сверка.

Одна реализация на всех: её зовёт и state/fingerprint.py, и
обвязка прогона. Прогон обязан сверяться с зафиксированным
состоянием ДО и ПОСЛЕ работы, иначе результаты двух разных
состояний кода смешаются незаметно.

Отпечаток НЕ обновляется автоматически ни при каком расхождении:
перезапись базы аудита — осознанное действие человека.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]

STORED = AUDIT / "state" / "sources.json"

# Что считается исходником аудита: код и справочники.
TREES = ("src", "reference")

SKIP = ("__pycache__", ".pytest_cache", ".mypy_cache")


class SourcesChanged(RuntimeError):
    """
    Дерево исходников не совпадает с зафиксированным.
    """


def files() -> list[Path]:

    found: list[Path] = []

    for tree in TREES:
        for path in sorted((ROOT / tree).rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP for part in path.parts):
                continue
            found.append(path)

    return found


def digest(path: Path) -> str:

    sha = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)

    return sha.hexdigest()


def snapshot() -> dict[str, str]:
    """
    Путь относительно корня -> sha256 файла.
    """

    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
        for path in files()
    }


def state_id(current: dict[str, str] | None = None) -> str:
    """
    Один идентификатор состояния кода.

    Считается по отсортированному списку пар «путь, хеш», поэтому
    меняется и от содержимого файла, и от появления или пропажи
    файла.
    """

    current = snapshot() if current is None else current

    joined = "\n".join(f"{name} {value}" for name, value in sorted(current.items()))

    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def stored() -> dict[str, str]:

    if not STORED.exists():
        raise SourcesChanged(
            f"нет {STORED}: состояние аудита не зафиксировано, "
            "выполните state/fingerprint.py"
        )

    return json.loads(STORED.read_text(encoding="utf-8"))


def differences(before: dict[str, str], after: dict[str, str]) -> list[str]:

    return sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )


def verify(when: str, expected: dict[str, str] | None = None) -> str:
    """
    Сверка с зафиксированным состоянием. Возвращает state_id.

    when — человеку понятное место проверки («до прогона»,
    «после прогона»), оно попадает в текст ошибки.
    """

    expected = stored() if expected is None else expected

    current = snapshot()

    changed = differences(expected, current)

    if changed:
        raise SourcesChanged(
            f"{when}: исходники не совпадают с зафиксированными "
            f"({len(changed)} файлов). Прогон остановлен, отпечаток НЕ обновлён.\n"
            + "\n".join(f"  {name}" for name in changed[:20])
            + ("\n  ..." if len(changed) > 20 else "")
        )

    return state_id(current)


__all__ = [
    "AUDIT",
    "ROOT",
    "STORED",
    "SourcesChanged",
    "differences",
    "snapshot",
    "state_id",
    "stored",
    "verify",
]
