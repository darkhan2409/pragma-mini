from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import dumps_json, read_json, sha256_bytes, sha256_file, write_json


# ============================================================
# ИДЕЯ
# ============================================================
#
# Результат этапа можно переиспользовать только если совпало
# всё, от чего он зависит: входные артефакты, секция конфига,
# реестры и версия реализации. Четыре части складываются в один
# отпечаток; он пишется рядом с выходом этапа и сверяется перед
# пропуском. Изменение любой части пересчитывает этап, а вместе
# с ним — всё, что читает его выход (нижестоящий этап держит
# отпечаток вышестоящего среди своих входов).
#
# preprocessing_manifest.json собирает результаты этапов по мере
# их выполнения; итоговый вид он получает на пятом этапе.
# ============================================================


FINGERPRINT_DIR = "stage_fingerprints"
MANIFEST_FILE = "preprocessing_manifest.json"


def stage_fingerprint(
    stage: str,
    version: str,
    inputs: Mapping[str, str],
    config_section: Mapping[str, Any],
    registries: Mapping[str, str] | None = None,
) -> dict:
    """
    inputs: имя входа -> sha256; registries: имя реестра -> sha256.
    """

    body = {
        "stage": stage,
        "version": version,
        "inputs": dict(sorted(inputs.items())),
        "config_sha256": sha256_bytes(dumps_json(dict(config_section)).encode("utf-8")),
        "registries": dict(sorted((registries or {}).items())),
    }

    body["fingerprint"] = sha256_bytes(dumps_json(body).encode("utf-8"))

    return body


def output_digests(paths: list[Path], root: Path) -> dict[str, str]:
    """
    sha256 выходов этапа, адресованных относительно набора.
    """

    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def outputs_intact(stored: dict | None, root: Path) -> bool:
    """
    Все выходы этапа на месте и не изменились с момента записи.
    """

    if not stored:
        return False

    outputs = stored.get("outputs")

    if not outputs:
        return False

    for name, digest in outputs.items():
        path = root / name
        if not path.exists() or sha256_file(path) != digest:
            return False

    return True


def fingerprint_path(processed_dir: Path, stage: str, group: str | None) -> Path:
    name = stage if group is None else f"{stage}__{group}"
    return Path(processed_dir) / FINGERPRINT_DIR / f"{name}.json"


def load_fingerprint(path: Path) -> dict | None:
    """
    Маркер этапа. Повреждённый файл это отсутствующий маркер:
    непрочитанный отпечаток не может подтвердить актуальность.
    """

    return read_json_or_none(Path(path))


def read_json_or_none(path: Path) -> dict | None:

    if not path.exists():
        return None

    try:
        value = read_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None

    return value if isinstance(value, dict) else None


def is_current(existing: dict | None, fingerprint: dict) -> bool:
    return existing is not None and existing.get("fingerprint") == fingerprint["fingerprint"]


def save_fingerprint(path: Path, fingerprint: dict) -> None:
    write_json(path, fingerprint)


# ============================================================
# МАНИФЕСТ ПРЕПРОЦЕССИНГА
# ============================================================


def update_manifest(processed_dir: Path, stage: str, group: str | None, entry: Mapping[str, Any]) -> dict:
    """
    Записывает результат этапа в preprocessing_manifest.json,
    не трогая записи других этапов и групп.

    Повреждённый файл начинается заново: восстановить чужие
    записи из него нельзя, а каждый этап всё равно сверяет свою
    при следующем запуске и вернёт её на место.
    """

    path = Path(processed_dir) / MANIFEST_FILE

    manifest = read_json_or_none(path) or {"stages": {}}

    stages = manifest.setdefault("stages", {})
    slot = stages.setdefault(stage, {})

    if group is None:
        slot.update(dict(entry))
    else:
        slot[group] = dict(entry)

    write_json(path, manifest)

    return manifest


def stage_entry(processed_dir: Path, stage: str, group: str | None) -> Any:
    """
    Запись этапа в общем манифесте либо None, если её там нет.
    """

    manifest = read_json_or_none(Path(processed_dir) / MANIFEST_FILE)

    if manifest is None:
        return None

    slot = manifest.get("stages", {}).get(stage)

    if slot is None:
        return None

    return slot if group is None else slot.get(group)


__all__ = [
    "FINGERPRINT_DIR",
    "MANIFEST_FILE",
    "fingerprint_path",
    "output_digests",
    "outputs_intact",
    "is_current",
    "load_fingerprint",
    "read_json_or_none",
    "save_fingerprint",
    "stage_entry",
    "stage_fingerprint",
    "update_manifest",
]
