from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .activity import is_client_action
from .config import FUTURE_DIR, FUTURE_LABEL_GROUPS, HORIZON, RAW_DIR, cutoff, manifest
from .raw import client_blocks


# ============================================================
# ИСТОЧНИКИ ГРУППЫ
# ============================================================
#
# У строки группы три источника, и каждый отпечатан отдельно:
#
#   история признаков   data/01_raw/<group>/events.parquet (t < T)
#   анкета признаков    data/01_raw/<group>/profile.parquet (на T)
#   метка               события (T, T + HORIZON]:
#                         export — та же выгрузка (val, test);
#                         future — продолжение группы (train),
#                                  FUTURE_DIR/<group>/.
#
# Продолжение принимается, только если оно продолжает ИМЕННО текущую
# выгрузку (её sha256 событий и анкеты записаны в future.json при
# генерации, и прошлое там сверено по клиентам), начинается с её
# конца, целиком покрывает окно метки и цело само (sha256 файла).
# Клиенты, у которых прошлое в продолжении разошлось с выгрузкой,
# записаны там же и из строк группы исключаются.
# ============================================================


FUTURE_FILE = "future.json"

FUTURE_EVENTS = "events.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_future(group: str, raw_dir: Path = RAW_DIR, future_dir: Path = FUTURE_DIR) -> dict:
    """
    future.json продолжения группы, проверенный против текущей выгрузки.
    """
    directory = future_dir / group
    path = directory / FUTURE_FILE
    if not path.exists():
        raise ValueError(
            f"нет {path}: метка {group} берётся из продолжения — python -m src.generator.continuation "
            f"{group} --days {HORIZON.days} --out {directory}"
        )

    record = json.loads(path.read_text(encoding="utf-8"))
    exported = manifest(group, raw_dir)
    moment = cutoff(group, raw_dir)

    source = record.get("source", {})
    for key in ("events_sha256", "profile_sha256", "period_end"):
        if source.get(key) != exported[key]:
            raise ValueError(f"{path}: продолжение не этой выгрузки {group} (другой {key})")

    if not record.get("profile_past_matches") or "prefix" not in record:
        raise ValueError(f"{path}: прошлое продолжения не сверено с выгрузкой")

    if datetime.fromisoformat(record["period_start"]) != moment:
        raise ValueError(f"{path}: продолжение начинается не с T {moment.isoformat()}")

    if datetime.fromisoformat(record["period_end"]) <= moment + HORIZON:
        raise ValueError(f"{path}: продолжение до {record['period_end']} не покрывает окно метки")

    if sha256(directory / FUTURE_EVENTS) != record["events_sha256"]:
        raise ValueError(f"{directory / FUTURE_EVENTS} не совпадает со своим {FUTURE_FILE}")

    return record


def provenance(group: str, raw_dir: Path = RAW_DIR, future_dir: Path = FUTURE_DIR) -> dict:
    """
    T, окно метки и отпечатки трёх источников группы.
    """
    exported = manifest(group, raw_dir)
    moment = cutoff(group, raw_dir)
    future = group in FUTURE_LABEL_GROUPS

    return {
        "T": moment.isoformat(),
        "target_window": f"({moment.isoformat()}, {(moment + HORIZON).isoformat()}]",
        "feature_history_events_sha256": exported["events_sha256"],
        "feature_profile_sha256": exported["profile_sha256"],
        "target_source": "future" if future else "export",
        "target_events_sha256": (
            read_future(group, raw_dir, future_dir)["events_sha256"] if future else exported["events_sha256"]
        ),
    }


def diverged_clients(group: str, raw_dir: Path = RAW_DIR, future_dir: Path = FUTURE_DIR) -> set[str]:
    """
    Клиенты, чьё прошлое в продолжении разошлось с выгрузкой: их
    продолжение — не продолжение наблюдаемой истории, и метка у них
    неизвестна. Из строк группы они исключаются, а не получают churn = 1.
    """
    return set(read_future(group, raw_dir, future_dir).get("diverged_clients", []))


def acting_clients(group: str, raw_dir: Path = RAW_DIR, future_dir: Path = FUTURE_DIR) -> set[str]:
    """
    Клиенты с собственным действием в окне метки (T, T + HORIZON] по
    продолжению группы. Событие ровно в T в окно не входит, ровно в
    T + HORIZON — входит; события банка не действие.
    """
    read_future(group, raw_dir, future_dir)
    moment = cutoff(group, raw_dir)

    acting: set[str] = set()
    for block in client_blocks(future_dir / group / FUTURE_EVENTS):
        t = block["t"]
        window = is_client_action(block) & ((t > moment) & (t <= moment + HORIZON)).to_numpy()
        acting.update(block.loc[window, "client_id"])
    return acting
