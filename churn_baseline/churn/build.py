from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .activity import is_client_action
from .config import DATA_DIR, GROUPS, HORIZON, RAW_DIR, REPORTS_DIR, cutoff, manifest
from .features import compute
from .profile import CATEGORICAL, DESCRIPTIONS as PROFILE_DESCRIPTIONS, profile_at
from .raw import client_blocks, read_profile
from .target import labels


# Колонки строки, которые не признаки.
KEYS: tuple[str, ...] = ("client_id", "group", "T", "churn")

CHANGE_COLUMNS = ["client_id", "t", "raw_row", "field_name", "old_value"]


def build(group: str, raw_dir: Path = RAW_DIR, data_dir: Path = DATA_DIR, reports_dir: Path = REPORTS_DIR) -> dict:
    """
    Признаки и target одной группы клиентов. Читает только выгрузку этой
    группы, пишет data/<group>/features.parquet и meta.json.
    """
    if group not in GROUPS:
        raise ValueError(f"группа {group!r} не из {GROUPS}")

    moment = cutoff(group, raw_dir)
    profile = read_profile(raw_dir / group / "profile.parquet")

    feature_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    change_parts: list[pd.DataFrame] = []
    descriptions: dict[str, str] | None = None

    for block in client_blocks(raw_dir / group / "events.parquet"):
        action = is_client_action(block)
        features, described = compute(block, moment, action)
        if descriptions is None:
            descriptions = described
        elif list(described) != list(descriptions):
            raise RuntimeError("состав признаков разошёлся между блоками")
        feature_parts.append(features)
        target_parts.append(labels(block, moment, action))
        change_parts.append(block.loc[block["type"] == "profile_change", CHANGE_COLUMNS])

    features = pd.concat(feature_parts)
    target = pd.concat(target_parts)
    stranger = set(features.index) - set(profile["client_id"])
    if stranger:
        raise ValueError(f"у {len(stranger)} клиентов лент нет анкеты своей группы")

    changes = pd.concat(change_parts, ignore_index=True)
    known = profile[profile["client_id"].isin(features.index)]
    at_cutoff = profile_at(known, changes, moment)

    table = target.join(at_cutoff).join(features)
    population = table[table["has_action_before"]].copy()
    population.insert(0, "T", pd.Timestamp(moment).tz_convert("UTC"))
    population.insert(0, "group", group)
    population = population.drop(columns=["has_action_before", "has_history_before"])
    population = population.reset_index()
    ordered = list(KEYS) + [name for name in population.columns if name not in KEYS]
    population = population[ordered].sort_values("client_id", ignore_index=True)

    out = data_dir / group
    out.mkdir(parents=True, exist_ok=True)
    population.to_parquet(out / "features.parquet", index=False)

    churn = int(population["churn"].sum())
    meta = {
        "group": group,
        "T": moment.isoformat(),
        "window": f"({moment.isoformat()}, {(moment + HORIZON).isoformat()}]",
        "clients_in_profile": int(len(profile)),
        "clients_with_events": int(len(table)),
        "without_history_before_T": int((~table["has_history_before"]).sum()),
        "history_without_client_action_before_T": int(
            (table["has_history_before"] & ~table["has_action_before"]).sum()
        ),
        "population": int(len(population)),
        "churn_1": churn,
        "churn_0": int(len(population) - churn),
        "churn_rate": churn / len(population),
        "features": len(population.columns) - len(KEYS),
        "raw_events_sha256": manifest(group, raw_dir)["events_sha256"],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    write_feature_list(reports_dir / "features.md", {**PROFILE_DESCRIPTIONS, **(descriptions or {})})
    return meta


def write_feature_list(path: Path, descriptions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Признаки churn_baseline",
        "",
        "Все признаки считаются по событиям строго раньше cutoff T и по анкете, откаченной на T.",
        "Окно «за w дней» — события с T − w ≤ t < T.",
        "",
        "| признак | тип | описание |",
        "|---|---|---|",
    ]
    for name, text in descriptions.items():
        kind = "категориальный" if name in CATEGORICAL else "числовой"
        lines.append(f"| `{name}` | {kind} | {text} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Признаки и target churn одной группы")
    parser.add_argument("group", choices=GROUPS)
    args = parser.parse_args(argv)
    meta = build(args.group)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
