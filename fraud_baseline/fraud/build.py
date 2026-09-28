from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DATA_DIR, ELIGIBLE_FROM, GROUPS, NEGATIVE_ONE_IN, RAW_DIR, REPORTS_DIR, manifest
from .features import CATEGORICAL, MILESTONES, PROFILE_FIELDS, compute, eligible, milliseconds
from .raw import client_blocks, read_profile


# Колонки строки, которые не признаки. episode_kind — вид эпизода из
# повтора генератора: только для отчёта о трудных отрицательных.
KEYS: tuple[str, ...] = ("client_id", "group", "event_time", "raw_row", "type", "fraud", "episode_kind", "weight")


def prepared_profile(path: Path) -> pd.DataFrame:
    """
    Анкета выгрузки для признаков на момент операции: поля на as_of (их
    откатывает features._profile) и моменты вех в мс.
    """
    raw = read_profile(path)
    out = raw.set_index("client_id")[["gender", "birth_date", *PROFILE_FIELDS]].astype(object)
    for milestone in MILESTONES:
        moments = [
            min((item["event_time"] for item in items if item["type"] == milestone), default=None)
            for items in raw["lifelong"]
        ]
        stamps = pd.Series(pd.to_datetime(moments, utc=True), index=out.index)
        out[f"{milestone}_ms"] = np.where(stamps.notna(), milliseconds(stamps.fillna(pd.Timestamp(0, tz="UTC"))), np.nan)
    return out


def sampled(raw_row: np.ndarray) -> np.ndarray:
    """
    Детерминированная одна из NEGATIVE_ONE_IN отрицательных строк train:
    по хешу номера строки RAW, без генератора случайных чисел.
    """
    return pd.util.hash_array(raw_row.astype(np.int64)) % NEGATIVE_ONE_IN == 0


def build(group: str, raw_dir: Path = RAW_DIR, data_dir: Path = DATA_DIR, reports_dir: Path = REPORTS_DIR) -> dict:
    """
    Строки и признаки одной группы. Метки берутся из data/<group>/labels.parquet
    (fraud.replay), признаки — только из выгрузки этой группы.
    """
    if group not in GROUPS:
        raise ValueError(f"группа {group!r} не из {GROUPS}")
    labels_path = data_dir / group / "labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"нет {labels_path}: сначала python -m fraud.replay {group}")

    labels = pd.read_parquet(labels_path).set_index("raw_row")
    profile = prepared_profile(raw_dir / group / "profile.parquet")
    out_path = data_dir / group / "features.parquet"
    temporary = out_path.with_suffix(".tmp")

    writer: pq.ParquetWriter | None = None
    descriptions: dict[str, str] | None = None
    counts = {"eligible": 0, "eligible_fraud": 0, "eligible_false_positive": 0, "rows": 0}
    matched: set[int] = set()

    for block in client_blocks(raw_dir / group / "events.parquet"):
        stranger = set(pd.unique(block["client_id"])) - set(profile.index)
        if stranger:
            raise ValueError(f"у {len(stranger)} клиентов ленты нет анкеты своей группы")

        episode = block["raw_row"].map(labels["episode_kind"])
        marked = episode.notna().to_numpy()
        fraud = block["raw_row"].map(labels["fraud"]).fillna(0).to_numpy(dtype=np.int8)
        matched |= set(block["raw_row"].to_numpy()[marked].tolist())

        rows_mask = eligible(block).copy()
        if (marked & (block["t"] >= ELIGIBLE_FROM).to_numpy() & ~rows_mask).any():
            raise ValueError("помеченная операция не попала в строки датасета")
        counts["eligible"] += int(rows_mask.sum())
        counts["eligible_fraud"] += int((rows_mask & (fraud == 1)).sum())
        counts["eligible_false_positive"] += int((rows_mask & marked & (fraud == 0)).sum())

        weight = np.ones(len(block), dtype=np.float32)
        if group == "train":
            keep_negative = sampled(block["raw_row"].to_numpy())
            rows_mask &= marked | keep_negative
            weight[~marked] = float(NEGATIVE_ONE_IN)
        rows = np.flatnonzero(rows_mask)
        if not len(rows):
            continue

        columns, described = compute(block, rows, profile)
        if descriptions is None:
            descriptions = described
        elif list(described) != list(descriptions):
            raise RuntimeError("состав признаков разошёлся между блоками")

        frame = pd.DataFrame(
            {
                "client_id": block["client_id"].to_numpy()[rows],
                "group": group,
                "event_time": block["t"].iloc[rows].to_numpy(),
                "raw_row": block["raw_row"].to_numpy()[rows],
                "type": block["type"].to_numpy()[rows],
                "fraud": fraud[rows],
                "episode_kind": episode.to_numpy()[rows],
                "weight": weight[rows],
                **columns,
            }
        )
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            # Вид эпизода у блока без помеченных строк весь пуст, и тип колонки
            # вывелся бы как null: он задаётся явно.
            schema = table.schema.set(
                table.schema.get_field_index("episode_kind"), pa.field("episode_kind", pa.string())
            )
            writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        writer.write_table(table.cast(schema))
        counts["rows"] += len(frame)

    if writer is None:
        raise RuntimeError(f"в группе {group} нет ни одной строки")
    writer.close()
    temporary.replace(out_path)

    missing = set(labels.index) - matched
    if missing:
        raise RuntimeError(f"{len(missing)} меток не нашли своей строки в выгрузке")

    written = pd.read_parquet(out_path, columns=["type", "fraud", "episode_kind", "weight"])
    meta = {
        "group": group,
        "eligible_from": ELIGIBLE_FROM.isoformat(),
        **counts,
        "negative_sampling": f"1 из {NEGATIVE_ONE_IN}, вес {NEGATIVE_ONE_IN}" if group == "train" else "нет",
        "fraud_1": int(written["fraud"].sum()),
        "fraud_0": int((written["fraud"] == 0).sum()),
        "false_positive_rows": int((written["episode_kind"] == "false_positive").sum()),
        "fraud_by_type": written[written["fraud"] == 1].groupby("type").size().to_dict(),
        "rows_by_type": written.groupby("type").size().to_dict(),
        "fraud_by_episode_kind": written[written["episode_kind"].notna()].groupby("episode_kind").size().to_dict(),
        "features": len(descriptions or {}),
        "raw_events_sha256": manifest(group, raw_dir)["events_sha256"],
    }
    (data_dir / group / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    write_feature_list(reports_dir / "features.md", descriptions or {})
    return meta


def write_feature_list(path: Path, descriptions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Признаки fraud_baseline",
        "",
        "Строка — покупка клиента (reason=purchase) или transfer_out. Момент решения — время операции.",
        "История — события того же клиента строго раньше операции; окно «за w» — t_i − w ≤ t < t_i.",
        "",
        "| признак | тип | описание |",
        "|---|---|---|",
    ]
    for name, text in descriptions.items():
        kind = "категориальный" if name in CATEGORICAL else "числовой"
        lines.append(f"| `{name}` | {kind} | {text} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Строки и признаки fraud одной группы")
    parser.add_argument("group", choices=GROUPS)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.group), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
