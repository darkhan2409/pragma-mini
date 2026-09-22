from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from src.preprocessing.artifacts import write_json, write_text

from .collate import collate
from .encoding import causes_as_of, encode_history
from .inputs import DatasetInputs
from .reader import Dataset
from .report import render_check_md
from .sample import build_sample
from .storage import INDEX_FILE
from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# Проверка собранного набора: не «собралось без падения», а
# «собралось правильно».
#
# Самая ценная проверка последняя: несколько примеров собираются
# заново из смыслового слоя на ту же дату и сверяются с
# записанными массив за массивом. Она ловит то, что не поймают
# инварианты, — например, молчаливую подмену версии события или
# сдвиг на границе среза.
# ============================================================


CHECK_JSON_FILE = "check_report.json"
CHECK_MD_FILE = "check_report.md"


class CheckError(ValueError):
    """
    Проверку выполнить нельзя.
    """


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def check_dataset(dataset: Dataset, inputs: DatasetInputs | None = None,
                  recompute: int = 0) -> dict:
    """
    Проверяет набор целиком и, если попросили, пересчитывает
    несколько примеров из семантики.
    """

    checks: list[dict] = []

    manifest = dataset.manifest
    index = dataset.index.to_pylist()

    # --- состав групп ---

    by_group: dict[str, set[str]] = {}

    for row in index:
        by_group.setdefault(row["group"], set()).add(row["client_id"])

    shared: list[str] = []

    names = sorted(by_group)

    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            common = by_group[names[first]] & by_group[names[second]]
            if common:
                shared.append(f"{names[first]} и {names[second]}: {len(common)}")

    checks.append(
        _check(
            "группы не пересекаются по клиентам",
            not shared,
            "; ".join(shared) if shared else f"групп {len(names)}, общих клиентов нет",
        )
    )

    if inputs is not None:

        mismatched = [
            group
            for group in by_group
            if group in inputs.groups and by_group[group] != set(inputs.groups[group].clients)
        ]

        checks.append(
            _check(
                "состав групп совпадает с разделением",
                not mismatched,
                "расходятся: " + ", ".join(mismatched) if mismatched else "совпадает во всех группах",
            )
        )

    # --- инварианты примеров ---

    artifacts = inputs.artifacts if inputs is not None else None

    broken: list[str] = []
    totals = {"events": 0, "tokens": 0, "values": 0, "eligible": 0}
    empty = 0
    without_targets = 0

    for group in dataset.groups:
        for sample in dataset.iter_samples(group):

            if artifacts is not None:
                try:
                    sample.check(artifacts)
                except Exception as error:  # noqa: BLE001 - причина попадает в отчёт
                    broken.append(f"{sample.sample_id}: {error}")

            totals["events"] += sample.n_events
            totals["tokens"] += sample.n_tokens
            totals["values"] += sample.n_values
            totals["eligible"] += sample.n_eligible_events

            empty += int(sample.n_events == 0)
            without_targets += int(not sample.has_targets)

    checks.append(
        _check(
            "инварианты примеров",
            not broken,
            "; ".join(broken[:3]) if broken else (
                "проверить нечем: словарь не передан" if artifacts is None
                else f"проверено примеров {len(index)}"
            ),
        )
    )

    # --- согласие счётчиков ---

    declared = manifest["counts"]

    checks.append(
        _check(
            "число примеров сходится с манифестом",
            len(index) == declared["samples"],
            f"в указателе {len(index)}, в манифесте {declared['samples']}",
        )
    )

    index_events = sum(row["n_events"] for row in index)
    index_tokens = sum(row["n_tokens"] for row in index)

    checks.append(
        _check(
            "суммы по примерам сходятся с указателем",
            index_events == totals["events"] and index_tokens == totals["tokens"],
            f"событий {totals['events']} против {index_events}, "
            f"токенов {totals['tokens']} против {index_tokens}",
        )
    )

    # --- пустые и без целей ---

    checks.append(
        _check(
            "пустые истории сохранены",
            empty == sum(1 for row in index if row["n_events"] == 0),
            f"примеров без событий {empty}",
        )
    )

    checks.append(
        _check(
            "примеры без целей помечены",
            without_targets == sum(1 for row in index if not row["has_targets"]),
            f"примеров без целей {without_targets}",
        )
    )

    # --- анкета ---

    known_empty = [
        row for row in index
        if row["profile_state"] == "known" and row["profile_tokens"] <= 1
    ]

    no_profile = [row for row in index if row["profile_state"] != "known"]

    checks.append(
        _check(
            "известная анкета отличается от отсутствующей",
            all(row["profile_tokens"] == 1 for row in no_profile),
            f"без анкеты {len(no_profile)} (у них только маркер), "
            f"известных, но пустых {len(known_empty)}",
        )
    )

    # --- потерянные цели в оценочных группах ---

    fit_group = manifest["identity"]["vocabulary"].get("artifact_id") and (
        inputs.artifacts.manifest["group"] if inputs is not None else None
    )

    lost = [
        row for row in index
        if row["excluded_eligible"] and (fit_group is None or row["group"] != fit_group)
    ]

    checks.append(
        _check(
            "в оценочных группах цели не потеряны",
            not lost,
            f"потерявших цели примеров {len(lost)}" if lost else "потерь нет",
        )
    )

    # --- словарь ---

    if inputs is not None:

        inputs.artifacts.verify()

        checks.append(
            _check(
                "словарь не изменился",
                True,
                f"отпечаток {inputs.artifacts.manifest['artifact_id']} проверен заново",
            )
        )

    # --- смешанный batch ---

    batch_report = _mixed_batch(dataset)

    checks.append(
        _check(
            "смешанный batch собирается",
            batch_report is not None,
            "примеров в batch {samples}, строк {rows}, ширина {width}".format(**batch_report)
            if batch_report else "в наборе нет примеров",
        )
    )

    # --- пересчёт ---

    recomputed = _recompute(dataset, inputs, recompute) if (inputs and recompute) else []

    if recomputed:
        checks.append(
            _check(
                "пересчёт из семантики совпадает",
                all(item["identical"] for item in recomputed),
                f"пересчитано примеров {len(recomputed)}",
            )
        )

    violations = sum(1 for item in checks if not item["ok"])

    return {
        "stage": "check",
        "format_version": FORMAT_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_id": dataset.dataset_id,
        "readiness": manifest["readiness"],
        "checks": checks,
        "violations": violations,
        "ok": violations == 0,
        "totals": totals,
        "batch": batch_report,
        "recomputed": recomputed,
    }


def _mixed_batch(dataset: Dataset) -> dict | None:
    """
    Batch из короткой, длинной и пустой истории.

    Именно такая смесь ловит ошибки границ: одинаковые примеры
    выравниваются одинаково и ничего не проверяют.
    """

    rows = dataset.index.to_pylist()

    if not rows:
        return None

    group = rows[0]["group"]

    same = [row for row in rows if row["group"] == group]

    wanted: list[str] = []

    empty = next((row["sample_id"] for row in same if row["n_events"] == 0), None)
    longest = max(same, key=lambda row: row["n_events"])["sample_id"]
    shortest = min(
        (row for row in same if row["n_events"]),
        key=lambda row: row["n_events"],
        default=None,
    )

    for item in (empty, shortest["sample_id"] if shortest else None, longest):
        if item is not None and item not in wanted:
            wanted.append(item)

    samples = [
        sample for sample in dataset.iter_samples(group) if sample.sample_id in wanted
    ]

    if not samples:
        return None

    batch = collate(samples)

    # Границы обязаны указывать на те же токены, что лежали в
    # примере: проверяется восстановлением, а не доверием.
    for number, sample in enumerate(samples):
        for slot in range(sample.n_events):

            row = int(batch.history_rows[number, slot])
            length = int(sample.event_lengths[slot])
            start = int(sample.event_starts[slot])

            stored = sample.key_ids[start : start + length]

            if not np.array_equal(batch.event_key_ids[row, :length], stored):
                raise CheckError(
                    f"{sample.sample_id}: событие {slot} в batch не совпадает с записанным"
                )

    return {
        "samples": batch.n_samples,
        "rows": batch.n_rows,
        "width": batch.width,
        "empty": int((batch.n_events == 0).sum()),
        "real_tokens": int(batch.event_token_mask.sum()),
        "target_candidates": int(batch.target_candidate_mask.sum()),
    }


def _recompute(dataset: Dataset, inputs: DatasetInputs, count: int) -> list[dict]:
    """
    Несколько примеров, собранных заново из смыслового слоя.
    """

    rows = dataset.index.to_pylist()

    if not rows:
        return []

    # Интереснее всего крайние: пустая история, самая длинная и
    # пример с исправленной записью.
    chosen = sorted(rows, key=lambda row: (row["n_events"], row["sample_id"]))

    picked = [chosen[0]] + chosen[-(count - 1):] if count > 1 else [chosen[0]]

    seen: set[str] = set()
    wanted = []

    for row in picked:
        if row["sample_id"] not in seen:
            seen.add(row["sample_id"])
            wanted.append(row)

    out: list[dict] = []

    limit = inputs.tokenizer_config.max_pieces_per_value

    for row in wanted:

        group = row["group"]
        entry = inputs.groups[group]
        cutoff = row["cutoff"] if isinstance(row["cutoff"], datetime) else datetime.fromisoformat(row["cutoff"])

        history = entry.history(row["client_id"], cutoff)

        encoded = encode_history(
            inputs.artifacts, history, limit,
            cause_of=causes_as_of(entry.corpus.store, row["client_id"], cutoff),
        )

        fresh = build_sample(
            artifacts=inputs.artifacts,
            encoded=encoded,
            group=group,
            window=entry.window,
            weight=entry.weight,
            sources=sources,
            policy=inputs.config.context,
        )

        stored = next(
            sample for sample in dataset.iter_samples(group)
            if sample.sample_id == row["sample_id"]
        )

        identical = all(
            np.array_equal(getattr(fresh, name), getattr(stored, name))
            for name in (
                "key_ids", "value_ids", "positions", "event_starts", "event_lengths",
                "value_event", "value_start", "value_length", "value_key_id",
                "profile_key_ids", "profile_value_ids", "profile_positions",
                "event_eligible",
            )
        ) and fresh.sample_id == stored.sample_id

        out.append(
            {
                "sample_id": row["sample_id"],
                "events": fresh.n_events,
                "tokens": fresh.n_tokens,
                "identical": identical,
            }
        )

    return out


def write_check(directory: Path, report: dict) -> list[Path]:

    directory = Path(directory)

    outputs = [directory / CHECK_JSON_FILE, directory / CHECK_MD_FILE]

    write_json(outputs[0], report)
    write_text(outputs[1], render_check_md(report))

    return outputs


__all__ = [
    "CHECK_JSON_FILE",
    "CHECK_MD_FILE",
    "CheckError",
    "check_dataset",
    "write_check",
]
