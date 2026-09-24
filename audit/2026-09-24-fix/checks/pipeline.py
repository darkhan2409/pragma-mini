"""
Граница RAW -> препроцессинг.

    python audit/2026-09-24-fix/checks/pipeline.py --run m-train

build_group принимает raw_dir и out_dir явно
(src/preprocessing/canonical/build.py:78-83), поэтому подменять
глобалы не нужно: настоящий data/ не участвует вовсе.

Проверяется:

  PREP-1  число строк сохраняется: одна строка canonical = одна строка RAW
  PREP-2  время переведено в UTC и совпадает с исходным моментом
  PREP-3  служебные поля в слой не проходят
  PREP-4  порядок клиента не перемешан
  PREP-5  битая выгрузка останавливает этап понятной ошибкой
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT / "harness"))

from sources import verify  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str, checked: int = 0, bad: int = 0) -> None:
    RESULTS.append(
        {"check": name, "verdict": verdict, "detail": detail, "checked": checked, "violations": bad}
    )
    print(f"[{verdict}] {name}: проверено {checked}, нарушений {bad} — {detail}")


def main() -> int:

    parser = argparse.ArgumentParser(prog="pipeline")
    parser.add_argument("--run", required=True)
    parser.add_argument("--group", default="train")

    args = parser.parse_args()

    code_state = verify("до границы конвейера")

    work = (RUNS / f"pipeline-{args.run}").resolve()

    if RUNS.resolve() not in work.parents:
        raise SystemExit(f"отказ: {work} вне {RUNS}")

    if work.exists():
        shutil.rmtree(work)

    raw = work / "01_raw" / args.group
    raw.mkdir(parents=True)

    source = RUNS / args.run

    for name in ("events.parquet", "profile.parquet", "manifest.json"):
        shutil.copy2(source / name, raw / name)

    from src.preprocessing.canonical.build import build_group
    from src.preprocessing.rawdata import RawContractError
    from src.preprocessing.settings import PreprocessingConfig

    before = pq.read_table(raw / "events.parquet")

    config = PreprocessingConfig.load(None)

    out_dir = work / "02_preprocessed" / args.group

    report = build_group(raw, out_dir, config, args.group)

    after_path = out_dir / "events.parquet"
    after = pq.read_table(after_path)

    # --- PREP-1 ---

    same = before.num_rows == after.num_rows

    record("PREP-1 число строк", "PASS" if same else "FAIL",
           f"RAW {before.num_rows} -> canonical {after.num_rows}",
           before.num_rows, 0 if same else abs(before.num_rows - after.num_rows))

    # --- PREP-2: время в UTC ---

    kind = str(after.schema.field("event_time").type)
    utc = kind == "timestamp[us, tz=UTC]"

    raw_times = sorted(before.column("event_time").to_pylist())
    new_times = sorted(
        moment.isoformat() for moment in after.column("event_time").to_pylist()
    )

    matched = 0

    for left, right in zip(raw_times, new_times):
        if datetime.fromisoformat(left) == datetime.fromisoformat(right):
            matched += 1

    record("PREP-2 время в UTC", "PASS" if utc and matched == len(raw_times) else "FAIL",
           f"тип {kind}; совпало моментов {matched} из {len(raw_times)}",
           len(raw_times), len(raw_times) - matched)

    # --- PREP-3: служебные поля ---

    from src.preprocessing.projection import INTERNAL_FIELDS

    leaked = sorted(set(after.column_names) & set(INTERNAL_FIELDS))

    record("PREP-3 служебные поля", "PASS" if not leaked else "FAIL",
           f"объявлено внутренними {len(INTERNAL_FIELDS)}, прошло в слой: {leaked or 'ничего'}",
           len(INTERNAL_FIELDS), len(leaked))

    # --- PREP-4: порядок клиента ---

    clients = after.column("client_id").to_pylist()
    moments = after.column("event_time").to_pylist()

    broken = 0
    seen: set = set()
    previous_client = None
    previous_time = None

    for client, when in zip(clients, moments):
        if client != previous_client:
            if client in seen:
                broken += 1
            seen.add(client)
            previous_client, previous_time = client, when
            continue
        if previous_time is not None and when < previous_time:
            broken += 1
        previous_time = when

    record("PREP-4 порядок внутри клиента", "PASS" if broken == 0 else "FAIL",
           "строки клиента идут подряд и по неубыванию времени",
           after.num_rows, broken)

    # --- PREP-5: битая выгрузка ---

    spoiled = work / "01_raw" / "val"
    spoiled.mkdir(parents=True, exist_ok=True)

    for name in ("events.parquet", "profile.parquet", "manifest.json"):
        shutil.copy2(source / name, spoiled / name)

    manifest = json.loads((spoiled / "manifest.json").read_text(encoding="utf-8"))
    manifest["events_rows"] = manifest["events_rows"] + 1
    (spoiled / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    try:
        build_group(spoiled, work / "02_preprocessed" / "val", config, "val")
        record("PREP-5 битая выгрузка", "FAIL",
               "этап прошёл на выгрузке, где манифест обещает лишнюю строку", 1, 1)
    except RawContractError as error:
        record("PREP-5 битая выгрузка", "PASS",
               f"доменная ошибка: {error}", 1, 0)
    except Exception as error:  # noqa: BLE001
        record("PREP-5 битая выгрузка", "FAIL",
               f"неожиданное исключение {type(error).__name__}: {error}", 1, 1)

    if verify("после границы конвейера") != code_state:
        record("состояние кода", "FAIL", "исходники изменились во время проверки")

    destination = AUDIT / "evidence" / f"pipeline-{args.run}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"report": report, "checks": RESULTS}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
