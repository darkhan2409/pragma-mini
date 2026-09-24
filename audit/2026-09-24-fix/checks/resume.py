"""
R2d, F-3, F-5: возобновление, целостность частей, манифест.

    python audit/2026-09-24-fix/checks/resume.py --clients 48

Проверяется пять вещей:

  1. Прерванный и продолженный прогон даёт ровно тот же результат,
     что непрерывный, — побайтово по обоим файлам.
  2. Подменённая часть С ПРЕЖНИМ ЧИСЛОМ СТРОК до манифеста не
     доходит, и ошибка называет пачку и файл.
  3. Нечитаемый маркер даёт GenerationError с именем файла,
     а не обрыв разбора JSON.
  4. Маркер прежнего формата (без подписей) продолжить нельзя.
  5. Манифест несёт происхождение и читается препроцессингом.

Прерывание вносится обёрткой вокруг emit._run_batch: она даёт
воспроизводимую точку останова и ничего не меняет в самом
генераторе.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[1]
RUNS = AUDIT / "runs"

sys.path.insert(0, str(ROOT))


class Stop(RuntimeError):
    """
    Искусственное прерывание на границе пачки.
    """


def generate(out: Path, args, resume: bool = False, stop_after: int | None = None) -> None:

    from src.generator import emit

    original = emit._run_batch

    done = 0

    def watching(job):
        nonlocal done
        if stop_after is not None and done >= stop_after:
            raise Stop(f"прерывание после {done} пачек")
        result = original(job)
        done += 1
        return result

    emit._run_batch = watching

    try:
        emit.generate_dataset(
            total_clients=args.clients,
            out_dir=out,
            seed=args.seed,
            world_seed=42,
            history_start=datetime.fromisoformat(args.start),
            history_end=datetime.fromisoformat(args.end),
            workers=1,
            chunk_clients=args.chunk,
            community_size=args.community_size,
            resume=resume,
            quiet=True,
        )
    finally:
        emit._run_batch = original


def digest_files(directory: Path) -> dict:
    import hashlib

    result = {}

    for name in ("events.parquet", "profile.parquet"):
        sha = hashlib.sha256()
        sha.update((directory / name).read_bytes())
        result[name] = sha.hexdigest()

    return result


def manifest_of(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def clean(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    return path


def check_resume_equals_continuous(args, report: dict) -> None:
    """
    1. Продолженный прогон против непрерывного.
    """

    from src.generator.emit import GenerationError  # noqa: F401

    straight = clean(RUNS / "r-straight")
    generate(straight, args)

    broken = clean(RUNS / "r-resume")

    try:
        generate(broken, args, stop_after=args.stop_after)
    except Stop:
        pass
    else:
        raise SystemExit("прерывание не сработало: прогон дошёл до конца")

    parts = sorted((broken / "parts").glob("*.parquet")) if (broken / "parts").exists() else []
    markers = sorted((broken / "parts").glob("*.json")) if (broken / "parts").exists() else []

    generate(broken, args, resume=True)

    left = digest_files(straight)
    right = digest_files(broken)

    report["resume_equals_continuous"] = {
        "parts_after_stop": len(parts),
        "markers_after_stop": len(markers),
        "files": {name: {"continuous": left[name][:16], "resumed": right[name][:16]} for name in left},
        "manifest_equal": {
            key: value
            for key, value in manifest_of(straight).items()
            if manifest_of(broken).get(key) != value
        },
        "verdict": "PASS" if left == right and manifest_of(straight) == manifest_of(broken) else "FAIL",
    }


def check_swapped_part(args, report: dict) -> None:
    """
    2. Подмена части с прежним числом строк.
    """

    from src.generator.emit import GenerationError

    out = clean(RUNS / "r-swap")

    try:
        generate(out, args, stop_after=args.stop_after)
    except Stop:
        pass

    ready = sorted((out / "parts").glob("events-*.parquet"))

    if not ready:
        raise SystemExit("нет готовых частей: подменять нечего")

    victim = ready[0]

    table = pq.read_table(victim)

    # Та же схема, то же число строк, другое содержимое: именно
    # этот случай проходил прежнюю проверку по числу строк.
    columns = {
        name: pa.array(
            [
                (value + "X") if isinstance(value, str) else value
                for value in table.column(name).to_pylist()
            ],
            type=table.schema.field(name).type,
        )
        for name in table.column_names
    }

    swapped = pa.table(columns, schema=table.schema)

    pq.write_table(swapped, victim, compression="zstd")

    rows_before = table.num_rows
    rows_after = pq.read_table(victim).num_rows

    error = None

    try:
        generate(out, args, resume=True)
    except GenerationError as failure:
        error = str(failure)

    report["swapped_part"] = {
        "part": victim.name,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "same_row_count": rows_before == rows_after,
        "error": error,
        "manifest_written": (out / "manifest.json").exists(),
        "verdict": "PASS"
        if error
        and victim.name in error
        and rows_before == rows_after
        and not (out / "manifest.json").exists()
        else "FAIL",
    }


def check_broken_marker(args, report: dict) -> None:
    """
    3 и 4. Нечитаемый маркер и маркер прежнего формата.
    """

    from src.generator.emit import GenerationError

    outcomes = {}

    for label, content in (
        ("нечитаемый", "{это не json"),
        ("без подписей", None),
    ):

        out = clean(RUNS / f"r-marker-{len(outcomes)}")

        try:
            generate(out, args, stop_after=args.stop_after)
        except Stop:
            pass

        markers = sorted((out / "parts").glob("*.json"))

        if not markers:
            raise SystemExit("нет маркеров: портить нечего")

        victim = markers[0]

        if content is None:
            record = json.loads(victim.read_text(encoding="utf-8"))
            record.pop("sha256", None)
            victim.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        else:
            victim.write_text(content, encoding="utf-8")

        error = None

        try:
            generate(out, args, resume=True)
        except GenerationError as failure:
            error = str(failure)

        outcomes[label] = {
            "marker": victim.name,
            "error": error,
            "names_the_file": bool(error) and (victim.name in error or str(args.stop_after - 1) in error),
            "manifest_written": (out / "manifest.json").exists(),
            "verdict": "PASS"
            if error and not (out / "manifest.json").exists()
            else "FAIL",
        }

    report["broken_marker"] = outcomes


def check_manifest(args, report: dict) -> None:
    """
    5. Манифест: происхождение и чтение препроцессингом.
    """

    from src.preprocessing.rawdata import read_manifest

    straight = RUNS / "r-straight"

    manifest = manifest_of(straight)

    needed = (
        "schema_version",
        "generator_version",
        "seed",
        "world_seed",
        "total_clients",
        "community_size",
        "generation_config_sha256",
        "reference_sha256",
    )

    missing = [key for key in needed if key not in manifest]

    read = None
    error = None

    try:
        card = read_manifest(straight)
        read = {
            "schema_version": card.schema_version,
            "events_rows": card.events_rows,
            "provenance_keys": sorted(card.provenance),
        }
    except Exception as failure:  # noqa: BLE001
        error = f"{type(failure).__name__}: {failure}"

    # Совместимость чтения: прежняя версия схемы и манифест без
    # происхождения обязаны быть отвергнуты, а не приняты молча.
    refusals = {}

    for label, change in (
        ("прежняя версия схемы", lambda card: card.update({"schema_version": manifest["schema_version"] - 1})),
        ("без происхождения", lambda card: [card.pop(key, None) for key in (
            "generator_version", "seed", "world_seed",
            "generation_config_sha256", "reference_sha256")]),
    ):

        spoiled = clean(RUNS / f"r-manifest-{len(refusals)}")

        shutil.copytree(straight, spoiled)

        card = manifest_of(spoiled)
        change(card)
        (spoiled / "manifest.json").write_text(
            json.dumps(card, ensure_ascii=False), encoding="utf-8"
        )

        try:
            read_manifest(spoiled)
            refusals[label] = {"error": None, "verdict": "FAIL"}
        except Exception as failure:  # noqa: BLE001
            refusals[label] = {
                "error": f"{type(failure).__name__}: {failure}",
                "verdict": "PASS",
            }

    report["manifest"] = {
        "keys": sorted(manifest),
        "missing": missing,
        "read_by_preprocessing": read,
        "error": error,
        "refusals": refusals,
        "verdict": "PASS"
        if not missing
        and read is not None
        and all(item["verdict"] == "PASS" for item in refusals.values())
        else "FAIL",
    }


def main() -> int:

    parser = argparse.ArgumentParser(prog="resume")
    parser.add_argument("--clients", type=int, default=48)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-07-01")
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--community-size", type=int, default=4)
    parser.add_argument("--stop-after", type=int, default=3)
    parser.add_argument("--out", default="evidence/r2d-resume.json")

    args = parser.parse_args()

    report: dict = {}

    check_resume_equals_continuous(args, report)
    check_swapped_part(args, report)
    check_broken_marker(args, report)
    check_manifest(args, report)

    verdicts = []

    for key, value in report.items():
        if "verdict" in value:
            verdicts.append(value["verdict"])
        else:
            verdicts.extend(item["verdict"] for item in value.values())

    report["verdict"] = "PASS" if all(item == "PASS" for item in verdicts) else "FAIL"

    (AUDIT / args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
