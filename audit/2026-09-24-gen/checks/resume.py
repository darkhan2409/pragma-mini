"""
Прерывание, продолжение и порча черновика.

    python audit/2026-09-24-gen/checks/resume.py

Прерывание делается воспроизводимо: обёртка над emit._run_batch
поднимает исключение после N завершённых пачек. Это тот же
момент, что и внешнее убийство процесса между пачками, но
повторяемый, а не случайный. Сама функция при этом не меняется —
подменяется глобал модуля, как и в наблюдении денег.

Проверяется:

  R1   продолжение даёт тот же результат, что непрерывный прогон
  R2a  маркер без части
  R2b  часть без маркера
  R2c  повреждённый маркер
  R2d  подменённая часть при прежнем числе строк
  R2e  чужая карточка прогона
  R2f  части без карточки прогона

Community_size уменьшен до 8, чтобы пачек было много: при
значениях по умолчанию 64 клиента дают одну пачку и прерывать
нечего.
"""

from __future__ import annotations

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

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.generator import emit  # noqa: E402

CLIENTS = 64
COMMUNITY = 8
CHUNK = 8
SEED = 100
START = datetime.fromisoformat("2024-01-01")
END = datetime.fromisoformat("2026-01-01")

RESULTS: list[dict] = []


def record(name: str, verdict: str, detail: str) -> None:
    RESULTS.append({"check": name, "verdict": verdict, "detail": detail})
    print(f"[{verdict}] {name}: {detail}")


def fresh(name: str) -> Path:
    out = (RUNS / name).resolve()
    if RUNS.resolve() not in out.parents:
        raise SystemExit(f"отказ: {out} вне {RUNS}")
    if out.exists():
        shutil.rmtree(out)
    return out


def generate(out: Path, resume: bool = False, crash_after: int | None = None):
    """
    Прогон; при crash_after — падение после N завершённых пачек.
    """

    original = emit._run_batch
    done = {"count": 0}

    def crashing(job):
        result = original(job)
        done["count"] += 1
        if crash_after is not None and done["count"] >= crash_after:
            raise RuntimeError(f"аудит: управляемое прерывание после {done['count']} пачек")
        return result

    emit._run_batch = crashing if crash_after is not None else original

    try:
        return emit.generate_dataset(
            total_clients=CLIENTS,
            out_dir=out,
            seed=SEED,
            world_seed=42,
            history_start=START,
            history_end=END,
            workers=1,
            chunk_clients=CHUNK,
            community_size=COMMUNITY,
            resume=resume,
            quiet=True,
        )
    finally:
        emit._run_batch = original


def digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_output(left: Path, right: Path) -> bool:
    return all(
        digest(left / name) == digest(right / name)
        for name in ("events.parquet", "profile.parquet", "manifest.json")
    )


def expect_error(name: str, call, fragment: str) -> None:
    """
    Ожидается отказ с определённой причиной.
    """

    try:
        call()
    except emit.GenerationError as error:
        if fragment in str(error):
            record(name, "PASS", f"отказано по нужной причине: {error}")
        else:
            record(name, "FAIL", f"отказано, но по другой причине: {error}")
    except Exception as error:  # noqa: BLE001 - тип исключения и есть результат
        record(name, "FAIL", f"неожиданное исключение {type(error).__name__}: {error}")
    else:
        record(name, "FAIL", "прогон прошёл там, где обязан был отказать")


# ============================================================


def main() -> int:

    code_state = verify("до проверок resume")

    # --- эталон: непрерывный прогон ---

    straight = fresh("r0-straight")
    generate(straight)

    batches = CLIENTS // COMMUNITY
    record("подготовка", "СПРАВКА", f"сообществ {batches}, пачек столько же при chunk={CHUNK}")

    # --- R1: прерывание и продолжение ---

    broken = fresh("r1-resume")

    try:
        generate(broken, crash_after=3)
        record("R1 прерывание", "ОШИБКА ПРОВЕРКИ", "прогон не прервался")
    except RuntimeError as error:
        record("R1 прерывание", "СПРАВКА", str(error))

    markers = sorted((broken / "parts").glob("batch-*.json"))
    record("R1 состояние после прерывания", "СПРАВКА",
           f"маркеров {len(markers)}, карточка прогона есть: {(broken / 'run.json').exists()}")

    generate(broken, resume=True)

    if same_output(straight, broken):
        record("R1 продолжение", "PASS",
               "продолженный прогон побайтово совпал с непрерывным по всем трём файлам")
    else:
        record("R1 продолжение", "FAIL", "результат отличается от непрерывного")

    # --- R2a: маркер без части ---

    case = fresh("r2a-marker-no-part")
    try:
        generate(case, crash_after=3)
    except RuntimeError:
        pass

    victim = sorted((case / "parts").glob("events-*.parquet"))[0]
    victim.unlink()

    expect_error("R2a маркер без части", lambda: generate(case, resume=True),
                 "маркер есть, а части")

    # --- R2b: часть без маркера ---

    case = fresh("r2b-part-no-marker")
    try:
        generate(case, crash_after=3)
    except RuntimeError:
        pass

    marker = sorted((case / "parts").glob("batch-*.json"))[-1]
    marker.unlink()

    generate(case, resume=True)

    if same_output(straight, case):
        record("R2b часть без маркера", "PASS",
               "пачка пересчитана заново, результат совпал с непрерывным: "
               "готовность определяет маркер, а не наличие части")
    else:
        record("R2b часть без маркера", "FAIL", "результат отличается")

    # --- R2c: повреждённый маркер ---

    case = fresh("r2c-broken-marker")
    try:
        generate(case, crash_after=3)
    except RuntimeError:
        pass

    marker = sorted((case / "parts").glob("batch-*.json"))[0]
    marker.write_text("{это не json", encoding="utf-8")

    try:
        generate(case, resume=True)
        record("R2c повреждённый маркер", "FAIL", "прогон прошёл на битом маркере")
    except json.JSONDecodeError as error:
        record("R2c повреждённый маркер", "СПРАВКА",
               f"отказ есть, но это JSONDecodeError из недр, а не доменная ошибка: {error}")
    except emit.GenerationError as error:
        record("R2c повреждённый маркер", "PASS", f"доменная ошибка: {error}")
    except Exception as error:  # noqa: BLE001
        record("R2c повреждённый маркер", "СПРАВКА",
               f"{type(error).__name__}: {error}")

    # --- R2d: подменённая часть при прежнем числе строк ---

    case = fresh("r2d-swapped-part")
    try:
        generate(case, crash_after=3)
    except RuntimeError:
        pass

    victim = sorted((case / "parts").glob("events-*.parquet"))[0]
    table = pq.read_table(victim)

    if table.num_rows:
        # Те же строки, но значения client_id испорчены: число
        # строк прежнее, содержимое другое.
        spoiled = table.set_column(
            table.schema.get_field_index("client_id"),
            "client_id",
            pa.array(["ПОРЧА"] * table.num_rows, type=pa.string()),
        )
        pq.write_table(spoiled, victim, compression="zstd")

        generate(case, resume=True)

        events = pq.read_table(case / "events.parquet")
        spoiled_rows = sum(
            1 for value in events.column("client_id").to_pylist() if value == "ПОРЧА"
        )

        if spoiled_rows:
            record("R2d подменённая часть", "СПРАВКА",
                   f"порча прошла в итог: {spoiled_rows} строк. Контракт защиты "
                   "содержимого части не обещает — сверяется только число строк")
        else:
            record("R2d подменённая часть", "PASS", "порча не попала в итог")
    else:
        record("R2d подменённая часть", "НЕ ПРОВЕРЕНО", "первая часть пуста")

    # --- R2e: чужая карточка прогона ---

    case = fresh("r2e-foreign-card")
    try:
        generate(case, crash_after=3)
    except RuntimeError:
        pass

    card = json.loads((case / "run.json").read_text(encoding="utf-8"))
    card["seed"] = card["seed"] + 1
    (case / "run.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")

    expect_error("R2e чужая карточка", lambda: generate(case, resume=True),
                 "продолжение чужого прогона")

    # --- R2f: части без карточки ---

    case = fresh("r2f-no-card")
    try:
        generate(case, crash_after=3)
    except RuntimeError:
        pass

    (case / "run.json").unlink()

    expect_error("R2f части без карточки", lambda: generate(case, resume=True),
                 "продолжать нечего")

    if verify("после проверок resume") != code_state:
        record("состояние кода", "FAIL", "исходники изменились во время проверки")

    destination = AUDIT / "evidence" / "resume.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n-> {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
