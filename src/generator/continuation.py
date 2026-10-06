from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config
from .config import RAW_DIR, DatasetGroup
from .emit import EVENTS_SCHEMA, default_workers, generate_dataset


# ============================================================
# ПРОДОЛЖЕНИЕ ГРУППЫ ДЛЯ МЕТОК
# ============================================================
#
#   python -m src.generator.continuation train --days 60 \
#       --out churn_baseline/data/future/train
#
# Те же клиенты группы, прожитые дальше конца её выгрузки на days
# суток, — источник меток, которым выгрузке взяться неоткуда.
# Генератор не меняется: окно выгрузки у него только режет ленту
# (config.py, «ГОРИЗОНТ ПЛАНИРОВАНИЯ»), поэтому та же группа с более
# поздним концом окна — это продолжение тех же жизней. Прежний
# последний приход в банк сохраняется: новых клиентов нет.
#
# Это обещание генератора проверяется на полном масштабе, а не
# принимается на веру: прошлое сверяется по каждому клиенту:
#
#   - строки раньше конца выгрузки те же и в том же порядке, что в
#     самой выгрузке (sha256 содержимого по клиенту);
#   - вехи и записи о работе анкеты раньше конца выгрузки те же;
#   - клиенты те же, новых нет.
#
# Клиент, у которого прошлое разошлось, — не продолжение
# наблюдаемой истории: в хвост он не идёт, а в future.json записан.
# Разошедшихся больше MAX_DIVERGED или другие клиенты — отказ, и
# ничего не записано.
#
# На диск ложится только хвост — события не раньше конца выгрузки —
# и future.json о том, из чего он получен. Анкета продолжения (снимок
# на новый конец) не сохраняется: признаки берутся из выгрузки.
#
# Каталог вывода — вне data/: этапы PRAGMA читают только
# data/01_raw/<группа>, и хвост до них не доходит.
# ============================================================


FUTURE_FILE = "future.json"

EVENTS_FILE = "events.parquet"

# Запас за концом окна метки: окно выгрузки полуоткрыто, а конец
# окна метки в метку входит.
MARGIN = timedelta(days=1)

# Время строк — местное со смещением банка; сравнивается по первым
# 19 символам, пока смещение у всех строк одно.
OFFSET = "+05:00"

# Строк за раз при проходе по ленте: группа строк train — до 900
# тысяч, и её копии в памяти не помещались в 6 ГБ.
BATCH_ROWS = 100_000

# Доля клиентов, чьё прошлое в продолжении может разойтись с
# выгрузкой. С Generator V1 (1.0.0) прошлое от конца окна не зависит
# совсем: решения дня читают только прошлое, а строки с будущим
# временем — только в свой день (tests/test_generator_prefix.py,
# проверка на 3 × 1000 клиентах — audit/2026-10-06-generator-v1).
# Любое расхождение — ошибка генератора, и продолжение отказывает.
MAX_DIVERGED = 0.0


class ContinuationError(RuntimeError):
    """
    Продолжение не совпало с выгрузкой или его нельзя собрать.
    """


def _file_sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _local(text: str) -> str:
    """
    Момент строки без смещения — для сравнения строк времени.
    """

    return text[:19]


def _split(table: pa.Table, boundary: str) -> tuple[pa.Table, pa.Table]:
    """
    Строки раньше границы и не раньше неё, в порядке файла.
    """

    times = table.column("event_time")

    if not pc.all(pc.ends_with(times, OFFSET)).as_py():
        raise ContinuationError(f"в ленте есть время не со смещением {OFFSET}")

    before = pc.less(pc.utf8_slice_codeunits(times, 0, 19), boundary)

    return table.filter(before), table.filter(pc.invert(before))


def _rows(table: pa.Table) -> bytes:
    """
    Строки как текст: поля через \\x1f, строка на строку — тот же
    отпечаток, что у тестов генератора, но не зависящий от байтов
    parquet.
    """

    if not table.num_rows:
        return b""

    joined = pc.binary_join_element_wise(
        *[table.column(name) for name in ("client_id", "event_time", "source", "payload")], "\x1f"
    )

    return ("\n".join(joined.to_pylist()) + "\n").encode("utf-8")


class _Tape:
    """
    Счёт, отпечаток и клиенты потока строк.
    """

    def __init__(self) -> None:
        self.rows = 0
        self.digest = hashlib.sha256()
        self.clients: set[str] = set()

    def add(self, table: pa.Table) -> None:
        self.rows += table.num_rows
        self.digest.update(_rows(table))
        self.clients.update(table.column("client_id").to_pylist())


def _batches(path: Path):
    """
    Лента по BATCH_ROWS строк, в порядке файла.
    """

    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=EVENTS_SCHEMA.names):
        yield pa.Table.from_batches([batch])


def _client_digests(path: Path, boundary: str) -> tuple[dict[str, str], int]:
    """
    Отпечаток строк раньше границы по каждому клиенту и их число.
    """

    found: dict[str, hashlib._Hash] = {}
    rows = 0

    for table in _batches(path):
        early, _ = _split(table, boundary)
        rows += early.num_rows
        for row in zip(*[early.column(name).to_pylist() for name in EVENTS_SCHEMA.names]):
            found.setdefault(row[0], hashlib.sha256()).update("\x1f".join(row).encode("utf-8"))

    return {client: digest.hexdigest() for client, digest in found.items()}, rows


def _write_tail(path: Path, boundary: str, tail: Path, exclude: set[str]) -> _Tape:
    """
    Строки не раньше границы, кроме клиентов exclude, — в tail.
    """

    written = _Tape()
    skip = pa.array(sorted(exclude), pa.string())

    with pq.ParquetWriter(tail, EVENTS_SCHEMA, compression="zstd") as writer:
        for table in _batches(path):
            _, late = _split(table, boundary)
            late = late.filter(pc.invert(pc.is_in(late.column("client_id"), value_set=skip)))
            if late.num_rows:
                written.add(late)
                writer.write_table(late.cast(EVENTS_SCHEMA))

    return written


def _profile_past(path: Path, moment: datetime) -> dict[str, tuple]:
    """
    Прошлое анкеты по клиентам: вехи и записи о работе раньше moment.
    """

    past = {}

    for row in pq.read_table(path, columns=["client_id", "lifelong", "employment"]).to_pylist():
        past[row["client_id"]] = (
            tuple((item["type"], item["event_time"], item["source_id"])
                  for item in row["lifelong"] or () if item["event_time"] < moment),
            tuple((item["start_date"], item["record_time"])
                  for item in row["employment"] or () if item["record_time"] < moment),
        )

    return past


def _differences(source: Path, extended: Path, boundary: str, clients: list[str]) -> list[dict]:
    """
    Где разошлись ленты клиентов: число строк до границы и первое
    расхождение.
    """

    wanted = set(clients)

    def rows_of(path: Path) -> dict[str, list[tuple]]:
        rows: dict[str, list[tuple]] = {}
        for table in _batches(path):
            early, _ = _split(table, boundary)
            for row in zip(*[early.column(name).to_pylist() for name in EVENTS_SCHEMA.names]):
                if row[0] in wanted:
                    rows.setdefault(row[0], []).append(row)
        return rows

    left, right = rows_of(source), rows_of(extended)

    found = []

    for client in clients:
        was, now = left.get(client, []), right.get(client, [])
        index = next((i for i, (a, b) in enumerate(zip(was, now)) if a != b), min(len(was), len(now)))
        found.append({
            "client_id": client,
            "rows": [len(was), len(now)],
            "first_difference": index,
            "time": (was[index] if index < len(was) else now[index])[1] if max(len(was), len(now)) > index else None,
        })

    return found


def extend(settings: DatasetGroup, source: Path, out: Path, days: int, *, workers: int = 1,
           quiet: bool = False, max_diverged: float = MAX_DIVERGED) -> dict:
    """
    Продолжение выгрузки source (группы settings) на days суток в out.
    """

    source, out = Path(source), Path(out)

    # generate_dataset начинает с того, что стирает свой каталог:
    # рядом с данными конвейера ему не место.
    for path in (out, out.parent):
        if path.resolve().is_relative_to(config.DATA_DIR.resolve()):
            raise ContinuationError(f"{out}: продолжение пишется вне {config.DATA_DIR}")

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))

    declared = {
        "seed": settings.seed,
        "total_clients": settings.clients,
        "period_start": config.event_time_text(settings.history_start),
        "period_end": config.event_time_text(settings.history_end),
        "registration_end": config.event_time_text(settings.registration_end),
    }

    changed = sorted(key for key, value in declared.items() if manifest.get(key) != value)

    if changed:
        raise ContinuationError(f"{source}: выгрузка не той группы, различаются {', '.join(changed)}")

    for name in ("events", "profile"):
        if _file_sha256(source / f"{name}.parquet") != manifest[f"{name}_sha256"]:
            raise ContinuationError(f"{source}/{name}.parquet не совпадает со своим manifest")

    end = settings.history_end + timedelta(days=days) + MARGIN
    boundary = _local(manifest["period_end"])
    work = out.parent / f".{out.name}.work"

    # Готовое продолжение с тем же паспортом — то же самое, что
    # сгенерировать его заново: генератор детерминирован. Так сверка,
    # прерванная после генерации, не повторяет сорок минут работы.
    planned = {
        "period_start": manifest["period_start"],
        "period_end": config.event_time_text(end),
        "registration_end": manifest["registration_end"],
        **{key: manifest[key] for key in (
            "seed", "world_seed", "total_clients", "community_size", "chunk_clients",
            "generation_config_sha256", "reference_sha256", "generator_version", "schema_version",
        )},
    }

    finished = work / "manifest.json"

    if not finished.exists() or {
        key: json.loads(finished.read_text(encoding="utf-8")).get(key) for key in planned
    } != planned:
        generate_dataset(
            total_clients=settings.clients,
            out_dir=work,
            seed=settings.seed,
            world_seed=manifest["world_seed"],
            history_start=settings.history_start,
            history_end=end,
            registration_end=settings.registration_end,
            workers=workers,
            chunk_clients=manifest["chunk_clients"],
            community_size=manifest["community_size"],
            quiet=quiet,
        )

    extended = json.loads(finished.read_text(encoding="utf-8"))

    if {key: extended.get(key) for key in planned} != planned:
        raise ContinuationError("продолжение сгенерировано не с параметрами выгрузки")

    for name in ("events", "profile"):
        if _file_sha256(work / f"{name}.parquet") != extended[f"{name}_sha256"]:
            raise ContinuationError(f"{work}/{name}.parquet не совпадает со своим manifest")

    moment = datetime.fromisoformat(manifest["period_end"])

    before, rows_before = _client_digests(source / EVENTS_FILE, boundary)
    after, rows_after = _client_digests(work / EVENTS_FILE, boundary)
    past_before = _profile_past(source / "profile.parquet", moment)
    past_after = _profile_past(work / "profile.parquet", moment)

    clients = set(past_before)

    if set(past_after) != clients or not set(before) <= clients:
        raise ContinuationError("в продолжении другие клиенты — ничего не записано")

    # Прошлое сверяется по каждому клиенту. Разошедшийся клиент не
    # продолжение наблюдаемой истории: его метка неизвестна, и в хвост
    # он не идёт.
    diverged = sorted(
        client for client in clients
        if before.get(client) != after.get(client) or past_before[client] != past_after[client]
    )

    if len(diverged) > max_diverged * len(clients):
        report = {
            "diverged": len(diverged),
            "clients": len(clients),
            "rows": [rows_before, rows_after],
            "first": _differences(source / EVENTS_FILE, work / EVENTS_FILE, boundary, diverged[:10]),
        }
        raise ContinuationError(
            f"продолжение меняет прошлое у {len(diverged)} клиентов из {len(clients)} — больше "
            f"{max_diverged:.0%}, ничего не записано; черновик оставлен в {work}:\n"
            + json.dumps(report, ensure_ascii=False, indent=1, default=str)
        )

    out.mkdir(parents=True, exist_ok=True)
    tail_path = out / f".{EVENTS_FILE}.tmp"

    tail = _write_tail(work / EVENTS_FILE, boundary, tail_path, set(diverged))

    if not tail.clients <= clients:
        tail_path.unlink(missing_ok=True)
        raise ContinuationError(f"в продолжении новые клиенты: {len(tail.clients - clients)}")

    tail_path.replace(out / EVENTS_FILE)

    record = {
        "group_source": str(source),
        "source": {
            "period_end": manifest["period_end"],
            "events_sha256": manifest["events_sha256"],
            "profile_sha256": manifest["profile_sha256"],
        },
        # Прошлое сверено по клиентам: строки до конца выгрузки, вехи и
        # записи о работе. Разошедшиеся исключены из хвоста.
        "prefix": {"rows_export": rows_before, "rows_continuation": rows_after, "clients": len(clients)},
        "profile_past_matches": True,
        "diverged_clients": diverged,
        "diverged_details": _differences(source / EVENTS_FILE, work / EVENTS_FILE, boundary, diverged),
        "period_start": manifest["period_end"],
        "period_end": config.event_time_text(end),
        "days": days,
        "events_rows": tail.rows,
        "events_sha256": _file_sha256(out / EVENTS_FILE),
        "clients": len(tail.clients),
        "generator": {
            "generator_version": extended["generator_version"],
            "seed": extended["seed"],
            "world_seed": extended["world_seed"],
            "generation_config_sha256": extended["generation_config_sha256"],
        },
    }

    (out / FUTURE_FILE).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.rmtree(work)

    return record


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="python -m src.generator.continuation")
    parser.add_argument("group", choices=sorted(config.DATASETS))
    parser.add_argument("--days", type=int, required=True, help="сколько суток после конца выгрузки")
    parser.add_argument("--out", type=Path, required=True, help="каталог хвоста, вне data/")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        record = extend(
            config.DATASETS[args.group], RAW_DIR / args.group, args.out, args.days,
            workers=args.workers or default_workers(),
        )
    except ContinuationError as error:
        print(f"[continuation] {error}")
        raise SystemExit(2)

    print(
        f"[continuation] {args.group}: прошлое совпало у {record['prefix']['clients'] - len(record['diverged_clients'])} "
        f"клиентов из {record['prefix']['clients']}; разошедшиеся исключены ({len(record['diverged_clients'])}); "
        f"хвост {record['events_rows']} строк до {record['period_end']} → {args.out}"
    )


if __name__ == "__main__":
    main()


__all__ = ["ContinuationError", "FUTURE_FILE", "extend", "main"]
