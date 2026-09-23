from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import params as params_module
from . import rng as rng_module
from . import config
from .config import (
    GENERATOR_VERSION,
    RAW_DIR,
    SCHEMA_VERSION,
    SEED,
    WORLD_SEED,
)
from .profile import PROFILE_SCHEMA
from .world import communities


# ============================================================
# ВЫГРУЗКА RAW
# ============================================================
#
# Единица симуляции это СООБЩЕСТВО, а runtime-чанк только
# распределяет сообщества между воркерами. Поэтому при одном
# seed и одних параметрах содержимое датасета не зависит ни от
# числа воркеров, ни от размера чанка, ни от порядка завершения.
#
# Выгрузка это ДВЕ таблицы: events.parquet и profile.parquet.
#
# event_time выгружается СТРОКОЙ ISO 8601 со смещением
# часового пояса, как его отдала бы банковская система.
# Перевод в UTC и timestamp — работа препроцессинга, а не
# генератора.
# Конверт события — четыре колонки, тип события внутри payload.
# Справочники мерчантов, продуктов и географии рядом не
# выкладываются: они вход генератора, а в событие попадают
# только поля выбранного объекта.
#
# Скрытые состояния симуляции — черты, стресс, жизненный цикл,
# мошенничество, отношения, покупки вне банка — влияют на
# события и наружу не выдаются.
# ============================================================


EVENTS_SCHEMA = pa.schema(
    [
        ("client_id", pa.string()),
        ("event_time", pa.string()),
        ("source", pa.string()),
        ("payload", pa.string()),
    ]
)

TABLES = {
    "events": ("events.parquet", EVENTS_SCHEMA),
    "profile": ("profile.parquet", PROFILE_SCHEMA),
}

PARTS_DIR = "parts"

# Карточка прогона: с чем он был начат. Продолжение чужого
# прогона собрало бы датасет из кусков разных миров.
RUN_FILE = "run.json"


class GenerationError(RuntimeError):
    """
    Выгрузку нельзя собрать: прогон не сходится сам с собой.
    """


# ============================================================
# ЗАПИСЬ
# ============================================================


def _write(path: Path, rows: list, schema: pa.Schema) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(rows, schema=schema)

    pq.write_table(table, path, compression="zstd")


def _write_json(path: Path, payload: dict) -> None:
    """
    Файл появляется целиком или не появляется вовсе.

    Прерванный прогон не имеет права оставить наполовину
    написанный маркер: по нему пачка считалась бы готовой.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def _batch_marker(out: Path, index: int) -> Path:
    return out / PARTS_DIR / f"batch-{index:05d}.json"


# ============================================================
# ЗАДАЧА ВОРКЕРА
# ============================================================


_WORKER: dict = {}


def _worker_init(seed: int, params_path: str | None, catalog_scale: float | None,
                 community_size: int | None, world_seed: int | None = None,
                 horizon: tuple | None = None) -> None:

    # Дочерний процесс импортирует config заново и о горизонте
    # группы ничего не знает: его надо поставить здесь.
    if horizon is not None:
        config.activate_horizon(datetime.fromisoformat(horizon[0]), datetime.fromisoformat(horizon[1]))

    settings = _build_params(params_path, catalog_scale, community_size)

    params_module.activate(settings)
    rng_module.configure(seed, settings.fingerprint(), world_seed)

    _WORKER["ready"] = True


def _build_params(params_path: str | None, catalog_scale: float | None,
                  community_size: int | None):

    settings = params_module.load(params_path)

    overrides: dict = {}

    if catalog_scale is not None:
        overrides["merchants"] = {"catalog_scale": float(catalog_scale)}

    if community_size is not None:
        overrides["relationships"] = {"community_size": int(community_size)}

    if overrides:
        settings = settings.with_overrides(overrides)

    return settings


def _run_batch(job: tuple) -> tuple:
    """
    Пакет сообществ: симуляция и запись своих part-файлов.
    """

    batch_index, community_ids, total_clients, out_dir = job

    from .engine import run_community

    rows: dict[str, list] = {name: [] for name in TABLES}

    for community_id in community_ids:

        members = communities.members(community_id, total_clients)

        if not members:
            continue

        result = run_community(community_id, members)

        rows["events"].extend(result.events)
        rows["profile"].extend(result.profile_rows)

    out = Path(out_dir)

    counts: dict[str, int] = {}

    for name, (_, schema) in TABLES.items():
        counts[name] = len(rows[name])
        _write(out / PARTS_DIR / f"{name}-{batch_index:05d}.parquet", rows[name], schema)

    # Маркер пишется ПОСЛЕДНИМ и целиком: пачка готова только
    # тогда, когда все её part-файлы на месте. Итоговые суммы
    # собираются из маркеров, поэтому продолженный прогон знает
    # и про те пачки, которых сам не считал.
    _write_json(
        _batch_marker(out, batch_index),
        {
            "batch": batch_index,
            "communities": list(community_ids),
            "counts": counts,
        },
    )

    return batch_index, counts


# ============================================================
# СКЛЕЙКА
# ============================================================


def _merge_parts(out: Path, name: str, batches: int) -> int:
    """
    Склейка part-файлов в итоговую таблицу.

    Пишется во временный файл и переименовывается: прерванная
    склейка не оставляет обрезанной таблицы на месте настоящей.
    Части не удаляются здесь — они нужны, пока манифест не
    записан, иначе прерывание отнимет и части, и результат.
    """

    target = out / TABLES[name][0]

    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_name(target.name + ".tmp")

    writer = None
    rows = 0

    for index in range(batches):

        part = out / PARTS_DIR / f"{name}-{index:05d}.parquet"

        # Каждая пачка пишет каждую таблицу, пусть и пустую, а
        # маркер появляется после всех частей. Части нет при
        # маркере — черновик испорчен, и молча собрать датасет
        # короче обещанного нельзя.
        if not part.exists():
            raise GenerationError(
                f"пачка {index}: маркер есть, а части {part.name} нет — "
                "черновик повреждён, прогон нужно начать заново"
            )

        table = pq.read_table(part)

        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")

        if table.num_rows:
            writer.write_table(table)
            rows += table.num_rows

    if writer is None:
        _write(temporary, [], TABLES[name][1])
    else:
        writer.close()

    os.replace(temporary, target)

    return rows


def _file_sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _run_card(
    settings,
    seed: int,
    world_seed,
    total_clients: int,
    chunk_clients: int,
    community_size: int,
) -> dict:
    """
    С чем начат прогон. Продолжать можно только его самого.
    """

    return {
        "generator_version": GENERATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "world_seed": world_seed,
        "total_clients": total_clients,
        "chunk_clients": chunk_clients,
        "community_size": community_size,
        "history_start": config.HISTORY_START.isoformat(),
        "history_end": config.HISTORY_END.isoformat(),
        "generation_config_sha256": settings.fingerprint(),
    }


# ============================================================
# ГЕНЕРАЦИЯ
# ============================================================


def generate_dataset(
    total_clients: int,
    out_dir: Path | None = None,
    seed: int = SEED,
    world_seed: int | None = None,
    history_start: datetime | None = None,
    history_end: datetime | None = None,
    workers: int = 1,
    chunk_clients: int = 256,
    params_path: str | None = None,
    catalog_scale: float | None = None,
    community_size: int | None = None,
    resume: bool = False,
    quiet: bool = False,
) -> dict:

    out = Path(out_dir) if out_dir is not None else RAW_DIR / f"clients_{total_clients}"

    if history_start is not None or history_end is not None:
        config.activate_horizon(
            history_start if history_start is not None else config.HISTORY_START,
            history_end if history_end is not None else config.HISTORY_END,
        )

    horizon = (config.HISTORY_START.isoformat(), config.HISTORY_END.isoformat())

    settings = _build_params(params_path, catalog_scale, community_size)

    params_module.activate(settings)
    rng_module.configure(seed, settings.fingerprint(), world_seed)

    if out.exists() and not resume:
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    size = settings.relationships.community_size

    community_count = communities.community_count(total_clients)

    per_batch = max(1, chunk_clients // size)

    batches = [
        tuple(range(start, min(start + per_batch, community_count)))
        for start in range(0, community_count, per_batch)
    ]

    jobs = [
        (index, community_ids, total_clients, str(out))
        for index, community_ids in enumerate(batches)
    ]

    card = _run_card(settings, seed, rng_module.current_world_seed(),
                     total_clients, chunk_clients, size)

    run_path = out / RUN_FILE

    if resume and run_path.exists():

        stored = json.loads(run_path.read_text(encoding="utf-8"))

        if stored != card:
            differing = sorted(
                key for key in set(stored) | set(card) if stored.get(key) != card.get(key)
            )
            raise GenerationError(
                "продолжение чужого прогона: не совпадает " + ", ".join(differing)
            )

    elif resume and (out / PARTS_DIR).exists():
        raise GenerationError(
            f"в {out} есть незавершённые части, но нет {RUN_FILE}: "
            "продолжать нечего, прогон не описан"
        )

    _write_json(run_path, card)

    if resume:
        # Готова та пачка, у которой есть маркер. Существование
        # part-файла ничего не значит: его мог оставить прогон,
        # прерванный на середине записи.
        jobs = [job for job in jobs if not _batch_marker(out, job[0]).exists()]

    done = 0

    def report(result) -> None:
        nonlocal done
        done += 1
        if not quiet:
            print(f"batches: {done}/{len(jobs)}")

    if workers <= 1 or len(jobs) <= 1:
        _worker_init(seed, params_path, catalog_scale, community_size, world_seed, horizon)
        for job in jobs:
            report(_run_batch(job))
    else:
        with Pool(
            processes=min(workers, len(jobs)),
            initializer=_worker_init,
            initargs=(seed, params_path, catalog_scale, community_size, world_seed, horizon),
        ) as pool:
            for result in pool.imap_unordered(_run_batch, jobs):
                report(result)

    # Итог собирается из маркеров ВСЕХ пачек, а не из того, что
    # посчитал текущий прогон: продолженная сборка обязана дать
    # тот же манифест, что и сборка без остановки.
    counts = {name: 0 for name in TABLES}

    for index in range(len(batches)):

        marker = _batch_marker(out, index)

        if not marker.exists():
            raise GenerationError(
                f"пачка {index} не завершена: без её маркера датасет собирать нельзя"
            )

        record = json.loads(marker.read_text(encoding="utf-8"))

        for name in counts:
            counts[name] += int(record["counts"][name])

    for name in TABLES:

        merged = _merge_parts(out, name, len(batches))

        # Склеенное обязано сойтись с обещанным маркерами: иначе
        # манифест назовёт строки, которых в файле нет.
        if merged != counts[name]:
            raise GenerationError(
                f"таблица {name}: склеено строк {merged}, а маркеры пачек обещают "
                f"{counts[name]} — черновик повреждён, прогон нужно начать заново"
            )

    # Технический паспорт выгрузки: версия контракта, окно,
    # число строк и sha256 двух основных файлов. Всё остальное —
    # схема событий, приоритеты типов, доступность источников —
    # статично и живёт в коде, а не в копии рядом с данными.
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timezone": config.TIMEZONE_NAME,
        "period_start": config.event_time_text(config.HISTORY_START),
        "period_end": config.event_time_text(config.HISTORY_END),
        "events_rows": counts["events"],
        "profile_rows": counts["profile"],
        "events_sha256": _file_sha256(out / TABLES["events"][0]),
        "profile_sha256": _file_sha256(out / TABLES["profile"][0]),
    }

    _write_json(out / "manifest.json", manifest)

    # Черновик убирается только теперь, когда результат записан.
    # Прерывание до этой строки оставляет части на месте, и
    # прогон продолжается, а не начинается заново.
    parts_dir = out / PARTS_DIR

    if parts_dir.exists():
        for leftover in sorted(parts_dir.iterdir()):
            leftover.unlink()
        parts_dir.rmdir()

    run_path.unlink(missing_ok=True)

    return counts


# ============================================================
# CLI
# ============================================================


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def generate_group(
    group: str,
    workers: int = 1,
    chunk_clients: int = 256,
    params_path: str | None = None,
    resume: bool = False,
    quiet: bool = False,
    root: Path | None = None,
) -> dict:
    """
    ОДНА группа в свой каталог data/01_raw/<группа>.

    Соседние группы не трогаются: запуск знает только про свой
    каталог. Мир у групп общий — один WORLD_SEED даёт одну
    географию, одни бренды и одни точки; клиентов и поведение
    делает seed группы, поэтому популяции не пересекаются.
    """

    if group not in config.DATASETS:
        known = ", ".join(sorted(config.DATASETS))
        raise GenerationError(f"группа {group!r} не объявлена в config.DATASETS: есть {known}")

    settings = config.DATASETS[group]

    out = (Path(root) if root is not None else RAW_DIR) / group

    if not quiet:
        print("=" * 60)
        print(f"ГРУППА {group}: {settings.clients} клиентов, "
              f"{settings.history_start.date()} … {settings.history_end.date()}, "
              f"seed {settings.seed}, world_seed {WORLD_SEED}")
        print("=" * 60)

    return generate_dataset(
        total_clients=settings.clients,
        out_dir=out,
        seed=settings.seed,
        world_seed=WORLD_SEED,
        history_start=settings.history_start,
        history_end=settings.history_end,
        workers=workers,
        chunk_clients=chunk_clients,
        params_path=params_path,
        resume=resume,
        quiet=quiet,
    )


def main() -> None:

    # Сообщения генератора на русском, а консоль Windows по
    # умолчанию живёт не в UTF-8. Без этой строки вывод
    # рассыпается в кракозябры, как и у остальных команд.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Генерация RAW: одна группа за запуск по config.DATASETS",
    )

    # Состав выгрузки — клиенты, горизонты, seed — живёт только
    # в config.DATASETS. Здесь остаётся имя группы и то, что
    # данных не меняет.
    parser.add_argument("group", choices=sorted(config.DATASETS),
                        help="какую группу генерировать; соседние не трогаются")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--chunk-clients", type=int, default=256)
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    counts = generate_group(
        args.group,
        workers=args.workers,
        chunk_clients=args.chunk_clients,
        params_path=args.params,
        resume=args.resume,
    )

    print()
    print("=" * 60)
    print(f"RAW GROUP GENERATED: {args.group}  ->  {RAW_DIR / args.group}")
    print("=" * 60)

    for table, value in counts.items():
        print(f"{table:24s}{value:,}")


if __name__ == "__main__":
    main()
