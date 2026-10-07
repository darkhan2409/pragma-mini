from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
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
from .truth import STATES_SCHEMA, TRANSITIONS_SCHEMA
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
# Рядом, в truth/, лежит скрытое состояние симуляции (truth.py):
# его читает только аудит, вход модели и метки — никогда.
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
    "transitions": ("truth/transitions.parquet", TRANSITIONS_SCHEMA),
    "states": ("truth/states.parquet", STATES_SCHEMA),
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


def _read_json(path: Path, what: str) -> dict:
    """
    JSON черновика. Битый файл — ошибка данных прогона, а не
    поломка генератора, поэтому наружу идёт GenerationError с
    именем файла, а не исключение из недр json.
    """

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"{what} {path.name} нечитаем: {error}") from error


def _job_marker(out: Path, batch: int, job: int) -> Path:
    return out / PARTS_DIR / f"job-{batch:05d}-{job:03d}.json"


def _part(out: Path, name: str, batch: int, job: int) -> Path:
    return out / PARTS_DIR / f"{name}-{batch:05d}-{job:03d}.parquet"


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
        config.activate_horizon(*(datetime.fromisoformat(moment) for moment in horizon))

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
    Задание воркера: сообщества одной пачки (сейчас одно), симуляция
    и запись своих part-файлов.

    Задание мельче пачки: пачка (chunk_clients) задаёт только
    раскладку итоговой таблицы — одну группу строк, — а считать её
    можно по частям. Иначе на 1000 клиентах было 4 задания, и из 12
    ядер работали 4.
    """

    batch_index, job_index, community_ids, total_clients, out_dir = job

    from .engine import run_community

    out = Path(out_dir)

    parts = {name: _part(out, name, batch_index, job_index) for name in TABLES}

    parts["events"].parent.mkdir(parents=True, exist_ok=True)

    # Строки пишутся по сообществу, как только оно досчитано: в
    # памяти воркера живут строки одного сообщества, а не всей
    # пачки. Пачка в сотни тысяч строк Python-словарями плюс её
    # копия в Arrow — это и было «почти вся память WSL» при многих
    # воркерах. Раскладка итоговой таблицы от этого не меняется:
    # склейка сводит часть в одну группу строк (_merge_parts).
    writers = {
        name: pq.ParquetWriter(parts[name], schema, compression="zstd")
        for name, (_, schema) in TABLES.items()
    }

    counts: dict[str, int] = {name: 0 for name in TABLES}

    try:
        for community_id in community_ids:

            members = communities.members(community_id, total_clients)

            if not members:
                continue

            result = run_community(community_id, members)

            for name, rows in (
                ("events", result.events),
                ("profile", result.profile_rows),
                ("transitions", result.transitions),
                ("states", result.states),
            ):
                if rows:
                    writers[name].write_table(pa.Table.from_pylist(rows, schema=TABLES[name][1]))
                    counts[name] += len(rows)

            del result

    finally:
        for writer in writers.values():
            writer.close()

    # Число строк не описывает содержимое: подменённая часть
    # прежней длины прошла бы в итог незамеченной. Маркер
    # подписывает сам файл.
    digests = {name: _file_sha256(path) for name, path in parts.items()}

    # Маркер пишется ПОСЛЕДНИМ и целиком: пачка готова только
    # тогда, когда все её part-файлы на месте. Итоговые суммы
    # собираются из маркеров, поэтому продолженный прогон знает
    # и про те пачки, которых сам не считал.
    _write_json(
        _job_marker(out, batch_index, job_index),
        {
            "batch": batch_index,
            "job": job_index,
            "communities": list(community_ids),
            "counts": counts,
            "sha256": digests,
        },
    )

    return batch_index, job_index, counts


# ============================================================
# СКЛЕЙКА
# ============================================================


def _merge_parts(out: Path, name: str, jobs: list[int], digests: dict[tuple, str]) -> int:
    """
    Склейка part-файлов в итоговую таблицу: jobs[b] — сколько заданий
    у пачки b. Части пачки идут подряд в одну группу строк, как если
    бы пачку считало одно задание.

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

    for index, count in enumerate(jobs):

        pieces = []

        for job in range(count):

            part = _part(out, name, index, job)

            # Каждое задание пишет каждую таблицу, пусть и пустую, а
            # маркер появляется после всех частей. Части нет при
            # маркере — черновик испорчен, и молча собрать датасет
            # короче обещанного нельзя.
            if not part.exists():
                raise GenerationError(
                    f"пачка {index}, задание {job}: маркер есть, а части {part.name} нет — "
                    "черновик повреждён, прогон нужно начать заново"
                )

            # Число строк содержимого не описывает: часть той же
            # длины с другими значениями прошла бы незамеченной.
            # Сверяется подпись, которую маркер поставил при записи.
            actual = _file_sha256(part)

            if actual != digests[index, job]:
                raise GenerationError(
                    f"пачка {index}, задание {job}: содержимое {part.name} не совпадает "
                    "с подписью маркера — черновик повреждён, прогон нужно начать заново"
                )

            pieces.append(pq.read_table(part))

        # Одна группа строк на пачку: части пишутся по сообществам и
        # заданиям, а итоговая раскладка — и с ней sha256 выгрузки — от
        # этого зависеть не должна.
        table = pa.concat_tables(pieces).combine_chunks()

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
        "registration_end": config.REGISTRATION_END.isoformat(),
        "generation_config_sha256": settings.fingerprint(),
        # Черновик из заданий по сообществу: прежний, по пачкам,
        # этой версией не продолжается.
        "job": "community",
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
    registration_end: datetime | None = None,
) -> dict:

    out = Path(out_dir) if out_dir is not None else RAW_DIR / f"clients_{total_clients}"

    if history_start is not None or history_end is not None or registration_end is not None:
        config.activate_horizon(
            history_start if history_start is not None else config.HISTORY_START,
            history_end if history_end is not None else config.HISTORY_END,
            registration_end,
        )

    horizon = (
        config.HISTORY_START.isoformat(),
        config.HISTORY_END.isoformat(),
        config.REGISTRATION_END.isoformat(),
    )

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

    # Задание — одно сообщество; пачка остаётся единицей раскладки.
    jobs = [
        (index, job, (community_id,), total_clients, str(out))
        for index, community_ids in enumerate(batches)
        for job, community_id in enumerate(community_ids)
    ]

    card = _run_card(settings, seed, rng_module.current_world_seed(),
                     total_clients, chunk_clients, size)

    run_path = out / RUN_FILE

    if resume and run_path.exists():

        stored = _read_json(run_path, "карточка прогона")

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
        # Готово то задание, у которого есть маркер. Существование
        # part-файла ничего не значит: его мог оставить прогон,
        # прерванный на середине записи.
        jobs = [job for job in jobs if not _job_marker(out, job[0], job[1]).exists()]

    done = 0

    def report(result) -> None:
        nonlocal done
        done += 1
        if not quiet:
            print(f"jobs: {done}/{len(jobs)}")

    if workers <= 1 or len(jobs) <= 1:
        _worker_init(seed, params_path, catalog_scale, community_size, world_seed, horizon)
        for job in jobs:
            report(_run_batch(job))
    else:
        # ProcessPoolExecutor, а не multiprocessing.Pool: у Pool
        # воркер, убитый нехваткой памяти, заменяется новым, а его
        # пачка не завершается никогда — прогон висит без вывода
        # (CPython gh-66587). Executor сообщает о гибели воркера.
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            initializer=_worker_init,
            initargs=(seed, params_path, catalog_scale, community_size, world_seed, horizon),
        ) as pool:
            try:
                for future in as_completed([pool.submit(_run_batch, job) for job in jobs]):
                    report(future.result())
            except BrokenProcessPool as error:
                raise GenerationError(
                    "воркер генератора погиб, скорее всего от нехватки памяти. Готовые пачки "
                    "сохранены маркерами: продолжите с --resume и меньшим --workers"
                ) from error

    # Итог собирается из маркеров ВСЕХ заданий, а не из того, что
    # посчитал текущий прогон: продолженная сборка обязана дать
    # тот же манифест, что и сборка без остановки.
    counts = {name: 0 for name in TABLES}

    # Подписи частей, обещанные маркерами: по ним сверяется
    # содержимое каждой готовой части перед склейкой.
    digests: dict[str, dict[tuple, str]] = {name: {} for name in TABLES}

    for index, community_ids in enumerate(batches):

        for job in range(len(community_ids)):

            marker = _job_marker(out, index, job)

            if not marker.exists():
                raise GenerationError(
                    f"пачка {index}, задание {job} не завершено: без его маркера "
                    "датасет собирать нельзя"
                )

            record = _read_json(marker, "маркер задания")

            promised = record.get("sha256")

            if not isinstance(promised, dict):
                raise GenerationError(
                    f"маркер задания {index}-{job} без подписей частей: черновик собран "
                    "прежней версией генератора, продолжить его нельзя"
                )

            for name in counts:
                counts[name] += int(record["counts"][name])
                digests[name][index, job] = str(promised[name])

    for name in TABLES:

        merged = _merge_parts(out, name, [len(item) for item in batches], digests[name])

        # Склеенное обязано сойтись с обещанным маркерами: иначе
        # манифест назовёт строки, которых в файле нет.
        if merged != counts[name]:
            raise GenerationError(
                f"таблица {name}: склеено строк {merged}, а маркеры пачек обещают "
                f"{counts[name]} — черновик повреждён, прогон нужно начать заново"
            )

    # Технический паспорт выгрузки: версия контракта, окно, число
    # строк, sha256 двух основных файлов и происхождение.
    #
    # Происхождение записано здесь, а не только в карточке
    # прогона: карточка удаляется при успехе, и готовая выгрузка
    # оставалась без единого следа того, каким seed и какими
    # параметрами получена. Схема событий, приоритеты типов и
    # доступность источников по-прежнему живут в коде, а не в
    # копии рядом с данными.
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "timezone": config.TIMEZONE_NAME,
        "period_start": config.event_time_text(config.HISTORY_START),
        "period_end": config.event_time_text(config.HISTORY_END),
        "registration_end": config.event_time_text(config.REGISTRATION_END),
        "seed": seed,
        "world_seed": rng_module.current_world_seed(),
        "total_clients": total_clients,
        "community_size": size,
        # Размер чанка данные не меняет, но меняет раскладку
        # групп строк в parquet, а значит и sha256 файла.
        "chunk_clients": chunk_clients,
        "generation_config_sha256": settings.fingerprint(),
        "reference_sha256": {
            "merchants": _file_sha256(config.MERCHANT_REFERENCE_PATH),
            "national_merchants": _file_sha256(config.NATIONAL_MERCHANTS_PATH),
            "product_timeline": _file_sha256(config.PRODUCT_TIMELINE_PATH),
        },
        **{f"{name}_rows": counts[name] for name in TABLES},
        **{f"{name}_sha256": _file_sha256(out / TABLES[name][0]) for name in TABLES},
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


# Сколько памяти держит один воркер на пике: состояние сообщества и
# его строки до записи. Замер (VmHWM) — пачка из 256 клиентов на
# самом длинном горизонте, test (31 месяц): 0.35–0.39 ГиБ. С запасом.
WORKER_MEMORY = 512 << 20

# Сколько памяти оставить остальному: главный процесс, склейка
# частей, система.
MEMORY_RESERVE = 1 << 30


def available_memory() -> int | None:
    """
    MemAvailable из /proc/meminfo в байтах; None, если его нет.
    """

    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None

    return None


def default_workers() -> int:
    """
    Воркеров по умолчанию: ядра минус одно, но не больше, чем
    помещается в свободную память. Одиннадцать воркеров на 7 ГБ
    WSL вытесняли систему в своп.
    """

    cpus = max(1, (os.cpu_count() or 2) - 1)

    available = available_memory()

    if available is None:
        return cpus

    return max(1, min(cpus, (available - MEMORY_RESERVE) // WORKER_MEMORY))


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
        registration_end=settings.registration_end,
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
    parser.add_argument(
        "--workers", type=int, default=None,
        help="процессов; по умолчанию — по ядрам и свободной памяти",
    )
    parser.add_argument("--chunk-clients", type=int, default=256)
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.workers is None:
        args.workers = default_workers()
        available = available_memory()
        print(
            f"[emit] воркеров {args.workers}: ядер {os.cpu_count()}, свободно "
            f"{available / 2**30 if available else float('nan'):.1f} ГиБ, на воркер "
            f"{WORKER_MEMORY / 2**30:.2f} ГиБ"
        )

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
