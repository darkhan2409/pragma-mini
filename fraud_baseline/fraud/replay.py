from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, GROUPS, RAW_DIR, REPO, REPORTS_DIR, manifest
from .labels import attach_rows


# ============================================================
# TARGET ИЗ ПОВТОРНОГО ПРОГОНА ГЕНЕРАТОРА
# ============================================================
#
# Явной метки мошенничества в выгрузке нет: состояние эпизода живёт
# только в памяти генератора, а на ленте мошенническая операция — обычная
# purchase или transfer_out. Поэтому группа прогоняется генератором ещё
# раз с тем же seed, окном и параметрами, и по дороге записывается,
# какие строки ленты выдал шаг мошенничества.
#
# Файлы генератора не меняются. В каждом процессе подменяются три имени:
#
#   engine_products._fraud_purchase,
#   engine_products._fraud_transfer   обёртка помечает событие, которое
#                                     вернул шаг, ключом (клиент, номер
#                                     выдачи) с видом эпизода и шага;
#   engine_month._row                 обёртка видит каждую строку выгрузки:
#                                     считает хеш ленты клиента и снимает
#                                     помеченные строки — ровно те строки,
#                                     что легли в файл.
#
# Повтору верят, только если хеш ленты КАЖДОГО клиента совпал с хешем его
# строк в RAW. Иначе меток нет.
#
# Метка — только target. В признаки из повтора не идёт ничего.
# ============================================================


if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


FALSE_POSITIVE = "false_positive"

# Состояние процесса-исполнителя.
_WORKER: dict = {}


def row_digest_bytes(client_id: str, event_time: str, source: str, payload: str) -> bytes:
    """
    Строка выгрузки в байтах хеша ленты: поля через \\x1f, конец строки \\x1e.
    Так же её собирает labels.raw_digests из файла RAW.
    """
    return f"{client_id}\x1f{event_time}\x1f{source}\x1f{payload}\x1e".encode()


def _install(seed: int, world_seed: int, horizon: tuple[str, str]) -> None:
    # Ошибка в инициализаторе заставила бы пул бесконечно перезапускать
    # процессы. Поэтому она запоминается и всплывает в первой же задаче.
    try:
        _patch(seed, world_seed, horizon)
    except Exception as error:  # noqa: BLE001
        _WORKER["error"] = f"{type(error).__name__}: {error}"


def _patch(seed: int, world_seed: int, horizon: tuple[str, str]) -> None:
    # engine первым: он сам подтягивает engine_month и engine_products,
    # прямой импорт engine_month замкнулся бы по кругу.
    from src.generator import emit, engine
    from src.generator import engine_month, engine_products

    emit._worker_init(seed, None, None, None, world_seed, horizon)

    marks: dict[tuple[str, int], tuple[str, str]] = {}
    captured: list[dict] = []
    digests: dict[str, "hashlib._Hash"] = {}

    def marking(original):
        def step(*args):
            event, subject = original(*args)
            # Сигнатура обеих функций кончается на (episode, step, rng).
            episode, fraud_step = args[-3], args[-2]
            if event is not None:
                marks[(event.client_id, event.ordinal)] = (episode.kind, fraud_step.kind)
            return event, subject

        return step

    original_row = engine_month._row

    def row(event) -> dict:
        out = original_row(event)
        digest = digests.get(out["client_id"])
        if digest is None:
            digest = digests[out["client_id"]] = hashlib.sha256()
        digest.update(row_digest_bytes(out["client_id"], out["event_time"], out["source"], out["payload"]))
        mark = marks.get((event.client_id, event.ordinal))
        if mark is not None:
            captured.append({**out, "episode_kind": mark[0], "step_kind": mark[1]})
        return out

    engine_products._fraud_purchase = marking(engine_products._fraud_purchase)
    engine_products._fraud_transfer = marking(engine_products._fraud_transfer)
    engine_month._row = row

    _WORKER.update(marks=marks, captured=captured, digests=digests)


def _run(job: tuple[tuple[int, ...], int]) -> tuple[list[dict], dict[str, str]]:
    from src.generator.engine import run_community
    from src.generator.world import communities

    if "error" in _WORKER:
        raise RuntimeError(f"процесс повтора не подготовлен: {_WORKER['error']}")
    community_ids, total_clients = job
    for community_id in community_ids:
        members = communities.members(community_id, total_clients)
        if members:
            run_community(community_id, members)

    captured = list(_WORKER["captured"])
    digests = {client: digest.hexdigest() for client, digest in _WORKER["digests"].items()}
    _WORKER["captured"].clear()
    _WORKER["digests"].clear()
    _WORKER["marks"].clear()
    return captured, digests


def replay(group: str, workers: int, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Помеченные строки группы и хеши лент всех клиентов повтора.
    """
    from src.generator import config as generator_config
    from src.generator import emit
    from src.generator.world import communities

    settings = generator_config.DATASETS[group]
    card = manifest(group, raw_dir)
    fingerprint = emit._build_params(None, None, None).fingerprint()
    expected = {
        "generator_version": generator_config.GENERATOR_VERSION,
        "generation_config_sha256": fingerprint,
        "seed": settings.seed,
        "world_seed": generator_config.WORLD_SEED,
        "total_clients": settings.clients,
    }
    differing = {key: (card.get(key), value) for key, value in expected.items() if card.get(key) != value}
    if differing:
        raise RuntimeError(f"генератор не тот, что сделал выгрузку {group}: {differing}")

    generator_config.activate_horizon(settings.history_start, settings.history_end)
    horizon = (generator_config.HISTORY_START.isoformat(), generator_config.HISTORY_END.isoformat())
    params = emit._build_params(None, None, None)
    per_batch = max(1, card["chunk_clients"] // params.relationships.community_size)
    count = communities.community_count(settings.clients)
    jobs = [(tuple(range(start, min(start + per_batch, count))), settings.clients) for start in range(0, count, per_batch)]

    captured: list[dict] = []
    digests: dict[str, str] = {}
    initargs = (settings.seed, generator_config.WORLD_SEED, horizon)
    with Pool(processes=max(1, min(workers, len(jobs))), initializer=_install, initargs=initargs) as pool:
        for done, (rows, hashes) in enumerate(pool.imap_unordered(_run, jobs), start=1):
            captured.extend(rows)
            if set(hashes) & set(digests):
                raise RuntimeError("клиент встретился в двух пачках повтора")
            digests.update(hashes)
            if done % 5 == 0 or done == len(jobs):
                print(f"replay {group}: {done}/{len(jobs)} пачек", flush=True)

    columns = ["client_id", "event_time", "source", "payload", "episode_kind", "step_kind"]
    return pd.DataFrame(captured, columns=columns), digests


def run(group: str, workers: int, raw_dir: Path = RAW_DIR, data_dir: Path = DATA_DIR, reports_dir: Path = REPORTS_DIR) -> dict:
    started = time.time()
    captured, digests = replay(group, workers, raw_dir)
    labels, check = attach_rows(raw_dir / group / "events.parquet", captured, digests)
    labels["fraud"] = (labels["episode_kind"] != FALSE_POSITIVE).astype("int8")

    out = data_dir / group
    out.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out / "labels.parquet", index=False)

    check.update(
        {
            "group": group,
            "steps": int(len(labels)),
            "fraud_1": int(labels["fraud"].sum()),
            "false_positive": int((labels["fraud"] == 0).sum()),
            "by_episode_kind": labels.groupby("episode_kind").size().to_dict(),
            "by_step_kind": labels.groupby("step_kind").size().to_dict(),
            "by_type": labels.groupby("type").size().to_dict(),
            "seconds": round(time.time() - started, 1),
        }
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "replay_check.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report[group] = check
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return check


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Метки fraud одной группы повторным прогоном генератора")
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.group, args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
