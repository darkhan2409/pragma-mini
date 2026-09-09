from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.artifacts import sha256_file, write_json
from src.preprocessing.config import artifacts_dir as prep_artifacts_dir
from src.preprocessing.config import processed_dir as prep_processed_dir
from src.tokenizer.build import check_order, client_runs, iter_client_blocks
from src.tokenizer.config import DATASET_MANIFEST_FILE, tokenized_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# session_id не переживает токенизацию: это metadata, и словарь
# её не знает. Но принадлежность экрана к сессии нужна модели
# как СТРУКТУРА, а не как признак.
#
# Поэтому рядом с токенизированным набором лежит sidecar:
# (client_id, seq, session_key). Ключ ничего не рассказывает
# о содержании события и в модель как значение не попадает.
#
# Ключ строится плотной нумерацией внутри клиента, а не хэшем
# строки. Разные session_id не могут слиться не потому, что
# хэш хороший, а потому что отображение инъективно.
#
# Сверка при загрузке идёт по (client_id, seq) поэлементно.
# Совпадения числа строк недостаточно: перепутанные местами
# клиенты дали бы ту же длину и другую историю.
#
# session_id есть у экранов всегда, а у операций и баннеров
# начиная с ревизии схемы RAW 2. Каждый тип хранит его в своей
# колонке, и строка берёт значение только из СВОЕЙ: значение
# в чужой колонке значило бы, что processed собран неверно.
#
# Сессия без экранов это не дефект. Так выглядят два честных
# случая: период, когда экраны ещё не сохранялись, и сессия,
# в которой вход не состоялся и до экрана дело не дошло.
# ============================================================


NO_SESSION = -1

APP_SCREEN = "app_screen"

# Типы событий приложения, у которых бывает session_id.
# Порядок фиксирован: он попадает в манифест.
APP_TYPES: tuple[str, ...] = ("app_screen", "app_operation", "banner")

SESSION_ID_COLUMNS: dict[str, str] = {
    namespace: f"{namespace}__session_id" for namespace in APP_TYPES
}

# Имя оставлено для читателей прежнего манифеста и отчётов.
SESSION_ID_COLUMN = SESSION_ID_COLUMNS[APP_SCREEN]

GROUPS: tuple[str, ...] = ("train", "val", "test")

SESSIONS_DIR = "sessions"
SESSIONS_MANIFEST = "sessions_manifest.json"

KEY_RULE = "dense_per_client"

SCHEMA = pa.schema(
    [
        pa.field("client_id", pa.int64()),
        pa.field("seq", pa.int64()),
        pa.field("session_key", pa.int64()),
    ]
)

BASE_COLUMNS = ("client_id", "seq", "ts", "event_type")

# Имя оставлено ради прежних читателей.
COLUMNS = (*BASE_COLUMNS, SESSION_ID_COLUMN)


class SessionSidecarError(RuntimeError):
    """Sidecar не подходит к этому набору данных."""


# ============================================================
# ПУТИ
# ============================================================


def sessions_path(root: Path, group: str) -> Path:
    return Path(root) / SESSIONS_DIR / f"{group}_clients.parquet"


def manifest_path(root: Path) -> Path:
    return Path(root) / SESSIONS_DIR / SESSIONS_MANIFEST


def events_path(processed_dir: Path, group: str) -> Path:
    return Path(processed_dir) / "clients" / f"{group}_clients" / "events.parquet"


# ============================================================
# КЛЮЧИ
# ============================================================


def session_key_of(values) -> np.ndarray:
    """
    Плотная нумерация session_id внутри одного клиента.

    None и пустая строка дают -1. Инъективность отображения
    проверяется здесь же: разные строки обязаны получить
    разные номера.
    """

    raw = [value if isinstance(value, str) else "" for value in values]

    present = np.array([bool(value) for value in raw], dtype=bool)

    keys = np.full(len(raw), NO_SESSION, dtype=np.int64)

    if not present.any():
        return keys

    named = np.asarray(raw, dtype=object)[present]

    unique, inverse = np.unique(named.astype(str), return_inverse=True)

    keys[present] = np.asarray(inverse, dtype=np.int64).ravel()

    distinct = len({value for value in named})

    if int(unique.size) != distinct:
        raise SessionSidecarError(
            f"разные session_id получили один ключ: строк {distinct}, ключей {unique.size}"
        )

    return keys


def fingerprint(client_id: np.ndarray, seq: np.ndarray, session_key: np.ndarray) -> str:
    """
    Отпечаток содержимого sidecar.
    """

    digest = hashlib.sha256()

    for array in (client_id, seq, session_key):
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.int64)).tobytes())

    return digest.hexdigest()


# ============================================================
# СБОРКА
# ============================================================


def is_named(value) -> bool:
    """
    Настоящий ли это session_id.

    Проверяется тип, а не истинность: pandas отдаёт пропуск
    строковой колонки как float nan, а bool(nan) это True.
    Через pyarrow сюда приходит None, но правило должно быть
    верным независимо от того, кто читал файл.
    """

    return isinstance(value, str) and value != ""


def available_columns(source: Path) -> dict[str, str]:
    """
    Колонки session_id, которые есть в этом processed-наборе.

    Наборы, собранные до ревизии 2, знают только колонку
    экранов. Читать несуществующую колонку нельзя, а требовать
    пересборку старого набора незачем: у него этих событий
    в сессиях и не было.
    """

    names = set(pq.read_schema(source).names)

    columns = {
        namespace: column
        for namespace, column in SESSION_ID_COLUMNS.items()
        if column in names
    }

    if APP_SCREEN not in columns:
        raise SessionSidecarError(
            f"{source}: нет колонки {SESSION_ID_COLUMNS[APP_SCREEN]}"
        )

    return columns


def session_ids_of_block(block, columns: dict[str, str], event_type: np.ndarray) -> np.ndarray:
    """
    session_id каждой строки из колонки её собственного типа.

    Заодно проверяется, что непустого значения нет в чужой
    колонке: preprocessing раскладывает поля по namespace, и
    нарушение значило бы, что раскладка сломана.
    """

    combined = np.full(event_type.size, None, dtype=object)

    for namespace, column in columns.items():

        values = np.asarray(block.column(column).to_pylist(), dtype=object)

        present = np.array([is_named(value) for value in values], dtype=bool)

        mine = event_type == namespace

        stray = present & ~mine

        if bool(stray.any()):
            raise SessionSidecarError(
                f"колонка {column} заполнена у события типа "
                f"{event_type[stray][0]}, а должна только у {namespace}"
            )

        take = present & mine

        combined[take] = values[take]

    return combined


def build_group(processed_dir: Path, group: str) -> dict:
    """
    Ключи сессий одной группы клиентов.
    """

    source = events_path(processed_dir, group)

    if not source.exists():
        raise SessionSidecarError(f"нет файла событий {source}")

    columns = available_columns(source)

    client_ids: list[np.ndarray] = []
    seqs: list[np.ndarray] = []
    keys: list[np.ndarray] = []

    n_named = 0
    n_screens = 0
    n_sessions = 0
    n_without_screens = 0

    named_by_type = {namespace: 0 for namespace in APP_TYPES}

    read = [*BASE_COLUMNS, *columns.values()]

    for block in iter_client_blocks(source, columns=read):

        column = block.column("client_id").to_numpy()

        if column.size == 0:
            continue

        seq = block.column("seq").to_numpy().astype(np.int64)
        ts = block.column("ts").to_numpy().astype("datetime64[us]")

        check_order(column, ts, seq)

        event_type = np.asarray(block.column("event_type").to_pylist(), dtype=object)

        session_id = session_ids_of_block(block, columns, event_type)

        for value, lo, hi in client_runs(column):

            local_seq = seq[lo:hi]
            local_type = event_type[lo:hi]

            if not np.array_equal(local_seq, np.arange(local_seq.size, dtype=np.int64)):
                raise SessionSidecarError(
                    f"клиент {value}: seq не плотный от нуля, префикс примера "
                    "нельзя резать по позиции"
                )

            local_key = session_key_of(session_id[lo:hi])

            named = local_key >= 0

            wrong = named & ~np.isin(local_type, APP_TYPES)

            if bool(wrong.any()):
                raise SessionSidecarError(
                    f"клиент {value}: session_id встретился у события типа "
                    f"{local_type[wrong][0]}, а не только у {APP_TYPES}"
                )

            n_named += int(named.sum())
            n_screens += int((local_type == APP_SCREEN).sum())

            for namespace in APP_TYPES:
                named_by_type[namespace] += int((named & (local_type == namespace)).sum())

            if named.any():

                unique, inverse = np.unique(local_key[named], return_inverse=True)

                screens_of_key = np.bincount(
                    np.asarray(inverse).ravel(),
                    weights=(local_type[named] == APP_SCREEN).astype(np.float64),
                    minlength=unique.size,
                )

                n_sessions += int(unique.size)
                n_without_screens += int((screens_of_key == 0).sum())

            client_ids.append(np.full(local_seq.size, int(value), dtype=np.int64))
            seqs.append(local_seq)
            keys.append(local_key)

    if not client_ids:
        raise SessionSidecarError(f"{source}: событий нет")

    client_id = np.concatenate(client_ids)
    seq = np.concatenate(seqs)
    session_key = np.concatenate(keys)

    return {
        "client_id": client_id,
        "seq": seq,
        "session_key": session_key,
        "summary": {
            "rows": int(client_id.size),
            "clients": int(np.unique(client_id).size),
            "screens": n_screens,
            "named": n_named,
            "named_by_type": named_by_type,
            "sessions": n_sessions,
            "sessions_without_screens": n_without_screens,
            "session_id_columns": sorted(columns.values()),
            "source_sha256": sha256_file(source),
            "fingerprint": fingerprint(client_id, seq, session_key),
        },
    }


def build_sidecar(
    processed_dir: Path,
    tokenized_root: Path,
    groups: tuple[str, ...] = GROUPS,
    quiet: bool = True,
) -> dict:
    """
    Собирает sidecar рядом с токенизированным набором.
    """

    processed_dir = Path(processed_dir)
    tokenized_root = Path(tokenized_root)

    manifest_file = tokenized_root / DATASET_MANIFEST_FILE

    if not manifest_file.exists():
        raise SessionSidecarError(f"нет манифеста набора {manifest_file}")

    report: dict = {
        "key_rule": KEY_RULE,
        "no_session": NO_SESSION,
        "session_id_columns": sorted(SESSION_ID_COLUMNS.values()),
        "tokenized_manifest_sha256": sha256_file(manifest_file),
        "processed": str(processed_dir),
        "groups": {},
    }

    (tokenized_root / SESSIONS_DIR).mkdir(parents=True, exist_ok=True)

    for group in groups:

        built = build_group(processed_dir, group)

        table = pa.Table.from_pydict(
            {
                "client_id": built["client_id"],
                "seq": built["seq"],
                "session_key": built["session_key"],
            },
            schema=SCHEMA,
        )

        pq.write_table(table, sessions_path(tokenized_root, group), compression="zstd")

        report["groups"][group] = built["summary"]

        if not quiet:
            summary = built["summary"]
            print(
                f"{group:<6s} строк {summary['rows']:>10,}  экранов {summary['screens']:>9,}"
                f"  сессий {summary['sessions']:>8,}".replace(",", " ")
            )

    write_json(manifest_path(tokenized_root), report)

    return report


# ============================================================
# ЗАГРУЗКА
# ============================================================


def load_session_keys(
    tokenized_root: Path,
    group: str,
    wanted: set[int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Ключи сессий по клиентам с проверкой происхождения.

    Проверяется отпечаток содержимого, привязка к этому же
    токенизированному набору и уникальность пар (client_id, seq).
    Совпадения количества строк недостаточно.
    """

    tokenized_root = Path(tokenized_root)

    path = sessions_path(tokenized_root, group)

    if not path.exists():
        raise SessionSidecarError(
            f"нет sidecar {path}: соберите его командой "
            "python -m src.model.sessions build --name <name>"
        )

    manifest = _read_manifest(tokenized_root)

    stored = manifest["groups"].get(group)

    if stored is None:
        raise SessionSidecarError(f"в манифесте sidecar нет группы {group}")

    table = pq.read_table(path, schema=SCHEMA)

    client_id = table.column("client_id").to_numpy().astype(np.int64)
    seq = table.column("seq").to_numpy().astype(np.int64)
    session_key = table.column("session_key").to_numpy().astype(np.int64)

    if fingerprint(client_id, seq, session_key) != stored["fingerprint"]:
        raise SessionSidecarError(
            f"{path}: содержимое не совпадает с отпечатком в манифесте"
        )

    pairs = client_id.astype(np.int64) * (int(seq.max()) + 2 if seq.size else 1) + seq

    if np.unique(pairs).size != pairs.size:
        raise SessionSidecarError(f"{path}: пара (client_id, seq) встречается дважды")

    found: dict[int, np.ndarray] = {}

    for value, lo, hi in client_runs(client_id):

        if wanted is not None and int(value) not in wanted:
            continue

        found[int(value)] = session_key[lo:hi]

        if not np.array_equal(seq[lo:hi], np.arange(hi - lo, dtype=np.int64)):
            raise SessionSidecarError(f"клиент {value}: seq в sidecar не плотный от нуля")

    if wanted is not None:

        missing = sorted(wanted - set(found))

        if missing:
            raise SessionSidecarError(f"{path}: нет клиентов {missing[:5]}")

    return found


def _read_manifest(tokenized_root: Path) -> dict:

    import json

    path = manifest_path(tokenized_root)

    if not path.exists():
        raise SessionSidecarError(f"нет манифеста sidecar {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))

    dataset_manifest = Path(tokenized_root) / DATASET_MANIFEST_FILE

    if not dataset_manifest.exists():
        raise SessionSidecarError(f"нет манифеста набора {dataset_manifest}")

    if manifest.get("tokenized_manifest_sha256") != sha256_file(dataset_manifest):
        raise SessionSidecarError(
            "sidecar собран для другого токенизированного набора: "
            "хэш tokenized_manifest.json не совпадает"
        )

    return manifest


def sidecar_digest(tokenized_root: Path) -> str | None:
    """
    Отпечаток sidecar для artifact_hashes. None, если его нет.
    """

    path = manifest_path(Path(tokenized_root))

    return sha256_file(path) if path.exists() else None


# ============================================================
# CLI
# ============================================================


def main() -> None:

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sidecar сессий приложения")

    parser.add_argument("command", choices=("build",))
    parser.add_argument("--name", default="smoke")
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)

    args = parser.parse_args()

    processed = args.processed or prep_processed_dir(args.name)
    root = args.root or tokenized_dir(args.name)

    report = build_sidecar(processed, root, quiet=False)

    print()
    print(f"записано: {manifest_path(root).parent}")
    print(f"правило ключа: {report['key_rule']}")


if __name__ == "__main__":
    main()


__all__ = [
    "GROUPS",
    "NO_SESSION",
    "SessionSidecarError",
    "build_sidecar",
    "load_session_keys",
    "session_key_of",
    "sessions_path",
    "sidecar_digest",
]
