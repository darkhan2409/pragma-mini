from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as pc

from ..artifacts import TableWriter
from ..rawdata import RawDataset, check_raw
from ..settings import PreprocessingConfig
from .events import build_batch, canonical_schema, iter_client_batches
from .schema import SCHEMA_VERSION, payload_columns


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап 1: выгрузка одной группы -> очищенная группа.
#
# Результат этапа — РОВНО ОДИН файл:
#
#   data/02_preprocessed/<group>/events.parquet
#
# Анкета клиента не копируется: чистить в ней нечего, и
# следующие этапы читают её прямо из выгрузки
# data/01_raw/<group>/profile.parquet. Двух почти одинаковых
# таблиц с одним смыслом быть не должно.
#
# Ни индекса клиентов, ни упоминаний сущностей, ни таблицы
# переводов, ни журнала отказов, ни реестра полей, ни отчётов
# рядом нет. Всё, что раньше лежало в спутниках, следующие этапы
# считают по самой ленте: история собирается группировкой по
# client_id, а переводы, договоры и карты читаются прямо из
# полей событий. Схему даёт код.
#
# Порядок этапа:
#
#   1. проверить пригодность RAW;
#   2. прочитать ленту и профиль;
#   3. раскрыть payload вместе с ключом type;
#   4. привести значения к объявленным типам;
#   5. пометить события-источники вех анкеты по их source_id;
#   6. упорядочить события клиента по времени, а при равном
#      времени — по причинному приоритету типа события;
#   7. записать ленту.
#
# Любая строка, которую нельзя разобрать по контракту,
# ОСТАНАВЛИВАЕТ этап: журнала отказов больше нет, а частичный
# слой на диске не остаётся.
#
# Лента читается пачками целых клиентов и пишется по одному row
# group на пачку, поэтому память ограничена пачкой, а не файлом.
# ============================================================


STAGE = "preprocess"

EVENTS_FILE = "events.parquet"

# Границы окна выгрузки едут метаданными самой ленты: отдельного
# файла-паспорта у слоя нет, а следующим этапам нужно знать, чем
# ограничена выгрузка.
PERIOD_START_KEY = b"period_start"
PERIOD_END_KEY = b"period_end"


@dataclass
class CanonicalResult:
    """
    Что получилось: один файл и числа для терминала.
    """

    outputs: list[Path]
    events_rows: int
    clients: int


def build_group(
    raw_dir: Path,
    out_dir: Path,
    config: PreprocessingConfig,
    group: str | None,
) -> CanonicalResult:

    # Пригодность входа проверяется ДО любой записи: непригодная
    # выгрузка не должна оставить ни куска слоя.
    raw = check_raw(raw_dir)

    manifest = raw.manifest

    out_dir = Path(out_dir)

    schema = canonical_schema(manifest)
    payload_names = [name for name, _ in payload_columns(manifest)]

    _clear(out_dir)

    metadata = {
        PERIOD_START_KEY: manifest.period_start.isoformat().encode("utf-8"),
        PERIOD_END_KEY: manifest.period_end.isoformat().encode("utf-8"),
    }

    events_writer = TableWriter(out_dir / EVENTS_FILE, schema.with_metadata(metadata))

    # Вехи анкеты нужны пометке их событий-источников. Строка на
    # клиента — это малая таблица рядом с лентой.
    profile = raw.read("profile", ["client_id", "lifelong"])

    milestones = dict(
        zip(profile.column("client_id").to_pylist(), profile.column("lifelong").to_pylist())
    )

    for batch in iter_client_batches(raw, config.batch_clients):

        result = build_batch(raw, config, batch, payload_names, schema, milestones)

        events_writer.write(result.table)

    events_rows = events_writer.close()

    return CanonicalResult(
        outputs=[out_dir / EVENTS_FILE],
        events_rows=events_rows,
        clients=len(_clients(raw)),
    )


def _clients(raw: RawDataset) -> set[str]:
    """
    Клиенты выгрузки: и те, у кого есть события, и те, у кого
    есть только анкета. В файл список не пишется: это число
    для терминала.
    """

    ids: set[str] = set()

    ids.update(raw.read("profile", ["client_id"]).column("client_id").to_pylist())

    for _, chunk in raw.iter_row_groups("events", ["client_id"]):
        ids.update(pc.unique(chunk.column("client_id")).to_pylist())

    return ids


def _clear(out_dir: Path) -> None:
    """
    Каталог этапа перед записью пуст.

    Чистится всё, что в нём лежит: прежняя сборка могла оставить
    файлы, которых этап больше не делает.
    """

    if not out_dir.exists():
        return

    for item in sorted(out_dir.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        else:
            item.rmdir()


__all__ = [
    "EVENTS_FILE",
    "PERIOD_END_KEY",
    "PERIOD_START_KEY",
    "SCHEMA_VERSION",
    "STAGE",
    "CanonicalResult",
    "build_group",
]
