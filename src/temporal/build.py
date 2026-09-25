from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from src.dataset.lineage import write_lineage
from src.dataset.build import SAMPLES_SCHEMA
from src.preprocessing.artifacts import TableWriter

from .position import check, check_profile, profile_time_log, time_log
from .samples import SamplesGroup
from .settings import TEMPORAL_FILE, temporal_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Этап читает готовые примеры и добавляет к каждому два канала:
#
#   data/06_temporal/<group>/temporal.parquet
#
#   event_time_log    давность события до последнего события;
#   profile_time_log  давность вехи анкеты до cutoff примера, ноль
#                     у [USR] и Attributes.
#
# Строка остаётся полным примером клиента, поэтому открыв её
# глазами, видно event_time[i] и event_time_log[i] рядом, а
# profile_time[j] и profile_time_log[j] — тоже. Ни агрегатов, ни
# скрытых значений внутри кода: позиция лежит в файле там же,
# где время, из которого она посчитана.
#
# Ничего не отбирается, не переставляется и не кодируется.
# ============================================================


# Схема это схема примера плюс две колонки, каждая следом за
# временем, из которого посчитана. Она выводится из
# SAMPLES_SCHEMA, а не переписывается рядом: иначе новое поле
# примера молча потерялось бы на этом этапе.
TIME_LOG = pa.field("event_time_log", pa.list_(pa.float32()))

PROFILE_TIME_LOG = pa.field("profile_time_log", pa.list_(pa.float32()))


def _schema() -> pa.Schema:

    fields = list(SAMPLES_SCHEMA)

    for column, after in ((TIME_LOG, "event_time"), (PROFILE_TIME_LOG, "profile_time")):

        index = [field.name for field in fields].index(after)

        fields.insert(index + 1, column)

    return pa.schema(fields)


TEMPORAL_SCHEMA = _schema()


@dataclass
class Counters:
    clients: int = 0
    events: int = 0
    silent: int = 0
    # Самая дальняя позиция и её исходное расстояние: по ним
    # видно, какой кусок шкалы вообще занят.
    max_position: float = 0.0
    max_seconds: int = 0


def build_group(group: str, directory: Path | None = None) -> dict:
    """
    Временные позиции одной группы.
    """

    source = SamplesGroup(group)

    directory = Path(directory) if directory is not None else temporal_dir(group)

    _clear(directory)

    counters = Counters()

    writer = TableWriter(directory / TEMPORAL_FILE, TEMPORAL_SCHEMA)

    try:
        for table in source.groups():

            rows = table.to_pylist()

            for row in rows:

                positions = time_log(row["client_id"], row["event_time"])

                check(row["client_id"], positions, len(row["event_time"]))

                row["event_time_log"] = positions

                ages = profile_time_log(row["client_id"], row["profile_time"], source.cutoff)

                check_profile(row["client_id"], ages, row["profile_time"])

                row["profile_time_log"] = ages

                _count(counters, row)

            writer.write(pa.Table.from_pylist(rows, schema=TEMPORAL_SCHEMA))

    finally:
        rows_written = writer.close()

    # Только после полной записи: прерванная сборка отметки не
    # получает, и читатель её отвергнет.
    write_lineage(directory)

    return {
        "group": group,
        "file": str(directory / TEMPORAL_FILE),
        "rows": rows_written,
        "counts": {
            "clients": counters.clients,
            "events": counters.events,
            "silent_clients": counters.silent,
            "max_position": counters.max_position,
            "max_days": counters.max_seconds / 86_400.0,
        },
    }


def _count(counters: Counters, row: dict) -> None:

    moments = row["event_time"]

    counters.clients += 1
    counters.events += len(moments)

    if not moments:
        counters.silent += 1
        return

    # Первое событие самое дальнее: позиции считаются до
    # последнего, а события упорядочены по времени.
    counters.max_position = max(counters.max_position, row["event_time_log"][0])

    seconds = int((moments[-1] - moments[0]).total_seconds())

    counters.max_seconds = max(counters.max_seconds, seconds)


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только файл позиций: прежний результат
    стирается целиком.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "TEMPORAL_SCHEMA",
    "Counters",
    "build_group",
]
