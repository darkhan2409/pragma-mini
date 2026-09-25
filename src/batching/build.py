from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from src.dataset.lineage import write_lineage
from src.preprocessing.artifacts import TableWriter
from src.tokenization.specials import PAD, load_special_tokens

from .batch import pack, widths
from .temporal import LENGTH, TemporalGroup
from .settings import BATCHES_FILE, BatchingConfig, batches_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сборка группы это один проход по её примерам и один файл на
# выходе:
#
#   data/07_batches/<group>/batches.parquet
#
# Строка это клиент, как и в примере, только массивы дополнены
# заполнителем до общей длины своего батча. Один батч это одна
# группа строк parquet: один вызов записи создаёт ровно одну
# группу, поэтому read_row_group(i) отдаёт батч i целиком и
# читать ради него весь файл не нужно.
#
# Группы не смешиваются: и примеры, и словарь приходят каждый из
# своего места.
#
# Из словаря берётся ровно одно число — код [PAD]. Читается он
# файлом, а не константой: так видно, что примеры и словарь
# рядом одни и те же.
# ============================================================


BATCHES_SCHEMA = pa.schema(
    [
        # --- где лежит клиент ---
        ("batch_index", pa.int32()),
        ("client_id", pa.string()),

        # --- исходные длины примера до выравнивания ---
        ("n_tokens", pa.int32()),
        ("n_events", pa.int32()),
        ("profile_n_tokens", pa.int32()),

        # --- события одной последовательностью, ширина T ---
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("token_mask", pa.list_(pa.bool_())),

        # --- границы событий и их каналы, ширина E ---
        ("event_starts", pa.list_(pa.int32())),
        ("event_lengths", pa.list_(pa.int32())),
        ("event_time", pa.list_(pa.timestamp("us", tz="UTC"))),
        ("event_time_log", pa.list_(pa.float32())),
        ("calendar", pa.list_(pa.float32())),
        ("event_mask", pa.list_(pa.bool_())),

        # --- что разрешено маскировать: только переносится ---
        ("target_event_mask", pa.list_(pa.bool_())),

        # --- профиль, ширина P ---
        ("profile_key_ids", pa.list_(pa.int32())),
        ("profile_value_ids", pa.list_(pa.int32())),
        ("profile_positions", pa.list_(pa.int32())),
        ("profile_token_mask", pa.list_(pa.bool_())),
    ]
)


@dataclass
class Counters:
    batches: int = 0
    clients: int = 0
    silent: int = 0
    last_batch: int = 0
    max_width: int = 0
    # Настоящее и слоты считаются по обеим осям: доля
    # заполнителя это единственное, ради чего окно вообще
    # упорядочивается по длине.
    tokens_real: int = 0
    tokens_slots: int = 0
    events_real: int = 0
    events_slots: int = 0
    profile_tokens_real: int = 0
    profile_tokens_slots: int = 0


def build_group(
    group: str,
    config: BatchingConfig,
    directory: Path | None = None,
) -> dict:
    """
    Батчи одной группы.
    """

    source = TemporalGroup(group)

    pad = load_special_tokens()[PAD]

    directory = Path(directory) if directory is not None else batches_dir(group)

    _clear(directory)

    counters = Counters()

    writer = TableWriter(directory / BATCHES_FILE, BATCHES_SCHEMA)

    try:
        for window in source.windows(config.window_clients):

            # Колонка длины нужна была только для порядка: в
            # файле батчей её место занимает n_tokens строки.
            rows = window.drop_columns([LENGTH]).to_pylist()

            for start in range(0, len(rows), config.batch_size):

                chunk = rows[start:start + config.batch_size]

                index = counters.batches

                packed = pack(index, chunk, pad)

                _count(counters, chunk, packed)

                writer.write(pa.Table.from_pylist(packed, schema=BATCHES_SCHEMA))

                counters.batches += 1
                counters.last_batch = len(chunk)

    finally:
        rows_written = writer.close()

    # Только после полной записи: прерванная сборка отметки не
    # получает, и читатель её отвергнет.
    write_lineage(directory)

    return {
        "group": group,
        "file": str(directory / BATCHES_FILE),
        "batch_size": config.batch_size,
        "window_clients": config.window_clients,
        "pad_id": pad,
        "rows": rows_written,
        "counts": {
            "batches": counters.batches,
            "clients": counters.clients,
            "silent_clients": counters.silent,
            "last_batch": counters.last_batch,
            "max_width": counters.max_width,
            "tokens_real": counters.tokens_real,
            "tokens_slots": counters.tokens_slots,
            "events_real": counters.events_real,
            "events_slots": counters.events_slots,
            "profile_tokens_real": counters.profile_tokens_real,
            "profile_tokens_slots": counters.profile_tokens_slots,
        },
    }


def _count(counters: Counters, chunk: list[dict], packed: list[dict]) -> None:

    size = widths(chunk)

    counters.clients += len(chunk)
    counters.silent += sum(1 for row in packed if row["n_events"] == 0)
    counters.max_width = max(counters.max_width, size.tokens)

    counters.tokens_real += sum(row["n_tokens"] for row in packed)
    counters.tokens_slots += len(packed) * size.tokens

    counters.events_real += sum(row["n_events"] for row in packed)
    counters.events_slots += len(packed) * size.events

    counters.profile_tokens_real += sum(row["profile_n_tokens"] for row in packed)
    counters.profile_tokens_slots += len(packed) * size.profile_tokens


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только файл батчей: прежний результат
    стирается целиком.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "BATCHES_SCHEMA",
    "Counters",
    "build_group",
]
