from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from src.dataset.lineage import write_lineage
from src.preprocessing.artifacts import TableWriter
from src.tokenization.specials import MASK, UNK, load_special_tokens

from .apply import apply
from .batches import BatchesGroup
from .choose import EVENT, KEY, VALUE, Selection, choose
from .settings import MASKED_FILE, MaskingConfig, masked_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сборка группы это один проход по её батчам и один файл на
# выходе:
#
#   data/08_masked/<group>/masked.parquet
#
# Строка это клиент, и это ТА ЖЕ строка, что в batches.parquet:
# тот же порядок, та же группа строк на батч. Поэтому модель
# читает два файла параллельно, а ключи, позиции, границы
# событий, время и календарь не дублируются — они уже лежат
# рядом, и второй их записи быть не должно.
#
# Из словаря берутся ровно два числа — коды [MASK] и [UNK].
# Читаются они файлом, а не константами: так видно, что данные и
# словарь рядом одни и те же.
# ============================================================


MASKED_SCHEMA = pa.schema(
    [
        # --- где лежит клиент ---
        ("batch_index", pa.int32()),
        ("client_id", pa.string()),

        # --- что было и что стало, ширина T ---
        ("value_ids_source", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),

        # --- что предсказывать: -100 значит «не в loss» ---
        ("labels", pa.list_(pa.int32())),

        # --- чем выбрано: event, key, value или пусто ---
        ("reason", pa.list_(pa.string())),
    ]
)


@dataclass
class Counters:
    batches: int = 0
    clients: int = 0
    without_targets: int = 0
    events: int = 0
    values: int = 0
    chosen: int = 0
    by_event: int = 0
    by_key: int = 0
    by_value: int = 0
    masked: int = 0
    unknown: int = 0
    labelled_tokens: int = 0


def build_group(
    group: str,
    config: MaskingConfig,
    directory: Path | None = None,
) -> dict:
    """
    Маски одной группы.
    """

    source = BatchesGroup(group)

    specials = load_special_tokens()

    mask = specials[MASK]
    unknown = specials[UNK]

    directory = Path(directory) if directory is not None else masked_dir(group)

    _clear(directory)

    counters = Counters()

    writer = TableWriter(directory / MASKED_FILE, MASKED_SCHEMA)

    try:
        for batch in source.batches():

            rows = []

            for row in batch.to_pylist():

                selection = choose(group, row, config)

                _count(counters, selection)

                masked = apply(
                    row["client_id"], row, selection.choices, mask, unknown
                )
                masked["batch_index"] = row["batch_index"]

                rows.append(masked)

            writer.write(pa.Table.from_pylist(rows, schema=MASKED_SCHEMA))

            counters.batches += 1

    finally:
        rows_written = writer.close()

    # Только после полной записи: прерванная сборка отметки не
    # получает, и читатель её отвергнет.
    write_lineage(directory)

    return {
        "group": group,
        "file": str(directory / MASKED_FILE),
        "seed": config.seed,
        "rows": rows_written,
        "counts": {
            "batches": counters.batches,
            "clients": counters.clients,
            "without_targets": counters.without_targets,
            "events": counters.events,
            "values": counters.values,
            "chosen": counters.chosen,
            "by_event": counters.by_event,
            "by_key": counters.by_key,
            "by_value": counters.by_value,
            "masked": counters.masked,
            "unknown": counters.unknown,
            "labelled_tokens": counters.labelled_tokens,
        },
    }


def _count(counters: Counters, selection: Selection) -> None:

    counters.clients += 1
    counters.events += selection.events
    counters.values += len(selection.values)

    if not selection.values:
        counters.without_targets += 1

    counters.chosen += len(selection.choices)

    for choice in selection.choices:

        if choice.reason == EVENT:
            counters.by_event += 1
        elif choice.reason == KEY:
            counters.by_key += 1
        elif choice.reason == VALUE:
            counters.by_value += 1

        if choice.unknown:
            counters.unknown += 1
        else:
            counters.masked += 1
            counters.labelled_tokens += choice.value.length


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только файл масок: прежний результат
    стирается целиком.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "MASKED_SCHEMA",
    "Counters",
    "build_group",
]
