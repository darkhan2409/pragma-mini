from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.settings import PreprocessingConfig
from src.tokenization.layout import FrozenArtifacts
from src.tokenization.specials import EMPTY, INVALID, MISSING, UNK

from .sample import Sample, SampleError, build_sample
from .settings import SAMPLES_FILE, DatasetConfig, dataset_dir
from .tokenized import TokenizedError, TokenizedGroup


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сборка группы это один проход по её закодированным клиентам и
# один файл на выходе:
#
#   data/dataset/<group>/samples.parquet
#
# Строка это клиент: его события во времени, его профиль,
# границы, каналы времени, маска допустимых целей и вес. Второй
# пример одного клиента не создаётся: срез у группы один — её
# конечный cutoff, и повторять клиента незачем.
#
# Группы не смешиваются нигде: и клиенты, и словарь приходят
# каждый из своего места, а период целей берётся из окна именно
# этой группы.
#
# Длинная история усекается объявленной политикой контекста, и
# усечение всегда названо: сколько событий и токенов осталось за
# границей примера, видно в самой строке.
# ============================================================


class BuildError(ValueError):
    """
    Набор собрать нельзя.
    """


SAMPLES_SCHEMA = pa.schema(
    [
        # --- тождество ---
        ("client_id", pa.string()),
        ("group", pa.string()),
        ("cutoff", pa.timestamp("us")),
        ("weight", pa.float64()),
        ("sample_seed", pa.int64()),

        # --- размеры ---
        ("n_events", pa.int32()),
        ("n_tokens", pa.int32()),
        ("n_values", pa.int32()),

        # --- события: входы модели ---
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("event_starts", pa.list_(pa.int32())),
        ("event_lengths", pa.list_(pa.int32())),

        # --- события: время ---
        ("event_time", pa.list_(pa.timestamp("us"))),
        ("hours_to_cutoff", pa.list_(pa.float64())),
        ("calendar", pa.list_(pa.float32())),

        # --- цели ---
        ("event_eligible", pa.list_(pa.bool_())),
        ("n_eligible_events", pa.int32()),
        ("has_targets", pa.bool_()),

        # --- значения событий ---
        ("value_event", pa.list_(pa.int32())),
        ("value_start", pa.list_(pa.int32())),
        ("value_length", pa.list_(pa.int32())),
        ("value_key_id", pa.list_(pa.int32())),

        # --- профиль ---
        ("profile_key_ids", pa.list_(pa.int32())),
        ("profile_value_ids", pa.list_(pa.int32())),
        ("profile_positions", pa.list_(pa.int32())),
        ("profile_value_start", pa.list_(pa.int32())),
        ("profile_value_length", pa.list_(pa.int32())),
        ("profile_value_key_id", pa.list_(pa.int32())),
        ("has_profile", pa.bool_()),

        # --- служебное ---
        ("truncated", pa.bool_()),
        ("excluded_events", pa.int32()),
        ("excluded_tokens", pa.int32()),
        ("excluded_eligible", pa.int32()),
        ("limitations", pa.list_(pa.string())),
    ]
)


@dataclass
class Counters:
    samples: int = 0
    events: int = 0
    # Токены и значения считаются по ОБЕИМ записям примера:
    # итог набора обязан сходиться с тем, что насчитало
    # кодирование, иначе две величины молча расходятся.
    tokens: int = 0
    values: int = 0
    profile_tokens: int = 0
    eligible: int = 0
    with_targets: int = 0
    silent: int = 0
    without_profile: int = 0
    truncated: int = 0
    excluded_events: int = 0
    max_tokens: int = 0
    specials: dict[str, int] = field(default_factory=dict)
    limitations: set[str] = field(default_factory=set)


def _row(sample: Sample) -> dict:
    return {
        "client_id": sample.client_id,
        "group": sample.group,
        "cutoff": sample.cutoff,
        "weight": sample.weight,
        "sample_seed": sample.sample_seed,
        "n_events": sample.n_events,
        "n_tokens": sample.n_tokens,
        "n_values": sample.n_values,
        "key_ids": sample.key_ids.tolist(),
        "value_ids": sample.value_ids.tolist(),
        "positions": sample.positions.tolist(),
        "event_starts": sample.event_starts.tolist(),
        "event_lengths": sample.event_lengths.tolist(),
        "event_time": sample.event_time.tolist(),
        "hours_to_cutoff": sample.hours_to_cutoff.tolist(),
        "calendar": sample.calendar.tolist(),
        "event_eligible": sample.event_eligible.tolist(),
        "n_eligible_events": sample.n_eligible_events,
        "has_targets": sample.has_targets,
        "value_event": sample.value_event.tolist(),
        "value_start": sample.value_start.tolist(),
        "value_length": sample.value_length.tolist(),
        "value_key_id": sample.value_key_id.tolist(),
        "profile_key_ids": sample.profile_key_ids.tolist(),
        "profile_value_ids": sample.profile_value_ids.tolist(),
        "profile_positions": sample.profile_positions.tolist(),
        "profile_value_start": sample.profile_value_start.tolist(),
        "profile_value_length": sample.profile_value_length.tolist(),
        "profile_value_key_id": sample.profile_value_key_id.tolist(),
        "has_profile": sample.has_profile,
        "truncated": sample.truncated,
        "excluded_events": sample.excluded_events,
        "excluded_tokens": sample.excluded_tokens,
        "excluded_eligible": sample.excluded_eligible,
        "limitations": list(sample.limitations),
    }


def _count_specials(artifacts: FrozenArtifacts, sample: Sample, counters: Counters) -> None:

    names = {
        artifacts.special(MISSING): "missing",
        artifacts.special(UNK): "unknown",
        artifacts.special(INVALID): "invalid",
        artifacts.special(EMPTY): "empty",
    }

    for values in (sample.value_ids, sample.profile_value_ids):
        for value_id in values.tolist():
            name = names.get(value_id)
            if name is not None:
                counters.specials[name] = counters.specials.get(name, 0) + 1


def build_group(
    artifacts: FrozenArtifacts,
    group: str,
    config: DatasetConfig,
    directory: Path | None = None,
) -> dict:
    """
    Собирает группу в один файл примеров.
    """

    directory = Path(directory) if directory is not None else dataset_dir(group)

    window = PreprocessingConfig.load(None).windows.get(group)

    if window is None:
        raise BuildError(f"для группы {group} не объявлено окно наблюдения")

    try:
        source = TokenizedGroup(group)
    except TokenizedError as error:
        raise BuildError(str(error)) from error

    cutoff = source.cutoff

    if cutoff != window.final_cutoff:
        raise BuildError(
            f"группа {group} закодирована до {cutoff.isoformat()}, а окно объявляет "
            f"{window.final_cutoff.isoformat()}: кодирование и настройки разошлись, "
            f"выполните python -m src.tokenization.run encode {group}"
        )

    _clear(directory)

    counters = Counters()

    writer: pq.ParquetWriter | None = None

    batch: list[dict] = []

    try:
        for client in source.clients():

            try:
                sample = build_sample(
                    artifacts=artifacts,
                    client=client,
                    group=group,
                    window=window,
                    cutoff=cutoff,
                    policy=config.context,
                )
            except SampleError as error:
                raise BuildError(str(error)) from error

            # Оценка обязана мериться одной линейкой. Потерянная
            # цель в validation или test делает её зависящей от
            # политики усечения, и молчать об этом нельзя.
            if group != artifacts.bundle["fit"]["group"] and sample.excluded_eligible:
                raise BuildError(
                    f"группа {group}, клиент {client.client_id}: отбор контекста выбросил "
                    f"{sample.excluded_eligible} событий периода целей. В оценочной группе это "
                    "запрещено: поднимите бюджет контекста или соберите её политикой all"
                )

            counters.samples += 1
            counters.events += sample.n_events
            counters.tokens += sample.n_tokens + sample.profile_tokens
            counters.values += sample.n_values + len(sample.profile_value_start)
            counters.profile_tokens += sample.profile_tokens
            counters.eligible += sample.n_eligible_events
            counters.with_targets += int(sample.has_targets)
            counters.silent += int(sample.n_events == 0)
            counters.without_profile += int(not sample.has_profile)
            counters.truncated += int(sample.truncated)
            counters.excluded_events += sample.excluded_events
            counters.max_tokens = max(counters.max_tokens, sample.n_tokens)
            counters.limitations.update(sample.limitations)

            _count_specials(artifacts, sample, counters)

            batch.append(_row(sample))

            if len(batch) >= config.row_group_samples:
                writer = _write(directory, writer, batch)
                batch = []

        if batch or writer is None:
            writer = _write(directory, writer, batch)

    finally:
        if writer is not None:
            writer.close()

    return {
        "group": group,
        "cutoff": cutoff.isoformat(),
        "file": str(directory / SAMPLES_FILE),
        "counts": {
            "samples": counters.samples,
            "events": counters.events,
            "tokens": counters.tokens,
            "values": counters.values,
            "profile_tokens": counters.profile_tokens,
            "eligible_events": counters.eligible,
            "samples_with_targets": counters.with_targets,
            "silent_clients": counters.silent,
            "clients_without_profile": counters.without_profile,
            "truncated": counters.truncated,
            "excluded_events": counters.excluded_events,
            "max_tokens": counters.max_tokens,
        },
        "specials": dict(sorted(counters.specials.items())),
        "limitations": sorted(counters.limitations),
    }


def _write(directory: Path, writer: pq.ParquetWriter | None, rows: list[dict]) -> pq.ParquetWriter:
    """
    Одна группа строк файла примеров.
    """

    table = pa.Table.from_pylist(rows, schema=SAMPLES_SCHEMA)

    if writer is None:
        directory.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(directory / SAMPLES_FILE, SAMPLES_SCHEMA, compression="zstd")

    if table.num_rows:
        writer.write_table(table)

    return writer


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только файл примеров: прежний результат
    стирается целиком.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "SAMPLES_SCHEMA",
    "BuildError",
    "build_group",
]
