from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.artifacts import write_json
from src.preprocessing.profile_state import INCLUDED_FIELDS, LIFELONG_TYPES, PROFILE_SEMANTICS
from src.preprocessing.settings import PreprocessingConfig
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.settings import TokenizerConfig
from src.tokenization.specials import UNK

from .sample import Sample, SampleError, build_sample
from .settings import DATASET_FORMAT, META_FILE, SAMPLES_FILE, DatasetConfig, dataset_dir
from .tokenized import TokenizedError, TokenizedGroup


# ============================================================
# ИДЕЯ
# ============================================================
#
# Сборка группы это один проход по её закодированным клиентам и
# один файл на выходе:
#
#   data/05_dataset/<group>/samples.parquet
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
        ("client_id", pa.string()),

        # --- события клиента одной последовательностью ---
        ("key_ids", pa.list_(pa.int32())),
        ("value_ids", pa.list_(pa.int32())),
        ("positions", pa.list_(pa.int32())),
        ("event_starts", pa.list_(pa.int32())),
        ("event_lengths", pa.list_(pa.int32())),
        ("event_time", pa.list_(pa.timestamp("us", tz="UTC"))),
        ("calendar", pa.list_(pa.float32())),

        # --- что разрешено маскировать ---
        ("target_event_mask", pa.list_(pa.bool_())),

        # --- профиль: Attributes на cutoff и вехи раньше него ---
        ("profile_key_ids", pa.list_(pa.int32())),
        ("profile_value_ids", pa.list_(pa.int32())),
        ("profile_positions", pa.list_(pa.int32())),
        # Время токена анкеты: у вехи её момент, у остальных null.
        ("profile_time", pa.list_(pa.timestamp("us", tz="UTC"))),
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
    empty_profiles: int = 0
    truncated: int = 0
    excluded_events: int = 0
    max_tokens: int = 0
    unknown: int = 0


def _row(sample: Sample) -> dict:
    return {
        "client_id": sample.client_id,
        "key_ids": sample.key_ids.tolist(),
        "value_ids": sample.value_ids.tolist(),
        "positions": sample.positions.tolist(),
        "event_starts": sample.event_starts.tolist(),
        "event_lengths": sample.event_lengths.tolist(),
        "event_time": sample.event_time.tolist(),
        "calendar": sample.calendar.tolist(),
        "target_event_mask": sample.target_event_mask.tolist(),
        "profile_key_ids": sample.profile_key_ids.tolist(),
        "profile_value_ids": sample.profile_value_ids.tolist(),
        "profile_positions": sample.profile_positions.tolist(),
        "profile_time": sample.profile_time.tolist(),
    }


def _count_unknown(artifacts: FrozenArtifacts, sample: Sample, counters: Counters) -> None:
    """
    Сколько значений словарь не знает.
    """

    unknown = artifacts.special(UNK)

    for values in (sample.value_ids, sample.profile_value_ids):
        counters.unknown += int((values == unknown).sum())


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
        source = TokenizedGroup(group, artifacts)
    except TokenizedError as error:
        raise BuildError(str(error)) from error

    # Cutoff у группы один: события строго раньше него, анкета —
    # состояние на него же. Закодированная группа обязана быть
    # собрана на тот же cutoff, иначе набор молча смешал бы два.
    cutoff = window.final_cutoff

    if source.meta.get("events_cutoff") != cutoff.isoformat():
        raise BuildError(
            f"группа {group} закодирована на cutoff {source.meta.get('events_cutoff')}, "
            f"а окно группы — {cutoff.isoformat()}: выполните "
            f"python -m src.tokenization.run encode {group} заново"
        )

    _clear(directory)

    # На какой группе учился словарь: остальные группы
    # оценочные, и терять их цели при усечении нельзя.
    fit_group = TokenizerConfig.load(None).fit_group

    counters = Counters()

    writer: pq.ParquetWriter | None = None

    batch: list[dict] = []

    try:
        for client in source.clients():

            try:
                sample = build_sample(
                    artifacts=artifacts,
                    client=client,
                    window=window,
                    policy=config.context,
                )
            except SampleError as error:
                raise BuildError(str(error)) from error

            # Оценка обязана мериться одной линейкой. Потерянная
            # цель в validation или test делает её зависящей от
            # политики усечения, и молчать об этом нельзя.
            if group != fit_group and sample.excluded_eligible:
                raise BuildError(
                    f"группа {group}, клиент {client.client_id}: отбор контекста выбросил "
                    f"{sample.excluded_eligible} событий периода целей. В оценочной группе это "
                    "запрещено: поднимите бюджет контекста или соберите её политикой all"
                )

            counters.samples += 1
            counters.events += sample.n_events
            counters.tokens += sample.n_tokens + sample.profile_tokens
            counters.values += sample.n_values + sample.profile_n_values
            counters.profile_tokens += sample.profile_tokens
            eligible = int(sample.target_event_mask.sum())

            counters.eligible += eligible
            counters.with_targets += int(eligible > 0)
            counters.silent += int(sample.n_events == 0)
            counters.empty_profiles += int(sample.profile_tokens <= 1)
            counters.truncated += int(sample.truncated)
            counters.excluded_events += sample.excluded_events
            counters.max_tokens = max(counters.max_tokens, sample.n_tokens)

            _count_unknown(artifacts, sample, counters)

            batch.append(_row(sample))

            if len(batch) >= config.row_group_samples:
                writer = _write(directory, writer, batch)
                batch = []

        if batch or writer is None:
            writer = _write(directory, writer, batch)

    finally:
        if writer is not None:
            writer.close()

    meta = {
        "format": DATASET_FORMAT,
        "group": group,
        "events_cutoff": cutoff.isoformat(),
        "profile_semantics": PROFILE_SEMANTICS,
        "profile_fields": list(INCLUDED_FIELDS),
        "profile_lifelong_types": list(LIFELONG_TYPES),
        "samples": counters.samples,
    }

    write_json(directory / META_FILE, meta)

    return {
        "group": group,
        "cutoff": cutoff.isoformat(),
        "meta": meta,
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
            "empty_profiles": counters.empty_profiles,
            "truncated": counters.truncated,
            "excluded_events": counters.excluded_events,
            "max_tokens": counters.max_tokens,
        },
        "unknown_values": counters.unknown,
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
