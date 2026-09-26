from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from src.mlm.backbone import initial_history, payload
from src.preprocessing.artifacts import TableWriter

from .encoder import HistoryEncoder
from .inputs import Client, Source
from .settings import HISTORY_FILE, WEIGHTS_FILE, HistoryConfig, history_dir


# ============================================================
# ИДЕЯ
# ============================================================
#
# У этапа два файла на выходе:
#
#   history.parquet — ИТОГОВЫЙ ВЕКТОР КЛИЕНТА, один на строку;
#   weights.pt      — веса энкодера.
#
# Строка это клиент, и это ТА ЖЕ строка, что в batches.parquet:
# тот же порядок, та же группа строк на батч.
#
# Обновлённые векторы СОБЫТИЙ энкодер тоже отдаёт — они нужны
# MLM-голове, — но на диск не пишутся: голова считает их сама в
# том же прямом проходе, и снимок ей не нужен. Наружу из функции
# они возвращаются, в файл не попадают.
#
# ВАЖНО, чем этот файл НЕ является. Векторы посчитаны начальным
# розыгрышем весов всех четырёх слоёв. При обучении они обязаны
# считаться вместе в одном прямом проходе, и файлы этапов 10-12
# замороженным входом обучения не являются.
#
# Этап — диагностика: начальные веса обучения даёт python -m
# src.mlm.init_backbone без прохода по данным. Веса здесь
# разыгрываются той же функцией (backbone.initial_history).
# ============================================================


HISTORY_SCHEMA = pa.schema(
    [
        # --- где лежит клиент ---
        ("batch_index", pa.int32()),
        ("client_id", pa.string()),

        # --- длина вектора ---
        ("dim", pa.int32()),

        # Итоговое представление клиента: выход позиции [USR]
        # после истории, d чисел.
        ("client", pa.list_(pa.float32())),
    ]
)


class HistoryError(ValueError):
    """
    Историю посчитать нельзя.
    """


def build_group(
    group: str,
    config: HistoryConfig,
    index: int = 0,
    directory: Path | None = None,
) -> dict:
    """
    Векторы истории всей группы и отчёт по одному её батчу.
    """

    source = Source(group)

    dim = source.dim

    device = _device(config.device)

    encoder = initial_history(config, dim)

    encoder.eval()

    # Веса разыграны на CPU и только теперь переезжают: сборка
    # сразу на карте тянула бы числа из другого генератора, и
    # «тот же seed» перестало бы значить «те же веса».
    encoder.to(device)

    directory = Path(directory) if directory is not None else history_dir(group)

    _clear(directory)

    table_path = directory / HISTORY_FILE
    weights_path = directory / WEIGHTS_FILE

    writer = TableWriter(table_path, HISTORY_SCHEMA)

    clients = 0
    events = 0
    longest = 0
    seconds = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        for number in range(source.count):

            batch = source.batch(number)

            rows = []

            for client in batch:

                started = time.perf_counter()

                with torch.no_grad():
                    vector, updated = encode_client(encoder, client, device)

                seconds += time.perf_counter() - started

                _check(client, vector, updated)

                rows.append(
                    {
                        "batch_index": client.batch_index,
                        "client_id": client.client_id,
                        "dim": dim,
                        "client": vector,
                    }
                )

                clients += 1
                events += client.n_events
                longest = max(longest, client.n_events)

            writer.write(_table(rows, dim))

    finally:
        written = writer.close()

    _save(encoder, config, dim, weights_path)

    return {
        "group": group,
        "table": str(table_path),
        "weights": str(weights_path),
        "dim": dim,
        "seed": config.seed,
        "layers": config.layers,
        "heads": config.heads,
        "device": str(device),
        "rows": written,
        "batches": source.count,
        "clients": clients,
        "events": events,
        "longest": longest,
        "seconds": seconds,
        "peak": (
            torch.cuda.max_memory_allocated() / 2 ** 20
            if device.type == "cuda"
            else None
        ),
        "size": table_path.stat().st_size,
    }


def encode_client(
    encoder: HistoryEncoder,
    client: Client,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Одна история через энкодер: вектор клиента и векторы событий.

    Последовательность это [z_a, события от старых к новым], а
    позиция [USR] — ноль, то есть момент самого свежего события.
    """

    # torch.tensor, а не from_numpy: векторы событий приходят
    # видом на буфер arrow, только для чтения, и from_numpy на
    # таком предупреждает. Копия здесь всё равно неизбежна —
    # тензор переезжает на устройство.
    profile = torch.tensor(client.profile, device=device)
    events = torch.tensor(client.events, device=device)

    sequence = torch.cat([profile[None, :], events], dim=0)[None]

    positions = torch.cat(
        [torch.zeros(1, dtype=torch.float32), torch.tensor(client.positions)]
    ).to(device)

    out = encoder(sequence, positions)

    return out[0, 0].cpu().numpy(), out[0, 1:].cpu().numpy()


def _check(client: Client, vector: np.ndarray, updated: np.ndarray) -> None:
    """
    Что посчиталось именно столько, сколько было событий.
    """

    if updated.shape[0] != client.n_events:
        raise HistoryError(
            f"клиент {client.client_id}: посчитано {updated.shape[0]} векторов "
            f"событий вместо {client.n_events}"
        )

    if not np.isfinite(vector).all() or not np.isfinite(updated).all():
        raise HistoryError(f"клиент {client.client_id}: в векторах истории не число")


def _table(rows: list[dict], dim: int) -> pa.Table:
    """
    Строки одного батча: по клиенту на строку.
    """

    return pa.table(
        {
            "batch_index": pa.array([row["batch_index"] for row in rows], pa.int32()),
            "client_id": pa.array([row["client_id"] for row in rows], pa.string()),
            "dim": pa.array([dim] * len(rows), pa.int32()),
            "client": pa.array(
                [row["client"] for row in rows], type=pa.list_(pa.float32())
            ),
        },
        schema=HISTORY_SCHEMA,
    )


def _device(name: str) -> torch.device:
    """
    Где считать.
    """

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise HistoryError("device cuda запрошен, но CUDA недоступна")

    return torch.device(name)


def _save(encoder: HistoryEncoder, config: HistoryConfig, dim: int, path: Path) -> None:
    """
    Веса энкодера рядом с векторами — в формате backbone.payload:
    состояние на CPU, чтобы файл не зависел от того, где считали.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(payload(encoder, config, dim), path)


def _clear(directory: Path) -> None:
    """
    Каталог группы держит только свои два файла.
    """

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "HISTORY_SCHEMA",
    "HistoryError",
    "build_group",
    "encode_client",
]
