from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F

from src.tokenization.names import Names
from src.preprocessing.artifacts import TableWriter, write_text
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import UNK, load_special_tokens

from .inputs import IGNORE, Client, Source
from .model import Model, Predicted, load_model, pack
from .varlen import autocast
from .report import Piece, Shot, render
from .settings import (
    PREVIEW_FILE,
    TARGETS_FILE,
    WEIGHTS_FILE,
    MlmConfig,
    mlm_dir,
)
from .version import IMPLEMENTATION_VERSION


# ============================================================
# ИДЕЯ
# ============================================================
#
# У этапа три файла на выходе:
#
#   targets.parquet — строка на КАЖДУЮ размеченную позицию:
#       что было целью, что предсказано и с какими потерями.
#       Полного массива логитов нет — только top-k;
#   preview.html    — один пример, показанный человеку;
#   weights.pt      — веса головы. Веса входного слоя и энкодеров
#       лежат в data/09_embeddings и data/09_backbone и здесь не
#       дублируются.
#
# Сам проход дифференцируемый и живёт в model.py. Здесь он
# вызывается под no_grad, потому что это отчёт: градиенты нужны
# обучению, а не диагностике. Внутри самого прохода no_grad нет.
#
# ВАЖНО. Всё, что тут посчитано, посчитано НЕОБУЧЕННОЙ моделью.
# Потери около ln(размер словаря) означают ровно случайное
# угадывание и качеством модели не являются.
#
# Этап — диагностика, а не вход обучения: python -m src.mlm.train
# его файлов не читает. Модель та же, что учится: входной слой
# train и начальные веса backbone, для любой группы.
# ============================================================


TARGETS_SCHEMA = pa.schema(
    [
        # --- где лежит цель ---
        ("batch_index", pa.int32()),
        ("client_id", pa.string()),
        ("event", pa.int32()),
        ("place", pa.int32()),
        ("position", pa.int32()),

        # --- что закрыто ---
        ("key_id", pa.int32()),
        ("key", pa.string()),
        ("label_id", pa.int32()),
        ("label", pa.string()),
        ("unknown_label", pa.bool_()),
        ("reason", pa.string()),

        # --- что предсказано: только top-k, не весь словарь ---
        ("top_ids", pa.list_(pa.int32())),
        ("top_names", pa.list_(pa.string())),
        ("top_probs", pa.list_(pa.float32())),
        ("predicted_id", pa.int32()),
        ("correct", pa.bool_()),
        ("loss", pa.float32()),
    ]
)


class MlmError(ValueError):
    """
    Этап посчитать нельзя.
    """


def build_group(
    group: str,
    config: MlmConfig,
    directory: Path | None = None,
) -> dict:
    """
    Потери и предсказания по всем целям группы.
    """

    source = Source(group)

    names = Names(FrozenArtifacts.load())

    unknown_id = load_special_tokens()[UNK]

    device = _device(config.device)

    model = load_model(
        seed=config.seed,
        events_per_chunk=config.events_per_chunk,
        label_smoothing=config.label_smoothing,
        device=device,
        attention_backend=config.attention_backend,
    )

    model.eval()

    directory = Path(directory) if directory is not None else mlm_dir(group)

    _clear(directory)

    targets_path = directory / TARGETS_FILE
    preview_path = directory / PREVIEW_FILE
    weights_path = directory / WEIGHTS_FILE

    writer = TableWriter(targets_path, TARGETS_SCHEMA)

    shot: Shot | None = None

    clients = events = targets = correct = unknown = 0
    loss_sum = 0.0
    by_reason: dict[str, int] = {}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        for number in range(source.count):

            batch: list[dict] = []

            for client in source.batch(number):

                # no_grad стоит ЗДЕСЬ: это отчёт. Отчёт поклиентный,
                # поэтому micro-batch здесь из одного клиента —
                # через то же ядро, что и обучение.
                with torch.no_grad(), autocast(device):
                    out = model(pack([client], device))

                rows, shot = _rows(client, out, names, config, unknown_id, shot)

                batch.extend(rows)

                clients += 1
                events += client.n_events
                targets += out.count
                loss_sum += float(out.loss) * out.count
                correct += sum(1 for row in rows if row["correct"])
                unknown += sum(1 for row in rows if row["unknown_label"])

                for row in rows:
                    by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1

            # Одна группа строк на батч, как во всех соседних
            # этапах: клиент это строка, а не группа.
            if batch:
                writer.write(pa.table(_columns(batch), schema=TARGETS_SCHEMA))

    finally:
        written = writer.close()

    _save(model, config, weights_path)

    if shot is None:
        raise MlmError(f"группа {group}: ни одной размеченной цели не нашлось")

    counts = {
        "dim": model.head.dim,
        "clients": clients,
        "events": events,
        "targets": targets,
        "loss": loss_sum / targets if targets else 0.0,
        "correct": correct,
        "unknown": unknown,
        "device": str(device),
        "by_reason": ", ".join(f"{name} {count}" for name, count in sorted(by_reason.items())),
    }

    write_text(
        preview_path,
        render(
            group=group,
            config=config,
            implementation=IMPLEMENTATION_VERSION,
            shot=shot,
            counts=counts,
            paths={
                "targets": targets_path,
                "weights": weights_path,
                "batches": source.batches_path,
                "masked": source.masked_path,
            },
        ),
    )

    peak = (
        torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None
    )

    return {
        "group": group,
        "targets_file": str(targets_path),
        "preview": str(preview_path),
        "weights": str(weights_path),
        "device": str(device),
        "dim": counts["dim"],
        "seed": config.seed,
        "label_smoothing": config.label_smoothing,
        "rows": written,
        "clients": clients,
        "events": events,
        "targets": targets,
        "loss": counts["loss"],
        "correct": correct,
        "unknown": unknown,
        "by_reason": counts["by_reason"],
        "peak": peak,
        "size": targets_path.stat().st_size,
        "client_id": shot.client_id,
        "event": shot.event,
        "pieces": len(shot.pieces),
    }


def _rows(
    client: Client,
    out: Predicted,
    names: Names,
    config: MlmConfig,
    unknown_id: int,
    shot: Shot | None,
) -> tuple[list[dict], Shot | None]:
    """
    Строки по целям одного клиента и, возможно, пример для страницы.
    """

    if out.count == 0:
        return [], shot

    # Под bf16 autocast логиты приходят в bf16: отчёт считает
    # вероятности и потери в fp32.
    logits = out.logits.float()

    losses = F.cross_entropy(
        logits, out.targets, reduction="none",
        label_smoothing=config.label_smoothing,
    )

    probabilities = logits.softmax(dim=-1)

    top = probabilities.topk(min(config.top_k, probabilities.shape[-1]), dim=-1)

    place = out.place.tolist()
    event = out.event.tolist()
    label = out.targets.tolist()
    top_ids = top.indices.tolist()
    top_probs = top.values.tolist()
    per_token = losses.tolist()

    rows = []

    for number, where in enumerate(place):

        key_id = int(client.key_ids[where])
        label_id = int(label[number])
        ids = [int(value) for value in top_ids[number]]

        rows.append(
            {
                "batch_index": client.batch_index,
                "client_id": client.client_id,
                "event": int(event[number]),
                "place": int(where),
                "position": int(client.positions[where]),
                "key_id": key_id,
                "key": names.short(key_id),
                "label_id": label_id,
                "label": names.short(label_id),
                "unknown_label": label_id == unknown_id,
                "reason": client.reason[where],
                "top_ids": ids,
                "top_names": [names.short(value) for value in ids],
                "top_probs": [float(value) for value in top_probs[number]],
                "predicted_id": ids[0],
                "correct": ids[0] == label_id,
                "loss": float(per_token[number]),
            }
        )

    return rows, shot or _shot(client, rows, names)


def _shot(client: Client, rows: list[dict], names: Names) -> Shot | None:
    """
    Пример для страницы: первое значение из нескольких кусков.

    Значение открывает позиция 0 внутри него, куски идут подряд.
    Если у клиента таких целей нет, страница подождёт следующего.
    """

    groups: list[list[dict]] = []

    for row in rows:
        if row["position"] == 0 or not groups:
            groups.append([row])
        else:
            groups[-1].append(row)

    chosen = next((group for group in groups if len(group) > 1), None)

    if chosen is None:
        return None

    return Shot(
        client_id=client.client_id,
        event=chosen[0]["event"],
        time=client.event_time[chosen[0]["event"]],
        key=chosen[0]["key"],
        target=names.text([row["label_id"] for row in chosen]),
        pieces=tuple(
            Piece(
                position=row["position"],
                label=row["label"],
                loss=row["loss"],
                correct=row["correct"],
                top=tuple(zip(row["top_names"], row["top_probs"])),
            )
            for row in chosen
        ),
    )


def _columns(rows: list[dict]) -> dict:
    return {name: [row[name] for row in rows] for name in TARGETS_SCHEMA.names}


def _device(name: str) -> torch.device:

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name == "cuda" and not torch.cuda.is_available():
        raise MlmError("device cuda запрошен, но CUDA недоступна")

    return torch.device(name)


def _save(model: Model, config: MlmConfig, path: Path) -> None:
    """
    Веса головы.

    Входной слой и энкодеры сюда не копируются: их веса лежат в
    data/09_embeddings и data/09_backbone, и второй их копии быть
    не должно.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "dim": model.head.dim,
            "config": config.as_dict(),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.head.state_dict().items()
            },
        },
        path,
    )


def _clear(directory: Path) -> None:

    directory.mkdir(parents=True, exist_ok=True)

    for path in sorted(directory.iterdir()):
        if path.is_file():
            path.unlink()


__all__ = [
    "TARGETS_SCHEMA",
    "MlmError",
    "build_group",
]
