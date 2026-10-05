from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# ЧТЕНИЕ ТЕЛЕМЕТРИИ ВО ВРЕМЯ ОБУЧЕНИЯ
# ============================================================
#
# Обучение (src.mlm.train) дописывает в telemetry.jsonl каталога
# прогона строки трёх видов:
#
#   run    начало прогона и продолжения: эпохи, горизонт шагов,
#          max_steps, max_grad_norm;
#   step   шаг оптимизатора: loss, lr, норма градиента до клипа,
#          токены и время окна;
#   epoch  полная эпоха: train_loss, val_loss, время, нормы, пики
#          памяти CUDA.
#
# Читатель только читает файл, с того места, где остановился в
# прошлый раз: обучение он не тормозит и в GPU не заходит.
#
#   - Хвост без перевода строки — строка, которую обучение ещё
#     пишет: он не берётся, пока строка не допишется.
#   - Целая, но битая строка (обрыв записи, после которого
#     продолжение дописало свою) пропускается и считается.
#   - Файл короче прочитанного или другой файл на том же месте —
#     новое обучение стёрло прежний: чтение начинается заново.
#   - Строка run продолжения отбрасывает шаги после своего шага и
#     эпохи с её эпохи: их писал прогон, оборвавшийся после
#     последнего чекпойнта, и продолжение пройдёт их заново.
# ============================================================


@dataclass
class Telemetry:
    """
    Всё, что прочитано из файла: прогоны по порядку, шаги по
    номеру, эпохи по номеру.
    """

    runs: list[dict] = field(default_factory=list)
    steps: dict[int, dict] = field(default_factory=dict)
    epochs: dict[int, dict] = field(default_factory=dict)

    # Целые строки, которые не удалось разобрать.
    skipped: int = 0

    # Время последнего изменения файла, секунды эпохи Unix.
    modified: float | None = None

    def add(self, record: dict) -> None:

        kind = record.get("kind")

        if kind == "run":
            step, epoch = int(record["step"]), int(record["epoch"])
            self.steps = {number: item for number, item in self.steps.items() if number <= step}
            self.epochs = {number: item for number, item in self.epochs.items() if number < epoch}
            self.runs.append(record)

        elif kind == "step":
            self.steps[int(record["step"])] = record

        elif kind == "epoch":
            self.epochs[int(record["epoch"])] = record

        else:
            self.skipped += 1


class TelemetryReader:
    """
    Читатель одного telemetry.jsonl, дочитывающий новые строки.
    """

    def __init__(self, path: Path):

        self.path = Path(path)
        self.telemetry = Telemetry()

        self._offset = 0
        self._identity: tuple[int, int] | None = None

    def poll(self) -> Telemetry:
        """
        Дочитать целые строки, дописанные с прошлого раза.
        """

        try:
            status = os.stat(self.path)
        except FileNotFoundError:
            self._reset()
            return self.telemetry

        identity = (status.st_dev, status.st_ino)

        if identity != self._identity or status.st_size < self._offset:
            self._reset()
            self._identity = identity

        self.telemetry.modified = status.st_mtime

        if status.st_size == self._offset:
            return self.telemetry

        with open(self.path, "rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read(status.st_size - self._offset)

        # Только до последнего перевода строки: дальше — строка,
        # которую обучение ещё не дописало.
        end = chunk.rfind(b"\n")

        if end < 0:
            return self.telemetry

        for line in chunk[:end].split(b"\n"):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.telemetry.skipped += 1
                continue

            if not isinstance(record, dict):
                self.telemetry.skipped += 1
                continue

            try:
                self.telemetry.add(record)
            except (KeyError, TypeError, ValueError):
                self.telemetry.skipped += 1

        self._offset += end + 1

        return self.telemetry

    def _reset(self) -> None:

        self.telemetry = Telemetry()
        self._offset = 0
        self._identity = None


def progress(telemetry: Telemetry) -> dict:
    """
    Где обучение сейчас: эпоха и шаг из всего и последние значения.

    Шагов всего — горизонт cosine из строки run. Если --epochs
    больше плана, эпохи сверх него идут тем же числом шагов на
    эпоху, и итог — оценка (estimated). max_steps ограничивает его.
    """

    run = telemetry.runs[-1] if telemetry.runs else None
    last_step = telemetry.steps[max(telemetry.steps)] if telemetry.steps else None
    last_epoch = telemetry.epochs[max(telemetry.epochs)] if telemetry.epochs else None

    step = max(
        [record["step"] for record in (run, last_step, last_epoch) if record is not None],
        default=0,
    )

    if last_step is not None:
        epoch = last_step["epoch"]
    elif run is not None:
        epoch = run["epoch"]
    else:
        epoch = None

    total_steps, estimated = None, False

    if run is not None:

        total_steps = run["total_steps"]

        if run["epochs"] > run["planned_epochs"] and run["planned_epochs"]:
            total_steps = round(total_steps * run["epochs"] / run["planned_epochs"])
            estimated = True

        if run["max_steps"] is not None:
            total_steps = min(total_steps, run["max_steps"])

    val = [
        (number, record["val_loss"])
        for number, record in sorted(telemetry.epochs.items())
        if record.get("val_loss") is not None
    ]

    return {
        "epoch": epoch,
        "epochs": run["epochs"] if run is not None else None,
        "step": step,
        "total_steps": total_steps,
        "estimated": estimated,
        "completed_epochs": len(telemetry.epochs),
        "last_step": last_step,
        "last_epoch": last_epoch,
        "val_loss": val[-1][1] if val else None,
        "best_val_loss": min(value for _, value in val) if val else None,
        "resumes": sum(1 for record in telemetry.runs if record.get("resumed")),
    }


__all__ = ["Telemetry", "TelemetryReader", "progress"]
