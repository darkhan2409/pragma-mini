from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.telemetry import TelemetryReader, progress


# ============================================================
# ИДЕЯ
# ============================================================
#
# Дашборд читает telemetry.jsonl, пока обучение его дописывает.
# Проверяется читатель, без Streamlit:
#
#   - строка без перевода строки не берётся, пока не допишется, и
#     потом берётся целиком ровно один раз;
#   - битая целая строка пропускается и считается, соседние целы;
#   - продолжение после обрыва отбрасывает шаги и эпохи,
#     записанные после последнего чекпойнта;
#   - новый файл на месте прежнего читается с начала;
#   - строки прежнего формата без kind — эпохи;
#   - на настоящем прогоне читатель видит все шаги, val только
#     после эпох и план из строки run.
# ============================================================


def run(epoch: int = 1, step: int = 0, epochs: int = 2, total: int = 4, resumed: bool = False,
        planned: int | None = None) -> dict:
    return {
        "kind": "run", "epoch": epoch, "step": step, "epochs": epochs,
        "planned_epochs": epochs if planned is None else planned,
        "total_steps": total, "max_steps": None, "max_grad_norm": 1.0, "resumed": resumed, "time": 0.0,
    }


def step(number: int, epoch: int = 1, loss: float = 2.0) -> dict:
    return {
        "kind": "step", "epoch": epoch, "step": number, "loss": loss, "targets": 10, "micro_batches": 1,
        "tokens": 100, "lr": 1e-4, "grad_norm": 0.5, "wait_seconds": 0.0, "seconds": 0.5, "time": 0.0,
    }


def epoch(number: int, at: int, val: float | None = 1.5) -> dict:
    return {"kind": "epoch", "epoch": number, "step": at, "train_loss": 2.0, "val_loss": val}


def lines(*records: dict) -> str:
    return "".join(json.dumps(record) + "\n" for record in records)


def append(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def test_a_line_being_written_is_taken_only_when_complete(tmp_path):

    path = tmp_path / "telemetry.jsonl"
    reader = TelemetryReader(path)

    whole = json.dumps(step(2))

    append(path, lines(run(), step(1)) + whole[:17])

    assert sorted(reader.poll().steps) == [1]

    append(path, whole[17:])

    # Всё ещё без перевода строки.
    assert sorted(reader.poll().steps) == [1]

    append(path, "\n")

    telemetry = reader.poll()

    assert sorted(telemetry.steps) == [1, 2]
    assert telemetry.steps[2] == step(2)
    assert telemetry.skipped == 0

    # Прочитанное второй раз не берётся.
    assert sorted(reader.poll().steps) == [1, 2]


def test_a_broken_line_is_skipped_and_counted(tmp_path):
    """
    Обрыв записи, после которого продолжение дописало свою строку:
    склеенная строка битая, остальные читаются.
    """

    path = tmp_path / "telemetry.jsonl"

    path.write_text(lines(run(), step(1)) + json.dumps(step(2))[:20] + lines(run(step=1, resumed=True), step(2, loss=1.0)))

    telemetry = TelemetryReader(path).poll()

    assert telemetry.skipped == 1
    assert sorted(telemetry.steps) == [1, 2]
    assert telemetry.steps[2]["loss"] == 1.0


def test_resume_drops_what_the_broken_run_wrote_after_the_last_checkpoint(tmp_path):
    """
    Прогон дошёл до шага 5 и записал эпоху 2, но чекпойнт остался
    на эпохе 1 (шаг 3). Продолжение начинает эпоху 2 с шага 3:
    шаги 4–5 и эпоха 2 прежнего прогона — не те, что будут.
    """

    path = tmp_path / "telemetry.jsonl"

    path.write_text(lines(
        run(), step(1), step(2), step(3), epoch(1, 3),
        step(4, epoch=2), step(5, epoch=2), epoch(2, 5, val=9.9),
        run(epoch=2, step=3, resumed=True),
    ))

    reader = TelemetryReader(path)
    telemetry = reader.poll()

    assert sorted(telemetry.steps) == [1, 2, 3]
    assert sorted(telemetry.epochs) == [1]

    append(path, lines(step(4, epoch=2, loss=1.25), epoch(2, 4, val=1.1)))

    telemetry = reader.poll()
    where = progress(telemetry)

    assert telemetry.steps[4]["loss"] == 1.25
    assert where["val_loss"] == 1.1 and where["best_val_loss"] == 1.1
    assert where["resumes"] == 1 and where["completed_epochs"] == 2


def test_a_new_file_in_place_of_the_old_one_is_read_from_the_start(tmp_path):
    """
    Новое обучение без --resume удаляет прежнюю телеметрию.
    """

    path = tmp_path / "telemetry.jsonl"
    reader = TelemetryReader(path)

    path.write_text(lines(run(), step(1), step(2), step(3), epoch(1, 3)))

    assert sorted(reader.poll().steps) == [1, 2, 3]

    path.unlink()

    assert reader.poll().steps == {}

    path.write_text(lines(run(epochs=5, total=10), step(1, loss=7.0)))

    telemetry = reader.poll()

    assert sorted(telemetry.steps) == [1]
    assert telemetry.steps[1]["loss"] == 7.0
    assert telemetry.epochs == {}
    assert progress(telemetry)["total_steps"] == 10


def test_old_lines_without_kind_are_epochs(tmp_path):

    path = tmp_path / "telemetry.jsonl"

    path.write_text(lines(
        {"epoch": 1, "step": 3, "train_seconds": 1.0, "steps": 3},
        {"epoch": 2, "step": 6, "train_seconds": 1.0, "steps": 3},
    ))

    telemetry = TelemetryReader(path).poll()
    where = progress(telemetry)

    assert sorted(telemetry.epochs) == [1, 2]
    assert where["step"] == 6 and where["val_loss"] is None and where["total_steps"] is None


def test_progress_estimates_epochs_beyond_the_plan_and_respects_max_steps():

    from src.dashboard.telemetry import Telemetry

    telemetry = Telemetry()
    telemetry.add(dict(run(epochs=3, total=4, planned=2)))

    where = progress(telemetry)

    assert where["total_steps"] == 6 and where["estimated"]

    telemetry.add(dict(run(epochs=2, total=4), max_steps=3))

    assert progress(telemetry)["total_steps"] == 3


def test_the_reader_follows_a_real_training_run(stage):
    """
    Настоящее обучение на крошечном мире: читатель видит каждый шаг
    оптимизатора, val — по точке на эпоху на шаге её конца, план —
    из строки run.
    """

    from src.mlm.settings import TELEMETRY_FILE, train_dir
    from src.mlm.train import train

    from tests.test_scheduler import many
    from tests.test_training_math import every_value, settle, tiny

    settle(stage, train_people=many())

    result = train(tiny(token_budget=6, early_stopping_patience=10), epochs=2, max_steps=None, masking=every_value())

    telemetry = TelemetryReader(train_dir() / TELEMETRY_FILE).poll()
    where = progress(telemetry)

    assert telemetry.skipped == 0
    assert sorted(telemetry.steps) == list(range(1, result["step"] + 1))
    assert sorted(telemetry.epochs) == [1, 2]
    assert where["step"] == where["total_steps"] == result["step"]
    assert where["epoch"] == where["epochs"] == 2
    assert where["best_val_loss"] == result["best_val_loss"]
    assert telemetry.epochs[2]["step"] == result["step"]


def test_the_page_shows_the_run_without_errors(tmp_path):
    """
    Страница целиком, без сервера: метрики из последних строк, по
    графику на loss, LR, норму, скорость и память, val — только
    точки эпох. Недописанная строка в конце файла ошибкой не
    становится.
    """

    pytest.importorskip("streamlit")

    from streamlit.testing.v1 import AppTest

    from src.dashboard import app

    rows = [run(epochs=2, total=4)]
    rows += [step(number, epoch=1 + (number > 2), loss=3.0 - number / 4) for number in range(1, 5)]
    rows += [epoch(1, 2, val=2.5), dict(epoch(2, 4, val=2.25), cuda_peak_allocated_gib=1.5, cuda_peak_reserved_gib=2.0)]

    (tmp_path / "telemetry.jsonl").write_text(lines(*rows) + '{"kind": "st')

    page = AppTest.from_file(app.__file__, default_timeout=60)
    page.run()

    assert not page.exception

    page.sidebar.text_input[0].input(str(tmp_path)).run()

    assert not page.exception

    shown = {metric.label: metric.value for metric in page.metric}

    assert shown["Эпоха"] == "2 / 2"
    assert shown["Шаг"] == "4 / 4"
    assert shown["Train loss"] == "2.0000"
    assert shown["Val loss"] == "2.2500"
    assert shown["Пик VRAM"] == "1.50 ГиБ"

    telemetry = TelemetryReader(tmp_path / "telemetry.jsonl").poll()
    chart = app.loss_chart(app.steps_frame(telemetry, 1), telemetry, 1).to_dict()

    assert sorted(len(data) for data in chart["datasets"].values()) == [2, 4]


def test_the_page_follows_lines_appended_between_refreshes(tmp_path):
    """
    Живое обновление: обучение дописывает файл между перерисовками.
    Страница дочитывает только новое (читатель живёт в сессии и
    ничего не берёт дважды), недописанную строку показывает, лишь
    когда та допишется, а после новой эпохи — её val.
    """

    pytest.importorskip("streamlit")

    from streamlit.testing.v1 import AppTest

    from src.dashboard import app

    path = tmp_path / "telemetry.jsonl"

    path.write_text(lines(run(epochs=2, total=4), step(1, loss=3.0)))

    page = AppTest.from_file(app.__file__, default_timeout=60)
    page.run()
    page.sidebar.text_input[0].input(str(tmp_path)).run()

    def shown() -> dict:
        assert not page.exception
        return {metric.label: metric.value for metric in page.metric}

    def reader() -> TelemetryReader:
        return page.session_state[f"reader:{tmp_path}"]

    assert shown()["Шаг"] == "1 / 4"
    assert shown()["Val loss"] == "—"

    # Шаг 2 целиком и начало шага 3 без перевода строки.
    tail = json.dumps(step(3, epoch=2, loss=2.5))
    append(path, lines(step(2, loss=2.75), epoch(1, 2, val=2.6)) + tail[:25])

    page.run()

    assert shown()["Шаг"] == "2 / 4"
    assert shown()["Train loss"] == "2.7500"
    assert shown()["Val loss"] == "2.6000"
    assert sorted(reader().telemetry.steps) == [1, 2]

    append(path, tail[25:] + "\n" + lines(step(4, epoch=2, loss=2.25), epoch(2, 4, val=2.4)))

    page.run()

    assert shown()["Эпоха"] == "2 / 2"
    assert shown()["Шаг"] == "4 / 4"
    assert shown()["Train loss"] == "2.2500"
    assert shown()["Val loss"] == "2.4000"

    telemetry = reader().telemetry

    assert sorted(telemetry.steps) == [1, 2, 3, 4]
    assert telemetry.steps[3] == step(3, epoch=2, loss=2.5)
    assert telemetry.skipped == 0

    # Перерисовка без новых строк ничего не меняет и не читает
    # прочитанное заново.
    page.run()

    assert shown()["Шаг"] == "4 / 4"
    assert sorted(reader().telemetry.steps) == [1, 2, 3, 4]
    assert reader()._offset == path.stat().st_size
