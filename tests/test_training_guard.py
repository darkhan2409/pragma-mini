from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.mlm.inputs import Source, micro_batches
from src.mlm.model import pack
from src.mlm.settings import (
    TELEMETRY_FILE,
    best_checkpoint_path,
    checkpoint_path,
    epoch_weights_path,
    train_dir,
)
from src.mlm.train import (
    CheckpointError,
    Scores,
    TrainingError,
    events_bin,
    load_trained,
    train,
    validate,
)

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import CPU, every_value, fresh, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Страховка и учёт обучения, которые не меняют самих шагов:
#
#   - нечисловой loss, градиент или val_loss останавливают
#     обучение ДО записи: на диске остаётся прежнее состояние;
#   - каждая полная эпоха оставляет свои веса, и их можно
#     загрузить обратно той же моделью;
#   - каталог прогона (--out) — единственное, что прогон пишет и
#     чистит;
#   - чекпойнт помнит backbone и данные, и продолжение или
#     загрузка на других отказывают;
#   - разбивка val — это NLL без сглаживания, честно поделённая
#     по целям, а не второе число «примерно того же».
#
# Телеметрия лежит в своём файле, а не в чекпойнте: время стены у
# продолжения другое, а чекпойнт обязан совпасть с непрерывным.
# ============================================================


def poisoned_loss(monkeypatch) -> None:
    """
    Потери модели становятся NaN — как при переполнении в bf16.

    Подменяется функция потерь самой модели, а не шаг обучения:
    проверяется реакция цикла, а не подмена.
    """

    import src.mlm.model as model

    original = model.mlm_loss

    def nan(*args, **kwargs):
        return original(*args, **kwargs) * float("nan")

    monkeypatch.setattr(model, "mlm_loss", nan)


# ============================================================
# НЕЧИСЛОВОЙ ШАГ
# ============================================================


def test_a_non_finite_loss_stops_before_the_optimizer_step(stage, monkeypatch):
    """
    Первая эпоха учится и сохраняется, во второй потери становятся
    NaN. Обучение обязано упасть до optimizer.step: чекпойнт на
    диске остаётся ровно тем, что записала первая эпоха, и весов
    второй эпохи нет.
    """

    settle(stage, train_people=many())

    config, masking = tiny(token_budget=6), every_value()

    train(config, epochs=1, max_steps=None, masking=masking)

    before = checkpoint_path().read_bytes()
    best_before = best_checkpoint_path().read_bytes()

    poisoned_loss(monkeypatch)

    with pytest.raises(TrainingError, match="шаг не сделан"):
        train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    assert checkpoint_path().read_bytes() == before
    assert best_checkpoint_path().read_bytes() == best_before
    assert not epoch_weights_path(2).exists()


def test_a_non_finite_val_loss_writes_no_checkpoint(stage, monkeypatch):
    """
    NaN на val не бывает ни улучшением, ни ухудшением: эпоха не
    пишет ни последнего, ни лучшего чекпойнта, ни своих весов.
    """

    settle(stage, train_people=many())

    monkeypatch.setattr(
        "src.mlm.train.validate",
        lambda model, source, device, token_budget: Scores(loss_sum=float("nan"), targets=5),
    )

    with pytest.raises(TrainingError, match="val_loss nan"):
        train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    assert not checkpoint_path().exists()
    assert not best_checkpoint_path().exists()
    assert not epoch_weights_path(1).exists()


# ============================================================
# ВЕСА ЭПОХ И КАТАЛОГ ПРОГОНА
# ============================================================


def test_every_whole_epoch_leaves_its_weights(stage):
    """
    Три полные эпохи — три файла весов. Последний совпадает с
    весами последнего чекпойнта побитно, а история в нём доходит
    ровно до своей эпохи.
    """

    settle(stage, train_people=many())

    train(tiny(token_budget=6, early_stopping_patience=10), epochs=3, max_steps=None,
          masking=every_value())

    latest = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)

    for epoch in (1, 2, 3):

        saved = torch.load(epoch_weights_path(epoch), map_location="cpu", weights_only=True)

        assert saved["epoch"] == epoch
        assert [row["epoch"] for row in saved["history"]] == list(range(1, epoch + 1))
        assert saved["backbone"] == latest["backbone"]
        assert saved["data"] == latest["data"]

    last = torch.load(epoch_weights_path(3), map_location="cpu", weights_only=True)

    for name, value in latest["model_state_dict"].items():
        assert torch.equal(value, last["model_state_dict"][name]), name

    first = torch.load(epoch_weights_path(1), map_location="cpu", weights_only=True)

    assert any(
        not torch.equal(value, first["model_state_dict"][name])
        for name, value in last["model_state_dict"].items()
    ), "веса первой и третьей эпохи совпали — обучение не шло"


def test_a_paused_epoch_leaves_no_weights(stage):
    """
    Эпоха, прерванная --max-steps, не пройдена целиком: ни
    validation, ни весов эпохи.
    """

    settle(stage, train_people=many())

    result = train(tiny(token_budget=6), epochs=2, max_steps=3, masking=every_value())

    assert result["reason"] == "max_steps"
    assert not epoch_weights_path(1).exists()


def test_weights_of_an_epoch_load_back_into_the_same_model(stage):
    """
    Веса эпохи, загруженные обратно, дают на val ровно те потери,
    что записаны в истории этой эпохи.
    """

    settle(stage, train_people=many())

    config = tiny(token_budget=6, early_stopping_patience=10, label_smoothing=0.1)

    train(config, epochs=2, max_steps=None, masking=every_value())

    model, state = load_trained(epoch_weights_path(1), CPU)

    assert not model.training

    scores = validate(model, Source("val", masking=every_value()), CPU, config.token_budget)

    assert scores.loss == pytest.approx(state["history"][0]["val"]["loss"], rel=1e-6)
    assert scores.detail == state["history"][0]["val_detail"]


def test_a_run_directory_keeps_everything_to_itself(stage):
    """
    Прогон в своём каталоге пишет только туда. Новое обучение в
    каталоге по умолчанию чужой каталог не трогает — и наоборот.
    """

    settle(stage, train_people=many())

    config, masking = tiny(token_budget=6), every_value()

    other = stage / "runs" / "a"

    train(config, epochs=1, max_steps=None, masking=masking, directory=other)

    assert checkpoint_path(other).exists()
    assert epoch_weights_path(1, other).exists()
    assert (other / TELEMETRY_FILE).exists()
    assert not checkpoint_path().exists()

    kept = checkpoint_path(other).read_bytes()

    train(config, epochs=1, max_steps=None, masking=masking)

    assert checkpoint_path().exists()
    assert checkpoint_path(other).read_bytes() == kept
    assert train_dir(other) != train_dir()


def test_a_fresh_run_clears_the_weights_and_telemetry_of_the_last_one(stage):
    """
    Новое обучение без --resume в том же каталоге не оставляет
    весов и телеметрии прошлого прогона: смешать эпохи двух
    прогонов было бы нельзя заметить.
    """

    settle(stage, train_people=many())

    masking = every_value()

    train(tiny(token_budget=6, early_stopping_patience=10), epochs=3, max_steps=None, masking=masking)

    assert epoch_weights_path(3).exists()

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=masking)

    assert epoch_weights_path(1).exists()
    assert not epoch_weights_path(2).exists()
    assert not epoch_weights_path(3).exists()

    rows = [json.loads(line) for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()]

    assert [row["epoch"] for row in rows if row["kind"] == "epoch"] == [1]
    assert [row["resumed"] for row in rows if row["kind"] == "run"] == [False]
    assert {row["epoch"] for row in rows if row["kind"] == "step"} == {1}


# ============================================================
# ПРОИСХОЖДЕНИЕ
# ============================================================


def test_resume_refuses_changed_data(stage):
    """
    После паузы val пересобран на других клиентах: продолжение
    ушло бы считать val по другой группе. Отказ называет файлы и
    ничего не пишет.
    """

    settle(stage, train_people=many())

    config, masking = tiny(token_budget=6), every_value()

    train(config, epochs=2, max_steps=3, masking=masking)

    paused = checkpoint_path().read_bytes()

    settle(stage, train_people=many(), val_people=world.population("w"))

    with pytest.raises(CheckpointError, match="данные изменились") as error:
        train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    assert "05_dataset/val" in str(error.value)
    assert checkpoint_path().read_bytes() == paused


def test_resume_refuses_a_rebuilt_backbone_but_loading_keeps_the_trained_architecture(stage):
    """
    backbone пересобран с другим dropout: те же формы тензоров,
    другая модель. Продолжение обучения обязано отказать: оно
    продолжило бы другой прогон. Загрузка обученной модели — нет: она
    строит архитектуру, записанную в чекпойнте, и считает ровно то
    же, что до пересборки, — иначе после init_backbone под следующий
    эксперимент прошлые модели стали бы незагружаемыми.
    """

    settle(stage, train_people=many())

    config, masking = tiny(token_budget=6), every_value()

    train(config, epochs=2, max_steps=3, masking=masking)
    train(config, epochs=1, max_steps=None, masking=masking, directory=stage / "done")

    before, _ = load_trained(best_checkpoint_path(stage / "done"), CPU)
    before_scores = validate(before, Source("val"), CPU, config.token_budget)

    settle(stage, train_people=many(), dropout=0.2)

    with pytest.raises(CheckpointError, match="07_backbone"):
        train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    after, state = load_trained(best_checkpoint_path(stage / "done"), CPU)

    assert state["backbone"]["encoders"]["history"]["config"]["dropout"] == 0.0
    assert validate(after, Source("val"), CPU, config.token_budget).loss == before_scores.loss


def test_loading_refuses_a_model_of_another_vocabulary(stage):
    """
    Архитектура берётся из чекпойнта, но словарь и входной слой —
    текущие: модель, обученная под другой словарь, не загружается.
    """

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    state["backbone"]["vocabulary"] = "0" * 64

    alien = stage / "alien.pt"
    torch.save(state, alien)

    with pytest.raises(CheckpointError, match="vocabulary"):
        load_trained(alien, CPU)


def test_vectors_are_taken_only_on_the_data_the_model_learned(stage):
    """
    Вход на T строится из текущих данных. После пересборки набора
    под другой эксперимент прежняя модель векторов не даёт.
    """

    from src.dataset.settings import SAMPLES_FILE, dataset_dir
    from src.downstream.embed import trained_model

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    trained_model(str(checkpoint_path()), CPU)

    path = dataset_dir("train") / SAMPLES_FILE
    path.write_bytes(path.read_bytes() + b"\0")

    with pytest.raises(CheckpointError, match="не на текущем"):
        trained_model(str(checkpoint_path()), CPU)


def test_loading_refuses_a_checkpoint_without_its_origin(stage):
    """
    Чекпойнт без отметки backbone (старый формат) не загружается:
    проверить, на той ли архитектуре он учился, нечем.
    """

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    del state["backbone"]

    old = stage / "old.pt"
    torch.save(state, old)

    with pytest.raises(CheckpointError, match="backbone"):
        load_trained(old, CPU)


# ============================================================
# РАЗБИВКА VAL
# ============================================================


@pytest.mark.parametrize("n_events, name", [
    (0, "0-100"), (99, "0-100"), (100, "100-300"), (299, "100-300"),
    (300, "300-1000"), (2999, "1000-3000"), (3000, "3000+"), (12000, "3000+"),
])
def test_events_bin_edges(n_events: int, name: str):
    assert events_bin(n_events) == name


def test_validation_detail_is_the_unsmoothed_loss_split_by_target(stage):
    """
    NLL разбивки — кросс-энтропия БЕЗ сглаживания, посчитанная
    независимо по логитам. Каждая разбивка делит одни и те же цели:
    сумма целей, NLL и попаданий по группам равна общей.
    """

    settle(stage, train_people=many())

    config = tiny(label_smoothing=0.1, token_budget=8)

    model = fresh(stage, config)
    model.eval()

    scores = validate(model, Source("val", masking=every_value()), CPU, config.token_budget)

    detail = scores.detail

    by_hand, count, first = 0.0, 0, 0

    with torch.no_grad():
        for clients in micro_batches(Source("val", masking=every_value()).clients(), config.token_budget):
            out = model(pack(clients, CPU))
            if out.count:
                by_hand += float(F.cross_entropy(out.logits, out.targets, reduction="sum"))
                count += out.count
                first += int((out.logits.argmax(dim=-1) == out.targets).sum())

    assert detail["targets"] == scores.targets == count
    assert detail["nll"] == pytest.approx(by_hand / count, rel=1e-5)

    # Сглаживание меняет число: иначе разбивка ничего не добавляла бы.
    assert detail["nll"] != pytest.approx(scores.loss, rel=1e-3)

    for kind in ("reason", "events", "key"):

        rows = detail[kind].values()

        assert sum(row["targets"] for row in rows) == count, kind
        assert sum(row["nll"] * row["targets"] for row in rows) == pytest.approx(by_hand, rel=1e-5)
        assert sum(round(row["top1"] * row["targets"]) for row in rows) == first, kind

    assert set(detail["reason"]) <= {"event", "key", "value"}
    assert len(detail["key"]) > 1, "ключей целей в мире несколько — разбивка обязана их различать"


# ============================================================
# ТЕЛЕМЕТРИЯ
# ============================================================


def test_telemetry_is_written_per_epoch_and_appended_on_resume(stage, capsys):
    """
    Строка телеметрии на каждую полную эпоху: число шагов совпадает
    с шагами эпохи, нормы градиента положительны, время не
    отрицательно, потери те же, что в истории чекпойнта.
    Продолжение дописывает, а не переписывает.
    """

    settle(stage, train_people=many())

    config, masking = tiny(token_budget=6, early_stopping_patience=10), every_value()

    first = train(config, epochs=1, max_steps=None, masking=masking)
    second = train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    rows = [json.loads(line) for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()]
    epochs = [row for row in rows if row["kind"] == "epoch"]

    assert [row["epoch"] for row in epochs] == [1, 2]
    assert epochs[0]["steps"] == first["step"]
    assert epochs[1]["steps"] == second["step"] - first["step"]

    for row in epochs:
        assert 0.0 < row["grad_norm_mean"] <= row["grad_norm_max"]
        assert 0.0 <= row["clipped_share"] <= 1.0
        assert row["train_seconds"] >= row["data_wait_seconds"] >= 0.0
        assert row["cuda_peak_allocated_gib"] is None

    history = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)["history"]

    for row, item in zip(epochs, history, strict=True):
        assert row["train_loss"] == item["train"]["loss"]
        assert row["val_loss"] == item["val"]["loss"]
        assert row["learning_rate"] == item["learning_rate"]

    printed = capsys.readouterr().out

    assert "grad_norm=" in printed and "wait=" in printed
    assert "val nll" in printed


def test_telemetry_has_the_plan_and_every_optimizer_step(stage, capsys):
    """
    Строка run называет план прогона, строка step — каждый шаг
    оптимизатора теми же числами, что печатает обучение. Шаги идут
    подряд через продолжение, токены шага — токены событий и анкет
    его micro-batch'ей.
    """

    settle(stage, train_people=many())

    config, masking = tiny(token_budget=6, early_stopping_patience=10), every_value()

    first = train(config, epochs=1, max_steps=None, masking=masking)
    second = train(config, epochs=2, max_steps=None, masking=masking, resume=True)

    rows = [json.loads(line) for line in (train_dir() / TELEMETRY_FILE).read_text().splitlines()]

    runs = [row for row in rows if row["kind"] == "run"]
    steps = [row for row in rows if row["kind"] == "step"]

    assert [(row["resumed"], row["epoch"], row["step"]) for row in runs] == [
        (False, 1, 0), (True, 2, first["step"]),
    ]
    assert all(row["planned_epochs"] == 1 and row["max_steps"] is None for row in runs)
    assert [row["epochs"] for row in runs] == [1, 2]
    assert runs[0]["total_steps"] == first["step"]
    assert runs[0]["max_grad_norm"] == config.max_grad_norm

    assert [row["step"] for row in steps] == list(range(1, second["step"] + 1))

    printed = [line for line in capsys.readouterr().out.splitlines() if " step=" in line]

    assert len(printed) == len(steps)

    for row, line in zip(steps, printed):
        assert line.startswith(f"epoch={row['epoch']} step={row['step']} loss={row['loss']:.4f} ")
        assert f"lr={row['lr']:.2e} grad_norm={row['grad_norm']:.3f} " in line
        assert row["tokens"] > 0 and row["targets"] > 0 and row["seconds"] >= row["wait_seconds"] >= 0.0

    # Шаги первой эпохи — ровно её micro-batch'и: tiny копит по
    # одному micro-batch на шаг, и цели есть в каждом.
    tokens = [
        sum(size.n_tokens + size.profile_n_tokens for size in batch)
        for batch in micro_batches(Source("train").sizes(), config.token_budget)
    ]

    assert config.grad_accum_steps == 1 and first["step"] == len(tokens)
    assert [row["tokens"] for row in steps if row["epoch"] == 1] == tokens


def test_history_in_the_checkpoint_stays_free_of_wall_time(stage):
    """
    Время стены в чекпойнт не попадает: иначе продолжение не могло
    бы совпасть с непрерывным прогоном (test_checkpoint_resume).
    """

    settle(stage, train_people=many())

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)

    assert set(state["history"][0]) == {"epoch", "step", "learning_rate", "train", "val", "val_detail"}
    assert np.isfinite(state["history"][0]["val_detail"]["nll"])
