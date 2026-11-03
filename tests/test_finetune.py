from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from src.downstream import finetune, settings, tasks
from src.downstream.finetune import ZERO_MASK, FinetuneConfig, fit, predict
from src.downstream.lora import ChurnHead, delta_ratio, inject, load_adapters, trainable
from src.mlm.inputs import IGNORE, Source, cost
from src.mlm.model import pack
from src.mlm.settings import ConfigError, best_checkpoint_path
from src.mlm.train import TrainingError, file_digest, train
from src.preprocessing.artifacts import read_json
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from tests import world
from tests.test_downstream import chain, raw_groups, same, write_churn_reports
from tests.test_profile_state import EARLY, QUIET_SNAPSHOT, RAW_CLIENT
from tests.test_scheduler import many
from tests.test_training_math import every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Эталон не меняется ни на бит; шаг делается, когда накоплено
# достаточно клиентов; нечисло останавливает обучение до шага;
# одинаковый seed на CPU даёт одинаковый результат; голова считает
# в fp32 даже под bf16-autocast.
#
# Данные и команда: отложенная часть — та же, что у CatBoost; вход
# train — набор 05 на T задачи без целей, вход val — сборка на T val;
# клиент задачи без входа — отказ, а не пропуск; чужие каталоги и
# заплатки чужого эталона не перезаписываются и не читаются.
# ============================================================


CPU = torch.device("cpu")


def people(count: int = 6):
    return [item.client for item in many(count)]


def labels_of(clients) -> dict[str, int]:
    # Поровну 0 и 1: log-loss и PR-AUC определены.
    return {c.client_id: i % 2 for i, c in enumerate(clients)}


def tiny_config(**overrides) -> FinetuneConfig:
    base = dict(rank=2, alpha=4.0, learning_rate=1e-2, clients_per_step=2, token_budget=10_000, max_epochs=2, patience=2)
    base.update(overrides)
    return FinetuneConfig(**base)


def build(dropout: float = 0.0, seed: int = 3):
    model = world.model(dropout=dropout)
    inject(model, 2, 4.0, seed)
    torch.manual_seed(seed)
    return model, ChurnHead(world.DIM, 0.5)


def run(model, head, clients, config, seed=11, telemetry=None):
    return fit(model, head, lambda epoch: clients, lambda: clients, labels_of(clients), config, seed, CPU, telemetry)


def test_config_is_strict():

    with pytest.raises(ConfigError):
        FinetuneConfig.from_dict({"rnak": 8})
    with pytest.raises(ConfigError):
        FinetuneConfig.from_dict({"rank": 0})

    config = FinetuneConfig.load(None)
    assert config.as_dict() == {
        "rank": 8, "alpha": 8.0, "learning_rate": 1e-3, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "clients_per_step": 32, "token_budget": 48_000, "max_epochs": 10, "patience": 3,
    }
    assert FinetuneConfig.from_dict(config.as_dict()) == config


def test_zero_mask_hides_nothing():

    assert ZERO_MASK.value_probability == ZERO_MASK.event_probability == 0.0
    assert ZERO_MASK.key_probability == ZERO_MASK.unknown_probability == 0.0


def test_fit_learns_only_adapters_and_keeps_the_reference(tmp_path):

    clients = people()
    model, head = build(dropout=0.1)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not n.endswith((".A", ".B"))}
    adapters = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}
    telemetry = tmp_path / "t.jsonl"

    result = run(model, head, clients, tiny_config(), telemetry=telemetry)

    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    assert any(not torch.equal(p, adapters[n]) for n, p in model.named_parameters() if n in adapters)
    assert len(result.history) == 2
    assert [r["epoch"] for r in result.history] == [1, 2]
    lines = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert len(lines) == 2 and all(line["kind"] == "epoch" for line in lines)
    assert result.best_epoch in (1, 2)
    # held_scores - прогнозы лучшей эпохи; метрики из них совпадают с записью той же эпохи.
    assert set(result.held_scores) == {c.client_id for c in clients}
    assert all(0.0 < p < 1.0 for p in result.held_scores.values())
    ids = sorted(result.held_scores)
    y, score = [labels_of(clients)[i] for i in ids], [result.held_scores[i] for i in ids]
    best = result.history[result.best_epoch - 1]
    assert float(log_loss(y, score, labels=[0, 1])) == best["held_log_loss"]
    assert float(average_precision_score(y, score)) == best["held_pr_auc"]
    assert delta_ratio(model) > 0


def test_fit_is_reproducible_on_cpu():

    clients = people()
    results = []
    for _ in range(2):
        model, head = build(dropout=0.1)
        results.append(run(model, head, clients, tiny_config()))

    first, second = results
    assert first.state.keys() == second.state.keys()
    assert all(torch.equal(first.state[k], second.state[k]) for k in first.state)
    assert first.held_scores == second.held_scores  # прогнозы лучшей эпохи
    assert first.best_epoch == second.best_epoch


@pytest.mark.parametrize(
    "per_step, steps",
    [(1, 3), (3, 2), (4, 2), (5, 1), (100, 1)],
)
def test_step_waits_for_enough_clients(monkeypatch, per_step, steps):
    """
    Шесть одинаковых клиентов по два в micro-batch'е: три прохода.
    Шаг делается, когда накоплено не меньше clients_per_step клиентов,
    и в конце эпохи, если что-то осталось.
    """

    clients = people(6)
    budget = 2 * cost(clients[0])
    model, head = build()
    calls = []
    original = torch.optim.AdamW.step
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1])

    run(model, head, clients, tiny_config(clients_per_step=per_step, token_budget=budget, max_epochs=1))

    assert len(calls) == steps


def test_nonfinite_loss_stops_before_the_step():

    clients = people()
    model, head = build()
    head.linear.weight.data.fill_(float("nan"))
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}

    with pytest.raises(TrainingError):
        run(model, head, clients, tiny_config())

    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before)


def test_patience_stops_training(monkeypatch):

    clients = people()
    model, head = build()
    calls = []

    def worse(model, head, held, device, token_budget):
        calls.append(1)
        # Постоянный прогноз уходит от 0.5, а метки поровну: log-loss растёт.
        return {c.client_id: 0.5 + 0.05 * len(calls) for c in held}

    monkeypatch.setattr(finetune, "predict", worse)

    result = run(model, head, clients, tiny_config(max_epochs=10, patience=2))

    assert len(result.history) == 3
    assert result.best_epoch == 1
    # Прогнозы - именно лучшей (первой) эпохи, не последней.
    assert result.held_scores == {c.client_id: 0.55 for c in clients}


def test_head_runs_in_fp32_under_autocast(monkeypatch):

    clients = people()
    model, head = build()
    head.linear.weight.data = torch.randn(head.linear.weight.shape, generator=torch.Generator().manual_seed(2))
    monkeypatch.setattr(finetune, "autocast", lambda device: torch.autocast("cpu", dtype=torch.bfloat16))

    # Эталон отдаёт [USR] в bf16, как это бывает на CUDA под autocast.
    inner = model.client_embeddings
    monkeypatch.setattr(model, "client_embeddings", lambda data: inner(data).bfloat16())

    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        usr = model.client_embeddings(pack(clients, CPU))
        in_bf16 = torch.sigmoid(head.linear(usr))
    assert usr.dtype == torch.bfloat16
    expected = torch.sigmoid(head(usr.float()))
    assert not torch.equal(expected, in_bf16.float())

    got = predict(model, head, clients, CPU, 10**6)

    assert expected.dtype == torch.float32
    assert list(got.values()) == expected.tolist()


# ============================================================
# ДАННЫЕ
# ============================================================


TASK = "churn_active90"


def test_split_follows_the_file_order(stage):
    """
    Отложенная часть — StratifiedShuffleSplit по меткам в порядке строк
    отчёта, как у churn.train.fit_task: по строкам, отсортированным по
    client_id, она была бы другой.
    """

    # 50 строк вразброс, каждая пятая ушла: доля 0.2 делится нацело.
    numbers = np.random.default_rng(4).permutation(50)
    rows = [(f"t{number:02d}", int(number % 5 == 0), True) for number in numbers]
    raw_groups("train", "val")
    write_churn_reports({"train": rows, "val": [("v1", 1, True)]})

    order = tasks.churn_order(TASK, "train")

    assert list(order) == [client for client, _, _ in rows]
    assert list(order) != sorted(order)

    y = tasks.churn_rows(TASK, "train").loc[order, "churn"].to_numpy()
    fit_rows, held_rows = finetune.split(y)

    expected = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    fit_expected, held_expected = next(expected.split(np.zeros(len(y)), y))
    assert np.array_equal(fit_rows, np.sort(fit_expected))
    assert np.array_equal(held_rows, np.sort(held_expected))

    assert not set(fit_rows) & set(held_rows) and len(fit_rows) + len(held_rows) == len(y)
    assert len(held_rows) == 10
    assert y[held_rows].mean() == y[fit_rows].mean() == y.mean() == 0.2

    by_id = tasks.churn_rows(TASK, "train")
    _, held_by_id = finetune.split(by_id["churn"].to_numpy())
    assert set(order[held_rows]) != set(by_id.index[held_by_id])

    write_churn_reports({"train": rows + rows[:1], "val": [("v1", 1, True)]})

    with pytest.raises(ValueError, match=rows[0][0]):
        tasks.churn_order(TASK, "train")


def test_preflight_refuses_other_cutoff_anchor_or_missing_client(stage, monkeypatch):
    """
    Вход train с нулевой маской — вход на T задачи, только если набор
    собран на этот T и время в нём отсчитано от T. Клиент задачи,
    которого в наборе нет, — отказ с его именем, а не пропуск.
    """

    settle(stage, train_people=many())
    monkeypatch.setattr(finetune, "cutoff", lambda group: world.CUTOFF)
    ids = pd.Index(["c3", "c0"])

    finetune.preflight(ids)

    with pytest.raises(ValueError, match="ghost"):
        finetune.preflight(ids.append(pd.Index(["ghost"])))

    monkeypatch.setattr(finetune, "cutoff", lambda group: world.CUTOFF + timedelta(days=1))
    with pytest.raises(ValueError, match="events_cutoff"):
        finetune.preflight(ids)

    monkeypatch.setattr(finetune, "cutoff", lambda group: world.CUTOFF)
    world.write_samples("train", [many()], anchor="last_event")
    with pytest.raises(ValueError, match="time_anchor"):
        finetune.preflight(ids)

    world.write_samples("train", [many(), many()[:1]])
    with pytest.raises(ValueError, match="c0"):
        finetune.preflight(ids)


def test_train_clients_follow_the_epoch_order_without_targets(stage):
    """
    Группы строк эпохи идут в перестановке от (seed, эпоха), отложенная
    часть — в порядке файла; клиенты целиком, без единой цели. Клиент
    без событий или неполный проход — отказ с именем клиента.
    """

    made = many(6)
    world.write_samples("train", [made[0:2], made[2:4], made[4:6]])
    every = {item.client.client_id for item in made}

    def ids(clients) -> list[str]:
        return [client.client_id for client in clients]

    assert ids(finetune.train_clients(every, None, 0)) == [f"c{n}" for n in range(6)]

    for epoch in (1, 2):
        groups = np.random.default_rng([3, epoch]).permutation(3)
        assert ids(finetune.train_clients(every, (3, epoch), 0)) == [f"c{2 * g + k}" for g in groups for k in (0, 1)]
    assert ids(finetune.train_clients(every, (3, 1), 0)) != ids(finetune.train_clients(every, (3, 2), 0))

    assert ids(finetune.train_clients({"c4", "c1"}, None, 0)) == ["c1", "c4"]

    source = {item.client.client_id: item.value_ids_source for item in made}
    for client in finetune.train_clients(every, (3, 1), 0):
        assert (client.labels == IGNORE).all()
        assert np.array_equal(client.value_ids, source[client.client_id])

    with pytest.raises(ValueError, match="ghost"):
        list(finetune.train_clients({"c0", "ghost"}, None, 0))

    empty = world.make("c-empty", [], [(world.KEY_A, [20])])
    world.write_samples("train", [made + [empty]])

    with pytest.raises(ValueError, match="c-empty: у клиента задачи нет событий"):
        list(finetune.train_clients({"c0", "c-empty"}, None, 0))


def test_val_clients_are_built_at_the_val_T(stage, monkeypatch):
    """
    Вход val — сборка at_cutoff на T val, а не набор 05/val: тот держит
    события до конца окна, позже T. Клиент задачи без входа на T или
    без событий до T — отказ с его именем.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff

    chain(stage, EARLY, QUIET_SNAPSHOT)
    moment = settings.cutoff("val")

    (client,) = list(finetune.val_clients("val", [RAW_CLIENT], 0))
    (stored,) = list(Source("val", ZERO_MASK).clients())

    same(client, ClientsAtCutoff("val", moment).client(RAW_CLIENT))
    assert client.n_events == 2 and stored.n_events == len(EARLY)
    assert all(event_time < moment for event_time in client.event_time)

    with pytest.raises(ValueError, match="ghost"):
        list(finetune.val_clients("val", [RAW_CLIENT, "ghost"], 0))

    first = min(datetime.fromisoformat(event["event_time"]) for event in EARLY)
    monkeypatch.setattr(finetune, "cutoff", lambda group: first - timedelta(days=1))

    with pytest.raises(ValueError, match=f"{RAW_CLIENT}: у клиента задачи нет событий до T val"):
        list(finetune.val_clients("val", [RAW_CLIENT], 0))


# ============================================================
# КОМАНДА
# ============================================================


# Порядок строк train в отчёте — не по client_id.
ROWS = {
    "train": [("c5", 1, True), ("c2", 0, True), ("c0", 1, True), ("c4", 0, True), ("c1", 1, True), ("c3", 0, True)],
    "val": [("v-thin", 1, True), ("v-long", 0, True), ("v-one", 1, True), ("v-quiet", 0, True), ("v-bare", 0, True)],
    "test": [("x-one", 0, True), ("x-long", 1, True), ("x-thin", 0, True)],
}


def stored_clients(group: str, ids: list[str], workers: int):
    """
    Вход на T группы мира: этапа 02 у мира нет, а его набор 05 собран
    ровно на world.CUTOFF.
    """

    wanted = set(ids)
    return (client for client in Source(group, ZERO_MASK).clients() if client.client_id in wanted)


def finetune_world(stage, monkeypatch, tmp_path, groups=("train", "val")) -> list[str]:
    """
    Мир, эталон, обученный на его train, и отчёты churn на T мира.
    Возвращает аргументы команды без --name.
    """

    settle(stage, train_people=many())
    if "test" in groups:
        world.write_samples("test", [world.population("x")])

    train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    raw_groups(*groups)
    write_churn_reports({group: ROWS[group] for group in groups}, moments={group: world.CUTOFF for group in groups})

    monkeypatch.setattr(tasks, "cutoff", lambda group: world.CUTOFF)
    monkeypatch.setattr(finetune, "cutoff", lambda group: world.CUTOFF)
    monkeypatch.setattr(finetune, "val_clients", stored_clients)
    # Этапа 02 у мира нет: проверка сборщика на T — заглушка; настоящая
    # проверена в test_val_side_is_checked_before_training.
    monkeypatch.setattr(finetune, "check_cutoff_inputs", lambda groups_rows: None)

    config = tmp_path / "ft.json"
    config.write_text(json.dumps(tiny_config().as_dict()))

    return ["--checkpoint", str(best_checkpoint_path()), "--device", "cpu", "--config", str(config), "--workers", "0"]


def record_inputs(monkeypatch) -> tuple[list[str], list[str]]:
    """
    Группы, с которыми вызвана val_clients, и группы, для которых
    создан Source из 05_dataset: val и test обязаны идти только через
    val_clients, а Source — только train.
    """

    via_val, via_source = [], []
    stub, real = finetune.val_clients, finetune.Source

    def recording_val(group, ids, workers):
        via_val.append(group)
        return stub(group, ids, workers)

    def recording_source(group, *rest, **kwargs):
        via_source.append(group)
        return real(group, *rest, **kwargs)

    monkeypatch.setattr(finetune, "val_clients", recording_val)
    monkeypatch.setattr(finetune, "Source", recording_source)

    return via_val, via_source


def cli(*argv: str) -> int:

    with pytest.raises(SystemExit) as stop:
        finetune.main(list(argv))

    return stop.value.code


def snapshot(root: Path) -> dict[str, bytes | None]:
    """
    Всё, что лежит под root: файлы с байтами, каталоги — None.
    """

    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None for path in root.rglob("*")}


def test_finetune_end_to_end_on_cpu(stage, monkeypatch, tmp_path):
    """
    Прогон пишет заплатки, телеметрию, прогнозы только val и meta;
    порог — максимум F1 на отложенной части, прогнозы — от лучшего
    состояния; повтор с тем же seed даёт те же прогнозы.
    """

    args = finetune_world(stage, monkeypatch, tmp_path)
    checkpoint = Path(args[1])
    real_fit = finetune.fit
    via_val, via_source = record_inputs(monkeypatch)

    def drifting_fit(model, head, *rest):
        # Поздние эпохи уводят заплатки от лучшего состояния: прогнозы
        # обязаны считаться лучшим состоянием, а не последним.
        result = real_fit(model, head, *rest)
        generator = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for p in trainable(model, head):
                p.add_(torch.randn(p.shape, generator=generator))
        return result

    monkeypatch.setattr(finetune, "fit", drifting_fit)

    assert cli(*args, "--name", "ft") == EXIT_OK

    # Вход val — только через val_clients; 05_dataset читается лишь для train.
    assert via_val == ["val"]
    assert via_source and set(via_source) == {"train"}

    run, out = settings.run_dir("ft"), settings.downstream_dir("ft")
    assert run == stage / "runs" / "ft"

    lora = torch.load(run / "lora.pt", map_location="cpu", weights_only=True)
    assert set(lora) == {"kind", "adapters", "config", "seed", "reference", "held", "best_epoch", "threshold"}
    assert lora["kind"] == "lora" and lora["seed"] == 0 and lora["config"] == tiny_config().as_dict()
    epoch = torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"]
    assert lora["reference"] == {"checkpoint": str(checkpoint), "sha256": file_digest(checkpoint), "epoch": int(epoch)}

    order = np.array([client for client, _, _ in ROWS["train"]])
    labels = {client: churn for client, churn, _ in ROWS["train"]}
    _, held_rows = finetune.split(np.array([labels[client] for client in order]))
    assert lora["held"] == sorted(order[held_rows])
    # По строкам, отсортированным по client_id, отложенными были бы другие.
    by_id = np.sort(order)
    assert sorted(by_id[finetune.split(np.array([labels[client] for client in by_id]))[1]]) != lora["held"]

    telemetry = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in telemetry] == list(range(1, len(telemetry) + 1))
    assert 1 <= lora["best_epoch"] <= len(telemetry) <= 2

    # Лучшее состояние, поднятое заново, даёт тот же порог и те же прогнозы.
    from src.downstream.embed import raw_record, trained_model
    from src.downstream.probe import threshold_max_f1

    model, _, _ = trained_model(str(checkpoint), CPU)
    inject(model, 2, 4.0, 0)
    head = ChurnHead(world.DIM, 0.5)
    load_adapters(model, head, lora["adapters"])

    held = predict(model, head, finetune.train_clients(set(lora["held"]), None, 0), CPU, 10_000)
    y_held = np.array([labels[client] for client in lora["held"]])
    assert lora["threshold"] == threshold_max_f1(y_held, np.array([held[client] for client in lora["held"]]))

    predictions = pd.read_parquet(out / "predictions.parquet")
    val = sorted(ROWS["val"])
    assert list(predictions.columns) == ["task", "set", "group", "client_id", "y", "score"]
    assert set(predictions["task"]) == {TASK} and set(predictions["set"]) == {"lora"}
    assert set(predictions["group"]) == {"val"}
    assert list(predictions["client_id"]) == [client for client, _, _ in val]
    assert list(predictions["y"]) == [churn for _, churn, _ in val]
    assert predictions["score"].dtype == np.float64

    scored = predict(model, head, stored_clients("val", list(predictions["client_id"]), 0), CPU, 10_000)
    assert predictions["score"].tolist() == [scored[client] for client in predictions["client_id"]]

    meta = read_json(out / "meta.json")
    assert meta["kind"] == "lora" and meta["name"] == "ft" and meta["final_test"] is False
    assert meta["reference"] == lora["reference"]
    assert meta["threshold"] == lora["threshold"] and meta["best_epoch"] == lora["best_epoch"]
    assert meta["seed"] == 0 and meta["config"] == tiny_config().as_dict()
    assert meta["trainable"] == sum(p.numel() for p in trainable(model, head))
    assert meta["groups"] == {group: {"cutoff": world.CUTOFF.isoformat(), **raw_record(group)} for group in ("train", "val")}
    assert meta["sources"] == {group: tasks.churn_sources(group) for group in ("train", "val")}

    # Повтор: те же прогнозы, телеметрия пересоздана, а не дописана.
    assert cli(*args, "--name", "ft") == EXIT_OK

    pd.testing.assert_frame_equal(pd.read_parquet(out / "predictions.parquet"), predictions, check_exact=True)
    assert len((run / "telemetry.jsonl").read_text().splitlines()) == len(telemetry)


@pytest.mark.parametrize("foreign", ["pretraining", "vectors", "checkpoint_as_lora", "vectors_beside_lora"])
def test_name_of_foreign_outputs_is_refused(stage, monkeypatch, tmp_path, capsys, foreign):
    """
    --name не занимает каталог предобучения (data/runs/<имя> с
    telemetry.jsonl и чекпойнтом), тег векторов embed — даже рядом с
    настоящими заплатками — и lora.pt, который не заплатки: отказ, и ни
    один файл не тронут и не создан.
    """

    # С группой test: --final-test дошёл бы до записи, не будь защиты.
    args = finetune_world(stage, monkeypatch, tmp_path, groups=("train", "val", "test"))
    checkpoint = Path(args[1])

    def vectors() -> None:
        out = settings.downstream_dir("ref")
        out.mkdir(parents=True)
        (out / "meta.json").write_text(json.dumps({"tag": "ref", "checkpoint": str(checkpoint), "groups": {}}))
        (out / "train.parquet").write_bytes(b"vectors")

    if foreign == "pretraining":
        run = settings.run_dir("ref")
        run.mkdir(parents=True)
        (run / "telemetry.jsonl").write_text(json.dumps({"kind": "epoch", "epoch": 1}) + "\n")
        shutil.copy(checkpoint, run / "best_checkpoint.pt")
    elif foreign == "vectors":
        vectors()
    elif foreign == "checkpoint_as_lora":
        run = settings.run_dir("ref")
        run.mkdir(parents=True)
        shutil.copy(checkpoint, run / "lora.pt")
    else:
        assert cli(*args, "--name", "ref") == EXIT_OK
        shutil.rmtree(settings.downstream_dir("ref"))
        vectors()

    before = snapshot(stage)
    capsys.readouterr()

    assert cli(*args, "--name", "ref") == EXIT_BLOCKED

    # Каталог без lora.pt бывает и прерванным прогоном LoRA: сообщение
    # называет каталог и говорит, что с ним делать.
    if foreign == "pretraining":
        message = capsys.readouterr().out
        assert str(settings.run_dir("ref")) in message and "нет lora.pt" in message
        assert "удалите каталог вручную" in message

    assert cli(*args, "--name", "ref", "--final-test") == EXIT_BLOCKED

    assert snapshot(stage) == before


def test_final_test_reuses_the_trained_adapters(stage, monkeypatch, tmp_path):
    """
    --final-test не обучает: те же заплатки дают те же прогнозы val и
    прогнозы test. Без lora.pt или с заплатками другого эталона — отказ.
    """

    args = finetune_world(stage, monkeypatch, tmp_path, groups=("train", "val", "test"))

    assert cli(*args, "--name", "ft") == EXIT_OK

    run, out = settings.run_dir("ft"), settings.downstream_dir("ft")
    trained = pd.read_parquet(out / "predictions.parquet")
    kept = {name: (run / name).read_bytes() for name in ("lora.pt", "telemetry.jsonl")}
    lora = torch.load(run / "lora.pt", map_location="cpu", weights_only=True)

    def no_fit(*args, **kwargs):
        pytest.fail("--final-test не обучает")

    monkeypatch.setattr(finetune, "fit", no_fit)
    via_val, via_source = record_inputs(monkeypatch)

    assert cli(*args, "--name", "ft", "--final-test") == EXIT_OK

    # Val и test — по порядку и только через val_clients; train в
    # --final-test не читается вовсе.
    assert via_val == ["val", "test"] and via_source == []

    final = pd.read_parquet(out / "predictions.parquet")
    assert list(final["group"].unique()) == ["val", "test"]
    pd.testing.assert_frame_equal(final[final["group"] == "val"].reset_index(drop=True), trained, check_exact=True)

    test = final[final["group"] == "test"]
    assert list(test["client_id"]) == sorted(client for client, _, _ in ROWS["test"])
    assert list(test["y"]) == [churn for _, churn, _ in sorted(ROWS["test"])]
    assert test["score"].between(0.0, 1.0, inclusive="neither").all()

    assert {name: (run / name).read_bytes() for name in kept} == kept

    meta = read_json(out / "meta.json")
    assert meta["final_test"] is True and set(meta["groups"]) == set(meta["sources"]) == {"train", "val", "test"}
    assert meta["reference"] == lora["reference"] and meta["threshold"] == lora["threshold"]
    assert meta["best_epoch"] == lora["best_epoch"] and meta["seed"] == lora["seed"]

    before = snapshot(stage)

    assert cli(*args, "--name", "other", "--final-test") == EXIT_BLOCKED
    assert snapshot(stage) == before

    lora["reference"]["sha256"] = "0" * 64
    torch.save(lora, run / "lora.pt")
    before = snapshot(stage)

    assert cli(*args, "--name", "ft", "--final-test") == EXIT_BLOCKED
    assert snapshot(stage) == before


def test_val_side_is_checked_before_training(stage, monkeypatch, tmp_path, capsys):
    """
    Сборщик at_cutoff на T каждой оцениваемой группы проверяется до
    обучения: клиент задачи, которого он не знает, — отказ с группой и
    id до первого вызова fit, а в --final-test — до inject.
    """

    real_check = finetune.check_cutoff_inputs
    args = finetune_world(stage, monkeypatch, tmp_path, groups=("train", "val", "test"))
    built = []

    class Builder:
        """Сборщик, не знающий v-bare и x-one."""

        def __init__(self, group, moment):
            built.append((group, moment))
            self.client_ids = [c for c, _, _ in ROWS[group] if c not in {"v-bare", "x-one"}]

    def fail(what):
        def refuse(*args, **kwargs):
            pytest.fail(what)
        return refuse

    with monkeypatch.context() as patch:
        patch.setattr(finetune, "check_cutoff_inputs", real_check)
        patch.setattr(finetune, "ClientsAtCutoff", Builder)

        known = pd.Index([c for c, _, _ in ROWS["val"] if c != "v-bare"])
        finetune.check_cutoff_inputs({"val": known})

        with pytest.raises(ValueError, match="val.*v-bare"):
            finetune.check_cutoff_inputs({"val": pd.Index([c for c, _, _ in ROWS["val"]])})

        built.clear()
        patch.setattr(finetune, "fit", fail("fit вызван до проверки сборщика val"))
        capsys.readouterr()
        before = snapshot(stage)

        assert cli(*args, "--name", "ft") == EXIT_BLOCKED
        assert "val" in capsys.readouterr().out and ("val", world.CUTOFF) in built
        assert snapshot(stage) == before

    # --final-test: заплатки есть, сборщик не знает v-bare и x-one — отказ до inject.
    assert cli(*args, "--name", "ft") == EXIT_OK

    with monkeypatch.context() as patch:
        patch.setattr(finetune, "check_cutoff_inputs", real_check)
        patch.setattr(finetune, "ClientsAtCutoff", Builder)
        patch.setattr(finetune, "inject", fail("inject вызван до проверки сборщика"))
        capsys.readouterr()

        assert cli(*args, "--name", "ft", "--final-test") == EXIT_BLOCKED
        assert "v-bare" in capsys.readouterr().out


def test_a_crash_while_publishing_never_leaves_the_old_meta(stage, monkeypatch, tmp_path, capsys):
    """
    Прежняя meta удаляется до записи прогнозов: сбой на записи прогнозов
    или между прогнозами и meta оставляет прогнозы без meta, но никогда —
    новые прогнозы рядом со старой meta. И в обучении, и в --final-test.
    """

    args = finetune_world(stage, monkeypatch, tmp_path, groups=("train", "val", "test"))
    out = settings.downstream_dir("ft")
    meta, predictions = out / "meta.json", out / "predictions.parquet"

    def crash(*args, **kwargs):
        raise RuntimeError("сбой записи")

    def crashed(where, what, *flags: str) -> None:
        with monkeypatch.context() as patch:
            patch.setattr(where, what, crash)
            with pytest.raises(RuntimeError, match="сбой записи"):
                finetune.main([*args, "--name", "ft", *flags])

    # Обучение: сбой между прогнозами и meta.
    assert cli(*args, "--name", "ft") == EXIT_OK
    assert read_json(meta)["kind"] == "lora"
    written = predictions.stat().st_ino

    crashed(finetune, "write_json")

    assert not meta.exists() and predictions.stat().st_ino != written

    # Прогнозы без meta — не законченный прогон: имя занято, пока каталог
    # не удалён вручную.
    capsys.readouterr()
    assert cli(*args, "--name", "ft") == EXIT_BLOCKED
    assert str(out) in capsys.readouterr().out
    shutil.rmtree(out)

    # --final-test: сбой на самой записи прогнозов — meta уже нет.
    assert cli(*args, "--name", "ft") == EXIT_OK
    assert read_json(meta)["kind"] == "lora"
    kept = predictions.read_bytes()

    crashed(pd.DataFrame, "to_parquet", "--final-test")

    assert not meta.exists() and predictions.read_bytes() == kept

    # --final-test: сбой между прогнозами val и test и meta.
    shutil.rmtree(out)
    assert cli(*args, "--name", "ft") == EXIT_OK
    assert read_json(meta)["kind"] == "lora"

    crashed(finetune, "write_json", "--final-test")

    assert not meta.exists()
    assert list(pd.read_parquet(predictions)["group"].unique()) == ["val", "test"]


# ============================================================
# CUDA
# ============================================================


@pytest.mark.cuda
@pytest.mark.skipif(not world.flash_ready(), reason="нужны CUDA и библиотека flash-attn")
def test_bf16_finetune_step_on_cuda(monkeypatch):

    cuda = torch.device("cuda")
    clients = people()
    model = world.model(attention="flash")
    inject(model, 2, 4.0, 3)
    torch.manual_seed(3)
    head = ChurnHead(world.DIM, 0.5)
    model.to(cuda)
    head.to(cuda)

    def is_patch(name: str) -> bool:
        return name.endswith((".A", ".B"))

    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not is_patch(n)}
    learning = {id(p) for p in trainable(model, head)}
    assert learning == {id(p) for n, p in model.named_parameters() if is_patch(n)} | {id(head.linear.weight), id(head.linear.bias)}
    assert all(p.dtype == torch.float32 for p in (*model.parameters(), *head.parameters()))

    # Градиенты видны только до шага: после него fit их сбрасывает.
    seen: list[set[int]] = []
    real_step = finetune.step

    def spy(optimizer, params, count, norm):
        seen.append({id(p) for p in (*model.parameters(), *head.parameters()) if p.grad is not None})
        real_step(optimizer, params, count, norm)

    monkeypatch.setattr(finetune, "step", spy)

    result = fit(model, head, lambda epoch: clients, lambda: clients, labels_of(clients), tiny_config(), 11, cuda, None)

    assert seen and all(ids == learning for ids in seen)
    assert all(p.dtype == torch.float32 for p in (*model.parameters(), *head.parameters()))
    assert all(np.isfinite(r["train_loss"]) and np.isfinite(r["held_log_loss"]) for r in result.history)
    assert all(0.0 < p < 1.0 for p in result.held_scores.values())
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
