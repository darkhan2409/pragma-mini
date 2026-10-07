from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.downstream.probe import paired
from src.downstream.settings import cutoff, downstream_dir

from tests import world
from tests.test_profile_state import (
    AFTER,
    BUSY_SNAPSHOT,
    EARLY,
    QUIET_SNAPSHOT,
    RAW_CLIENT,
    prepare,
    write_profile_vocab,
    write_raw,
)


# ============================================================
# ИДЕЯ
# ============================================================
#
# Вектор клиента на момент T честен, только если вход на T собран
# из прошлого и собран ТАК ЖЕ, как вход обучения. Поэтому:
#
#   - на конце окна группы вход в памяти обязан побитно совпасть с
#     тем, что обучение читает из набора 05, — второй реализации
#     цепочки здесь нет;
#   - всё, что случилось после T, — события, переезд, продукт, —
#     вход на T не меняет ни на бит, а анкета откатывается на T;
#   - строки задачи churn_active90 — строки churn-бейзлайна
#     текущей выгрузки, и все три сценария оцениваются на одних и
#     тех же клиентах;
#   - пока идут эксперименты, test не считается нигде: ни векторы,
#     ни метрики, ни сравнения. Только явный --final-test.
# ============================================================


NOTHING = dict(value_probability=0.0, event_probability=0.0, key_probability=0.0,
               unknown_probability=0.0)

FIELDS = (
    "key_ids", "value_ids", "positions", "labels", "event_starts", "event_lengths",
    "event_time_log", "calendar", "profile_key_ids", "profile_value_ids",
    "profile_positions", "profile_time_log",
)


def chain(stage, tape: list[dict], snapshot: dict, anchor: str | None = None) -> None:
    """
    Выгрузка → 02 → 04 → 05 для train и val: теми же функциями
    этапов, что и в бою. train нужен ради отбора истории и точки
    отсчёта времени, на которых «училась модель»
    (05_dataset/train/meta.json). Без anchor набор собирается со
    своим умолчанием, как в бою.
    """

    from dataclasses import replace

    from src.dataset.build import build_group as build_dataset
    from src.dataset.settings import DatasetConfig
    from src.tokenization.finalvocab import FrozenArtifacts
    from src.tokenization.settings import TokenizerConfig
    from src.tokenization.transform import encode_group

    write_profile_vocab(stage, event_types=("purchase", "profile_change", "product_opened"))

    artifacts = FrozenArtifacts.load()

    config = DatasetConfig.load(None)

    if anchor is not None:
        config = replace(config, time_anchor=anchor)

    for group in ("train", "val"):
        prepare(stage, tape, snapshot, group=group)
        encode_group(artifacts, group, TokenizerConfig.load(None))
        build_dataset(artifacts, group, config)


def same(left, right) -> None:

    for name in FIELDS:
        one, other = getattr(left, name), getattr(right, name)
        assert one.dtype == other.dtype and one.shape == other.shape, name
        assert np.array_equal(one, other), name

    assert left.reason == right.reason


# ============================================================
# ВХОД НА T
# ============================================================


@pytest.mark.parametrize("anchor", ["last_event", "cutoff"])
def test_input_at_the_group_cutoff_is_what_training_reads(stage, anchor: str):
    """
    T = конец окна val: вход в памяти побитно равен клиенту, которого
    модель читает из набора 05 без масок, — при любой точке отсчёта
    времени набора.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff
    from src.masking.settings import MaskingConfig
    from src.mlm.inputs import Source
    from src.preprocessing.settings import PreprocessingConfig

    chain(stage, EARLY, QUIET_SNAPSHOT, anchor)

    (stored,) = list(Source("val", masking=MaskingConfig(**NOTHING)).clients())

    end = PreprocessingConfig.load(None).windows["val"].final_cutoff

    built = ClientsAtCutoff("val", end).client(RAW_CLIENT)

    assert built.n_events == len(EARLY) and built.n_events > 0
    same(built, stored)


@pytest.mark.parametrize("anchor", ["last_event", "cutoff"])
def test_the_future_after_the_cutoff_does_not_change_the_input(stage, anchor: str):
    """
    Две выгрузки с общим прошлым до T: во второй после T ещё продукт
    и переезд, и снимок анкеты уже знает о переезде. Вход на T у них
    побитно один.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff
    from src.temporal.position import log_age

    moment = cutoff("val")

    chain(stage, EARLY, QUIET_SNAPSHOT, anchor)
    quiet = ClientsAtCutoff("val", moment).client(RAW_CLIENT)

    chain(stage, EARLY + AFTER, BUSY_SNAPSHOT, anchor)
    busy = ClientsAtCutoff("val", moment).client(RAW_CLIENT)

    same(quiet, busy)

    # T раньше конца окна: переезд 10 марта и продукт 20 марта в
    # ленту на T не попадают.
    assert quiet.n_events == 2
    assert all(event_time < moment for event_time in quiet.event_time)

    # Последнее событие стоит в точке отсчёта или на своей давности
    # до T — а не до события после T, которого во входе нет.
    age = 0 if anchor == "last_event" else int((moment - quiet.event_time[-1]) / timedelta(microseconds=1))
    assert quiet.event_time_log[-1] == np.float32(log_age(np.array([age], dtype=np.int64))[0])


def test_the_input_refuses_anything_at_or_after_T(stage, monkeypatch):
    """
    Защита в самом сборщике: если чтение истории вернёт событие не
    раньше T, вход на T не собирается, а не собирается с будущим.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff, CutoffError

    moment = cutoff("val")

    chain(stage, EARLY + AFTER, BUSY_SNAPSHOT)

    builder = ClientsAtCutoff("val", moment)

    read = builder._source.history
    monkeypatch.setattr(builder._source, "history", lambda client_id, at: read(client_id, at + timedelta(days=30)))

    with pytest.raises(CutoffError, match="не раньше T"):
        builder.client(RAW_CLIENT)


def test_a_cutoff_outside_the_window_is_refused(stage):

    from src.downstream.at_cutoff import CutoffError, window_at
    from src.preprocessing.settings import PreprocessingConfig

    window = PreprocessingConfig.load(None).windows["val"]

    with pytest.raises(CutoffError, match="вне окна"):
        window_at("val", window.final_cutoff + timedelta(seconds=1))

    with pytest.raises(CutoffError, match="без пояса"):
        window_at("val", window.final_cutoff.replace(tzinfo=None))


# ============================================================
# ВЕКТОРЫ
# ============================================================


def test_readouts_are_the_client_embedding_and_its_events():
    """
    usr — тот же вектор, что client_embeddings; mean_event и
    last_event — среднее и последнее из векторов событий того же
    прохода.
    """

    from src.mlm.model import pack

    model = world.model().eval()
    clients = [client for client in world.clients() if client.n_events]
    data = pack(clients, torch.device("cpu"))

    # Всё без графа, как при съёме: с графом SDPA выбирает другое
    # ядро, и последние биты расходятся.
    with torch.no_grad():
        vectors = model.readouts(data)
        embedded = model.client_embeddings(data)
        _, events, usr = model._encode(data)

    assert torch.equal(vectors["usr"], embedded)
    assert torch.equal(vectors["usr"], usr)

    owner = data.user_of_event.numpy()

    for number, client in enumerate(clients):
        mine = events[torch.as_tensor(np.flatnonzero(owner == number))]
        assert torch.allclose(vectors["mean_event"][number], mine.mean(dim=0), atol=1e-6)
        assert torch.equal(vectors["last_event"][number], mine[-1])


# ============================================================
# ЗАДАЧИ CHURN
# ============================================================


def write_churn_reports(rows: dict[str, list[tuple[str, int, bool]]], moments: dict | None = None,
                        damage: dict[str, str] | None = None) -> None:
    """
    Отчёты churn-бейзлайна в формате churn.train: строки задачи по
    группам (клиент, churn, active90), прогноз и источники групп — с
    отпечатками текущей выгрузки. У train метка из продолжения, и рядом
    лежит его future.json. damage — какой отпечаток группы подменить.
    """

    from src.downstream import settings
    from src.preprocessing.rawdata import read_manifest
    from src.preprocessing.settings import raw_group_dir

    parts: dict[str, list[dict]] = {"train_rows": [], "eval_rows": []}
    sources: dict[str, dict] = {}

    for group, clients in rows.items():
        moment = pd.Timestamp((moments or {}).get(group, cutoff(group)))
        for number, (client, churn, active) in enumerate(clients):
            if not active:
                continue
            parts["train_rows" if group == "train" else "eval_rows"].append({
                "task": "churn_active90", "client_id": client, "group": group, "T": moment, "churn": churn,
                "active90": active, "score": 0.1 + 0.6 * churn + 0.01 * number,
            })

        exported = read_manifest(raw_group_dir(group))
        future = group in settings.FUTURE_LABEL_GROUPS

        sources[group] = {
            "T": moment.isoformat(),
            "feature_history_events_sha256": exported.events_sha256,
            "feature_profile_sha256": exported.profile_sha256,
            "target_source": "future" if future else "export",
            "target_events_sha256": "continuation" if future else exported.events_sha256,
        }

        if future:
            directory = settings.CHURN_FUTURE / group
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "future.json").write_text(json.dumps({
                "events_sha256": "continuation",
                "source": {"events_sha256": exported.events_sha256, "profile_sha256": exported.profile_sha256},
            }))

    for group, key in (damage or {}).items():
        sources[group][key] = "export" if key == "target_source" else "previous-generation"

    settings.CHURN_REPORTS.mkdir(parents=True, exist_ok=True)

    for name, part in parts.items():
        pd.DataFrame(part).to_parquet(settings.CHURN_REPORTS / f"{name}.parquet", index=False)

    (settings.CHURN_REPORTS / "metrics.json").write_text(json.dumps({"sources": sources}))


CHURN_ROWS = {
    "train": [("t1", 1, True), ("t2", 0, True), ("t3", 1, False)],
    "val": [("v1", 1, True), ("v2", 0, False), ("v3", 0, True)],
}


def raw_groups(*groups: str) -> None:
    """
    Выгрузки групп с manifest: по ним сверяются отпечатки источников.
    """

    from src.preprocessing.settings import raw_group_dir

    for group in groups:
        write_raw(raw_group_dir(group), EARLY, QUIET_SNAPSHOT)


def test_churn_tasks_are_the_baseline_rows_of_the_current_sources(stage):
    """
    churn_active90 — строки churn-бейзлайна задачи на T группы. Отчёт
    без группы test или на другом T не принимается.
    """

    from src.downstream.tasks import churn_rows

    raw_groups("train", "val")
    write_churn_reports(CHURN_ROWS)

    active = churn_rows("churn_active90", "val")

    assert list(active.index) == ["v1", "v3"] and list(active["churn"]) == [1, 0]
    assert list(churn_rows("churn_active90", "train").index) == ["t1", "t2"]

    with pytest.raises(ValueError, match="--final-test"):
        churn_rows("churn_active90", "test")

    write_churn_reports(CHURN_ROWS, moments={"val": cutoff("val") - timedelta(days=1)})

    with pytest.raises(ValueError, match="разных моментах"):
        churn_rows("churn_active90", "val")


@pytest.mark.parametrize(
    ("group", "key", "message"),
    [
        ("val", "feature_history_events_sha256", "история признаков"),
        ("val", "feature_profile_sha256", "анкета признаков"),
        ("val", "target_events_sha256", "источник метки"),
        ("train", "feature_history_events_sha256", "история признаков"),
        ("train", "feature_profile_sha256", "анкета признаков"),
        ("train", "target_events_sha256", "продолжение — источник метки"),
        ("train", "target_source", "метка из export"),
    ],
)
def test_each_churn_source_is_checked_on_its_own(stage, group, key, message):
    """
    История признаков, анкета признаков и источник метки — три разных
    набора; каждый сверяется отдельно, и чужой любой из них — отказ.
    """

    from src.downstream.tasks import churn_rows

    raw_groups("train", "val")
    write_churn_reports(CHURN_ROWS, damage={group: key})

    with pytest.raises(ValueError, match=message):
        churn_rows("churn_active90", group)


def test_a_continuation_of_another_export_is_refused(stage):

    from src.downstream import settings
    from src.downstream.tasks import churn_rows

    raw_groups("train", "val")
    write_churn_reports(CHURN_ROWS)

    path = settings.CHURN_FUTURE / "train" / "future.json"
    record = json.loads(path.read_text())
    record["source"]["profile_sha256"] = "previous-generation"
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="продолжение другой выгрузки"):
        churn_rows("churn_active90", "train")


def test_train_T_is_the_end_of_the_pretraining_window_and_val_keeps_its_horizon():
    """
    T train — конец окна, на котором учился backbone: окно метки train
    за пределами истории предобучения. У val и test T — местная
    полночь 1-го числа, как у train, окно метки целиком внутри их
    выгрузки, а следующее 1-е число этому уже не удовлетворяет.
    """

    from src.preprocessing.settings import PreprocessingConfig

    config = PreprocessingConfig.load(None)
    windows = config.windows
    horizon = timedelta(days=60)

    assert cutoff("train") == windows["train"].final_cutoff
    for group in ("val", "test"):
        local = cutoff(group).astimezone(config.bank_timezone())
        assert (local.day, local.hour, local.minute, local.second) == (1, 0, 0, 0)
        assert cutoff(group) + horizon <= windows[group].final_cutoff
        following = (local + timedelta(days=32)).replace(day=1)
        assert following + horizon > windows[group].final_cutoff


def test_the_label_only_continuation_is_outside_every_pragma_stage():
    """
    Этапы PRAGMA читают только свои каталоги и RAW групп; каталог
    продолжения не внутри ни одного из них, и RAW-группы у PRAGMA —
    только train, val, test.
    """

    from importlib import import_module

    from src.downstream import settings
    from src.preprocessing.settings import normalize_group

    from tests.conftest import PLACES

    future = settings.CHURN_FUTURE.resolve()

    for module, attribute, _ in PLACES:
        if attribute == "CHURN_FUTURE":
            continue
        place = Path(getattr(import_module(module), attribute)).resolve()
        assert not future.is_relative_to(place), f"{module}.{attribute}"

    with pytest.raises(ValueError):
        normalize_group("future")


# ============================================================
# СРАВНЕНИЕ
# ============================================================


def test_paired_bootstrap_is_zero_for_the_same_scores_and_positive_for_a_better_one():

    rng = np.random.default_rng(1)

    y = (rng.random(400) < 0.2).astype(int)
    noise = rng.random(400)

    same_scores = paired(y, noise, noise, draws=200, seed=0)

    assert same_scores["pr_auc"]["mean"] == 0.0
    assert same_scores["roc_auc"]["not_better"] == 1.0

    better = paired(y, y + 0.1 * noise, noise, draws=200, seed=0)

    assert better["roc_auc"]["low"] > 0.0
    assert better["pr_auc"]["not_better"] == 0.0


def test_input_built_in_processes_is_the_same_and_in_order(stage):
    """
    Сборка входа на T в двух процессах: те же клиенты, в том же
    порядке, побитно. Процессы получают сборщик путями, а не
    глобалами, поэтому читают этот же тестовый каталог.
    """

    from src.downstream.at_cutoff import ClientsAtCutoff, clients_at

    chain(stage, EARLY, QUIET_SNAPSHOT)

    builder = ClientsAtCutoff("val", cutoff("val"))

    alone = list(clients_at(builder, 0))
    together = list(clients_at(builder, 2))

    assert [client.client_id for client in alone] == [client.client_id for client in together]

    for left, right in zip(alone, together):
        same(left, right)


# ============================================================
# ПРОТОКОЛ: VAL ВО ВРЕМЯ ЭКСПЕРИМЕНТОВ, TEST — ТОЛЬКО ФИНАЛЬНО
# ============================================================


SIZES = {"train": 120, "val": 60, "test": 60}


def probe_world(monkeypatch, plus: bool = False, drop: str | None = None) -> tuple[dict, list]:
    """
    Строки churn-бейзлайна групп — в памяти; какие группы запрошены,
    записывается. Метки заданы номером клиента, так что обе метки есть
    в задаче и в каждой группе. plus — есть и catboost_plus_usr на тех
    же строках; drop — клиент, которого у него нет.
    """

    import src.downstream.tasks as tasks

    baseline, asked = {}, []

    for group, size in SIZES.items():
        index = pd.Index([f"{group}_{number:03d}" for number in range(size)], name="client_id")
        number = np.arange(size)
        churn = pd.DataFrame({"churn": (number % 5 == 0).astype(int), "score": (number % 5 == 0) * 0.5 + number / 1000},
                             index=index)
        baseline[group] = {"churn_active90": churn[number % 4 != 0]}

    def churn_rows(task, group):
        asked.append((task, group))
        return baseline[group][task]

    def plus_usr(tag, used, embedded):
        if not plus:
            return None
        rows = {}
        for group in used:
            for task, frame in baseline[group].items():
                other = frame.assign(score=frame["score"] * 0.5 + 0.2)
                rows[(task, group)] = other.drop(index=drop) if drop in other.index else other
        return {"thresholds": {"churn_active90": 0.4}, "rows": rows}

    monkeypatch.setattr(tasks, "churn_rows", churn_rows)
    monkeypatch.setattr(tasks, "churn_thresholds", lambda: {"churn_active90": 0.3})
    monkeypatch.setattr(tasks, "plus_usr", plus_usr)

    return baseline, asked


def write_vectors(tag: str, groups: tuple[str, ...], seed: int = 0) -> None:
    """
    Векторы тега в формате embed для групп groups: на T группы, с
    отпечатками текущей выгрузки.
    """

    from src.downstream.embed import READOUTS, raw_record

    rng = np.random.default_rng(seed)
    directory = downstream_dir(tag)
    directory.mkdir(parents=True, exist_ok=True)

    for group in groups:
        frame = pd.DataFrame({
            "client_id": [f"{group}_{number:03d}" for number in range(SIZES[group])],
            "cutoff": pd.Timestamp(cutoff(group)),
        })
        for name in READOUTS:
            frame[name] = list(rng.normal(size=(SIZES[group], 3)))
        frame.to_parquet(directory / f"{group}.parquet", index=False)

    record = {group: {"cutoff": cutoff(group).isoformat(), **raw_record(group)} for group in groups}
    (directory / "meta.json").write_text(json.dumps({"tag": tag, "checkpoint": f"{tag}.pt", "groups": record}))


def test_the_probe_scores_val_by_default_and_test_only_in_the_final_evaluation(stage, monkeypatch):
    """
    По умолчанию test не запрашивается нигде — ни векторы, ни строки,
    ни метрики, ни сравнения. Все наборы — на одних клиентах, на
    строках бейзлайна задачи. С --final-test добавляется test, а val
    остаётся тем же.
    """

    from src.downstream.probe import PREDICTIONS_FILE, run_probe, show, vs_baseline

    baseline, asked = probe_world(monkeypatch)
    raw_groups("train", "val", "test")
    write_vectors("m", ("train", "val"))

    report = run_probe("m", draws=20, seed=0)

    assert report["groups"] == ["train", "val"] and not report["final_test"]
    assert all(group != "test" for _, group in asked)

    for block in report["tasks"].values():
        assert set(block["rows"]) == set(block["positive_rate"]) == {"train", "val"}
        for result in block["results"].values():
            assert "val" in result and "test" not in result
            assert set(result.get("vs_reference", {"val": None})) == {"val"}

    predictions = pd.read_parquet(downstream_dir("m") / PREDICTIONS_FILE)
    assert set(predictions["group"]) == {"val"}

    for _, part in predictions.groupby("task"):
        assert len({tuple(rows["client_id"]) for _, rows in part.groupby("set")}) == 1

    active = predictions[(predictions["task"] == "churn_active90") & (predictions["set"] == "catboost")]
    expected = baseline["val"]["churn_active90"]
    assert list(active["client_id"]) == list(expected.index)
    np.testing.assert_array_equal(active["score"].to_numpy(), expected["score"].to_numpy())
    assert report["tasks"]["churn_active90"]["rows"]["val"] == len(expected)

    # Отчёт: у задачи свой блок — строки, положительные и доля, — и ни
    # слова о test.
    text = show(report)
    assert list(report["tasks"]) == ["churn_active90"]
    for task in report["tasks"]:
        assert f"{task}: эталон catboost" in text
    assert "доля" in text and "val: сценарий" in text and "test" not in text

    # Сравнение моделей PRAGMA — парное, на тех же клиентах: та же
    # модель под другим тегом даёт нулевую разницу.
    write_vectors("same", ("train", "val"))
    versus = run_probe("same", draws=20, seed=0, baseline="m")
    delta = versus["tasks"]["churn_active90"]["results"]["usr"]["vs_baseline"]
    assert set(delta) == {"val"} and delta["val"]["pr_auc"]["mean"] == 0.0

    # Прогнозов test у пробы без --final-test нет — сравнивать не с чем.
    with pytest.raises(ValueError, match="--final-test"):
        vs_baseline(predictions, "churn_active90", "usr", "test", pd.Index(["test_000"]), np.zeros(1), np.zeros(1),
                    20, 0)

    # Финальная оценка: без векторов test — отказ, с ними — test
    # рядом с тем же val.
    with pytest.raises(FileNotFoundError, match="--final-test"):
        run_probe("m", draws=20, seed=0, final_test=True)

    write_vectors("m", ("train", "val", "test"))
    final = run_probe("m", draws=20, seed=0, final_test=True)

    assert final["groups"] == ["train", "val", "test"]
    assert "test: сценарий" in show(final)
    assert ("churn_active90", "test") in asked

    for task, block in final["tasks"].items():
        for name, result in block["results"].items():
            assert result["val"] == report["tasks"][task]["results"][name]["val"]
            assert "test" in result
            if "vs_reference" in result:
                assert set(result["vs_reference"]) == {"val", "test"}


def test_embed_takes_test_only_in_the_final_evaluation():

    from src.downstream.embed import build_parser, embed_groups, run
    from src.preprocessing.run import EXIT_BLOCKED

    assert embed_groups(None, False) == ["train", "val"]
    assert embed_groups(None, True) == ["train", "val", "test"]
    assert embed_groups(["test"], True) == ["test"]

    with pytest.raises(ValueError, match="--final-test"):
        embed_groups(["val", "test"], False)

    # Отказ до загрузки модели: чекпойнта может и не быть.
    args = build_parser().parse_args(["--checkpoint", "missing.pt", "--groups", "test"])
    assert run(args) == EXIT_BLOCKED


def test_vectors_of_another_export_are_refused(stage, monkeypatch):
    """
    Векторы сняты с одной выгрузки, а строки churn строятся из
    текущей: проба отказывает, а не сравнивает разные истории.
    """

    from src.downstream.probe import run_probe
    from src.preprocessing.settings import raw_group_dir

    probe_world(monkeypatch)
    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))

    write_raw(raw_group_dir("val"), EARLY + AFTER, BUSY_SNAPSHOT)

    with pytest.raises(ValueError, match="другой выгрузки"):
        run_probe("m", draws=20, seed=0)


# ============================================================
# ГОЛОВА ПРОБЫ: ВЛОЖЕННАЯ CV
# ============================================================


def shifted(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Train, у которого масштаб признаков меняется от клиента к клиенту:
    стандартизация по всему train и по части фолда дают разное.
    """

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(150, 4)) * np.linspace(1.0, 30.0, 150)[:, None]
    y = (x[:, 0] / np.linspace(1.0, 30.0, 150) + rng.normal(size=150) > 0.3).astype(int)
    return x, y


def test_the_scaler_is_fitted_inside_each_fold():
    """
    Оценки C совпадают с ручным вложенным расчётом: стандартизация и
    регрессия учатся на обучающей части фолда. Стандартизация по всему
    train до CV дала бы другие оценки — тест это различает.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from src.downstream.probe import CS, FOLDS, probe_model

    x, y = shifted()
    seed = 7

    model = probe_model(seed).fit(x, y)

    def mean_score(leaky: bool) -> np.ndarray:
        scores = []
        for c in CS:
            losses = []
            for fit, held in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(x, y):
                scaler = StandardScaler().fit(x if leaky else x[fit])
                head = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x[fit]), y[fit])
                losses.append(-log_loss(y[held], head.predict_proba(scaler.transform(x[held]))))
            scores.append(np.mean(losses))
        return np.asarray(scores)

    nested = mean_score(leaky=False)

    np.testing.assert_allclose(model.cv_results_["mean_test_score"], nested, rtol=0, atol=1e-10)
    assert model.best_params_["logisticregression__C"] == CS[int(np.argmax(nested))]
    assert not np.allclose(mean_score(leaky=True), nested, rtol=0, atol=1e-6)


def test_the_chosen_C_depends_on_train_only():
    """
    Другой val не меняет ни C, ни голову, ни кривую перебора: прогнозы
    на одинаковых строках те же. Кривая — средний log-loss фолдов при
    каждом C сетки, выбранный C — её минимум. Выход — вероятность на
    каждую строку каждой группы.
    """

    from src.downstream.probe import CS, fit_predict

    x, y = shifted()
    rng = np.random.default_rng(11)
    val_a, val_b = rng.normal(size=(40, 4)), rng.normal(size=(55, 4)) * 100

    (first, second), chosen, cv = fit_predict(x, y, [val_a, val_b], seed=7)
    (again,), chosen_again, cv_again = fit_predict(x, y, [val_a], seed=7)

    assert chosen == chosen_again and cv == cv_again
    assert [point["C"] for point in cv] == [float(c) for c in CS]
    assert chosen == min(cv, key=lambda point: point["log_loss"])["C"]
    assert all(point["log_loss"] > 0 for point in cv)
    np.testing.assert_array_equal(first, again)
    assert first.shape == (40,) and second.shape == (55,)
    assert ((first >= 0) & (first <= 1)).all() and ((second >= 0) & (second <= 1)).all()


# ============================================================
# ПОЛНЫЙ X + [USR] → CATBOOST И ПОРОГИ
# ============================================================


def test_catboost_plus_usr_is_compared_with_catboost_on_the_same_clients(stage, monkeypatch):
    """
    catboost_plus_usr — готовый набор на тех же строках, что catboost:
    парный bootstrap против него, прогнозы по тем же клиентам, в
    отчёте — главное сравнение и Δ с интервалом.
    """

    from src.downstream.probe import PREDICTIONS_FILE, run_probe, show

    probe_world(monkeypatch, plus=True)
    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))

    report = run_probe("m", draws=20, seed=0)
    predictions = pd.read_parquet(downstream_dir("m") / PREDICTIONS_FILE)

    assert report["plus_usr"]

    for task, block in report["tasks"].items():
        plus = block["results"]["catboost_plus_usr"]
        assert set(plus["vs_reference"]) == {"val"} and plus["threshold"] == 0.4
        assert block["results"]["catboost"]["threshold"] == 0.3
        part = predictions[(predictions["task"] == task) & (predictions["group"] == "val")]
        left = part[part["set"] == "catboost"]["client_id"].tolist()
        right = part[part["set"] == "catboost_plus_usr"]["client_id"].tolist()
        assert left == right and left

    text = show(report)
    assert text.count("Δ catboost_plus_usr − catboost (val)") == 1


def test_catboost_plus_usr_on_other_clients_is_refused(stage, monkeypatch):

    from src.downstream.probe import run_probe

    probe_world(monkeypatch, plus=True, drop="val_001")
    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))

    with pytest.raises(ValueError, match="другие клиенты или метки"):
        run_probe("m", draws=20, seed=0)


def test_without_catboost_plus_usr_the_report_says_so(stage, monkeypatch):

    from src.downstream.probe import run_probe, show

    probe_world(monkeypatch)
    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))

    report = run_probe("m", draws=20, seed=0)

    assert not report["plus_usr"] and "catboost_plus_usr нет" in show(report)


@pytest.mark.parametrize(("damage", "message"), [
    ("meta_cutoff", "сняты на"),
    ("row_cutoff", "не на T"),
    ("duplicate", "повторяется"),
])
def test_vectors_not_at_T_or_with_a_repeated_client_are_refused(stage, monkeypatch, damage, message):
    """
    Векторы годятся, только если сняты ровно на T группы — T строк
    churn — и по одному на клиента. Иначе отказ до обучения голов.
    """

    from src.downstream.probe import run_probe

    probe_world(monkeypatch)
    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))

    directory = downstream_dir("m")
    path = directory / "val.parquet"
    frame = pd.read_parquet(path)

    if damage == "meta_cutoff":
        meta = json.loads((directory / "meta.json").read_text())
        meta["groups"]["val"]["cutoff"] = (cutoff("val") - timedelta(days=1)).isoformat()
        (directory / "meta.json").write_text(json.dumps(meta))
    elif damage == "row_cutoff":
        frame.loc[3, "cutoff"] = frame.loc[3, "cutoff"] - pd.Timedelta(seconds=1)
        frame.to_parquet(path, index=False)
    else:
        pd.concat([frame, frame.iloc[[5]]], ignore_index=True).to_parquet(path, index=False)

    with pytest.raises(ValueError, match=message):
        run_probe("m", draws=20, seed=0)


def test_probe_thresholds_come_from_train_only(stage, monkeypatch):
    """
    Порог проб — max F1 по out-of-fold вероятностям train: другие
    векторы val не меняют ни порогов, ни метрик train-зависимых голов,
    а F1 на val посчитан при этом пороге.
    """

    from src.downstream.probe import PREDICTIONS_FILE, at_threshold, run_probe

    probe_world(monkeypatch)
    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))

    first = run_probe("m", draws=20, seed=0)
    predictions = pd.read_parquet(downstream_dir("m") / PREDICTIONS_FILE)

    # Векторы val другие, векторы train — прежние.
    kept = pd.read_parquet(downstream_dir("m") / "train.parquet")
    write_vectors("m", ("train", "val"), seed=5)
    kept.to_parquet(downstream_dir("m") / "train.parquet", index=False)

    second = run_probe("m", draws=20, seed=0)

    for task, block in first["tasks"].items():
        for name, result in block["results"].items():
            assert second["tasks"][task]["results"][name]["threshold"] == result["threshold"]
            part = predictions[(predictions["task"] == task) & (predictions["set"] == name)]
            expected = at_threshold(part["y"].to_numpy(), part["score"].to_numpy(), result["threshold"])
            assert {key: result["val"][key] for key in expected} == expected


def test_log_loss_of_every_set_is_the_cross_entropy_of_its_predictions(stage, monkeypatch):
    """
    log-loss сценария — средняя кросс-энтропия его вероятностей на тех
    же строках val, что и ROC-AUC и PR-AUC; у готовых прогнозов
    CatBoost — так же, как у регрессии.
    """

    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    from src.downstream.probe import PREDICTIONS_FILE, run_probe, show

    probe_world(monkeypatch, plus=True)
    raw_groups("train", "val")
    write_vectors("m", ("train", "val"))

    report = run_probe("m", draws=20, seed=0)
    predictions = pd.read_parquet(downstream_dir("m") / PREDICTIONS_FILE)

    for task, block in report["tasks"].items():
        for name, result in block["results"].items():
            part = predictions[(predictions["task"] == task) & (predictions["set"] == name)]
            y, score = part["y"].to_numpy(), part["score"].to_numpy()
            assert result["val"]["log_loss"] == log_loss(y, score, labels=[0, 1])
            assert result["val"]["roc_auc"] == roc_auc_score(y, score)
            assert result["val"]["pr_auc"] == average_precision_score(y, score)

    assert "LogLoss" in show(report)


def test_catboost_plus_usr_of_other_vectors_or_sources_is_refused(stage):
    """
    catboost_plus_usr годится, только если обучен ровно на этих
    векторах — тег, чекпойнт и записи групп (снятые заново векторы —
    другие) — и на тех же источниках строк, что CatBoost-бейзлайн.
    """

    from src.downstream import settings
    from src.downstream.tasks import plus_usr

    groups = {"train": {"cutoff": "T-train", "seconds": 1.0}, "val": {"cutoff": "T-val", "seconds": 2.0}}
    embedded = {"tag": "m", "checkpoint": "ckpt", "groups": groups}

    assert plus_usr("m", ("train", "val"), embedded) is None

    raw_groups("train", "val")
    write_churn_reports(CHURN_ROWS)
    baseline = json.loads((settings.CHURN_REPORTS / "metrics.json").read_text())

    directory = settings.CHURN_REPORTS / "plus_usr" / "m"
    directory.mkdir(parents=True)

    rows = pd.concat(
        [pd.read_parquet(settings.CHURN_REPORTS / f"{part}.parquet") for part in ("train_rows", "eval_rows")]
    )
    rows.iloc[:0].to_parquet(directory / "train_rows.parquet", index=False)
    rows.to_parquet(directory / "eval_rows.parquet", index=False)

    def write(recorded: dict, sources: dict) -> None:
        (directory / "metrics.json").write_text(json.dumps({
            "embeddings": recorded, "sources": sources, "tasks": {"churn_active90": {"threshold": 0.4}},
        }))

    write(embedded, baseline["sources"])
    found = plus_usr("m", ("train", "val"), embedded)
    assert found["thresholds"] == {"churn_active90": 0.4}
    assert ("churn_active90", "val") in found["rows"]

    with pytest.raises(ValueError, match="обучен на векторах"):
        plus_usr("m", ("train", "val"), dict(embedded, checkpoint="другой чекпойнт"))

    again = dict(embedded, groups=dict(groups, val={"cutoff": "T-val", "seconds": 3.0}))
    with pytest.raises(ValueError, match="векторы val сняты заново.*churn.plus_usr заново"):
        plus_usr("m", ("train", "val"), again)

    write(embedded, dict(baseline["sources"], val={"T": "другой"}))
    with pytest.raises(ValueError, match="источники val"):
        plus_usr("m", ("train", "val"), embedded)

    write(embedded, baseline["sources"])
    with pytest.raises(ValueError, match="--final-test"):
        plus_usr("m", ("train", "val", "test"), embedded)
