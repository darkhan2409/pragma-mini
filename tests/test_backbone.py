from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.dataset.lineage import LINEAGE_FILE
from src.mlm.backbone import BackboneError, init_backbone, state_digest
from src.mlm.settings import BACKBONE_FILES, backbone_dir

from tests import world
from tests.test_scheduler import many
from tests.test_training_math import every_value, settle, tiny


# ============================================================
# ИДЕЯ
# ============================================================
#
# Подготовка к обучению: 07 -> 08 -> 09 -> init_backbone -> 14.
#
#   init_backbone   веса трёх энкодеров без прохода по данным: ни
#                   батчей, ни прохода энкодеров, ни parquet — только
#                   три файла весов и lineage.json;
#   те же веса      что у диагностических этапов 10–12 при том же
#                   конфиге, что у прежнего кода (прямой вызов
#                   конструкторов) и — для итоговой архитектуры — те
#                   же отпечатки, что снял код коммита 60afd4f;
#   обучение        собирает модель только из 09 и backbone, этапы
#                   10–13 ему не нужны, val считается той же моделью,
#                   что учится;
#   градиент        доходит до таблицы, трёх энкодеров и головы, и
#                   шаг AdamW двигает каждую часть;
#   старое          backbone не под текущие словарь, веса 09, набор
#                   или код отвергается;
#   устройство      обучение не откатывается молча ни на CPU, ни на
#                   SDPA.
# ============================================================


CPU = torch.device("cpu")

# Отпечатки начальных весов итоговой архитектуры (d = 128, 4 головы;
# событие 5 блоков, анкета 1, история 2; FFN 512; seed 42), снятые
# кодом коммита 60afd4f — до лёгкой инициализации — на torch 2.13.
# Любая смена начальных весов, в том числе версией torch, видна здесь.
GOLDEN = {
    "event": "c1e4bce4aa476acd2c382950c68dda4e8a45586254cbdfb2b09549493d10d505",
    "profile": "8a19eed0d7f75d97f4547b0513d8bc28b4d98bfce15f7cc99a0f8814af8e84e1",
    "history": "a4d9c59081a6c6c82e34bc83dde27f2be975c4a8a671f71bde92ca4f9a83ee17",
}

DIAGNOSTIC_DIRS = ("10_events", "11_profiles", "12_history", "13_mlm")


def files(root: Path) -> set[Path]:
    return {path for path in root.rglob("*") if path.is_file()}


def saved(name: str) -> dict:
    return torch.load(backbone_dir() / BACKBONE_FILES[name], map_location="cpu", weights_only=True)


def same_state(left: dict, right: dict) -> None:

    assert left.keys() == right.keys()

    for key in left:
        assert torch.equal(left[key], right[key]), key


# ============================================================
# ЛЁГКАЯ ИНИЦИАЛИЗАЦИЯ
# ============================================================


def test_init_reads_no_data_and_runs_no_forward(stage, monkeypatch):
    """
    Батчей нет вовсе, и любой проход энкодера или чтение группы
    упали бы: init_backbone обходится весами 09 и словарём.
    """

    world.write_weights(stage, "train", backbone=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("init_backbone не читает данных и не считает проход")

    for target in (
        "src.event.encoder.EventEncoder.forward",
        "src.profile.encoder.ProfileEncoder.forward",
        "src.history.encoder.HistoryEncoder.forward",
        "src.embedding.layer.InputEmbedding.embed",
        "src.embedding.inputs.Source.__init__",
        "src.mlm.inputs.Source.__init__",
        "src.history.inputs.Source.__init__",
    ):
        monkeypatch.setattr(target, forbidden)

    before = files(stage)

    report = init_backbone(*world.encoder_configs())

    created = files(stage) - before

    directory = backbone_dir()

    assert created == {directory / name for name in (*BACKBONE_FILES.values(), LINEAGE_FILE)}
    assert not [path for path in files(stage) if path.suffix == ".parquet"]
    assert not [name for name in DIAGNOSTIC_DIRS if (stage / name).exists()]

    assert report["bytes"] == sum(path.stat().st_size for path in created)


def test_initial_weights_are_those_of_the_diagnostic_stages_and_the_old_code(stage):
    """
    Этапы 10–12 с тем же конфигом пишут те же веса, что init_backbone,
    бит в бит. И те же, что прямой вызов конструкторов — так начальные
    веса создавал код до init_backbone.
    """

    from src.event.build import build_group as build_events
    from src.event.encoder import EventEncoder
    from src.event.settings import WEIGHTS_FILE, events_dir
    from src.history.build import build_group as build_history
    from src.history.encoder import HistoryEncoder
    from src.history.settings import history_dir
    from src.profile.build import build_group as build_profiles
    from src.profile.encoder import ProfileEncoder
    from src.profile.settings import profiles_dir

    settle(stage, train_people=many())

    event, profile, history = world.encoder_configs()

    build_events("train", event)
    build_profiles("train", profile)
    build_history("train", history)

    stages = {
        "event": events_dir("train") / WEIGHTS_FILE,
        "profile": profiles_dir("train") / WEIGHTS_FILE,
        "history": history_dir("train") / WEIGHTS_FILE,
    }

    direct = {
        "event": EventEncoder(world.DIM, world.LAYERS, world.HEADS, world.FEEDFORWARD, 0.0, world.SEED),
        "profile": ProfileEncoder(world.DIM, world.LAYERS, world.HEADS, world.FEEDFORWARD, 0.0,
                                  world.ROPE_BASE, world.SEED),
        "history": HistoryEncoder(world.DIM, world.LAYERS, world.HEADS, world.FEEDFORWARD, 0.0,
                                  world.ROPE_BASE, world.SEED),
    }

    for name, path in stages.items():

        light = saved(name)
        legacy = torch.load(path, map_location="cpu", weights_only=True)

        assert (light["dim"], light["config"]) == (legacy["dim"], legacy["config"]), name

        same_state(light["state_dict"], legacy["state_dict"])
        same_state(light["state_dict"], direct[name].state_dict())


@pytest.mark.parametrize("name", ["event", "profile", "history"])
def test_final_architecture_starts_from_the_weights_of_the_old_code(name: str):

    from src.embedding.settings import EmbeddingConfig
    from src.event.settings import EventConfig
    from src.history.settings import HistoryConfig
    from src.mlm.backbone import initial_event, initial_history, initial_profile
    from src.profile.settings import ProfileConfig

    dim = EmbeddingConfig().dim

    build = {
        "event": lambda: initial_event(EventConfig(), dim),
        "profile": lambda: initial_profile(ProfileConfig(), dim),
        "history": lambda: initial_history(HistoryConfig(), dim),
    }

    assert state_digest(build[name]().state_dict()) == GOLDEN[name]


def test_default_configs_give_blocks_1_5_2(stage):
    """
    Без переопределений init_backbone собирает итоговую архитектуру,
    и load_model строит модель ровно из неё.
    """

    from src.event.settings import EventConfig
    from src.history.settings import HistoryConfig
    from src.mlm.model import load_model
    from src.profile.settings import ProfileConfig

    world.write_weights(stage, "train", backbone=False)

    report = init_backbone(EventConfig(device="cpu"), ProfileConfig(device="cpu"),
                           HistoryConfig(device="cpu"))

    blocks = {name: item["blocks"] for name, item in report["encoders"].items()}

    assert blocks == {"profile": 1, "event": 5, "history": 2}

    model = load_model(1, 512, 0.1, CPU, "sdpa")

    assert (len(model.profile.layers), len(model.event.layers), len(model.history.layers)) == (1, 5, 2)


# ============================================================
# ОБУЧЕНИЕ БЕЗ ЭТАПОВ 10–13
# ============================================================


def test_training_needs_no_diagnostic_stage(stage, monkeypatch):
    """
    Этапов 10–13 нет, их команды запрещены, а на месте отчёта 13
    лежит мусор: обучение читает только 07, 08, 09 и backbone.
    """

    from src.mlm.train import train

    settle(stage, train_people=many())

    assert not [name for name in DIAGNOSTIC_DIRS if (stage / name).exists()]

    def forbidden(*args, **kwargs):
        raise AssertionError("обучение не запускает диагностические этапы")

    for target in ("src.event.build.build_group", "src.profile.build.build_group",
                   "src.history.build.build_group", "src.mlm.build.build_group"):
        monkeypatch.setattr(target, forbidden)

    report = stage / "13_mlm" / "train"
    report.mkdir(parents=True)

    for name in ("weights.pt", "targets.parquet"):
        (report / name).write_bytes(b"not an artifact")

    result = train(tiny(token_budget=6), epochs=1, max_steps=None, masking=every_value())

    assert result["reason"] == "epochs"
    assert not [name for name in DIAGNOSTIC_DIRS[:3] if (stage / name).exists()]


def test_validation_scores_the_model_being_trained(stage, monkeypatch):
    """
    Модель собирается один раз, и validation получает тот же
    экземпляр — уже обученный на эпохе, а не свежий случайный.
    """

    import src.mlm.model as model_module
    import src.mlm.train as train_module

    settle(stage, train_people=many())

    built = []
    real_load = model_module.load_model

    def load(*args, **kwargs):
        model = real_load(*args, **kwargs)
        built.append((model, {name: value.detach().clone() for name, value in model.named_parameters()}))
        return model

    seen = []
    real_validate = train_module.validate

    def validate(model, source, device, token_budget):
        seen.append((model, {name: value.detach().clone() for name, value in model.named_parameters()}))
        return real_validate(model, source, device, token_budget)

    monkeypatch.setattr(model_module, "load_model", load)
    monkeypatch.setattr(train_module, "validate", validate)

    train_module.train(tiny(token_budget=6), epochs=2, max_steps=None, masking=every_value())

    ((model, initial),) = built

    assert len(seen) == 2
    assert all(item is model for item, _ in seen)

    # К validation эпохи веса уже сдвинуты шагами этой эпохи.
    for _, state in seen:
        assert [name for name in state if not torch.equal(state[name], initial[name])]


# ============================================================
# ГРАДИЕНТ ДО КАЖДОЙ ЧАСТИ
# ============================================================


PARTS = ("embedding", "event", "profile", "history", "head")


def test_one_step_trains_every_part_of_the_model(stage):
    """
    Сквозной проход модели этапа 14: градиент у каждого параметра
    таблицы, трёх энкодеров и головы, конечный, и шаг AdamW двигает
    каждую часть.
    """

    from src.mlm.model import load_model, pack

    settle(stage, train_people=many())

    model = load_model(3, 512, 0.0, CPU, "sdpa")
    model.train()

    assert all(value.requires_grad for value in model.parameters())

    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    out = model(pack([made.client for made in many()], CPU))

    assert out.count > 0 and bool(torch.isfinite(out.loss))

    out.loss.backward()

    for part in PARTS:

        grads = {name: value.grad for name, value in getattr(model, part).named_parameters()}

        assert grads, part
        assert all(grad is not None and bool(torch.isfinite(grad).all()) for grad in grads.values()), part
        assert any(bool(grad.abs().sum() > 0) for grad in grads.values()), part

    optimizer.step()

    for part in PARTS:
        moved = [name for name, value in getattr(model, part).named_parameters()
                 if not torch.equal(value.detach(), before[f"{part}.{name}"])]
        assert moved, part


# ============================================================
# СТАРОЕ И ЧУЖОЕ ОТВЕРГАЕТСЯ
# ============================================================


def rewrite_stamp(**changes) -> None:

    path = backbone_dir() / LINEAGE_FILE

    stamp = json.loads(path.read_text(encoding="utf-8"))

    stamp.update(changes)

    path.write_text(json.dumps(stamp), encoding="utf-8")


def test_backbone_of_another_vocabulary_dataset_or_code_is_refused(stage, monkeypatch):

    from src.mlm.model import load_model

    settle(stage, train_people=many())

    load_model(1, 512, 0.1, CPU, "sdpa")

    good = json.loads((backbone_dir() / LINEAGE_FILE).read_text(encoding="utf-8"))

    for key, value in (
        ("vocabulary", "0" * 64),
        ("dataset", dict(good["dataset"], dataset_format=good["dataset"]["dataset_format"] - 1)),
        ("format", good["format"] + 1),
    ):
        rewrite_stamp(**{key: value})

        with pytest.raises(BackboneError, match=key):
            load_model(1, 512, 0.1, CPU, "sdpa")

        rewrite_stamp(**{key: good[key]})

    load_model(1, 512, 0.1, CPU, "sdpa")

    # Код энкодера сменил версию после сборки backbone.
    monkeypatch.setattr("src.mlm.backbone.EVENT_VERSION", "0.0.0")

    with pytest.raises(BackboneError, match="implementation"):
        load_model(1, 512, 0.1, CPU, "sdpa")


def test_backbone_of_other_embedding_weights_is_refused(stage):
    """
    Этап 09 пересобран другим seed после init_backbone: начальные
    веса собраны под другой входной слой.
    """

    from src.mlm.model import load_model

    settle(stage, train_people=many())

    world.write_weights(stage, "train", seed=world.SEED + 1, backbone=False)

    with pytest.raises(BackboneError, match="embedding"):
        load_model(1, 512, 0.1, CPU, "sdpa")


def test_file_from_another_run_is_refused(stage):

    from src.event.settings import EventConfig
    from src.mlm.backbone import initial_event, payload
    from src.mlm.model import load_model

    settle(stage, train_people=many())

    other = EventConfig(seed=world.SEED, layers=world.LAYERS + 1, heads=world.HEADS,
                        feedforward=world.FEEDFORWARD, dropout=0.0, device="cpu")

    torch.save(payload(initial_event(other, world.DIM), other, world.DIM),
               backbone_dir() / BACKBONE_FILES["event"])

    with pytest.raises(BackboneError, match="не из того запуска"):
        load_model(1, 512, 0.1, CPU, "sdpa")


def test_old_stage_weights_are_not_a_backbone(stage):
    """
    Веса прежних этапов 10–12 лежат на месте, а backbone нет: модель
    их не подбирает, а называет команду init_backbone.
    """

    import shutil

    from src.event.build import build_group as build_events
    from src.mlm.model import load_model

    settle(stage, train_people=many())

    build_events("train", world.encoder_configs()[0])

    shutil.rmtree(backbone_dir())

    with pytest.raises(BackboneError, match="init_backbone"):
        load_model(1, 512, 0.1, CPU, "sdpa")


# ============================================================
# УСТРОЙСТВО ОБУЧЕНИЯ
# ============================================================


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_training_never_falls_back_to_the_cpu(stage, monkeypatch, device: str):

    from src.mlm.train import DeviceError, train

    settle(stage, train_people=many())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(DeviceError, match="CUDA недоступна"):
        train(tiny(device=device), epochs=1, max_steps=None, masking=every_value())


def test_training_on_cuda_never_falls_back_to_sdpa(stage, monkeypatch):
    """
    CUDA есть, flash-attn нет: auto на обучении — ошибка, а не тихие
    корзины SDPA. Модель падает на выборе бэкенда, до переезда на
    карту.
    """

    from src.mlm.train import train
    from src.mlm.varlen import BackendError

    settle(stage, train_people=many())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr("src.mlm.varlen.flash_available", lambda: False)

    with pytest.raises(BackendError, match="через FlashAttention"):
        train(tiny(device="auto", attention_backend="auto"), epochs=1, max_steps=None,
              masking=every_value())


def test_command_names_the_missing_backbone(stage, capsys):

    import shutil

    from src.mlm.train import main
    from src.preprocessing.run import EXIT_BLOCKED

    settle(stage, train_people=many())

    shutil.rmtree(backbone_dir())

    config = stage / "cpu.json"
    config.write_text(json.dumps({"device": "cpu", "attention_backend": "sdpa"}), encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        main(["--epochs", "1", "--config", str(config)])

    assert stopped.value.code == EXIT_BLOCKED
    assert "python -m src.mlm.init_backbone" in capsys.readouterr().out
