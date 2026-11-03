from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits

from src.dataset.settings import META_FILE, SAMPLES_FILE, dataset_dir
from src.downstream.at_cutoff import ClientsAtCutoff, clients_at
from src.downstream.embed import raw_record, trained_model
from src.downstream.lora import ChurnHead, adapter_state, delta_ratio, inject, load_adapters, trainable
from src.downstream.probe import PREDICTIONS_FILE, threshold_max_f1
from src.downstream.settings import EMBEDDINGS_META, cutoff, downstream_dir, groups, run_dir
from src.downstream.tasks import churn_order, churn_rows, churn_sources
from src.masking.settings import MaskingConfig
from src.mlm.backbone import BackboneError
from src.mlm.inputs import Client, Prefetch, Source, micro_batches
from src.mlm.model import Model, pack
from src.mlm.settings import TELEMETRY_FILE, ConfigError
from src.mlm.train import (
    CheckpointError,
    DeviceError,
    TrainingError,
    append_telemetry,
    file_digest,
    limit_cuda_memory,
    save_checkpoint,
    training_device,
)
from src.mlm.varlen import autocast
from src.preprocessing.artifacts import read_json, write_json
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK


# ============================================================
# ДООБУЧЕНИЕ LoRA
# ============================================================
#
#   python -m src.downstream.finetune --checkpoint data/12_train/best_checkpoint.pt --name lora-ref
#
# Эталон заморожен, учатся заплатки и голова на строках train задачи
# churn_active90 — тех же клиентах и метках, что у CatBoost. Доля и
# seed отложенной части — константы, а не поля конфига: иначе
# разбиение перестанет совпадать с churn.train.fit_task.
#
#   вход train   05_dataset/train без единой цели: набор собран на T
#                train, и вход на T побитно равен ему (preflight);
#   вход val     сборка at_cutoff на T val: 05_dataset/val держит
#                события до конца окна, позже T.
#
#   data/runs/<name>/lora.pt                 заплатки, голова, порог
#   data/runs/<name>/telemetry.jsonl         строка на эпоху
#   data/13_downstream/<name>/predictions.parquet   вероятности val
#   data/13_downstream/<name>/meta.json             из чего посчитаны
#
# test — только с --final-test: он не обучает, а берёт готовый
# lora.pt того же эталона и считает val и test той же моделью.
# ============================================================


HOLDOUT = 0.2
SPLIT_SEED = 42

TASK = "churn_active90"

LORA_FILE = "lora.pt"

# Процессов подготовки входа по умолчанию.
WORKERS = 2

# Маска без единой цели: дообучение читает клиентов целиком, как они есть.
ZERO_MASK = MaskingConfig(
    value_probability=0.0,
    event_probability=0.0,
    key_probability=0.0,
    unknown_probability=0.0,
    informativeness_weighted_masking=False,
)


@dataclass(frozen=True)
class FinetuneConfig:

    rank: int = 8
    alpha: float = 8.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Не меньше стольких клиентов на шаг; micro-batch неделим.
    clients_per_step: int = 32
    token_budget: int = 48_000
    max_epochs: int = 10

    # Эпох без улучшения log-loss отложенной части до останова.
    patience: int = 3

    def validate(self) -> None:

        for name in ("rank", "alpha", "learning_rate", "clients_per_step", "token_budget", "max_epochs", "patience"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} обязан быть положительным, получено {getattr(self, name)}")

        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay не может быть отрицательным, получено {self.weight_decay}")

        if self.max_grad_norm <= 0:
            raise ConfigError(f"max_grad_norm обязан быть положительным, получено {self.max_grad_norm}")

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "clients_per_step": self.clients_per_step,
            "token_budget": self.token_budget,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FinetuneConfig":

        base = FinetuneConfig()

        unknown = set(data) - set(base.as_dict())
        if unknown:
            raise ConfigError(f"неизвестные ключи конфига дообучения: {sorted(unknown)}")

        config = replace(
            base,
            rank=int(data.get("rank", base.rank)),
            alpha=float(data.get("alpha", base.alpha)),
            learning_rate=float(data.get("learning_rate", base.learning_rate)),
            weight_decay=float(data.get("weight_decay", base.weight_decay)),
            max_grad_norm=float(data.get("max_grad_norm", base.max_grad_norm)),
            clients_per_step=int(data.get("clients_per_step", base.clients_per_step)),
            token_budget=int(data.get("token_budget", base.token_budget)),
            max_epochs=int(data.get("max_epochs", base.max_epochs)),
            patience=int(data.get("patience", base.patience)),
        )
        config.validate()

        return config

    @staticmethod
    def load(path: Path | None) -> "FinetuneConfig":

        if path is None:
            config = FinetuneConfig()
            config.validate()
            return config

        return FinetuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class FitResult:
    state: dict[str, Tensor]
    best_epoch: int
    # Прогнозы отложенных клиентов лучшей эпохи: client_id -> вероятность.
    held_scores: dict[str, float]
    history: list[dict]


def logits_of(model: Model, head: ChurnHead, clients: list[Client], device: torch.device) -> Tensor:

    with autocast(device):
        usr = model.client_embeddings(pack(clients, device))

    # Голова вне autocast и на fp32: логит около -2.9 в bf16 теряет точность.
    return head(usr.float())


def predict(
    model: Model,
    head: ChurnHead,
    clients: Iterable[Client],
    device: torch.device,
    token_budget: int,
) -> dict[str, float]:

    model.eval()
    head.eval()

    probabilities: dict[str, float] = {}
    with torch.no_grad():
        for batch in micro_batches(clients, token_budget):
            scores = torch.sigmoid(logits_of(model, head, batch, device).float()).cpu().tolist()
            probabilities.update(zip((c.client_id for c in batch), scores))

    return probabilities


def step(optimizer: torch.optim.Optimizer, params: list, seen: int, max_norm: float) -> None:
    """
    Шаг по накопленной сумме потерь: градиент делится на число клиентов.
    """

    for p in params:
        if p.grad is not None:
            p.grad /= seen

    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if not torch.isfinite(norm):
        raise TrainingError(f"норма градиента не число: {float(norm)}")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def fit(
    model: Model,
    head: ChurnHead,
    fit_clients: Callable[[int], Iterable[Client]],
    held_clients: Callable[[], Iterable[Client]],
    labels: dict[str, int],
    config: FinetuneConfig,
    seed: int,
    device: torch.device,
    telemetry: Path | None,
) -> FitResult:

    torch.manual_seed(seed)

    params = trainable(model, head)
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict] = []
    best_state: dict[str, Tensor] = {}
    best_epoch = 0
    best_held: dict[str, float] = {}
    best_loss = float("inf")

    for epoch in range(1, config.max_epochs + 1):

        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # predict переводит в eval; dropout замороженных слоёв работает в обучении.
        model.train()
        head.train()

        total_loss, total_clients, waiting = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch in micro_batches(fit_clients(epoch), config.token_budget):

            target = torch.tensor([labels[c.client_id] for c in batch], dtype=torch.float32, device=device)
            loss = binary_cross_entropy_with_logits(logits_of(model, head, batch, device), target, reduction="sum")

            if not torch.isfinite(loss):
                raise TrainingError(f"потеря не число в эпохе {epoch}: {float(loss.detach())}")

            loss.backward()
            total_loss += float(loss.detach())
            total_clients += len(batch)
            waiting += len(batch)

            if waiting >= config.clients_per_step:
                step(optimizer, params, waiting, config.max_grad_norm)
                waiting = 0

        if waiting:
            step(optimizer, params, waiting, config.max_grad_norm)

        held = predict(model, head, held_clients(), device, config.token_budget)
        ids = sorted(held)
        y = [labels[i] for i in ids]
        score = [held[i] for i in ids]
        scores = {
            "log_loss": float(log_loss(y, score, labels=[0, 1])),
            "pr_auc": float(average_precision_score(y, score)),
        }

        record = {
            "kind": "epoch",
            "epoch": epoch,
            "train_loss": total_loss / max(total_clients, 1),
            "held_log_loss": scores["log_loss"],
            "held_pr_auc": scores["pr_auc"],
            "delta_ratio": delta_ratio(model),
            "seconds": time.perf_counter() - started,
            "cuda_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
            ),
        }
        history.append(record)
        if telemetry is not None:
            append_telemetry(telemetry, record)

        # Сразу, а не после fit: обучение идёт час-два, и строка эпохи
        # нужна, пока оно идёт.
        peak = record["cuda_peak_allocated_gib"]
        print(
            f"[finetune] эпоха {epoch}: train_loss {record['train_loss']:.4f}, отложенная часть "
            f"log-loss {record['held_log_loss']:.4f} PR-AUC {record['held_pr_auc']:.4f}, "
            f"delta_ratio {record['delta_ratio']:.4f}, {record['seconds'] / 60:.1f} мин"
            + (f", пик CUDA allocated {peak:.2f} ГиБ" if peak is not None else ""),
            flush=True,
        )

        if scores["log_loss"] < best_loss:
            best_loss, best_epoch, best_held = scores["log_loss"], epoch, held
            best_state = adapter_state(model, head)
        elif epoch - best_epoch >= config.patience:
            break

    return FitResult(state=best_state, best_epoch=best_epoch, held_scores=best_held, history=history)


# ============================================================
# ДАННЫЕ
# ============================================================


def split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Номера строк обучающей и отложенной части — как
    churn.train.holdout: стратифицированно по меткам в порядке строк
    отчёта (tasks.churn_order).
    """

    divider = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT, random_state=SPLIT_SEED)
    fit_rows, held_rows = next(divider.split(np.zeros(len(y)), y))

    return np.sort(fit_rows), np.sort(held_rows)


def preflight(ids: pd.Index) -> None:
    """
    Вход train с нулевой маской — вход на T задачи, только если набор
    05_dataset/train собран на этот T и время в нём отсчитано от T.
    Каждый клиент задачи обязан лежать в наборе ровно один раз:
    отсутствующий — отказ, а не пропуск.
    """

    directory = dataset_dir("train")
    meta = read_json(directory / META_FILE)

    if meta.get("time_anchor") != "cutoff":
        raise ValueError(
            f"{directory}: time_anchor {meta.get('time_anchor')!r}, а нужен 'cutoff' — вход train равен входу "
            "на T, только если время в нём отсчитано от T"
        )

    moment = cutoff("train")
    recorded = meta.get("events_cutoff")

    if recorded is None or datetime.fromisoformat(recorded) != moment:
        raise ValueError(
            f"{directory}: events_cutoff {recorded}, а T задачи {moment.isoformat()} — вход train был бы не на T"
        )

    stored = pd.Index(pq.read_table(directory / SAMPLES_FILE, columns=["client_id"]).column("client_id").to_pylist())

    repeated = stored[stored.duplicated()]

    if len(repeated):
        raise ValueError(f"{directory}: клиент {repeated[0]} повторяется в наборе")

    missing = ids.difference(stored)

    if len(missing):
        raise ValueError(
            f"клиента задачи {missing[0]} нет в {directory} (всего таких {len(missing)}): "
            "клиенты задачи не пропускаются"
        )


def train_clients(ids: set[str], order: tuple[int, int] | None, workers: int) -> Iterator[Client]:
    """
    Клиенты ids из 05_dataset/train целиком, без единой цели.

    order = (seed, эпоха) — группы строк эпохи в перестановке от них;
    None — в порядке файла (отложенная часть каждую эпоху та же).
    """

    source = Source("train", ZERO_MASK)

    if order is not None:
        source.order = np.random.default_rng(list(order)).permutation(source.count).tolist()

    seen: set[str] = set()

    for client in Prefetch(source, workers).clients():

        if client.client_id not in ids:
            continue

        # Такого входа эталон не видел: клиент задачи не пропускается молча.
        if not client.n_events:
            raise ValueError(f"{client.client_id}: у клиента задачи нет событий в 05_dataset/train")

        seen.add(client.client_id)
        yield client

    missing = ids - seen

    if missing:
        raise ValueError(
            f"проход 05_dataset/train неполон: нет клиента задачи {min(missing)} (всего таких {len(missing)})"
        )


def val_clients(group: str, ids: list[str], workers: int) -> Iterator[Client]:
    """
    Клиенты ids группы на её T, собранные из прошлого (at_cutoff).
    Отсутствующий клиент или клиент без событий до T — отказ.
    """

    wanted = set(ids)
    seen: set[str] = set()

    for client in clients_at(ClientsAtCutoff(group, cutoff(group)), workers):

        if client.client_id not in wanted:
            continue

        if not client.n_events:
            raise ValueError(f"{client.client_id}: у клиента задачи нет событий до T {group}")

        seen.add(client.client_id)
        yield client

    missing = wanted - seen

    if missing:
        raise ValueError(f"{group}: клиента задачи {min(missing)} нет на T (всего таких {len(missing)})")


def check_cutoff_inputs(groups_rows: dict[str, pd.Index]) -> None:
    """
    Сборщик at_cutoff каждой оцениваемой группы строится и сверяется с
    клиентами задачи до обучения: конструктор проверяет окно, политику
    истории и якорь времени против 05_dataset/train, и несовпадение
    конфигурации не должно вскрыться после часа GPU. Клиентов не собирает.
    """

    for group, ids in groups_rows.items():

        known = set(ClientsAtCutoff(group, cutoff(group)).client_ids)
        missing = sorted(set(ids) - known)

        if missing:
            raise ValueError(
                f"{group}: клиентов задачи нет у сборщика на T (всего таких {len(missing)}): "
                f"{', '.join(missing[:10])}"
            )


# ============================================================
# КОМАНДА
# ============================================================


def read_lora(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(f"нет {path}: сначала обучите заплатки (finetune без --final-test)")

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"{path} не читается: {error}") from error

    if not isinstance(state, dict) or state.get("kind") != "lora":
        raise CheckpointError(f"{path} — не заплатки LoRA (нет kind lora)")

    return state


def claim(name: str) -> None:
    """
    Каталоги --name либо новые, либо уже дообучения LoRA. Имена общие
    с прогонами предобучения (data/runs/<имя>: telemetry.jsonl,
    чекпойнты) и с векторами embed (data/13_downstream/<тег>): чужое
    не перезаписывается.
    """

    run = run_dir(name)

    if run.exists():
        if not (run / LORA_FILE).exists():
            raise ValueError(
                f"{run} — не законченный прогон LoRA (нет {LORA_FILE}): выберите другое --name; если это "
                "прерванный прогон LoRA, удалите каталог вручную"
            )
        read_lora(run / LORA_FILE)

    directory = downstream_dir(name)

    if directory.exists():
        path = directory / EMBEDDINGS_META
        meta = read_json(path) if path.exists() else None
        if not isinstance(meta, dict) or meta.get("kind") != "lora":
            raise ValueError(
                f"{directory} — не законченные прогнозы LoRA (нет {EMBEDDINGS_META} с kind lora): выберите "
                "другое --name; если это прогнозы прерванного прогона LoRA, удалите каталог вручную"
            )


def device_of(name: str) -> torch.device:

    device = training_device(name)

    if device.type == "cuda":
        limit_cuda_memory(device)

    return device


def publish(
    name: str,
    state: dict,
    model: Model,
    head: ChurnHead,
    device: torch.device,
    workers: int,
    rows: dict[str, pd.DataFrame],
    final_test: bool,
) -> None:
    """
    Прогнозы оцениваемых групп заплатками state и meta.json.

    Всё, что может отказать, считается до первой записи. Прежняя meta
    удаляется до записи прогнозов, новая пишется последней: прерванная
    запись оставляет прогнозы без meta (проба их не примет), но
    никогда — новые прогнозы рядом со старой meta.
    """

    parts = []

    for group, frame in rows.items():
        scores = predict(model, head, val_clients(group, list(frame.index), workers), device,
                         state["config"]["token_budget"])
        parts.append(pd.DataFrame({
            "task": TASK,
            "set": "lora",
            "group": group,
            "client_id": frame.index.to_numpy(),
            "y": frame["churn"].to_numpy(),
            "score": np.array([scores[client] for client in frame.index], dtype=np.float64),
        }))

    used = groups(final_test)

    meta = {
        "kind": "lora",
        "name": name,
        "reference": state["reference"],
        "groups": {group: {"cutoff": cutoff(group).isoformat(), **raw_record(group)} for group in used},
        "sources": {group: churn_sources(group) for group in used},
        "threshold": state["threshold"],
        "best_epoch": state["best_epoch"],
        "trainable": sum(p.numel() for p in trainable(model, head)),
        "seed": state["seed"],
        "config": state["config"],
        "final_test": final_test,
    }

    directory = downstream_dir(name)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / EMBEDDINGS_META).unlink(missing_ok=True)

    temporary = directory / f"{PREDICTIONS_FILE}.tmp"
    pd.concat(parts, ignore_index=True).to_parquet(temporary, index=False)
    os.replace(temporary, directory / PREDICTIONS_FILE)

    write_json(directory / EMBEDDINGS_META, meta)


def train_and_predict(args) -> None:

    config = FinetuneConfig.load(args.config)

    claim(args.name)

    # Строки val — до обучения: устаревший отчёт обнаружится сразу; как и
    # несовпадение сборщика val с набором train.
    evaluated = {group: churn_rows(TASK, group) for group in groups(False)[1:]}
    check_cutoff_inputs({group: frame.index for group, frame in evaluated.items()})

    device = device_of(args.device)

    model, _, source = trained_model(args.checkpoint, device)
    reference = {"checkpoint": source["checkpoint"], "sha256": file_digest(Path(args.checkpoint)),
                 "epoch": source["epoch"]}

    rows = churn_rows(TASK, "train").loc[churn_order(TASK, "train")]

    preflight(rows.index)

    y = rows["churn"].to_numpy()
    fit_rows, held_rows = split(y)
    labels = dict(zip(rows.index, y.tolist()))
    fit_ids, held_ids = set(rows.index[fit_rows]), set(rows.index[held_rows])

    inject(model, config.rank, config.alpha, args.seed)
    head = ChurnHead(model.embedding.dim, float(y.mean())).to(device)

    directory = run_dir(args.name)
    telemetry = directory / TELEMETRY_FILE

    # Телеметрия — только этого обучения: строки прежнего читались бы
    # как его эпохи.
    telemetry.unlink(missing_ok=True)

    print(
        f"[finetune] устройство {device}, клиентов обучения {len(fit_ids)}, отложенных {len(held_ids)}, "
        f"обучаемых параметров {sum(p.numel() for p in trainable(model, head)):,}",
        flush=True,
    )

    result = fit(
        model,
        head,
        lambda epoch: train_clients(fit_ids, (args.seed, epoch), args.workers),
        lambda: train_clients(held_ids, None, args.workers),
        labels,
        config,
        args.seed,
        device,
        telemetry,
    )

    load_adapters(model, head, result.state)

    held = sorted(held_ids)
    threshold = threshold_max_f1(np.array([labels[c] for c in held]), np.array([result.held_scores[c] for c in held]))

    state = {
        "kind": "lora",
        "adapters": result.state,
        "config": config.as_dict(),
        "seed": args.seed,
        "reference": reference,
        "held": held,
        "best_epoch": result.best_epoch,
        "threshold": threshold,
    }

    save_checkpoint(state, directory / LORA_FILE)

    print(f"[finetune] лучшая эпоха {result.best_epoch}, порог F1 {threshold:.4f} → {directory / LORA_FILE}")

    publish(args.name, state, model, head, device, args.workers, evaluated, final_test=False)


def predict_final(args) -> None:

    claim(args.name)

    path = run_dir(args.name) / LORA_FILE
    state = read_lora(path)

    digest = file_digest(Path(args.checkpoint))
    trained_on = state["reference"]

    if trained_on["sha256"] != digest:
        raise CheckpointError(
            f"{path} обучен на другом эталоне ({trained_on['checkpoint']}, sha256 {trained_on['sha256'][:12]}), "
            f"а --checkpoint {args.checkpoint} — sha256 {digest[:12]}"
        )

    config = FinetuneConfig.from_dict(state["config"])

    device = device_of(args.device)

    model, _, _ = trained_model(args.checkpoint, device)

    evaluated = {group: churn_rows(TASK, group) for group in groups(True)[1:]}
    check_cutoff_inputs({group: frame.index for group, frame in evaluated.items()})

    inject(model, config.rank, config.alpha, state["seed"])

    # Смещение головы придёт из заплаток вместе с весом.
    head = ChurnHead(model.embedding.dim, 0.5).to(device)
    load_adapters(model, head, state["adapters"])

    print(f"[finetune] финальная оценка заплатками {path}: val и test")

    publish(args.name, state, model, head, device, args.workers, evaluated, final_test=True)


def run(args) -> int:

    try:
        if args.final_test:
            predict_final(args)
        else:
            train_and_predict(args)
    except (ConfigError, ValueError, CheckpointError, BackboneError, DeviceError, TrainingError,
            FileNotFoundError) as error:
        print(f"[finetune] {error}")
        return EXIT_BLOCKED

    print(f"[finetune] → {downstream_dir(args.name)}")

    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.downstream.finetune")
    parser.add_argument("--checkpoint", required=True, help="чекпойнт эталона (data/12_train/best_checkpoint.pt)")
    parser.add_argument(
        "--name", required=True, help="имя прогона: data/runs/<name> и data/13_downstream/<name>"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed заплаток, порядка групп строк и dropout")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON FinetuneConfig; с --final-test конфиг и seed берутся из lora.pt",
    )
    parser.add_argument(
        "--final-test", action="store_true",
        help="финальная оценка без обучения: готовый lora.pt того же эталона, прогнозы val и test",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="auto и cuda — только CUDA; CPU — лишь явным cpu",
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help="процессов подготовки входа; 0 — в этом процессе"
    )
    parser.set_defaults(handler=run)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()


__all__ = [
    "HOLDOUT",
    "SPLIT_SEED",
    "ZERO_MASK",
    "FinetuneConfig",
    "FitResult",
    "build_parser",
    "check_cutoff_inputs",
    "fit",
    "main",
    "predict",
    "preflight",
    "run",
    "split",
    "train_clients",
    "val_clients",
]
