from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from src.generator.rng import stable_hash
from src.masking.settings import ConfigError as MaskingConfigError
from src.masking.settings import MaskingConfig
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from .settings import (
    EPOCHS_DIR,
    TELEMETRY_FILE,
    ConfigError,
    MlmConfig,
    best_checkpoint_path,
    checkpoint_path,
    epoch_weights_path,
    train_dir,
)


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
#
# Одна команда, только на train:
#
#   python -m src.mlm.train [--epochs N] [--max-steps N] [--config путь]
#                           [--masking-config путь] [--out каталог] [--resume]
#
# Вход: data/05_dataset/train и для validation data/05_dataset/val
# (время и маска считаются при чтении, src.mlm.inputs); начальные
# веса — входной слой
# data/06_embeddings/train и backbone data/07_backbone (python -m
# src.mlm.init_backbone). Этапы 08–11 не нужны: это диагностика.
# Выход — каталог прогона, по умолчанию data/12_train (--out задаёт
# другой): checkpoint.pt (последнее состояние), best_checkpoint.pt
# (лучший val_loss) и epochs/epoch_NN.pt — веса после каждой полной
# эпохи. Новое обучение без --resume очищает ТОЛЬКО свой каталог.
#
# Чекпойнт помнит, на чём учился: lineage каталога backbone (а с
# ним архитектуру энкодеров, словарь и входной слой) и sha256 трёх
# файлов данных. Продолжение и загрузка обученной модели сверяют
# его с текущими и при расхождении отказывают: другие веса на
# другой архитектуре или другие клиенты не подменяются молча.
#
# Шаг с нечисловым loss или нормой градиента не делается: обучение
# останавливается ошибкой до optimizer.step, веса и AdamW не
# тронуты, на диске остаётся чекпойнт прошлой эпохи. То же с
# нечисловым val_loss — до записи чекпойнтов.
#
# Устройство: CUDA обязательна (device auto или cuda), тихого
# отката на CPU нет — CPU только явным device=cpu. На CUDA внимание
# идёт только через FlashAttention под bf16 autocast: auto здесь
# значит «строго flash», SDPA — только явным attention_backend=sdpa.
#
# Маска разыгрывается при чтении (src.masking: choose + apply) по
# --masking-config. У train seed эпохи выводится из seed
# маскирования и номера эпохи, поэтому одна и та же эпоха даёт одну
# и ту же маску, а соседние эпохи — разные. У val seed тот же, что в
# конфиге, и маска одна на все эпохи.
#
# Клиенты идут потоком и собираются в micro-batch по бюджету
# позиций (inputs.micro_batches, token_budget). Группа строк набора
# — только единица чтения, а не батч модели. micro-batch
# собирается model.pack в плоские массивы без заполнителя.
#
#   micro-batch -> ОДИН проход: InputEmbedding -> Event -> Profile
#               -> History -> MLM -> потери -> backward
#   grad_accum_steps micro-batch'ей -> один шаг оптимизатора
#
# backward идёт по сумме потерь целей micro-batch, а перед шагом
# градиенты делятся на число целей окна: шаг получает градиент
# среднего по ВСЕМ целям окна, и цель в маленьком micro-batch
# весит столько же, сколько в большом. Затем норма градиента
# ограничивается max_grad_norm, делается шаг AdamW и шаг
# расписания LR. step — шаг оптимизатора, --max-steps ограничивает
# именно их.
#
# LR: линейный разгон за warmup_steps шагов, затем cosine до
# min_learning_rate к концу горизонта (lr_factor). Горизонт — это
# ПЛАН обучения: шаги всех --epochs, посчитанные до первого шага.
# --max-steps в него не входит, он только останавливает прогон, а
# --resume берёт горизонт из чекпойнта и продолжает ту же кривую.
#
# Промежуточные parquet этапов 08–11 сюда не читаются: через файл
# градиент не течёт. Весь проход собран в model.Model, и здесь он
# вызывается без no_grad.
#
# После каждой полностью пройденной эпохи та же модель считает
# потери на val: фиксированная маска val (seed конфига), eval и
# no_grad, те же micro-batch'и, без backward и шага. Среднее — по
# всем целям val. По нему обновляется лучший чекпойнт и считается
# early stopping.
#
# Кроме потерь считается точность MLM — Top-1 и Top-5 — только по
# настоящим целям (метка != -100): контекст, незакрытые токены и
# запрещённые цели в знаменатель не входят. На train она копится по
# micro-batch'ам эпохи, на val — по той же фиксированной маске,
# поэтому val сравним между эпохами. Строка эпохи печатает train и
# val, а история эпох едет в чекпойнте. Лучший чекпойнт выбирается
# по-прежнему по val_loss. test здесь не читается никогда.
#
# val дополнительно разбирается по целям (Detail): NLL без
# сглаживания меток и top-1 по механизму маски, по длине истории
# клиента и по ключу цели. На выбор лучшего чекпойнта это не
# влияет, но показывает, из чего сложен val_loss.
#
# Строка шага печатает норму градиента до клипа, ожидание данных и
# время шага; строка эпохи — долю ожидания, клипа и пик памяти
# CUDA. То же лежит в истории эпох.
#
# --resume продолжает с checkpoint.pt: веса, AdamW, расписание,
# счётчики, генераторы случайности и место внутри эпохи.
# ============================================================


# Всё, без чего продолжение невозможно.
CHECKPOINT_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scheduler_total",
    "scheduler_epochs",
    "epoch",
    "epoch_complete",
    "micro_batches_done",
    "step",
    "best_val_loss",
    "epochs_without_improvement",
    "config",
    "masking",
    "rng_state",
    "cuda_rng_state",
    "train_scores",
    "history",
    "backbone",
    "data",
)

# Корзины длины истории клиента (число событий) в разбивке val:
# короткие истории дают мало целей и в общем среднем не видны.
EVENT_BINS = (100, 300, 1000, 3000)

# Сколько байт читать за раз при подсчёте sha256 файла данных.
DIGEST_CHUNK = 8 << 20

# Сколько свободной памяти карты оставить сверх лимита процесса:
# драйверу, дисплею, соседям.
MEMORY_MARGIN = 256 << 20

# При какой доле лимита аллокатор начинает отдавать свои свободные
# блоки, не дожидаясь нехватки.
MEMORY_GC_THRESHOLD = 0.8


class CheckpointError(ValueError):
    """
    Чекпойнт нельзя прочитать или продолжить.
    """


class TrainingError(RuntimeError):
    """
    Обучение разошлось: loss, градиент или val_loss не числа.
    """


class DeviceError(RuntimeError):
    """
    Учиться негде: нужной CUDA нет, а тихий откат на CPU запрещён.
    """


def training_device(name: str):
    """
    Где учиться.

    auto и cuda — только CUDA: обучение не откатывается на CPU
    молча, иначе полный прогон шёл бы часами не там. CPU — лишь
    явным device=cpu (проверки и совместимость).
    """

    import torch

    if name == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise DeviceError(
            f"device={name}: CUDA недоступна, а обучение на CPU запускается "
            "только явным device=cpu"
        )

    return torch.device("cuda")


def limit_cuda_memory(device) -> float:
    """
    Кэш аллокатора CUDA не растёт больше свободной памяти карты;
    возвращает лимит в ГиБ.

    Проходы бывают от десятков до 70 тысяч позиций, и кэш
    кусков разного размера рос за эпохи до 4.4 ГиБ при 3.2 ГиБ
    свободных. Под WSL драйвер молча переносит перерасход в
    системную память, и шаги идут в разы медленнее — без ошибки,
    видно только по времени. С лимитом аллокатор у порога сперва
    отдаёт свои свободные блоки, а настоящая нехватка становится
    OOM, а не тихим переносом. Явный PYTORCH_CUDA_ALLOC_CONF
    пользователя не трогается.
    """

    import torch

    # Лимит ставится на карту по номеру: «cuda» без номера — текущая.
    index = device.index if device.index is not None else torch.cuda.current_device()

    free, total = torch.cuda.mem_get_info(index)

    fraction = max(0.1, min(1.0, (free - MEMORY_MARGIN) / total))

    torch.cuda.set_per_process_memory_fraction(fraction, index)

    if not any(name in os.environ for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")):
        # Переменная окружения читается при первом выделении, а CUDA к
        # этому моменту уже поднята — настройка ставится вызовом.
        setting = f"garbage_collection_threshold:{MEMORY_GC_THRESHOLD}"
        apply = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
        apply(setting) if apply is not None else torch.cuda.memory._set_allocator_settings(setting)

    return fraction * total / 2**30


def describe_model(model, device) -> dict:
    """
    Что и где учится: устройство, бэкенд внимания, архитектура.
    """

    import torch

    info = {
        "device": str(device),
        "attention": model.attention,
        "dim": int(model.embedding.dim),
        "heads": {
            "profile": model.profile.layers[0].heads,
            "event": model.event.layers[0].self_attn.num_heads,
            "history": model.history.layers[0].heads,
        },
        "blocks": {
            "profile": len(model.profile.layers),
            "event": len(model.event.layers),
            "history": len(model.history.layers),
        },
        "parameters": sum(value.numel() for value in model.parameters()),
        "torch": torch.__version__,
    }

    if device.type == "cuda":
        info.update(
            gpu=torch.cuda.get_device_name(device),
            cuda=torch.version.cuda,
            bf16=torch.cuda.is_bf16_supported(),
        )

    if model.attention == "flash":
        import flash_attn

        info["flash_attn"] = flash_attn.__version__

    return info


@dataclass
class Scores:
    """
    Счёт MLM по целям: сумма потерь, число целей и сколько из них
    угадано первым ответом (top1) и попало в первые пять (top5).

    Доли — по числу настоящих целей; без целей их нет (None), а не
    ноль и не деление на ноль.
    """

    loss_sum: float = 0.0
    targets: int = 0
    top1: int = 0
    top5: int = 0

    # Разбивка по целям (Detail.summary) — только у validation. В
    # чекпойнт как счёт эпохи не пишется.
    detail: dict | None = None

    def add(self, out) -> None:
        """
        Прибавить проход модели (model.Predicted).
        """

        from .model import hits

        if out.count == 0:
            return

        first, five = out.hits if out.hits is not None else hits(out.logits, out.targets, 5)

        self.loss_sum += out.loss.item() * out.count
        self.targets += out.count
        self.top1 += first
        self.top5 += five

    def share(self, value: float) -> float | None:
        return value / self.targets if self.targets else None

    @property
    def loss(self) -> float | None:
        return self.share(self.loss_sum)

    @property
    def top1_accuracy(self) -> float | None:
        return self.share(self.top1)

    @property
    def top5_accuracy(self) -> float | None:
        return self.share(self.top5)

    def as_dict(self) -> dict:
        return {"loss_sum": self.loss_sum, "targets": self.targets, "top1": self.top1, "top5": self.top5}

    def summary(self) -> dict:
        return {"loss": self.loss, "top1": self.top1_accuracy, "top5": self.top5_accuracy,
                "targets": self.targets}


def events_bin(n_events: int) -> str:
    """
    Корзина длины истории клиента: «0-100», …, «3000+».
    """

    low = 0

    for high in EVENT_BINS:
        if n_events < high:
            return f"{low}-{high}"
        low = high

    return f"{low}+"


def target_losses(logits, targets) -> tuple[np.ndarray, np.ndarray]:
    """
    По каждой цели: NLL без сглаживания меток и угадана ли она
    первым ответом.

    Кусками по TARGETS_PER_CHUNK: fp32-копия [M, словарь] целиком
    заняла бы сотни мегабайт.
    """

    import torch.nn.functional as F

    from .model import TARGETS_PER_CHUNK

    nll, first = [], []

    for piece, labels in zip(
        logits.detach().split(TARGETS_PER_CHUNK), targets.split(TARGETS_PER_CHUNK)
    ):
        scores = piece.float()
        nll.append(F.cross_entropy(scores, labels, reduction="none").cpu().numpy())
        first.append((scores.argmax(dim=-1) == labels).cpu().numpy())

    return np.concatenate(nll).astype(np.float64), np.concatenate(first)


@dataclass
class Detail:
    """
    Разбивка val по целям: NLL без сглаживания меток и top-1 — по
    механизму маски (event, key, value), по корзине длины истории
    клиента и по ключу цели.

    val_loss сглажен и усреднён по всем целям группы, а цели дают в
    основном длинные истории. По нему по-прежнему выбирается лучший
    чекпойнт; разбивка показывает, из чего он сложен.
    """

    nll_sum: float = 0.0
    targets: int = 0

    # вид разбивки -> группа -> [целей, сумма NLL, угадано первым]
    groups: dict = field(default_factory=dict)

    def add(self, out, clients: list, key_names: np.ndarray) -> None:
        """
        Прибавить проход модели по micro-batch clients.
        """

        from .inputs import IGNORE

        if out.count == 0:
            return

        nll, first = target_losses(out.logits, out.targets)

        # Цели идут в порядке pack: клиент за клиентом, внутри — по
        # номеру токена. В том же порядке берутся и их признаки.
        chosen = [client.labels != IGNORE for client in clients]

        labels = {
            "reason": np.concatenate(
                [np.asarray(client.reason, dtype=object)[mask] for client, mask in zip(clients, chosen)]
            ),
            "events": np.concatenate(
                [
                    np.full(int(mask.sum()), events_bin(client.n_events), dtype=object)
                    for client, mask in zip(clients, chosen)
                ]
            ),
            "key": key_names[
                np.concatenate([client.key_ids[mask] for client, mask in zip(clients, chosen)])
            ],
        }

        for kind, names in labels.items():

            if len(names) != len(nll):
                raise ValueError(
                    f"целей в проходе {len(nll)}, а признаков «{kind}» {len(names)}: "
                    "порядок целей pack разошёлся с клиентами"
                )

            table = self.groups.setdefault(kind, {})

            for name in np.unique(names):
                mask = names == name
                row = table.setdefault(str(name), [0, 0.0, 0])
                row[0] += int(mask.sum())
                row[1] += float(nll[mask].sum())
                row[2] += int(first[mask].sum())

        self.nll_sum += float(nll.sum())
        self.targets += len(nll)

    def summary(self) -> dict | None:

        if not self.targets:
            return None

        def order(kind: str, name: str):
            # Корзины длины — по возрастанию длины, а не по алфавиту.
            return int(name.split("-")[0].rstrip("+")) if kind == "events" else name

        return {
            "nll": self.nll_sum / self.targets,
            "targets": self.targets,
            **{
                kind: {
                    name: {"targets": count, "nll": total / count, "top1": first / count}
                    for name, (count, total, first) in sorted(
                        table.items(), key=lambda item: order(kind, item[0])
                    )
                }
                for kind, table in self.groups.items()
            },
        }


def describe_epoch(epoch: int, telemetry: dict, detail: dict | None) -> str:
    """
    Строки эпохи о времени, градиенте и памяти и о разбивке val.
    """

    train_seconds = telemetry["train_seconds"]
    waited = telemetry["data_wait_seconds"]

    parts = [
        f"[epoch {epoch}] train {train_seconds / 60:.1f} мин, из них ожидание данных "
        f"{waited / 60:.1f} мин ({waited / train_seconds if train_seconds else 0.0:.0%}), "
        f"val {telemetry['val_seconds'] / 60:.1f} мин"
    ]

    if telemetry["steps"]:
        parts.append(
            f"шагов {telemetry['steps']}, норма градиента средняя {telemetry['grad_norm_mean']:.3f} "
            f"наибольшая {telemetry['grad_norm_max']:.3f}, клип {telemetry['clipped_share']:.1%}"
        )

    if telemetry["cuda_peak_allocated_gib"] is not None:
        parts.append(
            f"пик CUDA allocated {telemetry['cuda_peak_allocated_gib']:.2f} ГиБ, reserved "
            f"{telemetry['cuda_peak_reserved_gib']:.2f} ГиБ"
        )

    text = "; ".join(parts)

    if detail is not None:

        def rows(kind: str) -> str:
            return ", ".join(
                f"{name} {row['nll']:.3f}/{row['top1']:.3f}" for name, row in detail[kind].items()
            )

        text += (
            f"\n[epoch {epoch}] val nll {detail['nll']:.4f}; nll/top1 по механизму: "
            f"{rows('reason')}; по длине истории: {rows('events')}"
        )

    return text


def file_digest(path: Path) -> str:
    """
    sha256 файла, по частям: файл батчей train — сотни мегабайт.
    """

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while chunk := handle.read(DIGEST_CHUNK):
            digest.update(chunk)

    return digest.hexdigest()


def data_record(
    groups: tuple[str, ...] = ("train", "val"),
    masking: MaskingConfig | None = None,
) -> dict:
    """
    Отпечаток данных обучения: sha256 наборов групп и точка отсчёта
    их времени. Маска val задана конфигом маскирования, а он лежит в
    чекпойнте сам; при взвешенном value-маскировании к нему добавлен
    sha256 весов value_weights.json словаря — перефит без переобучения
    дал бы другие цели.

    Имя — каталог этапа и группа, а не путь: каталог data/ у тестов
    и у настоящего обучения разный.
    """

    from src.dataset.settings import META_FILE, SAMPLES_FILE, dataset_dir
    from src.preprocessing.artifacts import read_json

    from .inputs import InputError

    record = {}

    for group in groups:

        record[f"05_dataset/{group}"] = file_digest(dataset_dir(group) / SAMPLES_FILE)

        # Время считается при чтении, и от точки отсчёта вход
        # зависит не меньше, чем от самих примеров.
        record[f"05_dataset/{group}/time_anchor"] = read_json(
            dataset_dir(group) / META_FILE
        ).get("time_anchor")

    anchors = {group: record[f"05_dataset/{group}/time_anchor"] for group in groups}

    if len(set(anchors.values())) > 1:
        raise InputError(
            f"время наборов считается от разных точек {anchors}: соберите их с одним time_anchor"
        )

    if masking is not None and masking.informativeness_weighted_masking:

        from src.tokenization.settings import VALUE_WEIGHTS_FILE, vocab_path

        record["03_vocab/value_weights"] = file_digest(vocab_path(VALUE_WEIGHTS_FILE))

    return record


def backbone_record() -> dict:
    """
    lineage каталога backbone: архитектура энкодеров, словарь,
    входной слой и версии кода, из которых собрана модель.
    """

    from src.dataset.lineage import LINEAGE_FILE
    from src.preprocessing.artifacts import read_json

    from .settings import backbone_dir

    return read_json(backbone_dir() / LINEAGE_FILE)


def origin_problems(state: dict, backbone: dict, data: dict | None) -> list[str]:
    """
    Чем текущие backbone и данные отличаются от записанных в
    чекпойнте. data=None — данные не сверяются.
    """

    problems = []

    if state.get("backbone") != backbone:
        was = state.get("backbone") or {}
        changed = sorted(key for key in set(was) | set(backbone) if was.get(key) != backbone.get(key))
        problems.append(
            f"data/07_backbone собран не так, как при обучении (разные {', '.join(changed)}): "
            "веса легли бы на другую архитектуру, словарь или входной слой"
        )

    if data is not None and state.get("data") != data:
        was = state.get("data") or {}
        changed = sorted(name for name in set(was) | set(data) if was.get(name) != data.get(name))
        problems.append(
            f"данные изменились после чекпойнта ({', '.join(changed)}): обучение пошло бы "
            "на других клиентах или другой маске val"
        )

    return problems


def load_trained(path: Path, device, attention_backend: str | None = None):
    """
    Обученная модель в режиме eval и её чекпойнт — из checkpoint.pt,
    best_checkpoint.pt или весов эпохи.

    Архитектура энкодеров берётся из самого чекпойнта (его lineage
    backbone), а не из текущего data/07_backbone: иначе другое число
    голов или rope_base загрузились бы в те же тензоры молча, а
    модель другой архитектуры не загрузилась бы вовсе после
    пересборки 07_backbone под следующий эксперимент. Данные, словарь
    и входной слой обязаны совпасть с текущими (recorded_backbone).
    attention_backend=None — бэкенд из конфига обучения.
    """

    import torch

    from .model import load_model

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"{path} не читается: {error}") from error

    missing = [key for key in ("model_state_dict", "config", "backbone") if key not in state]

    if missing:
        raise CheckpointError(
            f"{path}: нет полей {missing} — это не чекпойнт обученной модели или его формат старый"
        )

    from .backbone import BackboneError

    config = MlmConfig.from_dict(state["config"])

    try:
        model = load_model(
            seed=config.seed,
            events_per_chunk=config.events_per_chunk,
            label_smoothing=config.label_smoothing,
            device=device,
            attention_backend=attention_backend or config.attention_backend,
            backbone=state["backbone"],
        )
    except BackboneError as error:
        raise CheckpointError(f"{path}: {error}") from error

    if config.usr_aux_weight > 0.0:
        attach_recent(model, config)

    model.load_state_dict(state["model_state_dict"])

    if config.restricted_softmax:
        restrict(model)

    if config.hide_event_keys:
        hide_keys(model)

    return model.eval(), state


def attach_recent(model, config: MlmConfig) -> None:
    """
    Вспомогательная цель [USR] под текущий словарь. Свой seed —
    seed головы плюс один: веса MLM-головы от неё не зависят.
    """

    from src.tokenization.finalvocab import FrozenArtifacts

    from .model import recent_types

    model.attach_recent(recent_types(FrozenArtifacts.load(), int(model.embedding.dim), config.seed + 1))


def hide_keys(model) -> None:
    """
    Ключи событий под маской event закрываются тем же [MASK], что и
    их значения.
    """

    from src.tokenization.specials import MASK, load_special_tokens

    model.hide_event_keys(load_special_tokens()[MASK])


def restrict(model) -> None:
    """
    Кандидаты значения по ключу из текущего словаря — тем же, под
    который модель собрана (его отпечаток сверен при загрузке).
    """

    from src.tokenization.finalvocab import FrozenArtifacts

    from .model import candidate_table

    model.restrict(*candidate_table(FrozenArtifacts.load()))


def train_source(config: MlmConfig, masking: MaskingConfig, epoch: int):
    """
    Источник train эпохи: её маска и, с shuffle_row_groups, её
    перестановка групп строк. Одна и та же эпоха даёт тот же
    поток — на этом стоят resume и горизонт cosine.
    """

    from .inputs import Source

    source = Source("train", masking=for_epoch(masking, epoch))

    if config.shuffle_row_groups:
        source.shuffle(stable_hash("order", config.seed, epoch) % (2 ** 31))

    return source


def for_epoch(masking: MaskingConfig, epoch: int) -> MaskingConfig:
    """
    Конфиг маскирования эпохи: те же вероятности, свой seed.

    Маскер берёт всю случайность из seed конфига, поэтому смены
    seed достаточно, чтобы эпоха получила новую маску. stable_hash
    это blake2b: результат не зависит от процесса, в отличие от
    встроенного hash().
    """

    return replace(masking, seed=stable_hash("epoch", masking.seed, epoch) % (2 ** 31))


def lr_factor(done: int, warmup: int, total: int, floor: float) -> float:
    """
    Множитель к learning_rate для шага done + 1.

    done — сколько шагов оптимизатора уже сделано. Разгон:
    (done + 1) / warmup, поэтому первый шаг уже учит, а шаг warmup
    идёт с полным LR. Дальше cosine от 1 до floor =
    min_learning_rate / learning_rate к шагу total; после total
    множитель остаётся floor.
    """

    if done < warmup:
        return (done + 1) / warmup

    progress = min(1.0, (done - warmup) / max(1, total - warmup))

    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def horizon(config: MlmConfig, masking: MaskingConfig, epochs: int) -> int:
    """
    Конец cosine — сколько шагов оптимизатора займёт ПОЛНЫЙ план
    обучения по --epochs.

    --max-steps сюда не входит намеренно. Он останавливает текущий
    прогон, а не укорачивает план: иначе --epochs 10 --max-steps
    100 проехал бы весь cosine за сто шагов, а продолжение тех же
    десяти эпох поехало бы по другой кривой со скачком LR.

    Число micro-batch'ей эпохи считается по длинам клиентов, без
    масок и модели: разбиение от маски не зависит. Окно без целей
    шага не делает, поэтому это верхняя оценка. С перестановкой
    групп строк упаковка у каждой эпохи своя, и эпохи считаются
    по одной.
    """

    from .inputs import micro_batches

    def steps(epoch: int) -> int:
        sizes = train_source(config, masking, epoch).sizes()
        count = sum(1 for _ in micro_batches(sizes, config.token_budget))
        return math.ceil(count / config.grad_accum_steps)

    if not config.shuffle_row_groups:
        return steps(1) * epochs

    return sum(steps(epoch) for epoch in range(1, epochs + 1))


def resumed_horizon(state: dict, path: Path) -> tuple[int, int]:
    """
    Горизонт cosine при продолжении — (шаги, эпохи плана).

    Кривая LR принадлежит обучению, а не прогону, поэтому она
    берётся из чекпойнта, а не считается заново: пересчёт сдвинул
    бы уже пройденную часть расписания, и LR прыгнул бы на первом
    же шаге продолжения. Заодно проверяется, что расписание стоит
    ровно на сделанном шаге: LambdaLR и счётчик step идут вместе,
    и разойтись они могут только у испорченного чекпойнта.
    """

    total = int(state["scheduler_total"])
    epochs = int(state["scheduler_epochs"])

    if total < 1 or epochs < 1:
        raise CheckpointError(
            f"{path}: горизонт расписания {total} шагов на {epochs} эпох — продолжать нечего"
        )

    position = int(state["scheduler_state_dict"].get("last_epoch", -1))

    if position != int(state["step"]):
        raise CheckpointError(
            f"{path}: расписание стоит на шаге {position}, а шагов сделано "
            f"{state['step']} — LR продолжился бы не с того места"
        )

    return total, epochs


def load_checkpoint(path: Path) -> dict:
    """
    Чекпойнт целиком, проверенный до того, как им что-то заменят.
    """

    import torch

    if not path.exists():
        raise CheckpointError(
            f"чекпойнта {path} нет: продолжать нечего, запустите обучение без --resume"
        )

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"{path} не читается: {error}") from error

    missing = [key for key in CHECKPOINT_KEYS if key not in state]

    if missing:
        raise CheckpointError(
            f"{path}: нет полей {missing} — чекпойнт старого формата, продолжить его нельзя"
        )

    return state


def save_checkpoint(state: dict, path: Path) -> None:
    """
    Запись через временный файл: прерванная запись не портит
    прежний чекпойнт.
    """

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    torch.save(state, temporary)

    os.replace(temporary, path)


def validate(model, source, device, token_budget: int) -> Scores:
    """
    Потери и точность обучаемой модели на группе source, без
    обновления весов.

    Модель передаётся готовой: это тот же экземпляр, что только
    что учился. Своих весов validation не грузит. Клиенты идут
    теми же micro-batch'ами, что и в обучении, по одному проходу
    модели на каждый. Среднее и доли берутся по всем целям группы;
    micro-batch без целей в них не входит. У группы без целей
    потерь и долей нет (None).

    Разбивка по целям (Detail) едет в scores.detail.
    """

    import torch

    from src.tokenization.finalvocab import load_final_vocab

    from .inputs import micro_batches
    from .model import pack
    from .varlen import autocast

    model.eval()

    scores = Scores()
    detail = Detail()

    # Имя токена по номеру: ключи целей в разбивке — именами.
    vocab = load_final_vocab()
    key_names = np.empty(len(vocab), dtype=object)

    for token, number in vocab.items():
        key_names[number] = token

    with torch.no_grad():

        for clients in micro_batches(source.clients(), token_budget):

            with autocast(device):
                out = model(pack(clients, device))

            scores.add(out)
            detail.add(out, clients, key_names)

            del out

    scores.detail = detail.summary()

    return scores


def train(
    config: MlmConfig,
    epochs: int,
    max_steps: int | None,
    masking: MaskingConfig,
    resume: bool = False,
    directory: Path | None = None,
) -> dict:
    """
    Проход по micro-batch'ам train с обновлением весов всей модели.

    epochs и max_steps — общие пределы от начала обучения, в том
    числе при resume. При resume config и masking берутся из
    чекпойнта.

    epochs задаёт и горизонт cosine, max_steps — нет: он только
    останавливает текущий прогон. Горизонт первого запуска лежит в
    чекпойнте и при продолжении не пересчитывается.

    directory — каталог прогона (--out); None — data/12_train.
    """

    # torch импортируется здесь, а не в шапке: без него команда
    # обязана сказать, что поставить, а не упасть на импорте.
    import torch

    from .inputs import Prefetch, Source, micro_batches
    from .model import load_model, pack
    from .varlen import BackendError, autocast

    latest_path = checkpoint_path(directory)
    best_path = best_checkpoint_path(directory)
    telemetry_path = train_dir(directory) / TELEMETRY_FILE

    # Чекпойнт читается и проверяется первым: ни один файл не
    # пишется, пока продолжение не собрано целиком.
    state = load_checkpoint(latest_path) if resume else None

    if state is not None:
        config = MlmConfig.from_dict(state["config"])
        masking = MaskingConfig.from_dict(state["masking"])

    device = training_device(config.device)

    # На CUDA обучение идёт только через FlashAttention: auto здесь
    # значит «строго flash», а не «flash, если получится». SDPA на
    # CUDA — только явным attention_backend=sdpa.
    backend = config.attention_backend

    if device.type == "cuda" and backend == "auto":
        backend = "flash"

    if device.type == "cuda" and backend == "flash" and not torch.cuda.is_bf16_supported():
        raise DeviceError("FlashAttention считает в bf16, а эта CUDA bf16 не поддерживает")

    if device.type == "cuda":
        limit = limit_cuda_memory(device)
        print(f"[train] лимит памяти CUDA {limit:.2f} ГиБ: свободная на старте минус запас")

    try:
        model = load_model(
            seed=config.seed,
            events_per_chunk=config.events_per_chunk,
            label_smoothing=config.label_smoothing,
            device=device,
            attention_backend=backend,
        )
    except BackendError as error:
        raise BackendError(
            f"{error}. Обучение на CUDA идёт через FlashAttention; SDPA — только явным "
            "attention_backend=sdpa"
        ) from error

    if config.restricted_softmax:
        restrict(model)

    if config.hide_event_keys:
        hide_keys(model)

    if config.usr_aux_weight > 0.0:
        attach_recent(model, config)

    # Модель целиком на устройстве и учится целиком: таблица
    # эмбеддингов, три энкодера и голова.
    elsewhere = [name for name, value in model.named_parameters() if value.device.type != device.type]

    if elsewhere:
        raise DeviceError(f"параметры не на {device}: {elsewhere[:5]}")

    frozen = [name for name, value in model.named_parameters() if not value.requires_grad]

    if frozen:
        raise RuntimeError(f"замороженные параметры: {frozen[:5]} — модель обязана учиться целиком")

    described = describe_model(model, device)

    if device.type == "cuda":
        print(
            f"[train] {device}: {described['gpu']}; CUDA {described['cuda']}, PyTorch "
            f"{described['torch']}, bf16 {'да' if described['bf16'] else 'нет'}"
        )

    print(
        f"[train] внимание {described['attention']}"
        + (f" (flash-attn {described['flash_attn']}, активации bf16)" if "flash_attn" in described else "")
        + f"; d {described['dim']}, голов {described['heads']['event']}; блоков: анкета "
        f"{described['blocks']['profile']}, событие {described['blocks']['event']}, история "
        f"{described['blocks']['history']}; параметров {described['parameters']:,}"
    )

    # Параметры всей модели: общая таблица эмбеддингов, энкодеры
    # события, анкеты и истории и проекция головы.
    # fused на CUDA: один проход ядра по всем параметрам вместо
    # цепочки поэлементных.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )

    if state is None:
        total, planned_epochs = horizon(config, masking, epochs), epochs

    else:
        # План замораживается при первом запуске: продолжение едет
        # по той же кривой, а --epochs сверх плана добавляет эпохи
        # уже на min_learning_rate.
        total, planned_epochs = resumed_horizon(state, latest_path)

    floor = config.min_learning_rate / config.learning_rate

    # LambdaLR считает свои шаги сам: scheduler.step() вызывается
    # только после настоящего optimizer.step(), поэтому его счётчик
    # и есть число сделанных шагов оптимизатора.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda done: lr_factor(done, config.warmup_steps, total, floor)
    )

    if state is None:
        # Dropout энкодеров берёт случайность из глобального
        # генератора: одинаковый конфиг обязан давать одинаковые веса.
        torch.manual_seed(config.seed)

        step, best, stale = 0, None, 0
        first_epoch, skip = 1, 0
        epoch_scores, history = Scores(), []

    else:
        # Порядок важен: оптимизатор грузится после создания
        # расписания, иначе LR из чекпойнта затёрло бы начальным.
        try:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except (RuntimeError, ValueError, KeyError) as error:
            raise CheckpointError(f"{latest_path} не подходит к модели: {error}") from error

        # Те же генераторы, что были в момент сохранения: dropout
        # (в том числе flash-attn) продолжается той же случайностью.
        torch.set_rng_state(state["rng_state"])

        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)

        step = int(state["step"])
        best = state["best_val_loss"]
        stale = int(state["epochs_without_improvement"])
        history = list(state["history"])

        # Полная эпоха — продолжаем со следующей. Неполная (её
        # прервал --max-steps) — с той же, пропуская уже пройденные
        # micro-batch'и: порядок клиентов и маска эпохи
        # детерминированы, а окно на остановке было пустым.
        if state["epoch_complete"]:
            first_epoch, skip = int(state["epoch"]) + 1, 0
            epoch_scores = Scores()
        else:
            # Счёт эпохи продолжается с того же места, что и эпоха.
            first_epoch, skip = int(state["epoch"]), int(state["micro_batches_done"])
            epoch_scores = Scores(**state["train_scores"])

        if (
            stale >= config.early_stopping_patience
            or first_epoch > epochs
            or (max_steps is not None and step >= max_steps)
        ):
            return {
                "checkpoint": str(latest_path),
                "epoch": int(state["epoch"]),
                "step": step,
                "best_val_loss": best,
                "epochs_without_improvement": stale,
                "device": str(device),
                "model": described,
                "reason": "nothing",
            }

    optimizer.zero_grad(set_to_none=True)

    # val с фиксированной маской конфига: открывается сразу, чтобы
    # нехватка набора стала видна до первого шага.
    val_source = Source("val", masking=masking)

    # Происхождение прогона. Считается до первой записи: продолжение
    # на другом backbone или других данных отказывает, не тронув ни
    # одного файла.
    backbone = backbone_record()
    data = data_record(masking=masking)

    if state is not None:

        problems = origin_problems(state, backbone, data)

        if problems:
            raise CheckpointError(
                f"{latest_path}: {'; '.join(problems)} — продолжать нельзя, начните "
                "обучение заново без --resume"
            )

    if epochs > planned_epochs:
        print(
            f"[train] расписание рассчитано на {planned_epochs} эпох ({total} шагов) и "
            f"остаётся прежним: эпохи {planned_epochs + 1}-{epochs} пройдут на "
            "min_learning_rate, прошлая часть кривой не пересчитывается"
        )

    if state is None:
        # Новое обучение: чекпойнты прошлого прогона к нему не
        # относятся, и продолжать их без --resume было бы нечем.
        # Удаляются здесь, а не раньше: модель, оптимизатор,
        # расписание и val уже собрались, и падение на сборке не
        # стоило бы прошлого обучения. Чистится только свой каталог.
        latest_path.unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)

        for old in (train_dir(directory) / EPOCHS_DIR).glob("epoch_*.pt"):
            old.unlink()

        telemetry_path.unlink(missing_ok=True)

    epoch = first_epoch
    reason = "epochs"

    # Окно накопления: сколько micro-batch'ей в нём, сколько целей
    # и сумма потерь по целям.
    window_batches = 0
    window_targets = 0
    window_loss = 0.0

    # Телеметрия окна и эпохи: сколько цикл ждал данных, когда окно
    # открылось, нормы градиента до клипа.
    window_wait = 0.0
    window_started = time.perf_counter()
    epoch_wait = 0.0
    norms: list[float] = []

    # LR последнего сделанного шага — для строки эпохи.
    last_lr = optimizer.param_groups[0]["lr"]

    def close_window() -> None:
        """
        Шаг оптимизатора по накопленному окну.

        backward шёл по СУММЕ потерь целей каждого micro-batch,
        поэтому деление градиентов на число целей окна даёт
        градиент среднего по всем целям окна: цель одного
        micro-batch весит столько же, сколько цель другого. Клип
        стоит после деления: ограничивается норма именно этого
        среднего. Окно без целей шага не делает — ни оптимизатора,
        ни расписания: weight decay AdamW иначе сдвинул бы веса без
        обучающего сигнала.

        Нечисловые loss или норма градиента останавливают обучение
        до optimizer.step: один такой шаг сделал бы нечисловыми все
        веса и состояние AdamW, а следующий чекпойнт сохранил бы их.
        """

        nonlocal step, window_batches, window_targets, window_loss, last_lr
        nonlocal window_wait, window_started, epoch_wait

        if window_targets > 0:

            torch._foreach_div_(
                [parameter.grad for parameter in model.parameters() if parameter.grad is not None],
                window_targets,
            )

            # Норма ДО клипа: по ней видно, как часто и насколько клип
            # режет шаг.
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm))
            loss = window_loss / window_targets

            if not (math.isfinite(norm) and math.isfinite(loss)):
                raise TrainingError(
                    f"эпоха {epoch}, шаг {step + 1}: loss {loss}, норма градиента {norm} — шаг "
                    f"не сделан, веса и AdamW не тронуты; последнее сохранённое состояние — "
                    f"{latest_path}"
                )

            lr = optimizer.param_groups[0]["lr"]
            last_lr = lr

            optimizer.step()
            scheduler.step()

            step += 1
            norms.append(norm)

            print(
                f"epoch={epoch} step={step} loss={loss:.4f} "
                f"targets={window_targets} micro_batches={window_batches} lr={lr:.2e} "
                f"grad_norm={norm:.3f} wait={window_wait:.2f}s "
                f"time={time.perf_counter() - window_started:.2f}s"
            )

        optimizer.zero_grad(set_to_none=True)

        window_batches, window_targets, window_loss = 0, 0, 0.0

        epoch_wait += window_wait
        window_wait, window_started = 0.0, time.perf_counter()

    def snapshot(complete: bool, done: int) -> dict:
        """
        Всё состояние для продолжения. done — сколько micro-batch'ей
        эпохи пройдено; у полной эпохи он не нужен и равен нулю.
        """

        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scheduler_total": total,
            "scheduler_epochs": planned_epochs,
            "epoch": epoch,
            "epoch_complete": complete,
            "micro_batches_done": done,
            "step": step,
            "best_val_loss": best,
            "epochs_without_improvement": stale,
            "config": config.as_dict(),
            "masking": masking.as_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
            "train_scores": epoch_scores.as_dict(),
            "history": list(history),
            "backbone": backbone,
            "data": data,
        }

    for epoch in range(first_epoch, epochs + 1):

        # Каждая эпоха начинается в режиме обучения: validation
        # прошлой эпохи оставил модель в eval, и dropout был выключен.
        model.train()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        epoch_wait, norms, epoch_aux = 0.0, [], []
        window_wait, window_started = 0.0, started

        source = Prefetch(train_source(config, masking, epoch), config.loader_workers)

        stopped = False

        # micro-batch'и эпохи, пройденные до этого места, включая
        # пропущенные при resume.
        done = 0

        # Момент, когда цикл снова готов взять micro-batch: всё время
        # до следующего — ожидание данных (чтение, маска, сборка).
        ready = started

        for clients in micro_batches(source.clients(), config.token_budget):

            window_wait += time.perf_counter() - ready

            if done < skip:
                done += 1
                ready = time.perf_counter()
                continue

            # Предел считает шаги оптимизатора. Проверка стоит перед
            # новым micro-batch'ем: окно к этому моменту пустое,
            # потому что шаг закрывает окно.
            if max_steps is not None and step >= max_steps:
                stopped = True
                break

            # Один проход модели на весь micro-batch. На CUDA — под
            # bf16 autocast; backward ниже идёт уже вне него, а веса и
            # AdamW остаются fp32.
            # Без полных логитов: счёт top-1/top-5 приходит кусками.
            with autocast(device):
                out = model(pack(clients, device), logits=False)

            window_batches += 1
            done += 1

            if out.count > 0:
                objective = out.loss
                if out.aux is not None:
                    objective = objective + config.usr_aux_weight * out.aux
                    epoch_aux.append(float(out.aux.detach()))
                (objective * out.count).backward()
                window_targets += out.count
                window_loss += out.loss.item() * out.count

            epoch_scores.add(out)

            # Логиты [M, словарь] нужны только счёту точности. Без
            # del они жили бы до конца следующего прохода модели,
            # поверх его собственных.
            del out

            if window_batches == config.grad_accum_steps:
                close_window()

            ready = time.perf_counter()

        skip = 0

        # Эпоха, прерванная --max-steps, не пройдена целиком:
        # validation нет, лучший чекпойнт не трогается, последний
        # запоминает место внутри эпохи.
        if stopped:
            reason = "max_steps"
            save_checkpoint(snapshot(False, done), latest_path)
            break

        # Неполное окно в конце эпохи не выбрасывается: его
        # градиенты нормируются по его настоящему числу целей.
        if window_batches:
            close_window()

        train_seconds = time.perf_counter() - started

        val_scores = validate(
            model, Prefetch(val_source, config.loader_workers), device, config.token_budget
        )

        val_seconds = time.perf_counter() - started - train_seconds

        val_loss, val_targets = val_scores.loss, val_scores.targets

        # Нечисловой val_loss ни лучшим, ни худшим не бывает: модель
        # сломана, и чекпойнт этой эпохи не пишется.
        if val_loss is not None and not math.isfinite(val_loss):
            raise TrainingError(
                f"эпоха {epoch}: val_loss {val_loss} — чекпойнты эпохи не записаны; последнее "
                f"сохранённое состояние — {latest_path}"
            )

        # val без целей сигнала не даёт: ни улучшением, ни
        # ухудшением это не считается.
        improved = val_loss is not None and (
            best is None or val_loss < best - config.early_stopping_min_delta
        )

        if improved:
            best, stale = val_loss, 0
        elif val_loss is not None:
            stale += 1

        def shown(value: float | None) -> str:
            return f"{value:.4f}" if value is not None else "n/a"

        print(
            f"epoch={epoch} train_loss={shown(epoch_scores.loss)} "
            f"train_top1={shown(epoch_scores.top1_accuracy)} "
            f"train_top5={shown(epoch_scores.top5_accuracy)} "
            f"val_loss={shown(val_loss)} val_top1={shown(val_scores.top1_accuracy)} "
            f"val_top5={shown(val_scores.top5_accuracy)} val_targets={val_targets} "
            f"lr={last_lr:.2e} best_val_loss={shown(best)} "
            f"patience={stale}/{config.early_stopping_patience}"
        )

        telemetry = {
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "data_wait_seconds": epoch_wait,
            "steps": len(norms),
            "grad_norm_mean": float(np.mean(norms)) if norms else None,
            "grad_norm_max": max(norms) if norms else None,
            "clipped_share": (
                sum(norm > config.max_grad_norm for norm in norms) / len(norms) if norms else None
            ),
            "cuda_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
            ),
            "usr_aux_mean": float(np.mean(epoch_aux)) if epoch_aux else None,
            "cuda_peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None
            ),
        }

        print(describe_epoch(epoch, telemetry, val_scores.detail))

        # Телеметрия — строкой в свой файл, а не в историю чекпойнта:
        # время стены не результат, и продолжение обязано дать тот же
        # чекпойнт, что непрерывный прогон. У эпохи, продолженной
        # посередине, она описывает только часть после продолжения.
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)

        with open(telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "step": step, **telemetry}) + "\n")

        history.append({
            "epoch": epoch,
            "step": step,
            "learning_rate": last_lr,
            "train": epoch_scores.summary(),
            "val": val_scores.summary(),
            "val_detail": val_scores.detail,
        })

        current = snapshot(True, 0)

        epoch_scores = Scores()

        # Веса каждой полной эпохи: выбор эпохи по downstream-метрике
        # и кривые «эпоха → качество» без переобучения.
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": current["model_state_dict"],
                "config": current["config"],
                "masking": current["masking"],
                "backbone": backbone,
                "data": data,
                "history": current["history"],
            },
            epoch_weights_path(epoch, directory),
        )

        if improved:
            save_checkpoint(current, best_path)

        save_checkpoint(current, latest_path)

        if stale >= config.early_stopping_patience:
            reason = "early_stopping"
            break

        if max_steps is not None and step >= max_steps:
            reason = "max_steps"
            break

    return {
        "checkpoint": str(latest_path),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best,
        "epochs_without_improvement": stale,
        "device": str(device),
        "model": described,
        "reason": reason,
    }


def run_training(args) -> int:

    try:
        from .backbone import BackboneError
        from .build import MlmError
        from .inputs import InputError
        from .varlen import BackendError

    except ModuleNotFoundError as error:
        print(
            f"[train] нет модуля {error.name}: обучение считает тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    if args.resume and (args.config or args.masking_config):
        print(
            "[train] --resume берёт конфиг и маскирование из чекпойнта: "
            "--config и --masking-config с ним не задаются"
        )
        return EXIT_BLOCKED

    try:
        if args.resume:
            # Настоящие значения придут из чекпойнта внутри train.
            config, masking = MlmConfig(), MaskingConfig()
        else:
            config = MlmConfig.load(Path(args.config) if args.config else None)

            masking = MaskingConfig.load(
                Path(args.masking_config) if args.masking_config else None
            )

        result = train(
            config, args.epochs, args.max_steps, masking, resume=args.resume, directory=args.out
        )

    except (
        ConfigError, MaskingConfigError, InputError, MlmError, BackendError,
        BackboneError, CheckpointError, DeviceError, TrainingError, FileNotFoundError,
    ) as error:
        print(f"[train] {error}")
        return EXIT_BLOCKED

    if result["reason"] == "nothing":
        print(
            f"[train] продолжать нечего: эпоха {result['epoch']}, шагов {result['step']}, "
            f"patience {result['epochs_without_improvement']} — увеличьте --epochs или "
            "--max-steps, если early stopping ещё не сработал"
        )
        return EXIT_OK

    best = result["best_val_loss"]

    print(
        f"[train] эпох {result['epoch']}, шагов {result['step']} на {result['device']}, "
        f"остановка: {result['reason']}, best_val_loss "
        f"{f'{best:.4f}' if best is not None else 'n/a'} → {result['checkpoint']}"
    )

    return EXIT_OK


def _positive(text: str) -> int:

    value = int(text)

    if value < 1:
        raise argparse.ArgumentTypeError(f"нужно целое больше нуля, получено {value}")

    return value


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(prog="python -m src.mlm.train")

    parser.add_argument(
        "--epochs", type=_positive, default=10,
        help=(
            "сколько эпох всего, считая от начала обучения (по умолчанию 10, "
            "early stopping может остановить раньше); первый запуск ими же задаёт "
            "горизонт cosine"
        ),
    )
    parser.add_argument(
        "--max-steps", type=_positive, default=None,
        help=(
            "остановить прогон, когда шагов оптимизатора от начала обучения станет "
            "столько; на горизонт cosine не влияет"
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига головы и оптимизатора",
    )
    parser.add_argument(
        "--masking-config", type=Path, default=None,
        help="JSON конфига маскирования: по нему разыгрываются маски train и val",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=(
            "каталог прогона (по умолчанию data/12_train): чекпойнты и веса эпох; новое "
            "обучение очищает только его"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="продолжить с checkpoint.pt каталога прогона",
    )

    parser.set_defaults(handler=run_training)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


__all__ = [
    "CHECKPOINT_KEYS",
    "EVENT_BINS",
    "CheckpointError",
    "Detail",
    "DeviceError",
    "Scores",
    "TrainingError",
    "backbone_record",
    "build_parser",
    "data_record",
    "describe_epoch",
    "describe_model",
    "events_bin",
    "file_digest",
    "limit_cuda_memory",
    "for_epoch",
    "horizon",
    "load_checkpoint",
    "load_trained",
    "lr_factor",
    "main",
    "origin_problems",
    "attach_recent",
    "restrict",
    "resumed_horizon",
    "run_training",
    "save_checkpoint",
    "target_losses",
    "train",
    "training_device",
    "validate",
]


if __name__ == "__main__":
    main()
