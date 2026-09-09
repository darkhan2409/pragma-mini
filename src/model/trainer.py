from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from itertools import chain
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.artifacts import write_json, write_text
from src.tokenizer.dataset import collate
from src.preprocessing.config import DATASET_NAMES
from src.tokenizer.masking import SCHEME_BATCH, SCHEME_EXAMPLE, MaskingConfig, Masker

from .backbone import build_backbone
from .checkpoint import (
    check_resume,
    check_resume_splits,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from .config import ACTIVATIONS, STRUCTURE_SESSION, STRUCTURES, config_from_tokenizer
from .batching import BatchError
from .data import ClientStore, EpochSampler, FixedSplit, session_keys_from_examples
from .history_batching import metadata_from_examples, prepare_history_batch, to_model_inputs
from .losses import mlm_loss
from .metrics import MetricAccumulator, UnigramTable, render_metrics
from .mlm_batching import build_targets, representations, within_session
from .mlm_head import FieldLogits, FieldTable, MLMHead
from .targets import DEFAULT_POLICY, POLICY_NOTES, TARGET_POLICIES, exclude_patterns


# ============================================================
# ИДЕЯ
# ============================================================
#
# Обучаются backbone и голова вместе: замораживать энкодеры
# бессмысленно, они и есть то, что должно научиться.
#
# Бюджет считается в УСПЕШНЫХ шагах оптимизатора. Batch без
# целей не шаг: он не двигает ни веса, ни scheduler, и должен
# быть виден в логе как пропуск, а не растворяться в среднем.
#
# Оценка идёт на заранее собранных наборах: маски, обрезка,
# состав и порядок batch зафиксированы до первого шага. Иначе
# «метрика выросла» означало бы в том числе «маски стали
# проще».
# ============================================================


PRECISIONS: tuple[str, ...] = ("auto", "float32", "bf16")


def _merge_masking(records: list) -> dict | None:
    """
    Диагностика группы микро-batch'ей как одного шага.
    """

    present = [item for item in records if item]

    if not present:
        return None

    strategies: dict[str, int] = {}

    for item in present:
        for name, count in (item.get("selection") or {}).get("strategies", {}).items():
            strategies[name] = strategies.get(name, 0) + int(count)

    eligible = sum(item["n_eligible"] for item in present)
    masked = sum(item["n_masked"] for item in present)

    return {
        "mode": present[0]["mode"],
        "n_eligible": eligible,
        "n_masked": masked,
        "masked_fraction": masked / eligible if eligible else 0.0,
        "selection": {"strategies": strategies, "unique": masked},
    }


class TrainingInterrupted(RuntimeError):
    """
    Обучение остановлено намеренно: Ctrl+C или файл остановки.

    Отдельный тип, потому что это не сбой: веса целы, состояние
    сохранено, продолжать можно.
    """


class TrainingAborted(RuntimeError):
    """
    Обучение остановлено: продолжать нельзя без вмешательства.
    """


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================


# Поля, задающие архитектуру запуска. Порядок фиксирован:
# он же порядок в as_dict и в отчётах.
ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "d_model",
    "n_heads",
    "dim_feedforward",
    "activation",
    "n_event_layers",
    "n_profile_layers",
    "n_history_layers",
    "n_session_layers",
    "structure",
)


BEST_SCOPE_FULL = "full"
BEST_SCOPE_RECENT = "recent"

BEST_SCOPES: tuple[str, ...] = (BEST_SCOPE_FULL, BEST_SCOPE_RECENT)


@dataclass(frozen=True)
class TrainConfig:

    seed: int = 42
    val_seed: int = 20240607

    lr: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    warmup_steps: int = 100
    max_steps: int = 2000

    log_every: int = 100
    eval_every: int = 500

    batch_size: int = 2
    eval_batch_size: int = 8

    # Один optimizer step из нескольких микро-batch'ей: ответ на
    # нехватку памяти, при котором эффективный batch сохраняется.
    accumulation_steps: int = 1

    # None означает отсутствие лимита истории.
    max_events_per_history: int | None = 128
    event_microbatch: int = 1024
    num_workers: int = 0

    masking_mode: str = "field_balanced"

    # Какие поля вообще становятся целями. all это прежнее
    # поведение и умолчание: цифры старых запусков остаются
    # воспроизводимыми.
    target_policy: str = DEFAULT_POLICY

    # Как разыгрываются маски: batch это прежняя схема,
    # example привязывает маску к паре (клиент, cutoff).
    mask_scheme: str = SCHEME_BATCH

    # По какой метрике выбирается best.pt: full это прежний
    # выбор по всей истории, recent по целям месяца наблюдения.
    best_metric: str = BEST_SCOPE_FULL

    # Держать ли готовые batch'и validation в памяти. Поток
    # требует схемы масок example.
    stream_validation: bool = False

    # Как часто last.pt переписывается независимо от оценок.
    # Долгий прогон обязан переживать сбой, а не ждать
    # следующей validation через тысячи шагов.
    checkpoint_every: int = 2000

    # Наборы, на которых best.pt оценивается ОДИН раз после
    # обучения. Пусто значит не оценивать: test не должен
    # смотреться по ходу дела.
    final_splits: tuple[str, ...] = ()
    balanced_share: float = 0.15
    token_rate: float = 0.15
    event_rate: float = 0.15
    key_rate: float = 0.10
    keys_per_example: int = 1

    max_consecutive_skips: int = 20

    # Белый список на весь набор: клиенты с ID меньше значения.
    # Распределение по сплитам берётся из существующего разбиения.
    client_universe: int | None = None

    # Бюджет в эпохах вместо max_steps.
    epochs: int | None = None

    max_train_clients: int | None = 512
    max_val_clients: int | None = 64

    top_k: int = 5
    epsilon: float = 1e-8

    dropout: float = 0.1
    precision: str = "auto"

    # ----------------------------------------------------
    # АРХИТЕКТУРА ЗАПУСКА
    # ----------------------------------------------------
    #
    # None означает «умолчание ModelConfig». Это единственный
    # способ добавить новую архитектуру, не переписав молча
    # архитектуру всех прежних запусков: старый train_config
    # без этих полей даст None и соберёт свою исходную модель.
    # ----------------------------------------------------

    d_model: int | None = None
    n_heads: int | None = None
    dim_feedforward: int | None = None
    activation: str | None = None
    n_event_layers: int | None = None
    n_profile_layers: int | None = None
    n_history_layers: int | None = None
    n_session_layers: int | None = None
    structure: str | None = None

    def __post_init__(self) -> None:

        for name in ("max_steps", "warmup_steps", "log_every", "eval_every", "batch_size",
                     "eval_batch_size", "event_microbatch", "accumulation_steps",
                     "max_consecutive_skips", "top_k"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть положительным, получено {getattr(self, name)}")

        # None это осмысленное значение: лимита нет.
        for name in ("max_events_per_history", "client_universe", "epochs",
                     "max_train_clients", "max_val_clients"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} должен быть положительным или None, получено {value}")

        if self.lr <= 0:
            raise ValueError("lr должен быть положительным")

        if self.precision not in PRECISIONS:
            raise ValueError(f"precision должен быть одним из {PRECISIONS}, получено {self.precision!r}")

        if self.num_workers != 0:
            raise ValueError("num_workers > 0 в V1 не поддерживается: данные уже в памяти")

        for name in ARCHITECTURE_FIELDS:
            value = getattr(self, name)
            if name not in ("structure", "activation") and value is not None and value < 1:
                raise ValueError(f"{name} должен быть положительным или None, получено {value}")

        if self.structure is not None and self.structure not in STRUCTURES:
            raise ValueError(f"structure должен быть одним из {STRUCTURES}, получено {self.structure!r}")

        if self.activation is not None and self.activation not in ACTIVATIONS:
            raise ValueError(f"activation должен быть одним из {ACTIVATIONS}, получено {self.activation!r}")

        if self.stream_validation and self.mask_scheme != SCHEME_EXAMPLE:
            raise ValueError(
                "потоковая validation требует mask_scheme=example: при схеме "
                "batch маска зависит от номера batch, и пересборка дала бы "
                "другие цели"
            )

        if self.best_metric not in BEST_SCOPES:
            raise ValueError(
                f"best_metric должен быть одним из {BEST_SCOPES}, "
                f"получено {self.best_metric!r}"
            )

        unknown = [name for name in self.final_splits if name not in DATASET_NAMES]

        if unknown:
            raise ValueError(f"неизвестные наборы в final_splits: {unknown}")

        if self.target_policy not in TARGET_POLICIES:
            raise ValueError(
                f"target_policy должен быть одним из {TARGET_POLICIES}, "
                f"получено {self.target_policy!r}"
            )

        # Остальные границы проверит сам ModelConfig при сборке.

        # Проверяет свои границы сам.
        self.masking()

    # --------------------------------------------------------

    def architecture(self) -> dict:
        """
        Явно заданные размеры запуска. Пустой словарь означает
        «взять умолчания ModelConfig», то есть прежнюю модель.
        """

        return {
            name: getattr(self, name)
            for name in ARCHITECTURE_FIELDS
            if getattr(self, name) is not None
        }

    @property
    def uses_sessions(self) -> bool:
        return self.structure == STRUCTURE_SESSION

    def masking(self, seed: int | None = None) -> MaskingConfig:
        return MaskingConfig(
            mode=self.masking_mode,
            seed=self.seed if seed is None else int(seed),
            token_rate=self.token_rate,
            keys_per_example=self.keys_per_example,
            event_rate=self.event_rate,
            balanced_share=self.balanced_share,
            key_rate=self.key_rate,
            exclude_fields=exclude_patterns(self.target_policy),
            scheme=self.mask_scheme,
        )

    @staticmethod
    def from_dict(data: dict) -> "TrainConfig":
        """
        Конфигурация из checkpoint.

        as_dict пишет и пояснения (scheduler, budget), поэтому
        берутся только настоящие поля: диагностика обязана
        воспроизводить обучение, а не угадывать его.
        """

        known = {item.name for item in fields(TrainConfig)}

        taken = {name: value for name, value in data.items() if name in known}

        # JSON не знает кортежей: без обратного приведения
        # конфиг из checkpoint перестал бы равняться исходному.
        if "final_splits" in taken and taken["final_splits"] is not None:
            taken["final_splits"] = tuple(taken["final_splits"])

        return TrainConfig(**taken)

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "val_seed": self.val_seed,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "log_every": self.log_every,
            "eval_every": self.eval_every,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "client_universe": self.client_universe,
            "epochs": self.epochs,
            "max_events_per_history": self.max_events_per_history,
            "event_microbatch": self.event_microbatch,
            "num_workers": self.num_workers,
            "masking_mode": self.masking_mode,
            "target_policy": self.target_policy,
            "mask_scheme": self.mask_scheme,
            "best_metric": self.best_metric,
            "stream_validation": self.stream_validation,
            "checkpoint_every": self.checkpoint_every,
            "final_splits": list(self.final_splits),
            "balanced_share": self.balanced_share,
            "token_rate": self.token_rate,
            "event_rate": self.event_rate,
            "key_rate": self.key_rate,
            "keys_per_example": self.keys_per_example,
            "max_consecutive_skips": self.max_consecutive_skips,
            "max_train_clients": self.max_train_clients,
            "max_val_clients": self.max_val_clients,
            "top_k": self.top_k,
            "epsilon": self.epsilon,
            "dropout": self.dropout,
            "precision": self.precision,
            **{name: getattr(self, name) for name in ARCHITECTURE_FIELDS},
            "scheduler": "linear warmup, затем постоянный learning rate",
            "budget": "бюджет считается в успешных шагах оптимизатора, не в эпохах",
        }


def resolve_precision(precision: str, device: torch.device) -> str:
    """
    Модель целиком в half не переводится: autocast или float32.
    """

    if precision not in PRECISIONS:
        raise ValueError(f"precision должен быть одним из {PRECISIONS}")

    if device.type != "cuda":

        if precision == "bf16":
            raise ValueError("bf16 доступен только на CUDA; на CPU обучение идёт в float32")

        return "float32"

    if precision == "float32":
        return "float32"

    supported = torch.cuda.is_bf16_supported()

    if precision == "bf16" and not supported:
        raise ValueError("устройство не поддерживает bf16")

    return "bf16" if supported else "float32"


def autocast_for(mode: str, device: torch.device):
    if mode == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


# ============================================================
# РЕЗУЛЬТАТ ШАГА
# ============================================================


@dataclass(frozen=True)
class StepResult:
    step: int
    batch: int
    skipped: bool
    reason: str | None = None
    field_balanced: float | None = None
    token_weighted: float | None = None
    n_targets: int = 0
    n_fields: int = 0
    grad_norm: float | None = None
    learning_rate: float | None = None
    seconds: float = 0.0
    masking: dict | None = None


# ============================================================
# ТРЕНЕР
# ============================================================


# ============================================================
# СРЕЗ ПО МЕСЯЦУ НАБЛЮДЕНИЯ
# ============================================================
#
# Метрика на полной истории отвечает «как модель в среднем
# читает прошлое клиента». Метрика месяца наблюдения отвечает
# «как она читает то, что месяц добавил». Это разные вопросы,
# и путать их нельзя: в полной истории свежих событий единицы
# процентов, и улучшение на них там растворяется.
#
# Считается по ТЕМ ЖЕ маскам: срез, а не вторая оценка.
# ============================================================


RECENT_SCOPE = "цели событий месяца наблюдения, те же маски"


def recent_logits(field_logits, targets, device) -> list:
    """
    Те же logits, но только по свежим целям.
    """

    if targets.recent is None:
        return []

    keep = torch.from_numpy(np.ascontiguousarray(targets.recent)).to(device)

    sliced = []

    for item in field_logits:

        chosen = keep[item.index]

        if not bool(chosen.any()):
            continue

        sliced.append(
            FieldLogits(
                key_id=item.key_id,
                index=item.index[chosen],
                logits=item.logits[chosen],
            )
        )

    return sliced


class Trainer:

    def __init__(
        self,
        config: TrainConfig,
        tokenizer,
        table: FieldTable,
        unigram: UnigramTable,
        device="cpu",
        seed: int | None = None,
    ):

        self.config = config
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.table = table
        self.unigram = unigram

        self.device = torch.device(device)

        self.precision = resolve_precision(config.precision, self.device)

        self.model_config = config_from_tokenizer(
            tokenizer, dropout=config.dropout, **config.architecture()
        )

        seed = config.seed if seed is None else int(seed)

        # build_backbone задаёт seed один раз; голова строится
        # следом из того же потока чисел.
        self.backbone = build_backbone(self.model_config, seed=seed)
        self.head = MLMHead(self.model_config, table)

        self.backbone.to(self.device)
        self.head.to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.parameters()), lr=config.lr, weight_decay=config.weight_decay
        )

        warmup = max(1, config.warmup_steps)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / warmup)
        )

        self.masker = Masker(self.vocab, config.masking())

        # Поля, выведенные политикой из задачи. Отчёт обязан
        # называть их, иначе «целей нет» читалось бы как
        # дефект данных.
        self.excluded: frozenset[str] = self.masker.excluded_names

        self.n_batches = 0
        self.n_steps = 0
        self.n_skipped = 0
        self.consecutive_skips = 0

    # --------------------------------------------------------

    def parameters(self):
        return chain(self.backbone.parameters(), self.head.parameters())

    def n_parameters(self) -> int:
        return self.backbone.n_parameters() + self.head.n_parameters()

    def counters(self) -> dict:
        return {
            "n_batches": self.n_batches,
            "n_steps": self.n_steps,
            "n_skipped": self.n_skipped,
            "consecutive_skips": self.consecutive_skips,
        }

    def load_counters(self, counters: dict) -> None:
        self.n_batches = int(counters["n_batches"])
        self.n_steps = int(counters["n_steps"])
        self.n_skipped = int(counters["n_skipped"])
        self.consecutive_skips = int(counters.get("consecutive_skips", 0))

    def train_mode(self) -> None:
        self.backbone.train()
        self.head.train()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.head.eval()

    # --------------------------------------------------------

    def prepare(self, examples, mask_step: int, masker: Masker | None = None):
        """
        Пример → обрезка → раскладка → маски → цели → тензоры входа.
        """

        keys = session_keys_from_examples(examples)

        if self.config.uses_sessions and keys is None:
            raise BatchError(
                "структура session требует ключей сессий: "
                "ClientStore должен быть открыт с sessions=True"
            )

        history = prepare_history_batch(
            collate(examples),
            metadata_from_examples(examples),
            self.config.max_events_per_history,
            masker=self.masker if masker is None else masker,
            step=mask_step,
            session_keys=keys if self.config.uses_sessions else None,
            structure=self.model_config.structure,
        )

        targets = build_targets(history, self.table)

        return to_model_inputs(history, self.model_config), targets, history

    def compute(self, inputs, targets, attention_rule: str | None = None):
        """
        Один forward: локальные состояния, событие и клиент из него же.
        """

        moved = inputs.to(self.device)

        tensors = targets.tensors(self.device)

        with autocast_for(self.precision, self.device):

            out = self.backbone(
                moved,
                self.config.event_microbatch,
                gather=(tensors["event_row"], tensors["col"]),
                attention_rule=attention_rule,
            )

            h_local, h_event, h_usr = representations(out, tensors)

            within = within_session(out, tensors)

            field_logits = self.head(h_local, h_event, h_usr, tensors["key_ids"], within)

        return mlm_loss(field_logits, tensors["local_targets"]), field_logits, tensors["local_targets"]

    # --------------------------------------------------------

    def _skip(self, reason: str, seconds: float) -> StepResult:

        self.n_skipped += 1
        self.consecutive_skips += 1

        if self.consecutive_skips > self.config.max_consecutive_skips:
            raise TrainingAborted(
                f"{self.consecutive_skips} batch подряд без целей: обучение не двигается, "
                f"предел max_consecutive_skips={self.config.max_consecutive_skips}"
            )

        return StepResult(
            step=self.n_steps,
            batch=self.n_batches,
            skipped=True,
            reason=reason,
            seconds=seconds,
        )

    def optimize(self, inputs, targets) -> StepResult:
        """
        Полный шаг на уже подготовленном batch.
        """

        started = time.perf_counter()

        if targets.n == 0:
            return self._skip("в batch нет замаскированных позиций предсказуемых полей", 0.0)

        self.optimizer.zero_grad(set_to_none=True)

        result, _, _ = self.compute(inputs, targets)

        if result.empty:
            return self._skip(result.reason or "нет целей", time.perf_counter() - started)

        loss = result.field_balanced

        if not bool(torch.isfinite(loss)):
            raise TrainingAborted(f"loss не конечен на batch {self.n_batches}: {float(loss.detach())}")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.config.gradient_clip)

        if not bool(torch.isfinite(grad_norm)):
            raise TrainingAborted(
                f"норма градиента не конечна на batch {self.n_batches}: {float(grad_norm)}"
            )

        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        self.optimizer.step()
        self.scheduler.step()

        self.n_steps += 1
        self.consecutive_skips = 0

        return StepResult(
            step=self.n_steps,
            batch=self.n_batches,
            skipped=False,
            field_balanced=float(result.field_balanced.detach()),
            token_weighted=float(result.token_weighted.detach()),
            n_targets=result.n_targets,
            n_fields=result.n_fields,
            grad_norm=float(grad_norm),
            learning_rate=learning_rate,
            seconds=time.perf_counter() - started,
        )

    def optimize_group(self, prepared: list) -> StepResult:
        """
        Один шаг оптимизатора из нескольких микро-batch'ей.

        Loss делится на число микро-batch'ей С ЦЕЛЯМИ, а не на
        размер группы: пустой микро-batch не должен уменьшать
        вклад остальных.
        """

        started = time.perf_counter()

        useful = [(inputs, targets) for inputs, targets in prepared if targets.n > 0]

        if not useful:
            return self._skip("в группе нет замаскированных позиций предсказуемых полей", 0.0)

        self.optimizer.zero_grad(set_to_none=True)

        balanced: list[float] = []
        weighted: list[float] = []

        n_targets = 0
        fields: set[int] = set()

        for inputs, targets in useful:

            result, field_logits, _ = self.compute(inputs, targets)

            if result.empty:
                continue

            loss = result.field_balanced / len(useful)

            if not bool(torch.isfinite(loss)):
                raise TrainingAborted(
                    f"loss не конечен на batch {self.n_batches}: {float(loss.detach())}"
                )

            loss.backward()

            balanced.append(float(result.field_balanced.detach()))
            weighted.append(float(result.token_weighted.detach()))

            n_targets += result.n_targets
            fields.update(item.key_id for item in field_logits)

        if not balanced:
            return self._skip("все микро-batch'и группы остались без целей", time.perf_counter() - started)

        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.config.gradient_clip)

        if not bool(torch.isfinite(grad_norm)):
            raise TrainingAborted(
                f"норма градиента не конечна на batch {self.n_batches}: {float(grad_norm)}"
            )

        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        self.optimizer.step()
        self.scheduler.step()

        self.n_steps += 1
        self.consecutive_skips = 0

        return StepResult(
            step=self.n_steps,
            batch=self.n_batches,
            skipped=False,
            field_balanced=float(np.mean(balanced)),
            token_weighted=float(np.mean(weighted)),
            n_targets=n_targets,
            n_fields=len(fields),
            grad_norm=float(grad_norm),
            learning_rate=learning_rate,
            seconds=time.perf_counter() - started,
        )

    def train_group(self, groups: list) -> StepResult:
        """
        Один шаг оптимизатора из списка микро-batch'ей.

        При одном микро-batch'е совпадает с прежним поведением:
        деление на единицу ничего не меняет.
        """

        started = time.perf_counter()

        prepared = []
        masking = []

        for examples in groups:

            inputs, targets, history = self.prepare(examples, self.n_batches)

            self.n_batches += 1

            prepared.append((inputs, targets))
            masking.append(history.masking)

        result = self.optimize_group(prepared)

        return StepResult(
            **{
                **result.__dict__,
                "seconds": time.perf_counter() - started,
                "masking": masking[0] if len(masking) == 1 else _merge_masking(masking),
            }
        )

    def train_step(self, examples) -> StepResult:
        return self.train_group([examples])

    # --------------------------------------------------------

    def _fork_devices(self):
        return [self.device.index or 0] if self.device.type == "cuda" else []

    def evaluate(
        self,
        splits: dict[str, FixedSplit],
        attention_rule: str | None = None,
        exclude: frozenset[str] = frozenset(),
        keep_units: bool = False,
    ) -> dict:
        """
        Оценка не трогает ни веса, ни генераторы обучения.

        keep_units дополнительно хранит разбивку по примерам:
        она нужна парному bootstrap и больше никому.
        """

        was_training = self.backbone.training

        self.eval_mode()

        reports: dict[str, dict] = {}
        accumulators: dict[str, MetricAccumulator] = {}

        started = time.perf_counter()

        with torch.random.fork_rng(devices=self._fork_devices()):

            with torch.inference_mode():

                for name, split in splits.items():

                    accumulator = MetricAccumulator(
                        self.table,
                        self.unigram,
                        self.config.top_k,
                        self.config.epsilon,
                        keep_units=keep_units,
                    )

                    # Второй счёт по тем же самым маскам, но
                    # только по целям месяца наблюдения. Это не
                    # другая оценка, а срез той же: маски,
                    # порядок и веса общие.
                    recent = MetricAccumulator(
                        self.table,
                        self.unigram,
                        self.config.top_k,
                        self.config.epsilon,
                    )

                    offset = 0

                    for inputs, targets in split.iter_batches(self.model_config):

                        accumulator.note_batch(targets.n_masked, targets.n_degenerate)
                        recent.note_batch(targets.n_masked, targets.n_degenerate)

                        if targets.n:

                            _, field_logits, local = self.compute(inputs, targets, attention_rule)

                            units = None

                            if keep_units:
                                # Номер примера сквозной по набору,
                                # иначе примеры разных batch слились бы.
                                # Устройство то же, что у logits:
                                # индексировать CPU-тензор CUDA-индексом нельзя.
                                units = targets.tensors(self.device)["example"] + offset

                            accumulator.update(field_logits, local, units)

                            fresh = recent_logits(field_logits, targets, local.device)

                            if fresh:
                                recent.update(fresh, local)

                        offset += inputs.n_examples

                    reports[name] = {
                        **accumulator.finalize(exclude),
                        "recent": {
                            **recent.finalize(exclude),
                            "scope": RECENT_SCOPE,
                        },
                    }

                    accumulators[name] = accumulator

        if was_training:
            self.train_mode()

        reports["_seconds"] = time.perf_counter() - started

        if keep_units:
            reports["_accumulators"] = accumulators

        return reports


# ============================================================
# ЛОГ
# ============================================================


class JsonLog:
    """
    Одна строка JSON на событие: лог читается и после падения.

    Файл открывается и закрывается на каждой записи, и после
    записи содержимое сбрасывается на диск. Иначе последние
    события долгого прогона существовали бы только в буфере
    процесса, то есть ровно до сбоя.
    """

    def __init__(self, path: Path, append: bool = False):

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not append or not self.path.exists():
            self.path.write_text("", encoding="utf-8", newline="\n")

    def write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


# ============================================================
# ПРИЗНАКИ ЖИЗНИ
# ============================================================
#
# Долгий прогон надо уметь спрашивать «ты жив и где ты», не
# останавливая его и не читая гигабайт лога. Для этого рядом
# лежат три коротких файла: pid.txt, command.txt и status.json.
#
# status.json переписывается атомарно: читатель никогда не
# увидит половину файла.
# ============================================================


STATUS_FILE = "status.json"
PID_FILE = "pid.txt"
COMMAND_FILE = "command.txt"
STOP_FILE = "stop.request"


def write_atomic(path: Path, text: str) -> None:

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")

    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_status(out_dir: Path, status: dict) -> None:
    write_atomic(
        Path(out_dir) / STATUS_FILE,
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
    )


def stop_requested(out_dir: Path) -> bool:
    """
    Мягкая остановка файлом.

    На Windows фоновому процессу Ctrl+C не пошлёшь, поэтому
    просьба остановиться это файл. Обработка та же, что у
    Ctrl+C: досчитать шаг, сохранить состояние, выйти.
    """

    return (Path(out_dir) / STOP_FILE).exists()


def _memory(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1 << 20)


def _lengths(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"p50": 0, "p90": 0, "max": 0, "mean": 0.0}
    return {
        "p50": int(np.percentile(values, 50, method="inverted_cdf")),
        "p90": int(np.percentile(values, 90, method="inverted_cdf")),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def truncation_check(store, config: TrainConfig, batch_size: int = 64) -> dict:
    """
    Подтверждает, что ни одна история не обрезана.

    Проверяется не настройка, а результат: у каждого примера
    used == original и truncated == False. Долгий проход, зато
    утверждение «полные истории» перестаёт быть обещанием.
    """

    original: list[np.ndarray] = []
    used: list[np.ndarray] = []

    truncated = 0

    for start in range(0, len(store), batch_size):

        examples = store.examples(range(start, min(start + batch_size, len(store))))

        history = prepare_history_batch(
            collate(examples),
            metadata_from_examples(examples),
            config.max_events_per_history,
        )

        info = history.info

        original.append(info.original_history_length)
        used.append(info.used_history_length)

        truncated += int(info.truncated.sum())

    original_all = np.concatenate(original)
    used_all = np.concatenate(used)

    identical = bool(np.array_equal(original_all, used_all))

    return {
        "dataset": store.dataset,
        "max_events_per_history": config.max_events_per_history,
        "n_examples": int(original_all.size),
        "n_truncated": truncated,
        "used_equals_original": identical,
        "n_empty_histories": int((original_all == 0).sum()),
        "lengths": {
            "p50": int(np.percentile(original_all, 50)),
            "p90": int(np.percentile(original_all, 90)),
            "p95": int(np.percentile(original_all, 95)),
            "p99": int(np.percentile(original_all, 99)),
            "max": int(original_all.max()),
            "total_events": int(original_all.sum()),
        },
        "passed": identical and truncated == 0,
    }


def masking_summary(results: list) -> dict | None:
    """
    Средние по шагам: сколько было доступно, сколько скрыто и чем.

    У combined суммы стратегий больше числа скрытых позиций:
    одну позицию могли выбрать две стратегии, и разница между
    суммой и объединением это и есть величина пересечений.
    """

    records = [item.masking for item in results if item.masking is not None]

    if not records:
        return None

    strategies: dict[str, list[int]] = {}

    for record in records:
        for name, count in (record.get("selection") or {}).get("strategies", {}).items():
            strategies.setdefault(name, []).append(count)

    return {
        "mode": records[0]["mode"],
        "batches": len(records),
        "eligible": float(np.mean([item["n_eligible"] for item in records])),
        "masked": float(np.mean([item["n_masked"] for item in records])),
        "masked_fraction": float(np.mean([item["masked_fraction"] for item in records])),
        "selected_by": {name: float(np.mean(values)) for name, values in sorted(strategies.items())},
        "overlap": float(
            np.mean(
                [
                    sum((item.get("selection") or {}).get("strategies", {}).values())
                    - item["n_masked"]
                    for item in records
                ]
            )
        ),
    }


# ============================================================
# ОКРУЖЕНИЕ
# ============================================================


@dataclass(frozen=True)
class Environment:
    """
    Проверенные artifacts: словарь, кандидаты, baseline и хэши.
    """

    root: Path
    vocab_dir: Path
    artifacts_dir: Path
    tokenizer: object
    table: FieldTable
    unigram: UnigramTable
    hashes: dict

    @property
    def vocab(self):
        return self.tokenizer.vocab


def load_environment(root: Path, vocab_dir: Path, artifacts_dir: Path, epsilon: float = 1e-8) -> Environment:

    from src.preprocessing.artifacts import read_json
    from src.tokenizer.artifacts import Tokenizer

    from .data import UNIGRAM_FILE, artifact_hashes

    root, vocab_dir, artifacts_dir = Path(root), Path(vocab_dir), Path(artifacts_dir)

    tokenizer = Tokenizer.load(vocab_dir, artifacts_dir)

    table = FieldTable.load(tokenizer.vocab, vocab_dir)

    unigram = UnigramTable(
        read_json(artifacts_dir / UNIGRAM_FILE), tokenizer.vocab, table, epsilon=epsilon
    )

    return Environment(
        root=root,
        vocab_dir=vocab_dir,
        artifacts_dir=artifacts_dir,
        tokenizer=tokenizer,
        table=table,
        unigram=unigram,
        hashes=artifact_hashes(root, vocab_dir, artifacts_dir),
    )


VALIDATION_SPLITS: tuple[str, ...] = ("val_client", "val_time")


def client_whitelist(config: TrainConfig) -> range | None:
    """
    Клиенты набора: ID меньше client_universe.

    Возвращает None, когда вселенная не задана, и тогда работают
    прежние ограничения на число клиентов в сплите.
    """

    return None if config.client_universe is None else range(int(config.client_universe))


def store_for(
    env: Environment,
    config: TrainConfig,
    split: str,
    shared: ClientStore | None = None,
) -> ClientStore:
    """
    Единая точка создания хранилища: белый список и лимит вместе.
    """

    limit = config.max_train_clients if split == "train" else config.max_val_clients

    return ClientStore(
        env.root,
        split,
        env.vocab_dir,
        max_clients=limit,
        clients=client_whitelist(config),
        sessions=config.uses_sessions,
        shared=shared,
    )


def build_validation(
    env: Environment,
    config: TrainConfig,
    model_config,
    names: tuple[str, ...] = VALIDATION_SPLITS,
    shared: ClientStore | None = None,
) -> tuple[dict[str, FixedSplit], dict[str, dict]]:
    """
    Наборы собираются один раз и больше не меняются.

    При потоковой сборке в памяти остаются не batch'и, а
    рецепт: сами batch'и пересобираются на каждую оценку и
    получаются теми же самыми, потому что схема example
    привязывает маску к примеру.
    """

    splits: dict[str, FixedSplit] = {}
    stores: dict[str, dict] = {}

    masking = config.masking(seed=config.val_seed)

    for name in names:

        store = store_for(env, config, name, shared=shared)

        splits[name] = FixedSplit.build(
            name=name,
            store=store,
            vocab=env.vocab,
            table=env.table,
            model_config=model_config,
            masking=masking,
            max_events=config.max_events_per_history,
            batch_size=config.eval_batch_size,
            structure=model_config.structure,
            stream=config.stream_validation,
        )

        stores[name] = store.summary()

    return splits, stores


# ============================================================
# TINY OVERFIT
# ============================================================


def _last_row_per_client(store: ClientStore, limit: int) -> list[int]:
    """
    По одному примеру на клиента, самый поздний cutoff.
    """

    picked: dict[int, int] = {}

    for index, row in enumerate(store.rows):
        picked[int(row["client_id"])] = index

    return [picked[client_id] for client_id in sorted(picked)][:limit]


def tiny_overfit(
    env: Environment,
    config: TrainConfig,
    out_dir: Path,
    steps: int = 300,
    n_examples: int = 2,
    lr: float = 1e-2,
    max_events: int = 64,
    balanced_share: float = 0.5,
    device: str = "cpu",
    quiet: bool = False,
) -> dict:
    """
    Может ли модель вообще запомнить крошечный фиксированный batch.

    Отдельная модель, dropout ноль, маски зафиксированы. Если
    здесь CE не падает, дело не в данных и не в бюджете, а в
    mapping, masking или градиентах.
    """

    from dataclasses import replace as replace_config

    if not 100 <= steps <= 500:
        raise ValueError("tiny overfit имеет смысл на 100–500 шагах")

    settings = replace_config(
        config,
        dropout=0.0,
        lr=lr,
        warmup_steps=1,
        max_steps=steps,
        batch_size=n_examples,
        max_events_per_history=max_events,
        balanced_share=balanced_share,
    )

    trainer = Trainer(settings, env.tokenizer, env.table, env.unigram, device)

    store = ClientStore(
        env.root,
        "train",
        env.vocab_dir,
        max_clients=n_examples,
        clients=client_whitelist(settings),
    )

    indices = _last_row_per_client(store, n_examples)

    examples = store.examples(indices)

    inputs, targets, history = trainer.prepare(examples, mask_step=0)

    # --------------------------------------------------------
    # ПРОВЕРКА ПРИГОДНОСТИ BATCH
    # --------------------------------------------------------

    distinct = {
        int(key): int(np.unique(targets.local_targets[targets.key_ids == key]).size)
        for key in np.unique(targets.key_ids)
    }

    varied = sum(1 for count in distinct.values() if count >= 2)

    if targets.n_fields < 3 or varied == 0:
        raise TrainingAborted(
            f"batch не годится для проверки: полей с целями {targets.n_fields}, "
            f"полей с разными значениями {varied}; константные цели ничего не докажут"
        )

    # --------------------------------------------------------

    def measure() -> dict:

        was_training = trainer.backbone.training

        trainer.eval_mode()

        with torch.inference_mode():

            result, field_logits, local = trainer.compute(inputs, targets)

            correct = sum(
                int((item.logits.argmax(dim=-1) == local[item.index]).sum()) for item in field_logits
            )

        if was_training:
            trainer.train_mode()

        return {
            "field_balanced": float(result.field_balanced),
            "token_weighted": float(result.token_weighted),
            "accuracy": correct / targets.n,
        }

    trainer.train_mode()

    before = measure()

    history_log: list[dict] = []

    started = time.perf_counter()

    for _ in range(steps):

        trainer.n_batches += 1

        result = trainer.optimize(inputs, targets)

        if result.skipped:
            raise TrainingAborted(f"фиксированный batch внезапно без целей: {result.reason}")

        if trainer.n_steps % max(1, steps // 10) == 0:
            history_log.append(
                {"step": trainer.n_steps, "loss": result.field_balanced, "grad_norm": result.grad_norm}
            )

    seconds = time.perf_counter() - started

    after = measure()

    drop = (before["field_balanced"] - after["field_balanced"]) / before["field_balanced"]

    report = {
        "mode": "overfit",
        "device": str(trainer.device),
        "precision": trainer.precision,
        "steps": steps,
        "seconds": seconds,
        "settings": {
            "lr": lr,
            "dropout": 0.0,
            "n_examples": len(examples),
            "max_events_per_history": max_events,
            "balanced_share": balanced_share,
            "masking": settings.masking().as_dict(),
        },
        "batch": {
            "clients": [int(example.client_id) for example in examples],
            "cutoffs": [example.cutoff.isoformat() for example in examples],
            "events_used": [int(value) for value in history.info.used_history_length],
            "n_targets": targets.n,
            "n_fields": targets.n_fields,
            "n_degenerate_skipped": targets.n_degenerate,
            "distinct_targets_by_field": {
                env.table.name(key): count for key, count in sorted(distinct.items())
            },
            "fields_with_varied_targets": varied,
        },
        "before": before,
        "after": after,
        "ce_drop": drop,
        "log": history_log,
        "thresholds": {"ce_drop": 0.8, "accuracy": 0.9},
        "passed": bool(drop >= 0.8 and after["accuracy"] >= 0.9),
    }

    out_dir = Path(out_dir)

    write_json(out_dir / "overfit.json", report)
    write_text(out_dir / "overfit.txt", render_overfit(report))

    if not quiet:
        print(render_overfit(report))

    return report


def render_overfit(report: dict) -> str:

    lines = ["=" * 72, "TINY OVERFIT", "=" * 72]

    lines.append(f"  устройство {report['device']}, precision {report['precision']}, шагов {report['steps']}")

    batch = report["batch"]

    lines.append(
        f"  клиенты {batch['clients']}, событий {batch['events_used']}, "
        f"целей {batch['n_targets']} в {batch['n_fields']} полях"
    )
    lines.append(f"  полей с разными значениями целей: {batch['fields_with_varied_targets']}")
    lines.append("")

    header = f"  {'':<20s}{'field-balanced CE':>20s}{'token-weighted CE':>20s}{'accuracy':>12s}"

    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for name, key in (("до обучения", "before"), ("после обучения", "after")):
        item = report[key]
        lines.append(
            f"  {name:<20s}{item['field_balanced']:>20.4f}{item['token_weighted']:>20.4f}"
            f"{item['accuracy']:>12.3f}"
        )

    lines.append("")
    lines.append(f"  падение CE {report['ce_drop'] * 100:.1f} % при пороге 80 %")
    lines.append(f"  accuracy {report['after']['accuracy'] * 100:.1f} % при пороге 90 %")
    lines.append("")
    lines.append(f"  ВЕРДИКТ: {'пройдено' if report['passed'] else 'НЕ ПРОЙДЕНО'}")

    return "\n".join(lines)


# ============================================================
# BENCHMARK
# ============================================================


def benchmark(
    env: Environment,
    config: TrainConfig,
    out_dir: Path,
    warmup: int = 5,
    measured: int = 20,
    device: str = "cpu",
    quiet: bool = False,
) -> dict:
    """
    Сколько стоит полный шаг именно в той конфигурации, что пойдёт в run.
    """

    if warmup < 5 or measured < 20:
        raise ValueError("нужно минимум 5 разогревочных и 20 измеряемых шагов")

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, device)

    started = time.perf_counter()

    store = store_for(env, config, "train")

    splits, stores = build_validation(env, config, trainer.model_config, shared=store)

    load_seconds = time.perf_counter() - started

    sampler = EpochSampler(len(store), config.batch_size, config.seed)

    trainer.train_mode()

    # Самые длинные истории набора: обычный batch про среднюю
    # стоимость шага ничего не говорит о худшем случае, а
    # упереться в память можно именно на нём.
    longest = sorted(
        range(len(store)), key=lambda index: -int(store.rows[index]["seq_end"])
    )[: config.batch_size * max(1, measured // 4)]

    original: list[np.ndarray] = []
    used: list[np.ndarray] = []
    truncated: list[np.ndarray] = []

    step_seconds: list[float] = []

    masking: list[StepResult] = []

    targets_seen = 0

    for index in range(warmup + measured):

        examples = store.examples(sampler.next_batch())

        _sync(trainer.device)

        if index == warmup and trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

        moment = time.perf_counter()

        inputs, targets, history = trainer.prepare(examples, trainer.n_batches)

        trainer.n_batches += 1

        result = trainer.optimize(inputs, targets)

        _sync(trainer.device)

        if index >= warmup:
            step_seconds.append(time.perf_counter() - moment)
            original.append(history.info.original_history_length)
            used.append(history.info.used_history_length)
            truncated.append(history.info.truncated)
            targets_seen += result.n_targets
            masking.append(replace(result, masking=history.masking))

    peak_train = _memory(trainer.device)

    # --------------------------------------------------------
    # ХУДШИЙ СЛУЧАЙ
    # --------------------------------------------------------

    _sync(trainer.device)

    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)

    worst_seconds: list[float] = []
    worst_events: list[int] = []

    worst_failure = None

    try:

        for start in range(0, len(longest), config.batch_size):

            examples = store.examples(longest[start : start + config.batch_size])

            _sync(trainer.device)

            moment = time.perf_counter()

            inputs, targets, history = trainer.prepare(examples, trainer.n_batches)

            trainer.n_batches += 1

            trainer.optimize(inputs, targets)

            _sync(trainer.device)

            worst_seconds.append(time.perf_counter() - moment)
            worst_events.append(int(history.info.used_history_length.sum()))

    except torch.OutOfMemoryError as error:
        worst_failure = {
            "message": str(error).splitlines()[0],
            "batch_size": config.batch_size,
            "event_microbatch": config.event_microbatch,
            "longest_history": int(max(int(store.rows[i]["seq_end"]) for i in longest)),
        }

    worst = {
        "batches": len(worst_seconds),
        "seconds_per_step": float(np.mean(worst_seconds)) if worst_seconds else None,
        "seconds_per_step_max": float(np.max(worst_seconds)) if worst_seconds else None,
        "events_per_step": float(np.mean(worst_events)) if worst_events else None,
        "longest_history": int(max(int(store.rows[i]["seq_end"]) for i in longest)),
        "peak_mb": _memory(trainer.device),
        "failure": worst_failure,
    }

    # --------------------------------------------------------
    # VALIDATION И CHECKPOINT ОТДЕЛЬНО
    # --------------------------------------------------------

    _sync(trainer.device)

    reports = trainer.evaluate(splits, exclude=trainer.excluded)

    eval_seconds = reports.pop("_seconds")

    moment = time.perf_counter()

    save_checkpoint(
        Path(out_dir) / "benchmark.pt",
        backbone=trainer.backbone,
        head=trainer.head,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        counters=trainer.counters(),
        train_config=config.as_dict(),
        model_config=trainer.model_config.as_dict(),
        masking_config=config.masking().as_dict(),
        sampler_state=sampler.state(),
        splits={name: split.description() for name, split in splits.items()},
        artifacts=env.hashes,
    )

    checkpoint_seconds = time.perf_counter() - moment

    per_step = float(np.mean(step_seconds))

    def estimate(steps: int) -> dict:

        evaluations = steps // config.eval_every + 2

        total = steps * per_step + evaluations * (eval_seconds + checkpoint_seconds)

        return {
            "steps": steps,
            "evaluations": evaluations,
            "train_seconds": steps * per_step,
            "validation_seconds": evaluations * eval_seconds,
            "checkpoint_seconds": evaluations * checkpoint_seconds,
            "total_seconds": total,
        }

    report = {
        "mode": "benchmark",
        "device": str(trainer.device),
        "precision": trainer.precision,
        "parameters": {
            "backbone": trainer.backbone.n_parameters(),
            "head": trainer.head.n_parameters(),
            "total": trainer.n_parameters(),
        },
        "settings": {
            "batch_size": config.batch_size,
            "event_microbatch": config.event_microbatch,
            "max_events_per_history": config.max_events_per_history,
            "eval_batch_size": config.eval_batch_size,
            "eval_every": config.eval_every,
            "warmup": warmup,
            "measured": measured,
        },
        "data": {"train": store.summary(), **stores},
        "load_seconds": load_seconds,
        "history": {
            "original": _lengths(np.concatenate(original)),
            "used": _lengths(np.concatenate(used)),
            "truncated_share": float(np.concatenate(truncated).mean()),
        },
        "targets_per_step": targets_seen / measured,
        "masking": masking_summary(masking),
        "worst_case": worst,
        "steps_per_epoch": math.ceil(
            math.ceil(len(store) / config.batch_size) / config.accumulation_steps
        ),
        "epoch_examples": len(store),
        "seconds_per_step": per_step,
        "seconds_per_step_p90": float(np.percentile(step_seconds, 90)),
        "validation_seconds": eval_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "peak_train_mb": peak_train,
        "estimates": [estimate(value) for value in (1000, 2000, 5000, 10000)],
        "validation_summary": {
            name: {
                "n_targets": item["n_targets"],
                "field_balanced_ce": item["field_balanced_ce"],
            }
            for name, item in reports.items()
        },
    }

    write_json(Path(out_dir) / "benchmark.json", report)
    write_text(Path(out_dir) / "benchmark.txt", render_benchmark(report))

    if not quiet:
        print(render_benchmark(report))

    return report


def render_benchmark(report: dict) -> str:

    lines = ["=" * 78, "ЗАМЕР ОБУЧЕНИЯ", "=" * 78]

    def row(label: str, value) -> None:
        lines.append(f"  {label:<40s}{str(value):>28}")

    row("устройство", report["device"])
    row("precision", report["precision"])
    row("параметров", f"{report['parameters']['total']:,}".replace(",", " "))
    row("batch size", report["settings"]["batch_size"])
    row("event microbatch", report["settings"]["event_microbatch"])
    row("max_events_per_history", report["settings"]["max_events_per_history"])
    lines.append("")

    row("train клиентов / примеров", f"{report['data']['train']['clients']} / {report['data']['train']['examples']}")

    for name in ("val_client", "val_time"):
        if name in report["data"]:
            item = report["data"][name]
            row(f"{name} клиентов / примеров", f"{item['clients']} / {item['examples']}")

    lines.append("")

    history = report["history"]

    row("исходная длина p50 / p90 / max",
        f"{history['original']['p50']} / {history['original']['p90']} / {history['original']['max']}")
    row("использовано p50 / p90 / max",
        f"{history['used']['p50']} / {history['used']['p90']} / {history['used']['max']}")
    row("доля обрезанных историй", f"{history['truncated_share']:.2f}")
    row("целей на шаг", f"{report['targets_per_step']:.1f}")

    if report.get("masking"):
        masking = report["masking"]
        row("режим маскирования", masking["mode"])
        row("доступных значений на шаг", f"{masking['eligible']:.0f}")
        row("скрыто, доля", f"{masking['masked_fraction']:.3f}")
        row(
            "выбрано стратегиями",
            ", ".join(f"{name} {value:.0f}" for name, value in masking["selected_by"].items()),
        )
        row("пересечений выборок", f"{masking['overlap']:.0f}")

    lines.append("")

    row("секунд на шаг", f"{report['seconds_per_step']:.3f}")
    row("секунд на шаг, p90", f"{report['seconds_per_step_p90']:.3f}")
    row("одна validation, с", f"{report['validation_seconds']:.2f}")
    row("одно сохранение checkpoint, с", f"{report['checkpoint_seconds']:.2f}")
    row("загрузка данных, с", f"{report['load_seconds']:.1f}")

    if report["peak_train_mb"] is not None:
        row("пик GPU на обучении, МБ", f"{report['peak_train_mb']:.0f}")

    worst = report.get("worst_case")

    if worst:

        lines.append("")
        lines.append("  самые длинные истории набора")

        row("длиннейшая история, событий", worst["longest_history"])
        row("событий на шаг", "—" if worst["events_per_step"] is None else f"{worst['events_per_step']:.0f}")
        row("секунд на шаг", "—" if worst["seconds_per_step"] is None else f"{worst['seconds_per_step']:.3f}")
        row("секунд на шаг, максимум",
            "—" if worst["seconds_per_step_max"] is None else f"{worst['seconds_per_step_max']:.3f}")

        if worst["peak_mb"] is not None:
            row("пик GPU, МБ", f"{worst['peak_mb']:.0f}")

        if worst["failure"]:
            lines.append(f"  НЕ ПОМЕСТИЛОСЬ: {worst['failure']['message']}")

    lines.append("")

    row("примеров в эпохе", report.get("epoch_examples", "—"))
    row("шагов в эпохе", report.get("steps_per_epoch", "—"))

    if report.get("steps_per_epoch"):

        evaluations = report["steps_per_epoch"] // report["settings"]["eval_every"] + 2

        total = (
            report["steps_per_epoch"] * report["seconds_per_step"]
            + evaluations * (report["validation_seconds"] + report["checkpoint_seconds"])
        )

        row("ожидаемое время эпохи, мин", f"{total / 60:.1f}")
        row("  из них validation, мин", f"{evaluations * report['validation_seconds'] / 60:.1f}")

    lines.append("")

    header = f"  {'шагов':>8s}{'обучение':>12s}{'validation':>12s}{'checkpoint':>12s}{'всего, мин':>13s}"

    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for item in report["estimates"]:
        lines.append(
            f"  {item['steps']:>8,}{item['train_seconds']:>12.0f}{item['validation_seconds']:>12.0f}"
            f"{item['checkpoint_seconds']:>12.0f}{item['total_seconds'] / 60:>13.1f}".replace(",", " ")
        )

    return "\n".join(lines)


# ============================================================
# КОРОТКИЙ RUN
# ============================================================


BEST_SPLIT = "val_time"
BEST_METRIC = "field_balanced_ce"


def best_value(reports: dict, scope: str) -> float | None:
    """
    Значение критерия best у набора BEST_SPLIT.
    """

    item = reports.get(BEST_SPLIT)

    if item is None:
        return None

    if scope == BEST_SCOPE_RECENT:
        item = item.get("recent") or {}

    return item.get(BEST_METRIC)


def run_training(
    env: Environment,
    config: TrainConfig,
    out_dir: Path,
    device: str = "cpu",
    quiet: bool = False,
    preflight: bool = False,
    resume: Path | None = None,
) -> dict:
    """
    Один эксперимент: либо фиксированный бюджет шагов, либо эпоха.

    resume продолжает ТОТ ЖЕ прогон: остаток эпохи, то же
    расписание оценок, тот же лучший результат. Уже пройденные
    примеры второй раз не проходятся.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume = Path(resume) if resume else None

    # Состояние читается ДО всего тяжёлого: несовместимость
    # должна стоить секунды, а не загрузку набора.
    saved = verify_checkpoint(resume) if resume is not None else None

    log = JsonLog(out_dir / "log.jsonl", append=resume is not None)

    started_at = time.time()

    write_atomic(out_dir / PID_FILE, f"{os.getpid()}" + chr(10))

    write_atomic(out_dir / COMMAND_FILE, " ".join(sys.argv) + chr(10))

    # Просьба остановиться от прошлого запуска не должна
    # останавливать новый.
    (out_dir / STOP_FILE).unlink(missing_ok=True)

    trainer = Trainer(config, env.tokenizer, env.table, env.unigram, device)

    # Совместимость проверяется до загрузки набора: несовпадение
    # seed, политики целей или словаря должно стоить секунды,
    # а не полчаса чтения событий.
    if saved is not None:
        check_resume(
            saved,
            train_config=config.as_dict(),
            model_config=trainer.model_config.as_dict(),
            masking_config=config.masking().as_dict(),
            artifacts=env.hashes,
        )

    started = time.perf_counter()

    store = store_for(env, config, "train")

    splits, stores = build_validation(env, config, trainer.model_config, shared=store)

    load_seconds = time.perf_counter() - started

    sampler = EpochSampler(len(store), config.batch_size, config.seed)

    descriptions = {name: split.description() for name, split in splits.items()}

    sizes = {"train": store.summary(), **stores}

    # --------------------------------------------------------
    # БЮДЖЕТ
    # --------------------------------------------------------

    micro_per_epoch = math.ceil(len(store) / config.batch_size)

    steps_per_epoch = math.ceil(micro_per_epoch / config.accumulation_steps)

    # Эпоха меряется в batch'ах, а не в успешных шагах: batch без
    # целей шагом не считается, и по шагам граница эпохи уехала
    # бы, оставив часть примеров непройденной.
    micro_budget = micro_per_epoch * config.epochs if config.epochs else None

    # --------------------------------------------------------
    # ПРОДОЛЖЕНИЕ
    # --------------------------------------------------------

    restored: dict = {}

    if saved is not None:

        check_resume_splits(saved, descriptions)

        load_checkpoint(
            resume,
            backbone=trainer.backbone,
            head=trainer.head,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            sampler=sampler,
            map_location=str(trainer.device),
            restore_random=True,
        )

        trainer.load_counters(saved["counters"])

        restored = dict(saved.get("progress") or {})

        if not restored:
            raise TrainingAborted(
                f"{resume}: в checkpoint нет секции progress, "
                "продолжить эпоху с него нельзя"
            )

    checks = truncation_check(store, config) if preflight and saved is None else None

    if checks is not None:
        # Флаг токенизатора это НЕ обрезка: он лишь помечает
        # истории длиннее лимита манифеста. Стоит рядом, чтобы
        # два разных числа не путали друг с другом.
        checks["histories_over_manifest_limit"] = int(
            sum(1 for row in store.rows if row.get("history_over_limit"))
        )
        checks["note"] = (
            "обрезка это used < original; history_over_limit это отметка "
            "токенизатора о длине истории и обрезкой не является"
        )

    if checks is not None and not checks["passed"]:
        raise TrainingAborted(
            f"обрезка истории не отключена: обрезано {checks['n_truncated']} примеров из "
            f"{checks['n_examples']} при max_events_per_history={config.max_events_per_history}"
        )

    if not quiet:
        print(f"устройство {trainer.device}, precision {trainer.precision}, "
              f"параметров {trainer.n_parameters():,}".replace(",", " "))
        for name, item in sizes.items():
            print(f"  {name:<12s} клиентов {item['clients']:>5}  примеров {item['examples']:>7,}"
                  .replace(",", " "))
        for name, split in splits.items():
            print(f"  {name:<12s} batch {split.n_batches:>5}  целей {split.n_targets:>7,}"
                  .replace(",", " "))

        if config.epochs:
            print(f"  эпоха: {micro_per_epoch:,} batch'ей, {steps_per_epoch:,} шагов оптимизатора"
                  .replace(",", " "))

        if checks is not None:
            print(f"  обрезка: {checks['n_truncated']} примеров, длина p50 "
                  f"{checks['lengths']['p50']}, p95 {checks['lengths']['p95']}, "
                  f"max {checks['lengths']['max']}")

        print()

    log.write({"event": "start", "sizes": sizes, "splits": descriptions,
               "config": config.as_dict(), "device": str(trainer.device),
               "precision": trainer.precision, "load_seconds": load_seconds,
               "steps_per_epoch": steps_per_epoch, "micro_per_epoch": micro_per_epoch,
               "truncation_check": checks})

    # --------------------------------------------------------

    evaluations: list[dict] = list(restored.get("evaluations") or [])

    best = dict(restored.get("best") or {"step": None, "value": None, "reason": None})

    evaluated_steps: set[int] = {int(value) for value in restored.get("evaluated_steps") or []}

    train_log: list[dict] = list(restored.get("train_log") or [])

    final: dict | None = restored.get("final")

    micro_done = int(restored.get("micro_done") or 0)

    saved_checkpoint = {"path": None, "step": None, "at": None}

    def progress_state() -> dict:
        """
        Всё, что нужно, чтобы продолжить именно этот прогон.
        """

        return {
            "micro_done": micro_done,
            "micro_budget": micro_budget,
            "micro_per_epoch": micro_per_epoch,
            "evaluations": evaluations,
            "evaluated_steps": sorted(evaluated_steps),
            "best": best,
            "train_log": train_log,
            "final": final,
            "started_at": started_at,
        }

    def checkpoint(path: Path, metrics: dict | None, reason: str = "") -> None:

        save_checkpoint(
            path,
            backbone=trainer.backbone,
            head=trainer.head,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            counters=trainer.counters(),
            train_config=config.as_dict(),
            model_config=trainer.model_config.as_dict(),
            masking_config=config.masking().as_dict(),
            sampler_state=sampler.state(),
            splits=descriptions,
            artifacts=env.hashes,
            metrics=metrics,
            progress=progress_state(),
        )

        if Path(path).name == "last.pt":
            saved_checkpoint.update(
                {"path": str(path), "step": trainer.n_steps, "at": time.time()}
            )

        log.write({
            "event": "checkpoint",
            "file": Path(path).name,
            "step": trainer.n_steps,
            "micro_done": micro_done,
            "reason": reason,
        })

    def evaluate(step: int) -> dict:

        # Сохранение ДО оценки: она стоит десятки минут, и
        # упасть в ней значит потерять всё после последнего
        # шага, а не только саму оценку.
        checkpoint(out_dir / "last.pt", None, reason="before_eval")

        reports = trainer.evaluate(splits, exclude=trainer.excluded)

        seconds = reports.pop("_seconds")

        record = {"step": step, "seconds": seconds, "metrics": reports}

        evaluations.append(record)

        log.write({
            "event": "eval",
            "step": step,
            "seconds": seconds,
            "summary": {
                name: {
                    "n_targets": item["n_targets"],
                    "field_balanced_ce": item["field_balanced_ce"],
                    "token_weighted_ce": item["token_weighted_ce"],
                    "mean_nce_gain": item["mean_nce_gain"],
                    "accuracy": item["accuracy"],
                }
                for name, item in reports.items()
            },
        })

        if not quiet:
            for name, item in reports.items():
                print(f"  eval  step {step:>5}  {name:<12s} "
                      f"CE {_text(item['field_balanced_ce'])}  "
                      f"CE uni {_text(item['field_balanced_ce_unigram'])}  "
                      f"NCE {_text(item['mean_nce_gain'])}  "
                      f"Acc {_text(item['accuracy'])}  целей {item['n_targets']:,}".replace(",", " "))

        evaluated_steps.add(int(step))

        value = best_value(reports, config.best_metric)

        if value is None:
            best["reason"] = (
                f"у набора {BEST_SPLIT} нет целей в срезе {config.best_metric}: "
                "критерий best не применим, автоматической замены критерия нет"
            )
        elif best["value"] is None or value < best["value"]:
            best.update({"step": step, "value": value, "reason": None})
            # best.pt обновляется только после УСПЕШНОЙ оценки
            # и только на улучшении: иначе он перестал бы быть
            # лучшим и стал бы просто последним.
            checkpoint(out_dir / "best.pt", record, reason="best")

        checkpoint(out_dir / "last.pt", record, reason="after_eval")

        status(step=trainer.n_steps, phase="training")

        return reports

    # --------------------------------------------------------

    def status(step: int, phase: str, extra: dict | None = None) -> None:
        """
        Короткий файл «где я сейчас», переписываемый атомарно.
        """

        elapsed = time.time() - started_at

        share = micro_done / micro_budget if micro_budget else None

        write_status(out_dir, {
            "pid": os.getpid(),
            "phase": phase,
            "started_at": started_at,
            "elapsed_seconds": round(elapsed, 1),
            "step": step,
            "steps_per_epoch": steps_per_epoch,
            "micro_done": micro_done,
            "micro_per_epoch": micro_per_epoch,
            "epoch_share": round(share, 4) if share is not None else None,
            "eta_seconds": (
                round(elapsed * (1 - share) / share, 0) if share else None
            ),
            "last_checkpoint": saved_checkpoint,
            "last_validation_step": max(evaluated_steps) if evaluated_steps else None,
            "next_validation_step": (
                (step // config.eval_every + 1) * config.eval_every
                if config.eval_every
                else None
            ),
            "best": best,
            "device": str(trainer.device),
            "precision": trainer.precision,
            "gpu_mb": _memory(trainer.device),
            "resumed_from": str(resume) if resume else None,
            "config": {
                "name": str(out_dir),
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "eval_every": config.eval_every,
                "checkpoint_every": config.checkpoint_every,
                "structure": trainer.model_config.structure,
                "target_policy": config.target_policy,
                "mask_scheme": config.mask_scheme,
            },
            **(extra or {}),
        })

    trainer.train_mode()

    if resume is None:
        before = evaluate(0)
    else:
        # Оценка шага 0 уже сделана в этом прогоне: повторять её
        # значит потратить десятки минут на известный результат.
        before = evaluations[0]["metrics"] if evaluations else {}

        log.write({
            "event": "resume",
            "from": str(resume),
            "step": trainer.n_steps,
            "micro_done": micro_done,
            "micro_budget": micro_budget,
            "sampler_position": sampler.position,
            "evaluated_steps": sorted(evaluated_steps),
            "best": best,
            "final_done": final is not None,
        })

        if not quiet:
            print(
                f"продолжение с {resume}: шаг {trainer.n_steps}, "
                f"пройдено {micro_done:,} из {micro_budget or '?'} batch'ей, "
                f"оценки на шагах {sorted(evaluated_steps)}".replace(",", " ")
            )
            print()

    window: list[StepResult] = []
    seen: list[StepResult] = []

    last_batch: dict = {}

    interrupted: dict | None = None

    train_started = time.perf_counter()

    status(step=trainer.n_steps, phase="training")

    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)

    try:

        while (micro_done < micro_budget) if micro_budget else (trainer.n_steps < config.max_steps):

            groups = []

            for _ in range(config.accumulation_steps):

                if micro_budget is not None and micro_done >= micro_budget:
                    break

                indices = sampler.next_batch()

                groups.append(store.examples(indices))

                micro_done += 1

                last_batch = {
                    "rows": [int(value) for value in indices],
                    "clients": [int(example.client_id) for example in groups[-1]],
                    "cutoffs": [example.cutoff.isoformat() for example in groups[-1]],
                    "events": [int(example.events.n_events) for example in groups[-1]],
                }

            result = trainer.train_group(groups)

            window.append(result)
            seen.append(result)

            if result.skipped:
                log.write({"event": "skip", "batch": result.batch, "reason": result.reason,
                           "batch_context": last_batch})
                continue

            finished = (micro_done >= micro_budget) if micro_budget else (trainer.n_steps == config.max_steps)

            if trainer.n_steps % config.log_every == 0 or finished:

                done = [item for item in window if not item.skipped]

                record = {
                    "event": "train",
                    "step": trainer.n_steps,
                    "batches": trainer.n_batches,
                    "skipped": sum(1 for item in window if item.skipped),
                    "field_balanced": float(np.mean([item.field_balanced for item in done])),
                    "token_weighted": float(np.mean([item.token_weighted for item in done])),
                    "learning_rate": result.learning_rate,
                    "masked_targets": float(np.mean([item.n_targets for item in done])),
                    "fields_in_batch": float(np.mean([item.n_fields for item in done])),
                    "grad_norm": float(np.mean([item.grad_norm for item in done])),
                    "seconds_per_step": float(np.mean([item.seconds for item in done])),
                    "gpu_mb": _memory(trainer.device),
                    "masking": masking_summary(window),
                }

                elapsed = time.time() - started_at

                share = micro_done / micro_budget if micro_budget else None

                record.update({
                    "micro_done": micro_done,
                    "micro_budget": micro_budget,
                    "steps_per_epoch": steps_per_epoch,
                    "epoch_share": round(share, 4) if share is not None else None,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round(elapsed * (1 - share) / share, 0) if share else None,
                    "last_checkpoint_step": saved_checkpoint["step"],
                    "last_checkpoint_seconds_ago": (
                        round(time.time() - saved_checkpoint["at"], 1)
                        if saved_checkpoint["at"]
                        else None
                    ),
                })

                train_log.append(record)

                log.write(record)

                status(step=trainer.n_steps, phase="training")

                if not quiet:
                    print(
                        f"  step {record['step']:>6}/{steps_per_epoch}"
                        f" {(share or 0) * 100:5.1f}%  loss {record['field_balanced']:.4f}  "
                        f"lr {record['learning_rate']:.2e}  |g| {record['grad_norm']:.2f}  "
                        f"{record['seconds_per_step']:.3f} с/шаг  "
                        f"прошло {elapsed / 60:.0f} мин  "
                        f"ETA {(record['eta_seconds'] or 0) / 60:.0f} мин"
                        + (f"  {record['gpu_mb']:.0f} МБ" if record["gpu_mb"] is not None else "")
                        + (
                            f"  checkpoint {record['last_checkpoint_seconds_ago']:.0f} с назад"
                            if record["last_checkpoint_seconds_ago"] is not None
                            else "  checkpoint ещё не писали"
                        )
                    )

                window = []

            # Сохранение по расписанию, независимо от оценок:
            # между оценками десятки тысяч шагов, и терять их
            # из-за сбоя нельзя.
            if config.checkpoint_every and trainer.n_steps % config.checkpoint_every == 0:
                checkpoint(out_dir / "last.pt", None, reason="periodic")

            if trainer.n_steps % config.eval_every == 0 and not finished:
                if trainer.n_steps not in evaluated_steps:
                    evaluate(trainer.n_steps)

            if stop_requested(out_dir):
                raise TrainingInterrupted("получен файл остановки stop.request")

    except (KeyboardInterrupt, TrainingInterrupted) as error:

        # Это не сбой: текущий шаг досчитан, состояние цело.
        # Сохраняем и выходим, финальную оценку не запускаем.
        interrupted = {
            "reason": "stop.request" if isinstance(error, TrainingInterrupted) else "KeyboardInterrupt",
            "message": str(error),
            "step": trainer.n_steps,
            "counters": trainer.counters(),
            "micro_done": micro_done,
            "micro_budget": micro_budget,
            "sampler_position": sampler.position,
            "batch": last_batch,
        }

        try:
            checkpoint(out_dir / "interrupted.pt", None, reason="interrupted")
            interrupted["checkpoint"] = str(out_dir / "interrupted.pt")
        except BaseException as failure:  # noqa: BLE001
            interrupted["checkpoint_error"] = f"{type(failure).__name__}: {failure}"

        log.write({"event": "interrupted", **interrupted})

        status(step=trainer.n_steps, phase="interrupted", extra={"interrupted": interrupted})

        write_json(out_dir / "interrupted.json", interrupted)

        (out_dir / STOP_FILE).unlink(missing_ok=True)

        if not quiet:
            print()
            print(
                f"остановлено на шаге {trainer.n_steps} "
                f"({interrupted['reason']}); состояние в "
                f"{out_dir / 'interrupted.pt'}"
            )

        raise TrainingInterrupted(
            f"обучение остановлено на шаге {trainer.n_steps}: {interrupted['reason']}; "
            f"продолжить: --resume {out_dir / 'interrupted.pt'}"
        ) from error

    except (TrainingAborted, torch.OutOfMemoryError, RuntimeError) as error:

        last_step = train_log[-1] if train_log else {}

        diagnostics = {
            "error": type(error).__name__,
            "message": str(error).splitlines()[0] if str(error) else "",
            "step": trainer.n_steps,
            "counters": trainer.counters(),
            "micro_done": micro_done,
            "micro_budget": micro_budget,
            "sampler_position": sampler.position,
            "batch": last_batch,
            "loss": last_step.get("field_balanced"),
            "grad_norm": last_step.get("grad_norm"),
            "learning_rate": last_step.get("learning_rate"),
            "seconds_per_step": last_step.get("seconds_per_step"),
            "device": str(trainer.device),
            "precision": trainer.precision,
            "config": config.as_dict(),
            "model": trainer.model_config.as_dict(),
            "gpu_mb": _memory(trainer.device),
            "gpu_reserved_mb": (
                torch.cuda.max_memory_reserved(trainer.device) / (1 << 20)
                if trainer.device.type == "cuda"
                else None
            ),
            "resumed_from": str(resume) if resume else None,
            "note": (
                "batch size и обрезка истории внутри начатого прогона не меняются: "
                "это была бы другая задача под тем же именем"
            ),
        }

        write_json(out_dir / "diagnostics.json", diagnostics)

        log.write({"event": "abort", **diagnostics})

        try:
            checkpoint(out_dir / "crash.pt", None, reason="crash")
            diagnostics["checkpoint"] = str(out_dir / "crash.pt")
        except BaseException:  # noqa: BLE001 - диагностика важнее второй ошибки
            diagnostics["checkpoint"] = None

        status(step=trainer.n_steps, phase="crashed", extra={"diagnostics": diagnostics})

        raise TrainingAborted(
            f"обучение остановлено на шаге {trainer.n_steps}: {diagnostics['message']}; "
            f"диагностика в {out_dir / 'diagnostics.json'}"
        ) from error

    train_seconds = time.perf_counter() - train_started

    # Конец эпохи: сохранение ДО финальной оценки, чтобы
    # обученные веса пережили сбой в самой оценке.
    checkpoint(out_dir / "last.pt", None, reason="epoch_end")

    status(step=trainer.n_steps, phase="final_validation")

    after = (
        evaluations[-1]["metrics"]
        if trainer.n_steps in evaluated_steps and evaluations
        else evaluate(trainer.n_steps)
    )

    # --------------------------------------------------------
    # ФИНАЛЬНАЯ ОЦЕНКА
    # --------------------------------------------------------
    #
    # Один раз, лучшими весами, на наборах, которые по ходу
    # обучения не смотрели. Отдельный шаг именно поэтому:
    # test, увиденный на каждой оценке, перестаёт быть test.

    if config.final_splits and final is None:

        checkpoint(out_dir / "last.pt", None, reason="before_final")

        status(step=trainer.n_steps, phase="final_splits")

        best_path = out_dir / "best.pt"

        if best_path.exists():
            load_checkpoint(
                best_path,
                backbone=trainer.backbone,
                head=trainer.head,
                model_config=trainer.model_config.as_dict(),
                artifacts=env.hashes,
                restore_random=False,
            )

        final_splits, final_sizes = build_validation(
            env, config, trainer.model_config, names=config.final_splits, shared=store
        )

        final_reports = trainer.evaluate(final_splits, exclude=trainer.excluded)

        final_reports.pop("_seconds", None)

        final = {
            "weights": "best.pt" if best_path.exists() else "last.pt",
            "step": best["step"],
            "sizes": final_sizes,
            "splits": {name: split.description() for name, split in final_splits.items()},
            "metrics": final_reports,
        }

        log.write({"event": "final", "weights": final["weights"], "step": best["step"],
                   "summary": {
                       name: {
                           "n_targets": item["n_targets"],
                           "field_balanced_ce": item["field_balanced_ce"],
                           "recent_field_balanced_ce": item["recent"]["field_balanced_ce"],
                       }
                       for name, item in final_reports.items()
                   }})

    # --------------------------------------------------------

    report = {
        "mode": "run",
        "device": str(trainer.device),
        "precision": trainer.precision,
        "parameters": {
            "backbone": trainer.backbone.n_parameters(),
            "head": trainer.head.n_parameters(),
            "total": trainer.n_parameters(),
        },
        "config": config.as_dict(),
        "model": trainer.model_config.as_dict(),
        "masking": {
            "train": config.masking().as_dict(),
            "validation": config.masking(seed=config.val_seed).as_dict(),
        },
        "targets": {
            "policy": config.target_policy,
            "note": POLICY_NOTES[config.target_policy],
            "excluded": sorted(trainer.excluded),
        },
        "data": sizes,
        "splits": descriptions,
        "table": env.table.as_dict(),
        "unigram": env.unigram.as_dict(),
        "artifacts": env.hashes,
        "counters": trainer.counters(),
        "budget": {
            "epochs": config.epochs,
            "epoch_examples": len(store),
            "micro_per_epoch": micro_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "micro_batches_done": micro_done,
            "accumulation_steps": config.accumulation_steps,
        },
        "truncation_check": checks,
        "masking_diagnostics": masking_summary(seen),
        "load_seconds": load_seconds,
        "train_seconds": train_seconds,
        "peak_gpu_mb": _memory(trainer.device),
        "train_log": train_log,
        "evaluations": evaluations,
        "resumed_from": str(resume) if resume else None,
        "before": before,
        "after": after,
        "final": final,
        "best": best,
        "checkpoints": {
            "last": str(out_dir / "last.pt"),
            "best": str(out_dir / "best.pt") if best["step"] is not None else None,
        },
    }

    # Последняя запись состояния: в ней уже есть final, поэтому
    # повторный resume финальные наборы второй раз не считает.
    checkpoint(out_dir / "last.pt", None, reason="finished")

    status(step=trainer.n_steps, phase="finished")

    write_json(out_dir / "report.json", report)
    write_text(out_dir / "report.md", render_run(report))

    for name in splits:
        write_text(out_dir / f"metrics_{name}.txt", render_metrics(after[name], f"{name}: ПОСЛЕ ОБУЧЕНИЯ"))

    if not quiet:
        print()
        print(render_run(report))

    return report


def _text(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


AGGREGATES: tuple[tuple[str, str], ...] = (
    ("field-balanced CE", "field_balanced_ce"),
    ("field-balanced CE unigram", "field_balanced_ce_unigram"),
    ("token-weighted CE", "token_weighted_ce"),
    ("token-weighted CE unigram", "token_weighted_ce_unigram"),
    ("средний NCE gain", "mean_nce_gain"),
    ("accuracy", "accuracy"),
    ("accuracy unigram", "unigram_accuracy"),
    ("средний macro F1", "macro_f1_mean"),
)


def render_run(report: dict) -> str:
    """
    Отчёт читается без исходников: что запускали, что вышло, где хуже baseline.
    """

    lines: list[str] = []

    lines.append("# Короткий MLM-эксперимент mini_pragma_v2")
    lines.append("")
    lines.append(
        f"Устройство {report['device']}, precision {report['precision']}, "
        f"параметров {report['parameters']['total']:,}".replace(",", " ")
    )
    lines.append("")

    counters = report["counters"]

    lines.append(
        f"Шагов оптимизатора {counters['n_steps']} из {report['config']['max_steps']}, "
        f"batch'ей {counters['n_batches']}, пропущено {counters['n_skipped']}. "
        f"Обучение {report['train_seconds'] / 60:.1f} мин, загрузка данных {report['load_seconds']:.0f} с."
    )
    lines.append("")

    lines.append("## Выборки")
    lines.append("")
    lines.append("| набор | клиентов | примеров | событий в памяти | целей на оценке |")
    lines.append("|---|---|---|---|---|")

    for name, item in report["data"].items():
        targets = report["splits"].get(name, {}).get("n_targets")
        lines.append(
            f"| {name} | {item['clients']} | {item['examples']} | {item['events_in_memory']} | "
            f"{targets if targets is not None else '—'} |"
        )

    lines.append("")

    targets = report.get("targets")

    if targets:

        lines.append("## Цели")
        lines.append("")
        lines.append(f"Политика {targets['policy']}: {targets['note']}.")
        lines.append("")

        if targets["excluded"]:
            lines.append(
                "Выведены из задачи, но остаются входом: "
                + ", ".join(f"`{name}`" for name in targets["excluded"])
                + "."
            )
        else:
            lines.append("Исключённых полей нет.")

        lines.append("")

    # --------------------------------------------------------

    lines.append("## До и после обучения на одних и тех же масках")
    lines.append("")

    for name in report["splits"]:

        before = report["before"][name]
        after = report["after"][name]

        lines.append(f"### {name}")
        lines.append("")
        lines.append("| метрика | до | после |")
        lines.append("|---|---|---|")

        for label, key in AGGREGATES:
            lines.append(f"| {label} | {_text(before[key])} | {_text(after[key])} |")

        lines.append("")

        if before.get("recent") and after.get("recent"):

            lines.append(
                f"Только месяц наблюдения: целей {after['recent']['n_targets']:,} "
                f"из {after['n_targets']:,}.".replace(",", " ")
            )
            lines.append("")
            lines.append("| метрика | до | после |")
            lines.append("|---|---|---|")

            for label, key in AGGREGATES:
                lines.append(
                    f"| {label} | {_text(before['recent'][key])} | {_text(after['recent'][key])} |"
                )

            lines.append("")

    # --------------------------------------------------------

    final = report.get("final")

    if final:

        lines.append("## Финальная оценка")
        lines.append("")
        lines.append(
            f"Веса {final['weights']}"
            + (f" (шаг {final['step']})" if final["step"] is not None else "")
            + ". Наборы смотрели один раз, после обучения."
        )
        lines.append("")
        lines.append("| набор | целей | CE | CE месяц | NCE | NCE месяц |")
        lines.append("|---|---|---|---|---|---|")

        for name, item in final["metrics"].items():
            fresh = item.get("recent") or {}
            lines.append(
                f"| {name} | {item['n_targets']} | {_text(item['field_balanced_ce'])} | "
                f"{_text(fresh.get('field_balanced_ce'))} | {_text(item['mean_nce_gain'])} | "
                f"{_text(fresh.get('mean_nce_gain'))} |"
            )

        lines.append("")

    # --------------------------------------------------------

    lines.append("## По полям после обучения")
    lines.append("")

    for name in report["splits"]:

        lines.append(f"### {name}")
        lines.append("")
        lines.append("| поле | целей | кандидатов | CE | CE unigram | NCE gain | Acc | Acc unigram | Macro F1 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for item in report["after"][name]["fields"]:

            if item["n_targets"] == 0:
                lines.append(
                    f"| {item['field']} | 0 | {item['n_candidates']} | {item['status']} | | | | | |"
                )
                continue

            lines.append(
                f"| {item['field']} | {item['n_targets']} | {item['n_candidates']} | "
                f"{_text(item['ce_model'], 3)} | {_text(item['ce_unigram'], 3)} | "
                f"{_text(item['nce_gain'], 3)} | {_text(item['accuracy'], 3)} | "
                f"{_text(item['unigram_accuracy'], 3)} | {_text(item['macro_f1'], 3)} |"
            )

        lines.append("")

    # --------------------------------------------------------

    best = report["best"]

    lines.append("## Checkpoints")
    lines.append("")

    if best["step"] is None:
        lines.append(f"`best` не выбран: {best['reason']}")
    else:
        lines.append(
            f"`best` это шаг {best['step']} по {BEST_METRIC} на {BEST_SPLIT} "
            f"({best['value']:.4f})."
        )

    lines.append("")
    lines.append(f"- last: `{report['checkpoints']['last']}`")

    if report["checkpoints"]["best"]:
        lines.append(f"- best: `{report['checkpoints']['best']}`")

    lines.append("")
    lines.append(
        "Падение loss само по себе ничего не доказывает: полезность представлений "
        "показывает положительный NCE gain относительно unigram-baseline preprocessing "
        "на тех же позициях."
    )

    return "\n".join(lines) + "\n"
