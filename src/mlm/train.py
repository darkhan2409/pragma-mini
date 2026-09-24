from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

from src.generator.rng import stable_hash
from src.masking.settings import ConfigError as MaskingConfigError
from src.masking.settings import MaskingConfig
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from .settings import ConfigError, MlmConfig, best_checkpoint_path, checkpoint_path


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
#
# Одна команда, только на train:
#
#   python -m src.mlm.train [--epochs N] [--max-steps N] [--config путь]
#                           [--masking-config путь] [--resume]
#
# Вход: data/07_batches/train, для validation data/07_batches/val
# и data/08_masked/val; начальные веса энкодеров из этапов 09-12.
# Выход: data/14_train/checkpoint.pt (последнее состояние) и
# data/14_train/best_checkpoint.pt (лучший val_loss).
#
# Маска train НЕ читается из data/08_masked/train: каждая эпоха
# разыгрывает её заново тем же маскером этапа 08 по
# немаскированным value_ids батчей. Seed эпохи выводится из seed
# маскирования и номера эпохи, поэтому одна и та же эпоха даёт
# одну и ту же маску, а соседние эпохи — разные.
#
# Клиенты идут потоком и собираются в micro-batch по бюджету
# позиций (inputs.micro_batches, token_budget). Группа строк
# 07_batches — только хранение, а не батч модели. micro-batch
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
# Промежуточные parquet этапов 09-12 сюда не читаются: через файл
# градиент не течёт. Весь проход собран в model.Model, и здесь он
# вызывается без no_grad.
#
# После каждой полностью пройденной эпохи та же модель считает
# потери на val: фиксированная маска data/08_masked/val, eval и
# no_grad, те же micro-batch'и, без backward и шага. Среднее — по
# всем целям val. По нему обновляется лучший чекпойнт и считается
# early stopping.
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
)


class CheckpointError(ValueError):
    """
    Чекпойнт нельзя прочитать или продолжить.
    """


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
    шага не делает, поэтому это верхняя оценка.
    """

    from .inputs import Source, micro_batches

    count = sum(
        1 for _ in micro_batches(Source("train", masking=masking).sizes(), config.token_budget)
    )

    return math.ceil(count / config.grad_accum_steps) * epochs


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


def validate(model, source, device, token_budget: int) -> tuple[float | None, int]:
    """
    Потери обучаемой модели на группе source, без обновления весов.

    Модель передаётся готовой: это тот же экземпляр, что только
    что учился. Своих весов validation не грузит. Клиенты идут
    теми же micro-batch'ами, что и в обучении, по одному проходу
    модели на каждый. Среднее берётся по всем целям группы;
    micro-batch без целей в него не входит. Группа без целей даёт
    None.
    """

    import torch

    from .inputs import micro_batches
    from .model import pack
    from .varlen import autocast

    model.eval()

    total = 0.0
    targets = 0

    with torch.no_grad():

        for clients in micro_batches(source.clients(), token_budget):

            with autocast(device):
                out = model(pack(clients, device))

            if out.count == 0:
                continue

            total += out.loss.item() * out.count
            targets += out.count

    return (total / targets if targets else None), targets


def train(
    config: MlmConfig,
    epochs: int,
    max_steps: int | None,
    masking: MaskingConfig,
    resume: bool = False,
) -> dict:
    """
    Проход по micro-batch'ам train с обновлением весов всей модели.

    epochs и max_steps — общие пределы от начала обучения, в том
    числе при resume. При resume config и masking берутся из
    чекпойнта.

    epochs задаёт и горизонт cosine, max_steps — нет: он только
    останавливает текущий прогон. Горизонт первого запуска лежит в
    чекпойнте и при продолжении не пересчитывается.
    """

    # torch импортируется здесь, а не в шапке: без него команда
    # обязана сказать, что поставить, а не упасть на импорте.
    import torch

    from .build import _device
    from .inputs import Source, micro_batches
    from .model import load_model, pack
    from .varlen import autocast

    latest_path = checkpoint_path()
    best_path = best_checkpoint_path()

    # Чекпойнт читается и проверяется первым: ни один файл не
    # пишется, пока продолжение не собрано целиком.
    state = load_checkpoint(latest_path) if resume else None

    if state is not None:
        config = MlmConfig.from_dict(state["config"])
        masking = MaskingConfig.from_dict(state["masking"])

    device = _device(config.device)

    model = load_model(
        group="train",
        seed=config.seed,
        events_per_chunk=config.events_per_chunk,
        label_smoothing=config.label_smoothing,
        device=device,
        attention_backend=config.attention_backend,
    )

    # Параметры всей модели: общая таблица эмбеддингов, энкодеры
    # события, анкеты и истории и проекция головы.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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

        # Полная эпоха — продолжаем со следующей. Неполная (её
        # прервал --max-steps) — с той же, пропуская уже пройденные
        # micro-batch'и: порядок клиентов и маска эпохи
        # детерминированы, а окно на остановке было пустым.
        if state["epoch_complete"]:
            first_epoch, skip = int(state["epoch"]) + 1, 0
        else:
            first_epoch, skip = int(state["epoch"]), int(state["micro_batches_done"])

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
                "reason": "nothing",
            }

    optimizer.zero_grad(set_to_none=True)

    # val с фиксированной маской из 08_masked: открывается сразу,
    # чтобы нехватка файлов стала видна до первого шага.
    val_source = Source("val")

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
        # стоило бы прошлого обучения.
        latest_path.unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)

    epoch = first_epoch
    reason = "epochs"

    # Окно накопления: сколько micro-batch'ей в нём, сколько целей
    # и сумма потерь по целям.
    window_batches = 0
    window_targets = 0
    window_loss = 0.0

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
        """

        nonlocal step, window_batches, window_targets, window_loss

        if window_targets > 0:

            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_targets)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            lr = optimizer.param_groups[0]["lr"]

            optimizer.step()
            scheduler.step()

            step += 1

            print(
                f"epoch={epoch} step={step} loss={window_loss / window_targets:.4f} "
                f"targets={window_targets} micro_batches={window_batches} lr={lr:.2e}"
            )

        optimizer.zero_grad(set_to_none=True)

        window_batches, window_targets, window_loss = 0, 0, 0.0

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
        }

    for epoch in range(first_epoch, epochs + 1):

        # Каждая эпоха начинается в режиме обучения: validation
        # прошлой эпохи оставил модель в eval, и dropout был выключен.
        model.train()

        source = Source("train", masking=for_epoch(masking, epoch))

        stopped = False

        # micro-batch'и эпохи, пройденные до этого места, включая
        # пропущенные при resume.
        done = 0

        for clients in micro_batches(source.clients(), config.token_budget):

            if done < skip:
                done += 1
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
            with autocast(device):
                out = model(pack(clients, device))

            window_batches += 1
            done += 1

            if out.count > 0:
                (out.loss * out.count).backward()
                window_targets += out.count
                window_loss += out.loss.item() * out.count

            if window_batches == config.grad_accum_steps:
                close_window()

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

        val_loss, val_targets = validate(model, val_source, device, config.token_budget)

        # val без целей сигнала не даёт: ни улучшением, ни
        # ухудшением это не считается.
        improved = val_loss is not None and (
            best is None or val_loss < best - config.early_stopping_min_delta
        )

        if improved:
            best, stale = val_loss, 0
        elif val_loss is not None:
            stale += 1

        shown = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        best_shown = f"{best:.4f}" if best is not None else "n/a"

        print(
            f"epoch={epoch} val_loss={shown} val_targets={val_targets} "
            f"best_val_loss={best_shown} patience={stale}/{config.early_stopping_patience}"
        )

        current = snapshot(True, 0)

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
        "reason": reason,
    }


def run_training(args) -> int:

    try:
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

        result = train(config, args.epochs, args.max_steps, masking, resume=args.resume)

    except (
        ConfigError, MaskingConfigError, InputError, MlmError, BackendError,
        CheckpointError, FileNotFoundError,
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
        "--epochs", type=_positive, default=1,
        help=(
            "сколько эпох всего, считая от начала обучения; первый запуск ими же "
            "задаёт горизонт cosine"
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
        help="JSON конфига маскирования, тот же, что у python -m src.masking.run",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="продолжить с data/14_train/checkpoint.pt",
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
    "CheckpointError",
    "build_parser",
    "for_epoch",
    "horizon",
    "load_checkpoint",
    "lr_factor",
    "main",
    "resumed_horizon",
    "run_training",
    "save_checkpoint",
    "train",
    "validate",
]


if __name__ == "__main__":
    main()
