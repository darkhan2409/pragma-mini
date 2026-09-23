from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from src.generator.rng import stable_hash
from src.masking.settings import ConfigError as MaskingConfigError
from src.masking.settings import MaskingConfig
from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from .settings import ConfigError, MlmConfig, checkpoint_path


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
#
# Одна команда, только на train:
#
#   python -m src.mlm.train [--epochs N] [--max-steps N] [--config путь]
#                           [--masking-config путь]
#
# Вход: data/07_batches/train, для validation data/07_batches/val
# и data/08_masked/val; начальные веса энкодеров из этапов 09-12.
# Выход: data/14_train/checkpoint.pt.
#
# Маска train НЕ читается из data/08_masked/train: каждая эпоха
# разыгрывает её заново тем же маскером этапа 08 по
# немаскированным value_ids батчей. Seed эпохи выводится из seed
# маскирования и номера эпохи, поэтому одна и та же эпоха даёт
# одну и ту же маску, а соседние эпохи — разные.
#
# Шаг обучения это ОДИН граф на батч:
#
#   батч -> InputEmbedding -> Event -> Profile -> History -> MLM
#        -> потери -> backward -> step
#
# Промежуточные parquet этапов 09-12 сюда не читаются: через файл
# градиент не течёт. Весь проход собран в model.Model, и здесь он
# вызывается без no_grad.
#
# Единица прохода модели — клиент. Потери батча — среднее по всем
# его целям: потери клиента взвешиваются числом его целей, иначе
# клиент с одной целью весил бы столько же, сколько клиент с
# сотней.
#
# После каждой полностью пройденной эпохи та же модель считает
# потери на val: фиксированная маска data/08_masked/val, eval и
# no_grad, без backward и шага. Среднее — по всем целям val.
# ============================================================


def for_epoch(masking: MaskingConfig, epoch: int) -> MaskingConfig:
    """
    Конфиг маскирования эпохи: те же вероятности, свой seed.

    Маскер берёт всю случайность из seed конфига, поэтому смены
    seed достаточно, чтобы эпоха получила новую маску. stable_hash
    это blake2b: результат не зависит от процесса, в отличие от
    встроенного hash().
    """

    return replace(masking, seed=stable_hash("epoch", masking.seed, epoch) % (2 ** 31))


def validate(model, source, device) -> tuple[float | None, int]:
    """
    Потери обучаемой модели на группе source, без обновления весов.

    Модель передаётся готовой: это тот же экземпляр, что только
    что учился. Своих весов validation не грузит. Среднее берётся
    по всем целям группы, как у потерь батча train; клиент без
    целей в среднее не входит. Группа без целей даёт None.
    """

    import torch

    from .model import to_tensors

    model.eval()

    total = 0.0
    targets = 0

    with torch.no_grad():

        for number in range(source.count):

            for client in source.batch(number):

                out = model(to_tensors(client, device))

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
) -> dict:
    """
    Проход по батчам train с обновлением весов всей модели.
    """

    # torch импортируется здесь, а не в шапке: без него команда
    # обязана сказать, что поставить, а не упасть на импорте.
    import torch

    from .build import _device
    from .inputs import Source
    from .model import load_model, to_tensors

    device = _device(config.device)

    model = load_model(
        group="train",
        seed=config.seed,
        events_per_chunk=config.events_per_chunk,
        label_smoothing=config.label_smoothing,
        device=device,
    )

    # Dropout энкодеров берёт случайность из глобального
    # генератора: одинаковый конфиг обязан давать одинаковые веса.
    torch.manual_seed(config.seed)

    # Параметры всей модели: общая таблица эмбеддингов, энкодеры
    # события, анкеты и истории и проекция головы.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # val с фиксированной маской из 08_masked: открывается сразу,
    # чтобы нехватка файлов стала видна до первого шага.
    val_source = Source("val")

    epoch = 0
    step = 0

    for epoch in range(1, epochs + 1):

        # Каждая эпоха начинается в режиме обучения: validation
        # прошлой эпохи оставил модель в eval, и dropout был выключен.
        model.train()

        source = Source("train", masking=for_epoch(masking, epoch))

        stopped = False

        for number in range(source.count):

            if max_steps is not None and step >= max_steps:
                stopped = True
                break

            optimizer.zero_grad(set_to_none=True)

            outputs = [model(to_tensors(client, device)) for client in source.batch(number)]

            targets = sum(out.count for out in outputs)

            # Без целей шагу нечему учить: нулевые градиенты всё
            # равно сдвинули бы веса через weight decay.
            if targets == 0:
                continue

            loss = sum(out.loss * out.count for out in outputs) / targets

            loss.backward()
            optimizer.step()

            step += 1

            print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        # Эпоха, прерванная --max-steps, не пройдена целиком:
        # validation считается только после полной эпохи.
        if not stopped:

            val_loss, val_targets = validate(model, val_source, device)

            shown = f"{val_loss:.4f}" if val_loss is not None else "n/a"

            print(f"epoch={epoch} val_loss={shown} val_targets={val_targets}")

        if max_steps is not None and step >= max_steps:
            break

    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config.as_dict(),
            "masking": masking.as_dict(),
        },
        path,
    )

    return {"checkpoint": str(path), "epoch": epoch, "step": step, "device": str(device)}


def run_training(args) -> int:

    try:
        from .build import MlmError
        from .inputs import InputError

    except ModuleNotFoundError as error:
        print(
            f"[train] нет модуля {error.name}: обучение считает тензоры, "
            "установите зависимость командой pip install -e .[torch]"
        )
        return EXIT_BLOCKED

    try:
        config = MlmConfig.load(Path(args.config) if args.config else None)

        masking = MaskingConfig.load(
            Path(args.masking_config) if args.masking_config else None
        )

        result = train(config, args.epochs, args.max_steps, masking)

    except (ConfigError, MaskingConfigError, InputError, MlmError, FileNotFoundError) as error:
        print(f"[train] {error}")
        return EXIT_BLOCKED

    print(
        f"[train] эпох {result['epoch']}, шагов {result['step']} на {result['device']} "
        f"→ {result['checkpoint']}"
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
        help="сколько раз пройти по всем батчам train",
    )
    parser.add_argument(
        "--max-steps", type=_positive, default=None,
        help="остановиться после стольких шагов оптимизатора",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="JSON с переопределениями конфига головы и оптимизатора",
    )
    parser.add_argument(
        "--masking-config", type=Path, default=None,
        help="JSON конфига маскирования, тот же, что у python -m src.masking.run",
    )

    parser.set_defaults(handler=run_training)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


__all__ = ["build_parser", "for_epoch", "main", "run_training", "train", "validate"]


if __name__ == "__main__":
    main()
