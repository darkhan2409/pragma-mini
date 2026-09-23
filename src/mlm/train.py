from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.preprocessing.run import EXIT_BLOCKED, EXIT_OK

from .settings import ConfigError, MlmConfig, checkpoint_path


# ============================================================
# ОБУЧЕНИЕ
# ============================================================
#
# Одна команда, только на train:
#
#   python -m src.mlm.train [--epochs N] [--max-steps N] [--config путь]
#
# Вход: data/07_batches/train и data/08_masked/train; начальные
# веса энкодеров из этапов 09-12. Выход: data/14_train/checkpoint.pt.
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
# ============================================================


def train(config: MlmConfig, epochs: int, max_steps: int | None) -> dict:
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

    model.train()

    # Параметры всей модели: общая таблица эмбеддингов, энкодеры
    # события, анкеты и истории и проекция головы.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    source = Source("train")

    epoch = 0
    step = 0

    for epoch in range(1, epochs + 1):

        for number in range(source.count):

            if max_steps is not None and step >= max_steps:
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

        result = train(config, args.epochs, args.max_steps)

    except (ConfigError, InputError, MlmError, FileNotFoundError) as error:
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

    parser.set_defaults(handler=run_training)

    return parser


def main(argv: list[str] | None = None) -> None:

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    raise SystemExit(args.handler(args))


__all__ = ["build_parser", "main", "run_training", "train"]


if __name__ == "__main__":
    main()
