"""
Этап 13: MLM-голова и сквозной прямой проход.

Одна команда на группу:

    python -m src.mlm.run train|val|test

Этап собирает в памяти дифференцируемый проход
InputEmbedding -> Event Encoder -> Profile Encoder ->
History Encoder -> MLM и считает по нему кросс-энтропию на
размеченных токенах.

Веса четырёх энкодеров берутся из этапов 09-12, голова
разыгрывается заново.

Обучение — отдельная команда на train:

    python -m src.mlm.train [--epochs N] [--max-steps N]

Тот же проход, но без no_grad: потери батча, backward и шаг
AdamW по всем весам модели. Маска train разыгрывается заново на
каждую эпоху маскером этапа 08. Чекпойнт — data/14_train/checkpoint.pt.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
