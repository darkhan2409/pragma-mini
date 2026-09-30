"""
Этап 6: входной слой эмбеддингов.

Одна команда на группу:

    python -m src.embedding.run train|val|test

Вход — data/05_dataset/<group>/samples.parquet: ключи, номера кусков
и анкета; видимые модели значения разыгрывает маскер при чтении.
Выход — веса общей таблицы
data/06_embeddings/<group>/weights.pt и lineage.json.

Вектор токена — сумма трёх слагаемых — считает модель сама, в
прямом проходе этапов 08-11, по номерам токенов и этим весам.
Снимка векторов этап не пишет: при обучении веса меняются, и
градиент идёт в ту же таблицу. Ни внимания, ни энкодеров, ни
обучения здесь нет.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
