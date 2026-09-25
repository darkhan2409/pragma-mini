"""
Этап 9: входной слой эмбеддингов.

Одна команда на группу:

    python -m src.embedding.run train|val|test

Вход — два файла рядом: data/07_batches/<group>/batches.parquet даёт
ключи, номера кусков, анкету и маски, а data/08_masked/<group>/masked.parquet
даёт видимые модели значения. Выход — веса общей таблицы
data/09_embeddings/<group>/weights.pt и lineage.json.

Вектор токена — сумма трёх слагаемых — считает модель сама, в
прямом проходе этапов 10-13, по номерам токенов и этим весам.
Снимка векторов этап не пишет: при обучении веса меняются, и
градиент идёт в ту же таблицу. Ни внимания, ни энкодеров, ни
обучения здесь нет.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
