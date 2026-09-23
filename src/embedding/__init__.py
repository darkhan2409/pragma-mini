"""
Этап 9: входной слой эмбеддингов.

Одна команда на группу:

    python -m src.embedding.run train|val|test

Вход — два файла рядом: data/07_batches/<group>/batches.parquet даёт
ключи, номера кусков, анкету и маски, а data/08_masked/<group>/masked.parquet
даёт видимые модели значения. Выход — векторы токенов
data/09_embeddings/<group>/embeddings.parquet, страница для
человека preview.html и веса, которыми эти векторы посчитаны.

Этап складывает три слагаемых в один вектор токена и больше ничего:
ни внимания, ни энкодеров, ни обучения здесь нет.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
