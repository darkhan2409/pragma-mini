"""
Этап 7: маскирование значений.

Одна команда на группу:

    python -m src.masking.run train|val|test

Вход — data/07_batches/<group>/batches.parquet и коды [MASK] и
[UNK] из словаря. Выход — data/08_masked/<group>/masked.parquet
и ничего больше.

Этап решает, что скрыть от модели и что она должна предсказать.
Саму модель и обучение он не содержит.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
