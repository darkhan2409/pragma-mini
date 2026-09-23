"""
Этап 6: выравнивание примеров в батчи.

Одна команда на группу:

    python -m src.batching.run train|val|test

Вход — data/06_temporal/<group>/temporal.parquet, выход —
data/07_batches/<group>/batches.parquet и ничего больше.

Этап ничего не маскирует, не выбирает значения для предсказания
и не обучается: target_event_mask только переносится из примера
как есть.
"""

from __future__ import annotations

from .version import FORMAT_VERSION, IMPLEMENTATION_VERSION, SCHEMA_VERSION


__all__ = ["FORMAT_VERSION", "IMPLEMENTATION_VERSION", "SCHEMA_VERSION"]
